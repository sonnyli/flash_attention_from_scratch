#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

#include "common.h"
#include "ptx_functions.cuh"
#include "utils.h"

namespace flash {

// Dimensions of the mma instruction we're using
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define MMA_M_FRAGMENTS_PER_ITER 2 // (MMA_M / LDMATRIX_MAT_SIZE)
#define MMA_N_FRAGMENTS_PER_ITER 1 // (MMA_N / LDMATRIX_MAT_SIZE)
#define MMA_K_FRAGMENTS_PER_ITER 2 // (MMA_K / LDMATRIX_MAT_SIZE)

template <typename _A_t, typename _B_t, typename _C_t, int total_K_fragments,
          int load_K_fragments_per_iter, typename value_t_>
struct GEMM {
    using A_t = _A_t;
    using B_t = _B_t;
    using C_t = _C_t;
    using value_t = value_t_;

    static constexpr int TotalKTiles = total_K_fragments;
    static constexpr int LoadKTilesPerIter = load_K_fragments_per_iter;

    static constexpr bool DoubleBufferA =
        !A_t::load_entire_block_into_rf && A_t::mma_load_stages > 1;
    static constexpr bool DoubleBufferB =
        !B_t::load_entire_block_into_rf && B_t::mma_load_stages > 1;
    static constexpr bool DoubleBuffer = DoubleBufferA || DoubleBufferB;
};

// warp_fragment_mma_f32_accum
template <typename value_t, const int M_fragments, const int N_fragments,
          const int K_fragments_A, const int K_fragments_B,
          typename accum_t = float>
FA_DEVICE_CONSTEXPR void warp_fragment_mma_f32_accum(
    uint32_t (&regs_A)[M_fragments][K_fragments_A],
    uint32_t (&regs_B)[N_fragments][K_fragments_B],
    accum_t (&regs_C)[M_fragments][N_fragments * N_REGS_PER_F32_ACCUM_FRAGMENT],
    int A_col_fragment_offset = 0, int B_col_fragment_offset = 0) {
    constexpr int K_iters = constexpr_min(K_fragments_A, K_fragments_B);
    FA_UNROLL
    for (int k = 0; k < K_iters; k += MMA_K_FRAGMENTS_PER_ITER) {
        FA_UNROLL
        for (int m = 0; m < M_fragments; m += MMA_M_FRAGMENTS_PER_ITER) {
            FA_UNROLL
            for (int n = 0; n < N_fragments; n += MMA_N_FRAGMENTS_PER_ITER) {
                mma_m16n8k16_f32_accum<value_t>(
                    regs_C[m][n * 2], regs_C[m][n * 2 + 1],
                    regs_C[m + 1][n * 2], regs_C[m + 1][n * 2 + 1],
                    regs_A[m][k + A_col_fragment_offset],
                    regs_A[m + 1][k + A_col_fragment_offset],
                    regs_A[m][k + 1 + A_col_fragment_offset],
                    regs_A[m + 1][k + 1 + A_col_fragment_offset],
                    regs_B[n][k + B_col_fragment_offset],
                    regs_B[n][k + 1 + B_col_fragment_offset], regs_C[m][n * 2],
                    regs_C[m][n * 2 + 1], regs_C[m + 1][n * 2],
                    regs_C[m + 1][n * 2 + 1]);
            }
        }
    }
}

template <typename GEMM>
FA_DEVICE_CONSTEXPR void matmul(typename GEMM::A_t &A, typename GEMM::B_t &B,
                                typename GEMM::C_t &C) {
    using A_t = typename GEMM::A_t;
    using B_t = typename GEMM::B_t;
    using value_t = typename GEMM::value_t;

    constexpr int A_stage_toggle = A_t::mma_load_stages - 1;
    constexpr int B_stage_toggle = B_t::mma_load_stages - 1;

    int A_stage = 0;
    int B_stage = 0;

    if constexpr (GEMM::DoubleBufferA) {
        A.copy_SM2RF(A_stage);
    }
    if constexpr (GEMM::DoubleBufferB) {
        B.copy_SM2RF(B_stage);
    }

    FA_UNROLL
    for (int k_outer_fragment = 0; k_outer_fragment < GEMM::TotalKTiles;
         k_outer_fragment += GEMM::LoadKTilesPerIter) {
        if constexpr (!A_t::load_entire_block_into_rf ||
                      !B_t::load_entire_block_into_rf) {
            int k_load_fragment =
                k_outer_fragment +
                (GEMM::DoubleBuffer ? GEMM::LoadKTilesPerIter : 0);
            if (k_load_fragment < GEMM::TotalKTiles) {
                if constexpr (!A_t::load_entire_block_into_rf) {
                    A.copy_SM2RF(A_stage_toggle ^ A_stage, k_load_fragment);
                }
                if constexpr (!B_t::load_entire_block_into_rf) {
                    B.copy_SM2RF(B_stage_toggle ^ B_stage, k_load_fragment);
                }
            }
        }

        // Perform tile-wise outer products.
        int A_col_offset =
            A_t::load_entire_block_into_rf ? k_outer_fragment : 0;
        int B_col_offset =
            B_t::load_entire_block_into_rf ? k_outer_fragment : 0;
        warp_fragment_mma_f32_accum<value_t>(A.data(A_stage), B.data(B_stage),
                                             C.data(), A_col_offset,
                                             B_col_offset);

        A_stage ^= A_stage_toggle;
        B_stage ^= B_stage_toggle;
    }
}

} // namespace flash
