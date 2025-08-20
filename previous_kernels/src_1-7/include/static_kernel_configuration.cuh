#pragma once

#include "common.h"
#include "flash_attention.cuh"
#include "gemm.cuh"
#include "load_store.cuh"
#include "tensor.cuh"
#include "utils.h"

namespace flash {

template <int n, int K, bool double_buffer>
constexpr void static_assert_valid_load_k_fragments() {
    static_assert(((n & (n - 1)) == 0) && n != 1,
                  "load k is power of 2 and DNE 1");

    constexpr int max_frags = (double_buffer ? K / 2 : K) / 8;
    static_assert(n <= max_frags, "load k is <= max fragments");
}

template <FlashForwardKernelConfig cfg>
constexpr bool valid_config() {
    static_assert_valid_load_k_fragments<cfg.Q_mma_load_K_fragments, cfg.d_head,
                                         cfg.mma_double_buffer_loads>();
    static_assert_valid_load_k_fragments<cfg.K_mma_load_K_fragments, cfg.d_head,
                                         cfg.mma_double_buffer_loads>();
    static_assert_valid_load_k_fragments<cfg.V_mma_load_K_fragments, cfg.B_c,
                                         cfg.mma_double_buffer_loads>();

    static_assert((cfg.Q_mma_load_K_fragments == cfg.K_mma_load_K_fragments) ||
                  cfg.Q_mma_load_K_fragments == 0);

    return true;
}

template <FlashForwardKernelConfig CFG>
struct ForwardKernelTileShapes {
    static_assert(valid_config<CFG>());

    // The number of d_head tiles loaded and operated on by this thread
    // block.
    static constexpr int d_head_fragments = CFG.d_head / COLS_PER_FRAGMENT;
    static constexpr int d_head_accum_regs =
        d_head_fragments * N_REGS_PER_F32_ACCUM_FRAGMENT;

    // The number of Q/O rows/tiles each warp independently loads and computes
    // on, which corresponds to a (B_r / n_warps, d_head) chunk.
    static constexpr int QO_rows_per_warp = CFG.B_r / CFG.n_warps;
    static constexpr int QO_fragments_per_warp =
        QO_rows_per_warp / ROWS_PER_FRAGMENT;

    // For a K/V block, each warp will independently load a chunk of the (B_c,
    // d_head), but perform computations on the entire block loaded by the
    // thread-block.

    // The number of K/V tiles that each warp operates on, which corresponds to
    // a (B_c, d_head) chunks.
    static constexpr int KV_calc_fragments = CFG.B_c / ROWS_PER_FRAGMENT;
    static constexpr int KV_calc_accum_regs =
        KV_calc_fragments * N_REGS_PER_F32_ACCUM_FRAGMENT;

    // The number of K/V tiles that each warp loads into smem, which corresponds
    // to a (B_c/n_warps, d_head) chunk.
    static constexpr int KV_ldst_fragments_per_warp =
        KV_calc_fragments / CFG.n_warps;
    static constexpr int KV_ldst_rows_per_warp =
        KV_ldst_fragments_per_warp * ROWS_PER_FRAGMENT;

    // # tiles to load during matmuls between mma instructions.
    static constexpr int Q_mma_load_K_fragments =
        CFG.Q_mma_load_K_fragments == 0 ? d_head_fragments
                                        : CFG.Q_mma_load_K_fragments;
    static constexpr int Q_mma_load_stages =
        (CFG.Q_mma_load_K_fragments > 0 && CFG.mma_double_buffer_loads) ? 2 : 1;

    static constexpr int K_mma_load_K_fragments =
        CFG.K_mma_load_K_fragments == 0 ? d_head_fragments
                                        : CFG.K_mma_load_K_fragments;
    static constexpr int K_mma_load_stages =
        (CFG.K_mma_load_K_fragments > 0 && CFG.mma_double_buffer_loads) ? 2 : 1;

    static constexpr int V_mma_load_K_fragments =
        CFG.V_mma_load_K_fragments == 0 ? KV_calc_fragments
                                        : CFG.V_mma_load_K_fragments;
    static constexpr int V_mma_load_stages =
        (CFG.V_mma_load_K_fragments > 0 && CFG.mma_double_buffer_loads) ? 2 : 1;
};

template <FlashForwardKernelConfig CFG>
struct StaticForwardKernelConfig {
    using accum_t = float;
    using value_t = typename std::conditional_t<CFG.dtype == torch::kBFloat16,
                                                nv_bfloat16, half>;
    using N = ForwardKernelTileShapes<CFG>;

    // Static configuration fields accessed from the original CFG
    static constexpr bool async_copy = CFG.async_copy;
    static constexpr int B_r = CFG.B_r;
    static constexpr int B_c = CFG.B_c;
    static constexpr int d_head = CFG.d_head;
    static constexpr bool eager_load_blocks = CFG.eager_load_blocks;
    static constexpr bool optimized_softmax = CFG.optimized_softmax;

    static constexpr LDSTCommon Common{CFG.swizzled, CFG.async_copy};

    static constexpr TensorLDSTConfig make_ldst_config(
        TileLayout GSM, TileLayout RF, bool transposed, int block_size,
        int warp_ldst_rows, bool compute_over_entire_block,
        bool load_entire_block_into_rf = true, int mma_load_stages = 1) {

        return TensorLDSTConfig{GSM,
                                RF,
                                Common,
                                transposed,
                                block_size,
                                CFG.d_head,
                                warp_ldst_rows,
                                compute_over_entire_block,
                                load_entire_block_into_rf,
                                mma_load_stages};
    }

    static constexpr TensorLDSTConfig Q_LDST =
        make_ldst_config({N::QO_fragments_per_warp, N::d_head_fragments},
                         {N::QO_fragments_per_warp, N::Q_mma_load_K_fragments},
                         false /*transposed*/, CFG.B_r, N::QO_rows_per_warp,
                         false /*compute_over_entire_block*/,
                         CFG.Q_mma_load_K_fragments == 0, N::Q_mma_load_stages);
    using Q_t = MatrixLDST<Q_LDST, value_t>;

    static constexpr TensorLDSTConfig K_LDST = make_ldst_config(
        {N::KV_ldst_fragments_per_warp, N::d_head_fragments},
        {N::KV_calc_fragments, N::K_mma_load_K_fragments}, false /*transposed*/,
        CFG.B_c, N::KV_ldst_rows_per_warp, true /*compute_over_entire_block*/,
        CFG.K_mma_load_K_fragments == 0, N::K_mma_load_stages);
    using K_t = MatrixLDST<K_LDST, value_t>;

    static constexpr TensorLDSTConfig V_LDST = make_ldst_config(
        {N::KV_ldst_fragments_per_warp, N::d_head_fragments},
        {N::d_head_fragments, N::V_mma_load_K_fragments}, true /*transposed*/,
        CFG.B_c, N::KV_ldst_rows_per_warp, true /*compute_over_entire_block*/,
        CFG.V_mma_load_K_fragments == 0, N::V_mma_load_stages);
    using V_t = MatrixLDST<V_LDST, value_t>;

    static constexpr TensorLDSTConfig O_LDST =
        make_ldst_config({N::QO_fragments_per_warp, N::d_head_fragments},
                         {N::QO_fragments_per_warp, N::d_head_fragments},
                         false /*transposed*/, CFG.B_r, N::QO_rows_per_warp,
                         false /*compute_over_entire_block*/, true);
    using O_accum_t = MatrixLDST<O_LDST, accum_t>;
    using O_value_t = MatrixLDST<O_LDST, value_t>;

    // S/P is kept entirely in the RF during the entire duration of the kernel.
    static constexpr TensorLDSTConfig S_LDST = make_ldst_config(
        {N::QO_fragments_per_warp, N::KV_calc_fragments},
        {N::QO_fragments_per_warp, N::KV_calc_fragments}, CFG.B_r, false,
        0 /* only stored in RF, not smem or gmem */,
        false /*compute_over_entire_block*/);
    using S_accum_t = MatrixLDST<S_LDST, accum_t>;
    using P_value_t = MatrixLDST<S_LDST, value_t>;

    using S_QK_GEMM = GEMM<Q_t, K_t, S_accum_t, N::d_head_fragments,
                           constexpr_min(N::Q_mma_load_K_fragments,
                                         N::K_mma_load_K_fragments),
                           value_t>;
    using O_PV_GEMM = GEMM<P_value_t, V_t, O_accum_t, N::KV_calc_fragments,
                           N::V_mma_load_K_fragments, value_t>;

    using row_statistics_t = RFVector<accum_t, N::QO_fragments_per_warp>;
};

} // namespace flash
