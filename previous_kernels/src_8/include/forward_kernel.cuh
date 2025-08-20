#pragma once

#include <cuda/std/limits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "common.h"
#include "debug.cuh"
#include "flash_attention.cuh"
#include "gemm.cuh"
#include "ptx_functions.cuh"
#include "softmax.cuh"
#include "static_kernel_configuration.cuh"
#include "tensor.cuh"

namespace flash {

template <typename Kernel>
__global__ void
flash_forward_kernel(__grid_constant__ const ForwardKernelArgs args) {
    using accum_t = float;
    using index_t = int64_t;
    using N = typename Kernel::N;

    using value_t = typename Kernel::value_t;
    using Q_t = typename Kernel::Q_t;
    using K_t = typename Kernel::K_t;
    using V_t = typename Kernel::V_t;
    using P_t = typename Kernel::P_t;

    constexpr int async = Kernel::async_copy;

    // We initialize a CTA for each sample, seq tile, and head.
    const int sample = blockIdx.z;
    const int head = blockIdx.y;
    const int q_seq_block = blockIdx.x;

    const index_t gmem_seq_stride = args.seq_stride;

    const index_t sample_head_offset =
        sample * args.batch_stride + head * args.head_stride;
    // We only read/write one block for Q and O.
    // These offsets are the same for the whole thread-block.
    const index_t QO_gmem_block_offset =
        sample_head_offset + q_seq_block * Kernel::B_r * gmem_seq_stride;
    // We read the entire key sequence.
    const index_t KV_gmem_block_offset = sample_head_offset;

    value_t *gmem_Q = &static_cast<value_t *>(args.Q)[QO_gmem_block_offset];
    value_t *gmem_O = &static_cast<value_t *>(args.O)[QO_gmem_block_offset];
    value_t *gmem_K = &static_cast<value_t *>(args.K)[KV_gmem_block_offset];
    value_t *gmem_V = &static_cast<value_t *>(args.V)[KV_gmem_block_offset];

    extern __shared__ __align__(16) char smem[];
    value_t *smem_Q = reinterpret_cast<value_t *>(smem);
    value_t *smem_O = smem_Q;
    value_t *smem_K = &smem_Q[Kernel::B_r * Kernel::d_head];
    value_t *smem_V = &smem_K[Kernel::B_c * Kernel::d_head];

    // Pointers to the K&V locations in smem that the warp copies to.
    Q_t Q(gmem_Q, gmem_seq_stride, smem_Q);
    K_t K(gmem_K, gmem_seq_stride, smem_K);
    V_t V(gmem_V, gmem_seq_stride, smem_V);

    // The accumulator for O is only kept in registers. At the end of the
    // kernel, it is then converted into a 16-bit type and then copied into
    // gmem.
    typename Kernel::O_accum_t O_accum;
    auto O_accum_no_op_tiling = O_accum.view().with_op_tiling_removed();
    using O_accum_no_op_tiling_shape =
        decltype(O_accum_no_op_tiling)::Layout::Shape;

    // Start the async copy of the Q and K tiles.
    Q.copy_GM2SM();
    cp_async_commit<async>();
    if constexpr (Kernel::eager_load_blocks) {
        K.copy_GM2SM();
        K.advance_gmem_block();
        cp_async_commit<async>();
    }

    O_accum.zero();

    // Initialize softmax_scale, m, and l.
    const accum_t softmax_scale =
        rsqrt(static_cast<accum_t>(Kernel::d_head)) * M_LOG2E;
    constexpr accum_t neg_inf = -cuda::std::numeric_limits<float>::infinity();

    // Replace raw arrays with Array objects
    ArrayAligned<N::QO_fragments_per_warp, accum_t> m;
    ArrayAligned<N::QO_fragments_per_warp, accum_t> l;

    m.fill(neg_inf);
    l.fill(0.0);

    if constexpr (Q_t::load_entire_block_into_rf) {
        if constexpr (Kernel::eager_load_blocks) {
            // We only wait for the Q block to finish loading.
            cp_async_wait<1, async>();
        } else {
            cp_async_wait<0, async>();
        }
        // We need the __syncwarp() in addition to the cp_async_wait()
        // because cp_async_wait() only blocks until the current thread has
        // finished loading. The entire warp will read this block from
        // smem, so we need to wait on a warp-wide barrier.
        // For K and V, we will need a __syncthread() instead.
        __syncthreads();
        Q.copy_SM2RF_all_tiles();
    }

    for (int j = 0; j < args.n_KV_blocks; ++j) {
        typename Kernel::S_accum_t S_accum;
        // Initialize the registers for S to 0.
        S_accum.zero();

        // Block until we've copied the K block-tile for this iteration into
        // shared memory.
        cp_async_wait<0, async>();
        // After this barrier, it is safe to load the next V block, because all
        // warps have done the previous PV matmul.
        __syncthreads();

        if constexpr (Kernel::eager_load_blocks) {
            // Start the (async) copy for the V matrix from gmem to smem but
            // do not wait until after the S=QK matmul.
            V.copy_GM2SM();
            V.advance_gmem_block();
            cp_async_commit<async>();
        }
        if constexpr (K_t::load_entire_block_into_rf) {
            K.copy_SM2RF_all_tiles();
        }

        matmul<Kernel::GEMM_QK>(Q, K, S_accum);

        cp_async_wait<0, async>();
        // After this barrier, it is safe to load the next block of K.
        __syncthreads();

        if constexpr (Kernel::eager_load_blocks) {
            // Start the async copy for the next K block-tile from gmem to
            // smem, but do not wait for the copy until the next iteration
            // when we need it.
            if (j < args.n_KV_blocks - 1) {
                K.copy_GM2SM();
                K.advance_gmem_block();
                cp_async_commit<async>();
            }
        }

        // Online softmax
        auto S_accum_untiled = S_accum.view().with_op_tiling_removed();
        ArrayAligned<N::QO_fragments_per_warp, accum_t> m_next;
        calc_row_max(S_accum_untiled, m_next, m);
        scale_l_O_and_update_rowmax(m_next, m, l, O_accum_no_op_tiling,
                                    softmax_scale);
        exponentiate_tensor<Kernel::optimized_softmax>(S_accum_untiled, m,
                                                       softmax_scale);
        update_row_exp_sum(S_accum_untiled, l);

        typename Kernel::P_t P_b16;
        // Convert the S accumulator block into P fp16 input block.
        auto S_accum_view = S_accum.view();
        auto P_b16_view = P_b16.view();
        convert_to_16_bit_dtype(S_accum_view, P_b16_view);

        if constexpr (V_t::load_entire_block_into_rf) {
            V.copy_SM2RF_all_tiles();
        }

        matmul<typename Kernel::GEMM_PV>(P_b16, V, O_accum);
    }

    // Finish summing row_sums across all threads in the same row.
    final_softmax_normalization(O_accum_no_op_tiling, l);

    typename Kernel::O_t O_b16(gmem_O, gmem_seq_stride, smem_O);
    auto O_b16_view = O_b16.view();
    convert_to_16_bit_dtype(O_accum_no_op_tiling, O_b16_view);

    // Instead of writing directly to gmem, we write to smem as an intermediary
    // step. This allows us to
    // - use 16B vectorized stores, as opposed to 4B stores
    // - fully coalesce our stores
    //   - each warp can store 4x128B aligned lines (512B/warp) instead
    //   of 8x16B uncoalesced rows (128B/warp)
    O_b16.copy_RF2SM();

    // Wait until all threads in the same warp have written to smem.
    // We do not need __syncthreads() here because the warps operate on
    // independent chunks of O.
    __syncthreads();

    // Copy the final O tile from smem to gmem.
    O_b16.copy_SM2GM();
}

} // namespace flash
