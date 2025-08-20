#pragma once

#include <torch/torch.h>
#include <algorithm>

namespace flash {

struct ForwardKernelArgs {
    using index_t = int64_t;

    void *__restrict__ Q;
    void *__restrict__ K;
    void *__restrict__ V;
    void *__restrict__ O;

    // We assume all strides are the same across all inputs, and that
    // the tensors are all row major.
    const index_t batch_stride;
    const index_t seq_stride;
    const index_t head_stride;

    const index_t seq_len;
    const index_t n_heads;

    const int n_Q_blocks;
    const int n_KV_blocks;
};
} // namespace flash

// FlashForwardKernelConfig contains the configuration for a kernel.
// For choosing the kernel configuration at runtime, We use a map of kernel
// configs to kernels. The official repo uses static switches, which is cleaner
// and faster.
struct FlashForwardKernelConfig {
    const torch::ScalarType dtype;
    const int d_head;  // [64, 128]
    const int B_r;     // [64, 128]
    const int B_c;     // [32, 64, 128]
    const int n_warps; // [4, 8]. 8 only when B_r = 128

    const bool async_copy;
    // If true, load K and V block tiles into smem as soon as we can.
    const bool eager_load_blocks;
    const bool swizzled;

    const int Q_mma_load_K_fragments;
    const int K_mma_load_K_fragments;
    const int V_mma_load_K_fragments;

    // if true, call ldmatrix for the next iter before calling mma.
    const bool mma_double_buffer_loads;
    const bool optimized_softmax;

    int smem_bytes(int elem_size = 2) const {
        return (B_r + B_c * 2) * d_head * elem_size;
    }

    int num_ctas_per_sm(int max_smem_bytes) const {
        // The max # ctas will be 2 or less due to register limits.
        if ((n_warps == 8) || (max_smem_bytes < smem_bytes() * 2)) {
            return 1;
        }

        return 2;
    }

    bool operator<(const FlashForwardKernelConfig &other) const {
        if (dtype != other.dtype) {
            return dtype < other.dtype;
        }
        if (d_head != other.d_head) {
            return d_head < other.d_head;
        }
        if (B_r != other.B_r) {
            return B_r < other.B_r;
        }
        if (B_c != other.B_c) {
            return B_c < other.B_c;
        }
        if (n_warps != other.n_warps) {
            return n_warps < other.n_warps;
        }
        if (async_copy != other.async_copy) {
            return async_copy < other.async_copy;
        }
        if (eager_load_blocks != other.eager_load_blocks) {
            return eager_load_blocks < other.eager_load_blocks;
        }
        if (swizzled != other.swizzled) {
            return swizzled < other.swizzled;
        }
        if (Q_mma_load_K_fragments != other.Q_mma_load_K_fragments) {
            return Q_mma_load_K_fragments < other.Q_mma_load_K_fragments;
        }
        if (K_mma_load_K_fragments != other.K_mma_load_K_fragments) {
            return K_mma_load_K_fragments < other.K_mma_load_K_fragments;
        }
        if (V_mma_load_K_fragments != other.V_mma_load_K_fragments) {
            return V_mma_load_K_fragments < other.V_mma_load_K_fragments;
        }
        if (mma_double_buffer_loads != other.mma_double_buffer_loads) {
            return mma_double_buffer_loads < other.mma_double_buffer_loads;
        }
        if (optimized_softmax != other.optimized_softmax) {
            return optimized_softmax < other.optimized_softmax;
        }
        return false; // Equal configurations
    }
};
