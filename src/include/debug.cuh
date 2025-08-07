#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "common.h"

namespace flash {

constexpr int debug_warp_rank = 0;
constexpr int debug_block = 0;

using print_cast = float;

FA_DEVICE bool is_debug_block() {
    return (blockIdx.x + blockIdx.y * gridDim.x +
            blockIdx.z * gridDim.x * gridDim.y) == debug_block;
}

FA_DEVICE bool is_debug_warp() {
    return is_debug_block() && (threadIdx.x / 32) == debug_warp_rank;
}

FA_DEVICE bool is_warp_leader() { return threadIdx.x % 32 == 0; }

#define printf_leader(fmt, ...)                                                \
    if (is_debug_warp() && is_warp_leader())                                   \
        printf(fmt, ##__VA_ARGS__);

#define printf_warp(fmt, ...)                                                  \
    return;                                                                    \
    if (is_debug_warp())                                                       \
        printf(fmt, ##__VA_ARGS__);

template <typename value_t, typename castTo = print_cast>
FA_DEVICE void print_smem_matrix(value_t *t, int LD, int nrows = 32,
                                 int ncols = 32, const char *name = nullptr,
                                 const int iter = -1) {
    if (!is_debug_warp()) {
        return;
    }
    if (is_warp_leader() && name != nullptr && iter >= 0) {
        printf("%s_%d SMEM:\n", name, iter);
    }
    if (is_warp_leader()) {
        for (int i1 = 0; i1 < nrows; i1++) {
            for (int i2 = 0; i2 < ncols; i2++) {
                if constexpr (std::is_same_v<castTo, int>) {
                    printf("%d ", static_cast<castTo>(t[i1 * LD + i2]));
                } else {
                    printf("%5.2f ", static_cast<castTo>(t[i1 * LD + i2]));
                }
            }
            printf("\n");
        }
        printf("\n");
    }
}

template <const int n_row_fragments, const int n_col_fragments,
          typename cast_to = print_cast>
FA_DEVICE void
print_rf_matrix(uint32_t (&reg)[n_row_fragments][n_col_fragments],
                const char *name = nullptr, const int iter = -1) {
    const int lane_id = threadIdx.x % 32;
    if (!is_debug_warp()) {
        return;
    }
    if (is_warp_leader() && name != nullptr) {
        printf("%s_%d REGS:\n", name, iter);
    }
    for (int row_fragment = 0; row_fragment < n_row_fragments; ++row_fragment) {
        for (int r = 0; r < 8; ++r) {
            for (int col_fragment = 0; col_fragment < n_col_fragments;
                 ++col_fragment) {
                half2 vals = reinterpret_cast<half2 *>(
                    &reg[row_fragment][col_fragment])[0];
                __syncwarp();

                for (int col = 0; col < 4; ++col) {
                    if ((r * 4 + col) == lane_id) {
                        if constexpr (std::is_same_v<cast_to, int>) {
                            printf("%2d %2d ", static_cast<cast_to>(vals.x),
                                   static_cast<cast_to>(vals.y));
                        } else {
                            printf("%5.2f %5.2f ", static_cast<cast_to>(vals.x),
                                   static_cast<cast_to>(vals.y));
                        }
                    }
                    __syncwarp();
                }
            }

            if (is_warp_leader()) {
                printf("\n");
            }
        }
        __syncwarp();
    }
    if (is_warp_leader()) {
        printf("\n\n");
    }
    __syncwarp();
}

} // namespace flash

template <const int n_row_fragments, const int n_col_fragments,
          typename cast_to = print_cast>
FA_DEVICE void
print_rf_accum_matrix(float (&reg)[n_row_fragments][n_col_fragments],
                      const char *name = nullptr, const int iter = -1) {
    const int lane_id = threadIdx.x % 32;
    if (!is_debug_warp()) {
        return;
    }
    if (is_warp_leader() && name != nullptr && iter >= 0) {
        printf("%s_%d REGS:\n", name, iter);
    }
    for (int row_fragment = 0; row_fragment < n_row_fragments; ++row_fragment) {
        for (int r = 0; r < 8; ++r) {
            for (int col_fragment = 0; col_fragment < n_col_fragments / 2;
                 col_fragment += 2) {
                __syncwarp();

                for (int col = 0; col < 4; ++col) {
                    if ((r * 4 + col) == lane_id) {
                        if constexpr (std::is_same_v<cast_to, int>) {
                            printf(
                                "%2d %2d ",
                                static_cast<cast_to>(
                                    reg[row_fragment][col_fragment * 2]),
                                static_cast<cast_to>(
                                    reg[row_fragment][col_fragment * 2 + 1]));
                        } else {
                            printf(
                                "%5.2f %5.2f ",
                                static_cast<cast_to>(
                                    reg[row_fragment][col_fragment * 2]),
                                static_cast<cast_to>(
                                    reg[row_fragment][col_fragment * 2 + 1]));
                        }
                    }
                    __syncwarp();
                }
            }

            if (is_warp_leader()) {
                printf("\n");
            }
        }
        __syncwarp();
    }
    if (is_warp_leader()) {
        printf("\n\n");
    }
    __syncwarp();
}

} // namespace flash

template <const int n_row_fragments, typename value_t>
FA_DEVICE void print_rf_row(value_t (&regs)[n_row_fragments],
                            const char *name = nullptr, const int iter = -1,
                            bool print_entire_warp = true) {
    const int lane_id = threadIdx.x % 32;
    if (!is_debug_warp()) {
        return;
    }
    if (is_warp_leader() && name != nullptr && iter >= 0) {
        printf("%s_%d REGS:\n", name, iter);
    }

    for (int row_fragment = 0; row_fragment < n_row_fragments; ++row_fragment) {
        for (int t = 0; t < 32; ++t) {
            __syncwarp();
            if (lane_id == t) {
                printf("%5.2f ", static_cast<float>(regs[row_fragment]));
            }

            __syncwarp();
            if ((t + 1) % 4 == 0) {
                if (is_warp_leader()) {
                    printf("\n");
                }
            }
        }
    }
    __syncwarp();
    if (is_warp_leader()) {
        printf("\n\n");
    }
    __syncwarp();
}

} // namespace flash

template <const int n_row_fragments, typename value2_t = half2>
FA_DEVICE void print_rf_row_fp162(uint32_t (&regs)[n_row_fragments],
                                  const char *name = nullptr,
                                  const int iter = -1,
                                  bool print_entire_warp = true) {
    const int lane_id = threadIdx.x % 32;
    if (!is_debug_warp()) {
        return;
    }
    if (is_warp_leader() && name != nullptr && iter >= 0) {
        printf("%s_%d REGS:\n", name, iter);
    }

    for (int row_fragment = 0; row_fragment < n_row_fragments; ++row_fragment) {
        for (int t = 0; t < 32; ++t) {
            __syncwarp();
            if (lane_id == t) {
                printf("%5.2f %5.2f ", static_cast<float>(regs[row_fragment].x),
                       static_cast<float>(regs[row_fragment].y));
            }

            __syncwarp();
            if ((t + 1) % 4 == 0) {
                if (is_warp_leader()) {
                    printf("\n");
                }
            }
        }
    }
    __syncwarp();
    if (is_warp_leader()) {
        printf("\n\n");
    }
    __syncwarp();
}

} // namespace flash
