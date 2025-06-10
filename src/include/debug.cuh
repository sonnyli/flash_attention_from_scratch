#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "common.h"
#include "flash_attention.cuh"
#include "gemm.cuh"
#include "layout.cuh"
#include "load_store.cuh"
#include "static_kernel_configuration.cuh"

namespace flash {

constexpr int debug_warp_rank = 1;
constexpr int debug_block = 0;

using print_cast = float;

FA_DEVICE bool block0() {
    return blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
}

FA_DEVICE bool thread0() { return threadIdx.x == 0 && block0(); }

FA_DEVICE bool thread1() { return threadIdx.x == 1 && block0(); }

FA_DEVICE bool is_debug_block() {
    return (blockIdx.x + blockIdx.y * gridDim.x +
            blockIdx.z * gridDim.x * gridDim.y) == debug_block;
}

FA_DEVICE bool is_debug_warp() {
    return is_debug_block() && (threadIdx.x / 32) == debug_warp_rank;
}

FA_DEVICE bool is_warp_leader() {
    return is_debug_warp() && threadIdx.x % 32 == 0;
}

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
    __syncthreads();
    if (!is_warp_leader()) {
        return;
    }
    if (name != nullptr && iter >= 0) {
        printf("%s_%d SMEM:\n", name, iter);
    }
    for (int i1 = 0; i1 < nrows; i1++) {
        for (int i2 = 0; i2 < ncols; i2++) {
            if constexpr (std::is_same_v<castTo, int>) {
                printf("%d ", static_cast<castTo>(t[i1 * LD + i2]));
            } else {
                printf("%5.2f ", static_cast<castTo>(t[i1 * LD + i2]));
            }
        }
        printf("\n");

        if (i1 % 8 == 7) {
            printf("\n");
        }
    }
    printf("\n");
}

template <typename Tensor_t_, typename cast_to = print_cast>
FA_DEVICE void print_rmem_matrix(Tensor_t_ &t, const char *name = nullptr,
                                 const int iter = -1,
                                 const int print_tile = -1) {
    const int lane_id = threadIdx.x % 32;
    if (!is_debug_warp()) {
        return;
    }
    if (is_warp_leader() && name != nullptr && iter >= 0) {
        if (print_tile >= 0) {
            printf("%s_%d REGS (print_tile %d):\n", name, iter, print_tile);
        } else {
            printf("%s_%d REGS:\n", name, iter);
        }
    }

    auto view = t.view_with_op_tiling_removed().as_type2();
    using Tensor_t = decltype(view);

    for (int row_fragment = 0; row_fragment < Tensor_t::Shape::rows();
         ++row_fragment) {
        if (is_warp_leader()) {
            printf("row: %d\n", row_fragment * 8);
        }
        for (int thr_row = 0; thr_row < 8; ++thr_row) {

            for (int current_tile = 0; current_tile < Tensor_t::Shape::tiles();
                 ++current_tile) {
                if (print_tile >= 0 && current_tile != print_tile) {
                    continue;
                }

                for (int col_fragment = 0;
                     col_fragment < Tensor_t::Shape::cols(); ++col_fragment) {
                    __syncwarp();

                    for (int thr_col = 0; thr_col < 4; ++thr_col) {
                        int cur_lane = thr_row * 4 + thr_col;
                        if (cur_lane == lane_id) {
                            auto elem =
                                view(row_fragment, col_fragment, current_tile);
                            auto v1 = static_cast<cast_to>(elem.x);
                            auto v2 = static_cast<cast_to>(elem.y);

                            if constexpr (std::is_same_v<cast_to, int>) {
                                printf("%5d %5d ", v1, v2);
                            } else {
                                printf("%5.2f %5.2f ", v1, v2);
                            }
                        }

                        __syncwarp();
                    }
                    if (is_warp_leader()) {
                        printf("  ");
                    }
                }
            }

            if (is_warp_leader()) {
                printf("\n");
            }
        }
    }
    if (is_warp_leader()) {
        printf("\n");
    }
}

template <typename Tensor_t_, typename cast_to = print_cast>
FA_DEVICE void print_rmem_accum_matrix(Tensor_t_ &t, const char *name = nullptr,
                                       const int iter = -1,
                                       const int print_tile = -1) {
    const int lane_id = threadIdx.x % 32;
    if (!is_debug_warp()) {
        return;
    }
    if (is_warp_leader() && name != nullptr && iter >= 0) {
        if (print_tile >= 0) {
            printf("%s_%d REGS (print_tile %d):\n", name, iter, print_tile);
        } else {
            printf("%s_%d REGS:\n", name, iter);
        }
    }

    auto view = t.view_with_op_tiling_removed();
    using Tensor_t = decltype(view);

    for (int row_fragment = 0; row_fragment < Tensor_t::Shape::rows();
         ++row_fragment) {
        if (is_warp_leader()) {
            printf("row: %d\n", row_fragment * 8);
        }
        for (int thr_row = 0; thr_row < 8; ++thr_row) {
            for (int current_tile = 0; current_tile < Tensor_t::Shape::tiles();
                 ++current_tile) {
                if (print_tile >= 0 && current_tile != print_tile) {
                    continue;
                }

                for (int col_fragment = 0;
                     col_fragment < Tensor_t::Shape::cols();
                     col_fragment += 2) {
                    __syncwarp();

                    for (int thr_col = 0; thr_col < 4; ++thr_col) {
                        int cur_lane = thr_row * 4 + thr_col;
                        if (cur_lane == lane_id) {
                            auto v1 = static_cast<cast_to>(
                                view(row_fragment, col_fragment, current_tile));
                            auto v2 = static_cast<cast_to>(view(
                                row_fragment, col_fragment + 1, current_tile));

                            if constexpr (std::is_same_v<cast_to, int>) {
                                printf("%5d %5d ", v1, v2);
                            } else {
                                printf("%7.2f %7.2f ", v1, v2);
                            }
                        }

                        __syncwarp();
                    }
                    if (is_warp_leader()) {
                        printf("  ");
                    }
                }
            }

            if (is_warp_leader()) {
                printf("\n");
            }
        }
    }
    if (is_warp_leader()) {
        printf("\n");
    }
}

template <typename Array_t, typename cast_to = print_cast>
FA_DEVICE void print_rf_row(const Array_t &array, const char *name = nullptr,
                            const int iter = -1,
                            bool print_entire_warp = true) {
    const int lane_id = threadIdx.x % 32;
    if (!is_debug_warp()) {
        return;
    }
    if (is_warp_leader() && name != nullptr && iter >= 0) {
        printf("%s_%d REGS:\n", name, iter);
    }

    for (int row_fragment = 0; row_fragment < array.size(); ++row_fragment) {
        for (int t = 0; t < 32; ++t) {
            __syncwarp();
            if (lane_id == t) {
                printf("%5.2f ", static_cast<float>(array[row_fragment]));
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

// Helper function to print stride information
template <typename Stride>
FA_DEVICE static void print_stride(const char *name) {
    printf("  %s: {row: %d, col: %d, tile: %d", name, Stride::row(),
           Stride::col(), Stride::tile());

    // Only print op_row and op_col if they're non-zero
    if (Stride::op_row() != 0 || Stride::op_col() != 0) {
        printf(", op_row: %d, op_col: %d", Stride::op_row(), Stride::op_col());
    }

    printf("}\n");
}

// Helper function to print shape information
template <typename Shape>
FA_DEVICE static void print_shape(const char *name) {
    printf("  %s: {rows: %d, cols: %d, tiles: %d", name, Shape::rows(),
           Shape::cols(), Shape::tiles());

    // Only print op_rows and op_cols if they're not 1 (default)
    if (Shape::op_rows() != 1 || Shape::op_cols() != 1) {
        printf(", op_rows: %d, op_cols: %d", Shape::op_rows(),
               Shape::op_cols());
    }

    printf("}\n");
}

// Helper function to print tensor type information
template <typename Tensor>
FA_DEVICE static void print_tensor_type(const char *name) {
    printf("\n%s:\n", name);
    // print_shape<typename Tensor::StorageShape>("StorageShape");
    print_shape<typename Tensor::Shape>("Shape");
    print_stride<typename Tensor::Stride>("Stride");
    printf("  StorageSize: %d\n", Tensor::StorageSize);
    printf("  rmem_tile_buffer_size: %d\n", Tensor::rmem_tile_buffer_size);
    printf("  load_entire_block_into_rf: %d\n",
           Tensor::load_entire_block_into_rf);

    // Call the layout print function to show detailed layout information
    printf("\n  Layout Details for %s:\n", name);
    using Layout = typename Tensor::Layout;
    Layout::print();
}

// Print configuration as a static member function of FlashKernelTypes
template <typename Kernel>
FA_DEVICE static void print_config() {
    using N = typename Kernel::N;
    printf("\nFlashKernelTypes Configuration:\n");
    printf("----------------------------------------\n");

    // Print Kernel Configuration
    printf("\nKernel Configuration:\n");
    printf("  B_r: %d\n", Kernel::B_r);
    printf("  B_c: %d\n", Kernel::B_c);
    printf("  d_head: %d\n", Kernel::d_head);
    printf("  n_warps: %d\n", Kernel::n_warps);
    printf("  swizzled: %d\n", Kernel::swizzled);
    printf("  async_copy: %d\n", Kernel::async_copy);
    printf("  eager_load_blocks: %d\n", Kernel::eager_load_blocks);
    printf("  optimized_softmax: %d\n", Kernel::optimized_softmax);
    printf("  mma_double_buffer_loads: %d\n",
           Kernel::CFG.mma_double_buffer_loads);
    printf("  Q_mma_load_K_fragments: %d\n",
           Kernel::CFG.Q_mma_load_K_fragments);
    printf("  K_mma_load_K_fragments: %d\n",
           Kernel::CFG.K_mma_load_K_fragments);
    printf("  V_mma_load_K_fragments: %d\n",
           Kernel::CFG.V_mma_load_K_fragments);

    // Print Tile Configurations
    printf("\nTile Configurations:\n");
    printf("  n_threads: %d\n", N::n_threads);
    printf("  d_head_fragments: %d\n", N::d_head_fragments);
    printf("  QO_rows_per_warp: %d\n", N::QO_rows_per_warp);
    printf("  QO_fragments_per_warp: %d\n", N::QO_fragments_per_warp);
    printf("  KV_calc_fragments: %d\n", N::KV_calc_fragments);
    printf("  KV_ldst_fragments_per_warp: %d\n", N::KV_ldst_fragments_per_warp);
    printf("  KV_ldst_rows_per_warp: %d\n", N::KV_ldst_rows_per_warp);
    printf("  QK_rmem_tile_fragments: %d\n", N::QK_rmem_tile_fragments);
    printf("  QK_rmem_tile_size: %d\n", N::QK_rmem_tile_size);
    printf("  QK_rmem_tiles: %d\n", N::QK_rmem_tiles);
    printf("  PV_rmem_tile_fragments: %d\n", N::PV_rmem_tile_fragments);
    printf("  PV_rmem_tile_size: %d\n", N::PV_rmem_tile_size);
    printf("  PV_rmem_tiles: %d\n", N::PV_rmem_tiles);
    printf("  Q_rmem_tile_buffer_size: %d\n", N::Q_rmem_tile_buffer_size);
    printf("  K_rmem_tile_buffer_size: %d\n", N::K_rmem_tile_buffer_size);
    printf("  V_rmem_tile_buffer_size: %d\n", N::V_rmem_tile_buffer_size);

    // Print Common config
    printf("\nCommon Configuration:\n");
    printf("  swizzled: %d\n", Kernel::common.swizzled);
    printf("  async_copy: %d\n", Kernel::common.async_copy);

    // Print Strides
    printf("\nStrides:\n");
    print_stride<typename Kernel::OpGSMemStride>("OpGSMemStride");
    print_stride<typename Kernel::OpS2RSmemStride>("OpS2RSmemStride");
    print_stride<typename Kernel::OpS2RRmemStride>("OpS2RRmemStride");
    print_stride<typename Kernel::OpR2SSMemStride>("OpR2SSMemStride");
    print_stride<typename Kernel::OpR2SRmemStride>("OpR2SRmemStride");

    // Print Shapes and Memory Configurations
    printf("\nShapes and Memory:\n");
    print_shape<typename Kernel::GSMemShapeQO>("GSMemShapeQO");
    print_shape<typename Kernel::GSMemShapeKV>("GSMemShapeKV");
    print_stride<typename Kernel::GSMemQKVOStride>("GSMemQKVOStride");

    printf("\nS2R Memory Shapes:\n");
    print_shape<typename Kernel::S2RSmemShapeQ>("S2RSmemShapeQ");
    print_shape<typename Kernel::S2RSmemShapeK>("S2RSmemShapeK");
    print_shape<typename Kernel::S2RSmemShapeV>("S2RSmemShapeV");
    print_shape<typename Kernel::R2SSmemShapeO>("R2SSmemShapeO");

    printf("\nRmem Shapes:\n");
    print_shape<typename Kernel::RmemShapeQ>("RmemShapeQ");
    print_shape<typename Kernel::RmemShapeK>("RmemShapeK");
    print_shape<typename Kernel::RmemShapeV>("RmemShapeV");
    print_shape<typename Kernel::RmemShapeO>("RmemShapeO");
    print_shape<typename Kernel::RmemShapeOAccum>("RmemShapeOAccum");
    print_shape<typename Kernel::RmemShapeSAccum>("RmemShapeSAccum");
    print_shape<typename Kernel::RmemShapeP>("RmemShapeP");

    printf("\nStrides:\n");
    print_stride<typename Kernel::S2RSmemStrideQ>("S2RSmemStrideQ");
    print_stride<typename Kernel::S2RSmemStrideK>("S2RSmemStrideK");
    print_stride<typename Kernel::S2RSmemStrideV>("S2RSmemStrideV");
    print_stride<typename Kernel::R2SSmemStrideO>("R2SSmemStrideO");

    printf("\nRmemMatrix Configurations:\n");
    print_tensor_type<typename Kernel::O_accum_t>("O_accum_t");
    print_tensor_type<typename Kernel::S_accum_t>("S_accum_t");
    print_tensor_type<typename Kernel::P_t>("P_t");

    printf("\nTensor Types:\n");
    print_tensor_type<typename Kernel::Q_t>("Q_t (Base RmemTensor)");
    print_tensor_type<typename Kernel::K_t>("K_t (Base RmemTensor)");
    print_tensor_type<typename Kernel::V_t>("V_t (Base RmemTensor)");
    print_tensor_type<typename Kernel::O_t>("O_t (Base RmemTensor)");

    // Print GEMM Configurations
    printf("\nGEMM Configurations:\n");

    printf("GEMM_QK:\n");
    printf("  Tiles: %d\n", Kernel::GEMM_QK::Tiles);
    printf("  DoubleBufferA: %d\n", Kernel::GEMM_QK::DoubleBufferA);
    printf("  DoubleBufferB: %d\n", Kernel::GEMM_QK::DoubleBufferB);
    printf("  DoubleBuffer: %d\n", Kernel::GEMM_QK::DoubleBuffer);

    printf("\nGEMM_PV:\n");
    printf("  Tiles: %d\n", Kernel::GEMM_PV::Tiles);
    printf("  DoubleBufferA: %d\n", Kernel::GEMM_PV::DoubleBufferA);
    printf("  DoubleBufferB: %d\n", Kernel::GEMM_PV::DoubleBufferB);
    printf("  DoubleBuffer: %d\n", Kernel::GEMM_PV::DoubleBuffer);

    printf("----------------------------------------\n");
}

} // namespace flash
