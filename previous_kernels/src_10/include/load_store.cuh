#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tuple>

#include "common.h"
#include "debug.cuh"
#include "layout.cuh"
#include "ptx_functions.cuh"
#include "swizzling.cuh"
#include "utils.h"

namespace flash {

struct LDSTCommon {
    const bool swizzled;
    const bool async_copy;
};

template <typename ShapeT, int TileBufferSize, bool LoadEntireBlockIntoRF,
          bool RowMajorOpTile = true>
struct RmemLdstConfig {
    using rmem_shape = ShapeT;
    static constexpr int rmem_tile_buffer_size = TileBufferSize;
    static constexpr bool load_entire_block_into_rf = LoadEntireBlockIntoRF;
    static constexpr bool row_major_op_tile = RowMajorOpTile;
};

// constexpr non-type template parameter containing parameters for LDST for a
// block (Q, K, V, or O) from gmem to smem and vice versa, and also loading from
// smem to the rmem.
template <typename RmemConfigT, bool Swizzled, bool AsyncCopy, bool Transposed,
          int BlockSize, int SmemCols, int WarpLdstRows,
          bool ComputeOverEntireBlock>
struct GSRMemLdstConfig {
    using rmem = RmemConfigT;
    static constexpr LDSTCommon common{Swizzled, AsyncCopy};
    static constexpr bool transposed = Transposed;
    static constexpr int block_size = BlockSize;
    static constexpr int smem_cols = SmemCols;

    // This is the # of rows a warp in a thread-block independently
    // loads/stores. This is only used for Q and O.
    static constexpr int warp_ldst_rows = WarpLdstRows;

    // Whether not the warp will compute over the entire block.
    // This is false for (Q&O&S) and true for (K&V).
    static constexpr bool compute_over_entire_block = ComputeOverEntireBlock;
};

template <typename T>
struct GM2SM_async {
    FA_DEVICE_CONSTEXPR void operator()(T *gmem, T *smem) {
        cp_async<BYTES_PER_VEC4_ACCESS>(smem, gmem);
    }
};

template <typename T>
struct GM2SM {
    FA_DEVICE_CONSTEXPR void operator()(T *gmem, T *smem) {
        reinterpret_cast<uint4 *>(smem)[0] = reinterpret_cast<uint4 *>(gmem)[0];
    }
};

template <typename T>
struct SM2GM {
    FA_DEVICE_CONSTEXPR void operator()(T *gmem, T *smem) {
        reinterpret_cast<uint4 *>(gmem)[0] = reinterpret_cast<uint4 *>(smem)[0];
    }
};

template <typename Swizzle_, typename OpStride_, typename TensorShape_,
          typename SmemStride_>
struct GSMemLdstConfig {
    using Swizzle = Swizzle_;
    using OpStride = OpStride_;
    using TensorShape = TensorShape_;
    using SmemStride = SmemStride_;

    using OpIters = TShape<TensorShape::rows() / OpStride::row(),
                           TensorShape::cols() / OpStride::col()>;

    static constexpr int thrs_per_row = 8;

    static constexpr int tid_to_thr_row(int tid) { return tid / thrs_per_row; }

    static constexpr int tid_to_thr_col(int tid) {
        return (tid % thrs_per_row) * COLS_PER_FRAGMENT;
    }

    static constexpr int gmem_thr_offset(int tid, RuntimeStride stride) {
        return tid_to_thr_row(tid) * stride.row +
               tid_to_thr_col(tid) * stride.col;
    }

    static constexpr int smem_thr_offset(int tid) {
        return Swizzle::apply(tid_to_thr_row(tid) * SmemStride::row() +
                              tid_to_thr_col(tid) * SmemStride::col());
    }
};

// Copy a (B_r, d_head) or (B_c, d_head) block from gmem to smem or vice
// versa. Each warp independently loads a (seq_len_per_warp, d_head) block.
// Each inner iteration loads a (4, 64) tile, where each row is loaded by a
// group of 8 consecutive threads.
// In the edge case that we're loading a (128, 64) block with 8 warps, each warp
template <typename op, /* either GM2SM_async or SM2GM */
          typename Cfg, typename value_t = half, typename index_t = int64_t>
FA_DEVICE_CONSTEXPR void copy_block_GSM(value_t *gmem, value_t *smem,
                                        const int gmem_row_stride) {
    FA_UNROLL
    for (int ir = 0; ir < Cfg::OpIters::rows(); ++ir) {
        FA_UNROLL
        for (int ic = 0; ic < Cfg::OpIters::cols(); ++ic) {
            int r = ir * Cfg::OpStride::row();
            int c = ic * Cfg::OpStride::col();
            int smem_idx =
                r * Cfg::SmemStride::row() + c * Cfg::SmemStride::col();

            // We assume gmem is continguous along the d_head dimension.
            int gmem_idx = c * 1; // Use col stride of 1
            op()(&gmem[gmem_idx], &smem[smem_idx]);
        }

        gmem += Cfg::OpStride::row() * gmem_row_stride;
    }
}

template <typename Swizzle_, typename OpSmemStride_, typename OpRmemStride_,
          typename SmemStride_, typename SmemShape_,
          bool SmemRowMajorLdmatrix_ = false>
struct SRMemLdstConfig {
    using Swizzle = Swizzle_;
    using OpSmemStride = OpSmemStride_;
    using OpRmemStride = OpRmemStride_;
    using SmemStride = SmemStride_;
    using SmemShape = SmemShape_;
    static constexpr bool smem_row_major_ldmatrix = SmemRowMajorLdmatrix_;
    using OpIters = TShape<SmemShape::rows() / OpSmemStride::row(),
                           SmemShape::cols() / OpSmemStride::col()>;

    static constexpr int smem_col_fragments_per_tile = SmemShape::cols() / 8;

    static constexpr int lane_to_thr_offset_s2rmem(int lane_id) {
        int thread_row, thread_col;
        if constexpr (!smem_row_major_ldmatrix) {
            thread_row = lane_id % 16;
            thread_col = (lane_id / 16) * COLS_PER_FRAGMENT;
        } else {
            thread_row = (lane_id % 8) + 8 * (lane_id / 16);
            thread_col = lane_id & 8;
        }
        return Swizzle::apply(thread_row * SmemStride::row() +
                              thread_col * SmemStride::col());
    }

    static constexpr int lane_to_thr_offset_r2smem(int lane_id) {
        int thread_row = lane_id / 4;
        int thread_col = (lane_id % 4) * 2;
        return thread_row * SmemStride::row() + thread_col * SmemStride::col();
    }

    static constexpr SwizzleStride lane_to_thr_swizzle_stride(int lane_id) {
        if constexpr (std::is_same_v<Swizzle, NoSwizzle>) {
            return SwizzleStride{64, 32, 16};
        } else {
            int base_swizzle_offset = lane_to_thr_offset_s2rmem(lane_id);
            // Determine the swizzle offsets
            int base_offset_cmp = Swizzle::yy_mask_lowest_bit << 1;
            int s1 = 32 * binary_to_pm1((base_swizzle_offset &
                                         (base_offset_cmp << 1)) == 0);
            int s2 = 16 * binary_to_pm1(
                              (base_swizzle_offset & base_offset_cmp) == 0);

            // The 64 stride is for when we cross from one swizzle boundary to
            // the next.
            return SwizzleStride{64, s1, s2};
        }
    }
};

// Loads matrix fragments in smem into registers.
// Each ldmatrix.x4 instruction loads a (16, 16) chunk, i.e. (2, 2) fragments.
// For this non-transposed version, the shape of the smem matches rmem, i.e.
// shape(rmem) = (r_r, r_c) = (s_r / 8, s_c / 8).
// This will be used to copy Q and K.
template <typename Cfg, typename RmemType, typename value_t>
FA_DEVICE_CONSTEXPR void
copy_warp_fragment_SM2RF(RmemType &rmem, value_t *smem,
                         const SwizzleStride &swizzle_stride, const int &tile) {
    static_assert(is_supported_mma_input_type<value_t>(),
                  "value_t must be half or bfloat16");
    static_assert(Cfg::OpIters::rows() == RmemType::ViewType2x2::Shape::rows() /
                                              Cfg::OpRmemStride::row(),
                  "OpIters.rows must be equal to RmemType::Shape.rows / "
                  "Cfg::OpRmemStride.row");
    static_assert(RmemType::Shape::cols() == 1,
                  "RmemType::Shape.cols must be 2");
    auto rmem_uint = rmem.view2x2();
    int swizzle_offset = swizzle_stride.offset(tile);

    FA_UNROLL
    for (int ir = 0; ir < Cfg::OpIters::rows(); ++ir) {
        int smem_offset =
            ir * Cfg::OpSmemStride::row() * Cfg::SmemStride::row() +
            swizzle_offset;
        int rmem_row = ir * Cfg::OpRmemStride::row();
        if constexpr (!Cfg::smem_row_major_ldmatrix) {
            ldmatrix_x4<false>(&smem[smem_offset],
                               rmem_uint(rmem_row, 0, tile, 0, 0),
                               rmem_uint(rmem_row, 0, tile, 1, 0),
                               rmem_uint(rmem_row, 0, tile, 0, 1),
                               rmem_uint(rmem_row, 0, tile, 1, 1));
        } else {
            ldmatrix_x4<false>(&smem[smem_offset],
                               rmem_uint(rmem_row, 0, tile, 0, 0),
                               rmem_uint(rmem_row, 0, tile, 0, 1),
                               rmem_uint(rmem_row, 0, tile, 1, 0),
                               rmem_uint(rmem_row, 0, tile, 1, 1));
        }
    }
}

// Loads matrix fragments in smem into registers.
// Each ldmatrix.x4 instruction loads a (16, 16) chunk, i.e. (2, 2) fragments.
// For this transposed version, the shape of the smem matches the transpose of
// rmem, i.e. shape(rmem) = (r_r, r_c) = (s_c / 8, s_r / 8).
// This will be used to copy V.
template <typename Cfg, typename RmemType, typename value_t>
FA_DEVICE_CONSTEXPR void
copy_warp_fragment_transposed_SM2RF(RmemType &rmem, value_t *smem,
                                    const SwizzleStride &swizzle_stride,
                                    const int &tile) {
    static_assert(is_supported_mma_input_type<value_t>(),
                  "value_t must be half or bfloat16");
    using Swizzle = typename Cfg::Swizzle;
    static_assert(RmemType::Shape::cols() == 1,
                  "RmemType::Shape.cols must be 2");
    static_assert(Cfg::OpIters::cols() == RmemType::ViewType2x2::Shape::rows() /
                                              Cfg::OpRmemStride::col(),
                  "OpIters.cols must be equal to RmemType::Shape.rows / "
                  "Cfg::OpRmemStride.col");
    auto rmem_uint = rmem.view2x2();

    int base_offset = tile * Cfg::SmemStride::tile();

    FA_UNROLL
    for (int ic = 0; ic < Cfg::OpIters::cols(); ++ic) {
        int swizzle_offset = swizzle_stride.offset(ic);
        int smem_offset = base_offset + swizzle_offset;

        int rmem_row = ic * Cfg::OpRmemStride::row();
        ldmatrix_x4<true>(&smem[smem_offset],
                          rmem_uint(rmem_row, 0, tile, 0, 0),
                          rmem_uint(rmem_row, 0, tile, 0, 1),
                          rmem_uint(rmem_row, 0, tile, 1, 0),
                          rmem_uint(rmem_row, 0, tile, 1, 1));
    }
}

// Copies matrix fragments in rmem to smem.
// Each iteration of the inner loop copies a (8, 8) tile, i.e. a single
// fragment. This will be used to copy O.
template <typename Cfg, typename RmemType, typename value_t>
FA_DEVICE_CONSTEXPR void copy_warp_fragment_RF2SM(RmemType &rmem, value_t *smem,
                                                  const int &thread_offset) {
    static_assert(is_supported_mma_input_type<value_t>(),
                  "value_t must be half or bfloat16");
    using Swizzle = typename Cfg::Swizzle;
    auto rmem_uint = rmem.view();

    FA_UNROLL
    for (int ir = 0; ir < Cfg::OpIters::rows(); ++ir) {
        FA_UNROLL
        for (int ic = 0; ic < Cfg::OpIters::cols(); ++ic) {
            int smem_offset = Swizzle::apply(
                ir * Cfg::OpSmemStride::row() * Cfg::SmemStride::row() +
                ic * Cfg::OpSmemStride::col() * Cfg::SmemStride::col() +
                thread_offset);
            reinterpret_cast<uint32_t *>(&smem[smem_offset])[0] = rmem_uint(
                ir * Cfg::OpRmemStride::row(), ic * Cfg::OpRmemStride::col());
        }
    }
}

template <typename SrcType, typename DstType>
FA_DEVICE_CONSTEXPR void convert_to_16_bit_dtype(SrcType &src_view,
                                                 DstType &dst_view) {
    static_assert(std::is_same_v<typename SrcType::value_t, float>,
                  "Input tensor must be float type");
    static_assert(std::is_same_v<typename DstType::value_t, half> ||
                      std::is_same_v<typename DstType::value_t, nv_bfloat16>,
                  "Output tensor must be half or bfloat16 type");
    using value_t = typename DstType::value_t;

    auto src = src_view.with_op_tiling_removed();
    auto dst2 = dst_view.with_op_tiling_removed().as_type2();
    using SrcShape = decltype(src)::Layout::Shape;
    using DstShape = decltype(dst2)::Layout::Shape;

    static_assert(SrcShape::tiles() == 1, "Src must have 1 tile");
    static_assert(SrcShape::cols() * SrcShape::tiles() ==
                      DstShape::cols() * DstShape::tiles() * 2,
                  "A and B must have the same shape");
    static_assert(SrcShape::rows() == DstShape::rows(),
                  "A and B must have the same shape");

    FA_UNROLL
    for (int tile = 0; tile < DstShape::tiles(); ++tile) {
        int tile_offset = 2 * tile * DstShape::cols();
        FA_UNROLL
        for (int m = 0; m < DstShape::rows(); ++m) {
            FA_UNROLL
            for (int k = 0; k < DstShape::cols(); ++k) {
                int src_k = tile_offset + 2 * k;
                float2 src_val{src(m, src_k, 0), src(m, src_k + 1, 0)};
                if constexpr (std::is_same_v<value_t, half>) {
                    dst2(m, k, tile) = __float22half2_rn(src_val);
                } else {
                    dst2(m, k, tile) = __float22bfloat162_rn(src_val);
                }
            }
        }
    }
}

} // namespace flash
