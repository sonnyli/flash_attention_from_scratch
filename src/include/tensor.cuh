#pragma once

#include "array.cuh"
#include "common.h"
#include "debug.cuh"
#include "load_store.cuh"
#include "tensor_view.cuh"

namespace flash {

// TODO(sonny): refactor these structs
// - use tensorview for both gmem and smem and use swizzle
template <typename RmemConfig, typename value_t_, typename index_t = int64_t>
struct RmemBlockTensor {
    using Shape = typename RmemConfig::rmem_shape;
    using Stride =
        decltype(stride_for_shape<Shape, RmemConfig::row_major_op_tile>());
    using Layout = Layout<Stride, Shape>;
    using Layout2x2 = decltype(Layout::layout_as_2x2_op_tiled());

    using value_t = value_t_;
    using storage_t = decltype(value_storage_type<value_t>());

    using ViewType = TensorView<value_t, Layout>;
    using ViewType2x2 = TensorView<value_t, Layout2x2>;

    static constexpr int StorageSize = Shape::size();

    static constexpr int rmem_tile_buffer_size =
        RmemConfig::rmem_tile_buffer_size;
    static constexpr bool load_entire_block_into_rf =
        RmemConfig::load_entire_block_into_rf;

    ArrayAligned<StorageSize, storage_t> _storage;

    FA_DEVICE_CONSTEXPR void zero() { _storage.zero(); }

    FA_DEVICE_CONSTEXPR ViewType view() { return ViewType(_storage.data()); }
    FA_DEVICE_CONSTEXPR ViewType2x2 view2x2() {
        return ViewType2x2(_storage.data());
    }
    FA_DEVICE_CONSTEXPR auto view_as_type2() { return view().as_type2(); }

    FA_DEVICE_CONSTEXPR auto view_with_op_tiling_removed() {
        return view().with_op_tiling_removed();
    }
};

template <typename GSRConfig, typename value_t, typename gsmem, typename srmem,
          typename index_t = int64_t>
struct GSRBlockTensor
    : public RmemBlockTensor<typename GSRConfig::rmem, value_t, index_t> {
    using Base = RmemBlockTensor<typename GSRConfig::rmem, value_t, index_t>;
    using GM2SM_op = std::conditional_t<GSRConfig::common.async_copy,
                                        GM2SM_async<value_t>, GM2SM<value_t>>;

    using SM2GM_op = SM2GM<value_t>;

    // The location in memory that the warp reads from for Q, K, V from gmem to
    // smem and O for smem to gmem.
    value_t *gmem_ptr;
    RuntimeStride gmem_stride;

    // The location in memory that the warp writes to for Q, K, V from gmem
    // to smem and O for smem to gmem. It is offset to the specific position
    // that the thread reads.
    value_t *smem_gsmem_ptr;
    // The location in memory used to load fragments from smem to rmem. This is
    // different that the ptr for smem when copying from gmem to smem because
    // the threads load different values in a different pattern.
    value_t *smem_s2rmem_ptr;
    value_t *smem_r2smem_ptr;

    SwizzleStride s2rmem_swizzle_stride;
    SwizzleStride r2smem_swizzle_stride;

    FA_DEVICE GSRBlockTensor(value_t *gmem_block_ptr, index_t _gmem_seq_stride,
                             value_t *_smem_ptr)
        : Base(), gmem_stride(RuntimeStride{int(_gmem_seq_stride), 1, 0}) {
        const int tid = threadIdx.x;

        // We increment the pointers to the exact location for the thread.
        gmem_ptr = gmem_block_ptr + gsmem::gmem_thr_offset(tid, gmem_stride);
        smem_gsmem_ptr = _smem_ptr + gsmem::smem_thr_offset(tid);

        const int lane_id = tid % WARP_SIZE;
        const int warp_rank = tid / WARP_SIZE;
        s2rmem_swizzle_stride =
            srmem::lane_to_thr_swizzle_stride_s2rmem(lane_id);
        r2smem_swizzle_stride =
            srmem::lane_to_thr_swizzle_stride_r2smem(lane_id);

        auto smem_srmem_ptr =
            _smem_ptr + (GSRConfig::compute_over_entire_block
                             ? 0
                             : GSRConfig().warp_ldst_rows * warp_rank *
                                   GSRConfig().smem_cols);

        smem_s2rmem_ptr =
            smem_srmem_ptr + srmem::lane_to_thr_offset_s2rmem(lane_id);
        smem_r2smem_ptr =
            smem_srmem_ptr + srmem::lane_to_thr_offset_r2smem(lane_id);
    }

    FA_DEVICE_CONSTEXPR void advance_gmem_block() {
        gmem_ptr += GSRConfig().block_size * gmem_stride.row;
    }

    FA_DEVICE_CONSTEXPR void copy_GM2SM() {
        copy_block_GSM<GM2SM_op, gsmem>(gmem_ptr, smem_gsmem_ptr,
                                        gmem_stride.row);
    }

    FA_DEVICE_CONSTEXPR void copy_SM2GM() {
        copy_block_GSM<SM2GM_op, gsmem>(gmem_ptr, smem_gsmem_ptr,
                                        gmem_stride.row);
    }

    FA_DEVICE_CONSTEXPR void copy_SM2RF(int tile = 0) {
        if constexpr (!GSRConfig::transposed) {
            copy_warp_fragment_SM2RF<srmem>(*this, smem_s2rmem_ptr,
                                            s2rmem_swizzle_stride, tile);
        } else {
            copy_warp_fragment_transposed_SM2RF<srmem>(
                *this, smem_s2rmem_ptr, s2rmem_swizzle_stride, tile);
        }
    }

    FA_DEVICE_CONSTEXPR void copy_SM2RF_all_tiles() {
        for (int tile = 0; tile < Base::Shape::tiles(); ++tile) {
            copy_SM2RF(tile);
        }
    }

    FA_DEVICE_CONSTEXPR void copy_RF2SM() {
        copy_warp_fragment_RF2SM<srmem>(*this, smem_r2smem_ptr,
                                        r2smem_swizzle_stride);
    }
};

} // namespace flash
