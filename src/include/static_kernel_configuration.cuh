#pragma once

#include "common.h"
#include "flash_attention.cuh"
#include "gemm.cuh"
#include "layout.cuh"
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

template <const FlashForwardKernelConfig &cfg>
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

    static constexpr int n_threads = CFG.n_warps * WARP_SIZE;

    // The number of d_head tiles loaded and operated on by this thread
    // block.
    static constexpr int d_head_fragments = CFG.d_head / COLS_PER_FRAGMENT;

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

    // The number of K/V tiles that each warp loads into smem, which corresponds
    // to a (B_c/n_warps, d_head) chunk.
    static constexpr int KV_ldst_fragments_per_warp =
        KV_calc_fragments / CFG.n_warps;
    static constexpr int KV_ldst_rows_per_warp =
        KV_ldst_fragments_per_warp * ROWS_PER_FRAGMENT;

    static constexpr int get_tile_fragments(int val, int default_val) {
        return val == 0 ? default_val : val;
    }

    static constexpr int QK_rmem_tile_fragments = get_tile_fragments(
        constexpr_max(CFG.Q_mma_load_K_fragments, CFG.K_mma_load_K_fragments),
        d_head_fragments);
    static constexpr int QK_rmem_tile_size =
        QK_rmem_tile_fragments * COLS_PER_FRAGMENT;
    static constexpr int QK_rmem_tiles =
        d_head_fragments / QK_rmem_tile_fragments;
    static constexpr int PV_rmem_tile_fragments =
        get_tile_fragments(CFG.V_mma_load_K_fragments, KV_calc_fragments);
    static constexpr int PV_rmem_tile_size =
        PV_rmem_tile_fragments * COLS_PER_FRAGMENT;
    static constexpr int PV_rmem_tiles =
        KV_calc_fragments / PV_rmem_tile_fragments;

    static constexpr int get_rmem_tile_buffer_size(int load_K_fragments,
                                                   int tiles) {
        if (load_K_fragments == 0) {
            return tiles;
        }
        return CFG.mma_double_buffer_loads ? 2 : 1;
    }

    static constexpr int Q_rmem_tile_buffer_size =
        get_rmem_tile_buffer_size(CFG.Q_mma_load_K_fragments, QK_rmem_tiles);

    static constexpr int K_rmem_tile_buffer_size =
        get_rmem_tile_buffer_size(CFG.K_mma_load_K_fragments, QK_rmem_tiles);

    static constexpr int V_rmem_tile_buffer_size =
        get_rmem_tile_buffer_size(CFG.V_mma_load_K_fragments, PV_rmem_tiles);
};

template <FlashForwardKernelConfig _CFG>
struct StaticForwardKernelConfig {
    static constexpr FlashForwardKernelConfig CFG = _CFG;

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

    static constexpr int SwizzleTileSize = SWIZZLE_TILE_SIZE;
    static constexpr int DHeadSwizzleTiles = d_head / SwizzleTileSize;
    static constexpr int B_c_SwizzleTiles = B_c / SwizzleTileSize;

    using OpGSMemStride =
        SMemStride<N::n_threads / GSMEM_THR_PER_ROW, SwizzleTileSize, 0, 1>;

    using OpS2RSmemStride = SMemStride<16, 16, 1, 1>;
    using OpS2RRmemStride = RmemStride<1, 1, 1, 0, 0>;

    using OpR2SSMemStride = SMemStride<8, 8, 1, 1>;
    using OpR2SRmemStride = RmemStride<1, 1, 1, 0, 0>;

    // TODO: make sure this is consistent with tile size.
    using SmemSwizzle_ =
        CuteSwizzle<3, 3,
                    constexpr_log2_floor(SWIZZLE_TILE_SIZE) -
                        constexpr_log2_floor(ELEMS_PER_VEC4_ACCESS)>;
    using SmemSwizzle =
        std::conditional_t<CFG.swizzled, SmemSwizzle_, NoSwizzle>;

    using GSMemShapeQO = GSMemShape<B_r, SwizzleTileSize, 1, DHeadSwizzleTiles>;
    using GSMemShapeKV = GSMemShape<B_c, SwizzleTileSize, 1, DHeadSwizzleTiles>;

    // does not include row stride
    using GMemStride = SMemStride<CFG.d_head * N_HEADS, 1, 0, SwizzleTileSize>;

    using GSSMemQOStride =
        SMemStride<SwizzleTileSize, 1, 0, B_r * SwizzleTileSize>;
    using GSSMemKVStride =
        SMemStride<SwizzleTileSize, 1, 0, B_c * SwizzleTileSize>;

    using GSMemLdstConfigQO =
        GSMemLdstConfig<SmemSwizzle, OpGSMemStride, GSMemShapeQO,
                        GSSMemQOStride, GMemStride>;
    using GSMemLdstConfigKV =
        GSMemLdstConfig<SmemSwizzle, OpGSMemStride, GSMemShapeKV,
                        GSSMemKVStride, GMemStride>;

    static constexpr int SRMemTileSize = 16;
    static constexpr int SRMemTileFragments = SRMemTileSize / COLS_PER_FRAGMENT;
    static constexpr int SRMemTilesDHead = d_head / SRMemTileSize;
    static constexpr int SRMemTilesDHeadPerSwizzleTile =
        SRMemTilesDHead / DHeadSwizzleTiles;
    static constexpr int SRMemFragmentsDHead =
        SRMemTilesDHead * SRMemTileFragments;

    static constexpr int SRMemTilesB_c = B_c / SRMemTileSize;
    static constexpr int SRMemTilesB_cPerSwizzleTile =
        SRMemTilesB_c / B_c_SwizzleTiles;
    static constexpr int SRMemFragmentsB_c = SRMemTilesB_c * SRMemTileFragments;

    static constexpr int RSMemTileSize = 8;
    static constexpr int RSMemTilesDHead = d_head / RSMemTileSize;
    static constexpr int RSMemTilesDHeadPerSwizzleTile =
        RSMemTilesDHead / DHeadSwizzleTiles;

    using S2RSmemShapeQ =
        GSMemShape<N::QO_rows_per_warp, SRMemTileSize,
                   SRMemTilesDHeadPerSwizzleTile, DHeadSwizzleTiles>;
    using S2RSmemStrideQ = SMemStride<SwizzleTileSize, 1, SRMemTileSize,
                                      GSSMemQOStride::swizzle_space()>;
    using RmemShapeQ = RmemShape<N::QO_fragments_per_warp / 2,
                                 SRMemTileFragments / 2, SRMemTilesDHead, 2, 2>;

    using S2RMemLdstConfigQ =
        SRMemLdstConfig<SmemSwizzle, OpS2RSmemStride, OpS2RRmemStride,
                        S2RSmemStrideQ, S2RSmemShapeQ>;

    using S2RSmemShapeKV =
        GSMemShape<B_c, SRMemTileSize, SRMemTilesDHeadPerSwizzleTile,
                   DHeadSwizzleTiles>;

    using S2RSmemStrideK = SMemStride<SwizzleTileSize, 1, SRMemTileSize,
                                      GSSMemKVStride::swizzle_space()>;
    using RmemShapeK = RmemShape<SRMemFragmentsB_c / 1, SRMemTileFragments / 2,
                                 SRMemTilesDHead, 1, 2>;
    using S2RMemLdstConfigK =
        SRMemLdstConfig<SmemSwizzle, OpS2RSmemStride, OpS2RRmemStride,
                        S2RSmemStrideK, S2RSmemShapeKV,
                        true /*SmemRowMajorLdmatrix*/>;

    using S2RSmemStrideV =
        SMemStride<SwizzleTileSize, 1, SRMemTileSize * SwizzleTileSize,
                   GSSMemKVStride::swizzle_space()>;
    using RmemShapeV = RmemShape<SRMemFragmentsDHead / 1,
                                 SRMemTileFragments / 2, SRMemTilesB_c, 1, 2>;
    using S2RMemLdstConfigV =
        SRMemLdstConfig<SmemSwizzle, OpS2RSmemStride, OpS2RRmemStride,
                        S2RSmemStrideV, S2RSmemShapeKV>;

    using R2SSmemShapeO =
        GSMemShape<N::QO_rows_per_warp, RSMemTileSize,
                   RSMemTilesDHeadPerSwizzleTile, DHeadSwizzleTiles>;
    using R2SSmemStrideO = SMemStride<SwizzleTileSize, 1, RSMemTileSize,
                                      GSSMemQOStride::swizzle_space()>;
    using RmemShapeOAccum =
        RmemShape<N::QO_fragments_per_warp / 2,
                  N::d_head_fragments * THR_COLS_PER_ACCUM_FRAGMENT / 2, 1, 2,
                  2>;
    using RmemShapeO =
        RmemShape<N::QO_fragments_per_warp, N::d_head_fragments, 1, 1, 1>;
    using R2SMemLdstConfigO =
        SRMemLdstConfig<SmemSwizzle, OpR2SSMemStride, OpR2SRmemStride,
                        R2SSmemStrideO, R2SSmemShapeO>;

    using RmemShapeSAccum =
        RmemShape<N::QO_fragments_per_warp / 2,
                  N::KV_calc_fragments * THR_COLS_PER_ACCUM_FRAGMENT / 2, 1, 2,
                  2>;
    using RmemShapeP = RmemShape<N::QO_fragments_per_warp / 2,
                                 SRMemTileFragments / 2, SRMemTilesB_c, 2, 2>;

    using RmemConfigQ = RmemLdstConfig<RmemShapeQ, N::Q_rmem_tile_buffer_size,
                                       CFG.Q_mma_load_K_fragments == 0,
                                       false /*RowMajorOpTile*/>;
    using GSRConfigQ = GSRMemLdstConfig<RmemConfigQ, false, N::QO_rows_per_warp,
                                        false /*compute_over_entire_block*/
                                        >;
    using Q_t = GSRBlockTensor<GSRConfigQ, value_t, GSMemLdstConfigQO,
                               S2RMemLdstConfigQ>;

    using RmemConfigK = RmemLdstConfig<RmemShapeK, N::K_rmem_tile_buffer_size,
                                       CFG.K_mma_load_K_fragments == 0,
                                       true /*RowMajorOpTile*/>;
    using GSRConfigK =
        GSRMemLdstConfig<RmemConfigK, false, N::KV_ldst_rows_per_warp,
                         true /*compute_over_entire_block*/
                         >;
    using K_t = GSRBlockTensor<GSRConfigK, value_t, GSMemLdstConfigKV,
                               S2RMemLdstConfigK>;

    using RmemConfigV = RmemLdstConfig<RmemShapeV, N::V_rmem_tile_buffer_size,
                                       CFG.V_mma_load_K_fragments == 0,
                                       true /*RowMajorOpTile*/>;
    using GSRConfigV =
        GSRMemLdstConfig<RmemConfigV, true, N::KV_ldst_rows_per_warp,
                         true /*compute_over_entire_block*/
                         >;
    using V_t = GSRBlockTensor<GSRConfigV, value_t, GSMemLdstConfigKV,
                               S2RMemLdstConfigV>;

    // S/P is kept entirely in the rmem during the entire duration of the
    // kernel.
    using RmemConfigSAccum =
        RmemLdstConfig<RmemShapeSAccum, 1, true /*load_entire_block_into_rf*/,
                       true /*RowMajorOpTile*/>;
    using S_accum_t = RmemBlockTensor<RmemConfigSAccum, accum_t>;
    using RmemConfigP = RmemLdstConfig<RmemShapeP, RmemShapeP::tiles(),
                                       true /*load_entire_block_into_rf*/,
                                       false /*RowMajorOpTile*/>;
    using P_t = RmemBlockTensor<RmemConfigP, value_t>;

    using RmemConfigOAccum =
        RmemLdstConfig<RmemShapeOAccum, 1, true /*load_entire_block_into_rf*/,
                       true /*RowMajorOpTile*/>;
    using O_accum_t = RmemBlockTensor<RmemConfigOAccum, accum_t>;
    using RmemConfigO =
        RmemLdstConfig<RmemShapeO, 1, true /*load_entire_block_into_rf*/,
                       true /*RowMajorOpTile*/>;
    using GSRConfigO = GSRMemLdstConfig<RmemConfigO, false, N::QO_rows_per_warp,
                                        false /*compute_over_entire_block*/
                                        >;
    using O_t = GSRBlockTensor<GSRConfigO, value_t, GSMemLdstConfigQO,
                               R2SMemLdstConfigO>;

    using GEMM_QK = GEMM<Q_t, K_t, S_accum_t, SRMemTilesDHead, value_t>;
    using GEMM_PV = GEMM<P_t, V_t, O_accum_t, SRMemTilesB_c, value_t>;

    using row_statistics_t = ArrayAligned<N::QO_fragments_per_warp, accum_t>;
};

} // namespace flash
