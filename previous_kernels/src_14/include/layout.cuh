#pragma once

#include <stdio.h>
#include "common.h"
#include "debug.cuh"

namespace flash {

// This could be rewritten to be far more elegant with template
// metaprogramming, but I wanted to make the code more "readable".

template <int Row, int Col, int Tile = 0, int OpRow = 0, int OpCol = 0>
struct TStride {
    // Public accessor methods
    FA_DEVICE_CONSTEXPR static int row() { return Row; }
    FA_DEVICE_CONSTEXPR static int col() { return Col; }
    FA_DEVICE_CONSTEXPR static int tile() { return Tile; }
    FA_DEVICE_CONSTEXPR static int op_row() { return OpRow; }
    FA_DEVICE_CONSTEXPR static int op_col() { return OpCol; }
};

// This is a stride specific for swizzling. Ideally we should just use a
// tuple.
struct SwizzleStride {
    int s0;
    int s1;
    int s2;
    int s3;

    // This determines the iteration of the copy we're in.
    constexpr int offset_s2rmem(int iter) const {
        int i0 = (iter >> 2) & 1;
        int i1 = (iter >> 1) & 1;
        int i2 = iter & 1;
        return i0 * s0 + i1 * s1 + i2 * s2;
    }

    constexpr int offset_r2smem(int iter) const {
        int i0 = (iter >> 3) & 1;
        int i1 = (iter >> 2) & 1;
        int i2 = (iter >> 1) & 1;
        int i3 = iter & 1;
        return i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3;
    }
};

// RuntimeStride for global memory
template <typename index_t = int64_t>
struct RuntimeStride {
    index_t row;
    index_t col;
    index_t tile = 0;
};

template <int Rows, int Cols, int Tiles = 1, int OpRows = 1, int OpCols = 1,
          bool op_tiling_removed = false>
struct TShape {

    // Public accessor methods
    FA_DEVICE_CONSTEXPR static int rows() {
        return op_tiling_removed ? Rows * OpRows : Rows;
    }

    FA_DEVICE_CONSTEXPR static int cols() {
        return op_tiling_removed ? Cols * OpCols : Cols;
    }

    FA_DEVICE_CONSTEXPR static int tiles() { return Tiles; }

    FA_DEVICE_CONSTEXPR static int op_rows() {
        return op_tiling_removed ? 1 : OpRows;
    }

    FA_DEVICE_CONSTEXPR static int op_cols() {
        return op_tiling_removed ? 1 : OpCols;
    }

    FA_DEVICE_CONSTEXPR static int op_size() {
        return op_tiling_removed ? 1 : OpRows * OpCols;
    }

    FA_DEVICE_CONSTEXPR static int tile_size() {
        return rows() * cols() * op_size();
    }

    FA_DEVICE_CONSTEXPR static int size() { return Tiles * tile_size(); }

  private:
    template <typename, typename, bool>
    friend struct Layout;

    static constexpr int _op_rows = OpRows;
    static constexpr int _op_cols = OpCols;
};

template <typename Stride_, typename Shape_, bool OpTilingRemoved = false>
struct Layout {
    using Stride = Stride_;
    using Shape = Shape_;
    static constexpr bool op_tiling_removed = OpTilingRemoved;

    FA_DEVICE_CONSTEXPR static auto layout_as_2x2_op_tiled() {
        if constexpr (Shape::op_rows() == 2 && Shape::op_cols() == 2) {
            return Layout<Stride, Shape>{};
        } else {
            using NewShape =
                TShape<Shape::rows() / 2, Shape::cols(), Shape::tiles(),
                       Shape::op_rows() * 2, Shape::op_cols()>;
            using NewStride =
                TStride<Stride::row() * 2, Stride::col(), Stride::tile(),
                        Stride::op_row(), Stride::op_col()>;
            return flash::Layout<NewStride, NewShape>{};
        }
    }

    FA_DEVICE_CONSTEXPR static auto layout_as_type2() {
        using NewStride =
            TStride<Stride::row() / 2, Stride::col(), Stride::tile() / 2,
                    Stride::op_row(), Stride::op_col()>;
        using NewShape =
            TShape<Shape::rows(), Shape::cols() / 2, Shape::tiles(),
                   Shape::op_rows(), Shape::op_cols()>;
        return flash::Layout<NewStride, NewShape>{};
    }

    // This is a hacky way to make indexing into the accumulator block easier.
    // Since
    FA_DEVICE_CONSTEXPR static auto layout_with_op_tiling_removed() {
        using NewShape = TShape<Shape::rows(), Shape::cols(), Shape::tiles(),
                                Shape::op_rows(), Shape::op_cols(), true>;
        return flash::Layout<Stride, NewShape, true>{};
    }

    FA_DEVICE_CONSTEXPR static auto tiled_layout_with_2_cols_per_tile() {
        // static_assert(Shape::tiles() == 1, "Tiles must be 1");
        using NewShape = TShape<Shape::rows(), 2, Shape::cols() / 2,
                                Shape::op_rows(), Shape::op_cols()>;
        using NewStride =
            TStride<Stride::row(), Stride::col() * 2, Stride::tile(),
                    Stride::op_row(), Stride::op_col()>;
        return flash::Layout<NewStride, NewShape>{};
    }

    FA_DEVICE_CONSTEXPR static int crd2idx(int row, int col, int tile,
                                           int op_row = 0, int op_col = 0) {

        int _row = row;
        int _col = col;
        int _op_row = op_row;
        int _op_col = op_col;

        if constexpr (op_tiling_removed) {
            op_row = row % Shape::_op_rows;
            row = row / Shape::_op_rows;
            op_col = col % Shape::_op_cols;
            col = col / Shape::_op_cols;
        }

        auto offset = tile * Stride::tile() + row * Stride::row() +
                      col * Stride::col() + op_row * Stride::op_row() +
                      op_col * Stride::op_col();
        return offset;
    }

    __device__ static void print() {
        using MyLayout = Layout<Stride, Shape>;
        using type2_layout_t = decltype(MyLayout::layout_as_type2());
        using op_tiling_removed_layout_t =
            decltype(MyLayout::layout_with_op_tiling_removed());
        using tiled_layout_2cols_t =
            decltype(MyLayout::tiled_layout_with_2_cols_per_tile());
        using layout_2x2_op_tiled_t =
            decltype(MyLayout::layout_as_2x2_op_tiled());

        printf("Original Layout:\n");
        printf("  Shape: rows=%d, cols=%d, tiles=%d, op_rows=%d, op_cols=%d\n",
               Shape::rows(), Shape::cols(), Shape::tiles(), Shape::op_rows(),
               Shape::op_cols());
        printf("  Actual Shape: _rows=%d, _cols=%d, _tiles=%d, _op_rows=%d, "
               "_op_cols=%d\n",
               Shape::_rows, Shape::_cols, Shape::_tiles, Shape::_op_rows,
               Shape::_op_cols);
        printf("  Stride: row=%d, col=%d, tile=%d, op_row=%d, op_col=%d\n\n",
               Stride::row(), Stride::col(), Stride::tile(), Stride::op_row(),
               Stride::op_col());

        printf("Type2 Layout:\n");
        printf("  Shape: rows=%d, cols=%d, tiles=%d, op_rows=%d, op_cols=%d\n",
               type2_layout_t::Shape::rows(), type2_layout_t::Shape::cols(),
               type2_layout_t::Shape::tiles(), type2_layout_t::Shape::op_rows(),
               type2_layout_t::Shape::op_cols());
        printf("  Actual Shape: _rows=%d, _cols=%d, _tiles=%d, _op_rows=%d, "
               "_op_cols=%d\n",
               type2_layout_t::Shape::_rows, type2_layout_t::Shape::_cols,
               type2_layout_t::Shape::_tiles, type2_layout_t::Shape::_op_rows,
               type2_layout_t::Shape::_op_cols);
        printf("  Stride: row=%d, col=%d, tile=%d, op_row=%d, op_col=%d\n\n",
               type2_layout_t::Stride::row(), type2_layout_t::Stride::col(),
               type2_layout_t::Stride::tile(), type2_layout_t::Stride::op_row(),
               type2_layout_t::Stride::op_col());

        printf("Layout with Op Tiling Removed:\n");
        printf("  Shape: rows=%d, cols=%d, tiles=%d, op_rows=%d, op_cols=%d\n",
               op_tiling_removed_layout_t::Shape::rows(),
               op_tiling_removed_layout_t::Shape::cols(),
               op_tiling_removed_layout_t::Shape::tiles(),
               op_tiling_removed_layout_t::Shape::op_rows(),
               op_tiling_removed_layout_t::Shape::op_cols());
        printf("  Actual Shape: _rows=%d, _cols=%d, _tiles=%d, _op_rows=%d, "
               "_op_cols=%d\n",
               op_tiling_removed_layout_t::Shape::_rows,
               op_tiling_removed_layout_t::Shape::_cols,
               op_tiling_removed_layout_t::Shape::_tiles,
               op_tiling_removed_layout_t::Shape::_op_rows,
               op_tiling_removed_layout_t::Shape::_op_cols);
        printf("  Stride: row=%d, col=%d, tile=%d, op_row=%d, op_col=%d\n\n",
               op_tiling_removed_layout_t::Stride::row(),
               op_tiling_removed_layout_t::Stride::col(),
               op_tiling_removed_layout_t::Stride::tile(),
               op_tiling_removed_layout_t::Stride::op_row(),
               op_tiling_removed_layout_t::Stride::op_col());

        printf("Tiled Layout with 2 Cols Per Tile:\n");
        printf("  Shape: rows=%d, cols=%d, tiles=%d, op_rows=%d, op_cols=%d\n",
               tiled_layout_2cols_t::Shape::rows(),
               tiled_layout_2cols_t::Shape::cols(),
               tiled_layout_2cols_t::Shape::tiles(),
               tiled_layout_2cols_t::Shape::op_rows(),
               tiled_layout_2cols_t::Shape::op_cols());
        printf("  Actual Shape: _rows=%d, _cols=%d, _tiles=%d, _op_rows=%d, "
               "_op_cols=%d\n",
               tiled_layout_2cols_t::Shape::_rows,
               tiled_layout_2cols_t::Shape::_cols,
               tiled_layout_2cols_t::Shape::_tiles,
               tiled_layout_2cols_t::Shape::_op_rows,
               tiled_layout_2cols_t::Shape::_op_cols);
        printf("  Stride: row=%d, col=%d, tile=%d, op_row=%d, op_col=%d\n\n",
               tiled_layout_2cols_t::Stride::row(),
               tiled_layout_2cols_t::Stride::col(),
               tiled_layout_2cols_t::Stride::tile(),
               tiled_layout_2cols_t::Stride::op_row(),
               tiled_layout_2cols_t::Stride::op_col());

        printf("Layout as 2x2 Op Tiled:\n");
        printf("  Shape: rows=%d, cols=%d, tiles=%d, op_rows=%d, op_cols=%d\n",
               layout_2x2_op_tiled_t::Shape::rows(),
               layout_2x2_op_tiled_t::Shape::cols(),
               layout_2x2_op_tiled_t::Shape::tiles(),
               layout_2x2_op_tiled_t::Shape::op_rows(),
               layout_2x2_op_tiled_t::Shape::op_cols());
        printf("  Actual Shape: _rows=%d, _cols=%d, _tiles=%d, _op_rows=%d, "
               "_op_cols=%d\n",
               layout_2x2_op_tiled_t::Shape::_rows,
               layout_2x2_op_tiled_t::Shape::_cols,
               layout_2x2_op_tiled_t::Shape::_tiles,
               layout_2x2_op_tiled_t::Shape::_op_rows,
               layout_2x2_op_tiled_t::Shape::_op_cols);
        printf("  Stride: row=%d, col=%d, tile=%d, op_row=%d, op_col=%d\n\n",
               layout_2x2_op_tiled_t::Stride::row(),
               layout_2x2_op_tiled_t::Stride::col(),
               layout_2x2_op_tiled_t::Stride::tile(),
               layout_2x2_op_tiled_t::Stride::op_row(),
               layout_2x2_op_tiled_t::Stride::op_col());
    }
};

// Helper function to create a row-major stride from a shape
template <typename Shape>
constexpr auto row_major_stride() {
    static_assert(1 <= Shape::op_rows() && Shape::op_rows() <= 2);
    static_assert(1 <= Shape::op_cols() && Shape::op_cols() <= 2);

    constexpr int op_row_stride = 1;
    constexpr int op_col_stride = Shape::op_rows();

    return TStride<Shape::cols() * Shape::op_size(), 1, Shape::tile_size(),
                   op_row_stride, op_col_stride>{};
}

template <typename Shape, bool op_row_major>
constexpr auto stride_for_shape() {
    // We assume the outer shape is row major.
    constexpr int tile_stride = Shape::tile_size();
    constexpr int row_stride = Shape::op_size() * Shape::cols();
    constexpr int col_stride = Shape::op_size();
    constexpr int op_row_stride =
        op_row_major ? (Shape::op_rows() == 1 ? row_stride : Shape::op_cols())
                     : 1;
    constexpr int op_col_stride = op_row_major ? 1 : Shape::op_rows();
    return TStride<row_stride, col_stride, tile_stride, op_row_stride,
                   op_col_stride>{};
}

} // namespace flash