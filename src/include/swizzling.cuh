#pragma once

#include "common.h"
#include "utils.h"

namespace flash {

template <int col_fragments>
FA_DEVICE_CONSTEXPR int swizzled_col_fragment(int row, int col_fragment) {
    static_assert(col_fragments % ELEMS_PER_VEC4_ACCESS == 0,
                  "# col tiles is a multiple of # elems");

    // The % ELEMS_PER_VEC4_ACCESS makes sure that the swizzled column stays
    // within the same 8 element window.
    return (row % ELEMS_PER_VEC4_ACCESS) ^ col_fragment;
}

template <int col_fragments, bool swizzle>
FA_DEVICE_CONSTEXPR int get_smem_col_fragment(const int row,
                                              const int col_fragment) {
    return swizzle ? swizzled_col_fragment<col_fragments>(row, col_fragment)
                   : col_fragment;
}

template <const int col_fragments, const bool swizzled>
FA_DEVICE_CONSTEXPR int get_smem_offset(const int row, const int col) {
    const int offset = row * col_fragments + col;
    if constexpr (swizzled) {
        return swizzle_cute<col_fragments>(offset);
    } else {
        return offset;
    }
}

} // namespace flash
