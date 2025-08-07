#pragma once

#include "array.cuh"
#include "common.h"

namespace flash {

/*
Each group of 4 threads contains a row.

*/

template <typename S_accum_t, typename RowT, typename accum_t = float>
FA_DEVICE_CONSTEXPR void calc_row_max(S_accum_t &S_accum, RowT &m_cur,
                                      RowT &m_prev) {
    FA_UNROLL
    for (int q = 0; q < S_accum_t::Shape::rows(); ++q) {
        m_cur[q] = m_prev[q];

        // Calculate max for row across all in-thread registers.
        FA_UNROLL
        for (int k = 0; k < S_accum_t::Shape::cols(); ++k) {
            m_cur[q] = max(m_cur[q], S_accum(q, k));
        }

        // Group reduction
        m_cur[q] =
            max(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m_cur[q], 2), m_cur[q]);
        m_cur[q] =
            max(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m_cur[q], 1), m_cur[q]);
    }
}

template <typename O_accum_t, typename RowT, typename accum_t = float>
FA_DEVICE_CONSTEXPR void
scale_l_O_and_update_rowmax(RowT &m_cur, RowT &m_prev, RowT &l,
                            O_accum_t &O_accum, const accum_t &softmax_scale) {
    FA_UNROLL
    for (int q = 0; q < O_accum_t::Shape::rows(); ++q) {
        accum_t scale = exp2f((m_prev[q] - m_cur[q]) * softmax_scale);
        m_prev[q] = m_cur[q];
        l[q] *= scale;
        FA_UNROLL
        for (int d_head = 0; d_head < O_accum_t::Shape::cols(); ++d_head) {
            O_accum(q, d_head) *= scale;
        }
    }
}

template <bool optimized_softmax, typename S_accum_t, typename accum_t = float>
FA_DEVICE_CONSTEXPR void
exponentiate_tensor(S_accum_t &S_accum,
                    ArrayAligned<S_accum_t::Shape::rows(), accum_t> &m,
                    const accum_t &softmax_scale) {
    FA_UNROLL
    for (int q = 0; q < S_accum_t::Shape::rows(); ++q) {
        accum_t max_scaled = m[q] * softmax_scale;
        FA_UNROLL
        for (int k = 0; k < S_accum_t::Shape::cols(); ++k) {
            S_accum(q, k) = exp2f(S_accum(q, k) * softmax_scale - max_scaled);
        }
    }
}

template <typename P_accum_t, typename accum_t = float>
FA_DEVICE_CONSTEXPR void
update_row_exp_sum(P_accum_t &P_accum,
                   ArrayAligned<P_accum_t::Shape::rows(), accum_t> &l) {
    FA_UNROLL
    for (int q = 0; q < P_accum_t::Shape::rows(); ++q) {
        FA_UNROLL
        for (int d_head = 0; d_head < P_accum_t::Shape::cols(); ++d_head) {
            l[q] += P_accum(q, d_head);
        }
    }
}

template <typename O_accum_t, typename RowT, typename accum_t = float>
FA_DEVICE_CONSTEXPR void final_softmax_normalization(O_accum_t &O_accum,
                                                     RowT &l) {
    FA_UNROLL
    for (int q = 0; q < O_accum_t::Shape::rows(); ++q) {
        l[q] += __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, l[q], 2);
        l[q] += __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, l[q], 1);
    }

    FA_UNROLL
    for (int q = 0; q < O_accum_t::Shape::rows(); ++q) {
        FA_UNROLL
        for (int d_head = 0; d_head < O_accum_t::Shape::cols(); ++d_head) {
            O_accum(q, d_head) /= l[q];
        }
    }
}

} // namespace flash