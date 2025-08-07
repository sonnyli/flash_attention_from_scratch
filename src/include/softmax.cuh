#pragma once

#include "array.cuh"
#include "common.h"

namespace flash {

/*
Each group of 4 threads contains a row.

*/

template <bool is_first, typename S_accum_t, typename RowT,
          typename accum_t = float>
FA_DEVICE_CONSTEXPR void calc_row_max(S_accum_t &S_accum, RowT &m) {
    FA_UNROLL
    for (int q = 0; q < S_accum_t::Shape::rows(); ++q) {
        if constexpr (is_first) {
            m[q] = S_accum(q, 0);
        } else {
            m[q] = max(m[q], S_accum(q, 0));
        }

        // Calculate max for row across all in-thread registers.
        FA_UNROLL
        for (int k = 1; k < S_accum_t::Shape::cols(); ++k) {
            m[q] = max(m[q], S_accum(q, k));
        }

        // Group reduction
        m[q] = max(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m[q], 2), m[q]);
        m[q] = max(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m[q], 1), m[q]);
    }
}

template <typename O_accum_t, typename RowT, typename accum_t = float>
FA_DEVICE_CONSTEXPR void scale_l_O(RowT &m_cur, RowT &m_prev, RowT &l,
                                   O_accum_t &O_accum,
                                   const accum_t &softmax_scale) {
    FA_UNROLL
    for (int q = 0; q < O_accum_t::Shape::rows(); ++q) {
        accum_t scale = exp2f((m_prev[q] - m_cur[q]) * softmax_scale);
        l[q] *= scale;
        FA_UNROLL
        for (int d_head = 0; d_head < O_accum_t::Shape::cols(); ++d_head) {
            O_accum(q, d_head) *= scale;
        }
    }
}

template <typename S_accum_t, typename accum_t = float>
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

template <bool is_first, typename P_accum_t, typename accum_t = float>
FA_DEVICE_CONSTEXPR void
update_row_exp_sum(P_accum_t &P_accum,
                   ArrayAligned<P_accum_t::Shape::rows(), accum_t> &l) {
    FA_UNROLL
    for (int q = 0; q < P_accum_t::Shape::rows(); ++q) {
        if constexpr (is_first) {
            l[q] = P_accum(q, 0);
        } else {
            l[q] += P_accum(q, 0);
        }

        FA_UNROLL
        for (int d_head = 1; d_head < P_accum_t::Shape::cols(); ++d_head) {
            l[q] += P_accum(q, d_head);
        }
    }
}

template <bool is_first, bool optimized_softmax, typename S_accum_untiled_t,
          typename O_accum_untiled_t, typename row_statistics_t,
          typename accum_t = float>
FA_DEVICE_CONSTEXPR void local_softmax(S_accum_untiled_t &S_accum_untiled,
                                       O_accum_untiled_t &O_accum_untiled,
                                       row_statistics_t &m, row_statistics_t &l,
                                       const accum_t &softmax_scale) {
    if constexpr (is_first && optimized_softmax) {
        calc_row_max<is_first>(S_accum_untiled, m);
        exponentiate_tensor(S_accum_untiled, m, softmax_scale);
        update_row_exp_sum<is_first>(S_accum_untiled, l);
    } else {
        row_statistics_t m_prev;
        m_prev.copy(m);
        calc_row_max<is_first>(S_accum_untiled, m);

        scale_l_O(m, m_prev, l, O_accum_untiled, softmax_scale);
        exponentiate_tensor(S_accum_untiled, m, softmax_scale);
        update_row_exp_sum<is_first>(S_accum_untiled, l);
    }
}

template <typename O_accum_untiled_t, typename row_statistics_t,
          typename accum_t = float>
FA_DEVICE_CONSTEXPR void
final_softmax_normalization(O_accum_untiled_t &O_accum_untiled,
                            row_statistics_t &l) {
    // Finish reduction sum across all threads in the same row.
    FA_UNROLL
    for (int q = 0; q < row_statistics_t::size(); ++q) {
        l[q] += __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, l[q], 2);
        l[q] += __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, l[q], 1);
        l[q] = 1.0f / l[q];
    }

    FA_UNROLL
    for (int q = 0; q < O_accum_untiled_t::Shape::rows(); ++q) {
        FA_UNROLL
        for (int d_head = 0; d_head < O_accum_untiled_t::Shape::cols();
             ++d_head) {
            O_accum_untiled(q, d_head) *= l[q];
        }
    }
}

} // namespace flash