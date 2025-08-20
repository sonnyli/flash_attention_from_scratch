#pragma once

#include "common.h"

namespace flash {

/*
Each group of 4 threads contains a row.

*/

template <int QO_fragments, int KV_accum_fragments, typename accum_t = float>
FA_DEVICE_CONSTEXPR void
scale_S_accum(accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
              const accum_t &softmax_scale) {
    FA_UNROLL
    for (int q = 0; q < QO_fragments; ++q) {
        FA_UNROLL
        for (int k = 0; k < KV_accum_fragments; ++k) {
            S_accum[q][k] *= softmax_scale;
        }
    }
}

template <int QO_fragments, int KV_accum_fragments, typename accum_t = float>
FA_DEVICE_CONSTEXPR void
calc_row_max(accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
             accum_t (&m_next)[QO_fragments], accum_t (&m_cur)[QO_fragments]) {
    FA_UNROLL
    for (int q = 0; q < QO_fragments; ++q) {
        m_next[q] = m_cur[q];

        // Calculate max for row across all in-thread registers.
        FA_UNROLL
        for (int k = 0; k < KV_accum_fragments; ++k) {
            m_next[q] = max(m_next[q], S_accum[q][k]);
        }

        // Group reduction
        m_next[q] = max(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m_next[q], 2),
                        m_next[q]);
        m_next[q] = max(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m_next[q], 1),
                        m_next[q]);
    }
}

template <bool optimized_softmax, int QO_fragments, int d_head_accum_fragments,
          typename accum_t = float>
FA_DEVICE_CONSTEXPR void
scale_l_O(accum_t (&m_next)[QO_fragments], accum_t (&m_cur)[QO_fragments],
          accum_t (&l)[QO_fragments],
          accum_t (&O_accum)[QO_fragments][d_head_accum_fragments],
          accum_t softmax_scale) {
    FA_UNROLL
    for (int q = 0; q < QO_fragments; ++q) {
        accum_t scale;
        if constexpr (optimized_softmax) {
            scale = exp2f((m_cur[q] - m_next[q]) * softmax_scale);
        } else {
            scale = expf(m_cur[q] - m_next[q]);
        }
        m_cur[q] = m_next[q];
        l[q] *= scale;
        for (int d_head = 0; d_head < d_head_accum_fragments; ++d_head) {
            O_accum[q][d_head] *= scale;
        }
    }
}

template <bool optimized_softmax, int QO_fragments, int KV_accum_fragments,
          typename accum_t = float>
FA_DEVICE_CONSTEXPR void
exponentiate_tensor(accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
                    accum_t (&m)[QO_fragments], accum_t softmax_scale) {
    FA_UNROLL
    for (int q = 0; q < QO_fragments; ++q) {
        accum_t max_scaled;
        if constexpr (optimized_softmax) {
            max_scaled = m[q] * softmax_scale;
        }
        FA_UNROLL
        for (int k = 0; k < KV_accum_fragments; ++k) {
            if constexpr (optimized_softmax) {
                S_accum[q][k] =
                    exp2f(S_accum[q][k] * softmax_scale - max_scaled);
            } else {
                S_accum[q][k] = expf(S_accum[q][k] - m[q]);
            }
        }
    }
}

template <int QO_fragments, int d_head_accum_fragments,
          typename accum_t = float>
FA_DEVICE_CONSTEXPR void
update_row_exp_sum(accum_t (&P_accum)[QO_fragments][d_head_accum_fragments],
                   accum_t (&l)[QO_fragments]) {
    FA_UNROLL
    for (int q = 0; q < QO_fragments; ++q) {
        FA_UNROLL
        for (int d_head = 0; d_head < d_head_accum_fragments; ++d_head) {
            l[q] += P_accum[q][d_head];
        }
    }
}

template <int QO_fragments, int d_head_accum_fragments,
          typename accum_t = float>
FA_DEVICE_CONSTEXPR void final_softmax_normalization(
    accum_t (&O_accum)[QO_fragments][d_head_accum_fragments],
    accum_t (&l)[QO_fragments]) {
    // Finish summing row_sums across all threads in the same row.
    FA_UNROLL
    for (int q = 0; q < QO_fragments; ++q) {
        l[q] += __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, l[q], 2);
        l[q] += __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, l[q], 1);
    }

    // Final row-wise O softmax normalization.
    FA_UNROLL
    for (int q = 0; q < QO_fragments; ++q) {
        FA_UNROLL
        for (int d_head = 0; d_head < d_head_accum_fragments; ++d_head) {
            O_accum[q][d_head] /= l[q];
        }
    }
}

} // namespace flash