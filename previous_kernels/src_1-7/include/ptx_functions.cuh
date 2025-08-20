#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "common.h"

namespace flash {

template <bool async>
FA_DEVICE void cp_async_commit() {
    if constexpr (async) {
        asm volatile("cp.async.commit_group;");
    }
}

template <int ngroups, bool async>
FA_DEVICE void cp_async_wait() {
    if constexpr (async) {
        asm volatile("cp.async.wait_group %0;" ::"n"(ngroups));
    }
}

template <bool async>
FA_DEVICE_CONSTEXPR void cp_async_commit_and_wait_all() {
    if constexpr (async) {
        cp_async_commit<async>();
        cp_async_wait<0, async>();
    }
}

template <int size, typename T>
FA_DEVICE void cp_async(T *smem_to, T *gmem_from) {
    uint32_t smem_ptr = __cvta_generic_to_shared(smem_to);
    // The .cg option bypasses the L1 cache.
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
                 :
                 : "r"(smem_ptr), "l"(gmem_from), "n"(size));
}

template <typename T>
FA_DEVICE void ldmatrix_x4(T *load_from, uint32_t &a1, uint32_t &a2,
                           uint32_t &a3, uint32_t &a4) {
    uint32_t smem_ptr = __cvta_generic_to_shared(load_from);
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16"
                 "{%0, %1, %2, %3}, [%4];"
                 : "=r"(a1), "=r"(a2), "=r"(a3), "=r"(a4)
                 : "r"(smem_ptr));
}

template <typename T>
FA_DEVICE void ldmatrix_x4_transpose(T *load_from, uint32_t &a1, uint32_t &a2,
                                     uint32_t &a3, uint32_t &a4) {
    uint32_t smem_ptr = __cvta_generic_to_shared(load_from);
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16"
                 "{%0, %1, %2, %3}, [%4];"
                 : "=r"(a1), "=r"(a2), "=r"(a3), "=r"(a4)
                 : "r"(smem_ptr));
}

template <typename value_t>
FA_DEVICE void
mma_m16n8k16_f32_accum(float &d1, float &d2, float &d3, float &d4,
                       uint32_t const &a1, uint32_t const &a2,
                       uint32_t const &a3, uint32_t const &a4,
                       uint32_t const &b1, uint32_t const &b2, float const &c1,
                       float const &c2, float const &c3, float const &c4) {
    static_assert(std::is_same_v<value_t, half> ||
                      std::is_same_v<value_t, nv_bfloat16>,
                  "value_t must be either half or nv_bfloat16");

    if constexpr (std::is_same_v<value_t, half>) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     " { %0, %1, %2, %3 }, "
                     " { %4, %5, %6, %7 }, "
                     " { %8, %9 }, "
                     " { %10, %11, %12, %13 }; "
                     : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                     : "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(b1), "r"(b2),
                       "f"(c1), "f"(c2), "f"(c3), "f"(c4));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     " { %0, %1, %2, %3 }, "
                     " { %4, %5, %6, %7 }, "
                     " { %8, %9 }, "
                     " { %10, %11, %12, %13 }; "
                     : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                     : "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(b1), "r"(b2),
                       "f"(c1), "f"(c2), "f"(c3), "f"(c4));
    }
}

} // namespace flash