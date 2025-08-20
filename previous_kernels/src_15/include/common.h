#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

#ifdef FA_DEBUG
#define FA_UNROLL
#else
#define FA_UNROLL _Pragma("unroll")
#endif

#define FA_DEVICE __forceinline__ __device__
#define FA_DEVICE_CONSTEXPR __forceinline__ __device__ constexpr

#define WARP_SIZE 32
#define SHFL_ENTIRE_WARP_MASK 0xffffffff

#define B16_BYTES 2
#define BYTES_PER_VEC4_ACCESS 16
#define ELEMS_PER_VEC4_ACCESS (BYTES_PER_VEC4_ACCESS / B16_BYTES)

// mma/ldmatrix related constants
#define MMA_A_REGS_PER_ROW 2
#define MMA_A_REGS_PER_COL 2
#define MMA_B_REGS_PER_ROW 2
#define MMA_B_REGS_PER_COL 1
#define MMA_C_REGS_PER_ROW 1
#define MMA_C_REGS_PER_COL 2

#define THR_COLS_PER_ACCUM_FRAGMENT 2

#define LDMATRIX_MAT_SIZE 8
#define ROWS_PER_FRAGMENT LDMATRIX_MAT_SIZE
#define COLS_PER_FRAGMENT LDMATRIX_MAT_SIZE

#define N_BUFFER_STAGES 2

#define GSMEM_THR_PER_ROW 8
#define SWIZZLE_TILE_SIZE 64

namespace flash {

struct alignas(16) uint128_t {
    uint64_t low;
    uint64_t high;
};

template <typename value_t>
constexpr bool is_supported_mma_input_type() {
    return std::is_same_v<value_t, half> ||
           std::is_same_v<value_t, nv_bfloat16>;
}

template <typename value_t>
constexpr bool is_supported_mma_output_type() {
    return std::is_same_v<value_t, float>;
}

template <typename value_t>
constexpr auto value_storage_type() {
    if constexpr (is_supported_mma_input_type<value_t>()) {
        return uint32_t{};
    } else if constexpr (is_supported_mma_output_type<value_t>()) {
        return float{};
    }
}

template <typename value_t>
constexpr auto value2_storage_type() {
    if constexpr (std::is_same_v<value_t, half>) {
        return half2{};
    } else if constexpr (std::is_same_v<value_t, nv_bfloat16>) {
        return nv_bfloat162{};
    } else if constexpr (std::is_same_v<value_t, float>) {
        return float2{};
    }
}

} // namespace flash