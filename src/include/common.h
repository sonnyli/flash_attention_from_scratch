#pragma once

#define FA_UNROLL _Pragma("unroll")
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

#define N_REGS_PER_F32_ACCUM_FRAGMENT 2

#define LDMATRIX_MAT_SIZE 8
#define ROWS_PER_FRAGMENT LDMATRIX_MAT_SIZE
#define COLS_PER_FRAGMENT LDMATRIX_MAT_SIZE

#define GSM_LDST_ROWS_PER_ITER 4

#define N_BUFFER_STAGES 2
