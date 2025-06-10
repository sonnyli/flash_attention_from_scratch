// This file is auto-generated in "gen_kernel_instantiations.py".

#pragma once

#include <map>

#include "flash_attention.cuh"
#include "forward_kernel.cuh"

namespace flash {

typedef void (*forward_kernel_fn)(const ForwardKernelArgs);

std::map<FlashForwardKernelConfig, forward_kernel_fn>
    forward_kernels = {
        // (FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, false, false}>>},
        // (FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, false, true}>>},
        // (FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, true, false}>>},
        // (FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, true, true}>>},
        // (FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_2_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, false, false}>>},
        // (FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, false, true}>>},
        // (FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true, false}>>},
        // (FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true, true}>>},
        // (FP16, 128, 64, 64, 4): async+load_0_0_0_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, false, false, 0, 0, 0, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, false, false, 0, 0, 0, false, false}>>},
        // (FP16, 128, 64, 64, 4): async+swizzled+load_0_0_0_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, false, true, 0, 0, 0, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, false, true, 0, 0, 0, false, false}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_0_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, false, false}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, false, true}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, true, false}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, true, true}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_2_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, false, false}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, false, true}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, true, false}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, true, true}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_0_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, false, false}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, false, true}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, true, false}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, true, true}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, false}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, true}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, true, false}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, true, true}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_2_2_2_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 2, 2, 2, false, false}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_2_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 2, 2, 2, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 2, 2, 2, true, false}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 2, 2, 2, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 2, 2, 2, true, true}>>},
        // (FP16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_0_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, false, false}>>},
        // (FP16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, false, true}>>},
        // (FP16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, true, false}>>},
        // (FP16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, true, true}>>},
        // (FP16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_2_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, false, false}>>},
        // (FP16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, false, true}>>},
        // (FP16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, true, false}>>},
        // (FP16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, true, true}>>},
        // (FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_0_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, false, false}>>},
        // (FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, false, true}>>},
        // (FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, true, false}>>},
        // (FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, true, true}>>},
        // (FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, false, false}>>},
        // (FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, false, true}>>},
        // (FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, true, false}>>},
        // (FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, true, true}>>},
        // (BF16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, false, false}>>},
        // (BF16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, false, true}>>},
        // (BF16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, true, false}>>},
        // (BF16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, true, true}>>},
        // (BF16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_2_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, false, false}>>},
        // (BF16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, false, true}>>},
        // (BF16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true, false}>>},
        // (BF16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true, true}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_0_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, false, false}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, false, true}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, true, false}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, true, true}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_2_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, false, false}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, false, true}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, true, false}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, true, true}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_0_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, false, false}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, false, true}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, true, false}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, true, true}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, false}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, true}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, true, false}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, true, true}>>},
        // (BF16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_0_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, false, false}>>},
        // (BF16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, false, true}>>},
        // (BF16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, true, false}>>},
        // (BF16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, true, true}>>},
        // (BF16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_2_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, false, false}>>},
        // (BF16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, false, true}>>},
        // (BF16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, true, false}>>},
        // (BF16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, true, true}>>},
        // (BF16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_0_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, false, false}>>},
        // (BF16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, false, true}>>},
        // (BF16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, true, false}>>},
        // (BF16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, true, true}>>},
        // (BF16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, false, false}>>},
        // (BF16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, false, true}>>},
        // (BF16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, true, false}>>},
        // (BF16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, true, true}>>}
};
} // namespace flash