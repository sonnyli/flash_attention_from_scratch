#!/usr/bin/env python3

import argparse
import sys

import flash_attention
import torch
from flash_helpers.kernel_configs import (
    DType,
    get_kernel_configs,
    parse_kernel_name_into_config,
)
from flash_helpers.test.utils import (
    BATCH_SIZE_FOR_SEQ_LEN,
    BENCHMARK_N_HEADS,
    QKVConfig,
    generate_qkv,
    reference_forward_kernel_v2,
    reference_forward_kernel_v3,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run all flash attention kernels once"
    )
    parser.add_argument(
        "seq_len", type=int, nargs="?", default=4096, help="Sequence length"
    )
    parser.add_argument(
        "d_head", type=int, nargs="?", default=128, help="Head dimension"
    )
    parser.add_argument(
        "--ref", action="store_true", help="Run reference kernels"
    )
    parser.add_argument(
        "--ref_v3", action="store_true", help="Run reference v3 kernel"
    )
    parser.add_argument(
        "--n_runs", type=int, required=True, help="Run kernels N times"
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        help="List of kernel config strings to parse and run.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="FP16",
        help="Dtype for reference kernels (FP16 or BF16, default: BF16)",
    )
    args = parser.parse_args()

    # Error checking for command line arguments
    if args.seq_len <= 0 or args.d_head <= 0:
        print("Error: seq_len and d_head must be positive integers")
        sys.exit(1)

    if args.n_runs <= 0:
        print("Error: n_runs must be a positive integer")
        sys.exit(1)

    # Parse dtype argument
    try:
        ref_dtype = DType.from_string(args.dtype)
    except ValueError as e:
        print(f"Error parsing dtype: {e}")
        sys.exit(1)

    # Process kernel selection and parsing
    selected_kernels = []
    if args.kernels is not None:
        for kernel_str in args.kernels:
            try:
                kernel_config = parse_kernel_name_into_config(kernel_str)
                selected_kernels.append(kernel_config)
            except Exception as e:
                print(f"Error parsing kernel config '{kernel_str}': {e}")
                sys.exit(1)

    batch_size = BATCH_SIZE_FOR_SEQ_LEN[args.seq_len]
    n_heads = BENCHMARK_N_HEADS
    device = "cuda:0"

    # Create a map with two sets of q, k, v and O with either bfloat16 or float16 dtypes
    dtype_tensor_map = {}

    for dtype_enum in [DType.FP16, DType.BF16]:
        torch_dtype = dtype_enum.to_torch_dtype()

        q, k, v = generate_qkv(
            QKVConfig(
                n_heads=n_heads,
                d_head=args.d_head,
                batch_size=batch_size,
                seq_len=args.seq_len,
                dtype=torch_dtype,
                device=device,
            )
        )
        o = torch.empty_like(q)

        dtype_tensor_map[dtype_enum] = {"q": q, "k": k, "v": v, "o": o}

    torch.cuda.synchronize()

    if args.ref:
        # Run reference kernels with specified dtype only
        tensors = dtype_tensor_map[ref_dtype]
        torch_dtype = ref_dtype.to_torch_dtype()
        print(f"Running reference kernels with dtype: {ref_dtype.name}")

        for _ in range(args.n_runs):
            _ = reference_forward_kernel_v2(
                tensors["q"],
                tensors["k"],
                tensors["v"],
                tensors["o"],
            )
            torch.cuda.synchronize()

    if args.ref_v3:
        # Run reference v3 kernel with specified dtype only
        tensors = dtype_tensor_map[ref_dtype]
        torch_dtype = ref_dtype.to_torch_dtype()
        print(f"Running reference v3 kernel with dtype: {ref_dtype.name}")

        for _ in range(args.n_runs):
            _ = reference_forward_kernel_v3(
                tensors["q"],
                tensors["k"],
                tensors["v"],
                tensors["o"],
            )
            torch.cuda.synchronize()

    # Run selected kernels with their corresponding dtypes
    for cfg in selected_kernels:
        # Pick dtype for selected kernels from FlashForwardKernelConfig objects
        cfg_dtype = cfg.dtype
        tensors = dtype_tensor_map[cfg_dtype]

        print(f"Running kernel {cfg.short_form()} with dtype: {cfg_dtype.name}")

        # Create fresh output tensor
        o = torch.empty_like(tensors["q"])
        torch.cuda.synchronize()

        for _ in range(args.n_runs):
            _ = flash_attention.forward(
                cfg, tensors["q"], tensors["k"], tensors["v"], o
            )
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
