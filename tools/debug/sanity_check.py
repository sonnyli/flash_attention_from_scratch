#!/usr/bin/env python3
import argparse

import flash_attention
import torch
from flash_helpers.kernel_configs import get_kernel_configs
from flash_helpers.test.utils import (
    QKVConfig,
    evaluate_kernel,
    generate_qkv,
    reference_forward_kernel_v2,
)


def main():
    parser = argparse.ArgumentParser(
        description="",
    )
    parser.add_argument(
        "--small", action="store_true", dest="small", help="small test"
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        dest="print_diffs",
        help="print per row diffs",
    )
    parser.add_argument("--kernel", type=int, default=-1, help="kernel to test")
    args = parser.parse_args()

    if not args.small:
        batch_size = 16
        seq_len = 2048
        n_heads = 16
    else:
        batch_size = 1
        seq_len = 512
        n_heads = 1

    dtype = torch.bfloat16

    device = "cuda:0"

    cfgs = get_kernel_configs()
    if args.kernel >= 0:
        cfgs = [cfgs[args.kernel]]

    for d_head in [128]:
        print("d_head:", d_head)
        cfg = QKVConfig(
            n_heads=n_heads,
            d_head=d_head,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=dtype,
            device=device,
        )
        input_shape = (batch_size, seq_len, n_heads, d_head)

        q, k, v = generate_qkv(cfg)
        kernels = [cfg for cfg in cfgs if cfg.d_head == d_head]

        out_ref = reference_forward_kernel_v2(q, k, v).reshape(input_shape)

        for cfg in kernels:
            out = flash_attention.forward(cfg, q, k, v)
            evaluate_kernel(cfg, out_ref, out)

            if args.print_diffs:
                diff = (out - out_ref).abs() > 1e-3
                print(
                    diff.reshape((-1, diff.shape[-1])).sum(dim=-1, keepdim=True)
                )


if __name__ == "__main__":
    main()
