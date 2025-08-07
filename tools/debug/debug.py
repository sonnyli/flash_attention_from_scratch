#!/usr/bin/env python3
import argparse
import math
import sys
from typing import IO

import flash_attention
import torch
from flash_attn import flash_attn_func
from flash_helpers.kernel_configs import get_kernel_configs
from flash_helpers.test.utils import evaluate_kernel
from wurlitzer import pipes

PRINT_CHUNKS = False
INIT_RANGE = False


def print_tensor(tensor, out=sys.stdout):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")
    rows = list(tensor)
    sep_rows = 8
    sep_cols = 8
    for i in range(0, len(rows), sep_rows):
        group = rows[i : i + sep_rows]
        for row in group:
            # Group elements in chunks of 16
            elements = [elem.item() for elem in row]
            for i in range(0, len(elements), sep_cols):
                chunk = elements[i : i + sep_cols]
                out.write(" ".join("{:>5.2f}".format(x) for x in chunk))
                if i + sep_cols < len(
                    elements
                ):  # Don't add space after last group
                    out.write("  ")  # Two spaces between groups
            out.write("\n")
        out.write("\n")  # Add extra space between groups


def block_flash_attention(
    d_head,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    B_r: int,
    B_c: int,
    out: IO,
    n_warps: int = 4,
):
    warp_offset = B_r // n_warps
    warp_rank = 2
    base_seq = 0
    q_base_seq = warp_rank * warp_offset + base_seq
    row_idxs = slice(
        base_seq + warp_rank * warp_offset,
        base_seq + (warp_rank + 1) * warp_offset,
    )
    global d
    ks = k.split(B_c, dim=0)
    vs = v.split(B_c, dim=0)

    d_head = q.shape[-1]

    Q = q[row_idxs]

    out.write("Q:\n")
    print_tensor(Q, out)

    M = torch.full_like(Q, float("-inf"))[:, :1]
    L = torch.zeros_like(M)
    O = torch.zeros_like(Q)
    chunks = len(ks)

    fused_softmax = True
    softmax_scale = (d_head**-0.5) * math.log2(math.e)

    def pp(**kwargs):
        qk_chunks = d_head / 16
        v_chunks = B_c / 16
        k, v = next(iter(kwargs.items()))
        # if k in ["Q", "O"]:
        #     v = v[q_base_seq : q_base_seq + 16]
        #     if PRINT_CHUNKS:
        #         v = torch.stack(v.chunk(qk_chunks, dim=-1), dim=0)
        if k in ["K"]:
            v = v[base_seq : base_seq + 64]
            if PRINT_CHUNKS:
                v = torch.stack(v.chunk(qk_chunks, dim=-1), dim=0)
        elif k == "V":
            v = v[base_seq : base_seq + 64].T
            if PRINT_CHUNKS:
                v = torch.stack(v.chunk(v_chunks, dim=-1), dim=0)
        # elif k in ["L", "M"]:
        #     v = v[row_idxs]
        # elif v.shape[1] == 1:
        #     v = v[:, 0]

        out.write(f"{k}_{i}:\n")
        print_tensor(v, out)
        out.write("\n\n")

    for i in reversed(range(chunks)):
        K, V = ks[i], vs[i]

        S_pre_scaling = Q @ K.T
        if not fused_softmax:
            S = Q @ K.T / (d_head**0.5)
            M_new = torch.maximum(M, S.max(dim=-1, keepdim=True).values)

            M_delta_exp = (M - M_new).exp()

            P = (S - M_new).exp()
            L = M_delta_exp * L + P.sum(dim=-1, keepdim=True)
            if i > 0:
                O_scaled = O * M_delta_exp
            else:
                O_scaled = O
            PV = P @ V
            O_new = O_scaled + PV

        else:
            if i > 0:
                M_new = torch.maximum(
                    M, S_pre_scaling.max(dim=-1, keepdim=True).values
                )
            else:
                M_new = S_pre_scaling.max(dim=-1, keepdim=True).values
            scale = 2 ** ((M - M_new) * softmax_scale)
            L *= scale
            O_scaled = O * scale
            max_scaled = M_new * softmax_scale
            P = 2 ** (S_pre_scaling * softmax_scale - max_scaled)
            L += P.sum(dim=-1, keepdim=True)
            PV = P @ V
            O_new = O_scaled + PV

        if i == 0:
            pp(Q=Q)
        pp(K=K)
        pp(S_pre_scaling=S_pre_scaling)
        pp(M=M_new)
        pp(L=L)
        pp(P=P)
        pp(V=V)
        pp(O_scale=O_scaled)
        pp(O=O_new)

        O = O_new
        M = M_new

    O_final = O / L
    pp(O_final=O / L)
    return O_final


def self_attention(q, k, v):
    P = ((q @ k.T) / (q.shape[-1] ** 0.5)).softmax(dim=-1)
    return P @ v


def main():
    # Add argument parser
    parser = argparse.ArgumentParser(
        description="Debug flash attention implementation"
    )
    parser.add_argument(
        "kernel_idx", type=int, help="Index of kernel configuration to use"
    )
    parser.add_argument(
        "seq_B_r",
        type=int,
        help="Seq len in terms of multiple of B_r",
        default=2,
    )
    parser.add_argument(
        "-r",
        "--range",
        action="store_true",
        help="Initialize tensors with range values instead of random",
    )
    args = parser.parse_args()

    # Set INIT_RANGE based on command line argument
    global INIT_RANGE
    INIT_RANGE = args.range

    kernel_idx = args.kernel_idx
    kernel = get_kernel_configs("all")[kernel_idx]

    batch_size = 1
    seq_len = kernel.B_r * args.seq_B_r
    n_heads = 1

    d_head = 128
    dtype = torch.bfloat16

    device = "cuda:0"
    input_shape = (batch_size, seq_len, n_heads, d_head)
    torch.cuda.set_device(device)
    if INIT_RANGE:
        q = torch.arange(
            0, torch.tensor(input_shape).prod(), dtype=dtype, device=device
        ).reshape(input_shape)
        k = -q.clone()
        v = q.clone()
    else:
        q = torch.randn(input_shape, dtype=dtype, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

    print("q.shape:", q.shape)
    print("k.shape:", k.shape)
    print("v.shape:", v.shape)
    ref = flash_attn_func(q, k, v, causal=False)

    with open("debug/debug_pt.txt", "w") as pt_file:
        block_flash_attention(
            d_head,
            q[0, :, 0],
            k[0, :, 0],
            v[0, :, 0],
            kernel.B_r,
            kernel.B_c,
            pt_file,
        )
        print("O_full:", file=pt_file)
        print_tensor(
            self_attention(q[0, :, 0], k[0, :, 0], v[0, :, 0])[: kernel.B_r],
            pt_file,
        )
        print("O_ref:", file=pt_file)
        print_tensor(ref[0, :, 0], pt_file)
    with open("debug/debug_cuda.txt", "w") as cuda_file:
        with pipes(stdout=cuda_file):
            out = flash_attention.forward(kernel, q, k, v).reshape(input_shape)
            torch.cuda.synchronize()
            print("O_gmem:", file=cuda_file)
            print_tensor(out[0, :, 0], cuda_file)

    evaluate_kernel(kernel, ref, out)

    diff = (out - ref).abs() > 1e-3

    print("summed across d_head axis: |")
    print(diff.sum(dim=-1).reshape(-1))
    print()
    print("summed across seq axis: ---")
    print(diff.sum(dim=1).reshape(-1))


if __name__ == "__main__":
    main()
