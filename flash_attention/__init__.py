# ruff: noqa: F401

import torch as _  # isort: skip
import flash_attention_kernels


def forward(kernel_cfg, q, k, v, o=None):
    return flash_attention_kernels.forward(
        kernel_cfg, q, k, v, o, benchmark=False
    )[0]


def forward_timed(kernel_cfg, q, k, v, o=None):
    val, runtime = flash_attention_kernels.forward(
        kernel_cfg, q, k, v, o, benchmark=True
    )
    return val, runtime
