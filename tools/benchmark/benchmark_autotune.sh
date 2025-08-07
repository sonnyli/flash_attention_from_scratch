#!/usr/bin/env bash

seq_lens="${1:-1024,2048,4096,8192,16384}"
KERNELS=tune tools/benchmark/ncu_bench.py --seq_lens="${seq_lens}" --d_heads=128 --runs=1 --csv > profiles/autotune.csv