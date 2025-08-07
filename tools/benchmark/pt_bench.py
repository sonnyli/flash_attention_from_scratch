#!/usr/bin/env python3
"""This is a tool for comparing kernel runtimes.

The preferred tool is ncu_bench.py, which uses nsight compute due to its more hermetic
results. This tools tries to replicate the ncu profiling conditions using the techniques
in
https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch.
"""

import argparse
import statistics
import subprocess
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Callable, List

import flash_attention
import torch
from flash_helpers.kernel_configs import (
    DType,
    calc_self_attn_flop,
    get_kernel_configs,
)
from flash_helpers.test.utils import (
    BATCH_SIZE_FOR_SEQ_LEN,
    BENCHMARK_N_HEADS,
    QKVConfig,
    generate_qkvo,
    reference_forward_kernel_v2_timed,
    reference_forward_kernel_v3,
)
from prettytable import PrettyTable

# allocating 100MB for flushing L2 cache
x = torch.empty(int(100 * (1024**2)), dtype=torch.int8, device=0)


@dataclass
class BenchmarkStats:
    """Statistics for a benchmark run."""

    mean: float
    median: float
    min: float
    max: float
    stddev: float
    attn_tflops: float

    def relative_performance(self, baseline_mean: float) -> float:
        """Calculate relative performance as percentage compared to baseline."""
        return 100 * baseline_mean / self.mean


def calculate_benchmark_stats(
    runtime_samples: List[float], attn_flops: float
) -> BenchmarkStats:
    """Calculate statistics from runtime samples and compute TFLOP/s."""
    mean_runtime = statistics.mean(runtime_samples)
    median_runtime = statistics.median(runtime_samples)
    min_runtime = min(runtime_samples)
    max_runtime = max(runtime_samples)
    stddev = (
        statistics.stdev(runtime_samples) if len(runtime_samples) > 1 else 0
    )
    attn_tflops = (
        attn_flops / (mean_runtime * 1e6) / 1e3
    )  # convert ms to ns, then to TFLOP/s

    return BenchmarkStats(
        mean=mean_runtime,
        median=median_runtime,
        min=min_runtime,
        max=max_runtime,
        stddev=stddev,
        attn_tflops=attn_tflops,
    )


def get_cuda_device_info():
    """Return information about the CUDA device at index 0."""
    if not torch.cuda.is_available():
        return "CUDA not available"

    device = torch.cuda.get_device_properties(0)
    return {
        "name": device.name,
        "compute_capability": f"{device.major}.{device.minor}",
        "total_memory": f"{device.total_memory / (1024**3):.2f} GB",
        "multi_processor_count": device.multi_processor_count,
    }


def is_a100():
    return "A100" in get_cuda_device_info()["name"]


def flush_cache():
    x.zero_()


SHELL_OPTS = dict(
    shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)


def run_shell_command(cmd: str):
    subprocess.run(cmd, **SHELL_OPTS)


def set_clock_speed():
    run_shell_command("sudo nvidia-smi -pm ENABLED")
    # List of valid clock combinations can be found with
    # `nvidia-smi --query-supported-clocks=gr,mem --format=csv`

    # We want the clocks to be as high as possible, but not too high that they
    # fluctuate due to throttling.
    if is_a100():
        sm_clock = 1110
        mem_clock = 1512
    else:
        sm_clock = 1680
        mem_clock = 9501
    run_shell_command(f"sudo nvidia-smi -lgc {sm_clock}")
    run_shell_command(f"sudo nvidia-smi -lmc {mem_clock}")


def reset_clock_speed():
    run_shell_command("sudo nvidia-smi -pm ENABLED")
    sm_clock = 210
    mem_clock = 405
    run_shell_command(f"sudo nvidia-smi -lgc {sm_clock}")
    run_shell_command(f"sudo nvidia-smi -lmc {mem_clock}")
    run_shell_command("sudo nvidia-smi -rgc")


@dataclass
class BenchmarkData:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    o: torch.Tensor


@torch.inference_mode()
def benchmark_kernel(
    kernel: Callable[[None], None],
    n_warmups: int = 10,
    n_repeats: int = 50,
    ncu: bool = True,
):
    runtimes = []

    for _ in range(n_warmups):
        kernel()

    for _ in range(n_repeats):
        if ncu:
            flush_cache()
            torch.cuda._sleep(1_000_000)

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = kernel()
        end.record()
        torch.cuda.synchronize()
        if isinstance(out, tuple):
            runtimes.append(out[1])
        else:
            runtimes.append(start.elapsed_time(end))

    return runtimes


def generate_data(cfg: QKVConfig) -> BenchmarkData:
    return BenchmarkData(*generate_qkvo(cfg))


def parse_cmd_args():
    parser = argparse.ArgumentParser(
        description="Parse d_heads and seq_lens arguments, plus number of runs."
    )
    parser.add_argument(
        "--d_heads",
        type=str,
        default="128",
        help="Comma-separated list of integers for d_heads (e.g., '128,256,512').",
    )
    parser.add_argument(
        "--seq_lens",
        type=str,
        default="512,1024,2048,4096",
        help="Comma-separated list of integers for seq_lens (e.g., '64,128,256').",
    )
    parser.add_argument(
        "--num_warmups",
        type=int,
        default=10,
        help="Number of times to warm up the kernel.",
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=64,
        help="Number of times to run the kernel.",
    )
    parser.add_argument(
        "--noncu",
        action="store_true",
        help="do not try to stabilize",
    )

    args = parser.parse_args()

    # Convert the comma-separated string to a list of strings
    d_heads = list(map(int, args.d_heads.split(",")))
    seq_lens = list(map(int, args.seq_lens.split(",")))

    return d_heads, seq_lens, args.num_warmups, args.num_repeats, not args.noncu


def main():
    d_heads, seq_lens, num_warmups, num_repeats, ncu = parse_cmd_args()
    device = torch.device("cuda:0")

    n_heads = BENCHMARK_N_HEADS

    # Flag to control header printing - only print once
    first_table = True
    prev_seq_len = None

    def run_for_configs(configs, d_head, seq_len):
        nonlocal first_table, prev_seq_len
        batch_size = BATCH_SIZE_FOR_SEQ_LEN[seq_len]
        qkv_configs = {
            torch.float16: QKVConfig(
                n_heads=n_heads,
                d_head=d_head,
                batch_size=batch_size,
                seq_len=seq_len,
                dtype=torch.float16,
                device=device,
            ),
            torch.bfloat16: QKVConfig(
                n_heads=n_heads,
                d_head=d_head,
                batch_size=batch_size,
                seq_len=seq_len,
                dtype=torch.bfloat16,
                device=device,
            ),
        }

        runtimes = {}

        dtype_to_data = {
            torch.float16: generate_data(qkv_configs[torch.float16]),
            torch.bfloat16: generate_data(qkv_configs[torch.bfloat16]),
        }
        fp_data = dtype_to_data[torch.float16]

        def data_for_kernel(cfg):
            return dtype_to_data[cfg.dtype.to_torch_dtype()]

        kernels = OrderedDict()

        for cfg in configs:
            data = data_for_kernel(cfg)
            cfg_kernel = partial(
                flash_attention.forward_timed,
                kernel_cfg=cfg,
                q=data.q,
                k=data.k,
                v=data.v,
                o=data.o,
            )
            kernels[cfg] = partial(cfg_kernel)

        for cfg, kernel in kernels.items():
            runtime_samples = benchmark_kernel(
                kernel,
                n_repeats=num_repeats,
                n_warmups=num_warmups,
                ncu=ncu,
            )
            runtimes[cfg] = runtime_samples

        v2_times = benchmark_kernel(
            partial(
                reference_forward_kernel_v2_timed,
                q=fp_data.q,
                k=fp_data.k,
                v=fp_data.v,
                o=fp_data.o,
            ),
            n_repeats=num_repeats,
            n_warmups=num_warmups,
            ncu=ncu,
        )
        v3_times = benchmark_kernel(
            partial(
                reference_forward_kernel_v3,
                q=fp_data.q,
                k=fp_data.k,
                v=fp_data.v,
                o=fp_data.o,
            ),
            n_repeats=num_repeats,
            n_warmups=num_warmups,
            ncu=ncu,
        )

        # Sort by mean runtime
        runtimes = sorted(runtimes.items(), key=lambda x: statistics.mean(x[1]))
        field_names = [
            "Kernel Name",
            "d_head",
            "seq_len",
            "Mean (ms)",
            "Median (ms)",
            "Min (ms)",
            "Max (ms)",
            "StdDev (ms)",
            "Relative Performance",
            "Attn TFLOP/s",
        ]
        table = PrettyTable(field_names=field_names)
        table.align["Kernel Name"] = "l"
        table.align["d_head"] = "r"
        table.align["seq_len"] = "r"
        table.align["Mean (ms)"] = "r"
        table.align["Median (ms)"] = "r"
        table.align["Min (ms)"] = "r"
        table.align["Max (ms)"] = "r"
        table.align["StdDev (ms)"] = "r"
        table.align["Relative Performance"] = "r"
        table.align["Attn TFLOP/s"] = "r"

        # Calculate attn FLOP for this configuration
        attn_flops = calc_self_attn_flop(batch_size, n_heads, seq_len, d_head)

        # Add the reference kernels
        v2_stats = calculate_benchmark_stats(v2_times, attn_flops)
        table.add_row(
            [
                "Reference",
                f"{d_head}",
                f"{seq_len}",
                f"{v2_stats.mean:.4f}",
                f"{v2_stats.median:.4f}",
                f"{v2_stats.min:.4f}",
                f"{v2_stats.max:.4f}",
                f"{v2_stats.stddev:.4f}",
                "100.00%",
                f"{v2_stats.attn_tflops:.4f}",
            ]
        )

        v3_stats = calculate_benchmark_stats(v3_times, attn_flops)
        table.add_row(
            [
                "V3",
                f"{d_head}",
                f"{seq_len}",
                f"{v3_stats.mean:.4f}",
                f"{v3_stats.median:.4f}",
                f"{v3_stats.min:.4f}",
                f"{v3_stats.max:.4f}",
                f"{v3_stats.stddev:.4f}",
                f"{v3_stats.relative_performance(v2_stats.mean):.2f}%",
                f"{v3_stats.attn_tflops:.4f}",
            ]
        )

        # Add the custom kernels
        for cfg, runtime_samples in runtimes:
            stats = calculate_benchmark_stats(runtime_samples, attn_flops)
            table.add_row(
                [
                    cfg.short_form(),
                    f"{d_head}",
                    f"{seq_len}",
                    f"{stats.mean:.4f}",
                    f"{stats.median:.4f}",
                    f"{stats.min:.4f}",
                    f"{stats.max:.4f}",
                    f"{stats.stddev:.4f}",
                    f"{stats.relative_performance(v2_stats.mean):.2f}%",
                    f"{stats.attn_tflops:.4f}",
                ]
            )

        # Update prev_seq_len for next iteration
        prev_seq_len = seq_len

        # Only print the header for the first table
        print(table.get_csv_string(header=first_table))
        first_table = False

    set_clock_speed()
    for d_head in d_heads:
        for seq_len in seq_lens:
            run_for_configs(
                get_kernel_configs(),
                d_head,
                seq_len,
            )

    reset_clock_speed()


if __name__ == "__main__":
    main()
