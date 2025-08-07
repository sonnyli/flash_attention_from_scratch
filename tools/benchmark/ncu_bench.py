#!/usr/bin/env python3

import argparse
import csv
import itertools
import logging
import os
import re
import subprocess
import sys

from flash_helpers.kernel_configs import (
    calc_self_attn_flop,
    calc_total_flop,
    parse_flash_forward_kernel_config,
    parse_kernel_name_into_config,
)
from flash_helpers.test.utils import BENCHMARK_BATCH_SIZE, BENCHMARK_N_HEADS
from prettytable import PrettyTable

# --------------------------------------------------
# 1) Adjust this dictionary to add/remove metrics
#    that you want from the ncu CSV output.
# --------------------------------------------------
# Keys here are your "logical" metric names, as you'll see in your Python code.
# The "csv_name" is how the metric appears in the ncu CSV (under "Metric Name").
# "display_name" is how it will appear in the PrettyTable.
# "format_fn" is a function to format the numeric value for printing.
# "show_ratio" indicates whether to show a ratio column (comparing to baseline).
# --------------------------------------------------
METRICS_MAP = {
    "Duration": {
        "csv_name": "Duration",
        "display_name": "Dur (ms)",
        "format_fn": lambda val: f"{val / 1e6}",
        "show_ratio": True,
    },
    "Cycles": {
        "csv_name": "Elapsed Cycles",
        "display_name": "Cycles",
        "format_fn": lambda val: f"{val:.0f}",
        "show_ratio": True,
    },
    "Registers Per Thread": {
        "csv_name": "Registers Per Thread",
        "display_name": "Regs",
        "format_fn": lambda val: f"{val:.0f}",
        "show_ratio": False,
    },
    "L2 Hit Rate": {
        "csv_name": "L2 Hit Rate",
        "display_name": "L2 Hit %",
        "format_fn": lambda val: f"{val:.3f}",
        "show_ratio": False,
    },
}


def get_git_commit():
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        return commit_hash
    except subprocess.CalledProcessError:
        return "Not a git repository or no commits found"


REFERENCE_V2 = "V2"
REFERENCE_V3 = "V3"
REFERENCE_KERNELS = [REFERENCE_V2, REFERENCE_V3]


def parse_ncu_csv_rows(csv_reader):
    """
    Given a csv.DictReader (already positioned at the first data row),
    parse and return a dictionary of kernel metrics, keyed by kernel name.
    Example return:
      {
        REFERENCE_KERNEL_NAME: {
          "Duration": 123,
          "Cycles": 456,
          "Registers Per Thread": 32
        },
        "('x', 'y')": {...},
        ...
      }
    """
    kernels = {}

    for row in csv_reader:
        kernel_name = row["Kernel Name"]
        metric_name = row["Metric Name"]
        metric_value = row["Metric Value"]

        # Normalize the kernel name
        if "flash_fwd_kernel" in kernel_name:
            kernel_name = REFERENCE_V2
        elif "device_kernel" in kernel_name:
            kernel_name = REFERENCE_V3
        else:
            kernel_name = parse_flash_forward_kernel_config(
                kernel_name
            ).short_form()

        if kernel_name not in kernels:
            # Initialize all known metrics to 0.0
            kernels[kernel_name] = {m: 0.0 for m in METRICS_MAP.keys()}

        # If this metric name is in our METRICS_MAP, record it
        for user_metric, info in METRICS_MAP.items():
            if metric_name == info["csv_name"]:
                kernels[kernel_name][user_metric] = float(
                    metric_value.replace(",", "")
                )
                break

    return kernels


def merge_metrics_into_aggregator(aggregator, run_results):
    """
    Merges the single-run dictionary (run_results) into the aggregator.
    The aggregator is of the form:
      {
        kernel_name: {
          "count": int,
          "Duration": float,
          "Cycles": float,
          ...
        }
      }
    Summations are used, so we can average later.
    """
    for kname, metrics in run_results.items():
        if kname not in aggregator:
            aggregator[kname] = {"count": 0}
            # Initialize aggregator metrics to 0.0
            for m in METRICS_MAP.keys():
                aggregator[kname][m] = 0.0

        # Sum the metrics
        for m in METRICS_MAP.keys():
            aggregator[kname][m] += metrics[m]
        aggregator[kname]["count"] += 1


def average_aggregator(aggregator):
    """
    Convert sums in the aggregator to averages by dividing by 'count'.
    Returns a dict of the same structure but with final averaged numbers.
    """
    averaged = {}
    for kname, data in aggregator.items():
        count = data["count"]
        if count == 0:
            continue
        # Copy and replace sums with averages
        averaged[kname] = {}
        for m in METRICS_MAP.keys():
            averaged[kname][m] = data[m] / count
    return averaged


REFERENCE_BLOCK_SIZES = {(128, False): (128, 32)}


def generate_results_table(
    kernels,
    d_head,
    seq_len,
    print_csv=False,
    sort=True,
):
    """
    Takes a dictionary of kernel metrics (already averaged) and prints a table.
    'kernels' is of the form:
       {
         REFERENCE_KERNEL_NAME: {
           "Duration": X,
           "Cycles": Y,
           "Registers Per Thread": Z,
           ...
         },
         "('x', 'y')": {
           ...
         },
         ...
       }
    """
    if not kernels:
        print("No results to display.")
        return

    # Pick a baseline kernel (prefer REFERENCE_KERNEL_NAME)
    if REFERENCE_V2 in kernels:
        baseline_metrics = kernels[REFERENCE_V2]
    else:
        first_kname = list(kernels.keys())[0]
        baseline_metrics = kernels[first_kname]

    # Sort so that 'ref' is first, others by ascending Duration
    def get_duration(kname_data):
        kname, data = kname_data
        return data.get("Duration", float("inf"))

    # Separate ref from others
    ref_v2_item = (
        (REFERENCE_V2, kernels[REFERENCE_V2])
        if REFERENCE_V2 in kernels
        else None
    )
    ref_v3_item = (
        (REFERENCE_V3, kernels[REFERENCE_V3])
        if REFERENCE_V3 in kernels
        else None
    )
    other_items = [
        (k, v) for k, v in kernels.items() if k not in REFERENCE_KERNELS
    ]

    if sort:
        # Sort the non-ref items by Duration
        other_items.sort(key=get_duration)

    # Final list
    kernels_sorted = []
    if ref_v2_item:
        kernels_sorted.append(ref_v2_item)
    if ref_v3_item:
        kernels_sorted.append(ref_v3_item)
    kernels_sorted.extend(other_items)

    # Prepare the table
    # We'll put "Kernel Name" first, then each metric in METRICS_MAP,
    # plus ratio columns for those that set "show_ratio" to True.
    # Example columns: [ "Kernel Name", "<disp>", "% <disp>", "<disp2>", "% <disp2>", ... ]
    header = ["Kernel Name"]
    # We'll keep track of how we'll print each metric column
    # so the table rows can be built in the same order.
    metric_cols = []
    for m_key, info in METRICS_MAP.items():
        header.append(info["display_name"])
        metric_cols.append(m_key)
        if info["show_ratio"]:
            header.append(f"% inv {info['display_name']}")

    header.append("TFLOP/s")
    header.append("Attn TFLOP/s")

    if print_csv:
        header.extend(["d_head", "seq_len"])
    table = PrettyTable()
    table.field_names = header
    table.align["Kernel Name"] = "l"

    # Add rows
    for kernel_name, metrics in kernels_sorted:
        row = [kernel_name]
        for m_key in metric_cols:
            val = metrics[m_key]
            val_str = METRICS_MAP[m_key]["format_fn"](val)
            row.append(val_str)

            # Show ratio if needed
            if METRICS_MAP[m_key]["show_ratio"]:
                baseline_val = baseline_metrics[m_key]
                if baseline_val == 0:
                    ratio_str = "-"
                else:
                    ratio_str = f"{100.0 / (val / baseline_val)}%"
                row.append(ratio_str)

        duration_ns = metrics["Duration"]
        if kernel_name in REFERENCE_KERNELS:
            causal = False
            B_r, B_c = REFERENCE_BLOCK_SIZES[(d_head, causal)]
            flop = calc_total_flop(
                BENCHMARK_BATCH_SIZE,
                BENCHMARK_N_HEADS,
                seq_len,
                B_r,
                B_c,
                d_head,
            )
        else:
            flop = parse_kernel_name_into_config(kernel_name).total_flop(
                BENCHMARK_BATCH_SIZE, BENCHMARK_N_HEADS, seq_len
            )

        tflops = flop / duration_ns / 1e3
        attn_flops = (
            calc_self_attn_flop(
                BENCHMARK_BATCH_SIZE, BENCHMARK_N_HEADS, seq_len, d_head
            )
            / duration_ns
            / 1e3
        )
        row.append(f"{tflops}")
        row.append(f"{attn_flops}")

        if print_csv:
            row.extend([str(d_head), str(seq_len)])
        table.add_row(row)

    return table


def call_ncu_and_store_output(program: list[str]):
    """
    Call the `ncu` CLI with the given arguments, return a dict of kernel metrics.
    """
    try:
        # Run the command
        result = subprocess.run(
            [
                "ncu",
                "--csv",
                "--section=MemoryWorkloadAnalysis",
                "--set=basic",
                "-k",
                'regex:"device|flash"',
            ]
            + program,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Error running ncu: {result.stderr}")
            return {}

        # Skip lines until we reach the header row (the one that starts with "ID")
        output_lines = result.stdout.splitlines()
        if not output_lines:
            return {}

        start_idx = 0
        for line in output_lines:
            if not line.startswith('"ID"'):
                start_idx += 1
            else:
                break
        output_lines = output_lines[start_idx:]

        csv_reader = csv.DictReader(output_lines)
        # Parse into a dictionary
        return parse_ncu_csv_rows(csv_reader)

    except FileNotFoundError:
        print(
            "The `ncu` command was not found. Ensure it's installed and in your PATH."
        )
        return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}


PROGRAM = ["./tools/benchmark/run_kernels.py", "-kernels=all"]


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
        default="1024",
        help="Comma-separated list of integers for seq_lens (e.g., '64,128,256').",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to run each config (for averaging).",
    )
    parser.add_argument(
        "--csv", action="store_true", dest="csv", help="Output as csv"
    )
    parser.add_argument(
        "--no_sort",
        action="store_true",
        dest="no_sort",
        help="do not sort input by duration",
    )

    args = parser.parse_args()

    # Convert the comma-separated string to a list of strings
    d_heads = list(map(str, args.d_heads.split(",")))
    seq_lens = list(map(str, args.seq_lens.split(",")))

    return d_heads, seq_lens, args.runs, args.csv, not args.no_sort


def get_highest_profile_number(directory):
    profile_number = 0
    for file in os.listdir(directory):
        match = re.match(r"profile_(\d+)\.", file)
        if match:
            number = int(match.group(1))
            profile_number = max(profile_number, number)
    return profile_number


PROFILE_DIR = "./profiles/local_profiles"

if __name__ == "__main__":
    d_heads, seq_lens, runs, print_csv, sort = parse_cmd_args()
    n_profile = get_highest_profile_number(PROFILE_DIR) + 1
    ext = "csv" if print_csv else "txt"
    output_path = os.path.join(PROFILE_DIR, f"profile_{n_profile}.{ext}")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(output_path),
            logging.StreamHandler(sys.stdout),
        ],
    )

    commit = get_git_commit()
    if not print_csv:
        logging.info(f"commit: {commit}")

    first = True
    for d_head, seq_len in itertools.product(d_heads, seq_lens):
        if not print_csv:
            logging.info(f"--- d_head={d_head}, seq_len={seq_len} ---")

        # Accumulate results here
        aggregator = {}

        # Run multiple times
        for _ in range(runs):
            run_metrics = call_ncu_and_store_output(PROGRAM + [seq_len, d_head])
            merge_metrics_into_aggregator(aggregator, run_metrics)

        # Average results
        averaged_results = average_aggregator(aggregator)

        # Print the final table
        table = generate_results_table(
            averaged_results, int(d_head), int(seq_len), print_csv, sort=sort
        )
        if not print_csv:
            logging.info(table)
            logging.info("")
        else:
            header = first
            first = False

            logging.info(table.get_csv_string(header=header).strip())
