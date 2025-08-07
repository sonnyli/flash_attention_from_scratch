#!/usr/bin/env python3
"""
Script to compare instruction counts between two files.
Shows before/after values and absolute deltas.
"""

import argparse
import sys


def parse_count_file(filepath):
    """Parse a count file and return a dictionary of instruction -> count."""
    counts = {}
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Split on first space to separate count from instruction
                parts = line.split(None, 1)
                if len(parts) == 2:
                    count_str, instruction = parts
                    try:
                        count = int(count_str)
                        counts[instruction] = count
                    except ValueError:
                        print(
                            f"Warning: Could not parse count '{count_str}' in line: {line}"
                        )
                        continue
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        sys.exit(1)

    return counts


def compare_counts(before_counts, after_counts):
    """Compare two count dictionaries and return the differences."""
    all_instructions = set(before_counts.keys()) | set(after_counts.keys())

    results = []
    for instruction in sorted(all_instructions):
        before = before_counts.get(instruction, 0)
        after = after_counts.get(instruction, 0)
        delta = after - before
        abs_delta = abs(delta)

        # Calculate relative delta as percentage
        if before == 0:
            rel_delta = float("inf") if after > 0 else 0
        else:
            rel_delta = (delta / before) * 100

        if delta != 0:  # Only show instructions that changed
            results.append(
                {
                    "instruction": instruction,
                    "before": before,
                    "after": after,
                    "delta": delta,
                    "abs_delta": abs_delta,
                    "rel_delta": rel_delta,
                }
            )

    return results


def print_comparison(results, before_file, after_file):
    """Print the comparison results in markdown table format."""
    print("## Instruction Count Comparison")
    print(f"- **Before:** {before_file}")
    print(f"- **After:** {after_file}")
    print()

    if not results:
        print("No differences found!")
        return

    # Sort by absolute delta (largest changes first)
    results.sort(key=lambda x: x["abs_delta"], reverse=True)

    # Markdown table headers
    print("| Instruction | Before | After | Delta | Rel Delta |")
    print("|:------------|-------:|------:|------:|----------:|")

    # Data rows
    for result in results:
        instruction = result["instruction"]
        before = result["before"]
        after = result["after"]
        delta = result["delta"]
        rel_delta = result["rel_delta"]

        # Format delta with + sign for positive values
        delta_str = str(delta)
        if delta > 0:
            delta_str = f"+{delta}"

        # Format relative delta
        if rel_delta == float("inf"):
            rel_delta_str = "∞%"
        elif rel_delta == float("-inf"):
            rel_delta_str = "-∞%"
        else:
            rel_delta_str = f"{rel_delta:+.1f}%"

        print(
            f"| {instruction} | {before} | {after} | {delta_str} | {rel_delta_str} |"
        )

    print()
    print(f"**Total instructions changed:** {len(results)}")

    # Summary statistics
    increases = [r for r in results if r["delta"] > 0]
    decreases = [r for r in results if r["delta"] < 0]

    if increases:
        print(f"**Instructions increased:** {len(increases)}")
        print(
            f"- Largest increase: {max(increases, key=lambda x: x['delta'])['instruction']} (+{max(r['delta'] for r in increases)})"
        )

    if decreases:
        print(f"**Instructions decreased:** {len(decreases)}")
        print(
            f"- Largest decrease: {max(decreases, key=lambda x: x['abs_delta'])['instruction']} ({min(r['delta'] for r in decreases)})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compare instruction counts between two files"
    )
    parser.add_argument(
        "before_file", help='File with "before" instruction counts'
    )
    parser.add_argument(
        "after_file", help='File with "after" instruction counts'
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all instructions, including unchanged ones",
    )

    args = parser.parse_args()

    # Parse both files
    before_counts = parse_count_file(args.before_file)
    after_counts = parse_count_file(args.after_file)

    if not before_counts and not after_counts:
        print("No data found in either file!")
        sys.exit(1)

    # Compare the counts
    results = compare_counts(before_counts, after_counts)

    # If --all flag is used, include unchanged instructions too
    if args.all:
        all_instructions = set(before_counts.keys()) | set(after_counts.keys())
        all_results = []
        for instruction in sorted(all_instructions):
            before = before_counts.get(instruction, 0)
            after = after_counts.get(instruction, 0)
            delta = after - before
            abs_delta = abs(delta)

            # Calculate relative delta as percentage
            if before == 0:
                rel_delta = float("inf") if after > 0 else 0
            else:
                rel_delta = (delta / before) * 100

            all_results.append(
                {
                    "instruction": instruction,
                    "before": before,
                    "after": after,
                    "delta": delta,
                    "abs_delta": abs_delta,
                    "rel_delta": rel_delta,
                }
            )
        results = all_results

    # Print the comparison
    print_comparison(results, args.before_file, args.after_file)


if __name__ == "__main__":
    main()
