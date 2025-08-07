#!/usr/bin/env python3

import csv
import re
import subprocess
import sys

import flash_helpers.kernel_configs as kernel_configs


def demangle_function_name(mangled_name):
    """
    Attempt to demangle a C++ function name. If `cxxfilt` is available,
    use it. Otherwise, try external `c++filt`. If that also fails,
    return the original mangled name.
    """
    try:
        output = (
            subprocess.check_output(["cu++filt", mangled_name])
            .decode("utf-8")
            .strip()
        )
        return output
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 3) Fallback: Return the original mangled name
    return mangled_name


def parse_ptxas_log(logfile_path):
    """
    Parse a ptxas log containing lines like:
      - "... Compiling entry function '...' for 'sm_86'"
      - "... N bytes stack frame, M bytes spill stores, P bytes spill loads"
      - "... Used U registers, used B barriers, [Z bytes cumulative stack size,] C bytes cmem[0]"
      - "... Compile time = T ms"

    Return a list of dictionaries with the parsed info.
    """

    # Regex that allows for text like timestamps at the start of the line ('.*')
    re_compile_func = re.compile(
        r".*Compiling entry function '(?P<func_name>[^']+)' for 'sm_(?P<arch>\d+)'"
    )
    re_stack_spill = re.compile(
        r".*?(?P<stack>\d+)\s+bytes stack frame,\s+"
        r"(?P<spill_stores>\d+)\s+bytes spill stores,\s+"
        r"(?P<spill_loads>\d+)\s+bytes spill loads"
    )
    # This pattern makes the "XYZ bytes cumulative stack size," part optional:
    #   (?: ... )?  means "zero or one times"
    re_used_line = re.compile(
        r".*Used\s+(?P<regs>\d+)\s+registers,\s+"
        r"used\s+(?P<barriers>\d+)\s+barriers,"
        r"(?:\s+(?P<cum_stack>\d+)\s+bytes cumulative stack size,)?"
        r"\s+(?P<cmem0>\d+)\s+bytes cmem\[0\]"
    )
    re_compile_time = re.compile(r".*Compile time = (?P<time_ms>[\d\.]+)\s+ms")

    functions_info = []
    current_func = None

    with open(logfile_path, "r", encoding="utf-8") as f:
        for line in f:
            # Remove trailing newline but keep leading spaces if any
            line = line.rstrip("\n")

            # 1) "Compiling entry function '...' for 'sm_XX'"
            match_compile = re_compile_func.search(line)
            if match_compile:
                # If there's a previous function open with no compile_time, close it
                if current_func and "compile_time" not in current_func:
                    functions_info.append(current_func)

                mangled_name = match_compile.group("func_name")
                demangled_name = demangle_function_name(mangled_name)

                current_func = {
                    "function_name_mangled": mangled_name,
                    "function_name_demangled": demangled_name,
                    "architecture": "sm_" + match_compile.group("arch"),
                    "stack_frame": None,
                    "spill_stores": None,
                    "spill_loads": None,
                    "used_registers": None,
                    "used_barriers": None,
                    "cumulative_stack": None,
                    "cmem0": None,
                    "compile_time": None,
                }
                continue

            # If we haven't started a function context, skip
            if current_func is None:
                continue

            # 2) "N bytes stack frame, M bytes spill stores, P bytes spill loads"
            match_stack = re_stack_spill.search(line)
            if match_stack:
                current_func["stack_frame"] = int(match_stack.group("stack"))
                current_func["spill_stores"] = int(
                    match_stack.group("spill_stores")
                )
                current_func["spill_loads"] = int(
                    match_stack.group("spill_loads")
                )
                continue

            # 3) "Used X registers, used Y barriers, [Z bytes cumulative stack size,] W bytes cmem[0]"
            match_used = re_used_line.search(line)
            if match_used:
                current_func["used_registers"] = int(match_used.group("regs"))
                current_func["used_barriers"] = int(
                    match_used.group("barriers")
                )
                # If there's no cumulative stack portion in the line, group("cum_stack") will be None
                cum_stack = match_used.group("cum_stack")
                current_func["cumulative_stack"] = (
                    int(cum_stack) if cum_stack else None
                )

                current_func["cmem0"] = int(match_used.group("cmem0"))
                continue

            # 4) "Compile time = T ms"
            match_time = re_compile_time.search(line)
            if match_time:
                current_func["compile_time"] = float(
                    match_time.group("time_ms")
                )
                # Store the function and reset
                functions_info.append(current_func)
                current_func = None
                continue

    # If the file ended but the last function didn't have compile_time, store partial info
    if current_func and "compile_time" not in current_func:
        functions_info.append(current_func)

    return functions_info


def print_functions_info(functions_info):
    """
    Print all function information in a nicely formatted way.
    """
    for i, info in enumerate(functions_info, start=1):
        # If you prefer to see zeros instead of None, do something like:
        #   cum_stack = info.get("cumulative_stack") or 0
        # but by default let's just display None for missing data.
        cfg = kernel_configs.parse_kernel_name_into_config(
            info["function_name_demangled"]
        )
        print(f"=== Function #{i} ===")
        print(f"  Name        : {cfg.short_form()}")
        print(f"  Architecture          : {info['architecture']}")
        if info["stack_frame"] != 0:
            print(f"  Stack frame (bytes)   : {info['stack_frame']}")
        if info["spill_stores"] != 0:
            print(f"  Spill stores (bytes)  : {info['spill_stores']}")
        if info["spill_loads"] != 0:
            print(f"  Spill loads (bytes)   : {info['spill_loads']}")
        print(f"  Used registers        : {info['used_registers']}")
        print(f"  Used barriers         : {info['used_barriers']}")
        if info.get("cumulative_stack") is not None:
            print(f"  Cumulative stack      : {info['cumulative_stack']}")
        print(f"  cmem[0] (bytes)       : {info['cmem0']}")
        print(f"  Compile time (ms)     : {info['compile_time']}")
        print()


def write_csv_output(functions_info, output_file=None):
    """
    Write function information to a CSV file or stdout if output_file is None.
    Ignores cumulative_stack, cmem0, architecture, and compile_time fields.
    """
    fieldnames = [
        "kernel_config",
        "spilled",
        "stack_frame",
        "spill_stores",
        "spill_loads",
        "used_registers",
    ]

    # Create a file or use stdout
    if output_file:
        f = open(output_file, "w", newline="")
    else:
        f = sys.stdout

    try:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for info in functions_info:
            cfg = kernel_configs.parse_kernel_name_into_config(
                info["function_name_demangled"]
            )

            # Replace None values with 0
            stack_frame = info["stack_frame"] or 0
            spill_stores = info["spill_stores"] or 0
            spill_loads = info["spill_loads"] or 0

            row = {
                "kernel_config": cfg.short_form(),
                "spilled": stack_frame > 0
                or spill_stores > 0
                or spill_loads > 0,
                "stack_frame": stack_frame,
                "spill_stores": spill_stores,
                "spill_loads": spill_loads,
                "used_registers": info["used_registers"],
            }
            writer.writerow(row)
    finally:
        if output_file and f is not sys.stdout:
            f.close()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} LOGFILE [--csv] [--output OUTPUT_FILE]")
        sys.exit(1)

    logfile_path = sys.argv[1]
    csv_format = "--csv" in sys.argv
    output_file = None

    # Check if --output is specified
    if "--output" in sys.argv:
        output_index = sys.argv.index("--output")
        if output_index + 1 < len(sys.argv):
            output_file = sys.argv[output_index + 1]

    parsed_info = parse_ptxas_log(logfile_path)
    if not parsed_info:
        print("No functions were parsed from the log file.")
    else:
        if csv_format:
            write_csv_output(parsed_info, output_file)
        else:
            print_functions_info(parsed_info)


if __name__ == "__main__":
    main()
