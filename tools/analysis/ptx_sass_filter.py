#!/usr/bin/env python3

import argparse

from flash_helpers.build.ptx_instruction import Directive, Instruction, PtxLine
from flash_helpers.build.ptx_sass import (
    get_ptx_instructions_from_cuobjdump,
    get_ptx_to_sass_mapping_from_nvdisasm,
)


def filter_unrelated_instructions(
    instructions: list[PtxLine],
    line_number: int,
) -> list[PtxLine]:
    base_instr = instructions[line_number]
    if not isinstance(base_instr, Instruction):
        raise ValueError(
            f"Base line {base_instr.raw_line} is not an instruction"
        )
    relevant_regs = set(base_instr.src_registers)

    filtered_instructions = [base_instr]
    for instr in reversed(instructions[:line_number]):
        if isinstance(instr, Directive):
            if instr.is_entry_point():
                break
        if not isinstance(instr, Instruction):
            continue

        if instr.dst_register not in relevant_regs:
            continue

        relevant_regs.update(instr.src_registers)
        filtered_instructions.append(instr)

    return filtered_instructions[::-1]


def main():
    parser = argparse.ArgumentParser(
        description="Extract specific lines from a file"
    )
    parser.add_argument("filename", help="Path to the file to process")
    parser.add_argument("line_number", type=int, help="Line number (1-indexed)")
    parser.add_argument(
        "-ptx",
        action="store_true",
    )
    parser.add_argument(
        "-sass",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        action="store_true",
        help="Print the line numbers of the instructions",
    )

    args = parser.parse_args()

    ptx_instructions = get_ptx_instructions_from_cuobjdump(args.filename)

    line_number = args.line_number - 1

    filtered_instructions = filter_unrelated_instructions(
        ptx_instructions, line_number
    )
    if not args.ptx:
        assert args.sass, "Must provide SASS output if not providing PTX"
        mapping = get_ptx_to_sass_mapping_from_nvdisasm(args.filename)
        for instr in filtered_instructions:
            sass_instructions = mapping[instr.line_number]
            for sass_instr in sass_instructions:
                print(sass_instr)
        return

    if args.ptx:
        for instr in filtered_instructions:
            if args.n:
                print(f"{instr.line_number}: {instr.raw_line}")
            else:
                print(instr.raw_line)
    else:
        mapping = get_ptx_to_sass_mapping_from_nvdisasm(args.filename)

        # First pass to find the maximum length of PTX instructions for alignment
        max_ptx_length = 0
        for instr in filtered_instructions:
            max_ptx_length = max(max_ptx_length, len(instr.raw_line))

        # Format string with dynamic padding based on max PTX length
        format_string = "{:" + str(max_ptx_length + 2) + "s} {}"

        prev_instr_mapped = False
        for instr in filtered_instructions:
            if prev_instr_mapped:
                print()
            if instr.line_number in mapping:
                prev_instr_mapped = True
                sass_instructions = mapping[instr.line_number]
                # Print the first SASS instruction on the same line as PTX
                if sass_instructions:
                    print(
                        format_string.format(
                            instr.raw_line, sass_instructions[0]
                        )
                    )
                    # Print any additional SASS instructions with proper alignment
                    for sass_instr in sass_instructions[1:]:
                        padding = " " * (max_ptx_length + 3)
                        print(f"{padding}{sass_instr}")
                else:
                    print(format_string.format(instr.raw_line, ""))
            else:
                prev_instr_mapped = False
                print(format_string.format(instr.raw_line, ""))


if __name__ == "__main__":
    main()
