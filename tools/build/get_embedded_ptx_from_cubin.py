#!/usr/bin/env python3
import sys

from flash_helpers.build.ptx_sass import (
    get_cuobjdump_elf_dump,
    get_section_lines,
)


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python get_embedded_ptx_from_cubin.py <cubin_filename> [section_name]"
        )
        sys.exit(1)

    cubin_filename = sys.argv[1]

    # Default to .nv_debug_ptx_txt section if not specified
    section_name = sys.argv[2] if len(sys.argv) >= 3 else ".nv_debug_ptx_txt"

    # Get the ELF dump from cuobjdump
    lines = get_cuobjdump_elf_dump(cubin_filename)

    # Extract the section lines
    section_lines = get_section_lines(lines, section_name)

    if section_lines:
        # Skip the section header line
        for line in section_lines[1:]:
            print(line)
    else:
        print(f"Section '{section_name}' not found.")


if __name__ == "__main__":
    main()
