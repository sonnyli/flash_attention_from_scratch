import subprocess
from collections import defaultdict

from flash_helpers.build.ptx_instruction import (
    Instruction,
    PtxLine,
    parse_ptx_line,
)


def filter_line(line: str) -> bool:
    line = line.strip()
    if not line or line.startswith("//") or line.startswith("."):
        return True
    return False


def clean_sass_line(line: str) -> str:
    return " ".join(line.split()[1:])


def parse_ptx_to_sass_mapping(
    lines: list[str],
) -> dict[int, list[tuple[int, str]]]:
    """
    Parse a file that contains nvdisasm output with PTX line references.

    Returns a dictionary mapping PTX line numbers to a list of tuples,
    where each tuple contains (original_line_number, sass_instruction).

    Example input:
    //## File ".nv_debug_ptx_txt", line 35377
    /*0000*/                   MOV R1, c[0x0][0x28] ;
    //## File ".nv_debug_ptx_txt", line 35405
    /*0010*/                   S2R R25, SR_CTAID.Y ;

    Example output:
    {
        35377: [(2, "/*0000*/                   MOV R1, c[0x0][0x28] ;")],
        35405: [(4, "/*0010*/                   S2R R25, SR_CTAID.Y ;")]
    }
    """
    ptx_to_sass = defaultdict(list)
    current_ptx_line = None

    for line_num, line in enumerate(lines, 1):
        # Check if this is a PTX line reference
        if 'File ".nv_debug_ptx_txt"' in line:
            # Extract the line number from the format: //## File ".nv_debug_ptx_txt", line 35377
            current_ptx_line = int(line.split()[-1])

        # Check if this is a SASS instruction line (starts with /*XXXX*/)
        elif (
            current_ptx_line is not None
            and line.strip().startswith("/*")
            and "*/" in line
        ):
            # Associate this SASS line with the current PTX line
            ptx_to_sass[current_ptx_line].append(clean_sass_line(line))
        else:
            current_ptx_line = None

    return ptx_to_sass


### cuobjdump + nvdisasm helpers


def section_line_numbers(lines: list[str]) -> list[int]:
    return [n for n, line in enumerate(lines) if line.startswith(".section")]


def find_section_size(lines: list[str], section_name: str) -> int:
    sections = section_line_numbers(lines)
    for n, section in enumerate(sections):
        line = lines[section]
        if line.strip() == f".section {section_name}":
            if n < len(sections) - 1:
                return sections[n + 1] - section
            else:
                return len(lines) - section
    return 0


def get_section_lines(lines: list[str], section_name: str) -> list[str]:
    sections = section_line_numbers(lines)
    for n, section in enumerate(sections):
        line = lines[section]
        if line.strip() == f".section {section_name}":
            start = section
            if n < len(sections) - 1:
                end = sections[n + 1]
            else:
                end = len(lines)
            return lines[start:end]
    return []


def get_cuobjdump_elf_dump(filename: str, demangle: bool = True) -> list[str]:
    # also possible to do this with readelf to fetch a particular symbol but cuobjdump's output is cleaner by default
    result = subprocess.run(
        ["cuobjdump", "-elf", filename], capture_output=True, text=True
    ).stdout

    if demangle:
        result = subprocess.run(
            ["cu++filt", "-p"], input=result, capture_output=True, text=True
        ).stdout

    return result.splitlines()


def get_ptx_instructions_from_cuobjdump(filename: str) -> list[PtxLine]:
    lines = get_cuobjdump_elf_dump(filename)
    ptx_lines = get_section_lines(lines, ".nv_debug_ptx_txt")[1:]

    return [
        parse_ptx_line(line, line_number)
        for line_number, line in enumerate(ptx_lines, 1)
    ]


def get_sass_lines_from_nvdisasm(filename: str, with_ptx: bool) -> list[str]:
    cmd = ["nvdisasm", "-c"]
    if with_ptx:
        cmd.append("-gp")
    cmd.append(filename)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.splitlines()


def get_ptx_to_sass_mapping_from_nvdisasm(
    filename: str,
) -> dict[int, list[tuple[int, str]]]:
    lines = get_sass_lines_from_nvdisasm(filename, True)
    return parse_ptx_to_sass_mapping(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ptx_sass.py <file_path> [section_name]")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        lines = f.readlines()

    if len(sys.argv) >= 3:
        section_name = sys.argv[2]
        size = find_section_size(lines, section_name)

        section_lines = get_section_lines(lines, section_name)
        if section_lines:
            for line in section_lines[1:]:
                print(line, end="")
        else:
            print(f"Section '{section_name}' not found.")

    else:
        sections = section_line_numbers(lines)
        for n, section in enumerate(sections):
            line = lines[section]
            if line.strip() == ".section .nv_debug_ptx_txt":
                print(sections[n + 1] - section)
