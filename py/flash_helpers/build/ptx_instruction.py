import re
from abc import ABC
from dataclasses import dataclass


@dataclass
class PtxLine(ABC):
    line_number: int
    raw_line: str


@dataclass
class UnusedLine(PtxLine):
    pass


@dataclass
class Label(PtxLine):
    pass


@dataclass
class Directive(PtxLine):
    def is_entry_point(self) -> bool:
        return ".entry" in self.raw_line


@dataclass
class Instruction(PtxLine):
    op: str
    predicate: str | None
    dst_register: str | None
    src_registers: list[str]


def is_register(operand: str) -> bool:
    return "%" in operand


def parse_register(operand: str) -> str:
    match = re.search(r"%[a-zA-Z]+[0-9]+", operand)
    if match is None:
        return None
    return match.group()


def clean_operand(operand: str) -> str:
    if is_register(operand):
        reg = parse_register(operand)
        return reg if reg else operand
    return operand


def get_op_and_registers(line: str) -> tuple[str, list[str]]:
    line = line.strip().strip("{}")
    parts = line.split()
    start = 0
    predicate = parts[0] if parts[0].startswith("@") else None
    if predicate is not None:
        start = 1
    op = parts[start]
    operands = [clean_operand(operand) for operand in parts[start + 1 :]]

    dst_reg = None
    src_regs = []
    if operands:
        dst_reg = operands[0] if is_register(operands[0]) else None
        src_regs = [operand for operand in operands[1:] if is_register(operand)]
    return op, predicate, dst_reg, src_regs


def parse_ptx_line(line: str, line_number: int) -> PtxLine:
    line = line.strip()
    if len(line) <= 1:
        return UnusedLine(line_number=line_number, raw_line=line)
    elif line.startswith("."):
        return Directive(line_number=line_number, raw_line=line)
    elif line.startswith("$"):
        return Label(line_number=line_number, raw_line=line)
    else:
        op, predicate, dst_reg, src_regs = get_op_and_registers(line)

        raw_line = " ".join(line.split())
        return Instruction(
            line_number=line_number,
            raw_line=raw_line,
            op=op,
            predicate=predicate,
            dst_register=dst_reg,
            src_registers=src_regs,
        )
