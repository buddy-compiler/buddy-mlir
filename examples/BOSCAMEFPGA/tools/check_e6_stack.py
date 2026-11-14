#!/usr/bin/env python3
# ===- check_e6_stack.py ---------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------

"""Conservatively bound the linked E6 call-chain stack depth, including assembly register saves.

Inputs: Linked ELF path plus optional objdump/nm, entry/trap symbols, and explicit --indirect-call maps.
Outputs: JSON/text stack bound report or non-zero exit on unbounded/unknown SP updates.
When used: Post-link NR/NH-E6 image audits in FPGA example makefiles.

Walk both sides of every conditional branch and follow direct calls and tail
jumps. Propagate known register values for SP arithmetic, alignment and frame
pointer restoration. Reject unknown SP writes, unbounded stack loops,
recursion and unmapped indirect calls. This is a conservative control-flow
bound, not a proof of branch feasibility or arbitrary memory safety.

An indirect-call mapping is an explicit caller promise about all possible
targets (for these images, the installed and default DMA failure hooks).
It does not prove that a writable function pointer cannot be changed later.
The compiler-generated code is assumed to obey the RISC-V calling convention.
"""

import argparse
import hashlib
import json
import re
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from nr_isa import is_ame, parse_disassembly, validate_ame_word

REGISTERS = {"zero", "ra", "sp", "gp", "tp"} | {
    f"{prefix}{i}"
    for prefix, count in (("a", 8), ("s", 12), ("t", 7))
    for i in range(count)
}
PRESERVED = {"sp", "gp", "tp", "zero"} | {f"s{i}" for i in range(12)}
BRANCHES = {
    "beq",
    "bne",
    "blt",
    "bge",
    "bltu",
    "bgeu",
    "beqz",
    "bnez",
    "blez",
    "bgez",
    "bltz",
    "bgtz",
    "ble",
    "bgt",
    "bleu",
    "bgtu",
}
STORES = {"sb", "sh", "sw", "sd", "fsw", "fsd"}


@dataclass(frozen=True)
class Function:
    name: str
    start: int
    size: int


class AuditError(ValueError):
    pass


def immediate(value):
    return int(value, 0)


def signed(value, bits):
    return (value + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))


class Auditor:
    def __init__(self, functions, instructions, stack_top, indirect):
        self.functions = functions
        self.instructions = instructions
        self.stack_top = stack_top
        self.indirect = indirect
        self.cache = {}
        self.active = set()
        self.visited_calls = set()

    def owner(self, address):
        for function in self.functions.values():
            if function.start <= address < function.start + function.size:
                return function
        raise AuditError(f"no function owns instruction {address:#x}")

    def target(self, instruction):
        match = re.search(r"<([^>]+)>", instruction["instruction"])
        if match:
            symbol = match[1]
            if symbol in self.functions:
                return self.functions[symbol].start
        operand = instruction["instruction"].split("<", 1)[0]
        matches = re.findall(r"0x[0-9a-f]+", operand)
        if matches and instruction["mnemonic"] not in {"jalr", "jr"}:
            return int(matches[-1], 16)
        raise AuditError(f"unresolved target at {instruction['address']:#x}")

    def run(self, name, entry_sp, start=None):
        function = self.functions[name]
        start = function.start if start is None else start
        key = (name, entry_sp, start)
        if key in self.cache:
            return self.cache[key]
        if name in self.active:
            raise AuditError(f"recursive call/tail cycle through {name}")
        self.active.add(name)
        states = {start: {"sp": entry_sp, "zero": 0}}
        pending = deque([start])
        result = {
            "peak": self.stack_top - entry_sp,
            "path": [name],
            "address": hex(start),
            "returns": False,
        }

        def enqueue(pc, registers):
            if not function.start <= pc < function.start + function.size:
                raise AuditError(f"fallthrough leaves {name} at {pc:#x}")
            previous = states.get(pc)
            if previous is None:
                states[pc] = registers.copy()
                pending.append(pc)
            else:
                if previous.get("sp") != registers.get("sp"):
                    raise AuditError(
                        f"inconsistent/unbounded SP at {name}:{pc:#x}"
                    )
                merged = {
                    k: v for k, v in previous.items() if registers.get(k) == v
                }
                if merged != previous:
                    states[pc] = merged
                    pending.append(pc)

        def call(target, registers, pc):
            child = self.owner(target)
            self.visited_calls.add((name, hex(pc), child.name))
            summary = self.run(child.name, registers["sp"], target)
            if summary["peak"] > result["peak"]:
                result.update(
                    peak=summary["peak"],
                    path=[name, *summary["path"]],
                    address=summary["address"],
                )
            return summary["returns"]

        while pending:
            pc = pending.popleft()
            registers = states[pc].copy()
            instruction = self.instructions.get(pc)
            if instruction is None:
                raise AuditError(f"missing instruction at {name}:{pc:#x}")
            op = instruction["mnemonic"]
            args = instruction["instruction"].split(None, 1)
            args = (
                []
                if len(args) == 1
                else args[1].split("<", 1)[0].strip().split(", ")
            )
            next_pc = pc + instruction["size"]
            if op == "ret":
                if registers["sp"] != entry_sp:
                    raise AuditError(
                        f"unrestored SP at return from {name}:{pc:#x}"
                    )
                result["returns"] = True
                continue
            if op in {"unimp", "ebreak", "c.unimp"}:
                continue
            if op in BRANCHES:
                enqueue(self.target(instruction), registers)
                enqueue(next_pc, registers)
                continue
            if op in {"j", "jr", "jal", "jalr"}:
                tail = op in {"j", "jr"}
                if op == "j":
                    target = self.target(instruction)
                    if (
                        function.start
                        <= target
                        < function.start + function.size
                    ):
                        enqueue(target, registers)
                        continue
                    returning = call(target, registers, pc)
                elif (
                    op in {"jalr", "jr"}
                    and "<" not in instruction["instruction"]
                ):
                    targets = self.indirect.get(name)
                    if not targets:
                        raise AuditError(
                            f"unmapped indirect call at {name}:{pc:#x}"
                        )
                    returning = False
                    for target_name in targets:
                        returning |= call(
                            self.functions[target_name].start, registers, pc
                        )
                else:
                    returning = call(self.target(instruction), registers, pc)
                if tail:
                    if returning and registers["sp"] != entry_sp:
                        raise AuditError(
                            f"unrestored SP at tail return from {name}"
                        )
                    result["returns"] |= returning
                elif returning:
                    enqueue(
                        next_pc,
                        {k: v for k, v in registers.items() if k in PRESERVED},
                    )
                continue
            if op in {".word", "<unknown>"}:
                if not is_ame(instruction["word"]):
                    raise AuditError(f"unknown instruction at {name}:{pc:#x}")
                validate_ame_word(instruction["word"])
            elif args and args[0] in REGISTERS and op not in STORES:
                destination = args[0]
                value = None
                if op == "mv":
                    value = registers.get(args[1])
                elif op == "li":
                    value = immediate(args[1])
                elif op in {"lui", "auipc"}:
                    value = signed(immediate(args[1]), 20) << 12
                    if op == "auipc":
                        value += pc
                elif op in {"addi", "andi"} and args[1] in registers:
                    left, right = registers[args[1]], immediate(args[2])
                    value = left + right if op == "addi" else left & right
                elif op in {"add", "sub"} and all(
                    a in registers for a in args[1:]
                ):
                    left, right = (registers[a] for a in args[1:])
                    value = left + right if op == "add" else left - right
                if destination == "sp" and value is None:
                    raise AuditError(
                        f"unknown SP write at {name}:{pc:#x}: {args}"
                    )
                registers.pop(destination, None)
                if value is not None and destination != "zero":
                    registers[destination] = value
                registers["zero"] = 0
            depth = self.stack_top - registers["sp"]
            if depth < 0:
                raise AuditError(f"SP above stack top at {name}:{pc:#x}")
            if depth > result["peak"]:
                result.update(peak=depth, path=[name], address=hex(pc))
            enqueue(next_pc, registers)
        self.active.remove(name)
        self.cache[key] = result
        return result


def load(elf, objdump, nm):
    symbols = subprocess.check_output(
        [nm, "-S", "--defined-only", str(elf)], text=True
    )
    functions, values = {}, {}
    for line in symbols.splitlines():
        match = re.fullmatch(r"([0-9a-f]+) ([0-9a-f]+) ([A-Za-z]) (.+)", line)
        if not match:
            continue
        address, size, kind, name = match.groups()
        address, size = int(address, 16), int(size, 16)
        values[name] = address
        if kind in {"T", "t"} and not name.startswith("."):
            functions[name] = Function(name, address, size)
    ordered = sorted(functions.values(), key=lambda f: f.start)
    for i, function in enumerate(ordered):
        if function.size == 0 and i + 1 < len(ordered):
            functions[function.name] = Function(
                function.name,
                function.start,
                ordered[i + 1].start - function.start,
            )
    dump = subprocess.check_output([objdump, "-d", str(elf)], text=True)
    instructions = {i["address"]: i for i in parse_disassembly(dump)}
    return functions, instructions, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("elf", type=Path)
    parser.add_argument("--objdump", default="llvm-objdump")
    parser.add_argument("--nm", default="llvm-nm")
    parser.add_argument("--entry", default="e6_start")
    parser.add_argument("--trap-entry", default="e6_trap_entry")
    parser.add_argument(
        "--indirect-call",
        action="append",
        default=[],
        metavar="CALLER=TARGET,TARGET",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        indirect = {}
        for item in args.indirect_call:
            caller, targets = item.split("=", 1)
            indirect[caller] = targets.split(",")
        functions, instructions, values = load(args.elf, args.objdump, args.nm)
        top, bottom = values["__e6_stack_top"], values["__e6_stack_bottom"]
        auditor = Auditor(functions, instructions, top, indirect)
        normal = auditor.run(args.entry, top)
        # A trap runs on the interrupted stack. Audit its incremental use and
        # add it to the deepest normal/failure-hook chain, conservatively.
        trap = auditor.run(args.trap_entry, top - normal["peak"])
        report = {
            "elf_sha256": hashlib.sha256(args.elf.read_bytes()).hexdigest(),
            "entry": args.entry,
            "stack_capacity": top - bottom,
            "peak_bytes": trap["peak"],
            "normal_and_hook_peak": normal,
            "trap_extra_bytes": trap["peak"] - normal["peak"],
            "remaining_bytes": top - bottom - trap["peak"],
            "indirect_target_promises": indirect,
            "calls": [list(c) for c in sorted(auditor.visited_calls)],
            "status": "PASS" if trap["peak"] <= top - bottom else "FAIL",
        }
        output = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.write_text(output)
        print(output, end="")
        return report["status"] != "PASS"
    except (ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"stack audit failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
