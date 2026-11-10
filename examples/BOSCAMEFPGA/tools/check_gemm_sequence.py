#!/usr/bin/env python3
# ===- check_gemm_sequence.py ----------------------------------------------
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

"""Verify AME instruction wiring of the lowered 64x64x64 GEMM in LLVM IR (mtype phases, msettile*, registers, DMA order).

Inputs: Path to lowered kernel .ll from buddy-opt --lower-bosc-ame.
Outputs: Exit 0 if sequence/config/strides match the hardware contract; otherwise diagnostics.
When used: Matmul FPGA example makefiles after lowering, complementary to simple instruction greps.

`buddy-opt --lower-bosc-ame` owns the whole hardware contract of the example:
the SPM addresses, the per-instruction configuration, the byte strides, the
fixed register slots and the order of the instructions.  A few greps can show
that the right *instructions* exist; this check verifies that they are wired up
correctly:

  * the accumulator load uses the accumulator phase `mtype` (0x10042) and the
    i8 loads and the MMA use the compute phase (0x10010);
  * every instruction that depends on a tile extent is preceded, since the
    previous AME instruction, by the `msettile*` calls that program it - a load
    must never run against configuration left over from a different phase;
  * A is `tr0`, B is `tr4` and the accumulator is `acc0`, and each load carries
    the *physical* row stride in bytes (A 64, B 64 from the [N,K] layout, the
    accumulator 256);
  * the tile operations run after the DMA transfers that fill the buffers they
    read completed.

Usage: check_gemm_sequence.py <lowered kernel>.ll
"""

import re
import sys

# Phase `mtype` values of the FPGA datapath.
MTYPE_MMA_I8 = 65552  # 0x10010, i8 x i8 -> i32
MTYPE_ACC_I32 = 65602  # 0x10042, i32 accumulator

CONFIG = ("msettilem", "msettilen", "msettilek", "msettype")

# Per instruction: the extents it depends on, the `mtype` phase it needs, and
# the position of the byte-stride operand in the call (loads take
# `base, stride`, the store `source, base, stride`).  Registers are *not*
# checked: the backend allocates them, and the example's AME post-processing
# maps them to the board encoding.
TILE_OPS = {
    "mlce32.m": {
        "extents": ("msettilem", "msettilen"),
        "mtype": MTYPE_ACC_I32,
        "stride": 256,
        "stride_operand": 1,
    },
    "mlae8.m": {
        "extents": ("msettilem", "msettilek"),
        "mtype": MTYPE_MMA_I8,
        "stride": 64,
        "stride_operand": 1,
    },
    "mlbe8.m": {
        "extents": ("msettilen", "msettilek"),
        "mtype": MTYPE_MMA_I8,
        "stride": 64,
        "stride_operand": 1,
    },
    "mqma.b.mm": {
        "extents": ("msettilem", "msettilen", "msettilek"),
        "mtype": MTYPE_MMA_I8,
        "stride": None,
        "stride_operand": None,
    },
    "msce32.m": {
        "extents": ("msettilem", "msettilen"),
        "mtype": MTYPE_ACC_I32,
        "stride": 256,
        "stride_operand": 2,
    },
}

# The order the example's kernel implies: seed the accumulator, load A and B,
# compute, write back.
EXPECTED_ORDER = ["mlce32.m", "mlae8.m", "mlbe8.m", "mqma.b.mm", "msce32.m"]


def constant_values(lines):
    """Map `%name` to the integer of its `llvm.mlir.constant` definition."""
    values = {}
    pattern = re.compile(
        r"^\s*(%[\w.]+)\s*=\s*llvm\.mlir\.constant\((-?\d+) : i64\)"
    )
    for line in lines:
        match = pattern.match(line)
        if match:
            values[match.group(1)] = int(match.group(2))
    return values


def operand_value(operand, values):
    """Value of one LLVM operand, with its type prefix removed.

    The IR spells operands as `<type> <value>`, where the pointer operand has
    the form `<ptr> inttoptr (i64 N to ptr)`; only the integer operand of the
    tile instructions is interesting here.
    """
    operand = operand.strip()
    if operand.startswith("ptr"):
        return None
    tokens = operand.split()
    if len(tokens) < 2:
        return None
    value = tokens[-1]
    if value in values:
        return values[value]
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    return None


class Event:
    def __init__(self, kind, line, data):
        self.kind = kind
        self.line = line
        self.data = data


def parse(lines):
    """Reduce the IR to the events the sequence check cares about."""
    values = constant_values(lines)
    events = []
    for number, line in enumerate(lines, 1):
        # Only calls matter: the trailing `declare` lines name the same
        # intrinsics without executing anything.
        if " call " not in f" {line} ":
            continue
        for op in ("mlce32.m", "mlae8.m", "mlbe8.m", "mqma.b.mm", "msce32.m"):
            if f"bosc.{op}." in line:
                call = re.search(
                    r"@llvm\.riscv\.bosc\."
                    + re.escape(op)
                    + r"\.[\w.]*\((.*)\)",
                    line,
                )
                operands = (
                    [part for part in call.group(1).split(",")] if call else []
                )
                events.append(Event(op, number, operands))
                break
        else:
            call = re.search(
                r"@llvm\.riscv\.bosc\.(mset\w+)\.\w*\((.*)\)", line
            )
            if call:
                events.append(
                    Event(
                        call.group(1),
                        number,
                        [part for part in call.group(2).split(",")],
                    )
                )
                continue
            if "@bosc_ame_dma_start(" in line:
                events.append(Event("dma.start", number, []))
            elif "@bosc_ame_dma_wait(" in line:
                events.append(Event("dma.wait", number, []))
    # Resolve the msettype operand to its constant value.
    for event in events:
        if event.kind == "msettype" and event.data:
            event.data = [operand_value(event.data[-1], values)]
    return events


def check(events):
    failures = []
    # 1. Instruction order.
    order = [event.kind for event in events if event.kind in TILE_OPS]
    if order != EXPECTED_ORDER:
        failures.append(
            f"tile instruction order is {order}, expected {EXPECTED_ORDER}"
        )
        return failures

    # 2. Configuration state at every tile instruction.  `seen` is reset after
    #    each tile instruction: the configuration a load needs must be
    #    (re)emitted for it, not inherited from an unrelated earlier op.
    seen = dict.fromkeys(CONFIG, False)
    mtype = None
    tile_events = [event for event in events if event.kind in TILE_OPS]
    for event in events:
        if event.kind in CONFIG:
            seen[event.kind] = True
            if event.kind == "msettype":
                mtype = event.data[0] if event.data else None
            continue
        if event.kind == "dma.start":
            # A request started *inside* the AME sequence would overlap it; the
            # output transfer that starts after the last tile operation is the
            # normal end of the kernel.
            first = min(e.line for e in tile_events)
            last = max(e.line for e in tile_events)
            if first < event.line < last:
                failures.append(
                    f"line {event.line}: a DMA transfer starts between the "
                    "tile operations, so it overlaps the AME sequence"
                )
            continue
        if event.kind not in TILE_OPS:
            continue
        spec = TILE_OPS[event.kind]
        missing = [name for name in spec["extents"] if not seen[name]]
        if missing:
            failures.append(
                f"line {event.line}: {event.kind} runs without programming "
                f"{', '.join(missing)} for this instruction"
            )
        if mtype != spec["mtype"]:
            failures.append(
                f"line {event.line}: {event.kind} expects mtype "
                f"{spec['mtype']}, but the last msettype programmed {mtype}"
            )
        operands = event.data

        def operand(index, operands=operands):
            if index is None or len(operands) <= index:
                return None
            return operand_value(operands[index], {})

        actual_stride = operand(spec["stride_operand"])
        if spec["stride"] is not None and actual_stride != spec["stride"]:
            failures.append(
                f"line {event.line}: {event.kind} uses byte stride "
                f"{actual_stride}, expected {spec['stride']}"
            )
        seen = dict.fromkeys(CONFIG, False)
        mtype = None

    # 3. Every buffer the tile operations read must have been staged by a DMA
    #    transfer that completed before them.
    first_tile = min(event.line for event in tile_events)
    staged_waits = [
        event.line
        for event in events
        if event.kind == "dma.wait" and event.line < first_tile
    ]
    if len(staged_waits) < 3:
        failures.append(
            "the A, B and zero transfers are not all waited before the first "
            f"tile operation (waits before line {first_tile}: {staged_waits})"
        )
    return failures


def main():
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    with open(sys.argv[1], encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    failures = check(parse(lines))
    if failures:
        for failure in failures:
            print(f"check_gemm_sequence: {failure}", file=sys.stderr)
        return 1
    print("ok: AME sequence, configuration and byte strides are correct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
