#!/usr/bin/env python3
"""Rewrite RVV sequences unsupported by the current BOSC FPGA prototype.

LLVM lowers extractelement on an i1 vector to a mask-register move.  The
prototype does not reliably complete vmv.x.s for that sequence.  When the
mask was produced by comparing vid.v with a scalar tail length, materialize
the same low 16 mask bits with scalar integer instructions instead.

The current RISC-V backend can also allocate a hidden structure-return pointer
inside the outgoing-argument area of very large MLIR functions.  Preserve the
affected layer-wise entry points in private static slots so later calls do not
overwrite the pointer before the result descriptor is returned.
"""

from __future__ import annotations

import re
import sys
from collections import deque


VID_RE = re.compile(r"^\s*vid\.v\s+(v\d+)\s*(?:#.*)?$")
MASK_RE = re.compile(
    r"^\s*vmslt\.vx\s+(v\d+),\s*(v\d+),\s*([a-z][a-z0-9]*)\s*(?:#.*)?$"
)
EXTRACT_RE = re.compile(
    r"^(\s*)vmv\.x\.s\s+([a-z][a-z0-9]*),\s*(v\d+)\s*(?:#.*)?$"
)
VECTOR_DEST_RE = re.compile(r"^\s*v[a-z0-9.]+\s+(v\d+)(?:,|\s|$)")
SLIDE_RE = re.compile(
    r"^(\s*)vslideup\.vx\s+(v\d+),\s*(v\d+),\s*([a-z][a-z0-9]*)\s*$"
)
VSET_TU_RE = re.compile(
    r"^\s*vsetvli\s+zero,\s*([a-z][a-z0-9]*),\s*e32,\s*m1,\s*tu,\s*ma\s*$"
)
ADDI_ONE_RE = re.compile(
    r"^\s*addi\s+([a-z][a-z0-9]*),\s*([a-z][a-z0-9]*),\s*1\s*$"
)
FMV_SCALAR_RE = re.compile(r"^\s*vfmv\.s\.f\s+(v\d+),")
FUNCTION_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):")
SRET_FIXES = {
    "subgraph0_decoder_layer_prefill": ".Lbuddy_decoder_prefill_sret_ptr",
    "subgraph0_final_head": ".Lbuddy_final_head_sret_ptr",
    "subgraph0_decoder_layer_decode": ".Lbuddy_decoder_decode_sret_ptr",
}


def scalar_mask(indent: str, destination: str, threshold: str, index: int) -> str:
    scratch = next(
        register
        for register in ("t6", "t5", "t4")
        if register not in (destination, threshold)
    )
    zero = f".L__buddy_rvv_mask_{index}_zero"
    all_set = f".L__buddy_rvv_mask_{index}_all"
    done = f".L__buddy_rvv_mask_{index}_done"
    instructions = [
        f"{indent}# vmv.x.s mask extraction lowered for BOSC FPGA",
        f"{indent}addi\tsp, sp, -16",
        f"{indent}sd\t{scratch}, 0(sp)",
        f"{indent}blez\t{threshold}, {zero}",
        f"{indent}li\t{scratch}, 16",
        f"{indent}bge\t{threshold}, {scratch}, {all_set}",
        f"{indent}li\t{destination}, 1",
        f"{indent}sll\t{destination}, {destination}, {threshold}",
        f"{indent}addi\t{destination}, {destination}, -1",
        f"{indent}j\t{done}",
        f"{zero}:",
        f"{indent}li\t{destination}, 0",
        f"{indent}j\t{done}",
        f"{all_set}:",
        f"{indent}li\t{destination}, 65535",
        f"{done}:",
        f"{indent}ld\t{scratch}, 0(sp)",
        f"{indent}addi\tsp, sp, 16",
    ]
    return "\n".join(instructions)


def scalar_insert(
    indent: str,
    destination: str,
    source: str,
    lane: str,
    avl: str,
    index: int,
) -> str:
    scratch = [
        register
        for register in ("t6", "t5", "t4", "t3")
        if register not in (lane, avl)
    ][:3]
    base, value, address = scratch
    instructions = [
        f"{indent}# vslideup scalar insertion lowered for BOSC FPGA",
        f"{indent}addi\tsp, sp, -224",
        f"{indent}sd\t{base}, 192(sp)",
        f"{indent}sd\t{value}, 200(sp)",
        f"{indent}sd\t{address}, 208(sp)",
        f"{indent}addi\t{base}, sp, 63",
        f"{indent}andi\t{base}, {base}, -64",
        f"{indent}vsetivli\tzero, 16, e32, m1, ta, ma",
        f"{indent}vse32.v\t{destination}, ({base})",
        f"{indent}fence\trw, rw",
        f"{indent}addi\t{address}, {base}, 64",
        f"{indent}vse32.v\t{source}, ({address})",
        f"{indent}fence\trw, rw",
        f"{indent}lw\t{value}, 0({address})",
        f"{indent}slli\t{address}, {lane}, 2",
        f"{indent}add\t{address}, {base}, {address}",
        f"{indent}sw\t{value}, 0({address})",
        f"{indent}fence\trw, rw",
        f"{indent}vle32.v\t{destination}, ({base})",
        f"{indent}ld\t{base}, 192(sp)",
        f"{indent}ld\t{value}, 200(sp)",
        f"{indent}ld\t{address}, 208(sp)",
        f"{indent}addi\tsp, sp, 224",
        f"{indent}vsetvli\tzero, {avl}, e32, m1, tu, ma",
    ]
    return "\n".join(instructions)


def main() -> int:
    vid_registers: set[str] = set()
    masks: dict[str, str] = {}
    replacement_count = 0
    slide_count = 0
    current_function: str | None = None
    sret_functions_seen: set[str] = set()
    sret_stores: set[str] = set()
    sret_loads: set[str] = set()
    sret_offsets: dict[str, str] = {}
    sret_call_seen: set[str] = set()
    recent: deque[str] = deque(maxlen=3)

    for raw_line in sys.stdin:
        line = raw_line.rstrip("\n")
        function = FUNCTION_RE.match(line)
        if function:
            current_function = (
                function.group(1) if function.group(1) in SRET_FIXES else None
            )
            if current_function is not None:
                sret_functions_seen.add(current_function)
        if current_function in SRET_FIXES:
            symbol = SRET_FIXES[current_function]
            if re.match(r"^\s*call\s+", line):
                sret_call_seen.add(current_function)
            store = None
            # The hidden result pointer is the first a0 stack spill in these
            # generated entry points and is saved immediately before their
            # first call.  Discover its frame offset instead of enumerating
            # offsets: register allocation changes it whenever the W8A8
            # schedule, profiling, or vector legalization changes.
            if (
                current_function not in sret_stores
                and current_function not in sret_call_seen
            ):
                store = re.match(
                    r"^(\s*)sd\s+a0,\s*(-[0-9]+)"
                    r"\(s0\)\s*(?:#.*)?$",
                    line,
                )
            if store:
                indent = store.group(1)
                print(f"{indent}lla\tt0, {symbol}")
                print(
                    f"{indent}sd\ta0, 0(t0)"
                    " # preserve sret outside outgoing-argument area"
                )
                sret_stores.add(current_function)
                sret_offsets[current_function] = store.group(2)
                recent.append(line)
                continue
            load = None
            if (
                current_function in sret_offsets
                and current_function not in sret_loads
            ):
                load = re.match(
                    rf"^(\s*)ld\s+([a-z][a-z0-9]*),\s*"
                    rf"{re.escape(sret_offsets[current_function])}"
                    rf"\(s0\)\s*(?:#.*)?$",
                    line,
                )
            if load:
                indent = load.group(1)
                reload_register = load.group(2)
                print(f"{indent}lla\t{reload_register}, {symbol}")
                print(f"{indent}ld\t{reload_register}, 0({reload_register})")
                sret_loads.add(current_function)
                recent.append(line)
                continue
        slide = SLIDE_RE.match(line)
        if slide and len(recent) == 3:
            scalar_source = FMV_SCALAR_RE.match(recent[0])
            addi = ADDI_ONE_RE.match(recent[1])
            vset = VSET_TU_RE.match(recent[2])
            if (
                scalar_source
                and addi
                and vset
                and scalar_source.group(1) == slide.group(3)
                and addi.group(1) == vset.group(1)
                and addi.group(2) == slide.group(4)
            ):
                slide_count += 1
                print(
                    scalar_insert(
                        slide.group(1),
                        slide.group(2),
                        slide.group(3),
                        slide.group(4),
                        vset.group(1),
                        slide_count,
                    )
                )
                recent.append(line)
                continue
        extract = EXTRACT_RE.match(line)
        if extract and extract.group(3) in masks:
            replacement_count += 1
            print(
                scalar_mask(
                    extract.group(1),
                    extract.group(2),
                    masks[extract.group(3)],
                    replacement_count,
                )
            )
            recent.append(line)
            continue

        mask = MASK_RE.match(line)
        if mask and mask.group(2) in vid_registers:
            masks[mask.group(1)] = mask.group(3)
        else:
            destination = VECTOR_DEST_RE.match(line)
            if destination:
                masks.pop(destination.group(1), None)
                if not VID_RE.match(line):
                    vid_registers.discard(destination.group(1))

        vid = VID_RE.match(line)
        if vid:
            vid_registers.add(vid.group(1))
        print(line)
        recent.append(line)

    for function in sorted(sret_functions_seen):
        if function not in sret_stores or function not in sret_loads:
            print(
                f"error: incomplete sret rewrite for {function}", file=sys.stderr
            )
            return 1
        symbol = SRET_FIXES[function]
        print(f"\t.local\t{symbol}")
        print(f"\t.comm\t{symbol},8,8")

    print(
        f"# Rewrote {replacement_count} RVV mask extractions and "
        f"{slide_count} scalar vslideup operations plus "
        f"{len(sret_stores)} sret spills for BOSC FPGA",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
