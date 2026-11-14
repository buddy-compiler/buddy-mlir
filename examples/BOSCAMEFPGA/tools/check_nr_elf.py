#!/usr/bin/env python3
# ===- check_nr_elf.py -----------------------------------------------------
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

"""Audit every executable byte of a linked NR ELF for the NR AME/RVV/fence contract (including C runtime).

Inputs: ELF path; optional objdump, allowlists, --output report path.
Outputs: Human and optional JSON audit; non-zero if forbidden forms or missing fences.
When used: Post-link gate for NR FPGA images before upload.

Walk every SHF_EXECINSTR byte range from the ELF section table. LLVM objdump
supplies RVV mnemonic names, but raw ELF bytes determine CSR/AME detection and
fence adjacency, so data-mapped .word encodings cannot bypass the audit.
This checks the NR AME/RVV contract, not arbitrary scalar ISA compatibility.
"""

import argparse
import hashlib
import json
import shlex
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nr_isa import (
    FENCE_RW_RW,
    RVV_ALLOWED,
    is_ame,
    is_ame_store,
    is_vector,
    is_vector_memory,
    is_vector_store,
    parse_disassembly,
    validate_ame_word,
)


def executable_sections(data):
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("expected little-endian ELF64")
    if len(data) < 64 or struct.unpack_from("<H", data, 18)[0] != 243:
        raise ValueError("expected RISC-V ELF")
    offset = struct.unpack_from("<Q", data, 40)[0]
    size, count, strings_index = struct.unpack_from("<HHH", data, 58)
    if (
        size != 64
        or not count
        or strings_index >= count
        or offset + count * size > len(data)
    ):
        raise ValueError("invalid or unsupported ELF section table")
    headers = [
        struct.unpack_from("<IIQQQQIIQQ", data, offset + i * size)
        for i in range(count)
    ]
    strings_header = headers[strings_index]
    strings = data[strings_header[4] : strings_header[4] + strings_header[5]]
    result = []
    for name, kind, flags, address, start, length, *_ in headers:
        # NOLOAD state (notably nr_mailbox) may inherit AX flags from the
        # preceding linker-script section but contains no instruction bytes.
        if not flags & 4 or not length or kind == 8:
            continue
        if start + length > len(data) or name >= len(strings):
            raise ValueError("invalid executable section")
        label = (
            strings[name:].split(b"\0", 1)[0].decode("utf-8", errors="strict")
        )
        result.append((label, address, data[start : start + length]))
    if not result:
        raise ValueError("ELF has no executable sections")
    return result


def audit(
    data,
    dump,
    require_fences=True,
    allow_vector_ordered_reductions=False,
    allow_vector_max_reductions=False,
):
    decoded = {(r["section"], r["address"]): r for r in parse_disassembly(dump)}
    errors, sections = [], []
    vectors = Counter()
    instruction_count = ame_count = vector_memory_count = 0
    ame_store_count = vector_store_count = 0
    for name, address, code in executable_sections(data):
        sections.append(
            {"name": name, "address": hex(address), "bytes": len(code)}
        )
        offset = 0
        previous_instruction = None
        while offset < len(code):
            if offset + 2 > len(code):
                errors.append(f"{name}+{offset:#x}: incomplete instruction")
                break
            half = int.from_bytes(code[offset : offset + 2], "little")
            width = 4 if half & 3 == 3 else 2
            where = f"{name}@{address + offset:#x}"
            if width == 4 and (half & 31 == 31 or offset + 4 > len(code)):
                errors.append(f"{where}: unsupported instruction width")
                break
            word = int.from_bytes(code[offset : offset + width], "little")
            instruction_count += 1
            entry = decoded.get((name, address + offset))
            detail = entry["instruction"] if entry else f"raw {word:#010x}"
            if entry and entry["word"] != word:
                # ObjDump may coalesce scalar data/padding. AME/RVV must still
                # have exact 32-bit records to receive mnemonic validation.
                entry = None
            if width == 4:
                if (
                    word & 0x7F == 0x73
                    and (word >> 12) & 7
                    and word >> 20 in (0xC20, 0xC21, 0xC22)
                ):
                    errors.append(f"{where}: unsupported vector CSR: {detail}")
                fenced = False
                if is_ame(word):
                    ame_count += 1
                    fenced = True
                    if is_ame_store(word):
                        ame_store_count += 1
                    try:
                        validate_ame_word(word)
                    except ValueError as error:
                        errors.append(f"{where}: {error}")
                if is_vector(word):
                    mnemonic = entry["mnemonic"] if entry else "<undecoded>"
                    vectors[mnemonic] += 1
                    # Raw encoding also catches undecoded or stale objects.
                    if (
                        word & 0xFC00707F == 0x0C001057
                        and not allow_vector_ordered_reductions
                    ):
                        errors.append(
                            f"{where}: vector ordered reduction is disabled; "
                            "rebuild through the scalar workaround or use the "
                            "explicit diagnostic opt-out"
                        )
                    if (
                        word & 0xFC00707F == 0x1C001057
                        and not allow_vector_max_reductions
                    ):
                        errors.append(
                            f"{where}: vector max reduction is disabled; "
                            "rebuild through scalar-f32-max-reductions or use "
                            "the explicit diagnostic opt-out"
                        )
                    if mnemonic not in RVV_ALLOWED:
                        errors.append(f"{where}: unsupported RVV: {detail}")
                    if is_vector_memory(word):
                        vector_memory_count += 1
                        if is_vector_store(word):
                            vector_store_count += 1
                        fenced = True
                if fenced and require_fences:
                    after = int.from_bytes(
                        code[offset + 4 : offset + 8], "little"
                    )
                    # The four preceding bytes may straddle a 32-bit scalar
                    # instruction and a compressed instruction. Require an
                    # actual decoded predecessor, not a coincidental byte
                    # sequence equal to the fence encoding.
                    if (
                        previous_instruction != (4, FENCE_RW_RW)
                        or after != FENCE_RW_RW
                    ):
                        errors.append(
                            f"{where}: missing adjacent fence rw,rw: {detail}"
                        )
            previous_instruction = (width, word)
            offset += width
    return {
        "status": "FAIL" if errors else "PASS",
        "elf_sha256": hashlib.sha256(data).hexdigest(),
        "allow_vector_ordered_reductions": allow_vector_ordered_reductions,
        "allow_vector_max_reductions": allow_vector_max_reductions,
        "executable_sections": sections,
        "instructions": instruction_count,
        "ame_instructions": ame_count,
        "vector_memory_instructions": vector_memory_count,
        "ame_store_instructions": ame_store_count,
        "vector_store_instructions": vector_store_count,
        "rvv_instructions": {
            name: count for name, count in sorted(vectors.items())
        },
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-vector-ordered-reductions",
        action="store_true",
        help="diagnostic opt-out: permit vfredosum.vs in the linked image",
    )
    parser.add_argument(
        "--allow-vector-max-reductions",
        action="store_true",
        help="diagnostic opt-out: permit vfredmax.vs in the linked image",
    )
    parser.add_argument("elf", type=Path)
    parser.add_argument(
        "--objdump",
        default="llvm-objdump",
        help="shell-quoted llvm-objdump command, optionally with flags",
    )
    parser.add_argument(
        "--output", type=Path, help="write machine-readable audit report"
    )
    parser.add_argument(
        "--allow-unfenced",
        action="store_true",
        help="do not require fence rw,rw around AME/RVV memory instructions "
        "when the assembly filter intentionally ran with --no-fences",
    )
    parser.add_argument(
        "--require-direct-store",
        action="store_true",
        help="fail unless the ELF contains an AME or RVV direct store",
    )
    args = parser.parse_args()
    try:
        dump = subprocess.check_output(
            [
                *shlex.split(args.objdump),
                "-d",
                "--mattr=+v,+zicbom",
                str(args.elf),
            ],
            text=True,
        )
        report = audit(
            args.elf.read_bytes(),
            dump,
            require_fences=not args.allow_unfenced,
            allow_vector_ordered_reductions=args.allow_vector_ordered_reductions,
            allow_vector_max_reductions=args.allow_vector_max_reductions,
        )
        if args.require_direct_store and not (
            report["ame_store_instructions"]
            or report["vector_store_instructions"]
        ):
            report["errors"].append(
                "direct-store audit requested but no msce32.m or vse*.v was found"
            )
            report["status"] = "FAIL"
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"NR ELF audit {report['status']}: {report['instructions']} instructions, "
            f"{report['ame_instructions']} AME, "
            f"{sum(report['rvv_instructions'].values())} RVV; {args.elf}"
        )
        for error in report["errors"][:30]:
            print(error, file=sys.stderr)
        if len(report["errors"]) > 30:
            print(
                f"... {len(report['errors']) - 30} additional errors in report",
                file=sys.stderr,
            )
        return int(bool(report["errors"]))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"NR ELF audit failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
