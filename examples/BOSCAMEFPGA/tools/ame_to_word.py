#!/usr/bin/env python3
# ===- ame_to_word.py ------------------------------------------------------
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

"""Encode the verified NR FPGA AME mnemonic subset into raw instruction words with explicit GPR handling.

Inputs: RISC-V assembly text on stdin (or path args if provided); optional --gpr-mode and --preserve-matrix-registers.
Outputs: Assembly with AME ops rewritten to .word encodings on stdout; rejects unrepresentable schedules.
When used: NR FPGA example / lit build after llc emits assembly and before restrict_fpga_assembly.py.

Adapted from ModelZoo's examples/tools/ame_to_word.py, commit 8815b74fb6d3cd6288c4d99ac6fd7c5d041a7cc3:
https://gitlink.org.cn/michaelcjl/ModelZoo.git

This example-side tool targets NR only. GEM5 uses the compiler backend directly.
Run restrict_fpga_assembly.py afterwards to validate raw words and insert the
required fence before AND after every AME instruction. No UART/MMIO tracing,
legacy ISA fallback, or experimental environment-dependent encoding is provided.
The default fixed mode retains the board-tested operand-preservation wrappers.
Opt-in --gpr-mode=direct uses v0.5's full five-bit GPR fields for tile and memory
instructions only. msettype retains its existing caller-saved preservation in
both modes; this change makes no new claim about its hardware clobber behavior.
--preserve-matrix-registers keeps the allocated matrix registers while retaining
the selected GPR handling. Without it, fixed mode rejects schedules that cannot
be represented by the historical tr0/tr1/acc0 mapping.
"""

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nr_isa import validate_ame_word

ABI_NAMES = [
    "zero",
    "ra",
    "sp",
    "gp",
    "tp",
    "t0",
    "t1",
    "t2",
    "s0",
    "s1",
    "a0",
    "a1",
    "a2",
    "a3",
    "a4",
    "a5",
    "a6",
    "a7",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
    "s7",
    "s8",
    "s9",
    "s10",
    "s11",
    "t3",
    "t4",
    "t5",
    "t6",
]
GPR = {name: index for index, name in enumerate(ABI_NAMES)}
GPR.update({f"x{index}": index for index in range(32)})
GPR["fp"] = 8
CALLER_SAVED_GPRS = [
    "a0",
    "a1",
    "a2",
    "a3",
    "a4",
    "a5",
    "a6",
    "a7",
    "t0",
    "t1",
    "t2",
    "t3",
    "t4",
    "t5",
    "t6",
]
# Canonical patch6 NR operand encodings.  The updated AME image accepts
# tile configuration with a0 as the source and no architectural return,
# memory operations with a0/a1 as address/stride, and the tr0/tr1 pair for
# signed-i8 matrix multiply.  The previous ModelZoo words used return and
# operand fields from an older AME image (for example tr4 for B); they can
# leave the current engine waiting indefinitely even though they pass a
# syntactic AME audit.
CONFIG = {
    "msettype": ("a0", None, 0x00054077),
    "msettilem": ("a0", None, 0x04055077),
    "msettilen": ("a0", None, 0x04054077),
    "msettilek": ("a0", None, 0x04056077),
}
MEMORY = {
    "mlae8.m": ("tr", "a0", "a1", 0x04B50077),
    "mlbe8.m": ("tr", "a0", "a1", 0x08B500F7),
    "mlce32.m": ("acc", "a0", "a1", 0x00B52077),
    "msce32.m": ("acc", "a0", "a1", 0x02B52077),
}
GPR_MODES = ("fixed", "direct")
# Historical switches are deliberately rejected, including a value of "0":
# callers should remove stale configuration instead of relying on it silently.
REMOVED_ENVIRONMENT_OPTIONS = frozenset(
    {
        "AME_CONFIG_DELAY_NOPS",
        "AME_LOAD_DELAY_NOPS",
        "AME_STORE_DELAY_NOPS",
        "AME_MMA_DELAY_NOPS",
        "AME_MMA_FENCE_AFTER",
        "AME_FENCE_BEFORE_RESTORE",
        "AME_USE_QWEN_FIXED_REGS",
        "AME_TRACE_STAGES",
        "AME_TRACE_ADDRS",
    }
)


def validate_environment(environment=None):
    environment = os.environ if environment is None else environment
    old = sorted(REMOVED_ENVIRONMENT_OPTIONS.intersection(environment))
    if old:
        raise ValueError(
            "unsupported legacy AME environment options: "
            + ", ".join(old)
            + "; remove them for the NR-only build"
        )


def canonical_gpr(name):
    if name not in GPR:
        raise ValueError(f"invalid RISC-V GPR: {name}")
    return ABI_NAMES[GPR[name]]


def matrix_register(name, bank):
    match = re.fullmatch(re.escape(bank) + r"([0-7])", name)
    if not match:
        raise ValueError(f"expected {bank}0..{bank}7, got {name}")
    return int(match[1])


def emit_fixed_reg_sequence(
    indent, live_regs, moves, word, comment, result_regs=None
):
    """Preserve original operands, including aliases, overlap, sp and rd.

    Source operands must be treated as parallel moves: a store with base=t1,
    stride=t3 needs t3=old(t1), t0=old(t3), not t0=the new base. Configuration
    return values are saved separately so restoring old GPRs cannot erase rd.
    The subsequent restriction pass fences the word while its operands are live.
    """
    validate_ame_word(word)
    regs = list(dict.fromkeys(live_regs))
    offsets = {reg: 8 * index for index, reg in enumerate(regs)}
    result_offset = 8 * len(regs)
    slots = len(regs) + bool(result_regs)
    stack_bytes = max(16, ((8 * slots + 15) // 16) * 16)
    lines = [f"{indent}addi\tsp, sp, -{stack_bytes}"]
    for reg, offset in offsets.items():
        lines.append(f"{indent}sd\t{reg}, {offset}(sp)")
    for destination, source in moves:
        if destination == source:
            continue
        if source == "sp":
            lines.append(f"{indent}addi\t{destination}, sp, {stack_bytes}")
        elif source in offsets:
            lines.append(f"{indent}ld\t{destination}, {offsets[source]}(sp)")
        else:
            lines.append(f"{indent}mv\t{destination}, {source}")
    lines.append(f"{indent}.word 0x{word:08x} # {comment} [nr fixed]")
    if result_regs:
        lines.append(f"{indent}sd\t{result_regs[0]}, {result_offset}(sp)")
    for reg, offset in reversed(list(offsets.items())):
        lines.append(f"{indent}ld\t{reg}, {offset}(sp)")
    if result_regs and result_regs[1] != "sp":
        lines.append(f"{indent}ld\t{result_regs[1]}, {result_offset}(sp)")
    lines.append(f"{indent}addi\tsp, sp, {stack_bytes}")
    if result_regs and result_regs[1] == "sp":
        lines.append(f"{indent}ld\tsp, {result_offset - stack_bytes}(sp)")
    return "\n".join(lines)


def encode_mma(
    mnemonic,
    accumulator,
    lhs,
    rhs,
    gpr_mode="fixed",
    preserve_matrix_registers=False,
):
    if mnemonic != "mqma.b.mm":
        raise ValueError(
            f"NR only supports signed i8 mqma.b.mm, got {mnemonic}"
        )
    if gpr_mode == "fixed" and not preserve_matrix_registers:
        # Validate the source banks even though the virtual RHS register is
        # remapped to tr1 below.  This catches malformed lowering while
        # accepting the historical tr4 spelling emitted for B.
        matrix_register(accumulator, "acc")
        matrix_register(lhs, "tr")
        matrix_register(rhs, "tr")
        # patch6's validated signed-i8 operation is acc0 += tr0 * tr1.
        word = 0x28180877
    else:
        word = (
            0x28080877
            | (matrix_register(accumulator, "acc") << 7)
            | (matrix_register(lhs, "tr") << 15)
            | (matrix_register(rhs, "tr") << 20)
        )
    validate_ame_word(word)
    return word


def encode_acc_clear(
    mnemonic, destination, lhs, rhs, gpr_mode, preserve_matrix_registers
):
    indices = [matrix_register(name, "acc") for name in (destination, lhs, rhs)]
    if len(set(indices)) != 1:
        raise ValueError("NR accumulator clear requires dst == src1 == src2")
    accumulator = (
        0
        if gpr_mode == "fixed" and not preserve_matrix_registers
        else indices[0]
    )
    word = (
        (0x2C084077 if mnemonic == "mwsub.mm" else 0x2C081077)
        | (accumulator << 20)
        | (accumulator << 15)
        | (accumulator << 7)
    )
    validate_ame_word(word)
    return word


def convert_line(line, gpr_mode="fixed", preserve_matrix_registers=False):
    """Return (converted assembly, changed), rejecting unverified AME forms."""
    if gpr_mode not in GPR_MODES:
        raise ValueError(f"unsupported AME GPR mode: {gpr_mode}")
    code = line.split("#", 1)[0].strip()
    if not code or code.startswith("//"):
        return line, False
    # LLVM emits labels on separate lines; support same-line labels explicitly
    # so an unsupported mnemonic cannot hide after one.
    label = re.match(r"^([\w.$]+:)\s*(.*)$", code)
    if label:
        if not label[2]:
            return line, False
        converted, changed = convert_line(
            "\t" + label[2], gpr_mode, preserve_matrix_registers
        )
        return label[1] + "\n" + converted, changed
    if code.startswith("."):
        return line, False
    if ";" in code:
        raise ValueError(
            "multiple assembly statements must be on separate lines"
        )
    # tab-separated assembler output is common.
    fields = code.split(None, 1)
    mnemonic = fields[0]
    operand_text = fields[1] if len(fields) == 2 else ""
    operands = [part.strip() for part in operand_text.split(",")]
    indent = line[: len(line) - len(line.lstrip())]
    if mnemonic in CONFIG:
        if len(operands) != 2:
            raise ValueError(f"{mnemonic} expects rd, rs1")
        rd, rs1 = map(canonical_gpr, operands)
        fixed_rs1, fixed_rd, word = CONFIG[mnemonic]
        if gpr_mode == "direct" and mnemonic != "msettype":
            # AME v0.5 software contract section 2.1: rd and rs1 are
            # independent five-bit GPR fields. In particular rd=rs1 reads
            # the original source before writing the effective tile size.
            word = (
                (word & ~((31 << 15) | (31 << 7)))
                | (GPR[rs1] << 15)
                | (GPR[rd] << 7)
            )
            validate_ame_word(word)
            return f"{indent}.word 0x{word:08x} # {code} [nr direct]", True
        live = (
            list(CALLER_SAVED_GPRS) if mnemonic == "msettype" else [fixed_rs1]
        )
        if mnemonic == "msettype" and rd != "zero":
            fixed_rd = "a6"
            word |= GPR[fixed_rd] << 7
        if fixed_rd:
            live.append(fixed_rd)
        # patch6 tile configuration has no usable destination register.  Keep
        # the source value live in a0, but discard the compiler's historical
        # pseudo-result instead of encoding the stale a6/t2/a3 return fields.
        return emit_fixed_reg_sequence(
            indent, live, [(fixed_rs1, rs1)], word, code, None
        ), True
    if mnemonic in MEMORY:
        if len(operands) != 3:
            raise ValueError(
                f"{mnemonic} expects matrix register, (base), stride"
            )
        bank, fixed_base, fixed_stride, word = MEMORY[mnemonic]
        matrix = matrix_register(operands[0], bank)
        address = operands[1]
        if address.startswith("(") and address.endswith(")"):
            address = address[1:-1].strip()
        base, stride = canonical_gpr(address), canonical_gpr(operands[2])
        if gpr_mode == "direct":
            # AME v0.5 software contract section 2.2: rs2 is FIVE bits.
            # Bit 24 belongs to the GPR field, not to an eight-bit opcode.
            word = (
                (word & ~((31 << 20) | (31 << 15) | (15 << 7)))
                | (GPR[stride] << 20)
                | (GPR[base] << 15)
                | (matrix << 7)
            )
            validate_ame_word(word)
            return f"{indent}.word 0x{word:08x} # {code} [nr direct]", True
        # The patch6 contract uses tr0 for A, tr1 for B and acc0 for the
        # accumulator.  Matrix names in the generated LLVM are virtual and
        # may still say tr4/acc7; map them to the canonical physical banks.
        canonical_matrix = (
            matrix
            if preserve_matrix_registers
            else 1
            if mnemonic == "mlbe8.m"
            else 0
        )
        word &= ~(15 << 7)
        return emit_fixed_reg_sequence(
            indent,
            [fixed_base, fixed_stride],
            [(fixed_base, base), (fixed_stride, stride)],
            word | (canonical_matrix << 7),
            code,
        ), True
    if mnemonic in ("mwsub.mm", "mwsub.h.mm"):
        if len(operands) != 3:
            raise ValueError(
                f"{mnemonic} expects the same Acc register three times"
            )
        word = encode_acc_clear(
            mnemonic, *operands, gpr_mode, preserve_matrix_registers
        )
        return f"{indent}.word 0x{word:08x} # {code}", True
    if mnemonic == "mqma.b.mm":
        if len(operands) != 3:
            raise ValueError("mqma.b.mm expects acc, tr, tr")
        word = encode_mma(
            mnemonic,
            *operands,
            gpr_mode=gpr_mode,
            preserve_matrix_registers=preserve_matrix_registers,
        )
        return f"{indent}.word 0x{word:08x} # {code}", True
    if mnemonic.startswith(("mlbte", "mlbt")):
        raise ValueError(
            "NR does not support transposed B loads; pack B as [N,K]"
        )
    if (
        mnemonic.startswith(("mset", "ml", "ms", "mq", "mf", "mw"))
        # RVV mask logic uses the same suffix (for example vmor.mm). Leave
        # those instructions to the strict RVV filter rather than treating
        # them as unencoded AME matrix operations.
        or (".mm" in mnemonic and not mnemonic.startswith("v"))
        or re.search(r"\b(?:tr|acc)\d+\b", code)
    ):
        raise ValueError(f"unsupported NR AME instruction: {code}")
    return line, False


def verify_fixed_matrix_schedule(source):
    """Check that fixed physical registers still contain each allocated value.

    The compiler's assembly register names already have physical lifetimes.
    Collapsing them into one A, B and accumulator register is only valid when
    no later load overwrites an operand before its use. Diagnose the scheduled
    instruction at that use instead of silently computing a different product.
    """
    contents = {}
    for number, line in enumerate(source.splitlines(), 1):
        code = line.split("#", 1)[0].strip()
        code = re.sub(r"^[\w.$]+:\s*", "", code)
        fields = code.split(None, 1)
        if len(fields) != 2:
            continue
        mnemonic, operand_text = fields
        operands = [part.strip() for part in operand_text.split(",")]
        uses = []
        if mnemonic in ("mlae8.m", "mlbe8.m", "mlce32.m"):
            bank = "acc" if mnemonic == "mlce32.m" else "tr"
            matrix_register(operands[0], bank)
            physical = "tr1" if mnemonic == "mlbe8.m" else bank + "0"
            contents[physical] = operands[0]
        elif mnemonic in ("mwsub.mm", "mwsub.h.mm"):
            if len(operands) != 3:
                raise ValueError(f"{mnemonic} expects three Acc operands")
            indices = [matrix_register(value, "acc") for value in operands]
            if len(set(indices)) != 1:
                raise ValueError(
                    "NR accumulator clear requires dst == src1 == src2"
                )
            contents["acc0"] = operands[0]
        elif mnemonic == "mqma.b.mm" and len(operands) == 3:
            uses = list(zip(("acc0", "tr0", "tr1"), operands))
        elif mnemonic == "msce32.m":
            uses = [("acc0", operands[0])]
        for physical, allocated in uses:
            if contents.get(physical) != allocated:
                raise ValueError(
                    f"line {number}: fixed matrix mapping clobbers {allocated} "
                    f"before {mnemonic}; {physical} contains "
                    f"{contents.get(physical, 'no loaded value')}. "
                    "Use a serialized schedule or --preserve-matrix-registers "
                    "on hardware supporting the allocated matrix registers"
                )


def transform(source, gpr_mode="fixed", preserve_matrix_registers=False):
    validate_environment()
    if gpr_mode not in GPR_MODES:
        raise ValueError(f"unsupported AME GPR mode: {gpr_mode}")
    if gpr_mode == "fixed" and not preserve_matrix_registers:
        verify_fixed_matrix_schedule(source)
    converted_lines = []
    for number, line in enumerate(source.splitlines(), 1):
        try:
            converted, _ = convert_line(
                line, gpr_mode, preserve_matrix_registers
            )
        except ValueError as error:
            raise ValueError(f"line {number}: {error}") from error
        converted_lines.append(converted)
    return "\n".join(converted_lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpr-mode",
        choices=GPR_MODES,
        default="fixed",
        help="tile/MLS GPR handling (default: fixed); "
        "msettype always retains its fixed wrapper",
    )
    parser.add_argument(
        "--preserve-matrix-registers",
        action="store_true",
        help="preserve allocated tr/acc fields with fixed GPR wrappers; "
        "requires hardware support for those matrix registers",
    )
    args = parser.parse_args()
    try:
        output = transform(
            sys.stdin.read(), args.gpr_mode, args.preserve_matrix_registers
        )
    except ValueError as error:
        print(f"ame_to_word: {error}", file=sys.stderr)
        return 1
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
