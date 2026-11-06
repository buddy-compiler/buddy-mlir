# ===- nr_isa.py -----------------------------------------------------------
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

"""Library of the verified NR NH/RA FPGA instruction contract (RVV allowlist, AME form checks, helpers).

Inputs: Imported by encoding/audit tools; no CLI.
Outputs: Predicates and constants (RVV_ALLOWED, is_ame, validate_ame_word, ...).
When used: Shared by ame_to_word, restrict_fpga_assembly, check_nr_elf, and related lit helpers.

RVV allowlist and disassembly helpers adapted from ModelZoo's
examples/buddy-qwen35-fpga/python/qwen35/compiler/qwen35_fpga_isa.py,
commit 8815b74fb6d3cd6288c4d99ac6fd7c5d041a7cc3:
https://gitlink.org.cn/michaelcjl/ModelZoo.git

The AME validation below is NR-specific: accumulator memory contains raw i32,
only the signed-i8 datapath is verified, and transposed B loads are unsupported.
It validates instruction forms, not runtime values supplied to msettype.
"""

import re

RVV_ALLOWED = frozenset(
    {
        "vsetvli",
        "vsetivli",
        "vfredmax.vs",
        "vfcvt.x.f.v",
        "vfcvt.rtz.x.f.v",
        "vand.vx",
        "vor.vx",
        "vadd.vv",
        "vsub.vx",
        "vmax.vx",
        "vmin.vx",
        "vmv.v.x",
        "vfmv.v.f",
        "vfmv.s.f",
        "vfmv.f.s",
        "vle32.v",
        "vse32.v",
        "vnsra.wi",
        "vse8.v",
        # FPGA5 NR probe on 2026-09-16: all 21 finite-input checks passed at
        # e32,m1,VL=16, including signed inputs and reduction/conversion cases.
        "vmv.v.v",
        "vfadd.vv",
        "vfsub.vv",
        "vfmul.vv",
        "vfdiv.vv",
        "vfmacc.vv",
        "vfadd.vf",
        "vfmul.vf",
        "vfdiv.vf",
        "vfmacc.vf",
        "vfsqrt.v",
        "vfredusum.vs",
        "vfredosum.vs",
        "vfmax.vv",
        "vfmin.vv",
        "vfcvt.f.x.v",
        # Follow-up FPGA5 probe plus 384 copy cases also passed on 2026-09-16.
        "vmv.v.i",
        # Expanded FPGA5 probe run-6e31d5d081284634 (2026-09-16): arithmetic
        # passed; e8,mf4,VL16 and e64,m1,VL2/4 memory
        # checks passed. Whole-register memory and vector CSR reads FAILED and
        # remain excluded. See qwen3-0.6b/validation/rvv-capabilities-expanded.
        # vmv1r.v passed only e32/VL16, while compiler uses e8/VL1. The actual
        # FP32 matrix program fails on that path, so whole moves remain rejected.
        "vfmadd.vv",
        "vle8.v",
        "vle64.v",
        "vse64.v",
        # FPGA5 run-be16db38f65745f4 (2026-09-18), quant_ops.S: exact
        # abs/div/round/clamp/i32->i8 sequence, signed boundary inputs, VL1/7/16,
        # e32/m1 -> e16/mf2 -> e8/mf4, guards intact, zero illegal instructions.
        "vfabs.v",
        "vmfge.vf",
        "vmerge.vxm",
        "vfmax.vf",
        "vfmin.vf",
        "vnsrl.wi",
    }
)
VECTOR_MEMORY = frozenset(
    {"vle8.v", "vse8.v", "vle32.v", "vse32.v", "vle64.v", "vse64.v"}
)
FENCE_RW_RW = 0x0330000F

# Words emitted by the pre-patch6 fixed-register encoder.  They are kept
# explicit so an old generated object cannot silently pass a syntactic audit
# after being copied to the new board image.  In particular, the old B load
# selected tr4 and the old tile/store forms used non-canonical GPR fields.
PATCH6_AME_WORDS = frozenset(
    {
        0x00054077,
        0x04055077,
        0x04054077,
        0x04056077,
        0x04B50077,
        0x08B500F7,
        0x00B52077,
        0x02B52077,
        0x28180877,
    }
)
LEGACY_AME_WORDS = frozenset(
    {
        0x04055877,
        0x040543F7,
        0x0406E6F7,
        0x08B50277,
        0x005E2077,
        0x025E2077,
        0x28480877,
    }
)


def is_ame(word):
    return word & 0x7F == 0x77


def is_transposed_b_load(word):
    return (
        is_ame(word)
        and (word >> 26) == 2
        and not (word & (1 << 25))
        and bool(word & (1 << 11))
    )


def validate_ame_word(word):
    """Reject instructions outside NR's verified i8-to-i32 AME subset."""
    if not 0 <= word <= 0xFFFFFFFF or not is_ame(word):
        raise ValueError(f"not a 32-bit AME instruction: {word:#x}")
    if word in LEGACY_AME_WORDS:
        raise ValueError(f"legacy pre-patch6 AME encoding: {word:#010x}")
    if is_transposed_b_load(word):
        raise ValueError(
            "NR does not support transposed B loads; pack B as [N,K]"
        )
    if is_ame_acc_clear(word):
        return
    funct6 = word >> 26
    funct3 = (word >> 12) & 7
    immediate_or_store = (word >> 25) & 1
    # Register configuration: msettype and msettilem/n/k, not msettypei.
    if not immediate_or_store and ((word >> 20) & 31) == 0:
        if funct6 == 0 and funct3 == 4 or funct6 == 1 and funct3 in (4, 5, 6):
            return
    matrix_index = (word >> 7) & 15
    transpose = (word >> 11) & 1
    if not transpose and matrix_index < 8:
        if not immediate_or_store and funct3 == 0 and funct6 in (1, 2):
            return  # mlae8 / mlbe8
        if funct6 == 0 and funct3 == 2:
            return  # mlce32 / msce32
    # mqma.b.mm: quad widening, signed, integer, non-saturating, accumulate.
    if (
        funct6 == 10
        and not immediate_or_store
        and not (word & (1 << 24))
        and funct3 == 0
        and transpose == 1
        and (word & (1 << 19))
        and matrix_index < 8
        and ((word >> 15) & 15) < 8
        and ((word >> 20) & 15) < 8
    ):
        return
    raise ValueError(f"unverified NR AME instruction encoding: {word:#010x}")


def is_ame_acc_clear(word):
    """Recognize only the 16 board-verified self-clear encodings.

    Generic mwsub.mm requires msew=16; mwsub.h.mm selects 16-bit inputs
    explicitly. Both produce i32 zeros and require dst == src1 == src2.
    General subtraction, register conversion and other widths stay rejected.
    """
    accumulator = (word >> 7) & 15
    return accumulator < 8 and word in (
        base | (accumulator << 20) | (accumulator << 15) | (accumulator << 7)
        for base in (0x2C084077, 0x2C081077)
    )


def is_vector(word):
    return word & 0x7F == 0x57 or (
        word & 0x7F in (0x07, 0x27) and (word >> 12) & 7 in (0, 5, 6, 7)
    )


def is_vector_memory(word):
    return word & 0x7F in (0x07, 0x27) and (word >> 12) & 7 in (0, 5, 6, 7)


def is_ame_store(word):
    """Return whether *word* is the verified accumulator store form.

    The AME opcode uses bit 25 to select the memory-store form.  For NR,
    ``msce32.m`` is the non-transposed accumulator operation (funct6=0,
    funct3=2) with a physical accumulator index below eight.  Decode these
    fields from the raw word so an objdump ``.word`` line cannot hide a store
    from the ELF audit.
    """
    return (
        is_ame(word)
        and (word >> 26) == 0
        and ((word >> 12) & 7) == 2
        and bool((word >> 25) & 1)
        and not ((word >> 11) & 1)
        and ((word >> 7) & 15) < 8
    )


def is_vector_store(word):
    """Return whether *word* is an RVV vector memory store."""
    return is_vector_memory(word) and word & 0x7F == 0x27


def parse_disassembly(dump):
    records = []
    symbol = section = ""
    for line in dump.splitlines():
        if line.startswith("Disassembly of section "):
            section = line.removeprefix("Disassembly of section ").removesuffix(
                ":"
            )
        header = re.match(r"^([0-9a-f]+) <(.+)>:$", line)
        if header:
            symbol = header[2]
        # LLVM prints decoded instruction words in hex, but mapped .word data
        # (including our raw AME encodings) as little-endian byte pairs.
        match = re.match(
            r"^\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} ){1,7}[0-9a-f]{2}|[0-9a-f]{4}|[0-9a-f]{8})\s+(.+)$",
            line,
        )
        if match:
            instruction = match[3]
            encoded = match[2].strip()
            word = (
                int.from_bytes(bytes.fromhex(encoded), "little")
                if " " in encoded
                else int(encoded, 16)
            )
            records.append(
                {
                    "address": int(match[1], 16),
                    "word": word,
                    "size": len(encoded.replace(" ", "")) // 2,
                    "instruction": instruction,
                    "mnemonic": instruction.split()[0],
                    "symbol": symbol,
                    "section": section,
                }
            )
    return records
