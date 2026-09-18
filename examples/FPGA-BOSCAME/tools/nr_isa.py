"""Verified instruction contract for the current NR NH/RA FPGA.

RVV allowlist and disassembly helpers adapted from ModelZoo's
examples/buddy-qwen35-fpga/python/qwen35/compiler/qwen35_fpga_isa.py,
commit 8815b74fb6d3cd6288c4d99ac6fd7c5d041a7cc3:
https://gitlink.org.cn/michaelcjl/ModelZoo.git

The AME validation below is NR-specific: accumulator memory contains raw i32,
only the signed-i8 datapath is verified, and transposed B loads are unsupported.
It validates instruction forms, not runtime values supplied to msettype.
"""

import re


RVV_ALLOWED = frozenset({
    "vsetvli", "vsetivli",
    "vfredmax.vs", "vfcvt.x.f.v", "vfcvt.rtz.x.f.v",
    "vand.vx", "vor.vx", "vadd.vv", "vsub.vx", "vmax.vx", "vmin.vx",
    "vmv.v.x", "vfmv.v.f", "vfmv.s.f", "vfmv.f.s",
    "vle32.v", "vse32.v", "vnsra.wi", "vse8.v",
    # FPGA5 NR probe on 2026-09-16: all 21 finite-input checks passed at
    # e32,m1,VL=16, including signed inputs and reduction/conversion cases.
    "vmv.v.v", "vfadd.vv", "vfsub.vv", "vfmul.vv", "vfdiv.vv",
    "vfmacc.vv", "vfadd.vf", "vfmul.vf", "vfdiv.vf", "vfmacc.vf",
    "vfsqrt.v", "vfredusum.vs", "vfredosum.vs", "vfmax.vv", "vfmin.vv",
    "vfcvt.f.x.v",
    # Follow-up FPGA5 probe plus 384 copy cases also passed on 2026-09-16.
    "vmv.v.i", "vfmv.f.s",
    # Expanded FPGA5 probe run-6e31d5d081284634 (2026-09-16): arithmetic
    # passed; e8,mf4,VL16 and e64,m1,VL2/4 memory
    # checks passed. Whole-register memory and vector CSR reads FAILED and
    # remain excluded. See qwen3-0.6b/validation/rvv-capabilities-expanded.
    # vmv1r.v passed only e32/VL16, while compiler uses e8/VL1. The actual
    # FP32 matrix program fails on that path, so whole moves remain rejected.
    "vfmadd.vv", "vle8.v", "vle64.v", "vse64.v",
    # FPGA5 run-be16db38f65745f4 (2026-09-18), quant_ops.S: exact
    # abs/div/round/clamp/i32->i8 sequence, signed boundary inputs, VL1/7/16,
    # e32/m1 -> e16/mf2 -> e8/mf4, guards intact, zero illegal instructions.
    "vfabs.v", "vmfge.vf", "vmerge.vxm", "vfmax.vf", "vfmin.vf", "vnsrl.wi",
})
VECTOR_MEMORY = frozenset({"vle8.v", "vse8.v", "vle32.v", "vse32.v",
                           "vle64.v", "vse64.v"})
FENCE_RW_RW = 0x0330000F


def is_ame(word):
    return word & 0x7F == 0x77


def is_transposed_b_load(word):
    return (is_ame(word) and (word >> 26) == 2 and
            not (word & (1 << 25)) and bool(word & (1 << 11)))


def validate_ame_word(word):
    """Reject instructions outside NR's verified i8-to-i32 AME subset."""
    if not 0 <= word <= 0xFFFFFFFF or not is_ame(word):
        raise ValueError(f"not a 32-bit AME instruction: {word:#x}")
    if is_transposed_b_load(word):
        raise ValueError("NR does not support transposed B loads; pack B as [N,K]")
    funct6 = word >> 26
    funct3 = (word >> 12) & 7
    immediate_or_store = (word >> 25) & 1
    # Register configuration: msettype and msettilem/n/k, not msettypei.
    if not immediate_or_store and ((word >> 20) & 31) == 0:
        if (funct6 == 0 and funct3 == 4 or
                funct6 == 1 and funct3 in (4, 5, 6)):
            return
    matrix_index = (word >> 7) & 15
    transpose = (word >> 11) & 1
    if not transpose and matrix_index < 8:
        if (not immediate_or_store and funct3 == 0 and funct6 in (1, 2)):
            return  # mlae8 / mlbe8
        if funct6 == 0 and funct3 == 2:
            return  # mlce32 / msce32
    # mqma.b.mm: quad widening, signed, integer, non-saturating, accumulate.
    if (funct6 == 10 and not immediate_or_store and
            not (word & (1 << 24)) and funct3 == 0 and transpose == 1 and
            (word & (1 << 19)) and matrix_index < 8 and
            ((word >> 15) & 15) < 8 and ((word >> 20) & 15) < 8):
        return
    raise ValueError(f"unverified NR AME instruction encoding: {word:#010x}")


def is_vector(word):
    return (word & 0x7F == 0x57 or
            (word & 0x7F in (0x07, 0x27) and (word >> 12) & 7 in (0, 5, 6, 7)))


def is_vector_memory(word):
    return word & 0x7F in (0x07, 0x27) and (word >> 12) & 7 in (0, 5, 6, 7)


def parse_disassembly(dump):
    records = []
    symbol = section = ""
    for line in dump.splitlines():
        if line.startswith("Disassembly of section "):
            section = line.removeprefix("Disassembly of section ").removesuffix(":")
        header = re.match(r"^([0-9a-f]+) <(.+)>:$", line)
        if header:
            symbol = header[2]
        # LLVM prints decoded instruction words in hex, but mapped .word data
        # (including our raw AME encodings) as little-endian byte pairs.
        match = re.match(r"^\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} ){1,7}[0-9a-f]{2}|[0-9a-f]{4}|[0-9a-f]{8})\s+(.+)$", line)
        if match:
            instruction = match[3]
            encoded = match[2].strip()
            word = (int.from_bytes(bytes.fromhex(encoded), 'little')
                    if ' ' in encoded else int(encoded, 16))
            records.append(dict(address=int(match[1], 16), word=word,
                                size=len(encoded.replace(' ', '')) // 2, instruction=instruction,
                                mnemonic=instruction.split()[0],
                                symbol=symbol, section=section))
    return records
