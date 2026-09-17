#!/usr/bin/env python3
"""
Convert AME mnemonics in RISC-V assembly to .word directives.

llc outputs pseudo-instruction mnemonics (mlae8.m, mqma.b.mm, etc.) that
standard RISC-V assemblers don't recognize. This script converts them to
.word directives with correct 32-bit machine encodings.

The encoding is based on RVInstBOSCAME32 and related classes in
RISCVInstrInfoBuddyBOSCExt.td.

Usage:
  python3 ame_to_word.py < input.s > output.s
"""

import re
import sys
import os

# RISC-V GPR name to number
GPR = {
    "zero": 0, "ra": 1, "sp": 2, "gp": 3, "tp": 4,
    "t0": 5, "t1": 6, "t2": 7,
    "s0": 8, "fp": 8, "s1": 9,
    "a0": 10, "a1": 11, "a2": 12, "a3": 13, "a4": 14, "a5": 15,
    "a6": 16, "a7": 17,
    "s2": 18, "s3": 19, "s4": 20, "s5": 21, "s6": 22, "s7": 23,
    "s8": 24, "s9": 25, "s10": 26, "s11": 27,
    "t3": 28, "t4": 29, "t5": 30, "t6": 31,
}

# AME register index by name (tile/acc)
AME_REG = {
    "tr0": 0, "tr1": 1, "tr2": 2, "tr3": 3,
    "tr4": 4, "tr5": 5, "tr6": 6, "tr7": 7,
    "acc0": 0, "acc1": 1, "acc2": 2, "acc3": 3,
    "acc4": 4, "acc5": 5, "acc6": 6, "acc7": 7,
}

OPCODE_AME = 0b1110111  # bits[6:0]
AME_CONFIG_DELAY_NOPS = int(os.environ.get("AME_CONFIG_DELAY_NOPS", "0"))
AME_LOAD_DELAY_NOPS = int(os.environ.get("AME_LOAD_DELAY_NOPS", "0"))
AME_STORE_DELAY_NOPS = int(os.environ.get("AME_STORE_DELAY_NOPS", "0"))
AME_STORE_FENCE_AFTER = (
    int(os.environ.get("AME_STORE_FENCE_AFTER", "1")) != 0
)
AME_MMA_DELAY_NOPS = int(os.environ.get("AME_MMA_DELAY_NOPS", "0"))
AME_MMA_FENCE_AFTER = int(os.environ.get("AME_MMA_FENCE_AFTER", "0")) != 0
AME_FENCE_BEFORE_RESTORE = (
    int(os.environ.get("AME_FENCE_BEFORE_RESTORE", "0")) != 0
)
AME_USE_QWEN_FIXED_REGS = (
    int(os.environ.get("AME_USE_QWEN_FIXED_REGS", "1")) != 0
)
AME_TRACE_STAGES = int(os.environ.get("AME_TRACE_STAGES", "0")) != 0
AME_TRACE_ADDRS = int(os.environ.get("AME_TRACE_ADDRS", "0")) != 0
_trace_label = 0

QWEN_FIXED_CONFIG = {
    "msettype":  ("a0", None, 0x00054077),
    "msettilem": ("a0", "a6", 0x04055877),
    "msettilen": ("a0", "t2", 0x040543f7),
    "msettilek": ("a3", "a3", 0x0406e6f7),
}

QWEN_FIXED_LOAD = {
    ("mlce32.m", "acc0"): ("t3", "t0", 0x005e2077),
    ("mlce32.m", "acc1"): ("t3", "t0", 0x005e20f7),
    ("mlce32.m", "acc2"): ("t3", "t0", 0x005e2177),
    ("mlce32.m", "acc3"): ("t3", "t0", 0x005e21f7),
    ("mlce32.m", "acc4"): ("t3", "t0", 0x005e2277),
    ("mlce32.m", "acc5"): ("t3", "t0", 0x005e22f7),
    ("mlce32.m", "acc6"): ("t3", "t0", 0x005e2377),
    ("mlce32.m", "acc7"): ("t3", "t0", 0x005e23f7),
    ("mlae8.m", "tr0"): ("a0", "a1", 0x04b50077),
    ("mlae8.m", "tr2"): ("a0", "a1", 0x04b50177),
    ("mlae8.m", "tr4"): ("a0", "a1", 0x04b50277),
    ("mlbte8.m", "tr1"): ("a0", "a1", 0x08b508f7),
    ("mlbe8.m", "tr1"): ("a0", "a1", 0x08b500f7),
    ("mlbe8.m", "tr3"): ("a0", "a1", 0x08b501f7),
    ("mlbe8.m", "tr4"): ("a0", "a1", 0x08b50277),
    ("mlbe8.m", "tr5"): ("a0", "a1", 0x08b502f7),
    ("mlbe8.m", "tr6"): ("a0", "a1", 0x08b50377),
    ("mlbe8.m", "tr7"): ("a0", "a1", 0x08b503f7),
}

QWEN_FIXED_STORE = {
    ("msce32.m", "acc0"): ("t3", "t0", 0x025e2077),
    ("msce32.m", "acc1"): ("t3", "t0", 0x025e20f7),
    ("msce32.m", "acc2"): ("t3", "t0", 0x025e2177),
    ("msce32.m", "acc3"): ("t3", "t0", 0x025e21f7),
    ("msce32.m", "acc4"): ("t3", "t0", 0x025e2277),
    ("msce32.m", "acc5"): ("t3", "t0", 0x025e22f7),
    ("msce32.m", "acc6"): ("t3", "t0", 0x025e2377),
    ("msce32.m", "acc7"): ("t3", "t0", 0x025e23f7),
}

QWEN_FIXED_MMA = {
    ("mqma.b.mm", "acc0", "tr0", "tr1"): 0x28180877,
}

CALLER_SAVED_GPRS = [
    "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
    "t0", "t1", "t2", "t3", "t4", "t5", "t6",
]


def emit_uart_marker(indent, marker):
    """Emit a register-transparent UART marker for FPGA stage diagnosis."""
    global _trace_label
    label = _trace_label
    _trace_label += 1
    return "\n".join([
        f"{indent}addi\tsp, sp, -16",
        f"{indent}sd\tt0, 0(sp)",
        f"{indent}sd\tt1, 8(sp)",
        f"{indent}li\tt0, 0x10000014",
        f"{indent}.Lame_trace_wait_{label}:",
        f"{indent}lbu\tt1, 0(t0)",
        f"{indent}andi\tt1, t1, 32",
        f"{indent}beqz\tt1, .Lame_trace_wait_{label}",
        f"{indent}li\tt0, 0x10000000",
        f"{indent}li\tt1, {ord(marker)}",
        f"{indent}sb\tt1, 0(t0)",
        f"{indent}ld\tt1, 8(sp)",
        f"{indent}ld\tt0, 0(sp)",
        f"{indent}addi\tsp, sp, 16",
    ])


def emit_address_trace(indent):
    """Print fixed a0/a1 operands while preserving all caller-saved GPRs."""
    regs = ["ra"] + CALLER_SAVED_GPRS
    lines = [f"{indent}addi\tsp, sp, -128"]
    for i, reg in enumerate(regs):
        lines.append(f"{indent}sd\t{reg}, {8 * i}(sp)")
    lines.extend([
        f"{indent}call\tprint_uart_addr",
        f"{indent}li\ta0, 32",
        f"{indent}call\twrite_serial",
        f"{indent}ld\ta0, 16(sp)",
        f"{indent}call\tprint_uart_addr",
        f"{indent}li\ta0, 13",
        f"{indent}call\twrite_serial",
        f"{indent}li\ta0, 10",
        f"{indent}call\twrite_serial",
    ])
    for i, reg in reversed(list(enumerate(regs))):
        lines.append(f"{indent}ld\t{reg}, {8 * i}(sp)")
    lines.append(f"{indent}addi\tsp, sp, 128")
    return "\n".join(lines)


def emit_fixed_reg_sequence(indent, live_regs, moves, word, comment,
                            delay_nops=0, fence_after=False,
                            trace_marker=None, trace_address=False):
    """Emit a qwen3-style fixed-GPR AME instruction wrapper.

    Qwen3 FPGA AME instructions are validated with fixed integer-register
    encodings. LLVM may print the same AME mnemonic with arbitrary GPR
    operands, so preserve the fixed registers, move operands into the qwen3
    contract registers, execute the fixed .word, then restore the registers.
    """
    regs = list(dict.fromkeys(live_regs))
    stack_bytes = max(16, ((8 * len(regs) + 15) // 16) * 16)
    lines = [f"{indent}addi\tsp, sp, -{stack_bytes}"]
    for i, reg in enumerate(regs):
        lines.append(f"{indent}sd\t{reg}, {8 * i}(sp)")
    saved_offsets = {reg: 8 * i for i, reg in enumerate(regs)}
    for dst, src in moves:
        if dst == src:
            continue
        # Treat the fixed-register assignments as parallel moves.  A source
        # can itself be an earlier destination, e.g. an LLVM-emitted store
        #
        #   msce32.m acc0, (t1), t3
        #
        # needs t3=t1 and t0=old(t3).  Reading t3 after assigning the base
        # would pass the base address as the stride and wedge the AME store.
        # Fixed destination registers were saved above, so recover an
        # overlapping source from its pre-move stack slot.
        if src in saved_offsets:
            lines.append(f"{indent}ld\t{dst}, {saved_offsets[src]}(sp)")
        else:
            lines.append(f"{indent}mv\t{dst}, {src}")
    if AME_TRACE_ADDRS and trace_address:
        lines.append(emit_address_trace(indent))
    lines.append(f"{indent}.word 0x{word:08x}{comment}")
    if delay_nops:
        lines.append(f"{indent}# wait for asynchronous AME memory operation")
        for _ in range(delay_nops):
            lines.append(f"{indent}nop")
    if fence_after:
        lines.append(f"{indent}fence\trw, rw")
    if AME_FENCE_BEFORE_RESTORE:
        lines.append(f"{indent}fence\tiorw, iorw")
    # Keep the fixed operand registers live while the diagnostic MMIO
    # round-trip drains the asynchronous AME request.  Emitting the marker
    # after restoring the caller's registers lets the AME observe stale
    # dimensions, addresses, or strides and makes the trace perturb the very
    # operation it is meant to diagnose.
    if AME_TRACE_STAGES and trace_marker is not None:
        lines.append(emit_uart_marker(indent, trace_marker))
    for i, reg in reversed(list(enumerate(regs))):
        lines.append(f"{indent}ld\t{reg}, {8 * i}(sp)")
    lines.append(f"{indent}addi\tsp, sp, {stack_bytes}")
    return "\n".join(lines)

# ----- Config register instructions (msettype, msettilem/n/k) -----
# Format:
#   bits[6:0]   = 0b1110111
#   bits[11:7]  = rd
#   bits[14:12] = funct3
#   bits[19:15] = rs1
#   bits[24:20] = 0b00000
#   bits[25]    = 0
#   bits[31:26] = 0b000001

CONFIG_FUNCT3 = {
    "msettype":  0b100,  # funct3=4, but with opcode field considerations
    "msettilem": 0b101,
    "msettilen": 0b100,
    "msettilek": 0b110,
}


def encode_config_reg(mnemonic, rd_name, rs1_name):
    funct3 = CONFIG_FUNCT3[mnemonic]
    rd = GPR[rd_name]
    rs1 = GPR[rs1_name]
    # msettype uses funct6=0; msettile* use funct6=1 (verified against Qwen3 RTL)
    funct6 = 0b000000 if mnemonic == "msettype" else 0b000001
    word = 0
    word |= OPCODE_AME           # bits[6:0]
    word |= (rd & 0x1f) << 7     # bits[11:7]
    word |= (funct3 & 0x7) << 12 # bits[14:12]
    word |= (rs1 & 0x1f) << 15   # bits[19:15]
    # bits[24:20] = 0
    # bit[25] = 0
    word |= (funct6 & 0x3f) << 26  # bits[31:26]
    return word


# ----- Load tile instructions (mlae8.m, mlbe8.m, mlbte8.m) -----
# Format:
#   bits[6:0]   = 0b1110111
#   bits[10:7]  = md (tile reg index)
#   bits[11]    = tr (transpose flag)
#   bits[14:12] = eew
#   bits[19:15] = rs1 (base addr GPR)
#   bits[24:20] = rs2 (stride GPR)
#   bits[25]    = 0 (ls=0 for load)
#   bits[31:26] = funct6

LOAD_FUNCT6 = {
    "mlae": 0b000001,
    "mlbe": 0b000010,
    "mlce": 0b000000,  # load to accumulator
}

LOAD_EEW = {
    "8":  0b000,
    "16": 0b001,
    "32": 0b010,
    "64": 0b011,
}


def encode_load_tile(mnemonic, md_name, rs1_name, rs2_name):
    # Parse mnemonic: mlae8.m, mlbe8.m, mlbte8.m, mlce32.m, etc.
    m = re.match(r'(ml[abc]e|ml[abc]te|mltre|mlcte|mlacce)(\d+)\.m', mnemonic)
    if not m:
        return None
    base = m.group(1)
    bits = m.group(2)

    tr = 1 if base.endswith("te") or base in ("mltre", "mlcte") else 0
    base_key = base[:-2] + "e" if base.endswith("te") else base

    funct6 = LOAD_FUNCT6[base_key]
    eew = LOAD_EEW[bits]
    md = AME_REG[md_name]
    rs1 = GPR[rs1_name]
    rs2 = GPR[rs2_name]

    word = 0
    word |= OPCODE_AME
    word |= (md & 0xf) << 7
    word |= (tr & 0x1) << 11
    word |= (eew & 0x7) << 12
    word |= (rs1 & 0x1f) << 15
    word |= (rs2 & 0x1f) << 20
    # bit[25] = 0 (load)
    word |= (funct6 & 0x3f) << 26
    return word


# ----- Store acc instructions (msce32.m) -----
# Format:
#   bits[6:0]   = 0b1110111
#   bits[10:7]  = ms3 (acc reg index)
#   bits[11]    = tr
#   bits[14:12] = eew
#   bits[19:15] = rs1 (base addr GPR)
#   bits[24:20] = rs2 (stride GPR)
#   bits[25]    = 1 (store)
#   bits[31:26] = funct6

STORE_FUNCT6 = {
    "msce": 0b000000,  # store from accumulator
    "msae": 0b000001,
    "msbe": 0b000010,
}


def encode_store_acc(mnemonic, ms3_name, rs1_name, rs2_name):
    m = re.match(r'(ms[abc]e)(\d+)\.m', mnemonic)
    if not m:
        return None
    base = m.group(1)
    bits = m.group(2)

    funct6 = STORE_FUNCT6[base]
    eew = LOAD_EEW[bits]
    ms3 = AME_REG[ms3_name]
    rs1 = GPR[rs1_name]
    rs2 = GPR[rs2_name]
    tr = 0

    word = 0
    word |= OPCODE_AME
    word |= (ms3 & 0xf) << 7
    word |= (tr & 0x1) << 11
    word |= (eew & 0x7) << 12
    word |= (rs1 & 0x1f) << 15
    word |= (rs2 & 0x1f) << 20
    word |= (1 & 0x1) << 25  # store
    word |= (funct6 & 0x3f) << 26
    return word


# ----- MMA instructions (mma.w.mm, mqma.b.mm) -----
# Format:
#   bits[6:0]   = 0b1110111
#   bits[10:7]  = md (acc reg)
#   bits[11]    = ma (1 for accumulate)
#   bits[14:12] = eew
#   bits[18:15] = ms1 (tile reg 1)
#   bits[19]    = sn (signed: 1 for signed, 0 for unsigned)
#   bits[23:20] = ms2 (tile reg 2)
#   bits[24]    = sa (saturated)
#   bits[25]    = fp
#   bits[31:26] = funct6

# funct6: no-widen=001000, double=001001, quad=001010, oct=001011
MMA_FUNCT6 = {
    "mma":   0b001000,  # no-widen signed
    "mmau":  0b001000,  # no-widen unsigned (sn=0)
    "msma":  0b001000,  # no-widen saturated signed
    "msmau": 0b001000,  # no-widen saturated unsigned
    "mwma":  0b001001,  # double-widen signed
    "mwmau": 0b001001,  # double-widen unsigned
    "mqma":  0b001010,  # quad-widen signed
    "mqmau": 0b001010,  # quad-widen unsigned
    "moma":  0b001011,  # oct-widen signed
    "momau": 0b001011,  # oct-widen unsigned
}

# Element width suffixes for MMA
# mma.mm → eew=100 (use mtype.msew), mma.h.mm → eew=001, etc.
MMA_EEW = {
    "mm":   0b100,   # dynamic / use mtype
    "b.mm": 0b000,   # 8-bit
    "hb.mm": 0b111,  # 4-bit
    "h.mm": 0b001,   # 16-bit
    "w.mm": 0b010,   # 32-bit
    "dw.mm": 0b011,  # 64-bit
    "cf.mm": 0b000,  # 8-bit float
    "hf.mm": 0b001,  # 16-bit float
    "f.mm":  0b010,  # 32-bit float
    "d.mm":  0b011,  # 64-bit float
}

# For mnemonic: mqma.b.mm → base="mqma", suffix="b.mm"
MMA_MNEMONIC_RE = re.compile(
    r'^(m(?:ma|mau|sma|smau|wma|wmau|swma|swmau|qma|qmau|sqma|sqmau|oma|omau|soma|somau|fma|fwma|fqma))'
    r'\.([a-z0-9]+\.[a-z0-9]+)$'
)

# Wider regex for all mma variants
MMA_MNEMONIC_RE2 = re.compile(
    r'^(m\w+)\.([a-z0-9]+\.[a-z0-9]+)$'
)


def encode_mma(mnemonic, md_name, ms1_name, ms2_name):
    """Encode matrix multiply-accumulate instruction."""
    # Parse like: mqma.b.mm → base=mqma, suffix=b.mm
    m = MMA_MNEMONIC_RE2.match(mnemonic)
    if not m:
        return None
    base = m.group(1)
    suffix = m.group(2)

    # Determine properties
    funct6 = MMA_FUNCT6.get(base, 0b001000)
    eew = MMA_EEW.get(suffix, 0b100)

    # Signed/unsigned: 'u' appears in unsigned and saturated-unsigned variants
    # mma*, msma* → signed (sn=1)
    # mmau*, msmau* → unsigned (sn=0)
    # mf* → float (fp=1, sn=0)
    sn = 1  # default signed
    sa = 0  # default unsaturated
    fp = 0  # default integer

    if base.startswith("mf"):
        fp = 1
        sn = 0
    elif "u" in base and not base.startswith("ms"):
        # mmau, mqmau, etc. → unsigned
        sn = 0
    elif base.startswith("ms"):
        # msma, msmau, msqma, etc. → saturated
        sa = 1
        if "u" in base:
            sn = 0
    elif "u" in base:
        sn = 0

    md = AME_REG[md_name]
    ms1 = AME_REG[ms1_name]
    ms2 = AME_REG[ms2_name]

    word = 0
    word |= OPCODE_AME
    word |= (md & 0xf) << 7
    word |= (1 & 0x1) << 11     # ma = 1 (accumulate)
    word |= (eew & 0x7) << 12
    word |= (ms1 & 0xf) << 15
    word |= (sn & 0x1) << 19
    word |= (ms2 & 0xf) << 20
    word |= (sa & 0x1) << 24
    word |= (fp & 0x1) << 25
    word |= (funct6 & 0x3f) << 26
    return word


# ----- Element-wise instructions (msub.w.mm, etc.) -----
# Format: same as MMA but ma=0 (no accumulate)
# funct6 depends on operation type:
#   add/sub no-widen:   001000 (add) / 001010 (sub)
#   add/sub double:     001001 / 001011
#   min/max:            001100
#   mul:                001101
#   mulh/div:           001110
#   mul double:         001111
#   logic:              010000
#   shift:              010001
# For msub.w.mm specifically:
#   funct6 = 001010 (sub no-widen) — actually wait
# Looking at the .td: BOSC_AME_MSUB_NO_WIDEN has funct6=0b001010
# But BOSC_AME_MMA_NO_WIDEN has funct6=0b001000
# So msub uses funct6=001010

ELEWISE_FUNCT6 = {
    "maddu":  0b001000, "msaddu": 0b001000, "madd":   0b001000, "msadd":  0b001000,
    "msubu":  0b001010, "mssubu": 0b001010, "msub":   0b001010, "mssub":  0b001010,
    "mwaddu": 0b001001, "mwadd":  0b001001,
    "mwsubu": 0b001011, "mwsub":  0b001011,
    "mminu":  0b001100, "mmaxu":  0b001100, "mmin":   0b001100, "mmax":   0b001100,
    "msmulu": 0b001101, "mmul":   0b001101, "msmul":  0b001101,
    "mmulhu": 0b001110, "mmulh":  0b001110, "mmulhsu":0b001110, "msmulsu":0b001110,
    "mwmulu": 0b001111, "mwmul":  0b001111, "mwmulsu":0b001111,
    "mand":   0b010000, "mor":    0b010000, "mxor":   0b010000,
    "msll":   0b010001, "msrl":   0b010001, "msra":   0b010001,
}
ELEWISE_SA = {
    "msaddu": 1, "msadd": 1, "mssubu": 1, "mssub": 1,
    "mmaxu": 1, "mmax": 1,  # max uses sa=1 in encoding
    "msmulu": 1, "msmul": 1, "msmulsu": 1,
    "mor": 1, "mxor": 1, "msrl": 1, "msra": 1,
    # min uses sa=0
}
ELEWISE_SN = {
    "maddu": 0, "msaddu": 0, "madd": 1, "msadd": 1,
    "msubu": 0, "mssubu": 0, "msub": 1, "mssub": 1,
    "mminu": 0, "mmaxu": 0, "mmin": 1, "mmax": 1,
    "msmulu": 0, "mmul": 1, "msmul": 1,
    "mmulhu": 0, "mmulh": 1, "mmulhsu": 1, "msmulsu": 1,
    "mand": 0, "mor": 0, "mxor": 0,
    "msll": 0, "msrl": 0, "msra": 1,
}


def encode_elewise(mnemonic, md_name, ms1_name, ms2_name):
    """Encode element-wise instruction (msub.w.mm, etc.)."""
    m = MMA_MNEMONIC_RE2.match(mnemonic)
    if not m:
        return None
    base = m.group(1)
    suffix = m.group(2)

    funct6 = ELEWISE_FUNCT6.get(base, 0b001000)
    eew = MMA_EEW.get(suffix, 0b100)
    sa = ELEWISE_SA.get(base, 0)
    sn = ELEWISE_SN.get(base, 1)
    fp = 0
    if base.startswith("mf"):
        fp = 1
        sn = 0

    md = AME_REG[md_name]
    ms1 = AME_REG[ms1_name]
    ms2 = AME_REG[ms2_name]

    word = 0
    word |= OPCODE_AME
    word |= (md & 0xf) << 7
    # ma = 0 for element-wise
    word |= (eew & 0x7) << 12
    word |= (ms1 & 0xf) << 15
    word |= (sn & 0x1) << 19
    word |= (ms2 & 0xf) << 20
    word |= (sa & 0x1) << 24
    word |= (fp & 0x1) << 25
    word |= (funct6 & 0x3f) << 26
    return word


# ----- Config immediate instructions (msettypei, msettilemi/n/k) -----
# Format:
#   bits[6:0]   = 0b1110111
#   bits[11:7]  = rd
#   bits[14:12] = funct3
#   bits[24:15] = imm (10 bits)
#   bits[25]    = 1 (immediate)
#   bits[31:26] = funct6


def encode_config_imm(mnemonic, rd_name, imm_str):
    funct3 = {
        "msettypei":  0b100,
        "msettypehi": 0b101,
        "msettilemi": 0b101,
        "msettileni": 0b100,
        "msettileki": 0b110,
    }.get(mnemonic, 0)
    rd = GPR[rd_name]
    imm = int(imm_str, 0) & 0x3ff

    word = 0
    word |= OPCODE_AME
    word |= (rd & 0x1f) << 7
    word |= (funct3 & 0x7) << 12
    word |= (imm & 0x3ff) << 15
    word |= (1 & 0x1) << 25   # immediate
    word |= (0b000001 & 0x3f) << 26
    return word


# Main conversion logic

# Pattern for load instructions: mlae8.m tr0, (a1), a7  OR  mlae8.m tr0, a1, a7
LOAD_RE = re.compile(
    r'^\s*(ml[abc]e\d+\.m|ml[abc]te\d+\.m|mltre\d+\.m|mlcte\d+\.m|mlacce\d+\.m)\s+'
    r'(tr\d+|acc\d+)\s*,\s*\(?(\w+)\)?\s*,\s*(\w+)\s*$'
)

# Pattern for store instructions: msce32.m acc0, (a2), a7  OR  msce32.m acc0, a2, a7
STORE_RE = re.compile(
    r'^\s*(ms[abc]e\d+\.m|ms[abc]te\d+\.m|mstre\d+\.m|mscte\d+\.m|msacce\d+\.m)\s+'
    r'(tr\d+|acc\d+)\s*,\s*\(?(\w+)\)?\s*,\s*(\w+)\s*$'
)

# Pattern for MMA: mqma.b.mm acc0, tr0, tr1
MMA_RE = re.compile(
    r'^\s*(\w+\.\w+\.mm)\s+'
    r'(acc\d+)\s*,\s*(tr\d+|acc\d+)\s*,\s*(tr\d+|acc\d+)\s*$'
)

# Pattern for config register: msettype zero, a6  OR  msettilem zero, t4
CONFIG_REG_RE = re.compile(
    r'^\s*(msettype|msettilem|msettilen|msettilek)\s+'
    r'(\w+)\s*,\s*(\w+)\s*$'
)

# Pattern for config immediate: msettypei zero, 32
CONFIG_IMM_RE = re.compile(
    r'^\s*(msettypei|msettypehi|msettilemi|msettileni|msettileki)\s+'
    r'(\w+)\s*,\s*(\d+)\s*$'
)


def convert_line(line):
    """Convert a single line. Returns (converted_line, was_converted)."""
    stripped = line.strip()
    if not stripped or stripped.startswith('.') or stripped.startswith('#') \
            or stripped.startswith('//') or ':' in stripped.split()[0] \
            if stripped.split() else False:
        return line, False

    # Skip labels and directives
    first_token = stripped.split()[0] if stripped.split() else ""
    if first_token.endswith(':') or first_token.startswith('.'):
        return line, False

    # Try to match AME instructions
    word = None
    comment = ""
    delay_nops = 0
    fence_after = False

    # Config register (msettype, msettilem/n/k)
    m = CONFIG_REG_RE.match(stripped)
    if m:
        mnemonic, rd_name, rs1_name = m.group(1), m.group(2), m.group(3)
        if AME_USE_QWEN_FIXED_REGS and mnemonic in QWEN_FIXED_CONFIG:
            fixed_rs1, fixed_rd, fixed_word = QWEN_FIXED_CONFIG[mnemonic]
            live = CALLER_SAVED_GPRS if mnemonic == "msettype" else [fixed_rs1]
            if fixed_rd:
                live.append(fixed_rd)
            moves = [(fixed_rs1, rs1_name)]
            if mnemonic == "msettilek":
                moves = [("a3", rs1_name)]
            comment = f" # {mnemonic} {rd_name}, {rs1_name} [qwen fixed]"
            return emit_fixed_reg_sequence(line[:len(line) - len(line.lstrip())],
                                           live, moves, fixed_word, comment,
                                           AME_CONFIG_DELAY_NOPS,
                                           trace_marker=(
                                               {
                                                   "msettype": "T",
                                                   "msettilem": "M",
                                                   "msettilen": "N",
                                                   "msettilek": "K",
                                               }.get(mnemonic)
                                           )), True
        word = encode_config_reg(m.group(1), m.group(2), m.group(3))
        comment = f" # {m.group(1)} {m.group(2)}, {m.group(3)}"

    # Config immediate (msettypei, msettilemi/n/k)
    if word is None:
        m = CONFIG_IMM_RE.match(stripped)
        if m:
            word = encode_config_imm(m.group(1), m.group(2), m.group(3))
            comment = f" # {m.group(1)} {m.group(2)}, {m.group(3)}"

    # Load
    if word is None:
        m = LOAD_RE.match(stripped)
        if m:
            key = (m.group(1), m.group(2))
            if AME_USE_QWEN_FIXED_REGS and key in QWEN_FIXED_LOAD:
                fixed_rs1, fixed_rs2, fixed_word = QWEN_FIXED_LOAD[key]
                comment = (f" # {m.group(1)} {m.group(2)}, "
                           f"({m.group(3)}), {m.group(4)} [qwen fixed]")
                return emit_fixed_reg_sequence(
                    line[:len(line) - len(line.lstrip())],
                    [fixed_rs1, fixed_rs2],
                    [(fixed_rs1, m.group(3)), (fixed_rs2, m.group(4))],
                    fixed_word,
                    comment,
                    AME_LOAD_DELAY_NOPS,
                    trace_marker=(
                        {
                            ("mlce32.m", "acc3"): "Z",
                            ("mlae8.m", "tr0"): "A",
                            ("mlbe8.m", "tr4"): "4",
                            ("mlbe8.m", "tr5"): "5",
                            ("mlbe8.m", "tr6"): "6",
                            ("mlbe8.m", "tr7"): "7",
                        }.get(key)
                    ),
                    trace_address=(key == ("mlbe8.m", "tr4")),
                ), True
            word = encode_load_tile(m.group(1), m.group(2), m.group(3), m.group(4))
            comment = f" # {m.group(1)} {m.group(2)}, ({m.group(3)}), {m.group(4)}"

    # Store
    if word is None:
        m = STORE_RE.match(stripped)
        if m and any(c.isdigit() for c in m.group(1)):
            key = (m.group(1), m.group(2))
            if AME_USE_QWEN_FIXED_REGS and key in QWEN_FIXED_STORE:
                fixed_rs1, fixed_rs2, fixed_word = QWEN_FIXED_STORE[key]
                comment = (f" # {m.group(1)} {m.group(2)}, "
                           f"{m.group(3)}, {m.group(4)} [qwen fixed]")
                return emit_fixed_reg_sequence(
                    line[:len(line) - len(line.lstrip())],
                    [fixed_rs1, fixed_rs2],
                    [(fixed_rs1, m.group(3)), (fixed_rs2, m.group(4))],
                    fixed_word,
                    comment,
                    AME_STORE_DELAY_NOPS,
                    AME_STORE_FENCE_AFTER,
                    trace_marker=(
                        "S" if key == ("msce32.m", "acc3") else None
                    ),
                ), True
            word = encode_store_acc(m.group(1), m.group(2), m.group(3), m.group(4))
            comment = f" # {m.group(1)} {m.group(2)}, {m.group(3)}, {m.group(4)}"

    # MMA / element-wise
    if word is None:
        m = MMA_RE.match(stripped)
        if m:
            mnem = m.group(1)
            # Determine if this is MMA (ma=1) or element-wise (ma=0)
            base = mnem.split('.')[0]
            if base in MMA_FUNCT6 or base.startswith('mf'):
                word = encode_mma(mnem, m.group(2), m.group(3), m.group(4))
            else:
                word = encode_elewise(mnem, m.group(2), m.group(3), m.group(4))
            comment = f" # {mnem} {m.group(2)}, {m.group(3)}, {m.group(4)}"
            delay_nops = AME_MMA_DELAY_NOPS
            fence_after = AME_MMA_FENCE_AFTER

    if word is not None:
        indent = line[:len(line) - len(line.lstrip())]
        lines = [f"{indent}.word 0x{word:08x}{comment}"]
        if delay_nops:
            lines.append(f"{indent}# wait for asynchronous AME compute operation")
            for _ in range(delay_nops):
                lines.append(f"{indent}nop")
        if fence_after:
            lines.append(f"{indent}fence\trw, rw")
        if AME_TRACE_STAGES and m is not None and m.group(1) == "mqma.b.mm":
            marker = {
                "acc0": "a",
                "acc1": "b",
                "acc2": "c",
                "acc3": ".",
            }.get(m.group(2))
            if marker is not None:
                lines.append(emit_uart_marker(indent, marker))
        return "\n".join(lines), True

    return line, False


def main():
    converted = 0
    converted_lines = []
    for line in sys.stdin:
        new_line, was_converted = convert_line(line.rstrip('\r\n'))
        converted_lines.append(new_line + '\n')
        if was_converted:
            converted += 1
    for line in converted_lines:
        sys.stdout.write(line)
    if converted > 0:
        print(f"# Converted {converted} AME instructions to .word", file=sys.stderr)


if __name__ == "__main__":
    main()
