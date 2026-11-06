#!/usr/bin/env python3
# ===- restrict_fpga_assembly.py -------------------------------------------
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

"""Validate NR AME/RVV forms and insert fences around AME/vector memory ops in compiler assembly.

Inputs: Assembly from ame_to_word.py (stdin/files); opt-in expansion/fence flags.
Outputs: Filtered assembly safe for the NR FPGA assembler path; fails closed on unverified forms.
When used: Immediately after ame_to_word.py in the NR example assembly pipeline.

Adapted from ModelZoo examples/buddy-qwen35-fpga/python/qwen35/compiler/
restrict_fpga_assembly.py, commit 8815b74fb6d3cd6288c4d99ac6fd7c5d041a7cc3:
https://gitlink.org.cn/michaelcjl/ModelZoo.git

Run AFTER ame_to_word.py. NR rejects transposed B loads, unverified AME
encodings, vector CSR reads/spills and raw RVV encodings. Data sections are
preserved exactly: an AME-looking floating-point constant is not an instruction.
This is a checked compiler-output filter, not a general-purpose assembler.
Opt-in --no-fences skips fencing for inspection or variants that intentionally
provide ordering elsewhere; every check still runs. Opt-in
--coalesce-fences shares an adjacent identical fence between instructions;
every AME/vector memory instruction still has both adjacent fences. Labels,
directives, and all other instructions prevent sharing across that boundary.
Opt-in --expand-f32-sub replaces compiler-emitted e32 vfsub.vf with scalar
sign injection, allowed vfadd.vf, and scalar restoration. This needs no spare
register and preserves IEEE subtraction and floating-point exception flags.
Opt-in --expand-vector-merges replaces signed-five-bit vmerge.vim with allowed
vmerge.vxm. It saves t6 in a temporary 16-byte scalar stack slot, materializes
the immediate, and restores t6 and sp; no vector register is borrowed.
Opt-in --order-vector-copies moves an independent scalar counter increment
past a matching byte-vector store. NR can stall when that increment separates
the load and store; both memory instructions retain their surrounding fences.
Opt-in --expand-f32-scalar-moves replaces e32 vmv.s.x with allowed fmv.w.x
and vfmv.s.f, preserving ft11 through a temporary 16-byte scalar stack slot.
--expand-f32-sub also enables this bit-preserving scalar seed expansion for
existing NR build recipes. The new slot increases peak stack use by 16 bytes;
it stores one 64-bit scalar register and introduces no vector spill or buffer.
The f32 expansions accept a vector configuration proved on every reachable
control-flow path; unknown entry state, calls and opaque control flow cannot
provide that proof. No physical VLEN is assumed or read.
By default, unmasked fixed-AVL e32 vfredosum.vs is expanded into sequential
scalar fadd.s in lane order, using a bounded stack snapshot of active lanes.
All scratch registers are saved, and stack alignment and vector configuration
are preserved. This checked compiler-output path assumes vstart=0. Unknown,
dynamic, masked, fractional-LMUL and non-e32 forms fail closed. Diagnostic
--allow-vector-ordered-reductions explicitly retains the vector instruction.
Production recipes instead use --reject-vector-ordered-reductions after
scalar SSA lowering: the legacy vector/FPR roundtrip expansion is retained
for diagnosis and ISA regression, but is not reliable on the FPGA.
Opt-in --order-vector-adds handles the verified e32/16/m4 integer-add loop:
replace its independent counter increment with c.nop and emit the increment
after the vector store's trailing fence. Vector arithmetic stays unchanged.
"""

import argparse
import re
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ame_to_word import canonical_gpr
from nr_isa import (
    RVV_ALLOWED,
    VECTOR_MEMORY,
    is_ame,
    is_vector,
    validate_ame_word,
)


def _order_vector_copy(source_lines, position, code):
    """Move one independent self-add after an unmasked byte-copy store."""
    vector = r"(v(?:[0-9]|[12][0-9]|3[01]))"
    address = r"\(\s*(\w+)\s*\)"
    load = re.fullmatch(r"vle8\.v\s+" + vector + r"\s*,\s*" + address, code)
    if not load:
        return

    def next_statement(start):
        for index in range(start, len(source_lines)):
            statement = source_lines[index].split("#", 1)[0].strip()
            if statement and not statement.startswith("//"):
                return index, statement
        return len(source_lines), ""

    increment_index, increment = next_statement(position + 1)
    add = re.fullmatch(
        r"addi\s+(\w+)\s*,\s*(\w+)\s*,\s*"
        r"([+-]?(?:0x[0-9a-fA-F]+|[0-9]+))",
        increment,
    )
    if not add:
        return
    store_index, statement = next_statement(increment_index + 1)
    store = re.fullmatch(
        r"vse8\.v\s+" + vector + r"\s*,\s*" + address, statement
    )
    if not store or load[1] != store[1]:
        return
    try:
        destination, source = map(canonical_gpr, add.group(1, 2))
        bases = {canonical_gpr(load[2]), canonical_gpr(store[2])}
    except ValueError:
        return
    immediate = int(add[3], 16 if "x" in add[3].lower() else 10)
    if (
        destination != source
        or destination in bases | {"zero", "sp"}
        or not -2048 <= immediate <= 2047
    ):
        return
    # The existing executable-section parser calls this only for a validated
    # load. Every other statement, including labels and CFI, blocks matching.
    # Keep each original instruction and its inline comment intact.
    source_lines[increment_index], source_lines[store_index] = (
        source_lines[store_index],
        source_lines[increment_index],
    )


def _order_vector_add(source_lines, position, code):
    """Recognize only the measured two-load, in-place i32 add loop."""
    configuration = re.fullmatch(
        r"vsetivli\s+(\w+)\s*,\s*16\s*,\s*e32\s*,\s*m4\s*,\s*ta\s*,\s*ma",
        code,
    )
    if not configuration:
        return
    previous = next(
        (
            line.split("#", 1)[0].strip()
            for line in reversed(source_lines[:position])
            if line.split("#", 1)[0].strip()
            and not line.split("#", 1)[0].strip().startswith("//")
        ),
        "",
    )
    label = re.fullmatch(r"([\w.$]+):", previous)
    if not label:
        return
    statements = []
    for index in range(position + 1, len(source_lines)):
        statement = source_lines[index].split("#", 1)[0].strip()
        if statement and not statement.startswith("//"):
            statements.append((index, statement))
            if len(statements) == 8:
                break
    if len(statements) != 8:
        return
    vector = r"(v(?:[0-9]|[12][0-9]|3[01]))"
    address = r"\(\s*(\w+)\s*\)"
    load = r"vle32\.v\s+" + vector + r"\s*,\s*" + address
    increment = r"addi\s+(\w+)\s*,\s*(\w+)\s*,\s*([+-]?[0-9]+)"
    patterns = (
        load,
        load,
        r"vadd\.vv\s+" + vector + r"\s*,\s*" + vector + r"\s*,\s*" + vector,
        increment,
        r"vse32\.v\s+" + vector + r"\s*,\s*" + address,
        increment,
        increment,
        r"bge\s+(\w+)\s*,\s*(\w+)\s*,\s*([\w.$]+)",
    )
    matches = [
        re.fullmatch(pattern, statement)
        for pattern, (_, statement) in zip(patterns, statements)
    ]
    if not all(matches):
        return
    left, right, add, counter, store, left_step, right_step, branch = matches
    try:
        destination = canonical_gpr(left[2])
        source = canonical_gpr(right[2])
        induction = canonical_gpr(counter[1])
        if (
            canonical_gpr(configuration[1]) != "zero"
            or len({destination, source, induction}) != 3
            or induction in {"zero", "sp"}
            or canonical_gpr(counter[2]) != induction
            or int(counter[3]) != 16
            or canonical_gpr(store[2]) != destination
            or canonical_gpr(left_step[1]) != destination
            or canonical_gpr(left_step[2]) != destination
            or int(left_step[3]) != 64
            or canonical_gpr(right_step[1]) != source
            or canonical_gpr(right_step[2]) != source
            or int(right_step[3]) != 64
            or canonical_gpr(branch[1]) in {destination, source, induction}
            or canonical_gpr(branch[2]) != induction
            or branch[3] != label[1]
        ):
            return
    except ValueError:
        return
    if (
        not (left[1] == add[1] == add[2] == store[1])
        or right[1] != add[3]
        or left[1] == right[1]
        or int(left[1][1:]) % 4
        or int(right[1][1:]) % 4
    ):
        return
    counter_index, store_index = statements[3][0], statements[4][0]
    original_counter = source_lines[counter_index]
    # Preserve the measured spacing before the store, including its fences.
    # The caller emits the original counter only AFTER the trailing fence.
    source_lines[counter_index] = "\tc.nop"
    return store_index, original_counter


def _raw_words(code):
    match = re.fullmatch(r"\.(?:word|4byte|long)\s+(.+)", code)
    if not match:
        return None
    words = []
    for expression in match[1].split(","):
        expression = expression.strip()
        if not re.fullmatch(r"-?(?:0x[0-9a-fA-F]+|[0-9]+)", expression):
            raise ValueError("executable raw words must be integer literals")
        value = (
            int(expression, 0) if "x" in expression.lower() else int(expression)
        )
        if not -(1 << 31) <= value <= 0xFFFFFFFF:
            raise ValueError("raw instruction does not fit in 32 bits")
        words.append(value & 0xFFFFFFFF)
    return words


def _expand_vector_merge(code):
    """Preserve an immediate merge without needing a free vector or GPR."""
    vector = r"(v(?:[0-9]|[12][0-9]|3[01]))"
    operands = re.fullmatch(
        r"vmerge\.vim\s+"
        + vector
        + r"\s*,\s*"
        + vector
        + r"\s*,\s*([+-]?(?:0[xX][0-9a-fA-F]+|[0-9]+))\s*,\s*v0",
        code,
    )
    if not operands:
        raise ValueError(
            "vmerge.vim expansion requires vector registers, a signed "
            "five-bit immediate and explicit v0 mask"
        )
    destination, source, literal = operands.groups()
    immediate = int(literal, 16 if "x" in literal.lower() else 10)
    if destination == "v0" or not -16 <= immediate <= 15:
        raise ValueError(
            "vmerge.vim expansion requires a destination other than v0 "
            "and an immediate in [-16,15]"
        )
    # RISC-V has no red zone, so do not touch caller-owned stack contents.
    # Keep 16-byte stack alignment, and restore every scalar register before
    # the next original instruction. The scalar operand is sign-extended by
    # vmerge.vxm, exactly as the original signed-five-bit immediate for all SEW.
    return (
        "\taddi\tsp, sp, -16",
        "\tsd\tt6, 0(sp)",
        f"\tli\tt6, {immediate}",
        f"\tvmerge.vxm\t{destination}, {source}, t6, v0",
        "\tld\tt6, 0(sp)",
        "\taddi\tsp, sp, 16",
    )


def _proved_vector_sew(source_lines, *, full_config=False):
    """Meet SEW or fixed-AVL configuration facts over the compiler CFG."""
    nodes = []
    labels = {}
    entries = []
    executable = True
    previous = True
    sections = []
    function_entries = set()
    for number, line in enumerate(source_lines):
        code = line.split("#", 1)[0].strip()
        if not code or code.startswith("//"):
            continue
        mnemonic = code.split()[0]
        function = re.fullmatch(r"\.type\s+([\w.$]+)\s*,\s*[@%]function", code)
        exported = re.fullmatch(r"\.(?:globl|global)\s+([\w.$]+)", code)
        if function or exported:
            function_entries.add((function or exported)[1])
        if mnemonic in (".text", ".data", ".bss", ".rodata", ".sdata", ".sbss"):
            previous, executable = executable, mnemonic == ".text"
        elif mnemonic in (".section", ".pushsection"):
            if mnemonic == ".pushsection":
                sections.append(executable)
            fields = code.split(None, 1)[1].split(",")
            name = fields[0].strip().strip('"')
            flags = fields[1].strip().strip('"') if len(fields) > 1 else ""
            previous, executable = (
                executable,
                ("x" in flags or name == ".text" or name.startswith(".text.")),
            )
        elif mnemonic == ".popsection":
            if not sections:
                return {}
            previous, executable = executable, sections.pop()
        elif mnemonic == ".previous":
            executable, previous = previous, executable
        if not executable:
            continue
        label = re.match(r"^([\w.$]+):\s*(.*)$", code)
        if label:
            if label[1] in labels:
                return {}
            labels[label[1]] = len(nodes)
            if not label[1].startswith(".L") or label[1] in function_entries:
                entries.append(len(nodes))
            code = label[2]
        nodes.append((number, code))
    if not nodes:
        return {}
    conditional = {
        "beq",
        "bne",
        "blt",
        "bge",
        "bltu",
        "bgeu",
        "beqz",
        "bnez",
        "bltz",
        "bgez",
        "blez",
        "bgtz",
        "bgt",
        "ble",
        "bgtu",
        "bleu",
    }
    edges = []
    for i, (_, code) in enumerate(nodes):
        mnemonic = code.split()[0] if code else ""
        successor = [i + 1] if i + 1 < len(nodes) else []
        if mnemonic in conditional | {"c.beqz", "c.bnez"} or mnemonic in (
            "j",
            "c.j",
        ):
            target = code.split(",")[-1].split()[-1]
            if target not in labels:
                return {}  # An opaque edge can enter a local block anywhere.
            successor = ([] if mnemonic in ("j", "c.j") else successor) + [
                labels[target]
            ]
        elif mnemonic in ("ret", "c.ret"):
            successor = []
        elif (
            mnemonic == "tail"
            or mnemonic
            in (
                "jr",
                "jal",
                "jalr",
                "c.jr",
                "c.jalr",
                "c.jal",
            )
            or (
                mnemonic
                in (
                    "mret",
                    "sret",
                    "uret",
                    "dret",
                    "ecall",
                    "ebreak",
                    "c.ebreak",
                    "wfi",
                    ".insn",
                )
                or mnemonic.startswith(("cm.j", "cm.popret"))
                or mnemonic.startswith("c.b")
                or mnemonic.startswith("j")
                or (
                    mnemonic.startswith("b")
                    and mnemonic
                    not in {
                        "bclr",
                        "bclri",
                        "bext",
                        "bexti",
                        "binv",
                        "binvi",
                        "bset",
                        "bseti",
                    }
                )
            )
        ):
            return {}
        words = _raw_words(code)
        if words is not None and any(not is_ame(word) for word in words):
            return {}
        edges.append(successor)
    incoming = [None] * len(nodes)  # None = unreachable; 0 = unknown SEW.
    pending = deque()
    for entry in {0, *entries}:
        incoming[entry] = 0
        pending.append(entry)
    while pending:
        i = pending.popleft()
        _, code = nodes[i]
        mnemonic = code.split()[0] if code else ""
        state = incoming[i]
        if mnemonic in ("vsetvli", "vsetivli"):
            config = re.fullmatch(
                r"vseti?vli\s+\w+\s*,\s*(\w+)\s*,\s*"
                r"e(8|16|32|64)\s*,\s*(m(?:f[248]|[1248]))\s*,\s*"
                r"(t[au])\s*,\s*(m[au])",
                code,
            )
            state = 0
            if config:
                if not full_config:
                    state = int(config[2])
                elif mnemonic == "vsetivli":
                    avl = config[1]
                    if re.fullmatch(r"(?:0x[0-9a-fA-F]+|[0-9]+)", avl):
                        avl = int(avl, 16 if "x" in avl else 10)
                        if 0 <= avl <= 31:
                            state = (
                                avl,
                                int(config[2]),
                                *config.group(3, 4, 5),
                            )
        elif (
            mnemonic in ("call", "vsetvl")
            or mnemonic.startswith("csr")
            or mnemonic.startswith(".")
            and not mnemonic.startswith(
                (
                    ".cfi_",
                    ".p2align",
                    ".align",
                    ".file",
                    ".ident",
                    ".type",
                    ".size",
                    ".globl",
                    ".local",
                    ".attribute",
                    ".option",
                )
            )
        ):
            state = 0
        for target in edges[i]:
            old = incoming[target]
            merged = state if old is None else old if old == state else 0
            if old != merged:
                incoming[target] = merged
                pending.append(target)
    return {
        number: incoming[i]
        for i, (number, _) in enumerate(nodes)
        if incoming[i]
    }


def _expand_f32_scalar_move(code):
    operands = re.fullmatch(
        r"vmv\.s\.x\s+(v(?:[0-9]|[12][0-9]|3[01]))\s*,\s*(\w+)", code
    )
    if not operands:
        raise ValueError("f32 scalar move expansion requires compiler operands")
    destination, source = operands.groups()
    source = canonical_gpr(source)
    if source == "sp":
        raise ValueError(
            "f32 scalar move expansion cannot change its source sp"
        )
    return (
        "\taddi\tsp, sp, -16",
        "\tfsd\tft11, 0(sp)",
        f"\tfmv.w.x\tft11, {source}",
        f"\tvfmv.s.f\t{destination}, ft11",
        "\tfld\tft11, 0(sp)",
        "\taddi\tsp, sp, 16",
    )


def _expand_f32_ordered_reduction(code, configuration, label, insert_fences):
    """Snapshot active f32 lanes and add in order with saved scalar scratch.

    Normal compiler execution has vstart=0. Repeating the same fixed-AVL
    configuration returns the same actual VL without reading vector CSRs.
    AVL bounds the buffer independently of VLEN. VL=0 performs no FP work.
    """
    vector = r"(v(?:[0-9]|[12][0-9]|3[01]))"
    operands = re.fullmatch(
        r"vfredosum\.vs\s+"
        + vector
        + r"\s*,\s*"
        + vector
        + r"\s*,\s*"
        + vector,
        code,
    )
    if (
        not configuration
        or configuration[1] != 32
        or configuration[2] not in {"m1", "m2", "m4", "m8"}
        or not operands
    ):
        raise ValueError(
            "ordered f32 scalar reduction requires a CFG-proved fixed-AVL "
            "e32 integer-LMUL vsetivli and unmasked compiler operands; "
            "use --allow-vector-ordered-reductions only for hardware diagnostics"
        )
    destination, source, seed = operands.groups()
    avl, sew, lmul, tail, mask = configuration
    if int(source[1:]) % int(lmul[1:]):
        raise ValueError("ordered reduction source group is not LMUL-aligned")
    size = 32 + ((4 * avl + 15) // 16) * 16
    done, loop = label + "_done", label + "_loop"
    lines = [
        f"\taddi\tsp, sp, -{size}",
        "\tsd\tt5, 0(sp)",
        "\tsd\tt6, 8(sp)",
        "\tfsd\tft10, 16(sp)",
        "\tfsd\tft11, 24(sp)",
        f"\tvsetivli\tt6, {avl}, e{sew}, {lmul}, {tail}, {mask}",
        f"\tbeqz\tt6, {done}",
        f"\tvfmv.f.s\tft10, {seed}",
        "\taddi\tt5, sp, 32",
    ]
    if insert_fences:
        lines.append("\tfence\trw, rw")
    lines.append(f"\tvse32.v\t{source}, (t5)")
    if insert_fences:
        lines.append("\tfence\trw, rw")
    lines.extend(
        (
            loop + ":",
            "\tflw\tft11, 0(t5)",
            "\tfadd.s\tft10, ft10, ft11",
            "\taddi\tt5, t5, 4",
            "\taddi\tt6, t6, -1",
            f"\tbnez\tt6, {loop}",
            f"\tvfmv.s.f\t{destination}, ft10",
            done + ":",
            "\tfld\tft11, 24(sp)",
            "\tfld\tft10, 16(sp)",
            "\tld\tt6, 8(sp)",
            "\tld\tt5, 0(sp)",
            f"\taddi\tsp, sp, {size}",
        )
    )
    return lines


def transform(
    source,
    coalesce_fences=False,
    expand_f32_sub=False,
    order_vector_copies=False,
    order_vector_adds=False,
    insert_fences=True,
    expand_vector_merges=False,
    expand_f32_scalar_moves=False,
    scalarize_f32_ordered_reductions=True,
    reject_vector_ordered_reductions=False,
    reject_vector_max_reductions=False,
):
    output = []
    executable = True
    previous = True
    sections = []
    last_fence = False
    deferred_adds = {}

    def emit(line):
        nonlocal last_fence
        source_code, separator, source_comment = line.partition("#")
        code = source_code.strip()
        if not code or code.startswith("//"):
            output.append(line)
            return
        # Do not discard an annotation attached to a fence while sharing it;
        # generated fences have no comments, but preserving hand-written input
        # makes this transformation lossless outside instruction bytes.
        same_fence = bool(
            executable
            and not (separator and source_comment.strip())
            and re.fullmatch(r"fence\s+rw\s*,\s*rw", code)
        )
        if coalesce_fences and same_fence and last_fence:
            return
        output.append(line)
        # A label must break sharing even though it occupies no instruction
        # bytes: a branch to it must execute the following operation's fence.
        # Unknown directives are likewise conservative barriers, including
        # alignment, section transitions and assembler metadata.
        last_fence = same_fence

    source_lines = source.splitlines()
    proved_sew = _proved_vector_sew(source_lines)
    proved_config = (
        _proved_vector_sew(source_lines, full_config=True)
        if scalarize_f32_ordered_reductions
        else {}
    )
    # Avoid collisions even with forward references in the input.
    reduction_prefix = ".Lnr_scalar_sum"
    while reduction_prefix in source:
        reduction_prefix += "_"
    for number, original_line in enumerate(source_lines, 1):
        try:
            line = original_line
            code = line.split("#", 1)[0].strip()
            if not code or code.startswith("//"):
                emit(line)
                continue
            # Do not let a same-line label hide raw instructions from checks.
            label = re.match(r"^([\w.$]+:)\s*(.*)$", code)
            if label:
                if not label[2] or not executable:
                    emit(line)
                    continue
                emit(label[1])
                line = "\t" + label[2]
                code = label[2]
            # LLVM's custom xboscame ISA attribute is unknown to the assembler.
            if re.match(r"\.attribute\s+5,", code):
                last_fence = False
                continue
            mnemonic = code.split()[0]
            if mnemonic in (
                ".text",
                ".data",
                ".bss",
                ".rodata",
                ".sdata",
                ".sbss",
            ):
                previous, executable = executable, mnemonic == ".text"
            elif mnemonic in (".section", ".pushsection"):
                if mnemonic == ".pushsection":
                    sections.append(executable)
                fields = code.split(None, 1)[1].split(",")
                name = fields[0].strip().strip('"')
                flags = fields[1].strip().strip('"') if len(fields) > 1 else ""
                previous, executable = (
                    executable,
                    (
                        "x" in flags
                        or name == ".text"
                        or name.startswith(".text.")
                    ),
                )
            elif mnemonic == ".popsection":
                if not sections:
                    raise ValueError("unmatched .popsection")
                previous, executable = executable, sections.pop()
            elif mnemonic == ".previous":
                executable, previous = previous, executable
            if not executable:
                emit(original_line)
                continue
            if ";" in code:
                raise ValueError(
                    "multiple assembly statements must be on separate lines"
                )
            if reject_vector_ordered_reductions and mnemonic == "vfredosum.vs":
                raise ValueError(
                    "ordered reduction reached FPGA assembly; enable "
                    "lower-bosc-ame scalar-f32-ordered-reductions=true "
                    "for a supported scalar SSA lowering"
                )
            if reject_vector_max_reductions and mnemonic == "vfredmax.vs":
                raise ValueError(
                    "max reduction reached FPGA assembly; enable "
                    "lower-bosc-ame scalar-f32-max-reductions=true "
                    "for a supported scalar SSA lowering"
                )
            if scalarize_f32_ordered_reductions and mnemonic == "vfredosum.vs":
                for expanded_line in _expand_f32_ordered_reduction(
                    code,
                    proved_config.get(number - 1),
                    f"{reduction_prefix}_{number}",
                    insert_fences,
                ):
                    emit(expanded_line)
                continue
            if expand_f32_sub and mnemonic == "vfsub.vf":
                operands = re.fullmatch(
                    r"vfsub\.vf\s+(v(?:[12]?[0-9]|3[01])),\s*"
                    r"(v(?:[12]?[0-9]|3[01])),\s*"
                    r"(f(?:[0-9]|[12][0-9]|3[01])|ft(?:[0-9]|1[01])|"
                    r"fs(?:[0-9]|1[01])|fa[0-7])(,\s*v0\.t)?",
                    code,
                )
                if proved_sew.get(number - 1) != 32 or not operands:
                    raise ValueError(
                        "vfsub.vf expansion requires a proved e32 vsetvli "
                        "and compiler f32 operands"
                    )
                destination, vector, scalar, mask = operands.groups()
                # LLVM f32 scalar values are NaN-boxed. fneg.s is a sign-bit
                # operation (no rounding or exceptions); doing it twice also
                # restores signed zeros, infinities, and NaN payloads exactly.
                # x - y == x + (-y) for every IEEE rounding mode, including
                # signed zero. vfadd raises the same flags as vfsub.
                emit(f"\tfneg.s\t{scalar}, {scalar}")
                emit(
                    f"\tvfadd.vf\t{destination}, {vector}, {scalar}{mask or ''}"
                )
                emit(f"\tfneg.s\t{scalar}, {scalar}")
                continue
            if (
                expand_f32_scalar_moves or expand_f32_sub
            ) and mnemonic == "vmv.s.x":
                if proved_sew.get(number - 1) != 32:
                    raise ValueError(
                        "f32 scalar move requires a proved e32 configuration"
                    )
                for expanded_line in _expand_f32_scalar_move(code):
                    emit(expanded_line)
                continue
            if expand_vector_merges and mnemonic == "vmerge.vim":
                for expanded_line in _expand_vector_merge(code):
                    emit(expanded_line)
                continue
            if mnemonic.startswith("v") and mnemonic not in RVV_ALLOWED:
                raise ValueError(f"RVV instruction not allowed: {code}")
            if re.search(
                r"\b(?:vlenb|vtype|vl)\b", code
            ) and mnemonic.startswith("csr"):
                raise ValueError(f"vector CSR access not allowed: {code}")
            # Numeric vector CSRs are equivalent to their names.
            if mnemonic.startswith("csr"):
                operands = (
                    code.split(None, 1)[1]
                    if " " in code or "\t" in code
                    else ""
                )
                if re.search(
                    r"(?:^|[,\s])(?:0x[cC]2[012]|310[456])(?:$|[,\s])", operands
                ):
                    raise ValueError(f"vector CSR access not allowed: {code}")
            if mnemonic == ".insn":
                raise ValueError(
                    ".insn bypasses NR instruction checks; use encoder mnemonics"
                )
            if (
                mnemonic.startswith(("mset", "ml", "ms", "mq", "mf", "mw"))
                or ".mm" in mnemonic
            ):
                raise ValueError(
                    "unencoded AME mnemonic; run ame_to_word.py first"
                )
            words = _raw_words(code)
            if words is not None:
                # Multiple words are split so every AME word gets its own pair
                # of fences, even when emitted on the same original line.
                for word in words:
                    if is_vector(word):
                        raise ValueError("raw RVV encoding bypasses allowlist")
                    if (
                        word & 0x7F == 0x73
                        and (word >> 12) & 7
                        and word >> 20 in (0xC20, 0xC21, 0xC22)
                    ):
                        raise ValueError(
                            "raw vector CSR access bypasses allowlist"
                        )
                    if is_ame(word):
                        validate_ame_word(word)
                        if insert_fences:
                            emit("\tfence\trw, rw")
                    emit(line if len(words) == 1 else f"\t.word 0x{word:08x}")
                    if is_ame(word) and insert_fences:
                        emit("\tfence\trw, rw")
                continue
            if order_vector_adds and not label:
                moved = _order_vector_add(source_lines, number - 1, code)
                if moved:
                    store_index, counter = moved
                    deferred_adds[store_index] = counter
            if mnemonic in VECTOR_MEMORY:
                if order_vector_copies and not label:
                    _order_vector_copy(source_lines, number - 1, code)
                if insert_fences:
                    emit("\tfence\trw, rw")
            emit(line)
            if mnemonic in VECTOR_MEMORY and insert_fences:
                emit("\tfence\trw, rw")
            if number - 1 in deferred_adds:
                emit(deferred_adds.pop(number - 1))
        except (ValueError, IndexError) as error:
            raise ValueError(f"line {number}: {error}") from error
    if sections:
        raise ValueError("unclosed .pushsection")
    return "\n".join(output) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    reduction_policy = parser.add_mutually_exclusive_group()
    reduction_policy.add_argument(
        "--reject-vector-ordered-reductions",
        action="store_true",
        help="production safeguard: require ordered reductions to be lowered "
        "to scalar SSA before assembly; reject any residual vfredosum.vs",
    )
    reduction_policy.add_argument(
        "--allow-vector-ordered-reductions",
        action="store_true",
        help="diagnostic opt-out: retain vfredosum.vs instead of the default "
        "fixed-AVL f32 scalar workaround; requires matching ELF audit opt-out",
    )
    max_policy = parser.add_mutually_exclusive_group()
    max_policy.add_argument(
        "--reject-vector-max-reductions",
        action="store_true",
        help="production safeguard: require max reductions to be lowered "
        "to scalar SSA before assembly; reject any residual vfredmax.vs",
    )
    max_policy.add_argument(
        "--allow-vector-max-reductions",
        action="store_true",
        help="diagnostic opt-out: retain vfredmax.vs; requires matching "
        "lowering and ELF audit opt-outs",
    )
    parser.add_argument(
        "--expand-f32-scalar-moves",
        action="store_true",
        help="expand proved-e32 vmv.s.x through fmv.w.x and allowed vfmv.s.f; "
        "preserve ft11 with a temporary 16-byte scalar stack slot",
    )
    parser.add_argument(
        "--expand-f32-sub",
        action="store_true",
        help="expand compiler e32 vfsub.vf using scalar sign injection and "
        "allowed vfadd.vf, and e32 vmv.s.x scalar seeds through vfmv.s.f; "
        "requires a proved e32 vector configuration",
    )
    parser.add_argument(
        "--expand-vector-merges",
        action="store_true",
        help="expand signed-five-bit vmerge.vim using allowed vmerge.vxm; "
        "preserve t6 with a temporary 16-byte scalar stack slot",
    )
    parser.add_argument(
        "--order-vector-copies",
        action="store_true",
        help="move an independent counter increment after a matching byte "
        "vector store to avoid the NR load/store scheduling stall",
    )
    parser.add_argument(
        "--order-vector-adds",
        action="store_true",
        help="move the independent counter in the verified e32/16/m4 "
        "integer-add loop after its vector store's trailing fence",
    )
    parser.add_argument(
        "--no-fences",
        action="store_true",
        help="do not insert fence rw,rw around AME and vector memory "
        "instructions; the instruction, RVV and AME checks still run",
    )
    parser.add_argument(
        "--coalesce-fences",
        action="store_true",
        help="share only adjacent identical fence rw,rw; "
        "default retains separate pairs",
    )
    args = parser.parse_args()
    try:
        output = transform(
            sys.stdin.read(),
            coalesce_fences=args.coalesce_fences,
            expand_f32_sub=args.expand_f32_sub,
            expand_vector_merges=args.expand_vector_merges,
            expand_f32_scalar_moves=args.expand_f32_scalar_moves,
            order_vector_copies=args.order_vector_copies,
            order_vector_adds=args.order_vector_adds,
            insert_fences=not args.no_fences,
            scalarize_f32_ordered_reductions=not args.allow_vector_ordered_reductions,
            reject_vector_ordered_reductions=args.reject_vector_ordered_reductions,
            reject_vector_max_reductions=args.reject_vector_max_reductions,
        )
    except ValueError as error:
        print(f"restrict_fpga_assembly: {error}", file=sys.stderr)
        return 1
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
