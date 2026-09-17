#!/usr/bin/env python3
"""Validate NR AME/RVV forms and fence AME/vector memory instructions.

Adapted from ModelZoo examples/buddy-qwen35-fpga/python/qwen35/compiler/
restrict_fpga_assembly.py, commit 8815b74fb6d3cd6288c4d99ac6fd7c5d041a7cc3:
https://gitlink.org.cn/michaelcjl/ModelZoo.git

Run AFTER ame_to_word.py. NR rejects transposed B loads, unverified AME
encodings, vector CSR reads/spills and raw RVV encodings. Data sections are
preserved exactly: an AME-looking floating-point constant is not an instruction.
This is a checked compiler-output filter, not a general-purpose assembler.
Opt-in --coalesce-fences shares an adjacent identical fence between instructions;
every AME/vector memory instruction still has both adjacent fences. Labels,
directives, and all other instructions prevent sharing across that boundary.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nr_isa import (RVV_ALLOWED, VECTOR_MEMORY, is_ame, is_vector,
                    validate_ame_word)


def _raw_words(code):
    match = re.fullmatch(r"\.(?:word|4byte|long)\s+(.+)", code)
    if not match:
        return None
    words = []
    for expression in match[1].split(","):
        expression = expression.strip()
        if not re.fullmatch(r"-?(?:0x[0-9a-fA-F]+|[0-9]+)", expression):
            raise ValueError("executable raw words must be integer literals")
        value = int(expression, 0) if "x" in expression.lower() else int(expression)
        if not -(1 << 31) <= value <= 0xFFFFFFFF:
            raise ValueError("raw instruction does not fit in 32 bits")
        words.append(value & 0xFFFFFFFF)
    return words


def transform(source, coalesce_fences=False):
    output = []
    executable = True
    previous = True
    sections = []
    last_fence = False

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
        same_fence = bool(executable and not (separator and source_comment.strip())
                          and re.fullmatch(r"fence\s+rw\s*,\s*rw", code))
        if coalesce_fences and same_fence and last_fence:
            return
        output.append(line)
        # A label must break sharing even though it occupies no instruction
        # bytes: a branch to it must execute the following operation's fence.
        # Unknown directives are likewise conservative barriers, including
        # alignment, section transitions and assembler metadata.
        last_fence = same_fence

    for number, original_line in enumerate(source.splitlines(), 1):
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
            if mnemonic in (".text", ".data", ".bss", ".rodata", ".sdata", ".sbss"):
                previous, executable = executable, mnemonic == ".text"
            elif mnemonic in (".section", ".pushsection"):
                if mnemonic == ".pushsection":
                    sections.append(executable)
                fields = code.split(None, 1)[1].split(",")
                name = fields[0].strip().strip('"')
                flags = fields[1].strip().strip('"') if len(fields) > 1 else ""
                previous, executable = executable, (
                    "x" in flags or name == ".text" or name.startswith(".text."))
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
                raise ValueError("multiple assembly statements must be on separate lines")
            if mnemonic.startswith("v") and mnemonic not in RVV_ALLOWED:
                raise ValueError(f"RVV instruction not allowed: {code}")
            if re.search(r"\b(?:vlenb|vtype|vl)\b", code) and mnemonic.startswith("csr"):
                raise ValueError(f"vector CSR access not allowed: {code}")
            # Numeric vector CSRs are equivalent to their names.
            if mnemonic.startswith("csr"):
                operands = code.split(None, 1)[1] if " " in code or "\t" in code else ""
                if re.search(r"(?:^|[,\s])(?:0x[cC]2[012]|310[456])(?:$|[,\s])", operands):
                    raise ValueError(f"vector CSR access not allowed: {code}")
            if mnemonic == ".insn":
                raise ValueError(".insn bypasses NR instruction checks; use encoder mnemonics")
            if (mnemonic.startswith(("mset", "ml", "ms", "mq", "mf", "mw")) or
                    ".mm" in mnemonic):
                raise ValueError("unencoded AME mnemonic; run ame_to_word.py first")
            words = _raw_words(code)
            if words is not None:
                # Multiple words are split so every AME word gets its own pair
                # of fences, even when emitted on the same original line.
                for word in words:
                    if is_vector(word):
                        raise ValueError("raw RVV encoding bypasses allowlist")
                    if (word & 0x7F == 0x73 and (word >> 12) & 7 and
                            word >> 20 in (0xC20, 0xC21, 0xC22)):
                        raise ValueError("raw vector CSR access bypasses allowlist")
                    if is_ame(word):
                        validate_ame_word(word)
                        emit("\tfence\trw, rw")
                    emit(line if len(words) == 1 else f"\t.word 0x{word:08x}")
                    if is_ame(word):
                        emit("\tfence\trw, rw")
                continue
            if mnemonic in VECTOR_MEMORY:
                emit("\tfence\trw, rw")
            emit(line)
            if mnemonic in VECTOR_MEMORY:
                emit("\tfence\trw, rw")
        except (ValueError, IndexError) as error:
            raise ValueError(f"line {number}: {error}") from error
    if sections:
        raise ValueError("unclosed .pushsection")
    return "\n".join(output) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coalesce-fences", action="store_true",
                        help="share only adjacent identical fence rw,rw; "
                             "default retains separate pairs")
    args = parser.parse_args()
    try:
        output = transform(sys.stdin.read(), coalesce_fences=args.coalesce_fences)
    except ValueError as error:
        print(f"restrict_fpga_assembly: {error}", file=sys.stderr)
        return 1
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
