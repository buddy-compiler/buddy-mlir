#!/usr/bin/env python3
"""Audit the complete linked NR ELF, including C launch and runtime code.

Walk every SHF_EXECINSTR byte range from the ELF section table. LLVM objdump
supplies RVV mnemonic names, but raw ELF bytes determine CSR/AME detection and
fence adjacency, so data-mapped .word encodings cannot bypass the audit.
This checks the NR AME/RVV contract, not arbitrary scalar ISA compatibility.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shlex
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nr_isa import (FENCE_RW_RW, RVV_ALLOWED, is_ame, is_vector,
                    is_vector_memory, parse_disassembly, validate_ame_word)


def executable_sections(data):
    if data[:6] != b'\x7fELF\x02\x01':
        raise ValueError('expected little-endian ELF64')
    if len(data) < 64 or struct.unpack_from('<H', data, 18)[0] != 243:
        raise ValueError('expected RISC-V ELF')
    offset = struct.unpack_from('<Q', data, 40)[0]
    size, count, strings_index = struct.unpack_from('<HHH', data, 58)
    if size != 64 or not count or strings_index >= count or offset + count * size > len(data):
        raise ValueError('invalid or unsupported ELF section table')
    headers = [struct.unpack_from('<IIQQQQIIQQ', data, offset + i * size)
               for i in range(count)]
    strings_header = headers[strings_index]
    strings = data[strings_header[4]:strings_header[4] + strings_header[5]]
    result = []
    for name, kind, flags, address, start, length, *_ in headers:
        # NOLOAD state (notably nr_mailbox) may inherit AX flags from the
        # preceding linker-script section but contains no instruction bytes.
        if not flags & 4 or not length or kind == 8:
            continue
        if start + length > len(data) or name >= len(strings):
            raise ValueError('invalid executable section')
        label = strings[name:].split(b'\0', 1)[0].decode('utf-8', errors='strict')
        result.append((label, address, data[start:start + length]))
    if not result:
        raise ValueError('ELF has no executable sections')
    return result


def audit(data, dump):
    decoded = {(r['section'], r['address']): r for r in parse_disassembly(dump)}
    errors, sections = [], []
    vectors = Counter()
    instruction_count = ame_count = vector_memory_count = 0
    for name, address, code in executable_sections(data):
        sections.append({'name': name, 'address': hex(address), 'bytes': len(code)})
        offset = 0
        previous_instruction = None
        while offset < len(code):
            if offset + 2 > len(code):
                errors.append(f'{name}+{offset:#x}: incomplete instruction')
                break
            half = int.from_bytes(code[offset:offset+2], 'little')
            width = 4 if half & 3 == 3 else 2
            where = f'{name}@{address + offset:#x}'
            if width == 4 and (half & 31 == 31 or offset + 4 > len(code)):
                errors.append(f'{where}: unsupported instruction width')
                break
            word = int.from_bytes(code[offset:offset+width], 'little')
            instruction_count += 1
            entry = decoded.get((name, address + offset))
            detail = entry['instruction'] if entry else f'raw {word:#010x}'
            if entry and entry['word'] != word:
                # ObjDump may coalesce scalar data/padding. AME/RVV must still
                # have exact 32-bit records to receive mnemonic validation.
                entry = None
            if width == 4:
                if word & 0x7f == 0x73 and (word >> 12) & 7 and word >> 20 in (0xc20, 0xc21, 0xc22):
                    errors.append(f'{where}: unsupported vector CSR: {detail}')
                fenced = False
                if is_ame(word):
                    ame_count += 1
                    fenced = True
                    try:
                        validate_ame_word(word)
                    except ValueError as error:
                        errors.append(f'{where}: {error}')
                if is_vector(word):
                    mnemonic = entry['mnemonic'] if entry else '<undecoded>'
                    vectors[mnemonic] += 1
                    if mnemonic not in RVV_ALLOWED:
                        errors.append(f'{where}: unsupported RVV: {detail}')
                    if is_vector_memory(word):
                        vector_memory_count += 1
                        fenced = True
                if fenced:
                    after = int.from_bytes(code[offset+4:offset+8], 'little')
                    # The four preceding bytes may straddle a 32-bit scalar
                    # instruction and a compressed instruction. Require an
                    # actual decoded predecessor, not a coincidental byte
                    # sequence equal to the fence encoding.
                    if previous_instruction != (4, FENCE_RW_RW) or after != FENCE_RW_RW:
                        errors.append(f'{where}: missing adjacent fence rw,rw: {detail}')
            previous_instruction = (width, word)
            offset += width
    return {'status': 'FAIL' if errors else 'PASS',
            'elf_sha256': hashlib.sha256(data).hexdigest(),
            'executable_sections': sections, 'instructions': instruction_count,
            'ame_instructions': ame_count, 'vector_memory_instructions': vector_memory_count,
            'rvv_instructions': dict(sorted(vectors.items())), 'errors': errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('elf', type=Path)
    parser.add_argument('--objdump', default='llvm-objdump',
                        help='shell-quoted llvm-objdump command, optionally with flags')
    parser.add_argument('--output', type=Path, help='write machine-readable audit report')
    args = parser.parse_args()
    try:
        dump = subprocess.check_output([*shlex.split(args.objdump), '-d',
                                        '--mattr=+v,+zicbom', str(args.elf)], text=True)
        report = audit(args.elf.read_bytes(), dump)
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(f'NR ELF audit {report["status"]}: {report["instructions"]} instructions, '
              f'{report["ame_instructions"]} AME, '
              f'{sum(report["rvv_instructions"].values())} RVV; {args.elf}')
        for error in report['errors'][:30]:
            print(error, file=sys.stderr)
        if len(report['errors']) > 30:
            print(f'... {len(report["errors"])-30} additional errors in report', file=sys.stderr)
        return int(bool(report['errors']))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'NR ELF audit failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
