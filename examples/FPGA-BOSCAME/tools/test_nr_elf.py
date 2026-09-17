"""Check the final ELF audit at raw instruction and section boundaries."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import check_nr_elf
from nr_isa import FENCE_RW_RW, parse_disassembly


def sections_fixture(sections):
    """Build ELF64 directly, so tests need no cross compiler or objdump.

    Each section is (name, type, flags, address, bytes). The auditor must use
    the section flags and raw bytes even when disassembly marks code as data.
    """
    names = bytearray(b'\0')
    offsets = {}
    for name in [*(section[0] for section in sections), '.shstrtab']:
        offsets[name] = len(names)
        names.extend(name.encode() + b'\0')
    contents = bytearray(64)
    contents[:6] = b'\x7fELF\x02\x01'
    struct.pack_into('<H', contents, 18, 243)
    headers = [(0,) * 10]
    for name, kind, flags, address, data in sections:
        start = len(contents)
        if kind != 8:  # SHT_NOBITS has virtual extent but no file contents.
            contents.extend(data)
        headers.append((offsets[name], kind, flags, address, start, len(data),
                        0, 0, 2, 0))
    name_offset = len(contents)
    contents.extend(names)
    section_offset = len(contents)
    headers.append((offsets['.shstrtab'], 3, 0, 0, name_offset, len(names),
                    0, 0, 1, 0))
    struct.pack_into('<Q', contents, 40, section_offset)
    struct.pack_into('<HHH', contents, 58, 64, len(headers), len(headers)-1)
    for header in headers:
        contents.extend(struct.pack('<IIQQQQIIQQ', *header))
    return bytes(contents)


def words_bytes(*words):
    return b''.join(struct.pack('<I', word) for word in words)


def fixture(words, data_word=None, *, code=None):
    code = words_bytes(*words) if code is None else code
    sections = [('.text', 1, 6, 0x80000000, code)]
    if data_word is not None:
        sections.append(('.rodata', 1, 2, 0x81000000, words_bytes(data_word)))
    return sections_fixture(sections)


class NrElfTests(unittest.TestCase):
    def test_csr_is_detected_without_objdump_annotation(self):
        for word in (0xc2202373, 0xc2002373, 0xc2102373):
            report = check_nr_elf.audit(fixture([word]), '')
            self.assertEqual(report['status'], 'FAIL')
            self.assertIn('vector CSR', report['errors'][0])

    def test_raw_ame_is_checked_without_decoder_and_data_is_ignored(self):
        report = check_nr_elf.audit(fixture([FENCE_RW_RW, 0x04b50077, FENCE_RW_RW], 0xc2202373), '')
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['ame_instructions'], 1)
        report = check_nr_elf.audit(fixture([FENCE_RW_RW, 0x08b508f7, FENCE_RW_RW]), '')
        self.assertEqual(report['status'], 'FAIL')

    def test_fence_is_required_on_both_sides(self):
        for words in ([0x04b50077, FENCE_RW_RW], [FENCE_RW_RW, 0x04b50077],
                      [FENCE_RW_RW, 0x13, 0x04b50077, FENCE_RW_RW]):
            self.assertEqual(check_nr_elf.audit(fixture(words), '')['status'], 'FAIL')

    def test_unknown_vector_and_whole_move_are_rejected(self):
        word = 0x9e8034d7  # vmv1r.v v9, v8
        dump = 'Disassembly of section .text:\n80000000: 9e8034d7 vmv1r.v v9, v8\n'
        for source in ('', dump):
            self.assertEqual(check_nr_elf.audit(fixture([word]), source)['status'], 'FAIL')

    def test_normal_vector_memory_requires_decoder_and_fences(self):
        words = [FENCE_RW_RW, 0x0205e407, FENCE_RW_RW]
        dump = 'Disassembly of section .text:\n80000004: 0205e407 vle32.v v8, (a1)\n'
        report = check_nr_elf.audit(fixture(words), dump)
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['vector_memory_instructions'], 1)
        self.assertEqual(check_nr_elf.audit(fixture(words), '')['status'], 'FAIL')

    def test_byte_pair_disassembly_keeps_ame_words(self):
        dump = 'Disassembly of section .text:\n80000004: 77 00 b5 04  \t.word\t0x04b50077\n'
        record, = parse_disassembly(dump)
        self.assertEqual(record['word'], 0x04b50077)
        self.assertEqual(record['size'], 4)

    def test_c_aggregate_initializer_vector_is_rejected(self):
        # Actual LLVM-generated descriptor initialization from the failing
        # Triton tail launcher; these instructions bypassed kernel.s auditing.
        for word, instruction in (
            (0x42056457, 'vmv.s.x v8, a0'),
            (0x4a81a4d7, 'vsext.vf8 v9, v8'),
        ):
            dump = (f'Disassembly of section .text.launch:\n'
                    f'80001000: {word:08x} {instruction}\n')
            elf = sections_fixture([
                ('.text', 1, 6, 0x80000000, words_bytes(0x13)),
                ('.text.launch', 1, 6, 0x80001000, words_bytes(word)),
            ])
            with self.subTest(instruction=instruction):
                report = check_nr_elf.audit(elf, dump)
                self.assertEqual(report['status'], 'FAIL')
                self.assertTrue(any('.text.launch@0x80001000: unsupported RVV'
                                    in error for error in report['errors']))

    def test_all_vector_csr_forms_are_rejected_from_raw_bytes(self):
        # Register and immediate CSR instructions must both be caught, even
        # without mnemonic annotations or when dumped as mapping-symbol data.
        for csr in (0xc20, 0xc21, 0xc22):
            for funct3 in (1, 2, 3, 5, 6, 7):
                word = (csr << 20) | (funct3 << 12) | (10 << 7) | 0x73
                dump = ('Disassembly of section .text:\n80000000: ' +
                        words_bytes(word).hex(' ') + f' .word {word:#x}\n')
                with self.subTest(csr=hex(csr), funct3=funct3):
                    report = check_nr_elf.audit(fixture([word]), dump)
                    self.assertTrue(any('vector CSR' in error
                                        for error in report['errors']))
        # Scalar cycle reads are required by launchers and remain legal.
        self.assertEqual(check_nr_elf.audit(fixture([0xc0002573]), '')['status'], 'PASS')

    def test_whole_register_memory_is_rejected_even_when_fenced(self):
        for word, instruction in ((0x02856407, 'vl1re32.v v8, (a0)'),
                                  (0x02850427, 'vs1r.v v8, (a0)')):
            dump = ('Disassembly of section .text:\n'
                    f'80000004: {word:08x} {instruction}\n')
            with self.subTest(instruction=instruction):
                report = check_nr_elf.audit(fixture([FENCE_RW_RW, word, FENCE_RW_RW]), dump)
                self.assertEqual(report['status'], 'FAIL')
                self.assertEqual(report['vector_memory_instructions'], 1)
                self.assertTrue(any('unsupported RVV' in error for error in report['errors']))
                self.assertFalse(any('missing adjacent' in error for error in report['errors']))

    def test_executable_word_mapping_cannot_hide_vector_instruction(self):
        for word in (0x9e8034d7, 0x02856407, 0x02056407):
            dump = ('Disassembly of section .text:\n80000004: ' +
                    words_bytes(word).hex(' ') + f' .word {word:#x}\n')
            with self.subTest(word=hex(word)):
                report = check_nr_elf.audit(fixture([FENCE_RW_RW, word, FENCE_RW_RW]), dump)
                self.assertEqual(report['status'], 'FAIL')
                self.assertTrue(any('unsupported RVV' in error for error in report['errors']))

    def test_decoder_word_mismatch_cannot_authorize_vector(self):
        # A record for a different word/address/section must not lend its
        # allowlisted mnemonic to raw bytes that the decoder did not identify.
        elf = fixture([FENCE_RW_RW, 0x02856407, FENCE_RW_RW])
        for section, address, word in (('.text', 0x80000004, 0x02056407),
                                       ('.text', 0x80000008, 0x02856407),
                                       ('.other', 0x80000004, 0x02856407)):
            dump = (f'Disassembly of section {section}:\n'
                    f'{address:x}: {word:08x} vle32.v v8, (a0)\n')
            with self.subTest(section=section, address=address, word=word):
                self.assertEqual(check_nr_elf.audit(elf, dump)['status'], 'FAIL')

    def test_fence_bytes_straddling_instructions_are_not_a_fence(self):
        # The last two bytes of a 32-bit scalar instruction and the following
        # compressed instruction equal 0x0330000f, but neither is a fence.
        fake_predecessor = words_bytes(0x000f0013) + struct.pack('<H', 0x0330)
        self.assertEqual(fake_predecessor[-4:], words_bytes(FENCE_RW_RW))
        for word, annotation in ((0x04b50077, '.word 0x04b50077'),
                                  (0x02056407, 'vle32.v v8, (a0)')):
            code = fake_predecessor + words_bytes(word, FENCE_RW_RW)
            dump = f'Disassembly of section .text:\n80000006: {word:08x} {annotation}\n'
            with self.subTest(word=hex(word)):
                report = check_nr_elf.audit(fixture([], code=code), dump)
                self.assertEqual(report['status'], 'FAIL')
                self.assertTrue(any('missing adjacent fence' in error for error in report['errors']))

    def test_real_fences_after_compressed_instructions_are_accepted(self):
        code = struct.pack('<H', 0x0001) + words_bytes(FENCE_RW_RW, 0x04b50077, FENCE_RW_RW)
        report = check_nr_elf.audit(fixture([], code=code), '')
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['instructions'], 4)
        # A compressed NOP between fence and AME breaks adjacency.
        broken = words_bytes(FENCE_RW_RW) + struct.pack('<H', 0x0001) + words_bytes(0x04b50077, FENCE_RW_RW)
        self.assertEqual(check_nr_elf.audit(fixture([], code=broken), '')['status'], 'FAIL')

    def test_executable_noload_and_nonexecuting_data_are_not_code(self):
        data = sections_fixture([
            ('.text', 1, 6, 0x80000000, words_bytes(0x13)),
            ('.nr_mailbox', 8, 6, 0x80010000, b'\x00' * 4096),
            ('.rodata', 1, 2, 0x81000000, words_bytes(0xc2202573, 0x9e8034d7)),
        ])
        report = check_nr_elf.audit(data, '')
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['instructions'], 1)
        self.assertEqual([section['name'] for section in report['executable_sections']], ['.text'])

    def test_sections_do_not_share_fence_or_decoder_state(self):
        elf = sections_fixture([
            ('.text', 1, 6, 0x80000000, words_bytes(FENCE_RW_RW)),
            ('.text.launch', 1, 6, 0x80000004, words_bytes(0x04b50077, FENCE_RW_RW)),
        ])
        report = check_nr_elf.audit(elf, '')
        self.assertEqual(report['status'], 'FAIL')
        self.assertTrue(any('.text.launch' in error and 'missing adjacent' in error
                            for error in report['errors']))

    def test_truncated_and_unsupported_instruction_widths_fail_closed(self):
        for code in (b'\x01', b'\x13\x00', b'\x1f\x00\x00\x00'):
            with self.subTest(code=code):
                self.assertEqual(check_nr_elf.audit(fixture([], code=code), '')['status'], 'FAIL')
        for data in (b'', b'\x7fELF\x01\x01', fixture([0x13])[:-1]):
            with self.subTest(elf_size=len(data)), self.assertRaises(ValueError):
                check_nr_elf.audit(data, '')

    def test_cli_writes_failure_report_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            elf = Path(directory) / 'case.elf'
            output = Path(directory) / 'audit.json'
            elf.write_bytes(fixture([0xc2202573]))
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch('sys.argv', ['check_nr_elf.py', str(elf), '--output', str(output)]), \
                    patch.object(check_nr_elf.subprocess, 'check_output', return_value=''), \
                    redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(check_nr_elf.main(), 1)
            self.assertEqual(json.loads(output.read_text())['status'], 'FAIL')
            self.assertIn('NR ELF audit FAIL', stdout.getvalue())
            self.assertIn('unsupported vector CSR', stderr.getvalue())


if __name__ == '__main__':
    unittest.main()
