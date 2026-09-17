"""NR assembly regression checks; run with python3 -m unittest discover here."""

import os
from pathlib import Path
import re
import subprocess
import sys
import unittest
from unittest.mock import patch

import ame_to_word as encoder
import restrict_fpga_assembly as restriction
from nr_isa import validate_ame_word


def execute_wrapper(source, registers):
    """Evaluate scalar wrapper instructions and record the AME operand values.

    AME itself is mocked only at its architectural interface: memory ops read
    the two encoded GPRs; configuration writes a distinct return sentinel.
    """
    regs = dict(registers)
    memory = {}
    observed = []
    for line in source.splitlines():
        code = line.split('#', 1)[0].strip()
        if not code or code.startswith('fence'):
            continue
        parts = re.split(r'[\s,]+', code)
        if parts[0] in ('sd', 'ld'):
            address = re.fullmatch(r'(-?\d+)\((\w+)\)', parts[2])
            pointer = regs[address[2]] + int(address[1])
            if parts[0] == 'sd': memory[pointer] = regs[parts[1]]
            else: regs[parts[1]] = memory[pointer]
        elif parts[0] == 'mv':
            regs[parts[1]] = regs[parts[2]]
        elif parts[0] == 'addi':
            regs[parts[1]] = regs[parts[2]] + int(parts[3])
        elif parts[0] == '.word':
            word = int(parts[1], 0)
            rs1, rs2 = (word >> 15) & 31, (word >> 20) & 31
            observed.append((word, regs[encoder.ABI_NAMES[rs1]],
                             regs[encoder.ABI_NAMES[rs2]]))
            if (word >> 12) & 7 in (4, 5, 6):
                rd = (word >> 7) & 31
                if rd: regs[encoder.ABI_NAMES[rd]] = 13
        else:
            raise AssertionError(code)
        regs['zero'] = 0
    return regs, observed


class NrAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            key: value for key, value in os.environ.items()
            if key not in encoder.REMOVED_ENVIRONMENT_OPTIONS}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_verified_reference_words(self):
        # Fixed words from ModelZoo's bare_runtime.c, not recomputed encodings.
        for word in (0x00054077, 0x04055877, 0x040543f7, 0x0406e6f7,
                     0x04b50077, 0x08b500f7, 0x005e2077, 0x025e2077,
                     0x28180877):
            with self.subTest(word=hex(word)):
                validate_ame_word(word)
                output = restriction.transform(f'.word {word:#x}')
                self.assertEqual(output.count('fence\trw, rw'), 2)

    def test_fixed_memory_wrappers_preserve_operands_and_registers(self):
        original = {name: 0x10000 + index * 32
                    for index, name in enumerate(encoder.ABI_NAMES)}
        original['zero'] = 0
        for mnemonic, matrix in [('mlae8.m', 'tr3'), ('mlbe8.m', 'tr4'),
                                 ('mlce32.m', 'acc7'), ('msce32.m', 'acc0')]:
            for base in encoder.ABI_NAMES:
                for stride in encoder.ABI_NAMES:
                    assembly, _ = encoder.convert_line(
                        f'{mnemonic} {matrix}, ({base}), {stride}')
                    regs, seen = execute_wrapper(assembly, original)
                    self.assertEqual(regs, original, (mnemonic, base, stride))
                    self.assertEqual(seen[0][1:], (original[base], original[stride]))

    def test_configuration_return_value_survives_restore_and_aliases(self):
        original = {name: 0x10000 + index * 32
                    for index, name in enumerate(encoder.ABI_NAMES)}
        original['zero'] = 0
        for mnemonic in encoder.CONFIG:
            for rd in encoder.ABI_NAMES:
                for rs1 in ('a0', 'a3', 'a6', 't2', 'sp', 'zero'):
                    assembly, _ = encoder.convert_line(f'{mnemonic} {rd}, {rs1}')
                    regs, seen = execute_wrapper(assembly, original)
                    self.assertEqual(seen[0][1], original[rs1])
                    expected = dict(original)
                    if rd != 'zero': expected[rd] = 13
                    self.assertEqual(regs, expected, (mnemonic, rd, rs1))
        self.assertEqual(encoder.convert_line('msettilem x16, x10')[0],
                         encoder.convert_line('msettilem a6, a0')[0].replace(
                             'msettilem a6, a0', 'msettilem x16, x10'))

    def test_reject_unverified_mnemonics_and_banks(self):
        for text in ('mlbte8.m tr1, (a0), a1', '.Lx: mlbte8.m tr1, (a0), a1',
                     'msettypei zero, 8', 'mqma.h.mm acc0, tr0, tr1',
                     'mlce32.m tr0, (a0), a1', 'mlae8.m tr8, (a0), a1',
                     'mqma.b.mm acc0, acc0, tr1'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                encoder.transform(text)

    def test_reject_unsupported_raw_words_and_bypasses(self):
        for text in ('.word 0x08b508f7', '.Lx: .word 0x08b508f7',
                     '.word 0x04b50077, 0x08b508f7', '.word 0x04b51077',
                     '.word 0x2a180877', '.word 0xc2202573', '.word 0x57',
                     '.insn r 0x77, 0, 0, a0, a1, a2',
                     'csrr a0, vlenb', 'csrr a0, 0xc22',
                     'csrr a0, 3106', 'vl1re32.v v0, (a0)',
                     'vs1r.v v0, (a0)', 'csrr a0, vl', 'csrr a0, vtype',
                     'vmv1r.v v0, v1',
                     'nop; .word 0x08b508f7'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                restriction.transform(text)

    def test_multiple_words_receive_individual_fences(self):
        output = restriction.transform('.word 0x04b50077, 0x08b500f7')
        self.assertEqual(output.count('fence\trw, rw'), 4)

    def test_expanded_verified_memory_receives_fences(self):
        for mnemonic in ('vle8.v', 'vse8.v', 'vle32.v', 'vse32.v',
                         'vle64.v', 'vse64.v'):
            with self.subTest(mnemonic=mnemonic):
                output = restriction.transform(f'{mnemonic} v0, (a0)')
                self.assertEqual(output.count('fence\trw, rw'), 2)
        for instruction in ('vfmadd.vv v0, v1, v2',):
            self.assertIn(instruction, restriction.transform(instruction))

    def test_whole_moves_are_rejected_even_with_known_configuration(self):
        for instruction in ('vmv1r.v v9, v8', 'vmv2r.v v8, v10',
                            'vmv4r.v v8, v12', 'vmv8r.v v8, v16'):
            for setup in ('vsetivli zero, 1, e8, m1, ta, ma',
                          'vsetivli zero, 16, e32, m1, ta, ma'):
                with self.subTest(instruction=instruction, setup=setup), self.assertRaises(ValueError):
                    restriction.transform(setup + '\n' + instruction)

    def test_data_is_preserved_and_section_stack_tracks_code(self):
        data = '.rodata\n.Lx: .word 0x08b508f7\n.asciz "semi;colon"\n'
        self.assertEqual(restriction.transform(data), data)
        text = '.text\n.pushsection .rodata\n.word 0x08b508f7\n.popsection\n.word 0x04b50077\n'
        output = restriction.transform(text)
        self.assertEqual(output.count('fence\trw, rw'), 2)
        with self.assertRaises(ValueError):
            restriction.transform('.text\n.pushsection .rodata\n.popsection\n.word 0x08b508f7')

    def test_old_trace_options_rejected(self):
        for option in ('AME_TRACE_STAGES', 'AME_TRACE_ADDRS', 'AME_USE_QWEN_FIXED_REGS'):
            with self.subTest(option=option), self.assertRaises(ValueError):
                encoder.validate_environment({option: '0'})

    def test_cli_errors_do_not_emit_partial_assembly(self):
        for script, source in [('ame_to_word.py', 'msettype zero, a0\nmlbte8.m tr1, (a0), a1'),
                               ('restrict_fpga_assembly.py', '.word 0x04b50077\n.word 0x08b508f7')]:
            result = subprocess.run([sys.executable, str(Path(__file__).with_name(script))],
                                    input=source, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, '')
            self.assertIn('line 2', result.stderr)


if __name__ == '__main__':
    unittest.main()
