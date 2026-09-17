"""Strict intermediate trace acceptance and malformed/tampered evidence checks."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parents[1] / 'tools'
spec = importlib.util.spec_from_file_location('check_intermediates', TOOLS / 'check_intermediates.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def make_manifest():
    entries, cursor = [], 0
    for index in range(92):
        graph = 'prefill' if index < 46 else 'decode'
        count = 1 if graph == 'prefill' else 8
        suffix = 'model.layers.0_boundary_' + str(index % 46)
        entry = dict(index=index, graph=graph, layer=0, symbol=f'_mlir_ciface_qwen_graph_boundary_{index}',
                     operand=0, phase='after', reference_suffix=suffix, transform='identity',
                     shape=[1, 2], dtype='f32', rank=2, elements=2,
                     offsets=[cursor + 2 * i for i in range(count)],
                     reference_keys=[('prefill' if graph == 'prefill' else f'decode_{i}') + '_' + suffix
                                     for i in range(count)])
        entries.append(entry)
        cursor += 2 * count
    return dict(schema_version=1, layers=1, prefill_len=16, decode_steps=8,
                max_abs_tolerance=.001, mean_abs_tolerance=.0001,
                entries=entries, reference_bytes=cursor * 4, uncovered_reference_tensors=['uncovered'],
                generated_source_sha256='a' * 64, reference_blob_sha256='b' * 64,
                source_sha256={'arrays.npz': 'c' * 64},
                linker_flags=['--wrap=' + n for n in sorted(e['symbol'] for e in entries)])


def bits(value):
    return f'{struct.unpack("<I", struct.pack("<f", value))[0]:08x}'


def make_log(manifest):
    lines = ['[nr] launch model', '[model] diagnostic unrelated to acceptance']
    for position in checker.POSITIONS:
        graph = 'prefill' if position == 0 else 'decode'
        for entry in manifest['entries']:
            if entry['graph'] == graph:
                lines.append(f'[intermediate] position={position:08x} entry={entry["index"]:08x}'
                             f' calls=00000001 checked=00000001 count={entry["elements"]:08x}'
                             f' max_abs_bits={bits(.0005)} mean_abs_bits={bits(.00005)}')
        lines.append(f'[intermediate] complete position={position:08x} PASS')
        lines.append('[profile] unrelated timing handled by a separate checker')
    lines += ['[nr] RA returned: PASS', '[nr] launch model status=0x0']
    return lines


class IntermediateCheckerTests(unittest.TestCase):
    def setUp(self):
        self.manifest = make_manifest()
        self.lines = make_log(self.manifest)

    def assert_rejected(self, lines, text=None):
        report = checker.check('\r\n'.join(lines) + '\r\n', self.manifest)
        self.assertEqual(report['status'], 'NOT_ACCEPTED')
        self.assertTrue(report['errors'])
        if text:
            self.assertIn(text, '\n'.join(report['errors']))

    def test_accepts_exact_414_full_comparisons_and_reference_identity(self):
        result = checker.check('\r\n'.join(self.lines), self.manifest)
        self.assertEqual(result['status'], 'INTERMEDIATES_PASS', result['errors'])
        self.assertEqual(result['comparisons'], 414)
        self.assertEqual([r['position'] for r in result['completions']], [0, *range(16, 24)])
        self.assertEqual(result['records'][46]['reference_key'], 'decode_0_model.layers.0_boundary_0')
        self.assertEqual(result['records'][-1]['reference_offset'], self.manifest['entries'][-1]['offsets'][-1])
        self.assertIn('Partial', result['scope'])
        self.assertEqual(result['uncovered_reference_tensors'], ['uncovered'])

    def test_omission_duplicate_and_reordered_rows_cannot_pass(self):
        for mode in ('missing', 'duplicate', 'reversed'):
            with self.subTest(mode=mode):
                lines = self.lines[:]
                if mode == 'missing':
                    del lines[2]
                elif mode == 'duplicate':
                    lines.insert(2, lines[2])
                else:
                    lines[2], lines[3] = lines[3], lines[2]
                self.assert_rejected(lines, 'out-of-order')

    def test_wrong_position_kind_index_extent_or_call_counts(self):
        mutations = [('position=00000000', 'position=0000000f'),
                     ('entry=00000000', 'entry=0000002e'),
                     ('entry=00000000', 'entry=ffffffff'),
                     ('count=00000002', 'count=00000001'),
                     ('calls=00000001', 'calls=00000000'),
                     ('calls=00000001', 'calls=00000002'),
                     ('checked=00000001', 'checked=00000000'),
                     ('checked=00000001', 'checked=00000002')]
        for old, new in mutations:
            with self.subTest(new=new):
                lines = self.lines[:]
                lines[2] = lines[2].replace(old, new)
                self.assert_rejected(lines)

    def test_nonfinite_negative_and_excess_errors_reject_even_complete_pass(self):
        for field in ('max', 'mean'):
            for value in (float('nan'), float('inf'), -float('inf'), -1., .01):
                with self.subTest(field=field, value=value):
                    lines = self.lines[:]
                    old = bits(.0005 if field == 'max' else .00005)
                    lines[2] = lines[2].replace(f'{field}_abs_bits={old}', f'{field}_abs_bits={bits(value)}')
                    self.assert_rejected(lines, f'{field} error')
        lines = self.lines[:]
        lines[2] = lines[2].replace('max_abs_bits=' + bits(.0005), 'max_abs_bits=' + bits(.00001))
        self.assert_rejected(lines, 'mean error exceeds maximum')

    def test_malformed_rows_and_completions(self):
        for malformed in (self.lines[2] + ' trailing', self.lines[2][1:],
                          self.lines[2].replace('calls=', 'calls=G'),
                          self.lines[2].replace('checked=00000001 ', ''),
                          self.lines[2].replace('count=', 'elements=')):
            with self.subTest(malformed=malformed):
                lines = self.lines[:]
                lines[2] = malformed
                self.assert_rejected(lines)
        for extra in ('[intermediate complete position=00000000 PASS',
                      'intermediate] entry=00000000', '[nr] RA returned PASS'):
            with self.subTest(extra=extra):
                self.assert_rejected(self.lines + [extra], 'malformed')
        complete = next(i for i, line in enumerate(self.lines) if '[intermediate] complete' in line)
        for mode in ('missing', 'duplicate', 'wrong_position', 'fail', 'malformed', 'early'):
            with self.subTest(mode=mode):
                lines = self.lines[:]
                if mode == 'missing':
                    del lines[complete]
                elif mode == 'duplicate':
                    lines.insert(complete, lines[complete])
                elif mode == 'wrong_position':
                    lines[complete] = lines[complete].replace('00000000', '00000010')
                elif mode == 'fail':
                    lines[complete] = lines[complete].replace('PASS', 'FAIL')
                elif mode == 'malformed':
                    lines[complete] += ' junk'
                else:
                    lines[2], lines[complete] = lines[complete], lines[2]
                self.assert_rejected(lines)

    def test_missing_duplicate_early_or_malformed_return_and_traps(self):
        for mode in ('missing', 'duplicate', 'early', 'malformed', 'fail', 'trap', 'launch', 'post_return_record'):
            with self.subTest(mode=mode):
                lines = self.lines[:]
                if mode == 'missing':
                    lines.remove('[nr] RA returned: PASS')
                elif mode == 'duplicate':
                    lines.append('[nr] RA returned: PASS')
                elif mode == 'early':
                    lines.remove('[nr] RA returned: PASS')
                    lines.insert(0, '[nr] RA returned: PASS')
                elif mode == 'malformed':
                    lines[-2] += ' suffix'
                elif mode == 'fail':
                    lines[-2] = '[nr] RA returned: FAIL'
                elif mode == 'trap':
                    lines.append('[nr] TRAP mcause=0x2')
                elif mode == 'launch':
                    lines[-1] = '[nr] launch model status=0x1'
                else:
                    lines.append(lines[2])
                self.assert_rejected(lines)

    def test_manifest_dimensions_order_mapping_hashes_and_offsets_must_be_well_formed(self):
        mutations = [lambda m: m.update(layers=2), lambda m: m.update(prefill_len=15),
                     lambda m: m.update(decode_steps=7), lambda m: m['entries'].pop(),
                     lambda m: m['entries'][1].update(index=0), lambda m: m['entries'][0].update(graph='decode'),
                     lambda m: m['entries'][0].update(rank=1), lambda m: m['entries'][0].update(elements=1),
                     lambda m: m['entries'][0].update(dtype='f64'), lambda m: m['entries'][0].update(shape=[0, 2]),
                     lambda m: m['entries'][0].update(transform='transpose_maybe'),
                     lambda m: m['entries'][0].update(offsets=[1]),
                     lambda m: m['entries'][-1]['offsets'].pop(),
                     lambda m: m['entries'][1].update(symbol=m['entries'][0]['symbol']),
                     lambda m: m['entries'][-1]['reference_keys'].__setitem__(7, 'decode_6_wrong'),
                     lambda m: m.update(reference_bytes=m['reference_bytes'] + 4),
                     lambda m: m.update(max_abs_tolerance=float('nan')),
                     lambda m: m.update(mean_abs_tolerance=-1), lambda m: m.update(reference_blob_sha256='bad'),
                     lambda m: m.update(source_sha256={}), lambda m: m['linker_flags'].pop()]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                manifest = copy.deepcopy(self.manifest)
                mutate(manifest)
                with self.assertRaises(ValueError):
                    checker.check('\n'.join(self.lines), manifest)

    def test_cli_writes_machine_readable_failure_and_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manifest, uart, output = (path / name for name in ('manifest.json', 'uart.log', 'verification.json'))
            manifest.write_text(json.dumps(self.manifest))
            command = [sys.executable, str(TOOLS / 'check_intermediates.py'), '--uart', str(uart),
                       '--manifest', str(manifest), '--output', str(output)]
            uart.write_text('\n'.join(self.lines))
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(output.read_text())['status'], 'INTERMEDIATES_PASS')
            uart.write_text('\n'.join(self.lines).replace('max_abs_bits=' + bits(.0005), 'max_abs_bits=7fc00000', 1))
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            report = json.loads(output.read_text())
            self.assertIsNone(report['records'][0]['max_abs'])
            self.assertEqual(report['status'], 'NOT_ACCEPTED')
            self.assertEqual(len(report['inputs_sha256']), 2)
            manifest.write_text('{')
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(json.loads(output.read_text())['status'], 'NOT_ACCEPTED')


if __name__ == '__main__':
    unittest.main()
