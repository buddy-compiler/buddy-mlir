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


def make_selected_manifest(prefill_count=3, decode_count=4):
    manifest = make_manifest()
    manifest.update(schema_version=2, layers=28, selected_layers=[14, 15],
                    scope='selected norm/linear full tensors at external adapter boundaries',
                    limitations=['Attention and unlisted tensor boundaries are outside this probe.'])
    template = manifest['entries'][0]
    entries, cursor = [], 0
    for graph, count in [('prefill', prefill_count), ('decode', decode_count)]:
        steps = 1 if graph == 'prefill' else 8
        for number in range(count):
            entry = copy.deepcopy(template)
            index = len(entries)
            layer = 14 + number % 2
            suffix = f'model.layers.{layer}_boundary_{number}'
            entry.update(index=index, graph=graph, layer=layer,
                         symbol=f'_mlir_ciface_qwen_graph_selected_{index}',
                         reference_suffix=suffix,
                         reference_keys=[('prefill' if graph == 'prefill' else f'decode_{step}') + '_' + suffix
                                         for step in range(steps)],
                         offsets=[cursor + 2 * step for step in range(steps)])
            entries.append(entry)
            cursor += 2 * steps
    manifest.update(entries=entries, reference_bytes=cursor * 4,
                    linker_flags=['--wrap=' + n for n in sorted(e['symbol'] for e in entries)])
    if entries:
        entries[0]['reference_aliases'] = ['model.layers.14_shared_quantization_input']
    return manifest


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


class SelectedLayerIntermediateCheckerTests(unittest.TestCase):
    def setUp(self):
        self.manifest = make_selected_manifest()
        self.lines = make_log(self.manifest)

    def test_variable_counts_and_layers_preserve_reference_identity(self):
        for prefill, decode in [(3, 4), (50, 53)]:
            with self.subTest(prefill=prefill, decode=decode):
                manifest = make_selected_manifest(prefill, decode)
                result = checker.check('\r\n'.join(make_log(manifest)), manifest)
                self.assertEqual(result['status'], 'INTERMEDIATES_PASS', result['errors'])
                self.assertEqual(result['comparisons'], prefill + 8 * decode)
                self.assertEqual(result['expected_comparisons'], prefill + 8 * decode)
                self.assertEqual(result['entries_per_graph'], {'prefill': prefill, 'decode': decode})
                self.assertEqual(result['layers'], 28)
                self.assertEqual(result['selected_layers'], [14, 15])
                self.assertEqual(result['schema_version'], 2)
                self.assertIn('layers [14, 15] of 28', result['scope'])
                self.assertIn(manifest['limitations'][0], result['limits'])
                self.assertEqual(result['records'][prefill]['reference_key'],
                                 'decode_0_model.layers.14_boundary_0')
                self.assertEqual(result['records'][-1]['reference_offset'],
                                 manifest['entries'][-1]['offsets'][-1])
                self.assertEqual(len(result['completions']), 9)

    def test_selected_layer_scope_metadata_and_graph_grouping_reject_mutations(self):
        mutations = [lambda m: m.update(schema_version=True), lambda m: m.update(schema_version=3),
                     lambda m: m.update(coverage_includes_aliases='true'),
                     lambda m: m.update(layers=0), lambda m: m.update(layers=True),
                     lambda m: m.update(selected_layers=[]), lambda m: m.update(selected_layers=[15, 14]),
                     lambda m: m.update(selected_layers=[14, 14, 15]),
                     lambda m: m.update(selected_layers=[14, 28]),
                     lambda m: m.update(selected_layers=[True, 15]),
                     lambda m: m.update(selected_layers=[14, 15, 16]),
                     lambda m: m.update(scope=' '), lambda m: m.pop('scope'),
                     lambda m: m.update(limitations=[]), lambda m: m.update(limitations=['']),
                     lambda m: m.update(limitations='No attention coverage'),
                     lambda m: m.update(entries=[]),
                     lambda m: m['entries'][0].update(layer=13),
                     lambda m: m['entries'][0].update(layer=True),
                     lambda m: m['entries'][0].update(graph='unknown'),
                     lambda m: m['entries'][0].update(graph=['prefill']),
                     lambda m: m['entries'][0].update(graph='decode'),
                     lambda m: m['entries'][3].update(graph='prefill'),
                     lambda m: m['entries'][4].update(graph='prefill'),
                     lambda m: m['entries'][1].update(layer=14),
                     lambda m: [m['entries'][i].update(layer=14) for i in (4, 6)],
                     lambda m: m.update(prefill_len=15), lambda m: m.update(decode_steps=7)]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                manifest = copy.deepcopy(self.manifest)
                mutate(manifest)
                with self.assertRaises(ValueError):
                    checker.validate_manifest(manifest)
        for prefill, decode in [(0, 4), (3, 0)]:
            with self.subTest(prefill=prefill, decode=decode), self.assertRaisesRegex(ValueError, 'missing.*graph'):
                checker.validate_manifest(make_selected_manifest(prefill, decode))

    def test_common_abi_layout_hash_and_packing_checks_remain_strict(self):
        mutations = [lambda m: m['entries'][1].update(index=0),
                     lambda m: m['entries'][1].update(symbol=m['entries'][0]['symbol']),
                     lambda m: m['entries'][0].update(symbol='raw_symbol'),
                     lambda m: m['entries'][0].update(operand=-1),
                     lambda m: m['entries'][0].update(phase='during'),
                     lambda m: m['entries'][0].update(rank=1),
                     lambda m: m['entries'][0].update(elements=1),
                     lambda m: m['entries'][0].update(dtype='i32'),
                     lambda m: m['entries'][0].update(shape=[0, 2]),
                     lambda m: m['entries'][0].update(transform='unverified_transpose'),
                     lambda m: m['entries'][0].update(reference_suffix=''),
                     lambda m: m['entries'][0].update(offsets=[1]),
                     lambda m: m['entries'][-1]['offsets'].pop(),
                     lambda m: m['entries'][-1]['reference_keys'].__setitem__(7, 'decode_6_wrong'),
                     lambda m: m.update(reference_bytes=m['reference_bytes'] + 4),
                     lambda m: m.update(max_abs_tolerance=float('nan')),
                     lambda m: m.update(mean_abs_tolerance=-1),
                     lambda m: m.update(generated_source_sha256='bad'),
                     lambda m: m.update(reference_blob_sha256='bad'),
                     lambda m: m.update(source_sha256={}),
                     lambda m: m['linker_flags'].pop(),
                     lambda m: m.update(uncovered_reference_tensors=['z', 'a'])]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                manifest = copy.deepcopy(self.manifest)
                mutate(manifest)
                with self.assertRaises(ValueError):
                    checker.validate_manifest(manifest)

    def test_reference_aliases_optional_unique_nonempty_and_distinct_from_primary(self):
        for aliases in (None, [], ['second', 'first']):
            manifest = copy.deepcopy(self.manifest)
            if aliases is None:
                manifest['entries'][0].pop('reference_aliases')
            else:
                manifest['entries'][0]['reference_aliases'] = aliases
            checker.validate_manifest(manifest)
        suffix = self.manifest['entries'][0]['reference_suffix']
        for aliases in ('suffix', [''], [' '], [1], [['nested']], ['same', 'same'], [suffix]):
            with self.subTest(aliases=aliases):
                manifest = copy.deepcopy(self.manifest)
                manifest['entries'][0]['reference_aliases'] = aliases
                with self.assertRaisesRegex(ValueError, 'aliases'):
                    checker.validate_manifest(manifest)

    def test_shared_symbol_and_distinct_phases_are_valid_boundaries(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['entries'][1].update(symbol=manifest['entries'][0]['symbol'], phase='before')
        manifest['entries'][3]['symbol'] = manifest['entries'][0]['symbol']
        manifest['linker_flags'] = ['--wrap=' + symbol for symbol in
                                    sorted({entry['symbol'] for entry in manifest['entries']})]
        result = checker.check('\n'.join(make_log(manifest)), manifest)
        self.assertEqual(result['status'], 'INTERMEDIATES_PASS', result['errors'])

    def test_missing_duplicate_reordered_and_invalid_records_reject(self):
        mutations = [lambda lines: lines.pop(2), lambda lines: lines.insert(2, lines[2]),
                     lambda lines: lines.__setitem__(slice(2, 4), [lines[3], lines[2]])]
        for old, new in [('entry=00000000', 'entry=00000003'),
                         ('entry=00000000', 'entry=00000007'),
                         ('position=00000000', 'position=0000000f'),
                         ('count=00000002', 'count=00000001'),
                         ('calls=00000001', 'calls=00000002'),
                         ('checked=00000001', 'checked=00000000'),
                         ('max_abs_bits=' + bits(.0005), 'max_abs_bits=' + bits(.01)),
                         ('mean_abs_bits=' + bits(.00005), 'mean_abs_bits=7fc00000')]:
            mutations.append(lambda lines, old=old, new=new: lines.__setitem__(2, lines[2].replace(old, new)))
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                lines = self.lines[:]
                mutate(lines)
                result = checker.check('\n'.join(lines), self.manifest)
                self.assertEqual(result['status'], 'NOT_ACCEPTED')
                self.assertTrue(result['errors'])
                self.assertEqual(result['expected_comparisons'], 35)

    def test_missing_completion_return_and_runtime_failure_reject(self):
        complete = next(i for i, line in enumerate(self.lines) if '[intermediate] complete' in line)
        mutations = [lambda lines: lines.pop(complete),
                     lambda lines: lines.insert(complete, lines[complete]),
                     lambda lines: lines.remove('[nr] RA returned: PASS'),
                     lambda lines: lines.append('[nr] RA TRAP mcause=0x2'),
                     lambda lines: lines.__setitem__(-2, '[nr] RA returned: FAIL')]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                lines = self.lines[:]
                mutate(lines)
                result = checker.check('\n'.join(lines), self.manifest)
                self.assertEqual(result['status'], 'NOT_ACCEPTED')
                self.assertTrue(result['errors'])

    def test_cli_reports_selected_scope_and_rejects_malformed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manifest, uart, output = (path / name for name in ('manifest.json', 'uart.log', 'verification.json'))
            manifest.write_text(json.dumps(self.manifest))
            uart.write_text('\n'.join(self.lines))
            command = [sys.executable, str(TOOLS / 'check_intermediates.py'), '--uart', str(uart),
                       '--manifest', str(manifest), '--output', str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report['expected_comparisons'], 35)
            self.assertIn('layers [14, 15] of 28', report['scope'])
            self.assertEqual(len(report['inputs_sha256']), 2)
            manifest.write_text(json.dumps({**self.manifest, 'selected_layers': [28]}))
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report['status'], 'NOT_ACCEPTED')
            self.assertNotIn('one decoder layer', report['scope'])


if __name__ == '__main__':
    unittest.main()
