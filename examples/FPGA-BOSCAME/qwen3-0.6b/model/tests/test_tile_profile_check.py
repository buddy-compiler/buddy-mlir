"""Adversarial acceptance checks for optional raw tile diagnostics."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import test_kernel_profile as base
import test_profile_tile_probe as probe_fixture

checker = base.checker
generator = base.generator
HIGH, RAW = probe_fixture.HIGH, probe_fixture.RAW


def raw_adapter(grid=(1, 2, 1)):
    lines = [probe_fixture.RAW_ADAPTER.split('void ' + HIGH, 1)[0],
             f'void {HIGH}(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {{']
    for i, size in enumerate((1, 1, 4)):
        lines += [f'void *p{i} = (unsigned char *)a{i}->aligned + a{i}->offset * {size};',
                  f'MemRef0 m{i} = {{p{i}, p{i}, 0}};']
    lines += [f'for (int32_t {axis}=0; {axis}<{value}; ++{axis})' for axis, value in zip('xyz', grid)]
    lines += [f'{RAW}(0, &m0, 0, &m1, 0, &m2, ' + ', '.join(map(str, grid)) + ', x, y, z);', '}']
    return '\n'.join(lines) + '\n'


def log_fixture():
    lines = []
    expected = {'prefill': {HIGH: 2}, 'decode': {HIGH: 0}}
    for kind, position in (('prefill', 0), ('decode', 16)):
        lines.append(f'[model] {kind} begin position={position:08X} input_token=00000000')
        for call in range(expected[kind][HIGH]):
            lines.append(f'[kernel] begin {HIGH} call={call:016X}')
            if call == 1:
                for y in range(2):
                    for phase in ('begin', 'returned'):
                        lines.append(f'[tile-probe] {phase} {RAW} call={call:016X} '
                            f'grid_x=00000001 grid_y=00000002 grid_z=00000001 '
                            f'x=00000000 y={y:08X} z=00000000')
            lines.append(f'[kernel] end {HIGH} call={call:016X}')
        lines.append(f'[model] {kind} position={position:08X} token=0000002A '
                     'logit_bits=3F800000 compute_cycles=0000000000000064')
    return '\n'.join(lines) + '\n', expected


class TileProfileCheck(unittest.TestCase):
    def setup_tile(self, directory):
        d = Path(directory)
        adapters, raw = probe_fixture.TileProbeTests().fixture(d)
        raw.write_text(raw_adapter())
        source, _ = generator.generate_profile(adapters, d, progress=True,
            probe=HIGH + ':1', tile_probe=True, raw_adapter=raw)
        profile = json.loads((d / 'kernel-profile.json').read_text())
        return profile, adapters, raw, source

    def test_manifest_exact_abi_and_additional_wrap_only(self):
        with tempfile.TemporaryDirectory() as directory:
            tile_profile, _, _, _ = self.setup_tile(directory)
            original, report, adapters, irs, _, _ = base.fixture()
            adapters = adapters.replace('_mlir_ciface_kernel_project', HIGH)
            adapters = adapters.replace(f'extern void {HIGH}(MemRef2 *, MemRef2 *);',
                                        f'extern void {HIGH}(MemRef2 *, MemRef2 *, MemRef2 *);')
            for record in original['kernels']:
                if record['symbol'].endswith('_project'):
                    record['symbol'] = HIGH
                    record['argument_types'] = ['MemRef2 *'] * 3
            original['kernels'].sort(key=lambda record: record['symbol'])
            for index, record in enumerate(original['kernels']): record['index'] = index
            original.update(adapter_sha256=checker.sha256(adapters.encode()), progress_uart=True,
                phase_probe=tile_profile['phase_probe'], tile_probe=tile_profile['tile_probe'])
            original['linker_flags'] = ['--wrap=' + r['symbol'] for r in original['kernels']] + ['--wrap=' + RAW]
            counts, _ = checker.expected_counts(original, report, adapters, irs)
            self.assertEqual(counts['prefill'][HIGH], 2)
            self.assertNotIn(RAW, counts['prefill'])
            for field, value in (('raw_symbol', RAW + '_wrong'), ('call_index', 0),
                    ('raw_argument_types', ['int64_t'] * 12), ('descriptor_layout_lp64', {}),
                    ('raw_return_type', 'int'), ('linker_flag', '--wrap=wrong')):
                bad = copy.deepcopy(original)
                bad['tile_probe'][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    checker.expected_counts(bad, report, adapters, irs)
            for flags in (original['linker_flags'][:-1], original['linker_flags'] + ['--wrap=extra']):
                bad = copy.deepcopy(original); bad['linker_flags'] = flags
                with self.assertRaisesRegex(ValueError, 'linker wrappers'):
                    checker.expected_counts(bad, report, adapters, irs)
            bad = copy.deepcopy(original); del bad['tile_probe']
            with self.assertRaisesRegex(ValueError, 'linker wrappers'):
                checker.expected_counts(bad, report, adapters, irs)
            bad = copy.deepcopy(original); bad['nh_watch'] = {'selection': bad['phase_probe']}
            with self.assertRaisesRegex(ValueError, 'NH watch diagnostics'):
                checker.expected_counts(bad, report, adapters, irs)

    def test_grid_comes_from_exact_adapter_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            profile, _, raw, _ = self.setup_tile(directory)
            tile = profile['tile_probe']
            text = raw.read_text()
            self.assertEqual(checker.tile_probe_grid(tile, text), [1, 2, 1])
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                checker.tile_probe_grid(tile, text + '\n')
            for mutant in (text.replace('y<2', 'y<3'), text.replace('x, y, z);', 'x, x, z);'),
                           text.replace('for (int32_t x', 'if (a0) for (int32_t x')):
                changed = dict(tile, raw_adapter_sha256=checker.sha256(mutant.encode()))
                with self.assertRaisesRegex(ValueError, 'unsupported body'):
                    checker.tile_probe_grid(changed, mutant)

    def test_pairs_counts_order_scope_and_unselected_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            profile, _, _, _ = self.setup_tile(directory)
            tile = profile['tile_probe']
            log, expected = log_fixture()
            self.assertEqual(checker.check_tile_progress(log, expected, tile, [1, 2, 1]), [])
            record = next(line for line in log.splitlines() if '[tile-probe] returned' in line)
            begin = next(line for line in log.splitlines() if '[tile-probe] begin' in line)
            mutants = [log.replace(record + '\n', '', 1), log.replace(record, record + '\n' + record, 1),
                log.replace('grid_y=00000002', 'grid_y=00000003', 1),
                log.replace('x=00000000 y=00000000', 'x=00000000 y=00000001', 1),
                log.replace(RAW + ' call=0000000000000001', RAW + ' call=0000000000000000', 1),
                log.replace('[tile-probe] begin ' + RAW, '[tile-probe] begin wrong', 1),
                begin + '\n' + log, log + begin + '\n',
                '\n'.join(line for line in log.splitlines() if '[tile-probe]' not in line),
                log.replace(' grid_x=00000001', ' grid_x=X', 1),
                log.replace('[model] prefill position=', '[model] prefill invalid position=')]
            for mutant in mutants:
                with self.subTest(mutant=mutant):
                    self.assertTrue(checker.check_tile_progress(mutant, expected, tile, [1, 2, 1]))
            self.assertTrue(checker.check_tile_progress(log, expected))

    def test_cli_uses_hashed_adapter_and_rejects_lost_tile_return(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            _, adapters, raw, _ = self.setup_tile(directory)
            adapters.write_text(adapters.read_text() +
                f'void _mlir_ciface_qwen_graph_project(MemRef2 *a, MemRef2 *b, MemRef2 *c) {{\n'
                f'  {HIGH}(a, b, c);\n  {HIGH}(a, b, c);\n}}\n')
            source, _ = generator.generate_profile(adapters, d, progress=True,
                probe=HIGH + ':1', tile_probe=True, raw_adapter=raw)
            plan = {'profile_kernels': True, 'profile_progress': True,
                    'profile_probe': HIGH + ':1', 'completion_sync': 'ame-resync', 'profile_tile_probe': True}
            (d/'image-plan.json').write_text(json.dumps(plan))
            (d/'image.json').write_text(json.dumps({'input_sha256': {
                str(path): checker.sha256(path.read_bytes()) for path in (source, adapters)}}))
            for kind in ('prefill', 'decode'):
                (d / (kind + '.ll')).write_text(f'''define void @forward_{kind}() {{
  call void @qwen_graph_project()
  ret void
}}
define void @qwen_graph_project() {{
  call void @_mlir_ciface_qwen_graph_project()
  ret void
}}
''')
            (d / 'replacement.json').write_text(json.dumps({'graphs': {
                kind: {'external_calls': 1} for kind in ('prefill', 'decode')},
                'distinct_symbols': ['qwen_graph_project']}))
            log, _ = log_fixture()
            log = log.split('[model] decode begin', 1)[0]
            log += f'[profile] position=0000000F kernel={HIGH} calls=0000000000000002 cycles=0000000000000020\n'
            log += '[nr] RA returned: PASS\n'
            (d / 'uart.log').write_text(log)
            argv = ['check', '--uart', str(d/'uart.log'), '--profile', str(d/'kernel-profile.json'),
                '--replacement', str(d/'replacement.json'), '--adapters', str(adapters),
                '--prefill-ir', str(d/'prefill.ll'), '--decode-ir', str(d/'decode.ll'),
                '--image-plan', str(d/'image-plan.json'), '--image-build', str(d/'image.json'),
                '--steps', '0', '--output', str(d/'result.json')]
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(checker.main(), 0)
            result = json.loads((d / 'result.json').read_text())
            self.assertEqual(result['tile_probe']['grid'], [1, 2, 1])
            self.assertEqual(result['image_identity']['status'], 'BUILD_SOURCE_AND_PLAN_VERIFIED')
            self.assertEqual(result['inputs_sha256'][str(raw)], checker.sha256(raw.read_bytes()))
            lost = next(line for line in log.splitlines() if '[tile-probe] returned' in line)
            (d / 'uart.log').write_text(log.replace(lost + '\n', '', 1))
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(checker.main(), 1)
            self.assertTrue(any('tile probe:' in error for error in
                               json.loads((d/'result.json').read_text())['errors']))
            (d/'uart.log').write_text(log)
            (d/'image-plan.json').write_text(json.dumps({**plan, 'profile_tile_probe': False}))
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(checker.main(), 1)
            self.assertIn('tile configuration', json.loads((d/'result.json').read_text())['errors'][0])


if __name__ == '__main__':
    unittest.main()
