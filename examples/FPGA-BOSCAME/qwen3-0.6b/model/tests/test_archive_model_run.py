"""Tamper rejection for reusable model evidence archiving; no board required."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from test_decode_hang_watch import frame

TOOLS = Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('model_archive', TOOLS/'archive_model_run.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def dump(path, value):
    path.write_text(json.dumps(value))


class ArchiveModelRunTests(unittest.TestCase):
    def setup_boundary(self, directory):
        from test_image_boundary_sync import prior_manifest
        root = Path(directory)
        prior = prior_manifest()
        prior['layers'] = 28
        source_text = '#include "support.h"\n#include "nr_runtime.h"\n'
        for entry in prior['entries']:
            symbol = entry['symbol']
            entry['ir_line'] = ('call @' + symbol.removeprefix('_mlir_ciface_')
                               + '(%a, %b, %s) : (tensor<1x2xf32>, tensor<1x2xf32>, tensor<1xf32>) -> ()')
        symbols = sorted(entry['symbol'] for entry in prior['entries'])
        for symbol in symbols:
            params = 'MemRef2 *a0, MemRef2 *a1, MemRef1 *a2'
            source_text += (f'extern void __real_{symbol}({params});\n'
                            f'void __wrap_{symbol}({params}) {{\n'
                            f'  __real_{symbol}(a0, a1, a2);\n  ame_fence();\n}}\n')
        original = root/'prior.json'
        dump(original, prior)
        selection = root/'boundary-sync-selection.json'
        selection.write_bytes(original.read_bytes())
        source = root/'boundary-sync.c'
        source.write_text(source_text)
        dependency = root/'graph.mlir'
        dependency.write_text('module {}\n')
        flags = ['--wrap=' + symbol for symbol in symbols]
        manifest = {'schema_version': 1, 'symbols': symbols, 'linker_flags': flags,
                    'layers': 28, 'selected_layers': [0], 'prefill_len': 16, 'decode_steps': 8,
                    'completion_sync': 'ame-resync', 'scope': 'sync only', 'limitations': [],
                    'generated_source_sha256': archive.digest(source),
                    'source_sha256': {str(path): archive.digest(path) for path in (original, dependency)}}
        dump(root/'boundary-sync.json', manifest)
        plan = {'boundary_sync': manifest, 'layers': 28,
                'boundary_sync_requested': {'manifest': str(original), 'manifest_sha256': archive.digest(original)},
                'numeric_reference': {'prefill_len': 16, 'decode_steps': 8}}
        record = {'input_sha256': {str(path): archive.digest(path) for path in (source, original)},
                  'linker': {'path': '/fixture/ld.lld'},
                  'commands': [['/fixture/ld.lld', '-o', str(root/'model.elf'), *flags,
                                str(root/'boundary-sync.o')]]}
        return SimpleNamespace(image=root, source=source, dependency=dependency, selection=selection,
                               original=original, manifest=manifest, plan=plan, record=record)

    def test_boundary_sync_binds_plan_sources_and_build_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(archive.boundary_sync_identity(root, {}, {}))
            f = self.setup_boundary(directory)
            with self.assertRaisesRegex(ValueError, 'exact image plan'):
                archive.boundary_sync_identity(root, {}, {})
            result = archive.boundary_sync_identity(root, f.plan, f.record)
            self.assertEqual(result['status'], 'BUILD_SOURCE_AND_PLAN_VERIFIED')
            self.assertEqual(result['selection_sha256'], archive.digest(f.original))
            self.assertEqual(result['selected_layers'], [0])
            f.dependency.write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'source hash mismatch'):
                archive.boundary_sync_identity(root, f.plan, f.record)
            f.dependency.write_text('module {}\n')
            f.source.write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'build-time input hash'):
                archive.boundary_sync_identity(root, f.plan, f.record)

    def test_boundary_sync_rejects_relabelled_layers_and_symbol_subsets(self):
        for change in ('layers', 'symbols'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                f = self.setup_boundary(directory)
                if change == 'layers':
                    f.manifest['selected_layers'] = [1]
                else:
                    f.manifest['symbols'] = f.manifest['symbols'][:1]
                    f.manifest['linker_flags'] = f.manifest['linker_flags'][:1]
                dump(f.image/'boundary-sync.json', f.manifest)
                with self.assertRaisesRegex(ValueError, 'copied selection'):
                    archive.boundary_sync_identity(f.image, f.plan, f.record)

    def test_boundary_sync_selection_is_exact_build_input(self):
        for change in ('copy', 'rehash-prior', 'missing-build-hash', 'missing-copy', 'request-hash'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                f = self.setup_boundary(directory)
                if change == 'copy':
                    f.selection.write_bytes(f.selection.read_bytes() + b'\n')
                elif change == 'rehash-prior':
                    f.original.write_bytes(f.original.read_bytes() + b'\n')
                    f.selection.write_bytes(f.original.read_bytes())
                    f.manifest['source_sha256'][str(f.original)] = archive.digest(f.original)
                    f.plan['boundary_sync_requested']['manifest_sha256'] = archive.digest(f.original)
                    dump(f.image/'boundary-sync.json', f.manifest)
                elif change == 'missing-build-hash':
                    del f.record['input_sha256'][str(f.original)]
                elif change == 'missing-copy':
                    f.selection.unlink()
                else:
                    f.plan['boundary_sync_requested']['manifest_sha256'] = '0' * 64
                with self.assertRaises(ValueError):
                    archive.boundary_sync_identity(f.image, f.plan, f.record)

    def test_boundary_sync_rejects_non_identity_source_even_if_hashes_match(self):
        for change in ('swap', 'no-fence', 'oracle-read', 'extra-wrapper', 'wrong-rank'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                f = self.setup_boundary(directory)
                text = f.source.read_text()
                if change == 'swap':
                    text = text.replace('(a0, a1, a2);', '(a1, a0, a2);', 1)
                elif change == 'no-fence':
                    text = text.replace('  ame_fence();\n', '', 1)
                elif change == 'oracle-read':
                    text = text.replace('  ame_fence();', '  *(volatile float *)a0->aligned;\n  ame_fence();', 1)
                elif change == 'wrong-rank':
                    text = text.replace('MemRef2 *a0', 'MemRef1 *a0')
                else:
                    text += 'void __wrap_extra(void) { ame_fence(); }\n'
                f.source.write_text(text)
                f.record['input_sha256'][str(f.source)] = archive.digest(f.source)
                f.manifest['generated_source_sha256'] = archive.digest(f.source)
                dump(f.image/'boundary-sync.json', f.manifest)
                with self.assertRaisesRegex(ValueError, 'identity-forwarding'):
                    archive.boundary_sync_identity(f.image, f.plan, f.record)

    def test_boundary_sync_link_flags_match_exact_selection(self):
        for change in ('missing', 'extra', 'duplicate', 'separate-spelling', 'single-dash', 'two-links', 'no-linker'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                f = self.setup_boundary(directory)
                command = f.record['commands'][0]
                if change == 'missing':
                    command.remove(f.manifest['linker_flags'][0])
                elif change == 'extra':
                    command.append('--wrap=_mlir_ciface_qwen_graph_other')
                elif change == 'duplicate':
                    command.append(f.manifest['linker_flags'][0])
                elif change == 'separate-spelling':
                    index = command.index(f.manifest['linker_flags'][0])
                    command[index:index+1] = ['--wrap', f.manifest['symbols'][0]]
                elif change == 'single-dash':
                    command.append('-wrap=_mlir_ciface_qwen_graph_other')
                elif change == 'two-links':
                    f.record['commands'].append(command.copy())
                else:
                    del f.record['linker']
                with self.assertRaisesRegex(ValueError, 'link'):
                    archive.boundary_sync_identity(f.image, f.plan, f.record)

    def setup_profile(self, directory, tile=False):
        from kernel_profile import generate_profile
        from test_profile_tile_probe import TileProbeTests, HIGH
        root = Path(directory)
        adapters, raw = TileProbeTests().fixture(root)
        source, _ = generate_profile(adapters, root, progress=True, probe=HIGH + ':14',
            completion_sync='fence', tile_probe=tile, raw_adapter=raw)
        record = {'input_sha256': {str(path): archive.digest(path) for path in (source, adapters)}}
        plan = {'profile_kernels': True, 'profile_progress': True, 'profile_probe': HIGH + ':0xE',
                'completion_sync': 'fence'}
        if tile:
            plan['profile_tile_probe'] = True
        return SimpleNamespace(image=root, adapters=adapters, source=source, record=record, plan=plan)

    def test_profile_identity_binds_built_source_in_plain_and_tile_modes(self):
        for tile in (False, True):
            with self.subTest(tile=tile), tempfile.TemporaryDirectory() as directory:
                f = self.setup_profile(directory, tile)
                identity = archive.profile_identity(f.image, f.plan, f.record, f.adapters)
                self.assertEqual(identity['status'], 'BUILD_SOURCE_AND_PLAN_VERIFIED')
                self.assertEqual(identity['source_sha256'], archive.digest(f.source))
                self.assertEqual(identity['configuration']['completion_sync'], 'fence')
                self.assertEqual(bool(identity['configuration']['tile_probe']), tile)
                manifest_path = f.image/'kernel-profile.json'
                original_manifest = archive.read(manifest_path)
                for path, key in ((f.source, 'source_sha256'), (f.adapters, 'adapter_sha256')):
                    original_source = path.read_bytes()
                    path.write_bytes(original_source + b'/* altered after compilation */\n')
                    manifest = {**original_manifest, key: archive.digest(path)}
                    dump(manifest_path, manifest)
                    with self.assertRaisesRegex(ValueError, 'build-time input hash'):
                        archive.profile_identity(f.image, f.plan, f.record, f.adapters)
                    path.write_bytes(original_source)
                    dump(manifest_path, original_manifest)
                original_hashes = f.record['input_sha256'].copy()
                for path in (f.source, f.adapters):
                    f.record['input_sha256'] = {k: v for k, v in original_hashes.items() if k != str(path)}
                    with self.assertRaisesRegex(ValueError, 'build-time input hash'):
                        archive.profile_identity(f.image, f.plan, f.record, f.adapters)

    def test_profile_identity_rejects_plan_manifest_configuration_drift(self):
        for tile in (False, True):
            with self.subTest(tile=tile), tempfile.TemporaryDirectory() as directory:
                f = self.setup_profile(directory, tile)
                path = f.image/'kernel-profile.json'
                original = archive.read(path)
                for field, value in (('completion_sync', 'ame-resync'), ('progress_uart', False),
                                     ('phase_probe', None), ('nh_watch', {'selection': original['phase_probe']})):
                    dump(path, {**original, field: value})
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        archive.profile_identity(f.image, f.plan, f.record, f.adapters)
                dump(path, original)
                for field, value in (('completion_sync', 'ame-resync'), ('profile_progress', False),
                                     ('profile_probe', None), ('profile_tile_probe', not tile),
                                     ('profile_watch', True), ('profile_kernels', False)):
                    plan = {**f.plan, field: value}
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        archive.profile_identity(f.image, plan, f.record, f.adapters)
                if tile:
                    bad = copy.deepcopy(original)
                    bad['tile_probe']['raw_symbol'] = 'triton_other'
                    dump(path, bad)
                    with self.assertRaisesRegex(ValueError, 'raw symbol'):
                        archive.profile_identity(f.image, f.plan, f.record, f.adapters)

    def test_profile_identity_rejects_stray_evidence_and_source_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(archive.profile_identity(root, {}, {}, root/'absent.c'))
            f = self.setup_profile(root)
            with self.assertRaisesRegex(ValueError, 'exact image plan'):
                archive.profile_identity(root, {}, f.record, f.adapters)
            saved = root/'saved.c'
            f.source.rename(saved)
            f.source.symlink_to(saved)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                archive.profile_identity(root, f.plan, f.record, f.adapters)

    def test_profile_watch_binds_plan_source_and_nh_frames(self):
        from kernel_profile import generate_profile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            symbol = '_mlir_ciface_kernel_matmul_16x3072x1024'
            adapters = root/'adapters.c'
            adapters.write_text(f'extern void {symbol}(MemRef2 *, MemRef2 *, MemRef2 *);\n')
            source, _ = generate_profile(adapters, root, progress=True, probe=symbol+':38', watch=True)
            record = {'input_sha256': {str(p): archive.digest(p) for p in (source, adapters)}}
            plan = {'profile_watch': True, 'profile_kernels': True, 'profile_progress': True,
                    'profile_probe': symbol+':38'}
            clean, report, identity = archive.decode_run_uart(root, plan, record, adapters, b'prefix'+frame()+b'suffix')
            self.assertEqual(clean, b'prefixsuffix')
            self.assertEqual(report['frame_count'], 1)
            self.assertEqual(identity['reset_symbol'], 'qwen_profile_reset')
            for wrong in ({**plan, 'profile_probe': symbol+':37'}, {**plan, 'profile_kernels': False}):
                with self.assertRaises(ValueError):
                    archive.decode_run_uart(root, wrong, record, adapters, frame())
            for raw in (b'no frames', frame()[:-1], frame(dropped=1)):
                with self.assertRaises(ValueError):
                    archive.decode_run_uart(root, plan, record, adapters, raw)
            source.write_text(source.read_text()+'/* changed */')
            with self.assertRaisesRegex(ValueError, 'source hash'):
                archive.decode_run_uart(root, plan, record, adapters, frame())

    def setup_watch(self, directory):
        from hang_watch import generate_watch
        root = Path(directory)
        image = root/'image'
        adapters = root/'adapters.c'
        symbol = '_mlir_ciface_kernel_matmul_1x1024x2048'
        adapters.write_text(f'extern void {symbol}(MemRef2 *, MemRef2 *, MemRef2 *);\n')
        source, _ = generate_watch(adapters, image, symbol + ':14')
        record = {'input_sha256': {str(p): archive.digest(p) for p in (adapters, source)}}
        return SimpleNamespace(image=image, adapters=adapters, source=source,
            plan={'hang_watch': symbol + ':14', 'console': {'capacity_bytes': 524288}}, record=record)

    def decode_watch(self, fixture, uart):
        return archive.decode_run_uart(fixture.image, fixture.plan, fixture.record, fixture.adapters, uart)

    def test_hang_watch_restores_split_lines_and_preserves_raw_run_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            f = self.setup_watch(root)
            run, prepared, expected, result, _ = self.setup_run(root)
            raw = expected[:9] + frame(count=70000, consumed=0, pending=70000) + expected[9:]
            result['uart_bytes'] = len(raw)
            dump(run/'result.json', result)
            (run/'uart.raw.log').write_bytes(raw)
            archive.verify_run_identity(run, prepared, 'elf-hash', raw)
            clean, report, identity = self.decode_watch(f, raw)
            self.assertEqual(clean, expected)
            self.assertEqual(report['console_capacity_bytes'], 524288)
            self.assertEqual(report['raw_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(identity['ra_sha256'], hashlib.sha256(expected).hexdigest())
            self.assertTrue(identity['diagnostic_only'])
            self.assertEqual(identity['throughput_acceptance'], 'NOT_EVALUATED')
            with self.assertRaisesRegex(ValueError, 'UART byte count'):
                archive.verify_run_identity(run, prepared, 'elf-hash', clean)
            output = root/'archive'
            output.mkdir()
            path = archive.write_decoded_uart(output, clean, report)
            self.assertEqual(path.name, 'uart.ra.log')
            self.assertEqual(path.read_bytes(), expected)
            self.assertEqual(archive.read(output/'nh-watch.json')['raw_sha256'], identity['raw_sha256'])
            self.assertEqual((run/'uart.raw.log').read_bytes(), raw)

    def test_hang_watch_rejects_missing_dropped_truncated_inconsistent_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.setup_watch(directory)
            cases = (b'no frames', frame(dropped=1), frame()[:-1], frame() + b'\x1eNRWA',
                     frame(sample=2), frame(pending=11), frame(seq_after=4),
                     frame() + frame(sample=3), frame(count=600000, consumed=0, pending=600000),
                     frame() + b'\x1eNRWATCHwrong\n')
            for raw in cases:
                with self.subTest(raw=raw[:80]), self.assertRaises(ValueError):
                    self.decode_watch(f, raw)
            bad = frame().replace(b'pending=000000000000000A', b'pending=wrong')
            with self.assertRaisesRegex(ValueError, 'invalid hexadecimal'):
                self.decode_watch(f, bad)

    def test_no_watch_plan_rejects_nh_frames_and_stray_watch_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plain = b'[model] graph complete\r\n'
            self.assertEqual(archive.decode_run_uart(root, {}, {}, root/'absent.c', plain),
                             (plain, None, None))
            for raw in (frame(), frame()[:-1], b'\x1eNRWA', b'\x1eNRWATCHwrong\n'):
                with self.subTest(raw=raw[:80]), self.assertRaises(ValueError):
                    archive.decode_run_uart(root, {}, {}, root/'absent.c', raw)
            dump(root/'hang-watch.json', {})
            with self.assertRaisesRegex(ValueError, 'exact image plan'):
                archive.decode_run_uart(root, {}, {}, root/'absent.c', plain)

    def test_hang_watch_binds_selection_abi_and_built_source(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.setup_watch(directory)
            path = f.image/'hang-watch.json'
            manifest = archive.read(path)
            changes = {'schema_version': 2, 'watch': {'symbol': 'other', 'call_index': 14},
                       'argument_types': ['MemRef2 *'], 'return_type': 'int',
                       'reset_symbol': 'other', 'linker_flags': [],
                       'adapter_sha256': '0' * 64, 'source_sha256': '0' * 64}
            for key, value in changes.items():
                dump(path, {**manifest, key: value})
                with self.subTest(key=key), self.assertRaises(ValueError):
                    self.decode_watch(f, frame())
            dump(path, manifest)
            original = f.record['input_sha256'].copy()
            for source in (f.source, f.adapters):
                f.record['input_sha256'] = {**original, str(source): '0' * 64}
                with self.assertRaisesRegex(ValueError, 'build-time input hash'):
                    self.decode_watch(f, frame())
            f.record['input_sha256'] = original
            f.plan.pop('console')
            self.assertEqual(self.decode_watch(f, frame())[1]['console_capacity_bytes'], 65536)
            with self.assertRaisesRegex(ValueError, 'inconsistent'):
                self.decode_watch(f, frame(count=70000, consumed=0, pending=70000))

    def setup_intermediates(self, directory):
        """Small independent data exercises every generator layout transform."""
        d = Path(directory)
        arrays, blob, source, metadata = (d/name for name in
            ('arrays.npz', 'intermediate-reference.bin', 'intermediate-probe.c', 'quant-reference.json'))
        source.write_text('/* generated probe test fixture */\n')
        dump(metadata, {'layers': 1, 'capacity': 512, 'arithmetic_profile': 'nr-fpga', 'prompt_ids': list(range(16))})
        entries, data, packed, cursor = [], {}, [], 0
        for index in range(92):
            graph = 'prefill' if index < 46 else 'decode'
            transform = ('identity', 'last_row', 'heads_first', 'context_heads_first', 'projection_heads')[index % 5]
            shape = {'identity': [1, 2, 2], 'last_row': [1, 1, 2], 'heads_first': [2, 1, 3],
                     'context_heads_first': [16, 1, 128], 'projection_heads': [1, 2, 2, 3]}[transform]
            dtype = 'i8' if index % 5 == 0 else 'f32'
            elements = int(np.prod(shape))
            suffix = f'model.layers.0_tensor_{index}'
            entry = {'index': index, 'graph': graph, 'layer': 0, 'symbol': f'_mlir_ciface_qwen_graph_probe_{index}',
                     'operand': 0, 'phase': 'before', 'reference_suffix': suffix, 'transform': transform,
                     'shape': shape, 'dtype': dtype, 'rank': len(shape), 'elements': elements,
                     'reference_keys': [], 'offsets': []}
            for step in range(1 if graph == 'prefill' else 8):
                key = ('prefill' if graph == 'prefill' else f'decode_{step}') + '_' + suffix
                if transform == 'last_row':
                    value = np.arange(6, dtype=np.float32).reshape(3, 2) + step
                    expected = value[-1:]
                elif transform == 'heads_first':
                    value = np.arange(elements, dtype=np.float32).reshape(1, 2, 3) + step
                    expected = np.stack((value[:, 0, :], value[:, 1, :]))
                elif transform == 'context_heads_first':
                    value = np.arange(elements, dtype=np.float32).reshape(1, 2048) + step
                    expected = np.stack([value[:, i*128:(i+1)*128] for i in range(16)])
                elif transform == 'projection_heads':
                    value = np.arange(elements, dtype=np.float32).reshape(2, 6) + step
                    expected = value.reshape(2, 2, 3)
                else:
                    value = np.arange(elements, dtype=np.int8).reshape(2, 2) + step
                    expected = value
                data[key] = value
                packed.append(expected.astype('<f4').tobytes(order='C'))
                entry['reference_keys'].append(key)
                entry['offsets'].append(cursor)
                cursor += elements
            entries.append(entry)
        data['prefill_model.layers.0_unhooked'] = np.zeros(2, dtype=np.float32)
        np.savez(arrays, **data)
        blob.write_bytes(b''.join(packed))
        manifest = {'schema_version': 1, 'layers': 1, 'prefill_len': 16, 'decode_steps': 8,
                    'max_abs_tolerance': 0.001, 'mean_abs_tolerance': 0.0001, 'entries': entries,
                    'reference_bytes': cursor * 4, 'reference_blob_sha256': archive.digest(blob),
                    'generated_source_sha256': archive.digest(source),
                    'source_sha256': {str(p): archive.digest(p) for p in (arrays, metadata)},
                    'uncovered_reference_tensors': ['prefill_model.layers.0_unhooked'],
                    'linker_flags': ['--wrap=' + symbol for symbol in sorted(e['symbol'] for e in entries)]}
        return SimpleNamespace(manifest=manifest, arrays=arrays, blob=blob, source=source, metadata=metadata, data=data)

    def verify_intermediates(self, fixture):
        return archive.verify_intermediate_reference(fixture.manifest, fixture.arrays, fixture.blob,
                                                     fixture.source, fixture.metadata)

    def test_verified_alias_coverage_preserves_older_manifest_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.setup_intermediates(directory)
            entry = f.manifest['entries'][0]
            alias = 'model.layers.0_shared_input'
            alias_key = 'prefill_' + alias
            entry['reference_aliases'] = [alias]
            f.data[alias_key] = f.data[entry['reference_keys'][0]].copy()
            np.savez(f.arrays, **f.data)
            f.manifest['source_sha256'][str(f.arrays)] = archive.digest(f.arrays)
            # Existing image manifests counted only primary reference keys.
            f.manifest['uncovered_reference_tensors'].append(alias_key)
            f.manifest['uncovered_reference_tensors'].sort()
            self.verify_intermediates(f)
            f.manifest['coverage_includes_aliases'] = True
            f.manifest['uncovered_reference_tensors'].remove(alias_key)
            self.verify_intermediates(f)
            f.data[alias_key].flat[0] += 1
            np.savez(f.arrays, **f.data)
            f.manifest['source_sha256'][str(f.arrays)] = archive.digest(f.arrays)
            with self.assertRaisesRegex(ValueError, 'shared intermediate reference mismatch'):
                self.verify_intermediates(f)

    def setup_run(self, directory):
        d = Path(directory); run = d/'run'; prep = d/'prepared';run.mkdir();prep.mkdir()
        uart = b'[nr] RA returned: PASS\r\n'
        segments = []
        for name, file, address in [('model','image.bin',0x80000000),('weights','weights-w8a8.bin',0xb8000000)]:
            (prep/file).write_bytes(bytes([len(name)])*64)
            segments.append(dict(name=name,file=file,address=address,size=64,sha256=archive.digest(prep/file)))
        plan = ['version=1','ddr_base=0x80000000','ddr_size=0x400000000','alignment=64']
        for s in segments:
            plan += ['[[segments]]']+[f'{k}={json.dumps(v)}' for k,v in s.items()]
        (prep/'ddr-load.plan').write_text('\n'.join(plan))
        deployment = {'elf_sha256':'elf-hash','segments':segments}
        result = {'status':'OK','ddr_readback_matches':True,'completion_marker_seen':True,
                  'uart_bytes':len(uart),'sha256':segments[0]['sha256'],
                  'segment_readbacks':{s['file']:True for s in segments}}
        manifest = {'plan_sha256':archive.digest(prep/'ddr-load.plan'),
                    'segments':[{k:s[k] for k in ('file','size','sha256')} for s in segments]}
        dump(run/'result.json',result);dump(run/'run-manifest.json',manifest)
        dump(prep/'deployment.json',deployment)
        return run,prep,uart,result,manifest

    def test_exact_manifest_completion_and_optional_local_readbacks(self):
        with tempfile.TemporaryDirectory() as d:
            run,prep,uart,result,manifest=self.setup_run(d)
            validated=archive.verify_run_identity(run,prep,'elf-hash',uart)
            self.assertTrue(all(x['worker_readback_verified'] for x in validated[3]))
            self.assertFalse(any(x['downloaded_readback_rehashed'] for x in validated[3]))
            (run/'image.bin.readback').write_bytes((prep/'image.bin').read_bytes())
            self.assertTrue(archive.verify_run_identity(run,prep,'elf-hash',uart)[3][0]['downloaded_readback_rehashed'])
            (run/'image.bin.readback').write_bytes(b'X'*64)
            with self.assertRaisesRegex(ValueError,'downloaded DDR'):
                archive.verify_run_identity(run,prep,'elf-hash',uart)

    def test_rejects_missing_segment_wrong_hash_false_success_and_symlinks(self):
        with tempfile.TemporaryDirectory() as d:
            run,prep,uart,result,manifest=self.setup_run(d)
            mutants=[]
            r=copy.deepcopy(result);del r['segment_readbacks']['weights-w8a8.bin'];mutants.append(r)
            r=copy.deepcopy(result);r['status']='ERROR';mutants.append(r)
            r=copy.deepcopy(result);r['uart_bytes']+=1;mutants.append(r)
            r=copy.deepcopy(result);r['sha256']='wrong';mutants.append(r)
            for r in mutants:
                dump(run/'result.json',r)
                with self.assertRaises(ValueError): archive.verify_run_identity(run,prep,'elf-hash',uart)
            dump(run/'result.json',result)
            (prep/'weights-w8a8.bin').write_bytes(b'X'*64)
            with self.assertRaisesRegex(ValueError,'segment bytes'):
                archive.verify_run_identity(run,prep,'elf-hash',uart)
            with self.assertRaises(ValueError): archive.safe_file(prep,'../escape')
            (prep/'linked').symlink_to(prep/'image.bin')
            with self.assertRaises(ValueError): archive.safe_file(prep,'linked')

    def test_oracle_must_reconstruct_from_npz_and_match_full_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);npz=d/'a.npz';blob=d/'oracle.bin'
            logits=np.zeros((1,1,151936),dtype=np.float32);logits[0,0,42]=1
            kv=np.arange(8*3*128,dtype=np.float32).reshape(1,3,8,128)
            np.savez(npz,prefill_logits=logits,decode_logits_0=logits,kv_key_used=kv,kv_value_used=kv)
            blob.write_bytes(b''.join(x.astype('<f4').tobytes() for x in
                (logits,logits,kv.transpose(0,2,1,3),kv.transpose(0,2,1,3))))
            m={'schema_version':1,'layers':1,'prefill_len':2,'decode_steps':1,'vocab_size':151936,
               'arithmetic_profile':'nr-fpga','trajectory_verified':True,'capacity':512,'prompt_ids':[1,2],
               'source_sha256':archive.digest(npz),'blob_sha256':archive.digest(blob),'bytes':blob.stat().st_size}
            meta={'layers':1,'capacity':512,'arithmetic_profile':'nr-fpga','prompt_ids':[1,2],
                  'prefill_argmax_last':42,'decode_steps_recorded':[dict(step=0,cache_position=2,input_token=42,generated_token=42)]}
            self.assertEqual(archive.verify_reference(m,meta,npz,blob,layers=1,prefill=2,steps=1),[42,42])
            with self.assertRaisesRegex(ValueError,'layers'):
                archive.verify_reference(m,meta,npz,blob,layers=4,prefill=2,steps=1)
            bad=copy.deepcopy(meta);bad['decode_steps_recorded'][0]['input_token']=43
            with self.assertRaisesRegex(ValueError,'trajectory'):
                archive.verify_reference(m,bad,npz,blob,layers=1,prefill=2,steps=1)
            blob.write_bytes(b'\0'*blob.stat().st_size);m['blob_sha256']=archive.digest(blob)
            with self.assertRaisesRegex(ValueError,'reconstruct'):
                archive.verify_reference(m,meta,npz,blob,layers=1,prefill=2,steps=1)

    def test_elf_uses_actual_embedded_bytes_not_manifest_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);elf=d/'a.elf';blob=d/'blob.bin';blob.write_bytes(b'ORACLE')
            data=bytearray(256);data[:6]=b'\x7fELF\x02\x01'
            struct.pack_into('<H',data,18,243);struct.pack_into('<Q',data,32,64)
            struct.pack_into('<HH',data,54,56,1)
            struct.pack_into('<IIQQQQQQ',data,64,1,4,128,0x80000000,0x80000000,64,64,64)
            data[132:138]=b'ORACLE';elf.write_bytes(data)
            self.assertEqual(archive.elf_blob_matches(elf,0x80000004,blob),132)
            data[132]=0;elf.write_bytes(data)
            with self.assertRaisesRegex(ValueError,'embedded'):
                archive.elf_blob_matches(elf,0x80000004,blob)
            with self.assertRaisesRegex(ValueError,'file-backed'):
                archive.elf_blob_matches(elf,0x90000000,blob)

    def test_refuses_archive_overwrite_before_other_work(self):
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);(d/'old.txt').write_text('preserve')
            with self.assertRaisesRegex(ValueError,'overwrite'):
                archive.archive(SimpleNamespace(output=d))
            self.assertEqual((d/'old.txt').read_text(),'preserve')

    def test_intermediate_reference_reconstructs_every_layout_and_resolves_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.setup_intermediates(directory)
            self.assertEqual(self.verify_intermediates(f)['layers'], 1)
            arrays, metadata, sources = archive.intermediate_sources(f.manifest)
            self.assertEqual((arrays, metadata), (f.arrays, f.metadata))
            self.assertEqual(set(sources), {f.arrays, f.metadata})
            layout = Path(directory)/'weight-layout.json'
            layout.write_text('{}')
            f.manifest['source_sha256'][str(layout)] = archive.digest(layout)
            archive.intermediate_sources(f.manifest)
            layout.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'source hash'):
                archive.intermediate_sources(f.manifest)

    def test_intermediate_generated_source_npz_and_metadata_hashes_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.setup_intermediates(directory)
            original = f.source.read_bytes()
            f.source.write_bytes(original + b'changed')
            with self.assertRaisesRegex(ValueError, 'generated source hash'):
                self.verify_intermediates(f)
            f.source.write_bytes(original)
            for path in (f.arrays, f.metadata):
                expected = f.manifest['source_sha256'][str(path)]
                f.manifest['source_sha256'][str(path)] = '0' * 64
                with self.assertRaisesRegex(ValueError, 'input hash'):
                    self.verify_intermediates(f)
                f.manifest['source_sha256'][str(path)] = expected
            metadata = archive.read(f.metadata)
            metadata['prompt_ids'] = [1]
            dump(f.metadata, metadata)
            f.manifest['source_sha256'][str(f.metadata)] = archive.digest(f.metadata)
            with self.assertRaisesRegex(ValueError, 'metadata configuration'):
                self.verify_intermediates(f)

    def test_intermediate_rehashed_blob_still_must_match_independent_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.setup_intermediates(directory)
            original = f.blob.read_bytes()
            f.blob.write_bytes(b'\0' * len(original))
            with self.assertRaisesRegex(ValueError, 'blob hash/size'):
                self.verify_intermediates(f)
            f.manifest['reference_blob_sha256'] = archive.digest(f.blob)
            with self.assertRaisesRegex(ValueError, 'reconstruct'):
                self.verify_intermediates(f)

    def test_intermediate_npz_rejects_wrong_dtype_shape_missing_nonfinite_and_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.setup_intermediates(directory)
            key = f.manifest['entries'][0]['reference_keys'][0]
            float_key = f.manifest['entries'][1]['reference_keys'][0]
            cases = []
            bad = dict(f.data); bad[key] = bad[key].astype(np.float32); cases.append((bad, 'dtype'))
            bad = dict(f.data); bad[key] = bad[key].reshape(-1); cases.append((bad, 'shape'))
            bad = dict(f.data); del bad[key]; cases.append((bad, 'missing intermediate'))
            bad = dict(f.data); bad[float_key] = np.full_like(bad[float_key], np.nan); cases.append((bad, 'nonfinite'))
            bad = dict(f.data); bad['decode_0_model.layers.0_another_hidden'] = np.zeros(1); cases.append((bad, 'uncovered'))
            # Same flattened values and byte count do not establish a valid head view.
            projection_key = f.manifest['entries'][4]['reference_keys'][0]
            bad = dict(f.data); bad[projection_key] = bad[projection_key].reshape(1, 12); cases.append((bad, 'projection'))
            for data, message in cases:
                with self.subTest(message=message):
                    np.savez(f.arrays, **data)
                    f.manifest['source_sha256'][str(f.arrays)] = archive.digest(f.arrays)
                    with self.assertRaisesRegex(ValueError, message):
                        self.verify_intermediates(f)

    def test_intermediate_offsets_cannot_reorder_or_alias_the_reference_blob(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.setup_intermediates(directory)
            f.manifest['entries'][1]['offsets'][0] = 0
            with self.assertRaisesRegex(ValueError, 'offsets'):
                self.verify_intermediates(f)

    def test_optional_intermediate_manifest_is_bound_to_plan_and_uart(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory)
            self.assertIsNone(archive.intermediate_manifest(image, {}, b'[nr] RA returned: PASS'))
            for uart in (b'[intermediate] complete position=00000000 PASS', b'[intermediate broken',
                         b'intermediate] entry=00000000'):
                with self.assertRaisesRegex(ValueError, 'absent'):
                    archive.intermediate_manifest(image, {}, uart)
            with self.assertRaisesRegex(ValueError, 'missing intermediate'):
                archive.intermediate_manifest(image, {'intermediate_probe': {'claimed': True}}, b'')
            f = self.setup_intermediates(directory)
            dump(image/'intermediate-probe.json', f.manifest)
            self.assertEqual(archive.intermediate_manifest(image, {}, b''), f.manifest)
            self.assertEqual(archive.intermediate_manifest(image, {'intermediate_probe': f.manifest}, b''), f.manifest)
            with self.assertRaisesRegex(ValueError, 'plan intermediate'):
                archive.intermediate_manifest(image, {'intermediate_probe': None}, b'')


if __name__=='__main__':unittest.main()
