"""Boundary-only diagnostics must preserve model entry code and link evidence."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_image_abi as fixture

image = fixture.image
MODEL = fixture.MODEL
sys.path.insert(0, str(MODEL / 'tools'))
import intermediate_probe


def prior_manifest():
    entries = []
    cursor = 0
    for index, graph in enumerate(('prefill', 'decode')):
        suffix = 'model.layers.0_q_proj'
        count = 1 if graph == 'prefill' else 8
        entries.append({'index': index, 'graph': graph, 'layer': 0,
            'symbol': '_mlir_ciface_qwen_graph_' + graph, 'operand': 0, 'phase': 'after',
            'shape': [1, 2], 'rank': 2, 'elements': 2, 'dtype': 'f32', 'transform': 'identity',
            'reference_suffix': suffix,
            'reference_keys': [('prefill' if graph == 'prefill' else f'decode_{i}') + '_' + suffix
                               for i in range(count)],
            'offsets': [cursor + i * 2 for i in range(count)]})
        cursor += count * 2
    return {'schema_version': 2, 'layers': 1, 'prefill_len': 16, 'decode_steps': 8,
            'selected_layers': [0], 'scope': 'selected test boundaries', 'limitations': ['test fixture'],
            'max_abs_tolerance': 0.001, 'mean_abs_tolerance': 0.0001, 'entries': entries,
            'reference_bytes': cursor * 4,
            'linker_flags': ['--wrap=' + entry['symbol'] for entry in sorted(entries, key=lambda e: e['symbol'])],
            'generated_source_sha256': '1' * 64, 'reference_blob_sha256': '2' * 64,
            'source_sha256': {'original-adapter.c': '3' * 64}, 'uncovered_reference_tensors': []}


class ImageBoundarySyncTests(unittest.TestCase):
    def fixture(self, directory):
        args, report, segment = fixture.ImageABI().setup_case(directory, 1)
        args.boundary_sync_manifest = Path(directory) / 'prior-intermediate-probe.json'
        args.boundary_sync_manifest.write_text(json.dumps(prior_manifest()))
        return args, report, segment

    def test_selected_sync_preserves_generated_entry_and_default_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            args, report, segment = self.fixture(directory)
            source, plan = image.generate(report, segment, args)
            original = args.boundary_sync_manifest
            args.boundary_sync_manifest = None
            baseline, baseline_plan = image.generate(report, segment, args)
            self.assertEqual(source, baseline)
            self.assertNotIn('qwen_intermediate_', source)
            self.assertNotIn('intermediate_reference', source)
            request = plan.pop('boundary_sync_requested')
            self.assertEqual(plan, baseline_plan)
            self.assertEqual(request['manifest_sha256'], hashlib.sha256(original.read_bytes()).hexdigest())

    def test_mutually_exclusive_instrumentation_and_invalid_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            args, report, segment = self.fixture(directory)
            for name, value in (('profile_kernels', True), ('intermediate_arrays', Path('unused.npz')),
                                ('hang_watch', '_mlir_ciface_kernel_matmul_1x1024x2048:14')):
                setattr(args, name, value)
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'cannot combine'):
                    image.generate(report, segment, args)
                setattr(args, name, None)
            manifest = prior_manifest()
            manifest['layers'] = 28
            args.boundary_sync_manifest.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'dimensions differ'):
                image.generate(report, segment, args)
            manifest = prior_manifest()
            manifest['schema_version'] = 1
            args.boundary_sync_manifest.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'requires selected-layer schema_version 2'):
                image.generate(report, segment, args)
            args.boundary_sync_manifest.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'schema_version'):
                image.generate(report, segment, args)
            args.boundary_sync_manifest = Path(directory)/'absent.json'
            with self.assertRaisesRegex(ValueError, 'existing non-symlink'):
                image.generate(report, segment, args)

    def test_build_links_helper_records_plan_and_exact_input_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args, report, segment = self.fixture(directory)
            args.repo_root = MODEL.parents[3]
            args.uart_probe = False
            args.reference_arrays = None
            args.adapters = output/'adapters.c'
            args.adapters.write_text('/* graph adapters fixture */\n')
            args.archive = output/'lib.a'
            args.archive.write_bytes(b'archive fixture')
            source, plan = image.generate(report, segment, args)
            (output/'model_main.c').write_text(source)
            (output/'w8a8-image-plan.json').write_text(json.dumps(plan))
            generated = output/'boundary-sync.c'
            flags = ['--wrap=_mlir_ciface_qwen_graph_prefill']
            manifest = {'schema_version': 1, 'linker_flags': flags,
                        'scope': 'sync-only fixture', 'reference_bytes': 0}

            def generate_boundary_sync(actual_args, actual_output):
                self.assertIs(actual_args, args)
                self.assertEqual(actual_output, output)
                generated.write_text('/* wrapper fixture, no model arithmetic */\n')
                return generated, flags, manifest

            commands = []

            def run(command, **kwargs):
                commands.append(command)
                if '-o' in command:
                    Path(command[command.index('-o') + 1]).write_bytes(b'object fixture')
                if Path(command[0]).name == 'llvm-objcopy':
                    Path(command[-1]).write_bytes(b'object fixture')
                if kwargs.get('stdout') is not None:
                    kwargs['stdout'].write(b'assembly fixture\n')
                return SimpleNamespace(returncode=0, stdout='', stderr='')

            with patch.object(intermediate_probe, 'generate_boundary_sync', generate_boundary_sync, create=True), \
                    patch.object(image.subprocess, 'run', side_effect=run), \
                    patch.object(image, 'resolve_linker', return_value={'path': '/fixture/ld.lld'}), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(image.build(args, output), 0)
            actual_plan = json.loads((output/'w8a8-image-plan.json').read_text())
            self.assertEqual(actual_plan['boundary_sync'], manifest)
            record = json.loads((output/'image.json').read_text())
            for path in (args.boundary_sync_manifest, generated, Path(image.__file__).with_name('intermediate_probe.py')):
                self.assertEqual(record['input_sha256'][str(path)], hashlib.sha256(path.read_bytes()).hexdigest())
            compile_commands = [command for command in commands if str(generated) in command]
            self.assertEqual(len(compile_commands), 1)
            self.assertIn(str(output/'boundary-sync.o'), compile_commands[0])
            link = next(command for command in commands if command[0] == '/fixture/ld.lld')
            self.assertIn(flags[0], link)
            self.assertIn(str(output/'boundary-sync.o'), link)
            self.assertFalse(any('intermediate-reference' in part or 'intermediate-probe.c' in part
                                 for command in commands for part in command))

    def test_cli_help_and_dry_run_report_requested_mode(self):
        result = subprocess.run([sys.executable, image.__file__, '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--boundary-sync-manifest', result.stdout)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args, report, segment = self.fixture(directory)
            (output/'report.json').write_text(json.dumps(report))
            (output/'segment.json').write_text(json.dumps(segment))
            command = [sys.executable, image.__file__, '--dry-run', '--repo-root', str(MODEL.parents[3]),
                '--report', str(output/'report.json'), '--segment', str(output/'segment.json'),
                '--graph-ir', str(args.graph_ir), '--decode-ir', str(args.decode_ir),
                '--archive', str(output/'unused.a'), '--adapters', str(output/'unused.c'),
                '--output', str(output/'generated'), '--layers', '1', '--cache-len', '32',
                '--prompt-ids', ','.join(map(str, range(16))),
                '--boundary-sync-manifest', str(args.boundary_sync_manifest)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            plan = json.loads((output/'generated/w8a8-image-plan.json').read_text())
            self.assertIn('boundary_sync_requested', plan)
            self.assertNotIn('boundary_sync', plan)  # No compiled wrapper exists in a dry run.


if __name__ == '__main__':
    unittest.main()
