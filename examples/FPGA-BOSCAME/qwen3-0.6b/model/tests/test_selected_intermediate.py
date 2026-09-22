"""Check selected full-model probes against real typed IR and shared producers."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest

MODEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL / 'tools'))
spec = importlib.util.spec_from_file_location('selected_probe', MODEL / 'tools/intermediate_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def arguments():
    b = MODEL / 'build/quant-opt/shared'
    ref = MODEL / 'build/console-fix-20260921/intermediate-reference-28l'
    return SimpleNamespace(layers=28, intermediate_layers='14,15', interactive=False,
        reference_arrays=ref / 'arrays.npz', report=b / 'replacement/triton-call-replacement.json',
        adapters=b / 'replacement/qwen_triton_adapters.c',
        graph_ir=b / 'nr-prefill/forward_prefill.ll', decode_ir=b / 'nr-decode/forward_decode.ll',
        intermediate_layout=MODEL / 'build/ame-v05/model-28l-cap128/import/weight-layout.json',
        intermediate_graph_dir=b / 'replacement', intermediate_arrays=ref / 'arrays.npz',
        intermediate_atol=.001, intermediate_mean_atol=.0001,
        decode_steps=8, prefill_len=16, cache_len=128,
        prompt_ids=[151644,872,198,3838,374,9625,30,151645,198,151644,77091,198,151667,271,151668,271])


class SelectedIntermediateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = arguments()
        if not cls.args.report.exists():
            raise unittest.SkipTest('requires full-model graph fixtures')
        cls.report = json.loads(cls.args.report.read_text())

    def test_shared_consumers_and_both_layer_outputs_covered(self):
        entries, _, _ = probe.build_mapping(self.args, self.report)
        self.assertEqual(len(entries), 134)
        for graph in ('prefill', 'decode'):
            current = [e for e in entries if e['graph'] == graph]
            self.assertEqual(len(current), 67)
            for layer in (14, 15):
                for part in ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                             'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'):
                    for suffix, phase, operand in (('_q','before',0), ('_a_scale','before',1), ('_out','after',3)):
                        key = f'model.layers.{layer}.{part}.weight' + suffix
                        matches = [e for e in current if e['reference_suffix'] == key
                                   and e['phase'] == phase and e['operand'] == operand]
                        self.assertEqual(len(matches), 1, key)
                        self.assertEqual((matches[0]['phase'], matches[0]['operand']), (phase, operand))
                self.assertTrue(any(e['reference_suffix'] == f'model.layers.{layer}_output_hidden'
                                    for e in current))
            self.assertTrue(any('model.layers.15_input_hidden' in e.get('reference_aliases', [])
                                for e in current))

    def test_layer_selection_and_actual_dataflow_fail_closed(self):
        for selection in ('14,14', '-1', '28', ''):
            args = copy.copy(self.args)
            args.intermediate_layers = selection
            with self.assertRaises(ValueError):
                probe.build_mapping(args, self.report)
        report = copy.deepcopy(self.report)
        for row in report['w8a8_calls']:
            if row['node'] == 'w8a8_mm_99_matmul_16x1024x1024':
                row['operands'][0] = 'wrong_shared_quantized_buffer'
        with self.assertRaisesRegex(ValueError, 'quantize producer'):
            probe.build_mapping(self.args, report)
        report = copy.deepcopy(self.report)
        report['w8a8_calls'] = [r for r in report['w8a8_calls']
                               if r['node'] != 'w8a8_mm_103_dequantize_16x3072']
        with self.assertRaisesRegex(ValueError, 'linear call'):
            probe.build_mapping(self.args, report)
        report = copy.deepcopy(self.report)
        norms = {r['node']: r for r in report['replaced']
                 if r['symbol'].startswith('qwen_graph_rmsnorm_16x1024__forward_prefill')}
        a, b = norms['mul_199'], norms['mul_209']
        a['operands'][1], b['operands'][1] = b['operands'][1], a['operands'][1]
        with self.assertRaisesRegex(ValueError, 'LLVM parameter identity'):
            probe.build_mapping(self.args, report)

    def test_pack_reference_aliases_and_schema(self):
        if not self.args.intermediate_arrays.exists():
            self.skipTest('requires independent full-model intermediate reference')
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, manifest = probe.generate_intermediate(self.args, Path(directory))
            self.assertEqual(manifest['schema_version'], 2)
            self.assertEqual(manifest['layers'], 28)
            self.assertEqual(manifest['selected_layers'], [14,15])
            self.assertEqual(manifest['reference_bytes'], 7767360)
            self.assertIn('partial', manifest['scope'])
            self.assertTrue(manifest['coverage_includes_aliases'])
            for entry in manifest['entries']:
                for key in entry['reference_keys']:
                    for alias in entry.get('reference_aliases', []):
                        self.assertNotIn(key.removesuffix(entry['reference_suffix']) + alias,
                                         manifest['uncovered_reference_tensors'])

    def test_adapter_argument_swaps_are_rejected(self):
        args = copy.copy(self.args)
        original = args.adapters.read_text()
        cases = (
            ('_mlir_ciface_kernel_matmul_16x1024x1024',
             '(MemRef2 *)a0, (MemRef2 *)a1, (MemRef2 *)a2',
             '(MemRef2 *)a1, (MemRef2 *)a0, (MemRef2 *)a2'),
            ('_mlir_ciface_kernel_quantize_16x1024',
             '(MemRef2 *)a0, (MemRef2 *)a1, (MemRef1 *)a2',
             '(MemRef2 *)a1, (MemRef2 *)a0, (MemRef1 *)a2'),
            ('_mlir_ciface_kernel_dequantize_16x1024',
             '(MemRef2 *)a0, (MemRef1 *)a1, (MemRef1 *)a2, (MemRef2 *)a3',
             '(MemRef2 *)a0, (MemRef1 *)a2, (MemRef1 *)a1, (MemRef2 *)a3'),
            ('_mlir_ciface_kernel_rmsnorm_16x1024',
             '(MemRef2 *)a1, (MemRef1 *)a2, &scratch, (MemRef2 *)a0',
             '(MemRef2 *)a0, (MemRef1 *)a2, &scratch, (MemRef2 *)a1'),
        )
        with tempfile.TemporaryDirectory() as directory:
            args.adapters = Path(directory) / 'adapters.c'
            for kernel, before, after in cases:
                with self.subTest(kernel=kernel):
                    old = kernel + '(' + before + ');'
                    self.assertIn(old, original)
                    args.adapters.write_text(original.replace(old, kernel + '(' + after + ');'))
                    with self.assertRaisesRegex(ValueError, 'argument order mismatch'):
                        probe.build_mapping(args, self.report)

    def test_sync_only_preserves_actual_calls_without_reference_reads(self):
        args = copy.copy(self.args)
        prior = MODEL / 'build/console-fix-20260921/selected-intermediates/image/intermediate-probe.json'
        if not prior.exists():
            self.skipTest('requires selected-layer probe manifest')
        args.boundary_sync_manifest = prior
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source, flags, manifest = probe.generate_boundary_sync(args, path)
            text = source.read_text()
            self.assertEqual(len(flags), 90)
            self.assertEqual(text.count('  ame_fence();'), 90)
            self.assertEqual(text.count('  __real_'), 90)
            for forbidden in ('check(', 'reference_raw', 'nr_puts', 'qwen_intermediate', 'rdcycle'):
                self.assertNotIn(forbidden, text)
            self.assertEqual(manifest['selected_layers'], [14,15])
            wrong = json.loads(prior.read_text())
            wrong['entries'][0]['reference_suffix'] += '_incorrect'
            wrong['entries'][0]['reference_keys'] = [wrong['entries'][0]['reference_keys'][0] + '_incorrect']
            args.boundary_sync_manifest = path / 'wrong.json'
            args.boundary_sync_manifest.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError, 'prior mapping differs'):
                probe.generate_boundary_sync(args, path)

    def test_typed_ssa_disagreement_is_rejected_even_when_report_agrees(self):
        args = copy.copy(self.args)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for kind in ('prefill', 'decode'):
                name = f'subgraph0_{kind}.triton.mlir'
                text = (args.intermediate_graph_dir / name).read_text()
                if kind == 'decode':
                    old = 'call @qwen_graph_w8a8_mm_99_matmul_1x1024x1024(%arg1332,'
                    self.assertIn(old, text)
                    text = text.replace(old, old.replace('%arg1332', '%arg1333'), 1)
                (path / name).write_text(text)
            args.intermediate_graph_dir = path
            with self.assertRaisesRegex(ValueError, 'SSA mismatch'):
                probe.build_mapping(args, self.report)


if __name__ == '__main__':
    unittest.main()
