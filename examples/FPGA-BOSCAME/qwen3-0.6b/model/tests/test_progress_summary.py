"""Check progress mapping against graph order and optional post-call evidence."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'optimization/summarize_progress.py'
spec = importlib.util.spec_from_file_location('progress_summary', SCRIPT)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def decoder(shared=False):
    block = '''rmsnorm_1x1024 quantize_1x1024 matmul_1x2048x1024
        dequantize_1x2048 rmsnorm_16x128 quantize_1x1024 matmul_1x1024x1024
        dequantize_1x1024 rmsnorm_8x128 quantize_1x1024 matmul_1x1024x1024
        dequantize_1x1024 layout_context_8x1x128 kv_cache_update_position_1x8x128_cap128
        layout_context_8x1x128 kv_cache_update_position_1x8x128_cap128
        attention_qk_position_native_16x1x128x128 attention_scale_mask_position_16x1x128
        softmax_16x1x128 attention_pv_position_16x1x128x128 quantize_1x2048
        matmul_1x1024x2048 dequantize_1x1024 rmsnorm_1x1024 quantize_1x1024
        matmul_1x3072x1024 dequantize_1x3072 silu_1x3072 quantize_1x1024
        matmul_1x3072x1024 dequantize_1x3072 quantize_1x3072
        matmul_1x1024x3072 dequantize_1x1024'''.split()
    return [name for i, name in enumerate(block) if not shared or i not in (5, 9, 28)]


def graph_names(shared=False, layers=28):
    return ['embedding_w8a8_1x1024'] + decoder(shared) * layers + [
        'rmsnorm_1x1024', 'quantize_1x1024', 'matmul_1x151936x1024',
        'dequantize_1x151936']


def write_build(build, names):
    symbols = sorted(set(names))
    replacement = build / 'replacement'
    replacement.mkdir()
    (replacement / 'qwen_triton_adapters.c').write_text('\n'.join(
        f'void _mlir_ciface_qwen_graph_{name}(void) {{\n'
        f'  {summary.PREFIX}{name}();\n}}' for name in symbols))
    (replacement / 'triton-call-replacement.json').write_text(json.dumps({
        'distinct_symbols': ['qwen_graph_' + name for name in symbols],
        'graphs': {kind: {'external_calls': len(names)} for kind in ('prefill', 'decode')},
    }))
    for kind in ('prefill', 'decode'):
        folder = build / ('nr-' + kind)
        folder.mkdir()
        body = '\n'.join(f'  call void @qwen_graph_{name}()' for name in names)
        bridges = '\n'.join(f'define void @qwen_graph_{name}() {{\n'
                            f'  call void @_mlir_ciface_qwen_graph_{name}()\n'
                            '  ret void\n}' for name in symbols)
        (folder / f'forward_{kind}.ll').write_text(
            f'define void @forward_{kind}() {{\n{body}\n  ret void\n}}\n{bridges}\n')


class ProgressSummaryTests(unittest.TestCase):
    def test_actual_graph_structure_selects_shared_or_legacy_width(self):
        for shared, width, total in ((False, 34, 957), (True, 31, 873)):
            with self.subTest(shared=shared), tempfile.TemporaryDirectory() as directory:
                build = Path(directory)
                write_build(build, graph_names(shared))
                sequences, hashes = summary.expected_sequence(build, 28)
                self.assertEqual(len(hashes), 4)
                for seq in sequences.values():
                    self.assertEqual(len(seq), total)
                    self.assertEqual(seq[width]['decoder_block_index'], 0)
                    self.assertEqual(seq[width + 1]['decoder_block_index'], 1)
                    self.assertEqual(seq[width + 1]['call_in_decoder_block'], 0)
                    self.assertEqual(seq[-5]['decoder_block_index'], 27)
                    self.assertEqual(seq[-5]['call_in_decoder_block'], width - 1)
                    self.assertEqual(seq[-4]['region'], 'final_norm_lm_head')

    def test_rejects_changed_call_count_order_and_nonidentical_layers(self):
        valid = graph_names(True)
        bad_order = valid.copy()
        bad_order[2], bad_order[3] = bad_order[3], bad_order[2]
        different_layer = valid.copy()
        different_layer[32] = 'rmsnorm_1x2048'
        wrong_head = valid.copy()
        wrong_head[-2] = 'matmul_1x1024x1024'
        for names in (valid[:-1], bad_order, different_layer, wrong_head):
            with self.subTest(names=names[:4]):
                with self.assertRaises(ValueError):
                    summary.decoder_block_width(names, 28)

    def test_rejects_conditional_kernel_calls(self):
        with self.assertRaises(ValueError):
            summary.ordered_calls('''
  br i1 %condition, label %yes, label %no
yes:
  call void @qwen_graph_norm()
  br label %done
no:
  br label %done
done:
  ret void
''')

    def fixture(self):
        symbol = summary.PREFIX + 'norm'
        seq = {'index': 0, 'kernel': symbol, 'symbol_call_index': 0}
        sequences = {'prefill': [seq], 'decode': [seq]}
        begin = '[model] decode begin position=00000010 input_token=00000001\n'
        call = f'[kernel] begin {symbol} call=0000000000000000\n'
        returned = f'[kernel-phase] returned {symbol} call=0000000000000000\n'
        end = f'[kernel] end {symbol} call=0000000000000000\n'
        stage = ('[model] decode position=00000010 token=00000002 '
                 'logit_bits=3F800000 compute_cycles=0000000000000001\n')
        return sequences, begin, call, returned, end, stage

    def test_returned_narrows_unpaired_begin_without_claiming_end(self):
        sequences, begin, call, returned, end, stage = self.fixture()
        before = summary.summarize(begin + call, sequences)
        self.assertFalse(before['unpaired_begin']['real_kernel_return_observed'])
        after = summary.summarize(begin + call + returned, sequences)
        self.assertEqual(after['errors'], [])
        self.assertTrue(after['unpaired_begin']['real_kernel_return_observed'])
        self.assertEqual(after['unpaired_begin']['real_kernel_return_uart_line'], 3)
        self.assertEqual(after['graph_stages'][0]['completed_kernel_calls'], 0)
        self.assertFalse(after['graph_stages'][0]['graph_return_observed'])
        done = summary.summarize(begin + call + returned + end + stage, sequences)
        self.assertEqual(done['errors'], [])
        self.assertIsNone(done['unpaired_begin'])
        self.assertTrue(done['graph_stages'][0]['graph_return_observed'])

    def test_legacy_end_and_partial_return_line_remain_supported(self):
        sequences, begin, call, returned, end, stage = self.fixture()
        legacy = summary.summarize(begin + call + end + stage, sequences)
        self.assertEqual(legacy['errors'], [])
        self.assertFalse(legacy['graph_stages'][0]['last_completed_kernel'][
            'real_kernel_return_observed'])
        partial = summary.summarize(begin + call + returned[:-3], sequences)
        self.assertEqual(partial['errors'], [])
        self.assertFalse(partial['unpaired_begin']['real_kernel_return_observed'])

    def test_rejects_unmatched_duplicate_or_malformed_return_markers(self):
        sequences, begin, call, returned, end, _ = self.fixture()
        for log in (begin + returned, begin + call + returned + returned,
                    begin + call + end + returned,
                    begin + call + returned.replace('norm', 'wrong'),
                    begin + call + returned.replace('call=0000000000000000',
                                                    'call=0000000000000001'),
                    begin + call + '[kernel-phase] returned malformed\n'):
            with self.subTest(log=log):
                self.assertEqual(summary.summarize(log, sequences)['status'],
                                 'PROGRESS_SEQUENCE_ERROR')

    def test_retains_latest_graph_boundary_without_inferring_return(self):
        sequences, begin, call, _, _, _ = self.fixture()
        log = begin + '[graph-phase] graph call begin\n' + call
        result = summary.summarize(log, sequences)
        self.assertEqual(result['last_graph_phase'], {
            'phase': 'graph call begin', 'uart_line': 2, 'kind': 'decode', 'position': 16})
        self.assertFalse(result['graph_stages'][0]['graph_return_observed'])

    def test_summary_limits_match_selected_synchronization(self):
        sequences, begin, call, returned, _, _ = self.fixture()
        log = begin + call + returned
        legacy = summary.summarize(log, sequences)
        self.assertEqual(legacy['completion_sync'], 'ame-resync')
        self.assertIn('public ame_fence resync', ' '.join(legacy['limits']))
        fence = summary.summarize(log, sequences, completion_sync='fence')
        self.assertEqual(fence['completion_sync'], 'fence')
        self.assertNotIn('public ame_fence', ' '.join(fence['limits']))
        self.assertIn('fence rw,rw', ' '.join(fence['limits']))
        self.assertIn('does not by itself prove AME completion', ' '.join(fence['limits']))
        with self.assertRaisesRegex(ValueError, 'unknown profiler completion_sync'):
            summary.summarize(log, sequences, completion_sync='none')


if __name__ == '__main__':
    unittest.main()
