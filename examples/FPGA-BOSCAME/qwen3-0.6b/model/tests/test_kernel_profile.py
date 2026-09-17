"""Independent graph/adapter timing expectations and adversarial UART checks."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
spec = importlib.util.spec_from_file_location('profile_check', TOOLS / 'check_kernel_profile.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
spec = importlib.util.spec_from_file_location('profile_generate', TOOLS / 'kernel_profile.py')
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def fixture():
    adapters = '''extern void _mlir_ciface_kernel_norm(MemRef2 *, MemRef2 *);
extern void _mlir_ciface_kernel_project(MemRef2 *, MemRef2 *);
void _mlir_ciface_qwen_graph_norm(MemRef2 *out, MemRef2 *in) {
  _mlir_ciface_kernel_norm(in, out);
}
void _mlir_ciface_qwen_graph_project(MemRef2 *out, MemRef2 *in) {
  _mlir_ciface_kernel_project(in, out);
  _mlir_ciface_kernel_project(in, out);
}
'''
    graphs = {}
    for kind in ('prefill', 'decode'):
        # Calls in wrappers must not be counted as additional entry calls.
        graphs[kind] = f'''define void @forward_{kind}() {{
  call void @qwen_graph_norm()
  call void @qwen_graph_norm()
  call void @qwen_graph_project()
  ret void
}}
define void @qwen_graph_norm() {{
  call void @_mlir_ciface_qwen_graph_norm()
  ret void
}}
define void @qwen_graph_project() {{
  call void @_mlir_ciface_qwen_graph_project()
  ret void
}}
'''
    report = {'graphs': {k: {'external_calls': 3} for k in graphs},
              'distinct_symbols': ['qwen_graph_norm', 'qwen_graph_project']}
    symbols = [checker.PREFIX + n for n in ('norm', 'project')]
    profile = {'adapter_sha256': checker.sha256(adapters.encode()),
               'kernels': [{'symbol': n, 'index': i} for i, n in enumerate(symbols)],
               'linker_flags': ['--wrap=' + n for n in symbols]}
    expected = {k: {n: 2 for n in symbols} for k in graphs}
    lines = []
    for p in (15, 16, 17):
        kind, start = ('prefill', 0) if p == 15 else ('decode', p)
        lines.append(f'[model] {kind} position={start:08X} token=0000002A '
                     'logit_bits=3F800000 compute_cycles=0000000000000064')
        for n in symbols:
            lines.append(f'[profile] position={p:08X} kernel={n} '
                         'calls=0000000000000002 cycles=0000000000000020')
    lines.append('[nr] RA returned: PASS')
    return profile, report, adapters, graphs, expected, '\n'.join(lines) + '\n'


class KernelProfileTests(unittest.TestCase):
    def test_progress_pairs_and_counts_with_missing_duplicate_and_unknown_records(self):
        symbol = checker.PREFIX + 'norm'
        expected = {'prefill': {symbol: 1}, 'decode': {symbol: 1}}
        begin = '[model] prefill begin position=00000000 input_token=0000002A\n'
        call = f'[kernel] begin {symbol} call=0000000000000000\n'
        end = f'[kernel] end {symbol} call=0000000000000000\n'
        stage = ('[model] prefill position=00000000 token=0000002A '
                 'logit_bits=3F800000 compute_cycles=0000000000000064\n')
        good = begin + call + end + stage
        self.assertEqual(checker.check_progress(good, expected, enabled=True), [])
        for text in (begin + call + stage, begin + call + end + call + end + stage,
                     good.replace(symbol, symbol + '_wrong'), begin + stage,
                     begin + call + end, good.replace('call=0000000000000000', 'call=0000000000000001')):
            with self.subTest(text=text):
                self.assertTrue(checker.check_progress(text, expected, enabled=True))
        self.assertTrue(checker.check_progress(good, expected, enabled=False))

    def test_actual_entry_and_adapter_calls_not_bridge_or_scratch_double_counted(self):
        profile, report, adapters, irs, expected, _ = fixture()
        got, proof = checker.expected_counts(profile, report, adapters, irs)
        self.assertEqual(got, expected)
        self.assertEqual(proof['prefill']['total_raw_calls'], 3)
        self.assertEqual(proof['prefill']['expected_kernel_calls'], 4)
        self.assertTrue(proof['prefill']['unconditional_acyclic_calls_verified'])

    def test_rejects_insufficient_mismatched_and_conditional_count_evidence(self):
        profile, report, adapters, irs, _, _ = fixture()
        mutants = []
        p = copy.deepcopy(profile); p['adapter_sha256'] = '0' * 64
        mutants.append(('hash', p, report, adapters, irs))
        p = copy.deepcopy(profile); p['kernels'].append(p['kernels'][0])
        mutants.append(('duplicate_symbol', p, report, adapters, irs))
        p = copy.deepcopy(profile); p['linker_flags'].pop()
        mutants.append(('missing_wrap', p, report, adapters, irs))
        r = copy.deepcopy(report); r['distinct_symbols'].pop()
        mutants.append(('report_symbols', profile, r, adapters, irs))
        r = copy.deepcopy(report); r['graphs']['prefill']['external_calls'] = 2
        mutants.append(('report_count', profile, r, adapters, irs))
        r = copy.deepcopy(report); del r['graphs']['prefill']['external_calls']
        mutants.append(('missing_report_count', profile, r, adapters, irs))
        i = copy.deepcopy(irs); i['prefill'] = i['prefill'].replace(
            'call void @_mlir_ciface_qwen_graph_norm()', 'call void @wrong()')
        mutants.append(('wrong_bridge', profile, report, adapters, i))
        a = adapters.replace('  _mlir_ciface_kernel_norm(in, out);',
                             '  if (in) _mlir_ciface_kernel_norm(in, out);')
        p = copy.deepcopy(profile); p['adapter_sha256'] = checker.sha256(a.encode())
        mutants.append(('conditional_adapter', p, report, a, irs))
        a = adapters.replace('  _mlir_ciface_kernel_norm(in, out);',
                             '  return; _mlir_ciface_kernel_norm(in, out);')
        p = copy.deepcopy(profile); p['adapter_sha256'] = checker.sha256(a.encode())
        mutants.append(('early_return_adapter', p, report, a, irs))
        for name, p, r, a, i in mutants:
            with self.subTest(name=name), self.assertRaises(ValueError):
                checker.expected_counts(p, r, a, i)

    def test_cfg_rejects_conditional_calls_and_loops_but_allows_other_loops(self):
        conditional = '''  br i1 %a, label %yes, label %done
yes:
  call void @qwen_graph_norm()
  br label %done
done:
  ret void
'''
        loop = '''  br label %again
again:
  call void @qwen_graph_norm()
  br i1 %a, label %again, label %done
done:
  ret void
'''
        for body in (conditional, loop):
            with self.assertRaises(ValueError):
                checker.graph_calls(body)
        outside_loop = '''  call void @qwen_graph_norm()
  br label %again
again:
  %a = add i32 1, 1
  br i1 %stop, label %again, label %done
done:
  call void @qwen_graph_project()
  ret void
'''
        self.assertEqual(checker.graph_calls(outside_loop),
                         {'qwen_graph_norm': 1, 'qwen_graph_project': 1})

    def test_uart_missing_duplicate_unknown_wrong_counts_cycles_and_positions(self):
        _, _, _, _, expected, log = fixture()
        good = checker.check(log, expected, steps=2)
        self.assertEqual(good['status'], 'KERNEL_PROFILE_PASS')
        self.assertEqual(good['stages'][0]['kernel_calls'], 4)
        self.assertEqual(good['stages'][0]['kernel_cycles'], 64)
        self.assertEqual(good['stages'][0]['graph_cycles_outside_kernel_measurements'], 36)
        kernel = next(x for x in log.splitlines() if x.startswith('[profile]'))
        stage = next(x for x in log.splitlines() if x.startswith('[model] decode'))
        variants = {
            'missing_kernel': log.replace(kernel + '\n', '', 1),
            'duplicate_kernel': log.replace(kernel, kernel + '\n' + kernel, 1),
            'unknown_kernel': log.replace('kernel=_mlir_ciface_kernel_norm',
                                          'kernel=_mlir_ciface_kernel_unknown', 1),
            'wrong_calls': log.replace('calls=0000000000000002', 'calls=0000000000000003', 1),
            'zero_calls': log.replace('calls=0000000000000002', 'calls=0000000000000000', 1),
            'zero_cycles': log.replace('cycles=0000000000000020', 'cycles=0000000000000000', 1),
            'negative_cycles': log.replace('cycles=0000000000000020', 'cycles=-000000000000020', 1),
            'cycles_over_graph': log.replace('cycles=0000000000000020', 'cycles=0000000000000070', 1),
            'zero_graph': log.replace('compute_cycles=0000000000000064', 'compute_cycles=0000000000000000', 1),
            'wrong_profile_pos': log.replace('[profile] position=0000000F', '[profile] position=00000000', 1),
            'missing_stage': log.replace(stage + '\n', '', 1),
            'duplicate_stage': log.replace(stage, stage + '\n' + stage, 1),
            'prefill_wrong_semantics': log.replace('prefill position=00000000', 'prefill position=0000000F'),
            'no_completion': log.replace('[nr] RA returned: PASS', ''),
            'failed_runtime': log + 'verify NR runtime: FAIL\n',
            'nonzero_status': log + '[nr] launch cycles=0x1 status=0x1\n',
            'after_completion': log + kernel + '\n',
        }
        for name, bad in variants.items():
            with self.subTest(name=name):
                self.assertEqual(checker.check(bad, expected, steps=2)['status'], 'NOT_ACCEPTED')

    def test_cli_source_hash_and_exit_status(self):
        profile, report, adapters, irs, _, log = fixture()
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            (d / 'adapters.c').write_text(adapters)
            source, _ = generator.generate_profile(d / 'adapters.c', d)
            (d / 'replacement.json').write_text(json.dumps(report))
            (d / 'uart.log').write_text(log)
            for kind, text in irs.items(): (d / (kind + '.ll')).write_text(text)
            argv = ['check', '--uart', str(d/'uart.log'), '--profile', str(d/'kernel-profile.json'),
                    '--replacement', str(d/'replacement.json'), '--adapters', str(d/'adapters.c'),
                    '--prefill-ir', str(d/'prefill.ll'), '--decode-ir', str(d/'decode.ll'),
                    '--steps', '2', '--output', str(d/'out.json')]
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(checker.main(), 0)
            self.assertEqual(len(json.loads((d/'out.json').read_text())['inputs_sha256']), 7)
            source.write_text(source.read_text() + '/* changed */\n')
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(checker.main(), 1)
            self.assertIn('source hash mismatch', json.loads((d/'out.json').read_text())['errors'][0])

    def test_complete_phase_telemetry_must_sum_and_fit_total(self):
        _, _, _, _, expected, log = fixture()
        phase = ('preparation_cycles=0000000000000001 selection_cycles=0000000000000002 '
                 'cache_retention_cycles=0000000000000003 model_cycles=000000000000006A '
                 'total_with_uart_cycles=0000000000000080 scratch_bytes=0000000000001000')
        log = log.replace('compute_cycles=0000000000000064',
                          'compute_cycles=0000000000000064 ' + phase)
        report = checker.check(log, expected, steps=2)
        self.assertEqual(report['status'], 'KERNEL_PROFILE_PASS')
        self.assertEqual(report['stages'][0]['model_cycles'], 106)
        for bad in (log.replace('model_cycles=000000000000006A', 'model_cycles=000000000000006B', 1),
                    log.replace('total_with_uart_cycles=0000000000000080', 'total_with_uart_cycles=0000000000000050', 1),
                    log.replace('selection_cycles=0000000000000002 ', '', 1)):
            self.assertEqual(checker.check(bad, expected, steps=2)['status'], 'NOT_ACCEPTED')


if __name__ == '__main__': unittest.main()
