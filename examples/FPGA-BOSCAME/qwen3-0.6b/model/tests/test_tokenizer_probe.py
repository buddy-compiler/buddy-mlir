import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'tools/check_tokenizer_probe.py'
spec = importlib.util.spec_from_file_location('tokenizer_probe_verifier', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TokenizerProbeVerification(unittest.TestCase):
    def setUp(self):
        self.plan = {'fixtures':[{'case':0, 'expected_ids':[7], 'expected_decoded_hex':'61'}]}
        self.log = ('[tokenizer] case=00000000 count=00000001 ids=00000007\n'
                    '[tokenizer] case=00000000 decoded_hex=61\n'
                    '[tokenizer] case=00000000 encode_cycles=0000000000000001 decode_cycles=0000000000000002\n'
                    'verify tokenizer case 00000000: PASS\n'
                    '[tokenizer] checked=00000001 failures=00000000\n'
                    'verify tokenizer suite: PASS\n')

    def test_host_evidence_is_never_board_evidence(self):
        result = module.verify(self.log, self.plan, 'host')
        self.assertEqual(result['status'], 'PASS')
        self.assertFalse(result['board_tokenizer_verified'])
        self.assertEqual(module.verify(self.log, self.plan, 'fpga')['status'], 'FAIL')

    def test_pass_labels_do_not_hide_wrong_or_missing_results(self):
        for changed in (self.log.replace('ids=00000007', 'ids=00000008'),
                        self.log.replace('decoded_hex=61', 'decoded_hex=62'),
                        self.log.replace('count=00000001', 'count=00000002'),
                        self.log.replace('case=00000000 decoded_hex=61', 'missing')):
            self.assertEqual(module.verify(changed, self.plan, 'host')['status'], 'FAIL')

    def test_duplicate_results_and_failures_rejected(self):
        for changed in (self.log+self.log, self.log+'verify other: FAIL\n',
                        self.log+'[nr] TRAP mcause=2\n'):
            self.assertEqual(module.verify(changed, self.plan, 'host')['status'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
