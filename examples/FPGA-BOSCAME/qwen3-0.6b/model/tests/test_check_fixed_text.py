"""Official-tokenizer post-run verification and adversarial UART framing tests."""
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

MODEL = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(MODEL/'tools'))
spec=importlib.util.spec_from_file_location('fixed_check',MODEL/'tools/check_fixed_text.py')
checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)


def fixture(oracle, tokens=None):
    tokens = tokens or [32,33,151645,34,35,36,37,38,39]
    text='What is France?';raw=text.encode();ids=oracle.prompt_ids(text,False)
    chunks,finish,output=oracle.incremental_decode(tokens)
    plan={'text_mode':'fixed-validation','fixed_prompt':{'utf8_hex':raw.hex(),
        'bytes':len(raw),'sha256':checker.digest(raw),'thinking':False,'expected_ids':ids,
        'prefill_calls':1,'decode_calls':8,'predictions_recorded':9,'output_capacity':65536}}
    lines=[b'[nr] startup',b'[text] prompt count=00000010 ids='+b' '.join(f'{x:08X}'.encode() for x in ids)+
           b' encode_cycles=0000000000000123',b'verify fixed prompt tokenizer: PASS',checker.MODE]
    for i,(token,chunk) in enumerate(zip(tokens,chunks)):
        kind,pos=('prefill',0) if i==0 else ('decode',15+i)
        incoming=ids[0] if i==0 else tokens[i-1]
        lines += [f'[model] {kind} begin position={pos:08X} input_token={incoming:08X}'.encode(),
          f'[model] {kind} position={pos:08X} token={token:08X} logit_bits=3F800000 compute_cycles=0000000000000064'.encode(),
          f'[text] prediction={i:08X} token={token:08X} eos={int(token in oracle.eos):08X} bytes={len(chunk):08X} hex={chunk.hex()}'.encode()]
    lines += [b'[text] finish hex='+finish.hex().encode(),
              f'[text] output bytes={len(output):08X} hex={output.hex()}'.encode(),b'[text] BEGIN']
    log=b'\r\n'.join(lines)+b'\r\n'+output+b'\r\n[text] END\r\nverify fixed text validation: PASS\r\n'
    log+=b'[nr] launch cycles=0x1000 status=0x0\r\n[nr] RA returned: PASS\r\nverify NR runtime: PASS\r\n'
    return plan,log


class FixedTextCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets=MODEL/'assets/official'
        if not (cls.assets/'tokenizer.json').is_file():raise unittest.SkipTest('official tokenizer assets unavailable')
        cls.oracle=checker.OfficialOracle(cls.assets)

    def test_official_prompt_model_tokens_and_skip_special(self):
        plan,log=fixture(self.oracle)
        result=checker.verify(log,plan,self.oracle)
        self.assertEqual(result['status'],'FIXED_TEXT_PASS')
        self.assertEqual(result['output_text'],'ABCDEFGH')
        self.assertEqual(result['predictions'][2]['eos'],1)
        self.assertEqual(result['predictions'][2]['bytes'],0)
        self.assertEqual(result['actual_prompt']['count'],16)

    def test_utf8_can_span_tokens_and_finish_with_incomplete_sequence(self):
        inverse={b:c for c,b in self.oracle.byte_inverse.items()}
        raw=[0xc3,0xa9,0xe4,0xb8,0xad,0xff,65,0xe2,0x82]
        tokens=[self.oracle.tokenizer.convert_tokens_to_ids(inverse[b]) for b in raw]
        plan,log=fixture(self.oracle,tokens)
        result=checker.verify(log,plan,self.oracle)
        self.assertEqual(result['output_text'],'é中�A�')
        self.assertEqual(result['predictions'][0]['bytes'],0)
        self.assertEqual(result['predictions'][1]['hex'],'c3a9')
        self.assertEqual(result['finish_hex'],'efbfbd')

    def test_rejects_trace_and_framing_mutations(self):
        plan,log=fixture(self.oracle)
        prediction=next(l for l in log.split(b'\r\n') if l.startswith(b'[text] prediction='))
        begin=next(l for l in log.split(b'\r\n') if l.startswith(b'[model] decode begin'))
        prompt=next(l for l in log.split(b'\r\n') if l.startswith(b'[text] prompt'))
        variants={
            'missing_prediction':log.replace(prediction+b'\r\n',b'',1),
            'duplicate_prediction':log.replace(prediction,prediction+b'\r\n'+prediction,1),
            'unknown_text':log+b'[text] surprise\r\n',
            'malformed_prediction':log.replace(b'prediction=00000000',b'prediction=bad',1),
            'wrong_prediction_token':log.replace(b'prediction=00000000 token=00000020',b'prediction=00000000 token=00000021',1),
            'wrong_incremental_bytes':log.replace(b'bytes=00000001 hex=41',b'bytes=00000001 hex=42',1),
            'wrong_incremental_length':log.replace(b'bytes=00000001 hex=41',b'bytes=00000002 hex=41',1),
            'wrong_eos':log.replace(b'eos=00000001',b'eos=00000000',1),
            'wrong_decode_input':log.replace(begin,begin[:-8]+b'00000021',1),
            'missing_begin':log.replace(begin+b'\r\n',b'',1),
            'wrong_prefill_position':log.replace(b'prefill position=00000000',b'prefill position=0000000F',1),
            'wrong_prefill_input':log.replace(b'input_token=0002505C',b'input_token=00000000',1),
            'duplicate_prompt':log.replace(prompt,prompt+b'\r\n'+prompt,1),
            'wrong_prompt_count':log.replace(b'prompt count=00000010',b'prompt count=0000000F',1),
            'wrong_prompt_id':log.replace(b'ids=0002505C',b'ids=00000001',1),
            'wrong_output_length':log.replace(b'output bytes=00000008',b'output bytes=00000009'),
            'wrong_raw_bytes':log.replace(b'\r\nABCDEFGH\r\n',b'\r\nABCDEFGX\r\n'),
            'bad_frame_end':log.replace(b'[text] END',b'[text] ENX',1),
            'wrong_finish':log.replace(b'[text] finish hex=',b'[text] finish hex=41',1),
            'missing_runtime':log.replace(b'verify NR runtime: PASS\r\n',b''),
            'duplicate_runtime':log+b'[nr] RA returned: PASS\r\n',
            'firmware_fail':log+b'[nr] fail: FAIL\r\n',
            'nonzero_runtime':log.replace(b'status=0x0',b'status=0x1'),
            'truncated':log[:-1],
        }
        for name,bad in variants.items():
            with self.subTest(name=name):
                self.assertNotEqual(log,bad)
                with self.assertRaises(ValueError):checker.verify(bad,plan,self.oracle)

    def test_rejects_manifest_tampering(self):
        plan,log=fixture(self.oracle)
        for key,value in [('sha256','0'*64),('bytes',99),('thinking',True),
                          ('prefill_calls',2),('decode_calls',7),('predictions_recorded',8),
                          ('expected_ids',[1]*16),('output_capacity',1)]:
            bad=copy.deepcopy(plan);bad['fixed_prompt'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):checker.verify(log,bad,self.oracle)

    def test_payload_marker_text_is_not_parsed_as_control(self):
        class MarkersOracle:
            eos={151643,151645}
            def prompt_ids(inner,text,thinking):return self.oracle.prompt_ids(text,thinking)
            def incremental_decode(inner,tokens):
                payload=b'\r\n[text] END\r\n[nr] RA returned: PASS\r\nFAIL\x00'+ '中'.encode()
                return [payload]+[b'']*8,b'',payload
        oracle=MarkersOracle();plan,log=fixture(oracle)
        result=checker.verify(log,plan,oracle)
        self.assertEqual(result['status'],'FIXED_TEXT_PASS')
        self.assertIn('FAIL\x00',result['output_text'])

    def test_cli_failure_is_nonzero_and_records_hashes(self):
        plan,log=fixture(self.oracle)
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);(d/'plan.json').write_text(json.dumps(plan));(d/'uart.log').write_bytes(log)
            argv=['check','--uart',str(d/'uart.log'),'--image-plan',str(d/'plan.json'),
                  '--assets',str(self.assets),'--output',str(d/'out.json')]
            with patch.object(checker,'OfficialOracle',return_value=self.oracle),patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(checker.main(),0)
                result=json.loads((d/'out.json').read_text());self.assertEqual(len(result['official_assets_sha256']),3)
                (d/'uart.log').write_bytes(log[:-1]);self.assertEqual(checker.main(),1)
                self.assertEqual(json.loads((d/'out.json').read_text())['status'],'NOT_ACCEPTED')


if __name__=='__main__':unittest.main()
