"""Adversarial UART protocol tests; fixture text is not FPGA evidence."""
import hashlib
import importlib.util
from pathlib import Path
import unittest
import sys

MODEL=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(MODEL/'tools'))
spec=importlib.util.spec_from_file_location('fixed_text_check',MODEL/'tools/check_fixed_text.py')
checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)


def fixture(payload_chunks=None):
    prompt=b'What is France?';ids=list(range(16));tokens=[32,33,151645,34,35,36,37,38,39]
    chunks=payload_chunks or [b'A',b'',b'',b'\xe4\xb8\xad',b'B',b'\r\n',b'C',b'D',b'E']
    payload=b''.join(chunks)
    fixed={'utf8_hex':prompt.hex(),'sha256':hashlib.sha256(prompt).hexdigest(),'bytes':len(prompt),
           'thinking':False,'expected_ids':ids,'decode_calls':8,'prefill_calls':1,
           'predictions_recorded':9,'output_capacity':65536}
    plan={'text_mode':'fixed-validation','fixed_prompt':fixed}
    lines=['[text] prompt count=00000010 ids='+' '.join(f'{i:08X}' for i in ids)+' encode_cycles=0000000000000001',
           'verify fixed prompt tokenizer: PASS',
           '[text] mode=fixed-validation; EOS is recorded, never early-stops']
    for i,(token,chunk) in enumerate(zip(tokens,chunks)):
        kind='prefill' if i==0 else 'decode';position=0 if i==0 else 15+i
        incoming=ids[0] if i==0 else tokens[i-1]
        lines.append(f'[model] {kind} begin position={position:08X} input_token={incoming:08X}')
        lines.append(f'[model] {kind} position={position:08X} token={token:08X} logit_bits=3F800000 compute_cycles=0000000000000001')
        lines.append(f'[text] prediction={i:08X} token={token:08X} eos={int(token==151645):08X} bytes={len(chunk):08X} hex={chunk.hex()}')
    lines+=['[text] finish hex=',f'[text] output bytes={len(payload):08X} hex={payload.hex()}','[text] BEGIN']
    raw=('\r\n'.join(lines)+'\r\n').encode()+payload+b'\r\n[text] END\r\nverify fixed text validation: PASS\r\n[nr] RA returned: PASS\r\nverify NR runtime: PASS\r\n'
    def encode(text):
        if text!=prompt.decode():raise ValueError('unexpected prompt')
        return ids
    def decode(actual):
        if actual!=tokens:raise ValueError('unexpected token sequence')
        return chunks,b'',payload
    class FixtureOracle:
        eos={151643,151645}
        def prompt_ids(self,text,thinking):return encode(text)
        def incremental_decode(self,tokens):return decode(tokens)
    return raw,plan,FixtureOracle()


class FixedTextChecker(unittest.TestCase):
    def test_exact_raw_utf8_incremental_records_and_payload(self):
        raw,plan,oracle=fixture()
        result=checker.verify(raw,plan,oracle)
        self.assertEqual(result['status'],'FIXED_TEXT_PASS')
        self.assertEqual(result['output_text'],'A中B\r\nCDE')

    def test_generated_marker_spellings_are_payload_not_fake_records(self):
        chunks=[b'[text] strange\r\nFAIL\r\n',b'[text] output bytes=00000000 hex=\r\n[text] BEGIN\r\n']+[b'']*7
        self.assertEqual(checker.verify(*fixture(chunks))['status'],'FIXED_TEXT_PASS')

    def test_missing_duplicate_unknown_wrong_bytes_tokens_position_and_failures(self):
        raw,plan,oracle=fixture()
        prediction=next(line for line in raw.splitlines() if line.startswith(b'[text] prediction='))
        variants={
            'missing':raw.replace(prediction+b'\r\n',b'',1),
            'duplicate':raw+prediction+b'\r\n',
            'unknown':raw+b'[text] invented=yes\r\n',
            'token':raw.replace(b'prediction=00000000 token=00000020',b'prediction=00000000 token=00000021'),
            'count':raw.replace(b'bytes=00000001 hex=41',b'bytes=00000002 hex=41'),
            'chunk':raw.replace(b'bytes=00000001 hex=41',b'bytes=00000001 hex=42'),
            'literal':raw.replace(b'[text] BEGIN\r\nA',b'[text] BEGIN\r\nB'),
            'position':raw.replace(b'decode position=00000010',b'decode position=00000011'),
            'prompt':raw.replace(b'ids=00000000 ',b'ids=00000001 '),
            'eos':raw.replace(b'eos=00000001',b'eos=00000000'),
            'finish':raw+b'[text] finish hex=\r\n',
            'failure':raw+b'[nr] RA returned: FAIL\r\n',
            'truncated':raw[:-30]}
        for name,bad in variants.items():
            with self.subTest(name=name),self.assertRaises(ValueError):
                checker.verify(bad,plan,oracle)

    def test_official_oracle_checks_actual_template_and_cross_token_utf8(self):
        assets=MODEL/'assets/official'
        if not (assets/'tokenizer.json').exists():self.skipTest('official assets missing')
        try:oracle=checker.OfficialOracle(assets)
        except ImportError:self.skipTest('transformers/tokenizers unavailable')
        self.assertEqual(oracle.prompt_ids('What is France?',False),[151644,872,198,3838,374,9625,30,
                         151645,198,151644,77091,198,151667,271,151668,271])
        chunks,finish,whole=oracle.incremental_decode([32,33,151645,34])
        self.assertEqual(chunks,[b'A',b'B',b'',b'C']);self.assertEqual(finish,b'');self.assertEqual(whole,b'ABC')


if __name__=='__main__':unittest.main()
