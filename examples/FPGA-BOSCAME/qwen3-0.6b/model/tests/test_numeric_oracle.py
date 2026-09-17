"""Negative numerical-validation tests; the host C graph is an ABI stub only."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import test_image_abi as fixtures

MODEL = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('board_trace', MODEL/'tools/check_board_trace.py')
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)
image = fixtures.image


def bits(value):
    return f'{struct.unpack("<I",struct.pack("<f",value))[0]:08X}'


def fixture():
    manifest={'schema_version':1,'layers':1,'prefill_len':16,'decode_steps':8,
              'vocab_size':151936,'max_abs_tolerance':1e-3,'mean_abs_tolerance':1e-4}
    ref={'kv_key_used':np.zeros((1,8,24,128),dtype=np.float32),
         'kv_value_used':np.zeros((1,8,24,128),dtype=np.float32)}
    lines=[]
    for pos in range(15,24):
        key='prefill_logits' if pos==15 else f'decode_logits_{pos-16}'
        scores=np.zeros((1,1,151936),dtype=np.float32); scores[0,0,42]=1
        ref[key]=scores
        for name in ('logits','key_cache','value_cache'):
            count=151936 if name=='logits' else 8*(pos+1)*128
            lines.append(f'[compare] {name} position={pos:08X} count={count:08X} '
                         'max_abs_bits=00000000 mean_abs_bits=00000000 PASS')
        lines.append(f'[model] selected position={pos:08X} '+
                     ' '.join(f'{i:08X}:00000000' for i in trace.SELECTED_IDS))
        lines.append(f'[model] kv position={pos:08X} layer=00000000 k=00000000 v=00000000')
        kind='prefill' if pos==15 else 'decode'
        start=0 if kind=='prefill' else pos
        lines.append(f'[model] {kind} position={start:08X} token=0000002A '
                     'logit_bits=3F800000 compute_cycles=0000000000000064')
    lines.append('[nr] RA returned: PASS')
    return manifest, ref, '\n'.join(lines)+'\n'


class BoardTraceTests(unittest.TestCase):
    def test_full_summary_fails_missing_duplicate_wrong_count_error_and_nonfinite(self):
        manifest,ref,log=fixture()
        self.assertTrue(trace.check(log,ref)['sampled_numeric_pass_at_1e_3'])
        self.assertTrue(trace.check_full_reference(log,manifest)['pass'])
        first=log.splitlines()[0]
        variants={
            'missing':log.replace(first+'\n','',1),
            'duplicate':first+'\n'+log,
            'wrong_count':log.replace('count=00025180','count=00000001',1),
            'reported_failure':log.replace('00000000 PASS','00000000 FAIL',1),
            'unsampled_logit_error':log.replace('max_abs_bits=00000000',
                                              'max_abs_bits='+bits(.5),1),
            'kv_error':log.replace('[compare] key_cache position=0000000F count=00004000 max_abs_bits=00000000',
                                  '[compare] key_cache position=0000000F count=00004000 max_abs_bits='+bits(.5),1),
            'nonfinite_max':log.replace('max_abs_bits=00000000','max_abs_bits=7FC00000',1),
            'nonfinite_mean':log.replace('mean_abs_bits=00000000','mean_abs_bits=7F800000',1),
            'negative_error':log.replace('max_abs_bits=00000000','max_abs_bits=BF800000',1),
            'runtime_fail':log+'[nr] RA returned: FAIL\n'}
        for name,bad in variants.items():
            with self.subTest(name=name):
                self.assertNotEqual(bad,log,'test mutation did not change the log')
                self.assertFalse(trace.check_full_reference(bad,manifest)['pass'])
        for value in (float('nan'),float('inf'),-1):
            with self.subTest(tolerance=value),self.assertRaises(ValueError):
                trace.check_full_reference(log,dict(manifest,max_abs_tolerance=value))

    def test_generation_trace_rejects_duplicate_missing_empty_samples_and_wrong_order(self):
        _,ref,log=fixture()
        line=next(s for s in log.splitlines() if s.startswith('[model] decode'))
        selected=next(s for s in log.splitlines() if s.startswith('[model] selected'))
        variants=[log+line+'\n',log.replace(line+'\n',''),
                  log.replace(selected,'[model] selected position=0000000F'),
                  log.replace('[model] prefill position=00000000','[model] decode position=00000000'),
                  log.replace('[model] prefill position=00000000','[model] prefill position=0000000F')]
        for bad in variants:
            self.assertFalse(trace.check(bad,ref)['trace_complete'])

    def test_cli_full_success_is_zero_but_cannot_hide_missing_step_or_wrong_token(self):
        manifest,ref,log=fixture()
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);np.savez(p/'reference.npz',**ref)
            manifest['source_sha256']=hashlib.sha256((p/'reference.npz').read_bytes()).hexdigest()
            (p/'manifest.json').write_text(json.dumps(manifest))
            argv=['check','--uart',str(p/'uart.log'),'--host-graph',str(p/'reference.npz'),
                  '--quant-reference',str(p/'reference.npz'),'--embedded-reference',str(p/'manifest.json'),
                  '--output',str(p/'report.json')]
            missing=next(s for s in log.splitlines() if s.startswith('[model] decode'))
            for name,value,expected in [('valid',log,0),
                ('missing_step',log.replace(missing+'\n',''),1),
                ('wrong_token',log.replace('token=0000002A','token=0000002B',1),1),
                ('nan_selected',log.replace('logit_bits=3F800000','logit_bits=7FC00000',1),1)]:
                with self.subTest(name=name):
                    (p/'uart.log').write_text(value)
                    with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(trace.main(),expected)
                    result=json.loads((p/'report.json').read_text())
                    self.assertEqual(result['status'],'FULL_LOGITS_KV_PASS' if expected==0 else 'NOT_ACCEPTED')


class NativeOracleTests(unittest.TestCase):
    def test_prepare_reference_packs_transposed_kv_and_rejects_invalid_values(self):
        _,ref,_=fixture()
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            args,_,_=fixtures.ImageABI().setup_case(directory,1)
            args.reference_arrays=p/'reference.npz'
            args.reference_atol=.001;args.reference_mean_atol=.0001
            metadata={'prompt_ids':args.prompt_ids,'layers':1,'capacity':args.cache_len,
                      'arithmetic_profile':'nr-fpga','generated_ids':[42]*8,
                      'prefill_argmax_last':42,
                      'decode_steps_recorded':[{'step':position-16,
                                                'input_token':42,'generated_token':42,
                                                'cache_position':position}
                                               for position in range(16,24)]}
            (p/'quant-reference.json').write_text(json.dumps(metadata))
            original=np.arange(8*24*128,dtype=np.float32).reshape(1,8,24,128)
            ref['kv_key_used']=original.transpose(0,2,1,3)
            np.savez(args.reference_arrays,**ref)
            record=image.prepare_reference(args,p)
            self.assertEqual(record['layers'],1)
            self.assertEqual(record['vocab_size'],151936)
            self.assertEqual(record['purpose'],'validation only; not used to choose tokens or update model state')
            blob=np.fromfile(p/'numeric-reference.bin',dtype='<f4')
            np.testing.assert_array_equal(blob[9*151936:9*151936+original.size],original.ravel())
            args.interactive=True
            with self.assertRaisesRegex(ValueError,'interactive'):
                image.prepare_reference(args,p)
            args.interactive=False
            for tolerance in (float('inf'),float('nan'),-1):
                args.reference_atol=tolerance
                with self.assertRaisesRegex(ValueError,'threshold'):
                    image.prepare_reference(args,p)
            args.reference_atol=.001
            ref['prefill_logits']=ref['prefill_logits'].astype(np.float64)
            ref['prefill_logits'][0,0,12345]=-1e40
            np.savez(args.reference_arrays,**ref)
            with np.errstate(over='ignore'),self.assertRaisesRegex(ValueError,'non-finite'):
                image.prepare_reference(args,p)

    def test_generated_oracle_detects_unsampled_elements_without_selecting_tokens(self):
        cc=shutil.which('clang') or shutil.which('cc')
        if not cc:self.skipTest('host C compiler unavailable')
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            args,report,segment=fixtures.ImageABI().setup_case(directory,2)
            args.reference_arrays=p/'oracle.npz';args.reference_atol=.001;args.reference_mean_atol=.0001
            source,_=image.generate(report,segment,args)
            (p/'generated.c').write_text(source)
            code='''#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "generated.c"
const float model_reference_raw[REF_LOGITS_ROWS*VOCAB+2*LAYERS*8*REF_LENGTH*128]={
'''+','.join(f'[{row}*VOCAB+42]=1' for row in range(9))+'''};
static float logits[2*VOCAB+3];
void nr_puts(const char *s) {fputs(s,stdout);}
void nr_hex32(uint32_t v) {printf("%08X",v);}
void nr_hex64(uint64_t v) {printf("%016llX",(unsigned long long)v);}
uint64_t nr_cycles(void) {return 100;}
uintptr_t nr_heap_mark(void) {return 64;}
void nr_heap_reset(uintptr_t m) {(void)m;}
void ame_fence(void) {}
void nr_copy_bytes(void *d,const void *s,size_t n) {memcpy(d,s,n);}
'''
            for kind in ('prefill','decode'):
                params=['GraphResults *r','MemRef2 *ids']
                for layer in range(2):
                    params.extend([f'MemRef1 *p{layer}',f'MemRef4 *k{layer}',f'MemRef4 *v{layer}'])
                params.append('MemRef1 *bounds')
                code+=f'void _mlir_ciface_forward_{kind}('+','.join(params)+') {assert(0);}\n'
            code+='''
int main(int argc,char **argv) {
  assert(argc==2);int mode=atoi(argv[1]);
  reset_cache(); GraphResults r={0};
  for (unsigned l=0;l<LAYERS;++l) {
    r.cache[l].key=make_4(k_cache_f+l*KV_ELEMENTS,1,8,CAPACITY,128);
    r.cache[l].value=make_4(v_cache_f+l*KV_ELEMENTS,1,8,CAPACITY,128);
  }
  r.logits=make_3(logits,1,1,VOCAB);r.logits.offset=3;r.logits.strides[2]=2;
  float *scores=logits+3;scores[2*42]=1;
  unsigned position=mode==3?23:15;
  unsigned kv=((LAYERS-1)*8+7)*CAPACITY*128+position*128+127;
  if(mode==1)scores[2*12345]=.5f; /* not one of nine samples and not argmax */
  if(mode==2)k_cache_f[kv]=.5f;
  if(mode==3)v_cache_f[kv]=.5f;
  if(mode==4)scores[2*12345]=__builtin_nanf("");
  if(mode==5)k_cache_f[kv]=__builtin_inff();
  if(mode==6)k_cache_f[31*128+127]=.5f; /* unused future cache slot */
  if(mode==7)scores[2*53]=2; /* compiled graph chooses 53; oracle chooses 42 */
  unsigned token=999;float score=-1;
  int status=collect(&r,position,&token,&score,0);
  assert(token==(mode==7?53:42));
  assert(mode==0||mode==6 ? status==0 : status!=0);
  return 0;
}
'''
            (p/'harness.c').write_text(code)
            subprocess.run([cc,'-O1','-g','-fsanitize=address,undefined','-DHOST_TEST',
                            '-I'+str(MODEL.parent),'-I'+str(MODEL.parents[1]/'common/nr'),
                            str(p/'harness.c'),'-o',str(p/'check')],
                           check=True,capture_output=True,text=True)
            for mode in range(8):
                with self.subTest(mode=mode):
                    result=subprocess.run([str(p/'check'),str(mode)],capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
