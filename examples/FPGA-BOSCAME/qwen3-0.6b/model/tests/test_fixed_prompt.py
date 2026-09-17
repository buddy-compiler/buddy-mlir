"""ASan/UBSan host execution of firmware glue plus REAL freestanding tokenizer.

The graph functions are adversarial ABI fixtures, not model/FPGA evidence. Set
QWEN_TOKENIZER_BLOB to a packed official tokenizer if model/build/tokenizer.bin
has not been prepared. No host tokenization or text decoding supplies firmware.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import test_image_abi as fixtures

MODEL = Path(__file__).resolve().parents[1]
IDS = [151644,872,198,3838,374,9625,30,151645,198,151644,77091,198,151667,271,151668,271]


class FixedPrompt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cc = shutil.which('clang') or shutil.which('cc')
        cls.blob = Path(os.environ.get('QWEN_TOKENIZER_BLOB', MODEL/'build/tokenizer.bin'))
        if not cls.cc or not cls.blob.is_file():
            raise unittest.SkipTest('requires host compiler and packed official tokenizer')
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.flags = ['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer']
        cls.includes = ['-I'+str(MODEL.parent),'-I'+str(MODEL.parents[1]/'common/nr'),
                        '-I'+str(MODEL/'text')]
        cls.objects = []
        for name in ('tokenizer_encode','tokenizer_resource','unicode_tables'):
            obj = cls.root/(name+'.o')
            subprocess.run([cls.cc,*cls.flags,*cls.includes,'-c',str(MODEL/'text'/f'{name}.c'),
                            '-o',str(obj)],check=True,capture_output=True,text=True)
            cls.objects.append(str(obj))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_fixture(self, *, prompt='What is France?', mismatch=False,
                    oracle=False, output_capacity=65536, variant=False,
                    expected_calls=9, expected_error=None):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            args,report,segment=fixtures.ImageABI().setup_case(directory,1)
            args.prompt_ids=IDS.copy()
            if mismatch: args.prompt_ids[3]+=1
            args.prompt_text=prompt
            args.tokenizer_blob=self.blob
            args.text_output_bytes=output_capacity
            if oracle:
                args.reference_arrays=p/'unused.npz'
                args.reference_atol=10;args.reference_mean_atol=10
            source,summary=fixtures.image.generate(report,segment,args)
            self.assertEqual(summary['fixed_prompt']['predictions_recorded'],9)
            (p/'generated.c').write_text(source)
            harness=r'''
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "generated.c"
static unsigned calls, writes;
static float logits[VOCAB];
static char logs[65536]; static size_t logged;
static uint8_t actual_text[65536]; static size_t actual_bytes;
static const unsigned chosen[] = {FIRST_TOKEN,33,151645,34,35,36,37,38,39};
static const int64_t input_expected[] = {151644,872,198,3838,374,9625,30,151645,
                                       198,151644,77091,198,151667,271,151668,271};
void nr_puts(const char *s) {
  size_t n=strlen(s);assert(logged+n<sizeof(logs));
  memcpy(logs+logged,s,n+1);logged+=n;
}
void nr_write(const void *s,size_t n) {
  assert(actual_bytes+n<=sizeof(actual_text));
  memcpy(actual_text+actual_bytes,s,n);actual_bytes+=n;writes++;
}
void nr_hex32(uint32_t v) {char b[9];snprintf(b,sizeof(b),"%08X",v);nr_puts(b);}
void nr_hex64(uint64_t v) {char b[17];snprintf(b,sizeof(b),"%016llX",(unsigned long long)v);nr_puts(b);}
uint64_t nr_cycles(void) {return 100;}
uintptr_t nr_heap_mark(void) {return 64;}
void nr_heap_reset(uintptr_t m) {assert(m==64);}
void ame_fence(void) {}
void nr_copy_bytes(void *d,const void *s,size_t n) {memcpy(d,s,n);}
static void graph(GraphResults *r,MemRef2 *ids,MemRef1 *p,MemRef4 *k,MemRef4 *v,MemRef1 *bounds) {
  assert(calls<9);
  unsigned position = calls ? 15+calls : 0;
  unsigned length = calls ? 1 : 16;
  assert(ids->sizes[0]==1 && ids->sizes[1]==length);
  assert(*(int64_t*)p->aligned==position);
  for(unsigned i=0;i<length;i++) {
    assert(((int32_t*)bounds->aligned)[i]==position+i);
    assert(((int64_t*)ids->aligned)[i]==(calls ? chosen[calls-1] : input_expected[i]));
  }
  assert(((float*)k->aligned)[0]==calls);
  assert(((float*)v->aligned)[0]==calls);
  ((float*)k->aligned)[0]=(float)(calls+1);
  ((float*)v->aligned)[0]=(float)(calls+1);
  r->cache[0]=(CacheResult){*p,*k,*v};
  memset(logits,0,sizeof(logits)); logits[chosen[calls]]=1;
  r->logits=make_3(logits,1,1,VOCAB);calls++;
}
void _mlir_ciface_forward_prefill(GraphResults *r,MemRef2 *ids,MemRef1 *p,
  MemRef4 *k,MemRef4 *v,MemRef1 *b) {assert(calls==0);graph(r,ids,p,k,v,b);}
void _mlir_ciface_forward_decode(GraphResults *r,MemRef2 *ids,MemRef1 *p,
  MemRef4 *k,MemRef4 *v,MemRef1 *b) {assert(calls>0);graph(r,ids,p,k,v,b);}
#ifdef REF_LENGTH
/* Deliberately disagrees with the real graph's argmax. Large tolerances let
 * validation finish; using this array to choose a token would fail the test. */
const float model_reference_raw[REF_LOGITS_ROWS*VOCAB+2*8*REF_LENGTH*128] = {[999]=9};
#endif
int main(int argc,char **argv) {
  assert(argc==2);
  FILE *file=fopen(argv[1],"rb");assert(file);
  assert(fread(tokenizer_blob_raw,1,TOKENIZER_BYTES,file)==TOKENIZER_BYTES);fclose(file);
  int status=launch();
  assert(calls==EXPECTED_CALLS);
  if (EXPECT_SUCCESS) {
    assert(status==0 && writes==1 && actual_bytes==8);
    assert(!memcmp(actual_text,EXPECTED_TEXT,8));
    assert(strstr(logs,"mode=fixed-validation"));
    assert(strstr(logs,"prediction=00000002 token=0002505D eos=00000001"));
    assert(strstr(logs,"prediction=00000008"));
    assert(strstr(logs,"verify fixed prompt tokenizer: PASS"));
    assert(strstr(logs,"verify fixed text validation: PASS"));
  } else {
    assert(status!=0 && writes==0);
    assert(strstr(logs,EXPECTED_ERROR));
    assert(!strstr(logs,"verify fixed text validation: PASS"));
  }
  return 0;
}
'''
            replacements={'FIRST_TOKEN':str(57 if variant else 32),
                'TOKENIZER_BYTES':str(self.blob.stat().st_size),'EXPECTED_CALLS':str(expected_calls),
                'EXPECT_SUCCESS':'0' if expected_error else '1',
                'EXPECTED_TEXT':'"ZBCDEFGH"' if variant else '"ABCDEFGH"',
                'EXPECTED_ERROR':'"'+(expected_error or 'unused')+'"'}
            for key,value in replacements.items():harness=harness.replace(key,value)
            (p/'harness.c').write_text(harness)
            subprocess.run([self.cc,*self.flags,*self.includes,'-DHOST_TEST',str(p/'harness.c'),
                            *self.objects,'-o',str(p/'check')],check=True,capture_output=True,text=True)
            result=subprocess.run([str(p/'check'),str(self.blob)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_real_encoder_prefill_eight_decode_eos_and_actual_text(self):
        self.run_fixture()

    def test_different_logits_change_text_even_with_unrelated_oracle(self):
        self.run_fixture(oracle=True,variant=True)

    def test_expected_id_mismatch_rejected_before_graph(self):
        self.run_fixture(mismatch=True,expected_calls=0,expected_error='prompt IDs differ')

    def test_encoded_length_mismatch_rejected_before_graph(self):
        self.run_fixture(prompt='Hi',expected_calls=0,expected_error='prompt length differs')

    def test_decoded_output_overflow_fails_without_partial_success(self):
        self.run_fixture(output_capacity=1,expected_calls=2,expected_error='output capacity: FAIL')

    def test_build_contract_rejects_ambiguous_or_invalid_text_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            args,report,segment=fixtures.ImageABI().setup_case(directory,1)
            args.prompt_text='What is France?'
            args.tokenizer_blob=self.blob
            args.interactive=True
            with self.assertRaisesRegex(ValueError,'mutually exclusive'):
                fixtures.image.generate(report,segment,args)
            args.interactive=False;args.tokenizer_blob=None
            with self.assertRaisesRegex(ValueError,'requires tokenizer'):
                fixtures.image.generate(report,segment,args)
            args.prompt_text='x'*2049
            with self.assertRaisesRegex(ValueError,'2048'):
                fixtures.image.generate(report,segment,args)
            args.prompt_text=None;args.prompt_file=Path(directory)/'invalid.txt'
            args.prompt_file.write_bytes(b'\xff')
            with self.assertRaises(UnicodeDecodeError):fixtures.image.fixed_prompt_bytes(args)


if __name__=='__main__':unittest.main()
