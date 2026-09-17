"""Stage B mapping validation and ASan/UBSan comparator fault injection.

Requires the separately generated one-layer IR/report/reference fixtures. These
tests validate diagnostic plumbing, not FPGA/model numerical correctness.
"""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

MODEL=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('intermediate_probe',MODEL/'tools/intermediate_probe.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def arguments():
    p=MODEL/'build/review-native-1l'
    return SimpleNamespace(layers=1,interactive=False,
        reference_arrays=p/'quant-nr-trace/arrays.npz',
        report=p/'replacement/triton-call-replacement.json',
        adapters=p/'replacement/qwen_triton_adapters.c',
        graph_ir=p/'nr-prefill/forward_prefill.ll',decode_ir=p/'nr-decode/forward_decode.ll',
        intermediate_layout=MODEL/'build/import/probe-1layer/weight-layout.json',
        intermediate_graph_dir=p/'replacement',
        intermediate_arrays=p/'quant-nr-trace/arrays.npz',
        intermediate_atol=.001,intermediate_mean_atol=.0001,decode_steps=8,prefill_len=16,
        cache_len=512,prompt_ids=[151644,872,198,3838,374,9625,30,151645,198,151644,
                                77091,198,151667,271,151668,271])


class IntermediateProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args=arguments()
        cc=shutil.which('clang') or shutil.which('cc')
        if not cc or not cls.args.intermediate_arrays.exists():
            raise unittest.SkipTest('requires actual Stage B one-layer fixtures and C compiler')
        cls.temp=tempfile.TemporaryDirectory();cls.path=Path(cls.temp.name)
        source,asm,_,cls.manifest=probe.generate_intermediate(cls.args,cls.path)
        harness=r'''
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "intermediate-probe.c"
void nr_puts(const char *s) {(void)s;}
void nr_hex32(uint32_t x) {(void)x;}
void ame_fence(void) {}
static void sample(unsigned index, int fault) {
  const Probe *p=&probes[index];
  uint64_t descriptor[11]={0};
  size_t bytes=(p->count*2+3)*(p->dtype?1:4);
  void *memory=malloc(bytes);assert(memory);memset(memory,0,bytes);
  descriptor[0]=descriptor[1]=(uintptr_t)memory;descriptor[2]=3;
  int64_t stride=2;
  for(unsigned j=p->rank;j>0;j--) {
    descriptor[3+j-1]=p->shape[j-1];descriptor[3+p->rank+j-1]=stride;
    stride*=p->shape[j-1];
  }
  const float *gold=intermediate_reference_raw+p->offset[current_step];
  for(unsigned i=0;i<p->count;i++) {
    if(p->dtype)((int8_t*)memory)[3+2*i]=(int8_t)gold[i];
    else ((float*)memory)[3+2*i]=gold[i];
  }
  if(fault==3)descriptor[3]++;
  if(fault==4)((float*)memory)[3+2*(p->count-1)]+=1;
  if(fault==5) {union {uint32_t u;float f;} nan={0x7fc00000};((float*)memory)[3]=nan.f;}
  if(fault==6)descriptor[3+p->rank]=(uint64_t)-1;
  check(index,descriptor);
  if(fault==2)check(index,descriptor);
  free(memory);
}
int main(int argc,char **argv) {
  assert(argc==2);int fault=atoi(argv[1]);
  for(unsigned step=0;step<9;step++) {
    qwen_intermediate_begin(step?15+step:0,step?1:16);
    if(fault==7 && step==1)qwen_intermediate_begin(17,1);
    unsigned first=1;
    for(unsigned i=0;i<PROBE_COUNT;i++) if(probes[i].kind==current_kind) {
      int injected=first?fault:0;first=0;
      if(injected==1)continue;
      sample(i,injected);
    }
    int status=qwen_intermediate_end();
    if(fault && (fault!=7 || step==1)) {assert(status!=0);return 0;}
    assert(status==0);
  }
  assert(!fault && invocation==9);return 0;
}
'''
        (cls.path/'harness.c').write_text(harness)
        cls.binary=cls.path/'check'
        subprocess.run([cc,'-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer',
            '-ffunction-sections','-fdata-sections','-Wl,--gc-sections','-DHOST_TEST',
            '-I'+str(MODEL.parent),'-I'+str(MODEL.parents[1]/'common/nr'),
            str(cls.path/'harness.c'),str(asm),'-o',str(cls.binary)],check=True,capture_output=True,text=True)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_full_tensor_strides_offsets_and_nine_call_positions(self):
        result=subprocess.run([str(self.binary),'0'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_missing_duplicate_shape_outlier_nan_negative_stride_and_position_fail(self):
        for fault in range(1,8):
            with self.subTest(fault=fault):
                result=subprocess.run([str(self.binary),str(fault)],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)

    def test_mapping_rejects_wrong_layer_count_weight_identity_and_missing_callee(self):
        report=json.loads(self.args.report.read_text())
        changed=copy.copy(self.args);changed.layers=2
        with self.assertRaisesRegex(ValueError,'one decoder layer'):probe.build_mapping(changed,report)
        changed=copy.copy(self.args)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            layout=json.loads(self.args.intermediate_layout.read_text())
            layout['sections'][1]['checkpoint_tensor']='unsupported.norm.weight'
            changed.intermediate_layout=p/'layout.json';changed.intermediate_layout.write_text(json.dumps(layout))
            with self.assertRaisesRegex(ValueError,'unmapped norm'):probe.build_mapping(changed,report)
            changed=copy.copy(self.args)
            changed.graph_ir=p/'missing.ll';changed.graph_ir.write_text('')
            with self.assertRaisesRegex(ValueError,'final LLVM'):probe.build_mapping(changed,report)

    def test_unknown_reference_shape_and_nonfinite_rejected(self):
        import numpy as np
        for mode in ('transpose','nan'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory:
                p=Path(directory);args=copy.copy(self.args)
                with np.load(args.intermediate_arrays) as old:
                    arrays={k:old[k] for k in old.files}
                key='prefill_model.layers.0_input_hidden'
                if mode=='transpose':arrays[key]=arrays[key].T.copy()
                else:arrays[key][0,0]=float('nan')
                args.intermediate_arrays=p/'arrays.npz'
                np.savez(args.intermediate_arrays,**arrays)
                shutil.copyfile(self.args.intermediate_arrays.with_name('quant-reference.json'),p/'quant-reference.json')
                with self.assertRaisesRegex(ValueError,'shape/layout|finite'):
                    probe.generate_intermediate(args,p)


if __name__=='__main__':unittest.main()
