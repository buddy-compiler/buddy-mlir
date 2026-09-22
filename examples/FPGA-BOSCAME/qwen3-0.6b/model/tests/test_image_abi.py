"""Execute generated launch glue against adversarial graph ABI stubs, not kernels."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

MODEL = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('image', MODEL/'tools/build_nr_w8a8_image.py')
image = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image)


def ir_fixture(path, layers, name):
    def desc(r):
        return f'{{ ptr, ptr, i64, [{r} x i64], [{r} x i64] }}'
    result = '{ ' + ', '.join(desc(r) for r in [1,4,4]*layers+[3]) + ' }'
    path.write_text(f'define {result} @forward_{name}() {{\n}}\n'
                    f'define void @_mlir_ciface_forward_{name}(ptr %out) {{\n'
                    + '\n'.join(f'%x{i} = load {desc(r)}, ptr %p{i}'
                                for i,r in enumerate([2]+[1,4,4]*layers+[1]))+'\n}\n')


class ImageABI(unittest.TestCase):
    def setup_case(self, directory, layers):
        p = Path(directory)
        for kind in ('prefill','decode'):
            ir_fixture(p/f'{kind}.ll', layers, kind)
        args = SimpleNamespace(layers=layers, cache_len=32, head_dim=128,
            prefill_len=16, prompt_ids=list(range(16)), decode_steps=8,
            interactive=False, max_new_tokens=8, tokenizer_blob=None,
            graph='prefill', graph_ir=p/'prefill.ll', decode_ir=p/'decode.ll')
        report={'graphs':{kind:{'parameters':[], 'workspace_inputs':[{'name':'attention_position', 'shape':[length], 'dtype':'TensorDType.Int32', 'role':'cache_positions'}]} for kind,length in [('prefill',16),('decode',1)]}}
        for kind,length in [('prefill',16),('decode',1)]:
            def desc(index,shape,dtype,node):
                return {'index':index,'node':node,'shape':shape,'dtype':dtype,
                        'rank':len(shape),'descriptor_bytes_lp64':8*(3+2*len(shape))}
            inputs=[desc(0,[1,length],'i64','ids')]
            outputs=[]
            for l in range(layers):
                for shape,dtype,name in [([1],'i64','position'),([1,8,32,128],'f32','key'),([1,8,32,128],'f32','value')]:
                    inputs.append(desc(len(inputs),shape,dtype,f'{name}_{l}'))
                    outputs.append(desc(len(outputs),shape,dtype,f'{name}_{l}'))
            inputs.append(desc(len(inputs),[length],'i32','attention_position'))
            outputs.append(desc(len(outputs),[1,1,151936],'f32','logits'))
            cursor=0
            for item in outputs:
                item['aggregate_offset_bytes_lp64']=cursor
                cursor+=item['descriptor_bytes_lp64']
            report['graphs'][kind]['entry_abi']={'entry':'forward_'+kind,
                'inputs':inputs,'outputs':outputs,'parameter_count':0,
                'result_descriptor_count':len(outputs),'result_aggregate_bytes_lp64':cursor}
        segment={'placement':[], 'bytes':64}
        return args, report, segment

    def test_profile_sync_preserves_graph_calls_and_graph_final_resync(self):
        with tempfile.TemporaryDirectory() as d:
            args, report, segment = self.setup_case(d, 1)
            args.profile_kernels = True
            args.profile_progress = True
            default_source, default_plan = image.generate(report, segment, args)
            self.assertEqual(default_plan['completion_sync'], 'ame-resync')
            args.profile_sync = 'fence'
            fence_source, fence_plan = image.generate(report, segment, args)
            self.assertEqual(fence_source, default_source)
            self.assertEqual(fence_source.count('  ame_fence();'), 2)
            self.assertEqual(fence_plan['completion_sync'], 'fence')
            self.assertEqual(fence_plan['graph_completion_sync'], 'ame-resync')
            self.assertIn('do not establish AME completion',
                          ' '.join(fence_plan['profile_sync_limits']))

    def test_ame_cache_sync_is_opt_in_and_boundary_scoped(self):
        with tempfile.TemporaryDirectory() as d:
            args, report, segment = self.setup_case(d, 1)
            default, default_plan = image.generate(report, segment, args)
            self.assertNotIn('nr_ame_cache_clean(', default)
            self.assertEqual(default_plan['ame_cache_sync'], 'none')
            args.ame_cache_sync = 'workspace'
            diagnostic, plan = image.generate(report, segment, args)
            self.assertIn('nr_ame_cache_clean(k_cache_raw,', diagnostic)
            self.assertIn('nr_ame_cache_invalidate(ws_prefill_raw,', diagnostic)
            self.assertEqual(plan['ame_cache_sync'], 'workspace')
            self.assertTrue(any('not the platform SYNC_MEM ABI' in item
                                for item in plan['ame_cache_sync_limits']))

    def test_reject_diagnostic_sync_without_profiler_or_unknown_sync(self):
        with tempfile.TemporaryDirectory() as d:
            args, report, segment = self.setup_case(d, 1)
            args.profile_sync = 'fence'
            with self.assertRaisesRegex(ValueError, 'requires --profile-kernels'):
                image.generate(report, segment, args)
            args.profile_kernels = True
            args.profile_sync = 'none'
            with self.assertRaisesRegex(ValueError, '--profile-sync'):
                image.generate(report, segment, args)

    def test_reject_layer_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            args,report,segment = self.setup_case(d,4)
            args.layers=1
            with self.assertRaisesRegex(ValueError,'result ranks'):
                image.generate(report,segment,args)

    def test_reject_missing_workspace(self):
        with tempfile.TemporaryDirectory() as d:
            args,report,segment = self.setup_case(d,1)
            report['graphs']['prefill']['workspace_inputs']=[]
            with self.assertRaisesRegex(ValueError,'input ranks|shape/dtype'):
                image.generate(report,segment,args)

    def test_reject_cache_shape_with_unchanged_descriptor_ranks(self):
        with tempfile.TemporaryDirectory() as d:
            args,report,segment=self.setup_case(d,1)
            args.cache_len=512
            with self.assertRaisesRegex(ValueError,'shape/dtype'):
                image.generate(report,segment,args)

    def test_reject_token_shape_with_unchanged_descriptor_ranks(self):
        with tempfile.TemporaryDirectory() as d:
            args,report,segment=self.setup_case(d,1)
            args.prefill_len=8
            args.prompt_ids=list(range(8))
            with self.assertRaisesRegex(ValueError,'shape/dtype'):
                image.generate(report,segment,args)

    def test_reject_workspace_dtype_with_opaque_llvm_pointers(self):
        with tempfile.TemporaryDirectory() as d:
            args,report,segment=self.setup_case(d,1)
            report['graphs']['prefill']['workspace_inputs'][0]['dtype']='TensorDType.Float32'
            with self.assertRaisesRegex(ValueError,'shape/dtype'):
                image.generate(report,segment,args)

    def test_reject_wrong_output_dtype_or_aggregate_layout(self):
        for field,value in [('dtype','i32'),('aggregate_offset_bytes_lp64',128)]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
                args,report,segment=self.setup_case(d,1)
                report['graphs']['prefill']['entry_abi']['outputs'][-1][field]=value
                with self.assertRaisesRegex(ValueError,'shape/dtype|aggregate offset'):
                    image.generate(report,segment,args)

    def test_require_imported_abi(self):
        with tempfile.TemporaryDirectory() as d:
            args,report,segment=self.setup_case(d,1)
            del report['graphs']['prefill']['entry_abi']
            with self.assertRaisesRegex(ValueError,'entry_abi'):
                image.generate(report,segment,args)

    def test_reject_context_overflow(self):
        with tempfile.TemporaryDirectory() as d:
            args,report,segment=self.setup_case(d,1)
            args.decode_steps=17
            with self.assertRaisesRegex(ValueError,'exceeds cache'):
                image.generate(report,segment,args)

    def test_interactive_uses_last_prompt_prediction(self):
        cc=shutil.which('clang') or shutil.which('cc')
        if not cc: self.skipTest('host C compiler unavailable')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            args,report,segment=self.setup_case(d,1)
            args.interactive=True
            args.max_new_tokens=3
            args.tokenizer_blob=p/'tokenizer.bin'
            args.tokenizer_blob.write_bytes(b'fixture')
            source,_=image.generate(report,segment,args)
            (p/'generated.c').write_text(source)
            harness=r'''
#include <assert.h>
#include <string.h>
#include "generated.c"
static unsigned calls, emitted;
static float logits[VOCAB];
static int64_t expected[] = {3,7,9,10};
void nr_puts(const char *s) {(void)s;}
void nr_write(const void *s,size_t n) {(void)s;(void)n;}
void nr_hex32(uint32_t v) {(void)v;}
void nr_hex64(uint64_t v) {(void)v;}
uint64_t nr_cycles(void) {return 100;}
uintptr_t nr_heap_mark(void) {return 64;}
void nr_heap_reset(uintptr_t m) {(void)m;}
void ame_fence(void) {}
void nr_copy_bytes(void *d,const void *s,size_t n) {memcpy(d,s,n);}
int nr_getchar(void) { static unsigned i; static const char s[]="x\r\n/quit\n";
  assert(i<sizeof(s)-1); return s[i++]; }
int qwen_tokenizer_open(QwenTokenizerResource *r,const void *b,size_t n) {
  (void)r;(void)b;(void)n;return 0; }
int qwen_chat_single_turn(uint8_t *o,size_t c,size_t *n,const uint8_t *s,
  size_t sn,int hs,const uint8_t *u,size_t un,int t) {
  (void)o;(void)c;(void)s;(void)sn;(void)hs;(void)t;
  assert(un==1 && u[0]=='x'); *n=1;return 0; }
int qwen_encode(const QwenTokenizerResource *r,const uint8_t *s,size_t n,
  uint32_t *o,size_t c,size_t *written) {
  (void)r;(void)s;(void)n;(void)c; o[0]=3;o[1]=7;*written=2;return 0; }
int qwen_decode_token(const QwenTokenizerResource *r,QwenUtf8Decoder *s,
  uint32_t t,int skip,QwenEmit e,void *c) {
  (void)r;(void)s;(void)skip;(void)e;(void)c;
  assert(t==9+emitted); emitted++;return 0; }
void qwen_decode_finish(QwenUtf8Decoder *s,QwenEmit e,void *c) {(void)s;(void)e;(void)c;}
void _mlir_ciface_forward_prefill(GraphResults *r,MemRef2 *ids,MemRef1 *p,
  MemRef4 *k,MemRef4 *v,MemRef1 *bounds) {(void)bounds;(void)r;(void)ids;(void)p;(void)k;(void)v;assert(0);}
void _mlir_ciface_forward_decode(GraphResults *r,MemRef2 *ids,MemRef1 *p,
  MemRef4 *k,MemRef4 *v,MemRef1 *bounds) {
  assert(*(int32_t*)bounds->aligned == calls);
  assert(calls<4 && *(int64_t*)p->aligned==calls);
  assert(*(int64_t*)ids->aligned==expected[calls]);
  r->cache[0]=(CacheResult){*p,*k,*v};
  memset(logits,0,sizeof(logits)); logits[8+calls]=1;
  r->logits=make_3(logits,1,1,VOCAB);calls++;
}
int main(void) {assert(launch()==0);assert(calls==4 && emitted==3);return 0;}
'''
            (p/'harness.c').write_text(harness)
            subprocess.run([cc,'-O1','-g','-fsanitize=address,undefined','-DHOST_TEST',
                '-I'+str(MODEL.parent),'-I'+str(MODEL.parents[1]/'common/nr'),
                '-I'+str(MODEL/'text'),str(p/'harness.c'),'-o',str(p/'check')],
                check=True,capture_output=True,text=True)
            subprocess.run([str(p/'check')],check=True,capture_output=True,text=True)

    def test_multilayer_return_and_cache_lifetime(self):
        cc=shutil.which('clang') or shutil.which('cc')
        if not cc: self.skipTest('host C compiler unavailable')
        for layers in (1,4,28):
            with self.subTest(layers=layers), tempfile.TemporaryDirectory() as d:
                p=Path(d)
                args,report,segment=self.setup_case(d,layers)
                source,_=image.generate(report,segment,args)
                (p/'generated.c').write_text(source)
                code = '''#include <assert.h>
#include <string.h>
#include "generated.c"
static unsigned calls, resets;
static float scratch_k[LAYERS][KV_ELEMENTS], scratch_v[LAYERS][KV_ELEMENTS];
static float logits[VOCAB];
void nr_puts(const char *s) {(void)s;}
void nr_hex32(uint32_t v) {(void)v;}
void nr_hex64(uint64_t v) {(void)v;}
uint64_t nr_cycles(void) {return 100;}
uintptr_t nr_heap_mark(void) {return 64;}
void ame_fence(void) {}
void nr_heap_reset(uintptr_t m) {
  assert(m==64); resets++;
  memset(scratch_k,0xa5,sizeof(scratch_k));
  memset(scratch_v,0xa5,sizeof(scratch_v));
}
void nr_copy_bytes(void *d,const void *s,size_t n) {memcpy(d,s,n);}
'''
                for kind in ('prefill','decode'):
                    params=['GraphResults *r','MemRef2 *ids']
                    for l in range(layers):
                        params.extend([f'MemRef1 *p{l}',f'MemRef4 *k{l}',f'MemRef4 *v{l}'])
                    params.append('MemRef1 *bounds')
                    code+=f'void _mlir_ciface_forward_{kind}('+','.join(params)+') {\n'
                    code+='  assert(resets==calls);\n'
                    code+='  for (unsigned j=0; j<bounds->sizes[0]; ++j) assert(((int32_t*)bounds->aligned)[j] == (calls ? 15+calls : 0)+j);\n'
                    for l in range(layers):
                        code+=f'''
  assert(k{l}->aligned == k_cache_f + {l}*KV_ELEMENTS);
  assert(v{l}->aligned == v_cache_f + {l}*KV_ELEMENTS);
  assert(k{l}->aligned != v{l}->aligned);
  assert(*(int64_t *)p{l}->aligned == (calls ? 15+calls : 0));
  if (calls) {{
    assert(((float *)k{l}->aligned)[0] == (float)({l}*100+calls));
    assert(((float *)v{l}->aligned)[0] == (float)({l}*200+calls));
  }}
  scratch_k[{l}][0] = (float)({l}*100+calls+1);
  scratch_v[{l}][0] = (float)({l}*200+calls+1);
  r->cache[{l}].position = *p{l};
  r->cache[{l}].key = make_4(scratch_k[{l}],1,8,CAPACITY,128);
  r->cache[{l}].value = make_4(scratch_v[{l}],1,8,CAPACITY,128);
'''
                    code+='''
  if (calls) assert(((int64_t *)ids->aligned)[0] == 41+calls);
  memset(logits,0,sizeof(logits)); logits[42+calls]=1;
  r->logits=make_3(logits,1,1,VOCAB);
  calls++;
}
'''
                code+='int main(void) { assert(launch()==0); assert(calls==9 && resets==9); return 0; }\n'
                (p/'harness.c').write_text(code)
                nr=MODEL.parents[1]/'common/nr'
                cmd=[cc,'-O1','-g','-fsanitize=address,undefined','-DHOST_TEST',
                     '-I'+str(MODEL.parent),'-I'+str(nr), str(p/'harness.c'),'-o',str(p/'check')]
                subprocess.run(cmd,check=True,capture_output=True,text=True)
                subprocess.run([str(p/'check')],check=True,capture_output=True,text=True)

if __name__=='__main__': unittest.main()
