#!/usr/bin/env python3
"""Generate the Qwen3-0.6B linear and attention operator microbenchmarks."""
import json
from pathlib import Path

def write(root, name, ir, c, meta):
    d = root / name
    d.mkdir(exist_ok=True)
    (d/'kernel.mlir').write_text(ir)
    (d/'launch.c').write_text(c)
    (d/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    (d/'makefile').write_text('include ../common.mk\n')

def linear(root, m, n, k, roles, fp32=False):
    name = f'matmul_{m}x{n}x{k}'
    ir = f'''// Signed W8A8 deployment dot product; raw NR i32 accumulator ABI.
// B is stored [N,K], exposed as logical [K,N] without a data transpose.
module {{
  func.func @kernel_{name}(%a: memref<{m}x{k}xi8>,
      %b: memref<{k}x{n}xi8, strided<[1, {k}]>>,
      %c: memref<{m}x{n}xi32>) attributes {{llvm.emit_c_interface}} {{
    linalg.matmul ins(%a, %b : memref<{m}x{k}xi8>, memref<{k}x{n}xi8, strided<[1, {k}]>>)
      outs(%c : memref<{m}x{n}xi32>)
    return
  }}
}}
'''
    c = f'''#include "support.h"
#define M {m}
#define N {n}
#define K {k}
extern void _mlir_ciface_kernel_{name}(MemRef2 *, MemRef2 *, MemRef2 *);
int launch(void) {{
  int8_t *a = workspace(0), *b = workspace((M*K+63)&~63);
  int32_t *c = workspace(((size_t)M*K+(size_t)N*K+127)&~(size_t)63);
  /* Distinct row/column patterns exercise signs, K accumulation and tile tails. */
  for (int i=0;i<M;i++) for(int q=0;q<K;q++) a[i*K+q]=(i*3+q*5)%13-6;
  for (int j=0;j<N && j<11;j++) for(int q=0;q<K;q++) b[(size_t)j*K+q]=(j*7+q*3)%11-5;
  /* The same independently checked data repeat every eleven rows. Reuse
   * those rows through the NR RVV copy instead of scalar modulo per byte. */
  for (int j=11;j<N;j++) nr_copy_bytes(b+(size_t)j*K,b+(size_t)(j%11)*K,K*sizeof(*b));
  for (int i=0;i<M;i++) for(int j=0;j<N;j++) c[(size_t)i*N+j]=(i+j)%7-3;
  MemRef2 A=make_2(a,M,K), B=make_2(b,K,N), C=make_2(c,M,N);
  B.strides[0]=1; B.strides[1]=K;
#if !defined(HOST_TEST)
  ame_fence();
#endif
  uint64_t start=nr_cycles();
  _mlir_ciface_kernel_{name}(&A,&B,&C);
  nr_puts("cycles {name}: "); nr_hex64(nr_cycles()-start); nr_puts("\\r\\n");
  /* An independent CPU dot oracle computes all 143 residue combinations,
   * then checks EVERY output. This avoids repeating billions of scalar MACs. */
  int32_t oracle[13][11];
  for(int x=0;x<13;x++) for(int y=0;y<11;y++) {{
    int32_t dot=0;
    for(int q=0;q<K;q++) dot+=((x+q*5)%13-6)*((y+q*3)%11-5);
    oracle[x][y]=dot;
  }}
  unsigned errors=0; float max_error=0;
  for(int i=0;i<M;i++) for(int j=0;j<N;j++) {{
    int32_t want=oracle[(i*3)%13][(j*7)%11]+(i+j)%7-3;
    int32_t got=c[(size_t)i*N+j];
    if(got!=want) {{
      if(errors<4) {{ nr_puts("mismatch row/col/got/want "); nr_hex32(i);nr_puts(" ");nr_hex32(j);nr_puts(" ");nr_hex32(got);nr_puts(" ");nr_hex32(want);nr_puts("\\r\\n"); }}
      errors++;
      float d=(float)got-(float)want; if(d<0)d=-d; if(d>max_error)max_error=d;
    }}
  }}
  return print_check("{name}",errors,max_error);
}}
'''
    if fp32:
        oldname=name
        name += '_f32'
        ir=ir.replace(oldname,name).replace('xi8','xf32').replace('xi32','xf32')
        ir=ir.replace('// Signed W8A8 deployment dot product; raw NR i32 accumulator ABI.', '// FP32 linear operator; Buddy transpose-B RVV lowering, no quantization.')
        ir=ir.replace('// B is stored [N,K], exposed as logical [K,N] without a data transpose.', '// Physical [N,K] B permits contiguous K-vector loads without a copy.')
        ir=ir.replace(f'memref<{k}x{n}xf32, strided<[1, {k}]>>', f'memref<{n}x{k}xf32>')
        ir=ir.replace('linalg.matmul ins', 'linalg.matmul indexing_maps = [affine_map<(m,n,k)->(m,k)>, affine_map<(m,n,k)->(n,k)>, affine_map<(m,n,k)->(m,n)>] ins')
        c=c.replace(oldname,name).replace('int8_t *a', 'float *a').replace('int32_t *c', 'float *c')
        c=c.replace('#if !defined(HOST_TEST)\n  ame_fence();\n#endif\n', '')
        c=c.replace('workspace((M*K+63)&~63)', 'workspace(((M*K*4)+63)&~63)')
        c=c.replace('((size_t)M*K+(size_t)N*K+127)', '(((size_t)M*K+(size_t)N*K)*4+127)')
        # Small integers have exact FP32 sums for these K and are checked
        # against the independent int32 oracle without reusing float math.
        c=c.replace('int32_t got=c[(size_t)i*N+j];','float got=c[(size_t)i*N+j];')
        c=c.replace('B=make_2(b,K,N)', 'B=make_2(b,N,K)')
        c=c.replace('  B.strides[0]=1; B.strides[1]=K;\n', '')
    write(root,name,ir,c,dict(kind='matmul_f32' if fp32 else 'matmul_i8' ,shape=[m,n,k],roles=roles,dtype='f32' if fp32 else 'i8 x i8 -> i32',target='nr-rvv' if fp32 else 'nr-fpga',checked_elements=m*n))

def attention(root, s, t, pv=False):
    m,n,k = (s,128,t) if pv else (s,t,128)
    name=f'attention_{"pv" if pv else "qk"}_16x{m}x{n}x{k}'
    ir=f'''// FP32 attention, no integer quantization of probabilities or Q/K/V.
module {{
  func.func @kernel_{name}(%a: memref<16x{m}x{k}xf32>, %b: memref<16x{k}x{n}xf32>, %c: memref<16x{m}x{n}xf32>) attributes {{llvm.emit_c_interface}} {{
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%c : memref<16x{m}x{n}xf32>)
    linalg.batch_matmul ins(%a, %b : memref<16x{m}x{k}xf32>, memref<16x{k}x{n}xf32>) outs(%c : memref<16x{m}x{n}xf32>)
    return
  }}
}}
'''
    c=f'''#include "support.h"
#define M {m}
#define N {n}
#define K {k}
extern void _mlir_ciface_kernel_{name}(MemRef3*,MemRef3*,MemRef3*);
int launch(void) {{
  float *a=workspace(0),*b=workspace(16*M*K*4),*c=workspace(16*(M*K+K*N)*4);
  for(int x=0;x<16*M*K;x++) a[x]=(float)(x%17-8)*0.125f;
  for(int x=0;x<16*K*N;x++) b[x]=(float)(x%13-6)*0.0625f;
  for(int x=0;x<16*M*N;x++) c[x]=1234.0f;
  MemRef3 A=make_3(a,16,M,K),B=make_3(b,16,K,N),C=make_3(c,16,M,N);
  _mlir_ciface_kernel_{name}(&A,&B,&C);
  unsigned errors=0; float max_error=0;
  for(int h=0;h<16;h++) for(int i=0;i<M;i++) for(int j=0;j<N;j++) {{
    double want=0;
    for(int q=0;q<K;q++) want+=(double)a[(h*M+i)*K+q]*(double)b[(h*K+q)*N+j];
    float got=c[(h*M+i)*N+j],d=got-(float)want; if(d<0)d=-d;
    if(d>max_error)max_error=d;
    if(!check_close(got,(float)want,1e-6f,1e-6f))errors++;
  }}
  return print_check("{name}",errors,max_error);
}}
'''
    write(root,name,ir,c,dict(kind='attention_pv' if pv else 'attention_qk',shape=[16,m,n,k],dtype='f32',target='nr-rvv',checked_elements=16*m*n))

def generate(root):
    for m in (1,16):
        for n,k,roles in ((2048,1024,['q_proj']),(1024,1024,['k_proj','v_proj']),(1024,2048,['o_proj']),(3072,1024,['gate_proj','up_proj']),(1024,3072,['down_proj'])):
            linear(root,m,n,k,roles)
            linear(root,m,n,k,roles,True)
    linear(root,1,151936,1024,['lm_head_last_token'])
    linear(root,1,151936,1024,['lm_head_last_token'],True)
    # Hardware tile-tail regression, intentionally distinguished from model shapes.
    linear(root,3,19,70,['nr_tail_regression'])
    for s,t in ((16,16),(1,17)):
        attention(root,s,t)
        attention(root,s,t,True)

if __name__=='__main__': generate(Path(__file__).resolve().parents[1])
