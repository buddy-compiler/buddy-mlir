#!/usr/bin/env python3
"""Explicit deployment-only symmetric per-token quantization and scaling."""
from pathlib import Path
from generate_matmul import write

def quant(root,m,k):
    name=f'quantize_{m}x{k}'
    ir=f'''module {{
func.func @kernel_{name}(%x: memref<{m}x{k}xf32>, %q: memref<{m}x{k}xi8>, %scale: memref<{m}xf32>) attributes {{llvm.emit_c_interface}} {{
 %z = arith.constant 0.0 : f32
 %one = arith.constant 1.0 : f32
 %lim = arith.constant 127.0 : f32
 %neg = arith.constant -127.0 : f32
 %half = arith.constant 0.5 : f32
 %nhalf = arith.constant -0.5 : f32
 linalg.fill ins(%z : f32) outs(%scale : memref<{m}xf32>)
 linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i)>], iterator_types=["parallel","reduction"]}} ins(%x:memref<{m}x{k}xf32>) outs(%scale:memref<{m}xf32>) {{
 ^bb0(%v:f32,%a:f32):
  %abs = math.absf %v : f32
  %max = arith.maximumf %a,%abs : f32
  linalg.yield %max : f32
 }}
 linalg.generic {{indexing_maps=[affine_map<(i)->(i)>],iterator_types=["parallel"]}} outs(%scale:memref<{m}xf32>) {{
 ^bb0(%v:f32):
  %zero = arith.cmpf oeq,%v,%z : f32
  %s = arith.divf %v,%lim : f32
  %safe = arith.select %zero,%one,%s : f32
  linalg.yield %safe : f32
 }}
 linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}} ins(%x,%scale:memref<{m}x{k}xf32>,memref<{m}xf32>) outs(%q:memref<{m}x{k}xi8>) {{
 ^bb0(%v:f32,%s:f32,%unused:i8):
  %d = arith.divf %v,%s : f32
  %positive = arith.cmpf oge,%d,%z : f32
  %offset = arith.select %positive,%half,%nhalf : f32
  %rounded = arith.addf %d,%offset : f32
  %lo = arith.maximumf %rounded,%neg : f32
  %hi = arith.minimumf %lo,%lim : f32
  %i = arith.fptosi %hi : f32 to i32
  %byte = arith.trunci %i : i32 to i8
  linalg.yield %byte : i8
 }}
 return
}}
}}
'''
    c=f'''#include "support.h"
#define M {m}
#define K {k}
extern void _mlir_ciface_kernel_{name}(MemRef2*,MemRef2*,MemRef1*);
int launch(void) {{
 float *x=workspace(0),*s=workspace(M*K*4); int8_t *q=workspace(M*K*4+64);
 unsigned errors=0;
 for(int trial=0;trial<2;trial++) {{
 for(int i=0;i<M;i++) for(int j=0;j<K;j++) {{
  float base=j==0?127.0f:j==1?-127.0f:(float)(j%17-8)*0.5f+(j%3)*0.125f;
  x[i*K+j]=(trial==0&&i%3==0)?0.0f:base*(i%2?1.137f:1.0f);
 }}
 MemRef2 X=make_2(x,M,K),Q=make_2(q,M,K);MemRef1 S=make_1(s,M);
 _mlir_ciface_kernel_{name}(&X,&Q,&S);
 for(int i=0;i<M;i++) {{
  float max=0;
  for(int j=0;j<K;j++) {{float v=x[i*K+j];if(v<0)v=-v;if(v>max)max=v;}}
  float scale=max==0?1.0f:max/127.0f;
  if(!check_close(s[i],scale,1e-8f,1e-6f))errors++;
  for(int j=0;j<K;j++) {{
   float v=x[i*K+j]/scale;
   int want=(int)(v+(v>=0?0.5f:-0.5f));if(want>127)want=127;if(want<-127)want=-127;
   if(q[i*K+j]!=want)errors++;
  }}
 }}
 }}
 return print_check("{name}",errors,0);
}}
'''
    write(root,name,ir,c,dict(kind='per_token_quantization',shape=[m,k],rounding='nearest, ties away from zero',zero_row_scale=1,target='scalar',model_operator=False))

def dequant(root,m,n):
    name=f'dequantize_{m}x{n}'
    ir=f'''module {{
func.func @kernel_{name}(%x:memref<{m}x{n}xi32>, %row:memref<{m}xf32>, %col:memref<{n}xf32>, %y:memref<{m}x{n}xf32>) attributes {{llvm.emit_c_interface}} {{
 linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i)>,affine_map<(i,j)->(j)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}} ins(%x,%row,%col:memref<{m}x{n}xi32>,memref<{m}xf32>,memref<{n}xf32>) outs(%y:memref<{m}x{n}xf32>) {{
 ^bb0(%xv:i32,%rs:f32,%cs:f32,%unused:f32):
  %f = arith.sitofp %xv : i32 to f32
  %a = arith.mulf %f,%rs : f32
  %b = arith.mulf %a,%cs : f32
  linalg.yield %b : f32
 }}
 return
}}
}}
'''
    c=f'''#include "support.h"
#define M {m}
#define N {n}
extern void _mlir_ciface_kernel_{name}(MemRef2*,MemRef1*,MemRef1*,MemRef2*);
int launch(void) {{
 int32_t *x=workspace(0);float *r=workspace(M*N*4),*s=workspace(M*N*4+64),*y=workspace(M*N*4+64+N*4);
 for(int i=0;i<M;i++)r[i]=(i+1)*0.00390625f;
 for(int j=0;j<N;j++)s[j]=(j%31+1)*0.001953125f;
 for(int i=0;i<M*N;i++)x[i]=(i%1023-511)*7919;
 MemRef2 X=make_2(x,M,N),Y=make_2(y,M,N);MemRef1 R=make_1(r,M),S=make_1(s,N);
 _mlir_ciface_kernel_{name}(&X,&R,&S,&Y);
 unsigned errors=0;float max_error=0;
 for(int i=0;i<M;i++)for(int j=0;j<N;j++){{
  float want=(float)x[i*N+j]*r[i]*s[j],d=y[i*N+j]-want;if(d<0)d=-d;if(d>max_error)max_error=d;
  if(!check_close(y[i*N+j],want,1e-6f,1e-6f))errors++;
 }}
 return print_check("{name}",errors,max_error);
}}
'''
    write(root,name,ir,c,dict(kind='dequantization',shape=[m,n],dtype='i32 -> f32',target='scalar',model_operator=False))

def generate(root):
    for m in (1,16):
        for k in (1024,2048,3072): quant(root,m,k)
        for n in (1024,2048,3072): dequant(root,m,n)
    dequant(root,1,151936)
if __name__=='__main__':generate(Path(__file__).resolve().parents[1])
