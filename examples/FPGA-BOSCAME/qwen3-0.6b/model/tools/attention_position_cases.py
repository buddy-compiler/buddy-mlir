#!/usr/bin/env python3
"""Generate isolated Stage A tests for optional runtime-length attention.

Physical cache strides remain 512. Each executable grows and shrinks the valid
length, poisons invalid inputs with NaNs, and checks every output. Binary-exact
fraction inputs make the independent double oracle exact for either fused or
unfused FP32 dot products. No model deployment uses these cases implicitly.
"""
import argparse
import json
import os
from pathlib import Path

from model_kernel_cases import attention_metadata


def valid_lengths(sequence, boundary_probe=False, native_key=False):
    if boundary_probe:
        return [1, 2, 23, 2, 1] if sequence == 1 else [23, 16, 23]
    if native_key:
        lengths = [1, 2, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                   31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256,
                   257, 511, 512, 65, 64, 63, 24, 23, 17, 16, 2, 1]
        return [length for length in lengths if length >= sequence]
    return [16,17,24,63,64,65,511,512,511,65,64,63,24,17,16]


def launcher(name, sequence, qk, boundary_probe=False, native_key=False):
    n, k = (512, 128) if qk else (128, 512)
    lengths = ",".join(map(str, valid_lengths(sequence, boundary_probe, native_key)))
    return f'''#include "support.h"
#define HEADS 16
#define M {sequence}
#define N {n}
#define K {k}
#define QK {int(qk)}
#define NATIVE_KEY {int(native_key)}
extern void _mlir_ciface_kernel_{name}(MemRef3*,MemRef3*,MemRef1*,MemRef3*);
int launch(void) {{
  float *a=workspace(64),*b=workspace(128+HEADS*M*K*4);
  float *c=workspace(192+HEADS*(M*K+K*N)*4);
  int *position=workspace(256+HEADS*(M*K+K*N+M*N)*4);
  uint32_t *guards[]={{workspace(0),(uint32_t*)(a+HEADS*M*K),
                     (uint32_t*)(b+HEADS*K*N),(uint32_t*)(c+HEADS*M*N),
                     (uint32_t*)(position+M)}};
  MemRef3 A=make_3(a,HEADS,M,K),B=make_3(b,HEADS,NATIVE_KEY?N:K,NATIVE_KEY?K:N),C=make_3(c,HEADS,M,N);
  MemRef1 P=make_1(position,M);
  const int lengths[]={{{lengths}}};
  unsigned errors=0; float maximum=0;
  for(unsigned step=0;step<sizeof(lengths)/sizeof(lengths[0]);++step) {{
    int valid=lengths[step];
    for(int row=0;row<M;++row)position[row]=valid-M+row;
    for(int h=0;h<HEADS;++h)for(int row=0;row<M;++row)for(int q=0;q<K;++q) {{
      int x=(h*M+row)*K+q;
      a[x]=(!QK && q>=valid)?__builtin_nanf(""):(float)(x%17-8)*0.125f;
    }}
    for(int h=0;h<HEADS;++h)for(int q=0;q<K;++q)for(int col=0;col<N;++col) {{
      int x=(h*K+q)*N+col;
      int physical=NATIVE_KEY?(h*N+col)*K+q:x;
      b[physical]=((QK && col>=valid)||(!QK && q>=valid))?__builtin_nanf(""):(float)(x%13-6)*0.0625f;
    }}
    for(int x=0;x<HEADS*M*N;++x)c[x]=__builtin_nanf("");
    for(int g=0;g<5;++g)for(int x=0;x<16;++x)guards[g][x]=0xa55a5aa5u;
    uint64_t start=nr_cycles();
    _mlir_ciface_kernel_{name}(&A,&B,&P,&C);
    uint64_t elapsed=nr_cycles()-start;
    /* The inputs repeat with phases 17 and 13. Computing the 221 possible
       reference dots once avoids duplicating millions of scalar oracle MACs
       on the FPGA; this is exactly the same elementwise independent sum. */
    float expected[17][13];
    for(int pa=0;pa<17;++pa)for(int pb=0;pb<13;++pb) {{
      double sum=0;
      for(int q=0;q<(QK?K:valid);++q)
        sum+=(double)((pa+q)%17-8)*0.125*(double)((pb+q*N)%13-6)*0.0625;
      expected[pa][pb]=(float)sum;
    }}
    unsigned before=errors;
    for(int g=0;g<5;++g)for(int x=0;x<16;++x)if(guards[g][x]!=0xa55a5aa5u)++errors;
    for(int row=0;row<M;++row)if(position[row]!=valid-M+row)++errors;
    for(int h=0;h<HEADS;++h)for(int row=0;row<M;++row)for(int col=0;col<N;++col) {{
      float want=(QK && col>=valid)?0:expected[((h*M+row)*K)%17][(h*K*N+col)%13];
      float got=c[(h*M+row)*N+col],difference=got-want;
      if(difference<0)difference=-difference;
      if(!__builtin_isfinite(got)||got!=want)++errors;
      if(!__builtin_isfinite(got))maximum=__builtin_inff();
      else if(difference>maximum)maximum=difference;
    }}
    nr_puts("[attention-position] valid=");nr_hex32(valid);
    nr_puts(" errors=");nr_hex32(errors-before);
    nr_puts(" kernel_cycles=");nr_hex64(elapsed);nr_puts("\\r\\n");
  }}
  return print_check("{name}",errors,maximum);
}}
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--boundary-probe",action="store_true",
                        help="small separate cases covering short interaction prompts and length23")
    parser.add_argument("--native-key",action="store_true",
                        help="QK only, reading original row-major key cache (no global transpose)")
    args=parser.parse_args()
    qwen=Path(__file__).resolve().parents[2]
    names=[]
    for sequence in (1,16):
        for qk in (True,False):
            if args.native_key and not qk:
                continue
            family="attention_qk" if qk else "attention_pv"
            n,k=(512,128) if qk else (128,512)
            name=f"{family}_position_16x{sequence}x{n}x{k}"
            if args.native_key:
                name=f"{family}_position_native_16x{sequence}x{n}x{k}"
            if args.boundary_probe:
                name += "_bounds"
            directory=(args.output/name).resolve()
            directory.mkdir(parents=True,exist_ok=True)
            meta=attention_metadata(name,family,(16,sequence,n,k),
                "Runtime valid length; physical capacity 512; NaN tail validation")
            meta.update(runtime_position=True,valid_lengths=valid_lengths(sequence,args.boundary_probe,args.native_key),
                        native_key_layout=args.native_key,
                        oracle="Independent double sums for all 17x13 input phases; all output elements checked exactly",
                        poison="NaN in invalid cache; both reduction operands poisoned for PV",
                        bounds="64-byte sentinel before/after every buffer; Position remains unchanged")
            (directory/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
            (directory/"launch.c").write_text(launcher(name,sequence,qk,args.boundary_probe,args.native_key))
            relative=os.path.relpath(qwen,directory)
            (directory/"makefile").write_text(
                "# Generated by attention_position_cases.py\n"
                f"override ROOT := $(abspath {relative})\n"
                "override COMMON := $(ROOT)/../common\n"
                "override REPO_ROOT := $(abspath $(ROOT)/../../..)\n"
                "override TOOLS := $(ROOT)/../tools\n"
                "include $(ROOT)/common.mk\n")
            names.append(name)
    (args.output/"cases.json").write_text(json.dumps({"cases":names},indent=2)+"\n")
    print("\n".join(names))


if __name__=="__main__":
    main()
