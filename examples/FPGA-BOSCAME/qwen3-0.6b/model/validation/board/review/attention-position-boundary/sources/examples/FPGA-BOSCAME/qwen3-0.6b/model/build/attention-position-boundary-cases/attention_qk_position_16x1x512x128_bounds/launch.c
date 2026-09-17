#include "support.h"
#define HEADS 16
#define M 1
#define N 512
#define K 128
#define QK 1
extern void _mlir_ciface_kernel_attention_qk_position_16x1x512x128_bounds(MemRef3*,MemRef3*,MemRef1*,MemRef3*);
int launch(void) {
  float *a=workspace(64),*b=workspace(128+HEADS*M*K*4);
  float *c=workspace(192+HEADS*(M*K+K*N)*4);
  int *position=workspace(256+HEADS*(M*K+K*N+M*N)*4);
  uint32_t *guards[]={workspace(0),(uint32_t*)(a+HEADS*M*K),
                     (uint32_t*)(b+HEADS*K*N),(uint32_t*)(c+HEADS*M*N),
                     (uint32_t*)(position+M)};
  MemRef3 A=make_3(a,HEADS,M,K),B=make_3(b,HEADS,K,N),C=make_3(c,HEADS,M,N);
  MemRef1 P=make_1(position,M);
  const int lengths[]={1,2,23,2,1};
  unsigned errors=0; float maximum=0;
  for(unsigned step=0;step<sizeof(lengths)/sizeof(lengths[0]);++step) {
    int valid=lengths[step];
    for(int row=0;row<M;++row)position[row]=valid-M+row;
    for(int h=0;h<HEADS;++h)for(int row=0;row<M;++row)for(int q=0;q<K;++q) {
      int x=(h*M+row)*K+q;
      a[x]=(!QK && q>=valid)?__builtin_nanf(""):(float)(x%17-8)*0.125f;
    }
    for(int h=0;h<HEADS;++h)for(int q=0;q<K;++q)for(int col=0;col<N;++col) {
      int x=(h*K+q)*N+col;
      b[x]=((QK && col>=valid)||(!QK && q>=valid))?__builtin_nanf(""):(float)(x%13-6)*0.0625f;
    }
    for(int x=0;x<HEADS*M*N;++x)c[x]=__builtin_nanf("");
    for(int g=0;g<5;++g)for(int x=0;x<16;++x)guards[g][x]=0xa55a5aa5u;
    uint64_t start=nr_cycles();
    _mlir_ciface_kernel_attention_qk_position_16x1x512x128_bounds(&A,&B,&P,&C);
    uint64_t elapsed=nr_cycles()-start;
    /* The inputs repeat with phases 17 and 13. Computing the 221 possible
       reference dots once avoids duplicating millions of scalar oracle MACs
       on the FPGA; this is exactly the same elementwise independent sum. */
    float expected[17][13];
    for(int pa=0;pa<17;++pa)for(int pb=0;pb<13;++pb) {
      double sum=0;
      for(int q=0;q<(QK?K:valid);++q)
        sum+=(double)((pa+q)%17-8)*0.125*(double)((pb+q*N)%13-6)*0.0625;
      expected[pa][pb]=(float)sum;
    }
    unsigned before=errors;
    for(int g=0;g<5;++g)for(int x=0;x<16;++x)if(guards[g][x]!=0xa55a5aa5u)++errors;
    for(int row=0;row<M;++row)if(position[row]!=valid-M+row)++errors;
    for(int h=0;h<HEADS;++h)for(int row=0;row<M;++row)for(int col=0;col<N;++col) {
      float want=(QK && col>=valid)?0:expected[((h*M+row)*K)%17][(h*K*N+col)%13];
      float got=c[(h*M+row)*N+col],difference=got-want;
      if(difference<0)difference=-difference;
      if(!__builtin_isfinite(got)||got!=want)++errors;
      if(!__builtin_isfinite(got))maximum=__builtin_inff();
      else if(difference>maximum)maximum=difference;
    }
    nr_puts("[attention-position] valid=");nr_hex32(valid);
    nr_puts(" errors=");nr_hex32(errors-before);
    nr_puts(" kernel_cycles=");nr_hex64(elapsed);nr_puts("\r\n");
  }
  return print_check("attention_qk_position_16x1x512x128_bounds",errors,maximum);
}
