#include "support.h"
#define M 1
#define N 128
#define K 17
extern void _mlir_ciface_kernel_attention_pv_16x1x128x17(MemRef3*,MemRef3*,MemRef3*);
int launch(void) {
  float *a=workspace(0),*b=workspace(16*M*K*4),*c=workspace(16*(M*K+K*N)*4);
  for(int x=0;x<16*M*K;x++) a[x]=(float)(x%17-8)*0.125f;
  for(int x=0;x<16*K*N;x++) b[x]=(float)(x%13-6)*0.0625f;
  for(int x=0;x<16*M*N;x++) c[x]=1234.0f;
  MemRef3 A=make_3(a,16,M,K),B=make_3(b,16,K,N),C=make_3(c,16,M,N);
  _mlir_ciface_kernel_attention_pv_16x1x128x17(&A,&B,&C);
  unsigned errors=0; float max_error=0;
  for(int h=0;h<16;h++) for(int i=0;i<M;i++) for(int j=0;j<N;j++) {
    double want=0;
    for(int q=0;q<K;q++) want+=(double)a[(h*M+i)*K+q]*(double)b[(h*K+q)*N+j];
    float got=c[(h*M+i)*N+j],d=got-(float)want; if(d<0)d=-d;
    if(d>max_error)max_error=d;
    if(!check_close(got,(float)want,1e-6f,1e-6f))errors++;
  }
  return print_check("attention_pv_16x1x128x17",errors,max_error);
}
