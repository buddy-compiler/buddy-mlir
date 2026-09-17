#include "support.h"
#define M 1
#define N 1024
extern void _mlir_ciface_kernel_dequantize_1x1024(MemRef2*,MemRef1*,MemRef1*,MemRef2*);
int launch(void) {
 int32_t *x=workspace(0);float *r=workspace(M*N*4),*s=workspace(M*N*4+64),*y=workspace(M*N*4+64+N*4);
 for(int i=0;i<M;i++)r[i]=(i+1)*0.00390625f;
 for(int j=0;j<N;j++)s[j]=(j%31+1)*0.001953125f;
 for(int i=0;i<M*N;i++)x[i]=(i%1023-511)*7919;
 MemRef2 X=make_2(x,M,N),Y=make_2(y,M,N);MemRef1 R=make_1(r,M),S=make_1(s,N);
 _mlir_ciface_kernel_dequantize_1x1024(&X,&R,&S,&Y);
 unsigned errors=0;float max_error=0;
 for(int i=0;i<M;i++)for(int j=0;j<N;j++){
  float want=(float)x[i*N+j]*r[i]*s[j],d=y[i*N+j]-want;if(d<0)d=-d;if(d>max_error)max_error=d;
  if(!check_close(y[i*N+j],want,1e-6f,1e-6f))errors++;
 }
 return print_check("dequantize_1x1024",errors,max_error);
}
