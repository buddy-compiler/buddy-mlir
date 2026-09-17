#include "support.h"
#define M 16
#define K 1024
extern void _mlir_ciface_kernel_quantize_16x1024(MemRef2*,MemRef2*,MemRef1*);
int launch(void) {
 float *x=workspace(0),*s=workspace(M*K*4); int8_t *q=workspace(M*K*4+64);
 unsigned errors=0;
 for(int trial=0;trial<2;trial++) {
 for(int i=0;i<M;i++) for(int j=0;j<K;j++) {
  float base=j==0?127.0f:j==1?-127.0f:(float)(j%17-8)*0.5f+(j%3)*0.125f;
  x[i*K+j]=(trial==0&&i%3==0)?0.0f:base*(i%2?1.137f:1.0f);
 }
 MemRef2 X=make_2(x,M,K),Q=make_2(q,M,K);MemRef1 S=make_1(s,M);
 _mlir_ciface_kernel_quantize_16x1024(&X,&Q,&S);
 for(int i=0;i<M;i++) {
  float max=0;
  for(int j=0;j<K;j++) {float v=x[i*K+j];if(v<0)v=-v;if(v>max)max=v;}
  float scale=max==0?1.0f:max/127.0f;
  if(!check_close(s[i],scale,1e-8f,1e-6f))errors++;
  for(int j=0;j<K;j++) {
   float v=x[i*K+j]/scale;
   int want=(int)(v+(v>=0?0.5f:-0.5f));if(want>127)want=127;if(want<-127)want=-127;
   if(q[i*K+j]!=want)errors++;
  }
 }
 }
 return print_check("quantize_16x1024",errors,0);
}
