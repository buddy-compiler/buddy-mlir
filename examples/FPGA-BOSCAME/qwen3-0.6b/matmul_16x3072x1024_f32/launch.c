#include "support.h"
#define M 16
#define N 3072
#define K 1024
extern void _mlir_ciface_kernel_matmul_16x3072x1024_f32(MemRef2 *, MemRef2 *, MemRef2 *);
int launch(void) {
  float *a = workspace(0), *b = workspace(((M*K*4)+63)&~63);
  float *c = workspace((((size_t)M*K+(size_t)N*K)*4+127)&~(size_t)63);
  /* Distinct row/column patterns exercise signs, K accumulation and tile tails. */
  for (int i=0;i<M;i++) for(int q=0;q<K;q++) a[i*K+q]=(i*3+q*5)%13-6;
  for (int j=0;j<N && j<11;j++) for(int q=0;q<K;q++) b[(size_t)j*K+q]=(j*7+q*3)%11-5;
  /* The same independently checked data repeat every eleven rows. Reuse
   * those rows through the NR RVV copy instead of scalar modulo per byte. */
  for (int j=11;j<N;j++) nr_copy_bytes(b+(size_t)j*K,b+(size_t)(j%11)*K,K*sizeof(*b));
  for (int i=0;i<M;i++) for(int j=0;j<N;j++) c[(size_t)i*N+j]=(i+j)%7-3;
  MemRef2 A=make_2(a,M,K), B=make_2(b,N,K), C=make_2(c,M,N);
  uint64_t start=nr_cycles();
  _mlir_ciface_kernel_matmul_16x3072x1024_f32(&A,&B,&C);
  nr_puts("cycles matmul_16x3072x1024_f32: "); nr_hex64(nr_cycles()-start); nr_puts("\r\n");
  /* An independent CPU dot oracle computes all 143 residue combinations,
   * then checks EVERY output. This avoids repeating billions of scalar MACs. */
  int32_t oracle[13][11];
  for(int x=0;x<13;x++) for(int y=0;y<11;y++) {
    int32_t dot=0;
    for(int q=0;q<K;q++) dot+=((x+q*5)%13-6)*((y+q*3)%11-5);
    oracle[x][y]=dot;
  }
  unsigned errors=0; float max_error=0;
  for(int i=0;i<M;i++) for(int j=0;j<N;j++) {
    int32_t want=oracle[(i*3)%13][(j*7)%11]+(i+j)%7-3;
    float got=c[(size_t)i*N+j];
    if(got!=want) {
      if(errors<4) { nr_puts("mismatch row/col/got/want "); nr_hex32(i);nr_puts(" ");nr_hex32(j);nr_puts(" ");nr_hex32(got);nr_puts(" ");nr_hex32(want);nr_puts("\r\n"); }
      errors++;
      float d=(float)got-(float)want; if(d<0)d=-d; if(d>max_error)max_error=d;
    }
  }
  return print_check("matmul_16x3072x1024_f32",errors,max_error);
}
