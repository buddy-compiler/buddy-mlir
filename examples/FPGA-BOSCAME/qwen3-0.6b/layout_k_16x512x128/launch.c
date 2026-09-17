#include "support.h"
#ifdef HOST_TEST
typedef double oracle_float;
static float oracle_exp(float x) { return (float)exp((double)x); }
static float oracle_sqrt(float x) { return (float)sqrt((double)x); }
static float oracle_sin(float x) { return (float)sin((double)x); }
static float oracle_cos(float x) { return (float)cos((double)x); }
#else
typedef float oracle_float;
static float oracle_exp(float x) { return expf(x); }
static float oracle_sqrt(float x) { return sqrtf(x); }
static float oracle_sin(float x) { return sinf(x); }
static float oracle_cos(float x) { return cosf(x); }
#endif
static void compare(float actual, float expected, float atol, float rtol,
                    int *errors, float *maximum) {
  if (actual == expected) return;
  if (!__builtin_isfinite(actual) || !__builtin_isfinite(expected)) {
    ++*errors; *maximum = __builtin_inff(); return;
  }
  if (!check_close(actual, expected, atol, rtol)) ++*errors;
  float difference = actual - expected;
  if (difference < 0) difference = -difference;
  if (difference > *maximum) *maximum = difference;
}
extern void _mlir_ciface_kernel_layout_k_16x512x128(MemRef3 *, MemRef3 *);
int launch(void) {
  int errors = 0; float maximum = 0;
  float *x = (float *)workspace(0u);
  MemRef3 m_x = make_3(x, 16, 512, 128);
  float *out = (float *)workspace(4194304u);
  MemRef3 m_out = make_3(out, 16, 128, 512);
  for(int i=0;i<1048576;++i) { x[i]=(float)((i*19)%4093-2046)*0.015625f; out[i]=9999; }
  _mlir_ciface_kernel_layout_k_16x512x128(&m_x, &m_out);
  for(int a=0;a<16;++a) for(int b=0;b<512;++b) for(int d=0;d<128;++d) {
    int index[3]={a,b,d};
    int input=(a*512+b)*128+d;
    int output=(index[0]*128+index[2])*512+index[1];
    compare(out[output],x[input],0,0,&errors,&maximum);
  }
  return print_check("layout_k_16x512x128", errors, maximum);
}
