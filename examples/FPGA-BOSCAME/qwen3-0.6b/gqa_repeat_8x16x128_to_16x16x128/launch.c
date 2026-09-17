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
extern void _mlir_ciface_kernel_gqa_repeat_8x16x128_to_16x16x128(MemRef3 *, MemRef3 *);
int launch(void) {
  int errors = 0; float maximum = 0;
  float *x = (float *)workspace(0u);
  MemRef3 m_x = make_3(x, 8, 16, 128);
  float *out = (float *)workspace(65536u);
  MemRef3 m_out = make_3(out, 16, 16, 128);
  for(int i=0;i<16384;++i) x[i]=(float)((i*11)%127-63)*0.03125f;
  for(int i=0;i<32768;++i) out[i]=9999;
  _mlir_ciface_kernel_gqa_repeat_8x16x128_to_16x16x128(&m_x, &m_out);
  for(int h=0;h<16;++h) for(int t=0;t<16;++t) for(int d=0;d<128;++d) {
    compare(out[(h*16+t)*128+d],x[((h/2)*16+t)*128+d],0,0,&errors,&maximum);
  }
  return print_check("gqa_repeat_8x16x128_to_16x16x128", errors, maximum);
}
