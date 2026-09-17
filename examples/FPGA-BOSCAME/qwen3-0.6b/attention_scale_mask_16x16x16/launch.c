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
extern void _mlir_ciface_kernel_attention_scale_mask_16x16x16(MemRef3 *, MemRef3 *);
int launch(void) {
  int errors = 0; float maximum = 0;
  float *x = (float *)workspace(0u);
  MemRef3 m_x = make_3(x, 16, 16, 16);
  float *out = (float *)workspace(16384u);
  MemRef3 m_out = make_3(out, 16, 16, 16);
  for(int i=0; i<4096; ++i) { x[i]=(float)((i*11)%61-30)*0.25f; out[i]=9999; }
  _mlir_ciface_kernel_attention_scale_mask_16x16x16(&m_x, &m_out);
  for (int h=0; h<16; ++h) for(int s=0; s<16; ++s) for(int t=0; t<16; ++t) {
    int i=(h*16+s)*16+t; float expected=t<=s+0?x[i]/oracle_sqrt(128.0f):-__builtin_inff();
    compare(out[i],expected,1e-6f,2e-6f,&errors,&maximum);
  }
  return print_check("attention_scale_mask_16x16x16", errors, maximum);
}
