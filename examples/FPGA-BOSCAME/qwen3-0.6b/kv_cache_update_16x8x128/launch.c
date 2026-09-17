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
extern void _mlir_ciface_kernel_kv_cache_update_16x8x128(MemRef3 *, MemRef3 *);
int launch(void) {
  int errors = 0; float maximum = 0;
  float *x = (float *)workspace(0u);
  MemRef3 m_x = make_3(x, 16, 8, 128);
  float *cache = (float *)workspace(65536u);
  MemRef3 m_cache = make_3(cache, 8, 32, 128);
  for(int i=0;i<16384;++i) x[i]=(float)((i*17)%101-50)*0.0625f;
  for(int i=0;i<32768;++i) cache[i]=(float)((i*3)%47-23)*0.03125f;
  _mlir_ciface_kernel_kv_cache_update_16x8x128(&m_x, &m_cache);
  for(int h=0;h<8;++h) for(int t=0;t<32;++t) for(int d=0;d<128;++d) {
    int i=(h*32+t)*128+d; float expected=(float)((i*3)%47-23)*0.03125f;
    if(t>=0 && t<16) expected=x[((t-0)*8+h)*128+d];
    compare(cache[i],expected,0,0,&errors,&maximum);
  }
  return print_check("kv_cache_update_16x8x128", errors, maximum);
}
