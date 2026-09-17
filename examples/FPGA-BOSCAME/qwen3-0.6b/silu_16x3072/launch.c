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
extern void _mlir_ciface_kernel_silu_16x3072(MemRef2 *, MemRef2 *);
int launch(void) {
  int errors = 0; float maximum = 0;
  float *x = (float *)workspace(0u);
  MemRef2 m_x = make_2(x, 16, 3072);
  float *out = (float *)workspace(196608u);
  MemRef2 m_out = make_2(out, 16, 3072);
  for (int i=0; i<49152; ++i) { x[i] = (float)((i*13)%129-64)*0.125f; out[i] = 9999.0f; }
  _mlir_ciface_kernel_silu_16x3072(&m_x, &m_out);
  for (int i=0; i<49152; ++i) compare(out[i], x[i] / (1.0f + oracle_exp(-x[i])), 2e-6f, 3e-5f, &errors, &maximum);
  return print_check("silu_16x3072", errors, maximum);
}
