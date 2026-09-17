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
extern void _mlir_ciface_kernel_softmax_16x16x16(MemRef3 *, MemRef2 *, MemRef2 *, MemRef3 *);
int launch(void) {
  int errors = 0; float maximum = 0;
  float *x = (float *)workspace(0u);
  MemRef3 m_x = make_3(x, 16, 16, 16);
  float *maxima = (float *)workspace(16384u);
  MemRef2 m_maxima = make_2(maxima, 16, 16);
  float *sums = (float *)workspace(17408u);
  MemRef2 m_sums = make_2(sums, 16, 16);
  float *out = (float *)workspace(18432u);
  MemRef3 m_out = make_3(out, 16, 16, 16);
  for (int row=0; row<256; ++row) for (int k=0; k<16; ++k) {
    int i=row*16+k; x[i]=80.0f+(float)((i*7)%53-26)*0.25f; out[i]=9999;
    if (k>row%16) x[i]=-__builtin_inff();
  }
  _mlir_ciface_kernel_softmax_16x16x16(&m_x, &m_maxima, &m_sums, &m_out);
  for (int row=0; row<256; ++row) {
    float maximum_input=-__builtin_inff(); oracle_float sum=0;
    for (int k=0; k<16; ++k) if(x[row*16+k]>maximum_input) maximum_input=x[row*16+k];
    for (int k=0; k<16; ++k) sum+=oracle_exp(x[row*16+k]-maximum_input);
    oracle_float output_sum=0;
    for (int k=0; k<16; ++k) { int i=row*16+k; float expected=(float)(oracle_exp(x[i]-maximum_input)/sum); compare(out[i],expected,2e-6f,5e-5f,&errors,&maximum); output_sum+=out[i]; }
    compare((float)output_sum,1.0f,3e-6f,3e-6f,&errors,&maximum);
  }
  return print_check("softmax_16x16x16", errors, maximum);
}
