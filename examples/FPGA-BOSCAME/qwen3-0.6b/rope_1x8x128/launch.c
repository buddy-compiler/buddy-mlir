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
extern void _mlir_ciface_kernel_rope_1x8x128(MemRef4 *, MemRef2 *, MemRef2 *, MemRef4 *);
int launch(void) {
  int errors = 0; float maximum = 0;
  float *x = (float *)workspace(0u);
  MemRef4 m_x = make_4(x, 1, 8, 2, 64);
  float *cosine = (float *)workspace(4096u);
  MemRef2 m_cosine = make_2(cosine, 1, 64);
  float *sine = (float *)workspace(4352u);
  MemRef2 m_sine = make_2(sine, 1, 64);
  float *out = (float *)workspace(4608u);
  MemRef4 m_out = make_4(out, 1, 8, 2, 64);
  for (int i=0; i<1024; ++i) { x[i]=(float)((i*5)%53-26)*0.03125f; out[i]=9999; }
  for (int s=0; s<1; ++s) for (int k=0; k<64; ++k) {
    float frequency=oracle_exp(-13.815510557964274f*(float)k/64.0f);
    float angle=(float)(s+7)*frequency; cosine[s*64+k]=oracle_cos(angle); sine[s*64+k]=oracle_sin(angle);
  }
  _mlir_ciface_kernel_rope_1x8x128(&m_x, &m_cosine, &m_sine, &m_out);
  for (int s=0; s<1; ++s) for (int h=0; h<8; ++h) for (int k=0; k<64; ++k) {
    int first=(s*8+h)*128+k, second=first+64;
    float a=x[first]*cosine[s*64+k]-x[second]*sine[s*64+k];
    float b=x[second]*cosine[s*64+k]+x[first]*sine[s*64+k];
    compare(out[first],a,2e-6f,3e-5f,&errors,&maximum); compare(out[second],b,2e-6f,3e-5f,&errors,&maximum);
  }
  return print_check("rope_1x8x128", errors, maximum);
}
