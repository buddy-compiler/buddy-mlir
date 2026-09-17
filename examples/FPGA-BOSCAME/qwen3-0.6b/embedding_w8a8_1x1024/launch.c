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
extern void _mlir_ciface_kernel_embedding_w8a8_1x1024(MemRef1 *, MemRef2 *, MemRef2 *, MemRef1 *);
int launch(void) {
  int errors = 0; float maximum = 0;
  int64_t *ids = (int64_t *)workspace(0u);
  MemRef1 m_ids = make_1(ids, 1);
  float *out = (float *)workspace(64u);
  MemRef2 m_out = make_2(out, 1, 1024);
  int8_t *weight = (int8_t *)workspace(4160u);
  MemRef2 m_weight = make_2(weight, 151936, 1024);
  float *scale = (float *)workspace(155586624u);
  MemRef1 m_scale = make_1(scale, 151936);
  for (int s=0; s<1; ++s) {
    ids[s]=(s%4==0)?151935:((s*997+3)%151936);
    scale[ids[s]]=(float)((ids[s]%13)+1)*0.0078125f;
    for(int k=0;k<1024;++k) { weight[ids[s]*1024+k]=(int8_t)(((ids[s]+k*3)%255)-127); out[s*1024+k]=9999; }
  }
  _mlir_ciface_kernel_embedding_w8a8_1x1024(&m_ids, &m_out, &m_weight, &m_scale);
  for(int s=0;s<1;++s) for(int k=0;k<1024;++k) {
    float expected=(float)(int8_t)(((ids[s]+k*3)%255)-127)*scale[ids[s]]; compare(out[s*1024+k],expected,0,0,&errors,&maximum);
  }
  return print_check("embedding_w8a8_1x1024", errors, maximum);
}
