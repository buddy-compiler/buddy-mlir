/* Host oracle helpers only. Never included in the model or Triton archive. */
#include <stddef.h>

extern float fmaf(float, float, float);

extern float qwen_ref_nr_expf(float);
extern float qwen_ref_nr_sinf(float);
extern float qwen_ref_nr_cosf(float);

void qwen_ref_nr_unary(const float *input, float *output, size_t count, int op) {
  for (size_t i = 0; i < count; ++i)
    output[i] = op == 0 ? qwen_ref_nr_expf(input[i]) :
                op == 1 ? qwen_ref_nr_sinf(input[i]) : qwen_ref_nr_cosf(input[i]);
}

/* Independent scalar emulation of the observed RVV vfmacc.vf K order.
 * Each BK=64 partial starts from +0, then adds to the outer accumulator.
 * fmaf guarantees a single rounding even on hosts without native FMA. */
void qwen_ref_nr_dot(const float *left, const float *right, float *output,
                     size_t batches, size_t m, size_t n, size_t k, size_t bk) {
  for (size_t h = 0; h < batches; ++h)
    for (size_t row = 0; row < m; ++row)
      for (size_t column = 0; column < n; ++column) {
        float accumulated = 0;
        for (size_t base = 0; base < k; base += bk) {
          float partial = 0;
          size_t stop = base + bk < k ? base + bk : k;
          for (size_t reduction = base; reduction < stop; ++reduction)
            partial = fmaf(left[(h * m + row) * k + reduction],
                           right[(h * k + reduction) * n + column], partial);
          accumulated += partial;
        }
        output[(h * m + row) * n + column] = accumulated;
      }
}
