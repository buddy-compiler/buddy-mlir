/* Only the oracle is C. Timed functions are renamed, unmodified objects from
 * Triton -> Buddy, linked using the original descriptor adapters. */
#include "support.h"
#if Q_ROWS < 1 || Q_COLS < 1 || Q_REPEATS < 2
#error "Positive dimensions and alternating repetitions required"
#endif
#ifdef HOST_TEST
#define STORAGE __attribute__((aligned(64)))
#else
#define STORAGE NR_WORKSPACE
#endif
#define COUNT ((size_t)Q_ROWS * Q_COLS)
extern void qwen_quant_baseline(MemRef2 *, MemRef2 *, MemRef1 *);
extern void qwen_quant_optimized(MemRef2 *, MemRef2 *, MemRef1 *);
static struct { float pre[16], data[COUNT], post[16]; } input STORAGE;
static struct { int8_t pre[64], data[COUNT], post[64]; } output[2] STORAGE;
static struct { float pre[16], data[Q_ROWS], post[16]; } scales[2] STORAGE;
static uint32_t bits(float x) {
  union { float f; uint32_t u; } v = {x}; return v.u;
}
static float input_at(size_t i, unsigned round) {
  static const float cases[] = {-127.f, 127.f, -126.5f, 126.5f, -1.50001f,
      -1.5f, -0.5f, -0.f, 0.f, 0.49999f, 0.5f, 1.49999f, 1.5f};
  if ((i / Q_COLS + round) % 4 == 0) return 0.f;
  float factor = round & 1 ? 1.137f : 1.f;
  return cases[(i + round * 3) % (sizeof(cases) / sizeof(cases[0]))] * factor;
}
static float scale_at(unsigned row, unsigned round) {
  float maximum = 0.f;
  for (size_t j = 0; j < Q_COLS; ++j) {
    float x = input_at((size_t)row * Q_COLS + j, round);
    float a = x < 0.f ? -x : x;
    if (a > maximum) maximum = a;
  }
  return maximum == 0.f ? 1.f : maximum / 127.f;
}
static int8_t quant_at(size_t i, unsigned round, float scale) {
  volatile float d = input_at(i, round) / scale;
  volatile float r = d + (d >= 0.f ? 0.5f : -0.5f);
  float clamped = r < -127.f ? -127.f : r > 127.f ? 127.f : r;
  return (int8_t)(int32_t)clamped;
}
int launch(void) {
  unsigned errors = 0;
  nr_puts("[quant-ab] rows="); nr_hex32(Q_ROWS);
  nr_puts(" cols="); nr_hex32(Q_COLS); nr_puts("\r\n");
  for (unsigned round = 0; round < Q_REPEATS; ++round) {
    for (size_t i = 0; i < COUNT; ++i) input.data[i] = input_at(i, round);
    for (unsigned i = 0; i < 16; ++i) input.pre[i] = input.post[i] = -999.f;
    for (unsigned v = 0; v < 2; ++v) {
      for (size_t i = 0; i < COUNT; ++i) output[v].data[i] = -128;
      for (unsigned i = 0; i < Q_ROWS; ++i) scales[v].data[i] = -999.f;
      for (unsigned i = 0; i < 64; ++i) output[v].pre[i] = output[v].post[i] = -99;
      for (unsigned i = 0; i < 16; ++i) scales[v].pre[i] = scales[v].post[i] = -999.f;
    }
    for (unsigned pass = 0; pass < 2; ++pass) {
      unsigned v = pass ^ (round & 1);
      MemRef2 x = make_2(&input, Q_ROWS, Q_COLS);
      MemRef2 q = make_2(&output[v], Q_ROWS, Q_COLS);
      MemRef1 s = make_1(&scales[v], Q_ROWS);
      x.offset = s.offset = 16; q.offset = 64;
      __asm__ volatile("" ::: "memory");
      uint64_t start = nr_cycles();
      if (v) qwen_quant_optimized(&x, &q, &s);
      else qwen_quant_baseline(&x, &q, &s);
      uint64_t cycles = nr_cycles() - start;
      __asm__ volatile("" ::: "memory");
      unsigned wrong = 0;
      for (unsigned row = 0; row < Q_ROWS; ++row) {
        float expected = scale_at(row, round);
        wrong += bits(scales[v].data[row]) != bits(expected);
        for (size_t j = 0; j < Q_COLS; ++j) {
          size_t i = (size_t)row * Q_COLS + j;
          wrong += output[v].data[i] != quant_at(i, round, expected);
          wrong += bits(input.data[i]) != bits(input_at(i, round));
        }
      }
      for (unsigned i = 0; i < 16; ++i)
        wrong += input.pre[i] != -999.f || input.post[i] != -999.f;
      for (unsigned k = 0; k < 2; ++k) {
        for (unsigned i = 0; i < 64; ++i)
          wrong += output[k].pre[i] != -99 || output[k].post[i] != -99;
        for (unsigned i = 0; i < 16; ++i)
          wrong += scales[k].pre[i] != -999.f || scales[k].post[i] != -999.f;
      }
      errors += wrong;
      nr_puts("[quant-ab] round="); nr_hex32(round);
      nr_puts(v ? " variant=optimized" : " variant=baseline");
      nr_puts(" cycles="); nr_hex64(cycles);
      nr_puts(" errors="); nr_hex32(wrong); nr_puts("\r\n");
    }
    for (size_t i = 0; i < COUNT; ++i)
      errors += output[0].data[i] != output[1].data[i];
    for (unsigned i = 0; i < Q_ROWS; ++i)
      errors += bits(scales[0].data[i]) != bits(scales[1].data[i]);
  }
  nr_puts(errors ? "[quant-ab] FAIL errors=" : "[quant-ab] PASS errors=");
  nr_hex32(errors); nr_puts("\r\n");
  return errors != 0;
}
