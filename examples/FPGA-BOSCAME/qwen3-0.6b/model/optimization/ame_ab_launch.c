/* Validate real Triton INT8 linear kernels. C arithmetic is only the oracle.
 * The shared Triton linear semantic is C += A@B; B is physically [N,K]. */
#include "support.h"
#ifndef AM_M
#define AM_M 1
#endif
#ifndef AM_N
#define AM_N 151936
#endif
#ifndef AM_K
#define AM_K 1024
#endif
#ifndef AM_REPEATS
#define AM_REPEATS 2
#endif
#ifndef AM_FULL_INPUT_SCAN
#define AM_FULL_INPUT_SCAN 0
#endif
#if AM_M < 1 || AM_N < 1 || AM_K < 1 || AM_K > 65535 || AM_REPEATS < 2
#error "positive dimensions, K<=65535, and at least two rounds required"
#endif
#ifdef HOST_TEST
#define AM_STORAGE __attribute__((aligned(64)))
static void synchronize(void) {}
#else
#define AM_STORAGE NR_WORKSPACE
static void synchronize(void) { ame_fence(); }
#endif
#define A_COUNT ((size_t)AM_M * AM_K)
#define B_COUNT ((size_t)AM_N * AM_K)
#define C_COUNT ((size_t)AM_M * AM_N)
#define TEMPLATES 32

extern void qwen_ame_baseline(MemRef2 *, MemRef2 *, MemRef2 *);
extern void qwen_ame_optimized(MemRef2 *, MemRef2 *, MemRef2 *);
typedef void (*Kernel)(MemRef2 *, MemRef2 *, MemRef2 *);
static struct { uint8_t pre[64]; int8_t data[A_COUNT]; uint8_t post[64]; } a AM_STORAGE;
static struct { uint8_t pre[64]; int8_t data[B_COUNT]; uint8_t post[64]; } b AM_STORAGE;
static struct { uint32_t pre[16]; int32_t data[C_COUNT]; uint32_t post[16]; } c[2] AM_STORAGE;
static int8_t templates[TEMPLATES][AM_K] AM_STORAGE;
static int64_t dots[AM_M][TEMPLATES] AM_STORAGE;

static int8_t a_value(size_t row, size_t k, unsigned round) {
  if ((k + round) % 19 == 0) return -128;
  return (int8_t)((int)((k * 11 + row * 31 + round * 57) % 255) - 127);
}
static int8_t b_value(size_t pattern, size_t k, unsigned round) {
  if ((k + pattern + round) % 23 == 0) return -128;
  return (int8_t)((int)((k * 17 + pattern * 29 + round * 43 + k / 13) % 255) - 127);
}
static size_t column_pattern(size_t column, unsigned round) {
  return (column * 13 + column / 32 + column / 1024 + round * 7) % TEMPLATES;
}
static int32_t initial_c(size_t index, unsigned round) {
  /* Nonzero values ensure a missing/extra accumulator load is observable. */
  return (int32_t)((index * 48271 + (size_t)round * 104729) % 65521) - 32760;
}
static void guard8(uint8_t *pre, uint8_t *post) {
  for (unsigned i = 0; i < 64; ++i) pre[i] = post[i] = (uint8_t)(0x5a ^ i);
}
static unsigned bad8(const uint8_t *pre, const uint8_t *post) {
  unsigned errors = 0;
  for (unsigned i = 0; i < 64; ++i)
    errors += pre[i] != (uint8_t)(0x5a ^ i) || post[i] != (uint8_t)(0x5a ^ i);
  return errors;
}
static void guard32(uint32_t *pre, uint32_t *post) {
  for (unsigned i = 0; i < 16; ++i) pre[i] = post[i] = UINT32_C(0x95c31e6a) ^ i;
}
static unsigned bad32(const uint32_t *pre, const uint32_t *post) {
  unsigned errors = 0;
  for (unsigned i = 0; i < 16; ++i)
    errors += pre[i] != (UINT32_C(0x95c31e6a) ^ i) || post[i] != (UINT32_C(0x95c31e6a) ^ i);
  return errors;
}
static void initialize(unsigned round) {
  guard8(a.pre, a.post); guard8(b.pre, b.post);
  for (size_t row = 0; row < AM_M; ++row)
    for (size_t k = 0; k < AM_K; ++k) a.data[row * AM_K + k] = a_value(row, k, round);
  for (size_t pattern = 0; pattern < TEMPLATES; ++pattern)
    for (size_t k = 0; k < AM_K; ++k) templates[pattern][k] = b_value(pattern, k, round);
  /* Data preparation uses the already validated common RVV copy. It is outside
   * all kernel timing, and no host-generated inference result is involved. */
  for (size_t column = 0; column < AM_N; ++column)
    nr_copy_bytes(b.data + column * AM_K, templates[column_pattern(column, round)], AM_K);
  for (unsigned variant = 0; variant < 2; ++variant) {
    guard32(c[variant].pre, c[variant].post);
    for (size_t i = 0; i < C_COUNT; ++i) c[variant].data[i] = initial_c(i, round);
  }
  /* Exact independent oracle O(M*K*32), not a full scalar model-sized matmul.
   * Values are recomputed from formulas rather than reading the kernel inputs. */
  for (size_t row = 0; row < AM_M; ++row)
    for (size_t pattern = 0; pattern < TEMPLATES; ++pattern) {
      int64_t dot = 0;
      for (size_t k = 0; k < AM_K; ++k)
        dot += (int64_t)a_value(row, k, round) * b_value(pattern, k, round);
      dots[row][pattern] = dot;
    }
}
static unsigned input_errors(unsigned round) {
  unsigned errors = bad8(a.pre, a.post) + bad8(b.pre, b.post);
  for (size_t row = 0; row < AM_M; ++row)
    for (size_t k = 0; k < AM_K; ++k) errors += a.data[row * AM_K + k] != a_value(row, k, round);
  for (size_t column = 0; column < AM_N; ++column) {
    size_t pattern = column_pattern(column, round);
    if (AM_FULL_INPUT_SCAN || column < 32 || column + 32 >= AM_N || column % 1024 == 0) {
      for (size_t k = 0; k < AM_K; ++k)
        errors += b.data[column * AM_K + k] != b_value(pattern, k, round);
    } else {
      /* Every B row is sampled; boundary/template rows are checked in full. */
      for (size_t sample = 0; sample < 4; ++sample) {
        size_t k = sample == 3 ? AM_K - 1 : (column * 17 + sample * 257 + round) % AM_K;
        errors += b.data[column * AM_K + k] != b_value(pattern, k, round);
      }
    }
  }
  return errors;
}
static void f32(const char *name, uint32_t value) { nr_puts(name); nr_hex32(value); }
static void f64(const char *name, uint64_t value) { nr_puts(name); nr_hex64(value); }
static uint64_t double_bits(double value) {
  union { double f; uint64_t u; } v = {value}; return v.u;
}

int launch(void) {
  unsigned total_errors = 0;
  nr_puts("[ame-ab] config"); f32(" m=", AM_M); f32(" n=", AM_N); f32(" k=", AM_K);
  f32(" repeats=", AM_REPEATS);
  nr_puts(" semantic=accumulate input_check=");
  nr_puts(AM_FULL_INPUT_SCAN ? "full" : "full_A_sampled_B");
  nr_puts(" descriptor_offset_bytes=00000040\r\n");
  for (unsigned round = 0; round < AM_REPEATS; ++round) {
    nr_puts("[ame-ab] prepare"); f32(" round=", round); nr_puts("\r\n");
    initialize(round);
    MemRef2 A = make_2(&a, AM_M, AM_K), B = make_2(&b, AM_K, AM_N);
    A.offset = B.offset = 64;
    B.strides[0] = 1; B.strides[1] = AM_K;
    for (unsigned pass = 0; pass < 2; ++pass) {
      unsigned variant = pass ^ (round & 1);
      Kernel kernel = variant ? qwen_ame_optimized : qwen_ame_baseline;
      MemRef2 C = make_2(&c[variant], AM_M, AM_N); C.offset = 16;
      uint64_t begin = nr_cycles(); synchronize();
      uint64_t call_begin = nr_cycles();
      kernel(&A, &B, &C);
      uint64_t call_end = nr_cycles(); synchronize();
      uint64_t done = nr_cycles();
      unsigned errors = 0;
      uint64_t max_error = 0, sum_error = 0;
      for (size_t row = 0; row < AM_M; ++row)
        for (size_t column = 0; column < AM_N; ++column) {
          size_t i = row * AM_N + column;
          int64_t expected = dots[row][column_pattern(column, round)] + initial_c(i, round);
          int64_t delta = (int64_t)c[variant].data[i] - expected;
          if (delta < 0) delta = -delta;
          errors += delta != 0;
          if ((uint64_t)delta > max_error) max_error = (uint64_t)delta;
          sum_error += (uint64_t)delta;
        }
      unsigned guard_errors = bad32(c[0].pre, c[0].post) + bad32(c[1].pre, c[1].post);
      unsigned mutated_inputs = input_errors(round);
      total_errors += errors + guard_errors + mutated_inputs;
      nr_puts("[ame-ab] sample"); f32(" round=", round);
      nr_puts(variant ? " variant=optimized" : " variant=baseline");
      f32(" order=", pass); f64(" cycles=", done - call_begin);
      f64(" pre_sync_cycles=", call_begin - begin);
      f64(" adapter_kernel_cycles=", call_end - call_begin);
      f64(" post_sync_cycles=", done - call_end);
      f32(" mismatches=", errors); f32(" guard_errors=", guard_errors);
      f32(" input_errors=", mutated_inputs); f64(" max_abs_i64=", max_error);
      f64(" mean_abs_f64_bits=", double_bits((double)sum_error / C_COUNT));
      nr_puts(errors + guard_errors + mutated_inputs ? " status=FAIL\r\n" : " status=PASS\r\n");
    }
    unsigned mismatches = 0;
    for (size_t i = 0; i < C_COUNT; ++i) mismatches += c[0].data[i] != c[1].data[i];
    total_errors += mismatches;
    nr_puts("[ame-ab] pair"); f32(" round=", round); f32(" mismatches=", mismatches); nr_puts("\r\n");
  }
  nr_puts(total_errors ? "[ame-ab] FAIL" : "[ame-ab] PASS");
  f32(" errors=", total_errors); nr_puts("\r\n");
  return total_errors != 0;
}
