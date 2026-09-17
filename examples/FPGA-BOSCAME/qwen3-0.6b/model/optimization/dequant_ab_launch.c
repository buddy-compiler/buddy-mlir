/* A/B test of real Triton kernels through their existing descriptor adapters.
 * Arithmetic below is only the independent numerical oracle, never a kernel
 * fallback. Compile with -ffp-contract=off and no fast-math/vectorization. */
#include "support.h"

#ifndef DQ_ROWS
#define DQ_ROWS 1
#endif
#ifndef DQ_COLS
#define DQ_COLS 151936
#endif
#ifndef DQ_REPEATS
#define DQ_REPEATS 3
#endif
#if DQ_ROWS < 1 || DQ_COLS < 1 || DQ_REPEATS < 2
#error "Positive rows/columns and at least two alternating repetitions required"
#endif
#ifdef HOST_TEST
#define DQ_STORAGE __attribute__((aligned(64)))
#else
#define DQ_STORAGE NR_WORKSPACE
#endif
#define COUNT ((size_t)DQ_ROWS * DQ_COLS)
#define GUARD_WORDS 16
#define GUARD_BITS UINT32_C(0x5a39c76d)
#define POISON_BITS UINT32_C(0x7fcabcde)

extern void qwen_dequant_baseline(MemRef2 *, MemRef1 *, MemRef1 *, MemRef2 *);
extern void qwen_dequant_optimized(MemRef2 *, MemRef1 *, MemRef1 *, MemRef2 *);
typedef void (*Kernel)(MemRef2 *, MemRef1 *, MemRef1 *, MemRef2 *);

static struct {
  uint32_t before[GUARD_WORDS];
  int32_t data[COUNT];
  uint32_t after[GUARD_WORDS];
} input DQ_STORAGE;
static struct {
  uint32_t before[GUARD_WORDS];
  float data[DQ_ROWS];
  uint32_t after[GUARD_WORDS];
} row DQ_STORAGE;
static struct {
  uint32_t before[GUARD_WORDS];
  float data[DQ_COLS];
  uint32_t after[GUARD_WORDS];
} column DQ_STORAGE;
static struct {
  uint32_t before[GUARD_WORDS];
  float data[COUNT];
  uint32_t after[GUARD_WORDS];
} output[2] DQ_STORAGE;

static uint32_t bits(float value) {
  union { float f; uint32_t u; } v = {value};
  return v.u;
}
static float from_bits(uint32_t value) {
  union { uint32_t u; float f; } v = {value};
  return v.f;
}
static uint64_t double_bits(double value) {
  union { double f; uint64_t u; } v = {value};
  return v.u;
}
static int32_t input_at(size_t i, unsigned round) {
  /* Include the FP32 exact-integer boundary, tie rounding, and INT32 extrema. */
  static const int32_t cases[] = {
      0, 1, -1, 127, -127, 65537, -65537, 16777215, -16777215,
      16777216, -16777216, 16777217, -16777217, 16777219, -16777219,
      INT32_MAX, INT32_MIN, INT32_MAX - 64, INT32_MIN + 64,
      123456789, -987654321, 8388607, -8388609};
  return cases[(i + (size_t)round * 11) % (sizeof(cases) / sizeof(cases[0]))];
}
static float row_at(size_t i, unsigned round) {
  /* Positive, finite normal scales. Hex literals fix the exact input bits. */
  static const float scales[] = {
      0x1.2b020cp-7f, 0x1.020408p-7f, 0x1.6a09e6p-4f,
      0x1.abcde0p-9f, 0x1.234568p-2f};
  return scales[(i + round) % 5];
}
static float column_at(size_t i, unsigned round) {
  static const float scales[] = {
      0x1.45678ap-6f, 0x1.fedcbap-10f, 0x1.3b645ap-3f,
      0x1.010102p-8f, 0x1.789abcp-5f, 0x1.234568p-12f,
      0x1.333334p-4f};
  return scales[(i * 3 + (size_t)round * 2) % 7];
}
static float oracle(size_t i, unsigned round) {
  /* Force the documented three FP32 rounding points. In particular do not
   * reassociate to float(acc) * (row_scale * column_scale). */
  volatile float converted = (float)input_at(i, round);
  volatile float scaled_row = converted * row_at(i / DQ_COLS, round);
  volatile float result = scaled_row * column_at(i % DQ_COLS, round);
  return result;
}
static void set_guards(uint32_t *before, uint32_t *after) {
  for (unsigned i = 0; i < GUARD_WORDS; ++i)
    before[i] = after[i] = GUARD_BITS ^ i;
}
static unsigned guards_bad(const uint32_t *before, const uint32_t *after) {
  unsigned bad = 0;
  for (unsigned i = 0; i < GUARD_WORDS; ++i)
    bad += before[i] != (GUARD_BITS ^ i) || after[i] != (GUARD_BITS ^ i);
  return bad;
}
static void initialize(unsigned round) {
  set_guards(input.before, input.after);
  set_guards(row.before, row.after);
  set_guards(column.before, column.after);
  for (unsigned k = 0; k < 2; ++k) {
    set_guards(output[k].before, output[k].after);
    for (size_t i = 0; i < COUNT; ++i)
      output[k].data[i] = from_bits(POISON_BITS + k);
  }
  for (size_t i = 0; i < COUNT; ++i) input.data[i] = input_at(i, round);
  for (size_t i = 0; i < DQ_ROWS; ++i) row.data[i] = row_at(i, round);
  for (size_t i = 0; i < DQ_COLS; ++i) column.data[i] = column_at(i, round);
}
static unsigned inputs_bad(unsigned round) {
  unsigned bad = guards_bad(input.before, input.after)
               + guards_bad(row.before, row.after)
               + guards_bad(column.before, column.after);
  for (size_t i = 0; i < COUNT; ++i) bad += input.data[i] != input_at(i, round);
  for (size_t i = 0; i < DQ_ROWS; ++i) bad += bits(row.data[i]) != bits(row_at(i, round));
  for (size_t i = 0; i < DQ_COLS; ++i) bad += bits(column.data[i]) != bits(column_at(i, round));
  return bad;
}
static void field32(const char *name, uint32_t value) {
  nr_puts(name); nr_hex32(value);
}
static void field64(const char *name, uint64_t value) {
  nr_puts(name); nr_hex64(value);
}

int launch(void) {
  unsigned total_errors = 0;
  nr_puts("[dequant-ab] config");
  field32(" rows=", DQ_ROWS); field32(" cols=", DQ_COLS);
  field32(" repeats=", DQ_REPEATS);
  nr_puts(" offset_words=00000010 timing=adapter_plus_kernel\r\n");
  for (unsigned round = 0; round < DQ_REPEATS; ++round) {
    initialize(round);
    /* Nonzero descriptor offsets exercise the production adapter. The actual
     * data starts 64 bytes after aligned, with checked guards on both sides. */
    MemRef2 x = make_2(&input, DQ_ROWS, DQ_COLS);
    MemRef1 r = make_1(&row, DQ_ROWS), c = make_1(&column, DQ_COLS);
    x.offset = r.offset = c.offset = GUARD_WORDS;
    for (unsigned pass = 0; pass < 2; ++pass) {
      unsigned variant = pass ^ (round & 1);
      Kernel kernel = variant ? qwen_dequant_optimized : qwen_dequant_baseline;
      MemRef2 y = make_2(&output[variant], DQ_ROWS, DQ_COLS);
      y.offset = GUARD_WORDS;
      /* The existing adapters do not add an AME fence for this RVV-only
       * operation. Keep that policy; these are compiler barriers only. */
      __asm__ volatile("" ::: "memory");
      uint64_t begin = nr_cycles();
      kernel(&x, &r, &c, &y);
      uint64_t elapsed = nr_cycles() - begin;
      __asm__ volatile("" ::: "memory");
      unsigned mismatches = 0, nonfinite = 0;
      double maximum = 0.0, sum = 0.0;
      for (size_t i = 0; i < COUNT; ++i) {
        float expected = oracle(i, round), actual = output[variant].data[i];
        mismatches += bits(actual) != bits(expected);
        if ((bits(actual) & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
          ++nonfinite;
          continue;
        }
        double delta = (double)actual - (double)expected;
        if (delta < 0.0) delta = -delta;
        if (delta > maximum) maximum = delta;
        sum += delta;
      }
      unsigned guard_errors = guards_bad(output[0].before, output[0].after)
                            + guards_bad(output[1].before, output[1].after);
      unsigned input_errors = inputs_bad(round);
      unsigned errors = mismatches + nonfinite + guard_errors + input_errors;
      total_errors += errors;
      nr_puts("[dequant-ab] sample");
      field32(" round=", round);
      nr_puts(variant ? " variant=optimized" : " variant=baseline");
      field32(" order=", pass); field64(" cycles=", elapsed);
      field32(" mismatches=", mismatches); field32(" nonfinite=", nonfinite);
      field32(" guard_errors=", guard_errors); field32(" input_errors=", input_errors);
      field64(" max_abs_f64_bits=", double_bits(maximum));
      field64(" mean_abs_f64_bits=", double_bits(sum / COUNT));
      nr_puts(errors ? " status=FAIL\r\n" : " status=PASS\r\n");
    }
    unsigned disagreements = 0;
    for (size_t i = 0; i < COUNT; ++i)
      disagreements += bits(output[0].data[i]) != bits(output[1].data[i]);
    total_errors += disagreements;
    nr_puts("[dequant-ab] pair"); field32(" round=", round);
    field32(" mismatches=", disagreements); nr_puts("\r\n");
  }
  nr_puts("[dequant-ab] "); nr_puts(total_errors ? "FAIL" : "PASS");
  field32(" errors=", total_errors); nr_puts("\r\n");
  return total_errors != 0;
}
