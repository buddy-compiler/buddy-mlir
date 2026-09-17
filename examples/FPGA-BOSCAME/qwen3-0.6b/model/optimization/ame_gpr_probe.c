/* Focused v0.5 instruction-contract probe, not a model kernel replacement.
 * Link with the existing common NR runtime and compile with NR_CFLAGS.
 * Every AME instruction keeps both fences; every round keeps ame_fence(). */
#include "nr_runtime.h"

#define M 2
#define N 3
#define K 4
#define A_STRIDE 16
#define B_STRIDE 32
#define C_STRIDE 8
#define O_STRIDE 16
#define GUARDS 16
#define ROUNDS 4
#define GUARD UINT32_C(0x693ac57e)
#define PAD8 0x59
#define PAD32 INT32_C(0x397a125c)
#define POISON INT32_C(0x2468abcd)
#define STR_(value) #value
#define STR(value) STR_(value)

/* Five-bit GPR operands are intentionally x18..x31. The low destination
 * field in MLS names tr0/tr4/acc0, not a GPR operand or clobber. */
#define MSET(funct3, reg) \
  ((2 << 25) | ((reg) << 15) | ((funct3) << 12) | ((reg) << 7) | 0x77)
#define MLS(funct7, base, stride, width, matrix) \
  (((funct7) << 25) | ((stride) << 20) | ((base) << 15) | \
   ((width) << 12) | ((matrix) << 7) | 0x77)
#define FENCED_WORD(word) "fence rw, rw\n\t.word " STR(word) "\n\tfence rw, rw"

static struct {
  uint32_t before[GUARDS];
  int8_t data[M][A_STRIDE];
  uint32_t after[GUARDS];
} lhs NR_WORKSPACE;
static struct {
  uint32_t before[GUARDS];
  int8_t data[N][B_STRIDE];
  uint32_t after[GUARDS];
} rhs NR_WORKSPACE;
static struct {
  uint32_t before[GUARDS];
  int32_t data[M][C_STRIDE];
  uint32_t after[GUARDS];
} initial NR_WORKSPACE;
static struct {
  uint32_t before[GUARDS];
  int32_t data[M][O_STRIDE];
  uint32_t after[GUARDS];
} output NR_WORKSPACE;

/* Same instruction and clobber contract as common/nr/ame_sync.c. Keep this
 * probe's type setup conservative; direct mode does not optimize msettype. */
static inline void safe_msettype(uint64_t type) {
  register uint64_t a0 asm("a0") = type;
  __asm__ volatile(FENCED_WORD(0x00054077)
                   : "+r"(a0) :: "a1", "a2", "a3", "a4", "a5", "a6",
                     "a7", "t0", "t1", "t2", "t3", "t4", "t5", "t6",
                     "memory");
}
static inline uintptr_t direct_m(uintptr_t value) {
  register uintptr_t inout asm("s2") = value; /* x18, rd == rs1 */
  __asm__ volatile(FENCED_WORD(MSET(5, 18)) : "+r"(inout) :: "memory");
  return inout;
}
static inline uintptr_t direct_n(uintptr_t value) {
  register uintptr_t inout asm("s7") = value; /* x23, rd == rs1 */
  __asm__ volatile(FENCED_WORD(MSET(4, 23)) : "+r"(inout) :: "memory");
  return inout;
}
static inline uintptr_t direct_k(uintptr_t value) {
  register uintptr_t inout asm("t6") = value; /* x31, rd == rs1 */
  __asm__ volatile(FENCED_WORD(MSET(6, 31)) : "+r"(inout) :: "memory");
  return inout;
}
static inline void direct_load_a(void) {
  register const int8_t *base asm("s2") = &lhs.data[0][0];
  register uintptr_t stride asm("s3") = A_STRIDE; /* x18 / x19 */
  __asm__ volatile(FENCED_WORD(MLS(2, 18, 19, 0, 0))
                   :: "r"(base), "r"(stride) : "memory");
}
static inline void direct_load_b(void) {
  register const int8_t *base asm("s4") = &rhs.data[0][0];
  register uintptr_t stride asm("s5") = B_STRIDE; /* x20 / x21 */
  __asm__ volatile(FENCED_WORD(MLS(4, 20, 21, 0, 4))
                   :: "r"(base), "r"(stride) : "memory");
}
static inline void direct_load_c(void) {
  register const int32_t *base asm("s6") = &initial.data[0][0];
  register uintptr_t stride asm("s7") = C_STRIDE * sizeof(int32_t);
  __asm__ volatile(FENCED_WORD(MLS(0, 22, 23, 2, 0))
                   :: "r"(base), "r"(stride) : "memory");
}
static inline void direct_store_o(void) {
  register int32_t *base asm("t5") = &output.data[0][0];
  register uintptr_t stride asm("t6") = O_STRIDE * sizeof(int32_t);
  __asm__ volatile(FENCED_WORD(MLS(1, 30, 31, 2, 0))
                   :: "r"(base), "r"(stride) : "memory");
}
static int8_t a_at(unsigned m, unsigned k, unsigned round) {
  static const int8_t seed[M][K] = {{1, -2, 3, 4}, {-3, 5, 2, -1}};
  return (int8_t)(seed[m][k] + (int)round * ((int)k - 1));
}
static int8_t b_at(unsigned n, unsigned k, unsigned round) {
  static const int8_t seed[N][K] = {{2, 1, -1, 3}, {-4, 2, 1, -2},
                                  {3, -1, 2, 1}};
  return (int8_t)(seed[n][k] - (int)round * ((int)n + 1));
}
static int32_t c_at(unsigned m, unsigned n, unsigned round) {
  static const int32_t seed[M][N] = {{7, -5, 11}, {4, 9, -6}};
  return seed[m][n] + (int32_t)round * (int32_t)(7 + 3 * m + n);
}
static int32_t oracle(unsigned m, unsigned n, unsigned round) {
  int32_t result = c_at(m, n, round);
  for (unsigned k = 0; k < K; ++k)
    result += (int32_t)a_at(m, k, round) * (int32_t)b_at(n, k, round);
  return result;
}
static void set_guards(uint32_t *before, uint32_t *after) {
  for (unsigned i = 0; i < GUARDS; ++i)
    before[i] = after[i] = GUARD ^ i;
}
static unsigned guards_bad(const uint32_t *before, const uint32_t *after) {
  unsigned bad = 0;
  for (unsigned i = 0; i < GUARDS; ++i)
    bad += before[i] != (GUARD ^ i) || after[i] != (GUARD ^ i);
  return bad;
}
static void initialize(unsigned round) {
  set_guards(lhs.before, lhs.after);
  set_guards(rhs.before, rhs.after);
  set_guards(initial.before, initial.after);
  set_guards(output.before, output.after);
  for (unsigned m = 0; m < M; ++m) {
    for (unsigned k = 0; k < A_STRIDE; ++k)
      lhs.data[m][k] = k < K ? a_at(m, k, round) : PAD8;
    for (unsigned n = 0; n < C_STRIDE; ++n)
      initial.data[m][n] = n < N ? c_at(m, n, round) : PAD32;
    for (unsigned n = 0; n < O_STRIDE; ++n)
      output.data[m][n] = n < N ? POISON : PAD32;
  }
  for (unsigned n = 0; n < N; ++n)
    for (unsigned k = 0; k < B_STRIDE; ++k)
      rhs.data[n][k] = k < K ? b_at(n, k, round) : PAD8;
}
static unsigned validate(unsigned round) {
  unsigned errors = guards_bad(lhs.before, lhs.after)
                  + guards_bad(rhs.before, rhs.after)
                  + guards_bad(initial.before, initial.after)
                  + guards_bad(output.before, output.after);
  for (unsigned m = 0; m < M; ++m) {
    for (unsigned k = 0; k < A_STRIDE; ++k)
      errors += lhs.data[m][k] != (k < K ? a_at(m, k, round) : PAD8);
    for (unsigned n = 0; n < C_STRIDE; ++n)
      errors += initial.data[m][n] != (n < N ? c_at(m, n, round) : PAD32);
    for (unsigned n = 0; n < O_STRIDE; ++n) {
      int32_t expected = n < N ? oracle(m, n, round) : PAD32;
      int32_t actual = output.data[m][n];
      if (actual != expected) {
        ++errors;
        nr_puts("[ame-gpr] mismatch m="); nr_hex32(m);
        nr_puts(" n="); nr_hex32(n);
        nr_puts(" expected="); nr_hex32((uint32_t)expected);
        nr_puts(" actual="); nr_hex32((uint32_t)actual); nr_puts("\r\n");
      }
    }
  }
  for (unsigned n = 0; n < N; ++n)
    for (unsigned k = 0; k < B_STRIDE; ++k)
      errors += rhs.data[n][k] != (k < K ? b_at(n, k, round) : PAD8);
  return errors;
}
int launch(void) {
  unsigned total_errors = 0;
  nr_puts("[ame-gpr] direct GPR probe M=2 N=3 K=4 rounds=4\r\n");
  for (unsigned round = 0; round < ROUNDS; ++round) {
    /* No output is used until the common completion/resync sequence returns.
     * Rewriting the same buffers each round also tests the current runtime's
     * memory ownership behavior. No new cache-policy claim is made here. */
    ame_fence();
    initialize(round);
    uintptr_t actual_m = direct_m(M), actual_n = direct_n(N);
    safe_msettype((1ULL << 16) | (1ULL << 6) | 2);
    direct_load_c();
    safe_msettype((1ULL << 16) | (1ULL << 4));
    uintptr_t actual_k = direct_k(K);
    direct_load_a();
    direct_load_b();
    __asm__ volatile(FENCED_WORD(0x28480877) ::: "memory"); /* acc0 += tr0*tr4 */
    safe_msettype((1ULL << 16) | (1ULL << 6) | 2);
    direct_store_o();
    ame_fence();
    unsigned errors = validate(round);
    errors += actual_m != M || actual_n != N || actual_k != K;
    nr_puts("[ame-gpr] round="); nr_hex32(round);
    nr_puts(" tile_m="); nr_hex64(actual_m);
    nr_puts(" tile_n="); nr_hex64(actual_n);
    nr_puts(" tile_k="); nr_hex64(actual_k);
    nr_puts(" errors="); nr_hex32(errors); nr_puts("\r\n");
    total_errors += errors;
  }
  nr_puts(total_errors ? "verify AME direct GPR probe: FAIL\r\n"
                       : "verify AME direct GPR probe: PASS\r\n");
  return total_errors != 0;
}
