#include "nr_runtime.h"

typedef void (*Probe)(const void *, const void *, void *, size_t *);
#define DECLARE(name) extern void probe_##name(const void *, const void *, void *, size_t *)
DECLARE(load_store); DECLARE(copy); DECLARE(add_vv); DECLARE(sub_vv);
DECLARE(mul_vv); DECLARE(div_vv); DECLARE(macc_vv); DECLARE(add_vf);
DECLARE(mul_vf); DECLARE(div_vf); DECLARE(macc_vf); DECLARE(sqrt);
DECLARE(sum); DECLARE(ordered_sum); DECLARE(max_reduce); DECLARE(max_vv);
DECLARE(min_vv); DECLARE(f_to_i); DECLARE(f_to_i_rtz); DECLARE(i_to_f);
DECLARE(broadcast_f);
DECLARE(zero_i); DECLARE(extract_f);
DECLARE(madd_vv); DECLARE(whole_move); DECLARE(whole_memory); DECLARE(setivli);
DECLARE(csr_vlenb); DECLARE(csr_vl); DECLARE(csr_vtype);
DECLARE(byte_memory); DECLARE(word64_memory2); DECLARE(word64_memory4);
DECLARE(whole_move_vl1_e8);
extern int nr_copy_probe(void);
extern void nr_probe_trap(void);
volatile uint64_t nr_probe_illegal_count, nr_probe_last_pc, nr_probe_last_word;
static float a[16] __attribute__((aligned(64)));
static float b[16] __attribute__((aligned(64)));
static int32_t integers[16] __attribute__((aligned(64)));
static struct {
  uint32_t before[16];
  float values[16];
  uint32_t after[16];
} output __attribute__((aligned(64)));

enum Kind { COPY, ADD, SUB, MUL, DIV, MACC, ADD_F, MUL_F, DIV_F, MACC_F,
            SQRT, SUM, MAX_REDUCE, MAX, MIN, F_TO_I, F_TO_I_RTZ, I_TO_F,
            BROADCAST, ZERO_I, EXTRACT_F, MADD, COPY_BYTES, COPY64,
            CSR_LENB, CSR_VL, CSR_VTYPE };
static const struct { const char *name; Probe function; enum Kind kind; } cases[] = {
  {"vsetvli+vle32+vse32", probe_load_store, COPY},
  {"vmv.v.v", probe_copy, COPY},
  {"vfadd.vv", probe_add_vv, ADD}, {"vfsub.vv", probe_sub_vv, SUB},
  {"vfmul.vv", probe_mul_vv, MUL}, {"vfdiv.vv", probe_div_vv, DIV},
  {"vfmacc.vv", probe_macc_vv, MACC},
  {"vfadd.vf", probe_add_vf, ADD_F}, {"vfmul.vf", probe_mul_vf, MUL_F},
  {"vfdiv.vf", probe_div_vf, DIV_F}, {"vfmacc.vf", probe_macc_vf, MACC_F},
  {"vfsqrt.v", probe_sqrt, SQRT},
  {"vfredusum.vs", probe_sum, SUM}, {"vfredosum.vs", probe_ordered_sum, SUM},
  {"vfredmax.vs", probe_max_reduce, MAX_REDUCE},
  {"vfmax.vv", probe_max_vv, MAX}, {"vfmin.vv", probe_min_vv, MIN},
  {"vfcvt.x.f.v", probe_f_to_i, F_TO_I},
  {"vfcvt.rtz.x.f.v", probe_f_to_i_rtz, F_TO_I_RTZ},
  {"vfcvt.f.x.v", probe_i_to_f, I_TO_F},
  {"vfmv.v.f", probe_broadcast_f, BROADCAST},
  {"vmv.v.i", probe_zero_i, ZERO_I},
  {"vfmv.f.s", probe_extract_f, EXTRACT_F},
  {"vfmadd.vv", probe_madd_vv, MADD},
  {"vmv1r.v", probe_whole_move, COPY},
  {"vl1re32.v+vs1r.v", probe_whole_memory, COPY},
  {"vsetivli", probe_setivli, COPY},
  {"csrr.vlenb", probe_csr_vlenb, CSR_LENB},
  {"csrr.vl", probe_csr_vl, CSR_VL},
  {"csrr.vtype", probe_csr_vtype, CSR_VTYPE},
  {"vle8.v+vse8.v(e8,mf4,VL16)", probe_byte_memory, COPY_BYTES},
  {"vle64.v+vse64.v(e64,m1,VL2)", probe_word64_memory2, COPY64},
  {"vle64.v+vse64.v(e64,m1,VL4)", probe_word64_memory4, COPY64},
  {"vmv1r.v(after VL1/e8)", probe_whole_move_vl1_e8, COPY},
};
static uint32_t float_bits(float value) {
  union { float f; uint32_t u; } bits = {value};
  return bits.u;
}
static int32_t round_even(float value) {
  int32_t result = (int32_t)value;
  float fraction = value - (float)result;
  if (fraction > 0.5f || (fraction == 0.5f && (result & 1))) ++result;
  if (fraction < -0.5f || (fraction == -0.5f && (result & 1))) --result;
  return result;
}
static float expected(enum Kind kind, unsigned i, size_t vl) {
  float seed = (float)(i % 3) * 0.125f;
  switch (kind) {
  case COPY: case COPY_BYTES: case COPY64: return a[i];
  case ADD: return a[i] + b[i];
  case SUB: return a[i] - b[i];
  case MUL: return a[i] * b[i];
  case DIV: return a[i] / b[i];
  case MACC: return seed + a[i] * b[i];
  case MADD: return seed * a[i] + b[i];
  case ADD_F: return a[i] + b[0];
  case MUL_F: return a[i] * b[0];
  case DIV_F: return a[i] / b[0];
  case MACC_F: return seed + a[i] * b[0];
  case SQRT: return sqrtf(b[i]);
  case MAX: return a[i] > b[i] ? a[i] : b[i];
  case MIN: return a[i] < b[i] ? a[i] : b[i];
  case I_TO_F: return (float)integers[i];
  case BROADCAST: return b[0];
  case ZERO_I: return 0.0f;
  case EXTRACT_F: return a[0];
  case SUM: {
    float result = b[0];
    for (size_t j = 0; j < vl; ++j) result += a[j];
    return result;
  }
  case MAX_REDUCE: {
    float result = b[0];
    for (size_t j = 0; j < vl; ++j) if (a[j] > result) result = a[j];
    return result;
  }
  default: return 0;
  }
}
int launch(void) {
  uintptr_t previous_mtvec;
  __asm__ volatile("csrr %0, mtvec" : "=r"(previous_mtvec));
  __asm__ volatile("csrw mtvec, %0" :: "r"(&nr_probe_trap) : "memory");
  unsigned supported = 0, unsupported = 0, numeric_fail = 0;
  nr_puts("[probe] NR RVV e32,m1 requested VL=16; per-instruction results follow\r\n");
  for (unsigned test = 0; test < sizeof(cases) / sizeof(cases[0]); ++test) {
    for (unsigned i = 0; i < 16; ++i) {
      a[i] = ((float)i - 8.0f) * 0.25f;
      b[i] = ((float)(i % 5) + 1.0f) * 0.5f;
      if ((i & 1) && cases[test].kind != SQRT) b[i] = -b[i];
      integers[i] = (int32_t)i * 1234 - 8000;
      output.values[i] = (float)(i % 3) * 0.125f;
      output.before[i] = output.after[i] = 0x513a7e29u;
    }
    size_t vl = 0;
    uint64_t traps = nr_probe_illegal_count;
    nr_puts("[probe] BEGIN "); nr_puts(cases[test].name); nr_puts("\r\n");
    const void *input = cases[test].kind == I_TO_F ? (const void *)integers : a;
    __asm__ volatile("fence rw, rw" ::: "memory");
    cases[test].function(input, b, output.values, &vl);
    __asm__ volatile("fence rw, rw" ::: "memory");
    traps = nr_probe_illegal_count - traps;
    nr_puts("[probe] "); nr_puts(cases[test].name);
    nr_puts(" VL=0x"); nr_hex32((uint32_t)vl);
    if (traps) {
      ++unsupported;
      nr_puts(" UNSUPPORTED illegal=0x"); nr_hex32((uint32_t)traps);
      nr_puts(" last_pc=0x"); nr_hex64(nr_probe_last_pc);
      nr_puts(" mtval=0x"); nr_hex64(nr_probe_last_word); nr_puts("\r\n");
      continue;
    }
    unsigned errors = vl == 0 || vl > 16;
    unsigned checked = cases[test].kind == SUM || cases[test].kind == MAX_REDUCE || cases[test].kind == EXTRACT_F || cases[test].kind >= CSR_LENB ? 1 : (unsigned)vl;
    if (cases[test].kind == COPY_BYTES) checked = (unsigned)vl / 4;
    if (cases[test].kind == COPY64) checked = (unsigned)vl * 2;
    if (checked > 16) checked = 16;
    uint32_t first_got = 0, first_expected = 0;
    for (unsigned i = 0; i < 16; ++i)
      errors += output.before[i] != 0x513a7e29u || output.after[i] != 0x513a7e29u;
    if (cases[test].kind == COPY_BYTES || cases[test].kind == COPY64)
      for (unsigned i = checked; i < 16; ++i)
        errors += output.values[i] != (float)(i % 3) * 0.125f;
    for (unsigned i = 0; i < checked; ++i) {
      uint32_t got = float_bits(output.values[i]), want;
      int wrong;
      if (cases[test].kind == COPY || cases[test].kind == COPY_BYTES || cases[test].kind == COPY64) {
        want = float_bits(a[i]);
        wrong = got != want;
      } else if (cases[test].kind >= CSR_LENB) {
        want = cases[test].kind == CSR_LENB ? 64 : cases[test].kind == CSR_VL ? 16 : 0xd0;
        wrong = got != want;
      } else if (cases[test].kind == F_TO_I || cases[test].kind == F_TO_I_RTZ) {
        want = (uint32_t)(cases[test].kind == F_TO_I ? round_even(a[i]) : (int32_t)a[i]);
        wrong = got != want;
      } else {
        float reference = expected(cases[test].kind, i, vl);
        want = float_bits(reference);
        float delta = output.values[i] - reference;
        if (delta < 0) delta = -delta;
        float magnitude = reference < 0 ? -reference : reference;
        wrong = !(delta <= 0.000002f * (1.0f + magnitude));
      }
      if (wrong && !errors) { first_got = got; first_expected = want; }
      errors += (unsigned)wrong;
    }
    if (errors) {
      ++numeric_fail;
      nr_puts(" EXECUTED NUMERIC_FAIL errors=0x"); nr_hex32(errors);
      nr_puts(" got_bits=0x"); nr_hex32(first_got);
      nr_puts(" expected_bits=0x"); nr_hex32(first_expected);
    } else {
      ++supported;
      nr_puts(" SUPPORTED numeric=PASS");
    }
    nr_puts("\r\n");
  }
  __asm__ volatile("csrw mtvec, %0" :: "r"(previous_mtvec) : "memory");
  nr_puts("[probe] totals supported=0x"); nr_hex32(supported);
  nr_puts(" unsupported=0x"); nr_hex32(unsupported);
  nr_puts(" numeric_fail=0x"); nr_hex32(numeric_fail);
  nr_puts("\r\n[probe] completion does not imply all instructions are supported\r\n");
  nr_puts("verify RVV probe completion: PASS\r\n");
  int copy_status = nr_copy_probe();
  return copy_status || numeric_fail != 0;
}
