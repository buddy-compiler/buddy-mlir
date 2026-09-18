#include "nr_runtime.h"

extern void probe_quant(const float *, int8_t *, const float *, float *, size_t *);
extern void nr_probe_trap(void);
volatile uint64_t nr_probe_illegal_count, nr_probe_last_pc, nr_probe_last_word;
static float input[16] __attribute__((aligned(64)));
static struct { uint32_t pre[16]; float data[16]; uint32_t post[16]; }
    absolute __attribute__((aligned(64)));
static struct { uint8_t pre[64]; int8_t data[16]; uint8_t post[64]; }
    result __attribute__((aligned(64)));
static const float values[16] = {
    -200.f, -127.f, -126.5f, -1.50001f, -1.5f, -0.50001f, -0.5f, -0.f,
    0.f, 0.49999f, 0.5f, 1.49999f, 1.5f, 126.5f, 127.f, 200.f};
static uint32_t bits(float x) {
  union { float f; uint32_t u; } v = {x}; return v.u;
}
int launch(void) {
  uintptr_t old;
  __asm__ volatile("csrr %0, mtvec" : "=r"(old));
  __asm__ volatile("csrw mtvec, %0" :: "r"(&nr_probe_trap) : "memory");
  unsigned errors = 0;
  nr_puts("[quant-probe] BEGIN abs/mask/merge/clamp/narrow\r\n");
  for (unsigned test = 0; test < 12; ++test) {
    float scale = test % 2 ? 1.137f : 1.f;
    size_t requested = test % 3 == 0 ? 1 : test % 3 == 1 ? 7 : 16;
    size_t vl = requested;
    for (unsigned i = 0; i < 64; ++i)
      result.pre[i] = result.post[i] = 0xa5;
    for (unsigned i = 0; i < 16; ++i) {
      input[i] = test < 6 ? values[i] : values[15-i] * scale;
      absolute.pre[i] = absolute.post[i] = 0x513a7e29;
      absolute.data[i] = -999.f;
      result.data[i] = -128;
    }
    probe_quant(input, result.data, &scale, absolute.data, &vl);
    errors += vl != requested;
    for (unsigned i = 0; i < 64; ++i)
      errors += result.pre[i] != 0xa5 || result.post[i] != 0xa5;
    for (unsigned i = 0; i < 16; ++i) {
      errors += absolute.pre[i] != 0x513a7e29 || absolute.post[i] != 0x513a7e29;
      if (i >= requested) {
        errors += result.data[i] != -128 || absolute.data[i] != -999.f;
        continue;
      }
      float q = input[i] / scale;
      q += q >= 0.f ? 0.5f : -0.5f;
      q = q < -127.f ? -127.f : q > 127.f ? 127.f : q;
      errors += result.data[i] != (int8_t)(int32_t)q;
      errors += bits(absolute.data[i]) != (bits(input[i]) & 0x7fffffff);
    }
  }
  __asm__ volatile("csrw mtvec, %0" :: "r"(old) : "memory");
  nr_puts("[quant-probe] errors=0x"); nr_hex32(errors);
  nr_puts(" illegal=0x"); nr_hex64(nr_probe_illegal_count);
  nr_puts(" last_pc=0x"); nr_hex64(nr_probe_last_pc);
  nr_puts(" mtval=0x"); nr_hex64(nr_probe_last_word);
  nr_puts(errors || nr_probe_illegal_count ? " FAIL\r\n" : " PASS\r\n");
  return errors || nr_probe_illegal_count ? 1 : 0;
}
