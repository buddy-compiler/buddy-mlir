#include <stdint.h>

/*
 * The lowered attention graphs retain llvm.exp.f32, which llc materializes as
 * an expf call on RISC-V.  Bare metal has no libm, so use the same bounded
 * approximation as the validated Qwen3 C/AME runner.
 */
static inline float quake2_exp_from_affine(float affine) {
  const uint32_t mantissa_mask = 0x007fffffu;
  const uint32_t exponent_mask = 0xff800000u;
  const uint32_t one_bits = 0x3f800000u;
  int32_t integer = (int32_t)affine;
  uint32_t bits = (uint32_t)integer;
  union {
    uint32_t bits;
    float value;
  } mantissa = {(bits & mantissa_mask) | one_bits};
  union {
    uint32_t bits;
    float value;
  } result;

  mantissa.value =
      (mantissa.value * mantissa.value + 2.0f) * 0.3333333333333333f;
  result.bits = (bits & exponent_mask) + mantissa.bits - one_bits;
  return result.value;
}

float expf(float value) {
  if (value < -88.0f)
    return 0.0f;
  if (value > 88.0f)
    return 1.0e38f;
  return quake2_exp_from_affine(value * 12102203.161561486f +
                                1065353216.0f);
}
