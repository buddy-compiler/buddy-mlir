/* exp/log tables and freestanding routines migrated from ModelZoo
 * examples/buddy-qwen35-fpga/runtime/src/qwen35_bare_math.c, commit
 * 8815b74fb6d3cd6288c4d99ac6fd7c5d041a7cc3. See README.md for scope. */
#include "nr_runtime.h"

static uint32_t asuint(float x) {
  union { float f; uint32_t i; } u = {x};
  return u.i;
}

static float asfloat(uint32_t x) {
  union { uint32_t i; float f; } u = {x};
  return u.f;
}

static uint64_t asuint64(double x) {
  union { double f; uint64_t i; } u = {x};
  return u.i;
}

static double asdouble(uint64_t x) {
  union { uint64_t i; double f; } u = {x};
  return u.f;
}

/* Arm optimized-routines single-precision exp/log tables (MIT licensed).
 * These implementations use double only for range reduction and polynomial
 * evaluation.  They remain freestanding and generate ordinary RV64GC code. */
static const uint64_t exp2_tab[32] = {
  0x3ff0000000000000ULL, 0x3fefd9b0d3158574ULL,
  0x3fefb5586cf9890fULL, 0x3fef9301d0125b51ULL,
  0x3fef72b83c7d517bULL, 0x3fef54873168b9aaULL,
  0x3fef387a6e756238ULL, 0x3fef1e9df51fdee1ULL,
  0x3fef06fe0a31b715ULL, 0x3feef1a7373aa9cbULL,
  0x3feedea64c123422ULL, 0x3feece086061892dULL,
  0x3feebfdad5362a27ULL, 0x3feeb42b569d4f82ULL,
  0x3feeab07dd485429ULL, 0x3feea47eb03a5585ULL,
  0x3feea09e667f3bcdULL, 0x3fee9f75e8ec5f74ULL,
  0x3feea11473eb0187ULL, 0x3feea589994cce13ULL,
  0x3feeace5422aa0dbULL, 0x3feeb737b0cdc5e5ULL,
  0x3feec49182a3f090ULL, 0x3feed503b23e255dULL,
  0x3feee89f995ad3adULL, 0x3feeff76f2fb5e47ULL,
  0x3fef199bdd85529cULL, 0x3fef3720dcef9069ULL,
  0x3fef5818dcfba487ULL, 0x3fef7c97337b9b5fULL,
  0x3fefa4afa2a490daULL, 0x3fefd0765b6e4540ULL
};

float expf(float value) {
  uint32_t ax = asuint(value) & 0x7fffffffU;
  if (ax >= 0x42b00000U) {
    if (ax > 0x7f800000U) return value + value;
    if (value > 0x1.62e42ep6f) return asfloat(0x7f800000U);
    if (value < -0x1.9fe368p6f) return 0.0f;
  }
  double z = (0x1.71547652b82fep+0 * 32.0) * (double)value;
  const double shift = 0x1.8p+52;
  double kd = z + shift;
  uint64_t ki = asuint64(kd);
  kd -= shift;
  double r = z - kd;
  uint64_t t = exp2_tab[ki & 31U] + (ki << (52 - 5));
  double s = asdouble(t);
  double p = (0x1.c6af84b912394p-5 / (32.0*32.0*32.0) * r
             + 0x1.ebfce50fac4f3p-3 / (32.0*32.0)) * r * r
             + 0x1.62e42ff0c52d6p-1 / 32.0 * r + 1.0;
  return (float)(p * s);
}

struct log_entry { double invc, logc; };
static const struct log_entry log_tab[16] = {
  {0x1.661ec79f8f3bep+0,-0x1.57bf7808caadep-2},
  {0x1.571ed4aaf883dp+0,-0x1.2bef0a7c06ddbp-2},
  {0x1.49539f0f010bp+0,-0x1.01eae7f513a67p-2},
  {0x1.3c995b0b80385p+0,-0x1.b31d8a68224e9p-3},
  {0x1.30d190c8864a5p+0,-0x1.6574f0ac07758p-3},
  {0x1.25e227b0b8eap+0,-0x1.1aa2bc79c81p-3},
  {0x1.1bb4a4a1a343fp+0,-0x1.a4e76ce8c0e5ep-4},
  {0x1.12358f08ae5bap+0,-0x1.1973c5a611cccp-4},
  {0x1.0953f419900a7p+0,-0x1.252f438e10c1ep-5},
  {0x1p+0,0x0p+0}, {0x1.e608cfd9a47acp-1,0x1.aa5aa5df25984p-5},
  {0x1.ca4b31f026aap-1,0x1.c5e53aa362eb4p-4},
  {0x1.b2036576afce6p-1,0x1.526e57720db08p-3},
  {0x1.9c2d163a1aa2dp-1,0x1.bc2860d22477p-3},
  {0x1.886e6037841edp-1,0x1.1058bc8a07ee1p-2},
  {0x1.767dcf5534862p-1,0x1.4043057b6ee09p-2}
};

float logf(float value) {
  uint32_t ix = asuint(value);
  if (ix - 0x00800000U >= 0x7f800000U - 0x00800000U) {
    if ((ix << 1) == 0) return -asfloat(0x7f800000U);
    if (ix == 0x7f800000U) return value;
    if ((ix & 0x80000000U) || (ix << 1) >= 0xff000000U)
      return asfloat(0x7fc00000U);
    ix = asuint(value * 0x1p23f) - (23U << 23);
  }
  uint32_t tmp = ix - 0x3f330000U;
  int i = (int)((tmp >> 19) & 15U);
  int k = (int32_t)tmp >> 23;
  uint32_t iz = ix - (tmp & 0xff800000U);
  double r = (double)asfloat(iz) * log_tab[i].invc - 1.0;
  double y0 = log_tab[i].logc + (double)k * 0x1.62e42fefa39efp-1;
  double r2 = r * r;
  double y = 0x1.5575b0be00b6ap-2 * r - 0x1.ffffef20a4123p-2;
  y = -0x1.00ea348b88334p-2 * r2 + y;
  return (float)(y * r2 + y0 + r);
}

float powf(float base, float exponent) {
  int integer = (int)exponent;
  if ((float)integer == exponent && integer >= -64 && integer <= 64) {
    int negative = integer < 0;
    unsigned power = (unsigned)(negative ? -integer : integer);
    float factor = base;
    float result = 1.0f;
    while (power) {
      if (power & 1u)
        result *= factor;
      factor *= factor;
      power >>= 1;
    }
    return negative ? 1.0f / result : result;
  }
  if (base <= 0.0f)
    return 0.0f;
  return expf(exponent * logf(base));
}

float tanhf(float value) {
  if (value >= 9.0f)
    return 1.0f;
  if (value <= -9.0f)
    return -1.0f;
  if (value < 0.0f)
    return -tanhf(-value);
  float exponential = expf(2.0f * value);
  return (exponential - 1.0f) / (exponential + 1.0f);
}

float erff(float value) {
  int negative = value < 0.0f;
  float x = negative ? -value : value;
  float t = 1.0f / (1.0f + 0.3275911f * x);
  float polynomial =
      (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t -
         0.284496736f) * t + 0.254829592f) * t;
  float result = 1.0f - polynomial * expf(-x * x);
  return negative ? -result : result;
}

float _mlir_ciface_erff(float value) { return erff(value); }

float sqrtf(float value) {
#if defined(__riscv)
  float result;
  __asm__ volatile("fsqrt.s %0, %1" : "=f"(result) : "f"(value));
  return result;
#else
  return (float)__builtin_sqrt((double)value);
#endif
}

/* RoPE angles: range reduction uses double, polynomial evaluation avoids a
 * target libm dependency. Supported finite domain is |x| <= 2^20 radians;
 * outside it return NaN rather than claiming general libm accuracy. */
static float sine_or_cosine(float value, int cosine) {
  uint32_t bits = asuint(value) & 0x7fffffffU;
  if (bits > 0x49800000U) return asfloat(0x7fc00000U);
  if (!cosine && bits == 0) return value;
  const double half_pi = 0x1.921fb54442d18p+0;
  double angle = (double)value;
  double quadrant_value = angle * 0x1.45f306dc9c883p-1;
  int quadrant = (int)(quadrant_value + (quadrant_value >= 0 ? 0.5 : -0.5));
  double x = angle - quadrant * half_pi;
  double xx = x * x;
  double result;
  int kind = (quadrant + cosine) & 3;
  if (kind & 1) {
    result = 1.0 + xx * (-0.5 + xx * (1.0 / 24.0 + xx *
      (-1.0 / 720.0 + xx * (1.0 / 40320.0 + xx *
      (-1.0 / 3628800.0 + xx * (1.0 / 479001600.0))))));
  } else {
    result = x + x * xx * (-1.0 / 6.0 + xx * (1.0 / 120.0 + xx *
      (-1.0 / 5040.0 + xx * (1.0 / 362880.0 + xx *
      (-1.0 / 39916800.0 + xx * (1.0 / 6227020800.0))))));
  }
  return (float)((kind & 2) ? -result : result);
}
float sinf(float value) { return sine_or_cosine(value, 0); }
float cosf(float value) { return sine_or_cosine(value, 1); }
