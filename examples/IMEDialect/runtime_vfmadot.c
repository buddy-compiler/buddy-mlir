// Runtime for vfmadot test
#include <stdint.h>
#include <stdio.h>

// fp16 to float conversion
float fp16_to_float(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;

  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign << 31;
    } else {
      exp = 1;
      while ((mant & 0x400) == 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x3FF;
      f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f = (sign << 31) | 0x7F800000 | (mant << 13);
  } else {
    f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
  }

  union {
    uint32_t i;
    float f;
  } u;
  u.i = f;
  return u.f;
}

void print_header() {
  printf("Result matrix C (4x4) - vfmadot (floating-point fp16):\n");
  printf("Expected: [[36.0,18.0,36.0,18.0], [36.0,18.0,36.0,18.0], ...]\n");
}

void print_f16_row(int row, uint16_t v0, uint16_t v1, uint16_t v2,
                   uint16_t v3) {
  printf("Row %d: [%.2f, %.2f, %.2f, %.2f]\n", row, fp16_to_float(v0),
         fp16_to_float(v1), fp16_to_float(v2), fp16_to_float(v3));
}
