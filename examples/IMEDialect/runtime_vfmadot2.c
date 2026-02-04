// Runtime support for vfmadot2_print_test
// Prints matrix results in human-readable format (fp16 version)

#include <math.h>
#include <stdint.h>
#include <stdio.h>

// Convert half-precision float to single-precision
// Based on IEEE 754 half-precision format
static float f16_to_f32(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t mant = h & 0x3ff;

  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign << 31;
    } else {
      // Denormalized
      while (!(mant & 0x400)) {
        mant <<= 1;
        exp--;
      }
      exp++;
      mant &= ~0x400;
      exp = exp + (127 - 15);
      f = (sign << 31) | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f = (sign << 31) | 0x7f800000 | (mant << 13);
  } else {
    exp = exp + (127 - 15);
    f = (sign << 31) | (exp << 23) | (mant << 13);
  }

  union {
    uint32_t u;
    float f;
  } u;
  u.u = f;
  return u.f;
}

void print_header() {
  printf("======================================\n");
  printf("   vfmadot2 Result Matrix (fp16)\n");
  printf("   slide=2 fixed sliding window\n");
  printf("======================================\n");
  printf("Expected (slide=2): all 18s in row 0,\n");
  printf("                    all 22s in row 1,\n");
  printf("                    all 26s in row 2,\n");
  printf("                    all 30s in row 3\n");
  printf("--------------------------------------\n");
}

void print_row_f16(int32_t row, uint16_t v0, uint16_t v1, uint16_t v2,
                   uint16_t v3) {
  float f0 = f16_to_f32(v0);
  float f1 = f16_to_f32(v1);
  float f2 = f16_to_f32(v2);
  float f3 = f16_to_f32(v3);
  printf("Row %d: [%6.2f, %6.2f, %6.2f, %6.2f]\n", row, f0, f1, f2, f3);
}
