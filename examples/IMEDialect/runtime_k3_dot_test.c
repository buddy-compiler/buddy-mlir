// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
} MemRef;
typedef void (*Dot)(MemRef *, MemRef *, MemRef *);
extern void _mlir_ciface_signed_dot(MemRef *, MemRef *, MemRef *);
extern void _mlir_ciface_unsigned_dot(MemRef *, MemRef *, MemRef *);
extern void _mlir_ciface_signed_unsigned_dot(MemRef *, MemRef *, MemRef *);
extern void _mlir_ciface_unsigned_signed_dot(MemRef *, MemRef *, MemRef *);
extern void _mlir_ciface_full_dot(MemRef *, MemRef *, MemRef *);
extern void _mlir_ciface_full_offset_dot(MemRef *, MemRef *, MemRef *);
extern void _mlir_ciface_matmul(MemRef *, MemRef *, MemRef *);
extern void _mlir_ciface_generic_matmul(MemRef *, MemRef *, MemRef *);
typedef struct {
  void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
} BatchMemRef;
extern void _mlir_ciface_batch_matmul(BatchMemRef *, BatchMemRef *,
                                      BatchMemRef *);
extern void _mlir_ciface_fp16_dot(MemRef *, MemRef *, MemRef *);
extern void _mlir_ciface_loop_dot(MemRef *, MemRef *, MemRef *);

static int test_loop(void) {
  int8_t a = 1, b = 1;
  int32_t c = 0;
  MemRef ad = {&a, &a, 0, {1, 1}, {1, 1}};
  MemRef bd = {&b, &b, 0, {1, 1}, {1, 1}};
  MemRef cd = {&c, &c, 0, {1, 1}, {1, 1}};
  // Enough iterations to exceed the usual stack limit if scratch tiles leak.
  _mlir_ciface_loop_dot(&cd, &ad, &bd);
  if (c != 20000) {
    printf("FAIL loop got=%d expected=20000\n", c);
    return 1;
  }
  return 0;
}

static int test_integer(Dot dot, int unsignedA, int unsignedB, int full) {
  uint8_t a[256], b[256];
  int32_t c[128], expected[128];
  memset(a, 0xa5, sizeof(a));
  memset(b, 0x5a, sizeof(b));
  for (int i = 0; i < 128; ++i)
    c[i] = expected[i] = 0x12345678;
  int m = full ? 8 : 3, n = full ? 8 : 5, k = full ? 16 : 7;
  int inputStride = full ? 16 : 31, outputStride = full ? 8 : 17;
  int columnStride = full ? 1 : 2, offset = full == 1 ? 0 : 3;
  MemRef ad = {a, a, offset, {m, k}, {inputStride, columnStride}};
  MemRef bd = {b, b, offset, {n, k}, {inputStride, columnStride}};
  MemRef cd = {c, c, offset, {m, n}, {outputStride, columnStride}};
  for (int i = 0; i < m; ++i)
    for (int q = 0; q < k; ++q)
      a[offset + i * inputStride + q * columnStride] = 129 + i * 13 + q * 7;
  for (int j = 0; j < n; ++j)
    for (int q = 0; q < k; ++q)
      b[offset + j * inputStride + q * columnStride] = 151 + j * 9 + q * 3;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      c[offset + i * outputStride + j * columnStride] =
          expected[offset + i * outputStride + j * columnStride] =
              i * 11 - j * 7;
  // Two consecutive accumulations must preserve the previous C value.
  for (int repeat = 0; repeat < 2; ++repeat) {
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        for (int q = 0; q < k; ++q) {
          uint8_t av = a[offset + i * inputStride + q * columnStride];
          uint8_t bv = b[offset + j * inputStride + q * columnStride];
          int ai = unsignedA ? av : (int8_t)av;
          int bi = unsignedB ? bv : (int8_t)bv;
          expected[offset + i * outputStride + j * columnStride] += ai * bi;
        }
    dot(&cd, &ad, &bd);
    // Compare the entire allocation, including untouched stride padding.
    for (int i = 0; i < 128; ++i)
      if (c[i] != expected[i]) {
        printf("FAIL integer full=%d unsigned=%d%d repeat=%d index=%d got=%d "
               "expected=%d\n",
               full, unsignedA, unsignedB, repeat, i, c[i], expected[i]);
        return 1;
      }
  }
  return 0;
}

static int test_fp16(void) {
  _Float16 a[256], b[256], c[128], expected[128];
  for (int i = 0; i < 256; ++i)
    a[i] = b[i] = 0;
  for (int i = 0; i < 128; ++i)
    c[i] = expected[i] = 99;
  MemRef ad = {a, a, 3, {3, 7}, {31, 2}};
  MemRef bd = {b, b, 3, {5, 7}, {31, 2}};
  MemRef cd = {c, c, 3, {3, 5}, {17, 2}};
  for (int i = 0; i < 3; ++i)
    for (int q = 0; q < 7; ++q)
      a[3 + i * 31 + q * 2] = (i - q) * 0.5f;
  for (int j = 0; j < 5; ++j)
    for (int q = 0; q < 7; ++q)
      b[3 + j * 31 + q * 2] = (j + q) * 0.25f;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 5; ++j)
      c[3 + i * 17 + j * 2] = expected[3 + i * 17 + j * 2] = i - j;
  for (int repeat = 0; repeat < 2; ++repeat) {
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 5; ++j) {
        float sum = expected[3 + i * 17 + j * 2];
        for (int q = 0; q < 7; ++q)
          sum += (float)a[3 + i * 31 + q * 2] * (float)b[3 + j * 31 + q * 2];
        expected[3 + i * 17 + j * 2] = sum;
      }
    _mlir_ciface_fp16_dot(&cd, &ad, &bd);
    for (int i = 0; i < 128; ++i)
      if (c[i] != expected[i]) {
        printf("FAIL fp16 repeat=%d index=%d got=%g expected=%g\n", repeat, i,
               (double)c[i], (double)expected[i]);
        return 1;
      }
  }
  return 0;
}

static int test_matmul(void) {
  int8_t a[11 * 19], b[19 * 13];
  int32_t c[11 * 13], expected[11 * 13];
  for (int i = 0; i < 11 * 19; ++i)
    a[i] = i % 13 - 6;
  for (int i = 0; i < 19 * 13; ++i)
    b[i] = i % 7 - 3;
  for (int i = 0; i < 11 * 13; ++i)
    c[i] = expected[i] = 17 - i;
  for (int i = 0; i < 11; ++i)
    for (int j = 0; j < 13; ++j)
      for (int k = 0; k < 19; ++k)
        expected[i * 13 + j] += (int)a[i * 19 + k] * b[k * 13 + j];
  MemRef ad = {a, a, 0, {11, 19}, {19, 1}};
  MemRef bd = {b, b, 0, {19, 13}, {13, 1}};
  MemRef cd = {c, c, 0, {11, 13}, {13, 1}};
  _mlir_ciface_matmul(&ad, &bd, &cd);
  int bad = memcmp(c, expected, sizeof(c)) != 0;
  printf("K3 Linalg int8 matmul: %s\n", bad ? "FAIL" : "PASS");
  return bad;
}

static int test_generic(void) {
  _Float16 a[11 * 19], b[19 * 13], c[11 * 13], expected[11 * 13];
  for (int i = 0; i < 11 * 19; ++i)
    a[i] = (i % 7 - 3) * 0.5f;
  for (int i = 0; i < 19 * 13; ++i)
    b[i] = (i % 5 - 2) * 0.25f;
  for (int i = 0; i < 11 * 13; ++i)
    c[i] = expected[i] = (i % 7) * 0.25f;
  for (int i = 0; i < 11; ++i)
    for (int j = 0; j < 13; ++j) {
      float sum = expected[i * 13 + j];
      for (int k = 0; k < 19; ++k)
        sum += (float)a[i * 19 + k] * (float)b[k * 13 + j];
      expected[i * 13 + j] = sum;
    }
  MemRef ad = {a, a, 0, {11, 19}, {19, 1}};
  MemRef bd = {b, b, 0, {19, 13}, {13, 1}};
  MemRef cd = {c, c, 0, {11, 13}, {13, 1}};
  _mlir_ciface_generic_matmul(&ad, &bd, &cd);
  int bad = memcmp(c, expected, sizeof(c)) != 0;
  printf("K3 Linalg FP16 generic matmul: %s\n", bad ? "FAIL" : "PASS");
  return bad;
}

static int test_batch(void) {
  int8_t a[2 * 3 * 19], b[2 * 5 * 19];
  int32_t c[2 * 3 * 5], expected[2 * 3 * 5];
  for (int i = 0; i < 2 * 3 * 19; ++i)
    a[i] = i % 13 - 6;
  for (int i = 0; i < 2 * 5 * 19; ++i)
    b[i] = i % 7 - 3;
  for (int i = 0; i < 2 * 3 * 5; ++i)
    c[i] = expected[i] = 17 - i;
  for (int t = 0; t < 2; ++t)
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 5; ++j)
        for (int k = 0; k < 19; ++k)
          expected[t * 15 + i * 5 + j] +=
              (int)a[t * 57 + i * 19 + k] * b[t * 95 + j * 19 + k];
  BatchMemRef ad = {a, a, 0, {2, 3, 19}, {57, 19, 1}};
  BatchMemRef bd = {b, b, 0, {2, 5, 19}, {95, 19, 1}};
  BatchMemRef cd = {c, c, 0, {2, 3, 5}, {15, 5, 1}};
  _mlir_ciface_batch_matmul(&ad, &bd, &cd);
  int bad = memcmp(c, expected, sizeof(c)) != 0;
  printf("K3 Linalg batched transposed matmul: %s\n", bad ? "FAIL" : "PASS");
  return bad;
}

int ime_example_main(void) {
  int failures = test_integer(_mlir_ciface_signed_dot, 0, 0, 0) +
                 test_integer(_mlir_ciface_unsigned_dot, 1, 1, 0) +
                 test_integer(_mlir_ciface_signed_unsigned_dot, 0, 1, 0) +
                 test_integer(_mlir_ciface_unsigned_signed_dot, 1, 0, 0) +
                 test_integer(_mlir_ciface_full_dot, 0, 0, 1) +
                 test_integer(_mlir_ciface_full_offset_dot, 0, 0, 2) +
                 test_fp16() + test_loop() + test_matmul() + test_generic() +
                 test_batch();
  printf("K3 IME regression: %s (7 dot cases, 3 matmul cases + 20000-iteration "
         "loop)\n",
         failures ? "FAIL" : "PASS");
  return failures != 0;
}
