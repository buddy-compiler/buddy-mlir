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

enum Kind { DOT, FP, HP, SP, PACK };
struct Entry {
  const char *op;
  enum Kind kind;
  int sign, sew, imm;
  void (*run)(void *, const void *, const void *, const void *);
};
#include "a100-kernels.h"

static int value(uint8_t x, int isSigned) { return isSigned ? (int8_t)x : x; }

// Scalar layout reference: no IME or RVV instructions are used here.
static void referencePack(const struct Entry *t, uint8_t *out, const uint8_t *a,
                          const uint8_t *b) {
  int wide = !strcmp(t->op, "vpack") || !strcmp(t->op, "vupack");
  int unpack = !strcmp(t->op, "vupack");
  int nibble = strchr(t->op, '4') != NULL;
  int saturate = !strncmp(t->op, "vnspack", 7);
  int block = wide ? (t->imm ? (8 << t->imm) : t->sew / 8) : (4 << t->imm);
  if (wide) {
    if (!unpack) {
      for (int i = 0; i < 128 / block; ++i) {
        memcpy(out + 2 * i * block, a + i * block, block);
        memcpy(out + (2 * i + 1) * block, b + i * block, block);
      }
    } else {
      for (int i = 0; i < 64 / block; ++i) {
        memcpy(out + i * block, a + 2 * i * block, block);
        memcpy(out + 128 + i * block, a + (2 * i + 1) * block, block);
        memcpy(out + 64 + i * block, b + 2 * i * block, block);
        memcpy(out + 192 + i * block, b + (2 * i + 1) * block, block);
      }
    }
    return;
  }
  uint8_t narrowed[2][64] = {{0}};
  int destBits = nibble ? 4 : t->sew;
  int srcBytes = nibble ? 1 : t->sew / 4;
  for (int src = 0; src < 2; ++src) {
    const uint8_t *in = src ? b : a;
    for (int i = 0; i < 128 / srcBytes; ++i) {
      uint64_t bits = 0;
      memcpy(&bits, in + i * srcBytes, srcBytes);
      int64_t x = (int64_t)bits;
      if (srcBytes < 8)
        x = (int64_t)(bits << (64 - 8 * srcBytes)) >> (64 - 8 * srcBytes);
      if (saturate) {
        int64_t hi = ((int64_t)1 << (destBits - 1)) - 1, lo = -hi - 1;
        bits = x > hi ? hi : (x < lo ? lo : x);
      }
      if (nibble)
        narrowed[src][i / 2] |= (bits & 15) << (4 * (i % 2));
      else
        memcpy(narrowed[src] + i * (destBits / 8), &bits, destBits / 8);
    }
  }
  for (int i = 0; i < 64 / block; ++i) {
    memcpy(out + 2 * i * block, narrowed[0] + i * block, block);
    memcpy(out + (2 * i + 1) * block, narrowed[1] + i * block, block);
  }
}

static void referenceDot(const struct Entry *t, uint8_t *out, uint8_t *expected,
                         uint8_t *a, uint8_t *b, uint8_t *parameters,
                         int pattern) {
  int sa = t->sign == 0 || t->sign == 2;
  int sb = t->sign == 0 || t->sign == 3;
  if (t->kind == FP) {
    _Float16 *af = (_Float16 *)a, *bf = (_Float16 *)b;
    float *c = (float *)out, *e = (float *)expected;
    for (int i = 0; i < 64; ++i) {
      af[i] = (i % 11 - 5) * 0.5f;
      bf[i] = (i % 7 - 3) * 0.25f;
    }
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 8; ++j) {
        int p = i * 8 + j;
        c[p] = e[p] = p * 0.25f;
        for (int repeat = 0; repeat < 2; ++repeat)
          for (int k = 0; k < 8; ++k)
            e[p] += (float)af[i * 8 + k] * (float)bf[j * 8 + k];
      }
    return;
  }
  if (t->kind == HP) {
    for (int i = 0; i < 128; ++i) {
      a[i] = i % 16 == 0 ? 128 : 16 * ((i + pattern) % 3);
      b[i] = i % 16 == 0 ? 128 : 16 * ((2 * i + pattern) % 3);
    }
    _Float16 *scales = (_Float16 *)parameters;
    for (int g = 0; g < 8; ++g)
      for (int j = 0; j < 8; ++j)
        scales[g * 8 + j] = (g + 1) * (1 << (j % 4)) / 1024.0f;
  } else if (t->kind == SP) {
    const int masks[] = {9, 10, 12, 5, 6, 3, 0, 15};
    memset(parameters, 0, 128);
    for (int g = 0; g < 4; ++g)
      for (int j = 0; j < 8; ++j)
        for (int q = 0; q < 8; ++q) {
          int n = j * 8 + q;
          int mask = masks[(g + j + q) % (pattern == 2 ? 8 : 6)];
          parameters[g * 32 + n / 2] |= mask << ((n % 2) * 4);
        }
  }
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j) {
      int sum = 0, p = i * 8 + j;
      if (t->kind == SP) {
        for (int q = 0; q < 8; ++q) {
          int n = j * 8 + q;
          int mask = (parameters[t->imm * 32 + n / 2] >> ((n % 2) * 4)) & 15;
          int selected[4], count = 0;
          for (int k = 0; k < 4; ++k)
            if (mask & (1 << k))
              selected[count++] = k;
          if (count == 2)
            for (int k = 0; k < 2; ++k)
              sum += value(a[i * 32 + 4 * q + selected[k]], sa) *
                     value(b[j * 16 + 2 * q + k], sb);
        }
      } else {
        for (int k = 0; k < 16; ++k)
          sum += value(a[i * 16 + k], sa) * value(b[j * 16 + k], sb);
      }
      if (t->kind == HP) {
        _Float16 *c = (_Float16 *)out, *e = (_Float16 *)expected;
        _Float16 scale = ((_Float16 *)parameters)[t->imm * 8 + j];
        c[p] = e[p] = (p % 7) * 0.25f;
        for (int repeat = 0; repeat < 2; ++repeat)
          e[p] = (_Float16)((float)e[p] + (float)sum * (float)scale);
      } else {
        ((int32_t *)out)[p] = 17 - p;
        ((int32_t *)expected)[p] = 17 - p + 2 * sum;
      }
    }
}

// runtime_k3_runner.c establishes A100 affinity and checks VLEN before this
// entry point executes. Compile this file for the baseline scalar ISA.
int ime_example_main(void) {
  _Alignas(128) uint8_t a[256], b[128], params[128], out[288], expected[256];
  int cases = 0, failures = 0;
  for (unsigned t = 0; t < sizeof(tests) / sizeof(tests[0]); ++t) {
    const struct Entry *entry = &tests[t];
    int wide = !strcmp(entry->op, "vpack") || !strcmp(entry->op, "vupack");
    int bytes = entry->kind == HP || entry->kind == PACK ? 128 : 256;
    if (wide)
      bytes = 256;
    for (int pattern = 0; pattern < 3; ++pattern) {
      for (int i = 0; i < 256; ++i)
        a[i] = pattern == 0 ? i : i * 73 + 57 + pattern * 19;
      for (int i = 0; i < 128; ++i)
        b[i] = pattern == 0 ? 128 + i : i * 29 + 131 - pattern * 17;
      if (entry->kind == PACK && pattern == 2 &&
          strncmp(entry->op, "vn", 2) == 0) {
        int bits = strchr(entry->op, '4') ? 4 : entry->sew;
        int srcBytes = strchr(entry->op, '4') ? 1 : entry->sew / 4;
        int64_t hi = ((int64_t)1 << (bits - 1)) - 1, lo = -hi - 1;
        int64_t values[] = {lo - 1, lo, lo + 1, -1, 0, 1, hi - 1, hi, hi + 1};
        for (int i = 0; i < 128 / srcBytes; ++i) {
          memcpy(a + i * srcBytes, &values[i % 9], srcBytes);
          memcpy(b + i * srcBytes, &values[(i + 4) % 9], srcBytes);
        }
      }
      memset(out, 0x55, sizeof(out));
      if (entry->kind == PACK)
        referencePack(entry, expected, a, b);
      else
        referenceDot(entry, out, expected, a, b, params, pattern);
      int modes = entry->kind == PACK ? 8 : 1;
      for (int mode = 0; mode < modes; ++mode) {
        if (entry->kind == PACK)
          __asm__ volatile("csrw vxrm, %0; csrw vxsat, %1"
                           :
                           : "r"((long)(mode / 2)), "r"((long)(mode % 2))
                           : "memory");
        entry->run(out, a, b, params);
        int bad = memcmp(out, expected, bytes) != 0;
        for (int i = bytes; i < sizeof(out); ++i)
          bad |= out[i] != 0x55;
        if (entry->kind == PACK) {
          unsigned long flag;
          __asm__ volatile("csrr %0, vxsat" : "=r"(flag) : : "memory");
          bad |= flag != mode % 2;
        }
        if (bad)
          printf("FAIL %s e%d imm%d pattern%d mode%d\n", entry->op, entry->sew,
                 entry->imm, pattern, mode);
        failures += bad;
        ++cases;
      }
    }
  }
  printf("A100 SSA IME: %zu kernels, %d cases, %d failures\n",
         sizeof(tests) / sizeof(tests[0]), cases, failures);
  return failures != 0;
}
