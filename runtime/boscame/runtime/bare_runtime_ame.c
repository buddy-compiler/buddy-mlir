//===- bare_runtime_ame.c - AME fence helpers for bare runtime ------------===//
//
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
//
//===----------------------------------------------------------------------===//

//
// AME support of the bare single-hart runtime (opt-in).
//
// `ame_fence()` performs the AME resync sequence used by the bare samples that
// run AME tiles on the delivered platform.  Samples that do not use AME build
// with -DBARE_RUNTIME_ENABLE_AME=0, which removes this whole module; the
// declaration in bare_runtime.h is only present for the enabled build.
//
// This is not the NR `ame_sync.c` contract: NR adds a fence around every AME
// instruction and uses non-transposed B loads.  bare_runtime.c's callers keep
// their own ordering, so the sequence below stays byte-for-byte as shipped.
//

#include "bare_runtime_internal.h"
#include <stdint.h>

#if BARE_RUNTIME_ENABLE_AME
#define AME_MTYPE_INT8 ((1ULL << 16) | (1ULL << 4) | 0x0ULL)
#define AME_MTYPE_INT32 ((1ULL << 16) | (1ULL << 6) | 0x2ULL)
static float ame_sync_cell[1] __attribute__((aligned(64))) = {0.0f};
static int8_t ame_sync_a[1] __attribute__((aligned(64))) = {1};
static int8_t ame_sync_b[1] __attribute__((aligned(64))) = {1};

static inline int ame_msettilem(int rem) {
  register size_t in asm("a0") = (size_t)rem;
  register size_t out asm("a6");
  __asm__ volatile(".word 0x04055077" : "+r"(in), "=r"(out)::"memory");
  return (int)out;
}

static inline int ame_msettilen(int rem) {
  register size_t in asm("a0") = (size_t)rem;
  register size_t out asm("t2");
  __asm__ volatile(".word 0x04054077" : "+r"(in), "=r"(out)::"memory");
  return (int)out;
}

static inline int ame_msettilek(int rem) {
  register size_t inout asm("a0") = (size_t)rem;
  __asm__ volatile(".word 0x04056077" : "+r"(inout)::"memory");
  return (int)inout;
}

static inline void ame_msettype(uint64_t mtype) {
  register uint64_t a0 asm("a0") = mtype;
  __asm__ volatile(".word 0x00054077" ::"r"(a0) : "memory");
  __asm__ volatile("" ::
                       : "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "t0",
                         "t1", "t2", "t3", "t4", "t5", "t6", "memory");
}

static inline void ame_mlae8(const int8_t *base, int stride_bytes) {
  register const int8_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  __asm__ volatile(".word 0x04b50077" ::"r"(a0), "r"(a1) : "memory");
}

static inline void ame_mlbt8(const int8_t *base, int stride_bytes) {
  register const int8_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
#if defined(BARE_RUNTIME_AME_SYNC_NO_TRANSPOSE)
  // The resync tile is exactly 1x1: ordinary and transposed B loads have
  // identical layout. NR FPGA0 traps on the transposed form.
  __asm__ volatile(".word 0x08b500f7" ::"r"(a0), "r"(a1) : "memory");
#else
  __asm__ volatile(".word 0x08b500f7" ::"r"(a0), "r"(a1) : "memory");
#endif
}

static inline void ame_mqma_b(void) {
  __asm__ volatile(".word 0x28180877" ::: "memory");
}

static inline void ame_mlce32(const float *base, int stride_bytes) {
  register const float *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  __asm__ volatile(".word 0x00b52077" ::"r"(a0), "r"(a1) : "memory");
}

static inline void ame_msce32(float *base, int stride_bytes) {
  register float *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  __asm__ volatile(".word 0x02b52077" ::"r"(a0), "r"(a1) : "memory");
}

static void ame_resync_state(void) {
  (void)ame_msettilem(1);
  (void)ame_msettilen(1);
  (void)ame_msettilek(1);

  ame_msettype(AME_MTYPE_INT8);
  ame_mlae8(ame_sync_a, (int)sizeof(int8_t));
  ame_mlbt8(ame_sync_b, (int)sizeof(int8_t));
  ame_mqma_b();

  ame_msettype(AME_MTYPE_INT32);
  ame_mlce32(ame_sync_cell, (int)sizeof(float));
  ame_msce32(ame_sync_cell, (int)sizeof(float));
}

void ame_fence(void) {
  __asm__ volatile("fence rw, rw" ::: "memory");
  ame_resync_state();
  __asm__ volatile("fence rw, rw" ::: "memory");
}
#endif
