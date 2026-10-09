//===- ame_sync.c - NR ame_fence() helper ---------------------------------===//
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
// NR uses non-transposed B loads and integer accumulator stores. Every AME
// instruction is fenced. Call ame_fence() only at operator boundaries to
// resync 1x1 AME state after kernel execution — not inside hot loops.
//
//===----------------------------------------------------------------------===//

#include "nr_runtime.h"

// FPGA mtype bit-field (not the upstream raw element width):
//   bit 16 = mma, bit 4 = mint8, bit 6 = mint32, bits 1:0 = msew.
// int8 MMA is msew=e8; the accumulator cell is msew=e32.
#define AME_MTYPE_INT8 ((1ULL << 16) | (1ULL << 4) | 0x0ULL)
#define AME_MTYPE_INT32 ((1ULL << 16) | (1ULL << 6) | 0x2ULL)

// 64-byte aligned 1x1 tiles. A and B are the MMA inputs; the float cell is
// the int32 accumulator load/store target used only to retire the pipeline.
static float ame_sync_cell[1] __attribute__((aligned(64))) = {0.0f};
static int8_t ame_sync_a[1] __attribute__((aligned(64))) = {1};
static int8_t ame_sync_b[1] __attribute__((aligned(64))) = {1};

// Each helper is one fixed NR encoding from ame_to_word.py, wrapped in
// fence rw,rw. Registers match that tool's fixed-GPR contract.

// msettilem: a0 = remaining rows, result in a6. Encoding 0x04055877.
static inline int ame_msettilem(int rem) {
  register size_t in asm("a0") = (size_t)rem;
  register size_t out asm("a6");
  __asm__ volatile("fence rw, rw\n\t.word 0x04055877\n\tfence rw, rw"
                   : "+r"(in), "=r"(out)::"memory");
  return (int)out;
}

// msettilen: a0 = remaining columns, result in t2. Encoding 0x040543f7.
static inline int ame_msettilen(int rem) {
  register size_t in asm("a0") = (size_t)rem;
  register size_t out asm("t2");
  __asm__ volatile("fence rw, rw\n\t.word 0x040543f7\n\tfence rw, rw"
                   : "+r"(in), "=r"(out)::"memory");
  return (int)out;
}

// msettilek: a3 is both the K remainder and the result. Encoding 0x0406e6f7.
static inline int ame_msettilek(int rem) {
  register size_t inout asm("a3") = (size_t)rem;
  __asm__ volatile("fence rw, rw\n\t.word 0x0406e6f7\n\tfence rw, rw"
                   : "+r"(inout)::"memory");
  return (int)inout;
}

// msettype: a0 holds the mtype bit-field. Encoding 0x00054077.
static inline void ame_msettype(uint64_t mtype) {
  register uint64_t a0 asm("a0") = mtype;
  // Keep the NR configuration instruction and its register effects in one
  // asm contract. A later empty clobber allows the compiler to read an old
  // caller-saved value between the instruction and the empty asm. Match the
  // preservation used by ame_to_word.py for msettype.
  __asm__ volatile("fence rw, rw\n\t.word 0x00054077\n\tfence rw, rw"
                   : "+r"(a0)::"a1", "a2", "a3", "a4", "a5", "a6", "a7", "t0",
                     "t1", "t2", "t3", "t4", "t5", "t6", "memory");
}

// mlae8.m: load the A tile. a0 = base, a1 = stride in bytes. 0x04b50077.
static inline void ame_mlae8(const int8_t *base, int stride_bytes) {
  register const int8_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  __asm__ volatile("fence rw, rw\n\t.word 0x04b50077\n\tfence rw, rw" ::"r"(a0),
                   "r"(a1)
                   : "memory");
}

// mlbe8.m: non-transposed B load. a0 = base, a1 = stride. 0x08b50077.
// NR does not support transposed B loads (mlbte / mlbt).
static inline void ame_mlbe8(const int8_t *base, int stride_bytes) {
  register const int8_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  __asm__ volatile("fence rw, rw\n\t.word 0x08b50077\n\tfence rw, rw" ::"r"(a0),
                   "r"(a1)
                   : "memory");
}

// mqma.b.mm: int8*int8 -> int32 accumulate. No GPR operands. 0x28180877.
static inline void ame_mqma_b(void) {
  __asm__ volatile("fence rw, rw\n\t.word 0x28180877\n\tfence rw, rw" ::
                       : "memory");
}

// mlce32.m: load the accumulator. t3 = base, t0 = stride. 0x005e2077.
static inline void ame_mlce32(const float *base, int stride_bytes) {
  register const float *t3 asm("t3") = base;
  register size_t t0 asm("t0") = (size_t)stride_bytes;
  __asm__ volatile("fence rw, rw\n\t.word 0x005e2077\n\tfence rw, rw" ::"r"(t3),
                   "r"(t0)
                   : "memory");
}

// msce32.m: store the accumulator. t3 = base, t0 = stride. 0x025e2077.
// This store retires the 1x1 sync; it is not a lossless spill of a kernel tile.
static inline void ame_msce32(float *base, int stride_bytes) {
  register float *t3 asm("t3") = base;
  register size_t t0 asm("t0") = (size_t)stride_bytes;
  __asm__ volatile("fence rw, rw\n\t.word 0x025e2077\n\tfence rw, rw" ::"r"(t3),
                   "r"(t0)
                   : "memory");
}

static void ame_resync_state(void) {
  // 1x1 int8 MMA into an int32 accumulator cell, then store it back. Forces
  // the AME pipeline out of whatever tile/type state the kernel left behind.
  (void)ame_msettilem(1);
  (void)ame_msettilen(1);
  (void)ame_msettilek(1);

  ame_msettype(AME_MTYPE_INT8);
  ame_mlae8(ame_sync_a, (int)sizeof(int8_t));
  ame_mlbe8(ame_sync_b, (int)sizeof(int8_t));
  ame_mqma_b();

  ame_msettype(AME_MTYPE_INT32);
  ame_mlce32(ame_sync_cell, (int)sizeof(float));
  ame_msce32(ame_sync_cell, (int)sizeof(float));
}

void ame_fence(void) {
  // Outer fences serialize host/RA memory against the AME resync sequence.
  __asm__ volatile("fence rw, rw" ::: "memory");
  ame_resync_state();
  __asm__ volatile("fence rw, rw" ::: "memory");
}
