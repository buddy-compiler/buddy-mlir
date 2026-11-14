//===- test_private_ddr.c - Private DDR frame tests for bare runtime ------===//
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

// Host validation of the production invocation-private DDR frame allocator.
#include <assert.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static _Alignas(64) unsigned char arena[262144];
static uintptr_t window_start, window_end, hart;
static unsigned writes, reads;
static unsigned check_release_policy, release_read_seen, release_write_seen;
static uintptr_t release_top, release_active;
static uint64_t release_count;
static void test_shared_writes(void);
static void test_shared_reads(void);
#ifndef BOSC_AME_PRIVATE_DDR_TEST_RESET
#define BOSC_AME_PRIVATE_DDR_TEST_RESET() ((void)0)
#endif
#ifndef BOSC_AME_PRIVATE_DDR_TEST_SHARED
#define BOSC_AME_PRIVATE_DDR_TEST_SHARED() ((void)0)
#endif
#define BOSC_AME_PRIVATE_DDR_START ((uintptr_t)arena)
#define BOSC_AME_PRIVATE_DDR_END ((uintptr_t)(arena + sizeof(arena)))
#define BOSC_AME_PRIVATE_DDR_WINDOW_START window_start
#define BOSC_AME_PRIVATE_DDR_WINDOW_END window_end
#define BOSC_AME_PRIVATE_DDR_HART_ID() hart
#define BOSC_AME_DMA_SHARED_WRITES()                                           \
  (++writes, test_shared_writes(), BOSC_AME_PRIVATE_DDR_TEST_SHARED())
#define BOSC_AME_DMA_SHARED_READS()                                            \
  (++reads, test_shared_reads(), BOSC_AME_PRIVATE_DDR_TEST_SHARED())
#include "bosc_ame_dma.c"

// The board's read policy does not publish CPU stores to DDR. The first
// write-policy transition after reading the frame guards must therefore
// precede any allocator-state update. Extra transitions are harmless.
static void test_shared_reads(void) {
  if (check_release_policy)
    release_read_seen = 1;
}
static void test_shared_writes(void) {
  if (check_release_policy && release_read_seen && !release_write_seen) {
    assert(bosc_ame_private_ddr_top == release_top);
    assert(bosc_ame_private_ddr_active == release_active);
    assert(bosc_ame_private_ddr_releases == release_count);
    release_write_seen = 1;
  }
}
static void release_checked(uint64_t address, uint64_t bytes) {
  release_top = bosc_ame_private_ddr_top;
  release_active = bosc_ame_private_ddr_active;
  release_count = bosc_ame_private_ddr_releases;
  release_read_seen = release_write_seen = 0;
  check_release_policy = 1;
  bosc_ame_private_ddr_release(address, bytes);
  check_release_policy = 0;
  assert(release_read_seen && release_write_seen);
}

static jmp_buf failure;
static uint32_t last_reason;
static void fail_hook(uint32_t reason, uint64_t address, uint64_t bytes) {
  (void)address;
  (void)bytes;
  last_reason = reason;
  longjmp(failure, 1);
}
static void reset(void) {
  memset(arena, 0xA5, sizeof(arena));
  window_start = (uintptr_t)arena;
  window_end = (uintptr_t)(arena + sizeof(arena));
  hart = 0;
  writes = reads = 0;
  check_release_policy = release_read_seen = release_write_seen = 0;
  bosc_ame_private_ddr_top = bosc_ame_private_ddr_active =
      bosc_ame_private_ddr_owner = 0;
  bosc_ame_private_ddr_busy = 0;
  bosc_ame_private_ddr_peak_bytes = bosc_ame_private_ddr_acquires =
      bosc_ame_private_ddr_releases = 0;
  bosc_ame_private_ddr_fail_hook = fail_hook;
#if BOSC_AME_PRIVATE_DDR_TRACE
  memset((void *)bosc_ame_private_ddr_trace, 0,
         sizeof(bosc_ame_private_ddr_trace));
  bosc_ame_private_ddr_trace_count = 0;
#endif
  BOSC_AME_PRIVATE_DDR_TEST_RESET();
}
#define FAILS(reason, action)                                                  \
  do {                                                                         \
    last_reason = 0;                                                           \
    if (setjmp(failure) == 0) {                                                \
      action;                                                                  \
      assert(!"expected checked failure");                                     \
    }                                                                          \
    assert(last_reason == (reason));                                           \
  } while (0)

#if BOSC_AME_PRIVATE_DDR_TRACE
static void
expect_trace(unsigned sequence,
             const struct bosc_ame_private_ddr_trace_event *expected) {
  assert(sequence < bosc_ame_private_ddr_trace_count);
  assert(bosc_ame_private_ddr_trace_count - sequence <=
         BOSC_AME_PRIVATE_DDR_TRACE_EVENTS);
  const volatile struct bosc_ame_private_ddr_trace_event *actual =
      &bosc_ame_private_ddr_trace[sequence % BOSC_AME_PRIVATE_DDR_TRACE_EVENTS];
  assert(actual->op == expected->op);
  assert(actual->bytes == expected->bytes);
  assert(actual->address == expected->address);
  assert(actual->top_before == expected->top_before);
  assert(actual->active_before == expected->active_before);
  assert(actual->top_after == expected->top_after);
  assert(actual->active_after == expected->active_after);
  assert(actual->acquires == expected->acquires);
  assert(actual->releases == expected->releases);
}

static void trace_failure_and_wrap_cases(void) {
  reset();
  uint64_t maximum =
      bosc_ame_private_ddr_acquire(BOSC_AME_PRIVATE_DDR_MAX_PAYLOAD);
  uintptr_t end = (uintptr_t)(arena + sizeof(arena));
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_CAPACITY, bosc_ame_private_ddr_acquire(64));
  assert(bosc_ame_private_ddr_trace_count == 2);
  struct bosc_ame_private_ddr_trace_event expected = {
      0x100u | BOSC_AME_PRIVATE_DDR_FAIL_CAPACITY,
      64,
      end,
      end,
      maximum,
      end,
      maximum,
      1,
      0};
  expect_trace(1, &expected);

  reset();
  const uint64_t outer = bosc_ame_private_ddr_acquire(64);
  const uint64_t inner = bosc_ame_private_ddr_acquire(64);
  const uintptr_t nested_top = (uintptr_t)arena + 384u;
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_LIFETIME,
        bosc_ame_private_ddr_release(outer, 64));
  assert(bosc_ame_private_ddr_trace_count == 3);
  expected = (struct bosc_ame_private_ddr_trace_event){
      0x100u | BOSC_AME_PRIVATE_DDR_FAIL_LIFETIME,
      64,
      outer,
      nested_top,
      inner,
      nested_top,
      inner,
      2,
      0};
  expect_trace(2, &expected);

  reset();
  const uint64_t guarded = bosc_ame_private_ddr_acquire(64);
  const uintptr_t top = (uintptr_t)arena + 192u;
  arena[0] ^= 1;
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_GUARD,
        bosc_ame_private_ddr_release(guarded, 64));
  assert(bosc_ame_private_ddr_trace_count == 2);
  expected = (struct bosc_ame_private_ddr_trace_event){
      0x100u | BOSC_AME_PRIVATE_DDR_FAIL_GUARD,
      64,
      guarded,
      top,
      guarded,
      top,
      guarded,
      1,
      0};
  expect_trace(1, &expected);

  reset();
  const unsigned total = 2u * (BOSC_AME_PRIVATE_DDR_TRACE_EVENTS + 3u);
  for (unsigned i = 0; i < total / 2u; ++i) {
    uint64_t address = bosc_ame_private_ddr_acquire(64);
    assert(address == outer);
    release_checked(address, 64);
  }
  assert(bosc_ame_private_ddr_trace_count == total);
  for (unsigned i = total - BOSC_AME_PRIVATE_DDR_TRACE_EVENTS; i < total; ++i) {
    unsigned release = i & 1u;
    expected = (struct bosc_ame_private_ddr_trace_event){release ? 2u : 1u,
                                                         64,
                                                         outer,
                                                         release ? top : 0,
                                                         release ? outer : 0,
                                                         release ? 0 : top,
                                                         release ? 0 : outer,
                                                         i / 2u + 1u,
                                                         i / 2u + release};
    expect_trace(i, &expected);
  }
}
#endif

int main(void) {
  reset();
  uint64_t outer = bosc_ame_private_ddr_acquire(8192);
  assert(outer == (uintptr_t)arena + 64 && !(outer & 63));
  uint64_t inner = bosc_ame_private_ddr_acquire(4096);
  assert(inner == outer + 8192 + 128);
  memset((void *)(uintptr_t)outer, 0x71, 8192);
  memset((void *)(uintptr_t)inner, 0x93, 4096);
  release_checked(inner, 4096);
  release_checked(outer, 8192);
  assert(!bosc_ame_private_ddr_top && !bosc_ame_private_ddr_owner);
  assert(writes >= 4 && reads >= 2);
  assert(bosc_ame_private_ddr_acquires == 2 &&
         bosc_ame_private_ddr_releases == 2 &&
         bosc_ame_private_ddr_peak_bytes == 8192 + 4096 + 256);
#if BOSC_AME_PRIVATE_DDR_TRACE
  const uintptr_t outer_top = outer + 8192u + 64u;
  const uintptr_t inner_top = inner + 4096u + 64u;
  const struct bosc_ame_private_ddr_trace_event nested[] = {
      {1, 8192, outer, 0, 0, outer_top, outer, 1, 0},
      {1, 4096, inner, outer_top, outer, inner_top, inner, 2, 0},
      {2, 4096, inner, inner_top, inner, outer_top, outer, 2, 1},
      {2, 8192, outer, outer_top, outer, 0, 0, 2, 2}};
  assert(bosc_ame_private_ddr_trace_count == 4);
  for (unsigned i = 0; i < 4; ++i)
    expect_trace(i, &nested[i]);
#endif
  for (unsigned i = 0; i != 3; ++i) {
    uint64_t address = bosc_ame_private_ddr_acquire(8192);
    assert(address == outer);
    release_checked(address, 8192);
  }

  reset();
  uint64_t maximum =
      bosc_ame_private_ddr_acquire(BOSC_AME_PRIVATE_DDR_MAX_PAYLOAD);
  assert(bosc_ame_private_ddr_peak_bytes == sizeof(arena));
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_CAPACITY, bosc_ame_private_ddr_acquire(64));
  reset();
  maximum = bosc_ame_private_ddr_acquire(BOSC_AME_PRIVATE_DDR_MAX_PAYLOAD);
  bosc_ame_private_ddr_release(maximum, BOSC_AME_PRIVATE_DDR_MAX_PAYLOAD);
  reset();
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_CAPACITY, bosc_ame_private_ddr_acquire(0));
  reset();
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_CAPACITY, bosc_ame_private_ddr_acquire(65));
  reset();
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_CAPACITY,
        bosc_ame_private_ddr_acquire(UINT64_MAX));
  reset();
  window_start += 64;
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_WINDOW, bosc_ame_private_ddr_acquire(64));
  reset();
  window_end -= 64;
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_WINDOW, bosc_ame_private_ddr_acquire(64));
  reset();
  outer = bosc_ame_private_ddr_acquire(64);
  hart = 1;
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_OWNER, bosc_ame_private_ddr_acquire(64));
  reset();
  bosc_ame_private_ddr_busy = 1;
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_CONCURRENT, bosc_ame_private_ddr_acquire(64));
  reset();
  outer = bosc_ame_private_ddr_acquire(64);
  inner = bosc_ame_private_ddr_acquire(64);
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_LIFETIME,
        bosc_ame_private_ddr_release(outer, 64));
  reset();
  outer = bosc_ame_private_ddr_acquire(64);
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_LIFETIME,
        bosc_ame_private_ddr_release(outer, UINT64_MAX));
  reset();
  outer = bosc_ame_private_ddr_acquire(64);
  bosc_ame_private_ddr_release(outer, 64);
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_LIFETIME,
        bosc_ame_private_ddr_release(outer, 64));
  reset();
  outer = bosc_ame_private_ddr_acquire(64);
  arena[0] ^= 1;
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_GUARD,
        bosc_ame_private_ddr_release(outer, 64));
  reset();
  outer = bosc_ame_private_ddr_acquire(64);
  arena[128] ^= 1;
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_GUARD,
        bosc_ame_private_ddr_release(outer, 64));
  reset();
  outer = bosc_ame_private_ddr_acquire(64);
  arena[32] ^= 1; // Parent metadata must be guarded too.
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_GUARD,
        bosc_ame_private_ddr_release(outer, 64));
#if BOSC_AME_PRIVATE_DDR_TRACE
  trace_failure_and_wrap_cases();
#endif
  puts("private DDR frame: all checks passed");
  return 0;
}
