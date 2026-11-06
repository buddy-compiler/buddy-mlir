//===- test_bosc_ame_dma.c - Unit tests for BOSC AME DMA driver -----------===//
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
// Host-side test for the BOSCAME SPM/DMA runtime helper.
//
// The test redirects the DMA base addresses at compile time to an in-memory
// fake register file and includes bosc_ame_dma.c directly, so the exact code
// that runs on the board (same register offsets, ordering barriers and
// completion protocol) is exercised on the host.
//
// Build and run (from the repository root):
//   cc -std=c11 -Wall -Wextra -Werror \
//      -I runtime/boscame/runtime \
//      runtime/boscame/runtime/test_bosc_ame_dma.c \
//      -o /tmp/bosc_ame_dma_test && /tmp/bosc_ame_dma_test
//
// bosc_ame_dma_wait has no result: it performs the completion barrier and then
// checks the status itself.  The test replaces bosc_ame_dma_fail_hook with a
// setjmp/longjmp hook, so every failure path (bus_error, busy, bad channel,
// "not launched" token, token mismatch) is observed without trapping.
//
// The wait checks do not use BOSC_AME_DMA_ASSERT, so the same failure suite
// runs with and without NDEBUG.  Only the bosc_ame_dma_start argument checks
// are gated on NDEBUG, because in a debug build they deliberately assert.
//
// The last cases are numerical: a simulated device performs the block move
// the runtime programmed, so the staged bytes can be compared against a
// DDR-direct copy of the same region, target-external guard bytes can be
// checked, and a bus error can be shown to take the error path and leave the
// destination untouched.
//

#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Fake MMIO register file: [channel][register index].
static volatile uint64_t fake_regs[2][5];

#define BOSC_AME_DMA0_BASE ((uintptr_t)&fake_regs[0][0])
#define BOSC_AME_DMA1_BASE ((uintptr_t)&fake_regs[1][0])

//
// Ordering probe for the wait completion barrier.  It replaces the fence so
// that the test can observe the state of the fake device *between* the SRC
// write-back and the STATUS read: the first fence call inside wait must see
// the write-back already applied, and it publishes completion afterwards, so
// a wait that succeeds provably read STATUS after the barrier.
//
static void test_fence(void);
static uint64_t test_mmio_read64(uint32_t channel, uintptr_t offset);
static void test_mmio_write64(uint32_t channel, uintptr_t offset,
                              uint64_t value);
static void test_shared_writes(void);
static void test_shared_reads(void);

#define BOSC_AME_DMA_FENCE() test_fence()
#define BOSC_AME_DMA_SHARED_WRITES() test_shared_writes()
#define BOSC_AME_DMA_SHARED_READS() test_shared_reads()
#define BOSC_AME_DMA_MMIO_READ64(channel, offset)                              \
  test_mmio_read64((channel), (offset))
#define BOSC_AME_DMA_MMIO_WRITE64(channel, offset, value)                      \
  test_mmio_write64((channel), (offset), (value))

#include "bosc_ame_dma.c"

// ------------------------------------------------------------------
// Tiny check harness (independent of NDEBUG, unlike assert)
// ------------------------------------------------------------------

static int failures;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

// Register indices inside the fake file (byte offset / 8).
#define REG_SRC 0u
#define REG_DST 1u
#define REG_LEN 2u
#define REG_START 3u
#define REG_STATUS 4u

static unsigned mmio_reads[2];
static unsigned mmio_writes[2];
static unsigned mmio_write_count;
static uintptr_t mmio_write_offsets[16];
static uint64_t mmio_write_values[16];
static enum { POLICY_UNSET, POLICY_READ, POLICY_WRITE } ddr_policy;

static void test_shared_writes(void) { ddr_policy = POLICY_WRITE; }
static void test_shared_reads(void) { ddr_policy = POLICY_READ; }

static void mmio_reset(void) {
  memset(mmio_reads, 0, sizeof(mmio_reads));
  memset(mmio_writes, 0, sizeof(mmio_writes));
  mmio_write_count = 0;
}

static uint64_t test_mmio_read64(uint32_t channel, uintptr_t offset) {
  CHECK(ddr_policy == POLICY_READ);
  CHECK(channel < 2u);
  CHECK(offset == BOSC_AME_DMA_STATUS_OFFSET);
  if (channel >= 2u || offset != BOSC_AME_DMA_STATUS_OFFSET)
    return UINT64_C(0);
  ++mmio_reads[channel];
  return fake_regs[channel][offset / sizeof(uint64_t)];
}

static void test_mmio_write64(uint32_t channel, uintptr_t offset,
                              uint64_t value) {
  CHECK(ddr_policy == POLICY_WRITE);
  CHECK(channel < 2u);
  CHECK(offset <= BOSC_AME_DMA_START_OFFSET);
  CHECK(offset % sizeof(uint64_t) == 0u);
  if (channel >= 2u || offset > BOSC_AME_DMA_START_OFFSET ||
      offset % sizeof(uint64_t) != 0u)
    return;
  ++mmio_writes[channel];
  CHECK(mmio_write_count < 16u);
  if (mmio_write_count < 16u) {
    mmio_write_offsets[mmio_write_count] = offset;
    mmio_write_values[mmio_write_count] = value;
  }
  ++mmio_write_count;
  fake_regs[channel][offset / sizeof(uint64_t)] = value;
}

static void reset_regs(void) {
  ddr_policy = POLICY_UNSET;
  for (unsigned channel = 0; channel < 2u; ++channel) {
    bosc_ame_dma_last_src[channel] = BOSC_AME_DMA_INVALID_TOKEN;
    for (unsigned reg = 0; reg < 5u; ++reg)
      fake_regs[channel][reg] = UINT64_C(0);
  }
  mmio_reset();
}

// ------------------------------------------------------------------
// Failure hook capture
// ------------------------------------------------------------------

static jmp_buf fail_jmp;
static int fail_seen;
static uint32_t fail_channel;
static uint32_t fail_reason;
static uint64_t fail_status;
static uint64_t fail_token;

static void test_fail_hook(uint32_t channel, uint32_t reason, uint64_t status,
                           uint64_t src_token) {
  fail_channel = channel;
  fail_reason = reason;
  fail_status = status;
  fail_token = src_token;
  ++fail_seen;
  longjmp(fail_jmp, 1);
}

static void fail_reset(void) {
  fail_seen = 0;
  fail_channel = 0;
  fail_reason = 0;
  fail_status = 0;
  fail_token = 0;
}

//
// Run `bosc_ame_dma_wait` and require that it takes the failure path with
// `reason` and `channel`.  `fail_status`/`fail_token` are left for the caller
// to inspect.
//
#define EXPECT_FAIL(channel_value, token_value, reason_value)                  \
  do {                                                                         \
    fail_reset();                                                              \
    if (setjmp(fail_jmp) == 0) {                                               \
      bosc_ame_dma_wait((channel_value), (token_value));                       \
      CHECK(0 && "bosc_ame_dma_wait returned instead of failing");             \
    } else {                                                                   \
      CHECK(fail_seen == 1);                                                   \
      CHECK(fail_channel == (channel_value));                                  \
      CHECK(fail_reason == (reason_value));                                    \
      CHECK(fail_token == (token_value));                                      \
    }                                                                          \
  } while (0)

// ------------------------------------------------------------------
// Ordering probe state
// ------------------------------------------------------------------

static int fence_armed;
static int fence_calls;
static int fence_saw_write_back;
static uint64_t fence_expected_src;
static uint64_t fence_expected_status;
static unsigned fence_channel;

static void test_fence(void) {
  if (!fence_armed)
    return;
  ++fence_calls;
  if (fence_calls == 1) {
    // The completion barrier (SRC write-back) must already have landed.
    fence_saw_write_back =
        (fake_regs[fence_channel][REG_SRC] == fence_expected_src) ? 1 : 0;
    // Publish completion only now, so the status the wait checks proves that
    // the STATUS read happened after this barrier.
    fake_regs[fence_channel][REG_STATUS] = fence_expected_status;
  }
}

static void fence_reset(void) {
  fence_armed = 0;
  fence_calls = 0;
  fence_saw_write_back = 0;
  fence_expected_src = UINT64_C(0);
  fence_expected_status = UINT64_C(0);
  fence_channel = 0;
}

static void prepare_wait(uint32_t channel, uint64_t src) {
  CHECK(bosc_ame_dma_start(channel, src, UINT64_C(0x80000000), UINT64_C(128)) ==
        src);
  mmio_reset();
  fence_reset();
}

// ------------------------------------------------------------------
// 1: start programs the exact register layout and returns src
// ------------------------------------------------------------------

static void test_start_layout_and_token(void) {
  static const uint64_t src = UINT64_C(0x20000000);
  static const uint64_t dst = UINT64_C(0x80000000);
  static const uint64_t len = UINT64_C(128);

  reset_regs();
  fence_reset();
  fake_regs[0][REG_STATUS] = UINT64_C(0xdeadbeef); // STATUS sentinel

  const uint64_t token = bosc_ame_dma_start(0u, src, dst, len);

  CHECK(fake_regs[0][REG_SRC] == src);                          // SRC at 0x00
  CHECK(fake_regs[0][REG_DST] == dst);                          // DST at 0x08
  CHECK(fake_regs[0][REG_LEN] == len);                          // LEN at 0x10
  CHECK(fake_regs[0][REG_START] == BOSC_AME_DMA_START_COMMAND); // 0x18
  CHECK(fake_regs[0][REG_STATUS] == UINT64_C(0xdeadbeef));      // untouched
  CHECK(token == src); // start returns the src token
  CHECK(bosc_ame_dma_last_src[0] == src);
  CHECK(mmio_reads[0] == 0u && mmio_reads[1] == 0u);
  CHECK(mmio_writes[0] == 4u && mmio_writes[1] == 0u);
  CHECK(mmio_write_offsets[0] == BOSC_AME_DMA_SRC_OFFSET);
  CHECK(mmio_write_offsets[1] == BOSC_AME_DMA_DST_OFFSET);
  CHECK(mmio_write_offsets[2] == BOSC_AME_DMA_LEN_OFFSET);
  CHECK(mmio_write_offsets[3] == BOSC_AME_DMA_START_OFFSET);
}

// ------------------------------------------------------------------
// 2: a clean wait returns normally and never calls the failure hook
// ------------------------------------------------------------------

static void test_wait_success(void) {
  static const uint64_t src = UINT64_C(0x20000000);
  static const uint64_t dst = UINT64_C(0x80000000);
  static const uint64_t len = UINT64_C(128);

  reset_regs();
  fence_reset();
  fail_reset();
  prepare_wait(0u, src);
  fake_regs[0][REG_STATUS] = UINT64_C(0x0); // completed, not busy, no error

  bosc_ame_dma_wait(0u, src);

  CHECK(ddr_policy == POLICY_READ);
  CHECK(fail_seen == 0);
  CHECK(mmio_reads[0] == 1u && mmio_reads[1] == 0u);
  CHECK(mmio_writes[0] == 1u && mmio_writes[1] == 0u);
  CHECK(bosc_ame_dma_last_src[0] == BOSC_AME_DMA_INVALID_TOKEN);
  // The write-back only re-arms SRC: no DST/LEN rewrite and no new START.
  CHECK(fake_regs[0][REG_SRC] == src);
  CHECK(fake_regs[0][REG_DST] == dst);
  CHECK(fake_regs[0][REG_LEN] == len);
  CHECK(fake_regs[0][REG_START] == BOSC_AME_DMA_START_COMMAND);

  // Channel 1 was not touched.
  for (unsigned reg = 0; reg < 5u; ++reg)
    CHECK(fake_regs[1][reg] == UINT64_C(0));

  // A consumed token cannot perform another completion transaction.
  mmio_reset();
  EXPECT_FAIL(0u, src, BOSC_AME_DMA_FAIL_INVALID_TOKEN);
  CHECK(mmio_reads[0] == 0u && mmio_writes[0] == 0u);

  // Reserved STATUS bits do not turn a completed transfer into an error.
  prepare_wait(0u, src);
  fail_reset();
  fake_regs[0][REG_STATUS] = UINT64_C(0x100);
  bosc_ame_dma_wait(0u, src);
  CHECK(fail_seen == 0);
}

// ------------------------------------------------------------------
// 3: every defined wait failure takes the error path
// ------------------------------------------------------------------

static void test_wait_failures(void) {
  static const uint64_t src = UINT64_C(0x20000000);

  // bus_error: the transferred data is invalid.
  reset_regs();
  fence_reset();
  prepare_wait(0u, src);
  fake_regs[0][REG_STATUS] = BOSC_AME_DMA_STATUS_BUS_ERROR;
  EXPECT_FAIL(0u, src, BOSC_AME_DMA_FAIL_BUS_ERROR);
  CHECK(fail_status == BOSC_AME_DMA_STATUS_BUS_ERROR);
  CHECK(bosc_ame_dma_last_src[0] == BOSC_AME_DMA_INVALID_TOKEN);

  // busy still visible after the barrier: the completion guarantee did not
  // hold.  The check uses the defined bits, not "status == 0".
  reset_regs();
  fence_reset();
  prepare_wait(0u, src);
  fake_regs[0][REG_STATUS] = BOSC_AME_DMA_STATUS_BUSY;
  EXPECT_FAIL(0u, src, BOSC_AME_DMA_FAIL_BUSY);
  CHECK(fail_status == BOSC_AME_DMA_STATUS_BUSY);
  CHECK(bosc_ame_dma_last_src[0] == src);

  // bus_error takes precedence when both bits are set.
  reset_regs();
  fence_reset();
  prepare_wait(0u, src);
  fake_regs[0][REG_STATUS] =
      BOSC_AME_DMA_STATUS_BUSY | BOSC_AME_DMA_STATUS_BUS_ERROR;
  EXPECT_FAIL(0u, src, BOSC_AME_DMA_FAIL_BUS_ERROR);

  // Bad channel: reported before any register access.
  reset_regs();
  fence_reset();
  EXPECT_FAIL(2u, src, BOSC_AME_DMA_FAIL_BAD_CHANNEL);
  CHECK(fail_status == UINT64_C(0));
  CHECK(mmio_write_count == 0u);
  CHECK(mmio_reads[0] == 0u && mmio_reads[1] == 0u);
  for (unsigned channel = 0; channel < 2u; ++channel)
    for (unsigned reg = 0; reg < 5u; ++reg)
      CHECK(fake_regs[channel][reg] == UINT64_C(0));

  // Token 0 is bosc_ame_dma_start's "not launched" sentinel.
  reset_regs();
  fence_reset();
  EXPECT_FAIL(0u, BOSC_AME_DMA_INVALID_TOKEN, BOSC_AME_DMA_FAIL_INVALID_TOKEN);
  CHECK(fail_status == UINT64_C(0));
  CHECK(mmio_write_count == 0u);
  CHECK(mmio_reads[0] == 0u && mmio_reads[1] == 0u);
  for (unsigned channel = 0; channel < 2u; ++channel)
    for (unsigned reg = 0; reg < 5u; ++reg)
      CHECK(fake_regs[channel][reg] == UINT64_C(0));

  // Even a nonzero token cannot wait a channel that was never started.
  EXPECT_FAIL(0u, src, BOSC_AME_DMA_FAIL_INVALID_TOKEN);
  CHECK(mmio_write_count == 0u);
  CHECK(mmio_reads[0] == 0u && mmio_reads[1] == 0u);

  // Invalid input also leaves an already launched request untouched.
  prepare_wait(0u, src);
  EXPECT_FAIL(0u, BOSC_AME_DMA_INVALID_TOKEN, BOSC_AME_DMA_FAIL_INVALID_TOKEN);
  CHECK(mmio_write_count == 0u);
  CHECK(mmio_reads[0] == 0u && mmio_reads[1] == 0u);
  CHECK(bosc_ame_dma_last_src[0] == src);
  bosc_ame_dma_wait(0u, src);

  //
  // Token mismatch: the completion barrier is still performed (the SRC
  // write-back lands) so the wrong request is not left active, and then the
  // mismatch fails with the status that was read after the barrier.
  //
  reset_regs();
  fence_reset();
  prepare_wait(0u, src);
  // A write-only register has no meaningful read-back value.
  fake_regs[0][REG_SRC] = UINT64_C(0xdeadbeef);
  fence_armed = 1;
  fence_expected_src = src;
  fence_expected_status = BOSC_AME_DMA_STATUS_BUS_ERROR;
  const uint64_t mismatched_token = src + UINT64_C(0x40);
  EXPECT_FAIL(0u, mismatched_token, BOSC_AME_DMA_FAIL_TOKEN_MISMATCH);
  CHECK(fail_status == BOSC_AME_DMA_STATUS_BUS_ERROR);
  CHECK(fake_regs[0][REG_SRC] == src);
  CHECK(fake_regs[0][REG_START] == BOSC_AME_DMA_START_COMMAND);
  CHECK(mmio_write_count == 1u);
  CHECK(mmio_write_offsets[0] == BOSC_AME_DMA_SRC_OFFSET);
  CHECK(mmio_write_values[0] == src);
  CHECK(mmio_reads[0] == 1u && mmio_reads[1] == 0u);
  CHECK(fence_calls == 2 && fence_saw_write_back == 1);
  CHECK(bosc_ame_dma_last_src[0] == BOSC_AME_DMA_INVALID_TOKEN);
}

// ------------------------------------------------------------------
// 4: the SRC write-back is the completion barrier, ordered before the
// STATUS read, and never relaunches the transfer
// ------------------------------------------------------------------

static void test_wait_write_back_barrier(void) {
  static const uint64_t src = UINT64_C(0x20000000);
  static const uint64_t dst = UINT64_C(0x80000000);
  static const uint64_t len = UINT64_C(128);

  // Success case: the probe publishes a clean STATUS during the first fence,
  // so a normal return proves the read happened after the barrier.
  reset_regs();
  fence_reset();
  fail_reset();
  prepare_wait(0u, src);
  fake_regs[0][REG_STATUS] = BOSC_AME_DMA_STATUS_BUSY; // still busy

  fence_armed = 1;
  fence_expected_src = src;
  fence_expected_status = UINT64_C(0);
  fence_channel = 0u;
  bosc_ame_dma_wait(0u, src);

  CHECK(fail_seen == 0);
  CHECK(fence_calls == 2);
  CHECK(fence_saw_write_back == 1);
  CHECK(fake_regs[0][REG_STATUS] == UINT64_C(0));
  CHECK(fake_regs[0][REG_DST] == dst);
  CHECK(fake_regs[0][REG_LEN] == len);
  CHECK(fake_regs[0][REG_START] == BOSC_AME_DMA_START_COMMAND);
  CHECK(mmio_writes[0] == 1u && mmio_reads[0] == 1u);

  // Failure case: the same probe publishes bus_error during the first fence.
  // The wait must report that status, which again places the read after the
  // write-back.
  reset_regs();
  fence_reset();
  fail_reset();
  prepare_wait(0u, src);
  fake_regs[0][REG_STATUS] = UINT64_C(0);
  fence_armed = 1;
  fence_expected_src = src;
  fence_expected_status = BOSC_AME_DMA_STATUS_BUS_ERROR;
  fence_channel = 0u;
  EXPECT_FAIL(0u, src, BOSC_AME_DMA_FAIL_BUS_ERROR);
  CHECK(fence_calls == 2);
  CHECK(fence_saw_write_back == 1);
  CHECK(fail_status == BOSC_AME_DMA_STATUS_BUS_ERROR);
  CHECK(mmio_writes[0] == 1u && mmio_reads[0] == 1u);
}

// ------------------------------------------------------------------
// 5: channel 1 uses the channel-1 base
// ------------------------------------------------------------------

static void test_channel1_base(void) {
  static const uint64_t src = UINT64_C(0x20400000);
  static const uint64_t dst = UINT64_C(0x90000000);
  static const uint64_t len = UINT64_C(64);

  reset_regs();
  fence_reset();
  fail_reset();
  const uint64_t token = bosc_ame_dma_start(1u, src, dst, len);

  // Channel 1 register file.
  CHECK(fake_regs[1][REG_SRC] == src);
  CHECK(fake_regs[1][REG_DST] == dst);
  CHECK(fake_regs[1][REG_LEN] == len);
  CHECK(fake_regs[1][REG_START] == BOSC_AME_DMA_START_COMMAND);
  CHECK(token == src);

  // Channel 0 must be untouched.
  for (unsigned reg = 0; reg < 5u; ++reg)
    CHECK(fake_regs[0][reg] == UINT64_C(0));

  // A clean wait on channel 1 writes back channel 1's SRC.
  fake_regs[1][REG_STATUS] = UINT64_C(0x0);
  bosc_ame_dma_wait(1u, src);
  CHECK(fail_seen == 0);
  CHECK(fake_regs[1][REG_SRC] == src);
  CHECK(fake_regs[1][REG_STATUS] == UINT64_C(0x0));
  for (unsigned reg = 0; reg < 5u; ++reg)
    CHECK(fake_regs[0][reg] == UINT64_C(0));
}

static void test_mismatch_preserves_other_channel(void) {
  static const uint64_t src0 = UINT64_C(0x20000000);
  static const uint64_t src1 = UINT64_C(0x20400000);

  reset_regs();
  fence_reset();
  CHECK(bosc_ame_dma_start(0u, src0, UINT64_C(0x80000000), UINT64_C(128)) ==
        src0);
  CHECK(bosc_ame_dma_start(1u, src1, UINT64_C(0x80000100), UINT64_C(128)) ==
        src1);
  mmio_reset();
  fake_regs[0][REG_STATUS] = BOSC_AME_DMA_STATUS_BUS_ERROR;
  fake_regs[1][REG_STATUS] = UINT64_C(0);

  EXPECT_FAIL(0u, src1, BOSC_AME_DMA_FAIL_TOKEN_MISMATCH);
  CHECK(fake_regs[0][REG_SRC] == src0);
  CHECK(bosc_ame_dma_last_src[0] == BOSC_AME_DMA_INVALID_TOKEN);
  CHECK(bosc_ame_dma_last_src[1] == src1);
  CHECK(mmio_writes[0] == 1u && mmio_reads[0] == 1u);
  CHECK(mmio_writes[1] == 0u && mmio_reads[1] == 0u);

  // The test's longjmp replaces task abort; the sibling request is intact.
  bosc_ame_dma_wait(1u, src1);
  CHECK(bosc_ame_dma_last_src[1] == BOSC_AME_DMA_INVALID_TOKEN);
  CHECK(mmio_writes[1] == 1u && mmio_reads[1] == 1u);
}

// ------------------------------------------------------------------
// 6: a staged transfer reproduces a direct copy, a failed one takes
// the error path and leaves the destination untouched
// ------------------------------------------------------------------

//
// Simulated device for the numerical check.  The two SPM banks and the DDR
// source window are backed by host arrays that the test itself declares as the
// compile-time bases, so the model below sees exactly the addresses the
// runtime programmed: it performs a 64-byte-granular contiguous block move and
// raises bus_error when an endpoint falls outside the simulated windows.
//
// Guard bytes surround every bank, so a transfer that strayed outside its
// [dst, dst + len) window is caught even at the edge of a bank.
//
#define SIM_BANK_COUNT 2u
#define SIM_BANK_BYTES ((size_t)4096)
#define SIM_DDR_BYTES ((size_t)4096)
#define SIM_DDR_BASE UINT64_C(0x80000000)
#define SIM_GUARD_BYTES ((size_t)64)
#define SIM_SENTINEL ((uint8_t)0xA5)

// The SPM bases of the hardware map, used as register payloads only.
#define SPM0_BASE_VAL ((uint64_t)BOSC_AME_SPM0_BASE)
#define SPM1_BASE_VAL ((uint64_t)BOSC_AME_SPM1_BASE)

static uint8_t sim_bank[SIM_BANK_COUNT][SIM_BANK_BYTES + 2 * SIM_GUARD_BYTES];
static uint8_t sim_ddr[SIM_DDR_BYTES];

static uint64_t sim_bank_base(unsigned bank) {
  return bank == 0u ? (uint64_t)BOSC_AME_SPM0_BASE
                    : (uint64_t)BOSC_AME_SPM1_BASE;
}

// A byte range that stays inside [base, base + size).
static int sim_in_window(uint64_t addr, uint64_t base, size_t size,
                         uint64_t len) {
  return addr >= base && len <= (uint64_t)size &&
         addr - base <= (uint64_t)size - len;
}

static int sim_bytes_are(const uint8_t *data, size_t size, uint8_t value) {
  for (size_t i = 0; i < size; ++i) {
    if (data[i] != value)
      return 0;
  }
  return 1;
}

//
// Everything the bank holds outside [offset, offset + len) must still be the
// sentinel: a transfer that wrote even one byte too many (or too early) is
// caught here, including the guard bytes at the bank edges.
//
static int sim_bank_outside_window_is_sentinel(unsigned bank, size_t offset,
                                               size_t len) {
  const size_t begin = SIM_GUARD_BYTES + offset;
  const size_t end = begin + len;
  for (size_t i = 0; i < sizeof(sim_bank[bank]); ++i) {
    if (i >= begin && i < end)
      continue;
    if (sim_bank[bank][i] != SIM_SENTINEL)
      return 0;
  }
  return 1;
}

//
// Play the part of the hardware for one channel: run the request the runtime
// left in the configuration registers and publish its completion in STATUS.
// Returns 1 when the move was performed, 0 on a modelled bus error.
//
static int sim_commit(uint32_t channel, int force_bus_error) {
  const uint64_t src = fake_regs[channel][REG_SRC];
  const uint64_t dst = fake_regs[channel][REG_DST];
  const uint64_t len = fake_regs[channel][REG_LEN];

  // No device activity before the runtime launched the transfer.
  if (fake_regs[channel][REG_START] != BOSC_AME_DMA_START_COMMAND) {
    fake_regs[channel][REG_STATUS] = BOSC_AME_DMA_STATUS_BUS_ERROR;
    return 0;
  }

  int bank = -1;
  for (unsigned i = 0; i < SIM_BANK_COUNT; ++i) {
    if (sim_in_window(dst, sim_bank_base(i), SIM_BANK_BYTES, len))
      bank = (int)i;
  }
  if (force_bus_error || bank < 0 ||
      !sim_in_window(src, SIM_DDR_BASE, SIM_DDR_BYTES, len)) {
    // Completion must clear busy and report the error: the destination keeps
    // whatever it had, so a failed wait never exposes half-moved data.
    fake_regs[channel][REG_STATUS] =
        BOSC_AME_DMA_STATUS_BUS_ERROR; // busy cleared
    return 0;
  }

  memcpy(sim_bank[(unsigned)bank] + SIM_GUARD_BYTES +
             (size_t)(dst - sim_bank_base((unsigned)bank)),
         sim_ddr + (size_t)(src - SIM_DDR_BASE), (size_t)len);
  fake_regs[channel][REG_STATUS] = UINT64_C(0); // completed, not busy
  return 1;
}

// A staged move into the bank, and the same bytes read straight from DDR.
static void test_staged_copy_matches_direct_copy(void) {
  static const uint64_t src = SIM_DDR_BASE + UINT64_C(64); // non-zero offset
  static const uint64_t dst = SPM0_BASE_VAL + UINT64_C(256);
  static const uint64_t len = UINT64_C(1024);
  uint8_t baseline[1024];

  for (size_t i = 0; i < SIM_DDR_BYTES; ++i)
    sim_ddr[i] = (uint8_t)((i * 7u + 3u) & 0xffu);
  for (size_t b = 0; b < SIM_BANK_COUNT; ++b)
    for (size_t i = 0; i < sizeof(sim_bank[b]); ++i)
      sim_bank[b][i] = SIM_SENTINEL;

  reset_regs();
  fence_reset();
  fail_reset();
  const uint64_t token = bosc_ame_dma_start(0u, src, dst, len);
  CHECK(token == src);
  CHECK(sim_commit(0u, 0) == 1);
  bosc_ame_dma_wait(0u, token);
  CHECK(fail_seen == 0);

  // The DDR-direct baseline: staging must reproduce it byte for byte.
  memcpy(baseline, sim_ddr + (size_t)(src - SIM_DDR_BASE), (size_t)len);
  CHECK(memcmp(sim_bank[0] + SIM_GUARD_BYTES + (size_t)(dst - SPM0_BASE_VAL),
               baseline, (size_t)len) == 0);

  // Nothing outside [dst, dst + len) moved, guards included.
  CHECK(sim_bank_outside_window_is_sentinel(0u, (size_t)(dst - SPM0_BASE_VAL),
                                            (size_t)len) == 1);

  // The second bank was not touched by a channel-0 transfer.
  CHECK(sim_bytes_are(sim_bank[1], sizeof(sim_bank[1]), SIM_SENTINEL) == 1);
}

// Two engines staging into the two banks at the same time, then two waits.
static void test_dual_channel_staging(void) {
  static const uint64_t src0 = SIM_DDR_BASE + UINT64_C(0);
  static const uint64_t src1 = SIM_DDR_BASE + UINT64_C(2048);
  static const uint64_t len = UINT64_C(1024);
  const uint64_t dst0 = SPM0_BASE_VAL + UINT64_C(0);
  const uint64_t dst1 = SPM1_BASE_VAL + UINT64_C(128);
  uint8_t baseline0[1024];
  uint8_t baseline1[1024];

  for (size_t i = 0; i < SIM_DDR_BYTES; ++i)
    sim_ddr[i] = (uint8_t)((i * 13u + 11u) & 0xffu);
  for (size_t b = 0; b < SIM_BANK_COUNT; ++b)
    for (size_t i = 0; i < sizeof(sim_bank[b]); ++i)
      sim_bank[b][i] = SIM_SENTINEL;

  reset_regs();
  fence_reset();
  fail_reset();
  // Both requests are launched before either is completed.
  const uint64_t token0 = bosc_ame_dma_start(0u, src0, dst0, len);
  const uint64_t token1 = bosc_ame_dma_start(1u, src1, dst1, len);
  CHECK(token0 == src0);
  CHECK(token1 == src1);
  CHECK(fake_regs[0][REG_START] == BOSC_AME_DMA_START_COMMAND);
  CHECK(fake_regs[1][REG_START] == BOSC_AME_DMA_START_COMMAND);
  CHECK(bosc_ame_dma_last_src[0] == src0);
  CHECK(bosc_ame_dma_last_src[1] == src1);

  CHECK(sim_commit(1u, 0) == 1);
  CHECK(sim_commit(0u, 0) == 1);
  bosc_ame_dma_wait(0u, token0);
  CHECK(bosc_ame_dma_last_src[0] == BOSC_AME_DMA_INVALID_TOKEN);
  CHECK(bosc_ame_dma_last_src[1] == src1);
  bosc_ame_dma_wait(1u, token1);
  CHECK(bosc_ame_dma_last_src[1] == BOSC_AME_DMA_INVALID_TOKEN);
  CHECK(fail_seen == 0);

  memcpy(baseline0, sim_ddr + (size_t)(src0 - SIM_DDR_BASE), (size_t)len);
  memcpy(baseline1, sim_ddr + (size_t)(src1 - SIM_DDR_BASE), (size_t)len);
  CHECK(memcmp(sim_bank[0] + SIM_GUARD_BYTES, baseline0, (size_t)len) == 0);
  CHECK(memcmp(sim_bank[1] + SIM_GUARD_BYTES + 128u, baseline1, (size_t)len) ==
        0);

  // Each engine wrote only its own bank, and only inside its own window.
  CHECK(sim_bank_outside_window_is_sentinel(0u, 0u, (size_t)len) == 1);
  CHECK(sim_bank_outside_window_is_sentinel(1u, 128u, (size_t)len) == 1);
}

// A bus error must take the error path and leave the destination direct.
static void test_failed_staging_is_not_consumed(void) {
  static const uint64_t src = SIM_DDR_BASE + UINT64_C(64);
  static const uint64_t len = UINT64_C(1024);
  // One bank-width past the start of bank 0: outside the bank.
  const uint64_t bad_dst = SPM0_BASE_VAL + (uint64_t)SIM_BANK_BYTES;

  for (size_t i = 0; i < SIM_DDR_BYTES; ++i)
    sim_ddr[i] = (uint8_t)((i * 5u + 1u) & 0xffu);
  for (size_t i = 0; i < sizeof(sim_bank[0]); ++i)
    sim_bank[0][i] = SIM_SENTINEL;

  reset_regs();
  fence_reset();
  const uint64_t token = bosc_ame_dma_start(0u, src, bad_dst, len);
  CHECK(token == src);
  CHECK(sim_commit(0u, 0) == 0);
  EXPECT_FAIL(0u, token, BOSC_AME_DMA_FAIL_BUS_ERROR);
  CHECK((fail_status & BOSC_AME_DMA_STATUS_BUS_ERROR) != 0u);
  CHECK((fail_status & BOSC_AME_DMA_STATUS_BUSY) == 0u);
  // The failure path returns before the tile load: no byte of the bank was
  // written, so there is no half-moved data to be mistaken for a result.
  CHECK(sim_bytes_are(sim_bank[0], sizeof(sim_bank[0]), SIM_SENTINEL) == 1);
}

// ------------------------------------------------------------------
// start argument checks (only with NDEBUG: a debug build asserts)
// ------------------------------------------------------------------

#ifdef NDEBUG
static void test_invalid_arguments(void) {
  static const uint64_t token_src = UINT64_C(0x20000000);

  reset_regs();
  fence_reset();
  // Bad channel (>= 2).
  CHECK(bosc_ame_dma_start(2u, token_src, UINT64_C(0x80000000), UINT64_C(64)) ==
        BOSC_AME_DMA_INVALID_TOKEN);
  // Zero length.
  CHECK(bosc_ame_dma_start(0u, token_src, UINT64_C(0x80000000), UINT64_C(0)) ==
        BOSC_AME_DMA_INVALID_TOKEN);
  // Length not a multiple of 64.
  CHECK(bosc_ame_dma_start(0u, token_src, UINT64_C(0x80000000), UINT64_C(63)) ==
        BOSC_AME_DMA_INVALID_TOKEN);
  // Misaligned source.
  CHECK(bosc_ame_dma_start(0u, token_src | UINT64_C(1), UINT64_C(0x80000000),
                           UINT64_C(64)) == BOSC_AME_DMA_INVALID_TOKEN);
  // Misaligned destination.
  CHECK(bosc_ame_dma_start(0u, token_src, UINT64_C(0x80000000) | UINT64_C(8),
                           UINT64_C(64)) == BOSC_AME_DMA_INVALID_TOKEN);
  CHECK(bosc_ame_dma_start(0u, UINT64_C(0), UINT64_C(0x80000000),
                           UINT64_C(64)) == BOSC_AME_DMA_INVALID_TOKEN);
  CHECK(bosc_ame_dma_start(0u, token_src, UINT64_C(0), UINT64_C(64)) ==
        BOSC_AME_DMA_INVALID_TOKEN);
  CHECK(mmio_write_count == 0u);
  CHECK(mmio_reads[0] == 0u && mmio_reads[1] == 0u);
  // No register may have been written by any rejected call.
  for (unsigned channel = 0; channel < 2u; ++channel) {
    for (unsigned reg = 0; reg < 5u; ++reg) {
      CHECK(fake_regs[channel][reg] == UINT64_C(0));
    }
    CHECK(bosc_ame_dma_last_src[channel] == BOSC_AME_DMA_INVALID_TOKEN);
  }

  // A rejected call must not overwrite the token of an existing launch.
  prepare_wait(0u, token_src);
  CHECK(bosc_ame_dma_start(0u, token_src + UINT64_C(0x40), UINT64_C(0x80000000),
                           UINT64_C(63)) == BOSC_AME_DMA_INVALID_TOKEN);
  CHECK(bosc_ame_dma_last_src[0] == token_src);
  CHECK(mmio_write_count == 0u && mmio_reads[0] == 0u);
  CHECK(fake_regs[0][REG_SRC] == token_src);
  bosc_ame_dma_wait(0u, token_src);
}
#endif // NDEBUG

static void test_cpu_ddr_policy(void) {
  reset_regs();
  bosc_ame_ddr_write_begin();
  CHECK(ddr_policy == POLICY_WRITE);
  bosc_ame_ddr_read_begin();
  CHECK(ddr_policy == POLICY_READ);
  bosc_ame_ddr_write_begin();
  CHECK(ddr_policy == POLICY_WRITE);
  CHECK(mmio_write_count == 0u);
  CHECK(mmio_reads[0] == 0u && mmio_reads[1] == 0u);
  CHECK(bosc_ame_dma_last_src[0] == BOSC_AME_DMA_INVALID_TOKEN);
  CHECK(bosc_ame_dma_last_src[1] == BOSC_AME_DMA_INVALID_TOKEN);
}

int main(void) {
  bosc_ame_dma_fail_hook = test_fail_hook;

  test_cpu_ddr_policy();
  test_start_layout_and_token();
  test_wait_success();
  test_wait_failures();
  test_wait_write_back_barrier();
  test_channel1_base();
  test_mismatch_preserves_other_channel();
  test_staged_copy_matches_direct_copy();
  test_dual_channel_staging();
  test_failed_staging_is_not_consumed();
#ifdef NDEBUG
  test_invalid_arguments();
#endif

  if (failures != 0) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return EXIT_FAILURE;
  }
  printf("bosc_ame_dma: all checks passed%s\n",
#ifdef NDEBUG
         " (NDEBUG: start argument checks included)"
#else
         ""
#endif
  );
  return EXIT_SUCCESS;
}
