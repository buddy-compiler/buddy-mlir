//===- bosc_ame_dma.h - BOSC AME DMA driver public API --------------------===//
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

#ifndef BOSC_AME_DMA_H
#define BOSC_AME_DMA_H

//
// BOSCAME SPM/DMA runtime helper.
//
// Hardware contract (fixed, do not invent alternatives):
//   - 2 SPM banks, 128 KiB each: SPM0 at 0x20000000, SPM1 at 0x20400000.
//   - 2 DMA channels: DMA0 registers at 0x60000000, DMA1 at 0x60010000.
//   - Per-channel register file, 64-bit little-endian MMIO:
//       0x00 SRC     write-only source physical address
//       0x08 DST     write-only destination physical address
//       0x10 LEN     write-only transfer length (multiple of 64)
//       0x18 START   write-only launch command
//       0x20 STATUS  read-only bit0 busy, bit1 bus_error (cleared by START)
//   - Transfer granularity is 64 bytes; both endpoints must be 64 B aligned.
//   - Launch protocol: write SRC/DST/LEN, fence, write START.
//   - Completion protocol (dma_wait): write the request's effective SRC back
//     to the SRC register, wait for that configuration write transaction and
//     fence iorw, iorw, then take one volatile 64-bit STATUS load and keep the
//     necessary order barrier before later accesses.  The SRC write-back is
//     the completion barrier because the hardware only accepts configuration
//     writes while it is idle; busy is live state, not a sticky done flag, so
//     the wait must not be redefined as mere busy polling.  There is no cancel
//     and no timeout in this version.
//   - On the delivered E6 platform, initialize the PMP region boundaries
//     before using this helper.  The helper follows the hardware reference's
//     CSR 0xbc0 write-through-store / uncached-load policy, which assumes that
//     platform setup.  Address-space setup belongs to the caller as well.
//
// The DMA base addresses are compile-time overridable so the same runtime can
// be linked against a host fake register file in tests.  The pointer to one
// register is always formed as:
//
//     (volatile uint64_t *)(BOSC_AME_DMAx_BASE + byte_offset)
//
// This header is intentionally C and C++ compatible.
//

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ------------------------------------------------------------------
// SPM banks
// ------------------------------------------------------------------

#ifndef BOSC_AME_SPM0_BASE
#define BOSC_AME_SPM0_BASE ((uintptr_t)UINT32_C(0x20000000))
#endif
#ifndef BOSC_AME_SPM1_BASE
#define BOSC_AME_SPM1_BASE ((uintptr_t)UINT32_C(0x20400000))
#endif

//
// The SPM bases are compile-time hardware constants, not runtime parameters.
// The compiler (`-lower-bosc-ame`) compiles `bosc_ame.spm.view {bank, offset}`
// directly into the pointer `BOSC_AME_SPMx_BASE + offset`, so an SPM kernel
// takes no workspace-root argument: the kernel ABI only carries the DDR
// buffers, and both banks are windows in the same address space.
//

// Each SPM bank is 128 KiB.
#define BOSC_AME_SPM_BANK_BYTES UINT32_C(0x20000)

// ------------------------------------------------------------------
// DMA channels and their register bases
// ------------------------------------------------------------------

#ifndef BOSC_AME_DMA0_BASE
#define BOSC_AME_DMA0_BASE ((uintptr_t)UINT32_C(0x60000000))
#endif
#ifndef BOSC_AME_DMA1_BASE
#define BOSC_AME_DMA1_BASE ((uintptr_t)UINT32_C(0x60010000))
#endif

#define BOSC_AME_DMA_CHANNELS UINT32_C(2)
#define BOSC_AME_DMA_CHANNEL_STRIDE UINT32_C(0x10000)

// Byte offsets inside one DMA channel register file.
#define BOSC_AME_DMA_SRC_OFFSET UINT32_C(0x00)
#define BOSC_AME_DMA_DST_OFFSET UINT32_C(0x08)
#define BOSC_AME_DMA_LEN_OFFSET UINT32_C(0x10)
#define BOSC_AME_DMA_START_OFFSET UINT32_C(0x18)
#define BOSC_AME_DMA_STATUS_OFFSET UINT32_C(0x20)

// STATUS bits.
#define BOSC_AME_DMA_STATUS_BUSY UINT64_C(0x1)
#define BOSC_AME_DMA_STATUS_BUS_ERROR UINT64_C(0x2)

// Transfer granularity / required endpoint alignment, in bytes.
#define BOSC_AME_DMA_ALIGNMENT UINT32_C(64)

// Value written to START to launch a transfer.
#define BOSC_AME_DMA_START_COMMAND UINT64_C(1)

//
// Value returned by bosc_ame_dma_start for malformed arguments.
//
// bosc_ame_dma_start normally returns the effective source address (the token
// carried by the lowered IR).  Physical address 0 is not a legal aligned
// transfer endpoint on this target, so 0 is used as the "not launched"
// sentinel.  The function performs no register write in that case.
// bosc_ame_dma_wait explicitly rejects this token.
//
#define BOSC_AME_DMA_INVALID_TOKEN UINT64_C(0)

//
// Assertion hook used for programming errors inside bosc_ame_dma_start.
//
// Defaults to the standard assert() so debug builds trap on a bad call.
// Override it (e.g. -DBOSC_AME_DMA_ASSERT(x)=...) for freestanding builds or
// to exercise the defined error-return paths in tests.  Defining NDEBUG also
// removes the default assertion.  Error handling that must survive a
// freestanding build does NOT use this macro: bosc_ame_dma_wait reports every
// failure through bosc_ame_dma_fail_hook below, which is active in release
// builds too.
//
#ifndef BOSC_AME_DMA_ASSERT
#include <assert.h>
#define BOSC_AME_DMA_ASSERT(cond) assert(cond)
#endif

//
// Failure reasons passed to the bosc_ame_dma_wait error hook.
//
typedef enum {
  BOSC_AME_DMA_FAIL_BAD_CHANNEL = 1,
  BOSC_AME_DMA_FAIL_INVALID_TOKEN = 2,
  BOSC_AME_DMA_FAIL_TOKEN_MISMATCH = 3,
  BOSC_AME_DMA_FAIL_BUS_ERROR = 4,
  BOSC_AME_DMA_FAIL_BUSY = 5
} bosc_ame_dma_fail_reason;

//
// Error hook of bosc_ame_dma_wait.  It receives the channel, a
// bosc_ame_dma_fail_reason, the raw STATUS word (0 when no register was read)
// and the token the caller passed.
//
// The shipped default traps, so a failed transfer can never be consumed as if
// it were data.  A board that wants a richer report replaces the pointer
// before the first wait; the replacement must not return.
//
// Because the hook does not return, a failure on one channel aborts the task
// without draining the other channel: the two engines are independent, so the
// sibling request may still be in flight.  A policy that returns to the caller
// (buffer reuse, next task) has to add explicit drain/reset handling for the
// other channel first.
//
typedef void (*bosc_ame_dma_fail_fn)(uint32_t channel, uint32_t reason,
                                     uint64_t status, uint64_t src_token);
extern bosc_ame_dma_fail_fn bosc_ame_dma_fail_hook;

// ------------------------------------------------------------------
// ABI implemented by bosc_ame_dma.c and emitted by the MLIR pass
// ------------------------------------------------------------------

//
// Select the delivered E6 DDR access policy, draining earlier accesses with
// a fence before changing CSR 0xbc0. Platform PMP setup is required first.
// read_begin selects uncached DDR loads so DMA-written data is visible;
// write_begin restores the policy required for CPU DDR stores. A DDR
// read/modify/write must select reads, load its operands, then select writes
// before storing. dma_wait leaves the read policy active. These helpers do
// not access SPM or launch/complete a DMA; they are no-ops on host builds.
//
void bosc_ame_ddr_read_begin(void);
void bosc_ame_ddr_write_begin(void);

//
// Program SRC, DST, LEN and then START on the given channel, in that order.
// Returns the effective source address on success.  Invalid arguments make
// no register write and return BOSC_AME_DMA_INVALID_TOKEN.  Successful launches
// save the source address in software; SRC is write-only and is never read.
// The runtime owns both channels and requires serialized calls, with exactly
// one wait per start before the channel is reused.
//
uint64_t bosc_ame_dma_start(uint32_t channel, uint64_t src, uint64_t dst,
                            uint64_t len);

//
// Complete the request identified by src_token (the effective source address
// returned by the matching bosc_ame_dma_start) and check its STATUS.
//
// A normal return means "the transfer completed and succeeded".  The hardware
// contract is honoured exactly: SRC write-back (the completion barrier), fence,
// one volatile 64-bit STATUS read, fence, then the checks.
//
// Failure conditions, each reported through bosc_ame_dma_fail_hook with the
// raw status when one was read:
//   - invalid channel (no register access),
//   - src_token == BOSC_AME_DMA_INVALID_TOKEN or no outstanding launch recorded
//     on the channel (no register access),
//   - a token that differs from the source saved by start: the originally
//     launched source is written back so the barrier is never skipped, then
//     the mismatch fails,
//   - STATUS with bus_error set: the transfer data is invalid,
//   - STATUS with busy still set after the barrier: the completion guarantee
//     did not hold.
//
// The check uses the defined bits, not "status == 0".  It does not depend on
// BOSC_AME_DMA_ASSERT or NDEBUG.  A completed request is removed from the
// software record even on bus_error or token mismatch; an unexpectedly busy
// request remains recorded until the task is aborted or the device is reset.
//
void bosc_ame_dma_wait(uint32_t channel, uint64_t src_token);

// Explicit NH/E6-only runtime selection. Generic and host builds retain the
// atomic allocator. A value of 1 requires the delivered single E6 hart0 flow,
// real disjoint NH/E6 linker stacks, and no resumable interrupt/NMI/monitor
// allocator entry. Its bounded metadata operations save/clear/restore MIE;
// generated kernel execution and logical nested-frame lifetimes stay unmasked.
//
#ifndef BOSC_AME_PRIVATE_DDR_NR_SINGLE_E6
#define BOSC_AME_PRIVATE_DDR_NR_SINGLE_E6 0
#endif
#if BOSC_AME_PRIVATE_DDR_NR_SINGLE_E6 != 0 &&                                  \
    BOSC_AME_PRIVATE_DDR_NR_SINGLE_E6 != 1
#error "BOSC_AME_PRIVATE_DDR_NR_SINGLE_E6 must be 0 or 1"
#endif

// Managed private DDR frame ABI. The linker reserves a nonoverlapping arena
// with __bosc_ame_private_ddr_start/end inside __nh_e6_ddr_window_start/end.
// Payload and both 64-byte guards are invocation-local; nested calls are LIFO.
// Other harts/concurrent allocator entry, exhaustion, malformed releases and
// overwritten guards trap through the failure hook. Payload is uninitialized.
// Calls must occur before/after all AME register lifetimes, respectively.
// CPU accesses use bosc_ame_ddr_read_begin/write_begin exactly as external DDR.
//
#define BOSC_AME_PRIVATE_DDR_MAX_PAYLOAD UINT64_C(262016)
enum {
  BOSC_AME_PRIVATE_DDR_FAIL_WINDOW = 1,
  BOSC_AME_PRIVATE_DDR_FAIL_CAPACITY = 2,
  BOSC_AME_PRIVATE_DDR_FAIL_OWNER = 3,
  BOSC_AME_PRIVATE_DDR_FAIL_LIFETIME = 4,
  BOSC_AME_PRIVATE_DDR_FAIL_GUARD = 5,
  BOSC_AME_PRIVATE_DDR_FAIL_CONCURRENT = 6
};
typedef void (*bosc_ame_private_ddr_fail_fn)(uint32_t reason, uint64_t address,
                                             uint64_t bytes);
extern bosc_ame_private_ddr_fail_fn bosc_ame_private_ddr_fail_hook;
extern uint64_t bosc_ame_private_ddr_peak_bytes;
extern uint64_t bosc_ame_private_ddr_acquires;
extern uint64_t bosc_ame_private_ddr_releases;
#ifndef BOSC_AME_PRIVATE_DDR_TRACE
#define BOSC_AME_PRIVATE_DDR_TRACE 0
#endif
#if BOSC_AME_PRIVATE_DDR_TRACE
// Optional diagnostic ring, published before a terminal failure hook. Keep
// these control cells outside the PMP window whose cache policy changes.
#define BOSC_AME_PRIVATE_DDR_TRACE_EVENTS 64u
struct bosc_ame_private_ddr_trace_event {
  uint64_t op; // 1 = acquire, 2 = release, 0x100 | failure reason
  uint64_t bytes;
  uint64_t address;
  uint64_t top_before;
  uint64_t active_before;
  uint64_t top_after;
  uint64_t active_after;
  uint64_t acquires;
  uint64_t releases;
};
extern volatile struct bosc_ame_private_ddr_trace_event
    bosc_ame_private_ddr_trace[BOSC_AME_PRIVATE_DDR_TRACE_EVENTS];
extern volatile unsigned bosc_ame_private_ddr_trace_count;
#endif
uint64_t bosc_ame_private_ddr_acquire(uint64_t bytes);
void bosc_ame_private_ddr_release(uint64_t address, uint64_t bytes);

#ifdef __cplusplus
}
#endif

#endif // BOSC_AME_DMA_H
