//===- bosc_ame_dma.c - BOSC AME DMA driver for bare-metal targets --------===//
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
// BOSCAME SPM/DMA runtime helper implementation.
//
// See bosc_ame_dma.h for the hardware contract and the ABI documentation.
// This file is additionally designed to be included directly by the host test
// after it has overridden BOSC_AME_DMA0_BASE / BOSC_AME_DMA1_BASE, so it must
// not define any name that could collide with the test translation unit.
//

#include "bosc_ame_dma.h"

// ------------------------------------------------------------------
// MMIO order barrier
// ------------------------------------------------------------------

//
// Order barrier for MMIO.  On the BOSCAME RISC-V target this is the same
// "fence iorw, iorw" used by the board reference runtime.  Everywhere else
// (host tests) it lowers to a sequentially-consistent atomic fence.
//
#ifndef BOSC_AME_DMA_FENCE
#if defined(__riscv)
#define BOSC_AME_DMA_FENCE() __asm__ volatile("fence iorw, iorw" ::: "memory")
#else
#define BOSC_AME_DMA_FENCE() __atomic_thread_fence(__ATOMIC_SEQ_CST)
#endif
#endif

// ------------------------------------------------------------------
// Delivered E6 cache policy (CSR 0xbc0)
// ------------------------------------------------------------------

//
// The delivered patch7 DMA/SPM reference uses CSR 0xbc0 to select write-through
// stores and uncached loads. Follow its `shared_writes` before DDR stores or
// programming SRC/DST/LEN/START and `shared_reads` before DDR loads or STATUS.
// The read policy left by dma_wait must be switched back before subsequent
// CPU DDR stores; a read/modify/write needs both transitions. Drain prior
// accesses before each policy change.
// The caller must first configure the platform's PMP regions as that reference
// does; these CSR values must not be used with unspecified region boundaries.
// See test/src/cases/nh_e6_dma_spm/src/main.c in the hardware delivery.
//
// The host test has no such CSR, so the transition compiles to nothing there;
// the transfer ordering under test is the fence pair.
//
#ifndef BOSC_AME_DMA_SHARED_WRITES
#if defined(__riscv)
#define BOSC_AME_DMA_SHARED_WRITES()                                           \
  __asm__ volatile(                                                            \
      "fence iorw, iorw\n\tcsrw 0xbc0, %0" ::"r"(UINT64_C(0x0f0f0f0f0f0f0f0f)) \
      : "memory")
#define BOSC_AME_DMA_SHARED_READS()                                            \
  __asm__ volatile(                                                            \
      "fence iorw, iorw\n\tcsrw 0xbc0, %0" ::"r"(UINT64_C(0x0f0f000f000f000f)) \
      : "memory")
#else
#define BOSC_AME_DMA_SHARED_WRITES() ((void)0)
#define BOSC_AME_DMA_SHARED_READS() ((void)0)
#endif
#endif

//
// Effective source address of the outstanding launch per channel, or 0.
//
// SRC is a write-only MMIO register (patch7 section 2.5), so the token check
// in `bosc_ame_dma_wait` cannot read it back.  The address is tracked in
// software instead, exactly like the board reference runtime's `dma_last_src`.
//
static uint64_t bosc_ame_dma_last_src[BOSC_AME_DMA_CHANNELS];

// ------------------------------------------------------------------
// Register access
// ------------------------------------------------------------------

// Base address of one DMA channel's register file.
static inline uintptr_t bosc_ame_dma_base(uint32_t channel) {
  return (channel == 1u) ? (uintptr_t)BOSC_AME_DMA1_BASE
                         : (uintptr_t)BOSC_AME_DMA0_BASE;
}

//
// Volatile 64-bit MMIO register pointer.  The pointer is always formed as
// (volatile uint64_t *)(base + byte_offset) so that overriding the base with
// the address of a fake register array is enough to redirect every access.
//
static inline volatile uint64_t *bosc_ame_dma_reg(uint32_t channel,
                                                  uintptr_t byte_offset) {
  return (volatile uint64_t *)(bosc_ame_dma_base(channel) + byte_offset);
}

// Tests can observe register transactions without making write-only SRC
// readable.
#ifndef BOSC_AME_DMA_MMIO_READ64
#define BOSC_AME_DMA_MMIO_READ64(channel, offset)                              \
  (*bosc_ame_dma_reg((channel), (offset)))
#endif
#ifndef BOSC_AME_DMA_MMIO_WRITE64
#define BOSC_AME_DMA_MMIO_WRITE64(channel, offset, value)                      \
  (*bosc_ame_dma_reg((channel), (offset)) = (value))
#endif

static inline int bosc_ame_dma_valid_channel(uint32_t channel) {
  return channel < BOSC_AME_DMA_CHANNELS;
}

// Both endpoints and the transfer length must be 64-byte aligned/multiples.
static inline int bosc_ame_dma_valid_transfer(uint64_t src, uint64_t dst,
                                              uint64_t len) {
  const uint64_t align = (uint64_t)BOSC_AME_DMA_ALIGNMENT;
  if (src == BOSC_AME_DMA_INVALID_TOKEN || dst == 0u)
    return 0;
  if (len == 0u)
    return 0;
  if ((len % align) != 0u)
    return 0;
  if (((src | dst) & (align - 1u)) != 0u)
    return 0;
  return 1;
}

// ------------------------------------------------------------------
// Failure hook
// ------------------------------------------------------------------

//
// Shipped failure hook: trap.  On the board this reaches the E6 trap entry of
// the platform startup module (common/nh_e6/nh_e6_boot.S), which captures the
// exception CSRs, publishes 0xed and stops E6; on the host it is the
// compiler's trap.  Override bosc_ame_dma_fail_hook before the first
// bosc_ame_dma_wait call to publish channel/status diagnostics first --
// common/nh_e6's nh_e6_publish_dma_failure is that override for the NH/E6
// examples.
//
// The hook must not return.  Every call below is followed by
// __builtin_unreachable() so that a violating hook is undefined behaviour
// rather than a silent continuation with stale data.
//
static void bosc_ame_dma_default_fail(uint32_t channel, uint32_t reason,
                                      uint64_t status, uint64_t src_token) {
  (void)channel;
  (void)reason;
  (void)status;
  (void)src_token;
  __builtin_trap();
}

bosc_ame_dma_fail_fn bosc_ame_dma_fail_hook = bosc_ame_dma_default_fail;

// ------------------------------------------------------------------
// Public ABI
// ------------------------------------------------------------------

void bosc_ame_ddr_read_begin(void) { BOSC_AME_DMA_SHARED_READS(); }

void bosc_ame_ddr_write_begin(void) { BOSC_AME_DMA_SHARED_WRITES(); }

uint64_t bosc_ame_dma_start(uint32_t channel, uint64_t src, uint64_t dst,
                            uint64_t len) {
  //
  // Invalid arguments are a programming error: trap in debug builds (the
  // default BOSC_AME_DMA_ASSERT), and with NDEBUG report the failure through
  // the defined BOSC_AME_DMA_INVALID_TOKEN return without touching any
  // register, so a malformed call can never launch a transfer or corrupt the
  // channel configuration.
  //
  if (!bosc_ame_dma_valid_channel(channel) ||
      !bosc_ame_dma_valid_transfer(src, dst, len)) {
    BOSC_AME_DMA_ASSERT(0 &&
                        "bosc_ame_dma_start: bad channel/length/alignment");
    return BOSC_AME_DMA_INVALID_TOKEN;
  }

  // Publish the CPU-side buffers before handing the channel to the engine.
  BOSC_AME_DMA_SHARED_WRITES();

  // Config writes in the required order: SRC, DST, LEN.
  BOSC_AME_DMA_MMIO_WRITE64(channel, BOSC_AME_DMA_SRC_OFFSET, src);
  BOSC_AME_DMA_MMIO_WRITE64(channel, BOSC_AME_DMA_DST_OFFSET, dst);
  BOSC_AME_DMA_MMIO_WRITE64(channel, BOSC_AME_DMA_LEN_OFFSET, len);

  //
  // Fence #1, after the config writes and before START: makes SRC/DST/LEN
  // visible to the DMA engine before the launch store is observed.  Matches
  // the board reference runtime (linear_ame_dma_spm.c dma_start).
  //
  BOSC_AME_DMA_FENCE();

  // START is the last store; it launches the transfer and clears bus_error.
  BOSC_AME_DMA_MMIO_WRITE64(channel, BOSC_AME_DMA_START_OFFSET,
                            BOSC_AME_DMA_START_COMMAND);

  //
  // Fence #2, after START: keeps the launch store ordered ahead of any later
  // MMIO access in program order, again matching the board reference runtime.
  //
  BOSC_AME_DMA_FENCE();

  // The lowered IR carries the effective source address as the token.
  bosc_ame_dma_last_src[channel] = src;
  return src;
}

void bosc_ame_dma_wait(uint32_t channel, uint64_t src_token) {
  if (!bosc_ame_dma_valid_channel(channel)) {
    bosc_ame_dma_fail_hook(channel, BOSC_AME_DMA_FAIL_BAD_CHANNEL, 0u,
                           src_token);
    __builtin_unreachable();
  }

  //
  // Token 0 is bosc_ame_dma_start's "not launched" sentinel: the request was
  // rejected before it programmed any register, so there is nothing to
  // complete.  A channel without an outstanding launch likewise has no
  // completion barrier to perform.  Reject both cases before MMIO.
  //
  if (src_token == BOSC_AME_DMA_INVALID_TOKEN ||
      bosc_ame_dma_last_src[channel] == BOSC_AME_DMA_INVALID_TOKEN) {
    bosc_ame_dma_fail_hook(channel, BOSC_AME_DMA_FAIL_INVALID_TOKEN, 0u,
                           src_token);
    __builtin_unreachable();
  }

  //
  // Token validity: the token carried by the lowered IR is the effective SRC
  // of the request being completed.  SRC is write-only, so the expected value
  // comes from the software record written by `bosc_ame_dma_start`.  The check
  // only records the bug; the write-back below is always performed because it
  // is the completion barrier, and skipping it would leave an active transfer
  // behind.
  //
  const uint64_t launched_src = bosc_ame_dma_last_src[channel];
  const int token_mismatch = (launched_src != src_token);

  //
  // SRC write-back: the completion barrier required by the hardware contract.
  // The engine only accepts configuration writes while it is idle, so this
  // store cannot be observed to complete before the transfer it completes.
  // It only re-arms the source address; only a START store launches a
  // transfer, so the write-back never restarts anything.
  //
  BOSC_AME_DMA_SHARED_WRITES();
  BOSC_AME_DMA_MMIO_WRITE64(channel, BOSC_AME_DMA_SRC_OFFSET, launched_src);

  //
  // Order barrier: wait for the configuration write transaction (and the
  // request it retires) and order it ahead of the STATUS observation that
  // follows.  busy is live state, so this version deliberately performs a
  // single STATUS read instead of polling it.
  //
  BOSC_AME_DMA_FENCE();

  // STATUS is a read transaction: take the read side of the transition.
  BOSC_AME_DMA_SHARED_READS();
  uint64_t status =
      BOSC_AME_DMA_MMIO_READ64(channel, BOSC_AME_DMA_STATUS_OFFSET);

  // Order the STATUS read before any later access to the transferred buffer.
  BOSC_AME_DMA_FENCE();

  if ((status & BOSC_AME_DMA_STATUS_BUSY) == 0u)
    bosc_ame_dma_last_src[channel] = BOSC_AME_DMA_INVALID_TOKEN;

  //
  // The checks below only use the defined STATUS bits.  `bus_error` means the
  // transferred data is invalid; a `busy` that is still visible after the
  // completion barrier means the completion guarantee did not hold.  Both are
  // failures even though they are reported with different reasons.
  //
  if (token_mismatch) {
    bosc_ame_dma_fail_hook(channel, BOSC_AME_DMA_FAIL_TOKEN_MISMATCH, status,
                           src_token);
    __builtin_unreachable();
  }
  if ((status & (BOSC_AME_DMA_STATUS_BUSY | BOSC_AME_DMA_STATUS_BUS_ERROR)) !=
      0u) {
    uint32_t reason = (status & BOSC_AME_DMA_STATUS_BUS_ERROR) != 0u
                          ? BOSC_AME_DMA_FAIL_BUS_ERROR
                          : BOSC_AME_DMA_FAIL_BUSY;
    bosc_ame_dma_fail_hook(channel, reason, status, src_token);
    __builtin_unreachable();
  }
}

// ------------------------------------------------------------------
// Invocation-private AME DDR frames
// ------------------------------------------------------------------

//
// Ordinary E6 stack memory stays cached under both CSR policies. Native AME
// writes cannot use it as CPU-visible workspace. A private frame instead lives
// in a linker-reserved, nonoverlapping part of PMP operand region 5. The same
// read/write helpers used by external operands therefore give it identical
// CPU/device visibility. Missing linker symbols are a checked runtime error,
// never an implicit stack/global fallback.
//
#ifndef BOSC_AME_PRIVATE_DDR_START
extern char __bosc_ame_private_ddr_start[] __attribute__((weak));
extern char __bosc_ame_private_ddr_end[] __attribute__((weak));
extern char __nh_e6_ddr_window_start[] __attribute__((weak));
extern char __nh_e6_ddr_window_end[] __attribute__((weak));
#define BOSC_AME_PRIVATE_DDR_START ((uintptr_t)__bosc_ame_private_ddr_start)
#define BOSC_AME_PRIVATE_DDR_END ((uintptr_t)__bosc_ame_private_ddr_end)
#define BOSC_AME_PRIVATE_DDR_WINDOW_START ((uintptr_t)__nh_e6_ddr_window_start)
#define BOSC_AME_PRIVATE_DDR_WINDOW_END ((uintptr_t)__nh_e6_ddr_window_end)
#endif
#ifndef BOSC_AME_PRIVATE_DDR_HART_ID
#if defined(__riscv)
static uintptr_t bosc_ame_private_ddr_hart_id(void) {
  uintptr_t value;
  __asm__ volatile("csrr %0, mhartid" : "=r"(value));
  return value;
}
#define BOSC_AME_PRIVATE_DDR_HART_ID() bosc_ame_private_ddr_hart_id()
#else
#define BOSC_AME_PRIVATE_DDR_HART_ID() ((uintptr_t)0)
#endif
#endif

#define BOSC_AME_PRIVATE_DDR_MAGIC UINT64_C(0xb05ca6e0dd4f4a6e)
#define BOSC_AME_PRIVATE_DDR_GUARD UINT64_C(0xd3d3d3d3d3d3d3d3)
#define BOSC_AME_PRIVATE_DDR_OVERHEAD 128u
static uintptr_t bosc_ame_private_ddr_top;
static uintptr_t bosc_ame_private_ddr_active;
#if BOSC_AME_PRIVATE_DDR_NR_SINGLE_E6
static volatile uintptr_t bosc_ame_private_ddr_owner;
#else
static uintptr_t bosc_ame_private_ddr_owner;
#endif
#if BOSC_AME_PRIVATE_DDR_NR_SINGLE_E6
static volatile unsigned char bosc_ame_private_ddr_busy;
static uintptr_t bosc_ame_private_ddr_saved_mie;
#else
static unsigned char bosc_ame_private_ddr_busy;
#endif
uint64_t bosc_ame_private_ddr_peak_bytes;
uint64_t bosc_ame_private_ddr_acquires;
uint64_t bosc_ame_private_ddr_releases;

#if BOSC_AME_PRIVATE_DDR_TRACE
volatile struct bosc_ame_private_ddr_trace_event
    bosc_ame_private_ddr_trace[BOSC_AME_PRIVATE_DDR_TRACE_EVENTS];
volatile unsigned bosc_ame_private_ddr_trace_count;

// The ring, like allocator controls, requires a stable cache policy outside
// the operand window. Publish the count last so NH can inspect complete
// records after E6 stops, without E6 UART/progress delays.
static void bosc_ame_private_ddr_trace_record(uint64_t op, uint64_t bytes,
                                              uint64_t address,
                                              uintptr_t top_before,
                                              uintptr_t active_before) {
  unsigned count = bosc_ame_private_ddr_trace_count;
  volatile struct bosc_ame_private_ddr_trace_event *event =
      &bosc_ame_private_ddr_trace[count % BOSC_AME_PRIVATE_DDR_TRACE_EVENTS];
  event->op = op;
  event->bytes = bytes;
  event->address = address;
  event->top_before = top_before;
  event->active_before = active_before;
  event->top_after = bosc_ame_private_ddr_top;
  event->active_after = bosc_ame_private_ddr_active;
  event->acquires = bosc_ame_private_ddr_acquires;
  event->releases = bosc_ame_private_ddr_releases;
  BOSC_AME_DMA_FENCE();
  bosc_ame_private_ddr_trace_count = count + 1u;
  BOSC_AME_DMA_FENCE();
}
#endif

static void bosc_ame_private_ddr_default_fail(uint32_t reason, uint64_t address,
                                              uint64_t bytes) {
  (void)reason;
  (void)address;
  (void)bytes;
  __builtin_trap();
}
bosc_ame_private_ddr_fail_fn bosc_ame_private_ddr_fail_hook =
    bosc_ame_private_ddr_default_fail;

static void bosc_ame_private_ddr_fail(uint32_t reason, uintptr_t address,
                                      uint64_t bytes) {
#if BOSC_AME_PRIVATE_DDR_TRACE
  uintptr_t top = bosc_ame_private_ddr_top;
  uintptr_t active = bosc_ame_private_ddr_active;
  bosc_ame_private_ddr_trace_record(0x100u | reason, bytes, address, top,
                                    active);
#endif
  bosc_ame_private_ddr_fail_hook(reason, address, bytes);
  __builtin_unreachable();
}
#if BOSC_AME_PRIVATE_DDR_NR_SINGLE_E6

// This opt-in is valid only for the NH/E6 single executor image. Reject an
// unsupported hart and NH/wrong-stack context before touching allocator state.
// Resumable interrupt, NMI or monitor handlers must not invoke the allocator.
// Only the installed E6 mtvec path is proven terminal; no NMI routing is
// assumed. Never hold MIE off across generated kernel work.
#ifndef BOSC_AME_PRIVATE_DDR_E6_CONTEXT
#ifndef BOSC_AME_PRIVATE_DDR_E6_STACK_BOTTOM
extern char __e6_stack_bottom[] __attribute__((weak));
extern char __e6_stack_top[] __attribute__((weak));
extern char __nh_stack_bottom[] __attribute__((weak));
extern char __nh_stack_top[] __attribute__((weak));
#define BOSC_AME_PRIVATE_DDR_E6_STACK_BOTTOM ((uintptr_t)__e6_stack_bottom)
#define BOSC_AME_PRIVATE_DDR_E6_STACK_TOP ((uintptr_t)__e6_stack_top)
#define BOSC_AME_PRIVATE_DDR_NH_STACK_BOTTOM ((uintptr_t)__nh_stack_bottom)
#define BOSC_AME_PRIVATE_DDR_NH_STACK_TOP ((uintptr_t)__nh_stack_top)
#endif
#ifndef BOSC_AME_PRIVATE_DDR_STACK_POINTER
static uintptr_t bosc_ame_private_ddr_stack_pointer(void) {
  uintptr_t current_sp;
  __asm__ volatile("mv %0, sp" : "=r"(current_sp));
  return current_sp;
}
#define BOSC_AME_PRIVATE_DDR_STACK_POINTER()                                   \
  bosc_ame_private_ddr_stack_pointer()
#endif
static int bosc_ame_private_ddr_e6_context(void) {
  uintptr_t current_sp = BOSC_AME_PRIVATE_DDR_STACK_POINTER();
  uintptr_t bottom = BOSC_AME_PRIVATE_DDR_E6_STACK_BOTTOM;
  uintptr_t top = BOSC_AME_PRIVATE_DDR_E6_STACK_TOP;
  uintptr_t nh_bottom = BOSC_AME_PRIVATE_DDR_NH_STACK_BOTTOM;
  uintptr_t nh_top = BOSC_AME_PRIVATE_DDR_NH_STACK_TOP;
  return bottom && !(bottom & 15u) && !(top & 15u) && top > bottom &&
         top - bottom >= 4096u && nh_bottom && nh_top > nh_bottom &&
         !(nh_bottom & 15u) && !(nh_top & 15u) &&
         (top <= nh_bottom || bottom >= nh_top) && current_sp >= bottom &&
         current_sp < top;
}
#define BOSC_AME_PRIVATE_DDR_E6_CONTEXT() bosc_ame_private_ddr_e6_context()
#endif
#ifndef BOSC_AME_PRIVATE_DDR_IRQ_SAVE
static uintptr_t bosc_ame_private_ddr_irq_save(void) {
  uintptr_t status;
  __asm__ volatile("csrrci %0, mstatus, 8" : "=r"(status) : : "memory");
  return status & 8u;
}
#define BOSC_AME_PRIVATE_DDR_IRQ_SAVE() bosc_ame_private_ddr_irq_save()
#endif
#ifndef BOSC_AME_PRIVATE_DDR_IRQ_RESTORE
#define BOSC_AME_PRIVATE_DDR_IRQ_RESTORE(saved)                                \
  __asm__ volatile("csrs mstatus, %0" : : "r"((saved) & 8u) : "memory")
#endif
static uintptr_t bosc_ame_private_ddr_enter(uint64_t bytes) {
  uintptr_t hart_id = BOSC_AME_PRIVATE_DDR_HART_ID();
  if (hart_id != 0u || !BOSC_AME_PRIVATE_DDR_E6_CONTEXT())
    bosc_ame_private_ddr_fail(BOSC_AME_PRIVATE_DDR_FAIL_OWNER, 0u, bytes);
  uintptr_t saved = BOSC_AME_PRIVATE_DDR_IRQ_SAVE();
  if (bosc_ame_private_ddr_busy)
    bosc_ame_private_ddr_fail(BOSC_AME_PRIVATE_DDR_FAIL_CONCURRENT, 0u, bytes);
  bosc_ame_private_ddr_busy = 1u;
  bosc_ame_private_ddr_saved_mie = saved;
  uintptr_t owner = hart_id + 1u;
  uintptr_t expected = bosc_ame_private_ddr_owner;
  if (expected && expected != owner)
    bosc_ame_private_ddr_fail(BOSC_AME_PRIVATE_DDR_FAIL_OWNER, 0u, bytes);
  if (!expected)
    bosc_ame_private_ddr_owner = owner;
  return owner;
}
static void bosc_ame_private_ddr_leave(void) {
  uintptr_t saved = bosc_ame_private_ddr_saved_mie;
  BOSC_AME_DMA_FENCE();
  bosc_ame_private_ddr_busy = 0u;
  BOSC_AME_PRIVATE_DDR_IRQ_RESTORE(saved);
}
#else
static uintptr_t bosc_ame_private_ddr_enter(uint64_t bytes) {
  uintptr_t owner = BOSC_AME_PRIVATE_DDR_HART_ID() + 1u;
  if (!owner ||
      __atomic_test_and_set(&bosc_ame_private_ddr_busy, __ATOMIC_ACQUIRE))
    bosc_ame_private_ddr_fail(BOSC_AME_PRIVATE_DDR_FAIL_CONCURRENT, 0u, bytes);
  uintptr_t expected = 0;
  if (!__atomic_compare_exchange_n(&bosc_ame_private_ddr_owner, &expected,
                                   owner, 0, __ATOMIC_ACQ_REL,
                                   __ATOMIC_ACQUIRE) &&
      expected != owner)
    bosc_ame_private_ddr_fail(BOSC_AME_PRIVATE_DDR_FAIL_OWNER, 0u, bytes);
  return owner;
}
static void bosc_ame_private_ddr_leave(void) {
  __atomic_clear(&bosc_ame_private_ddr_busy, __ATOMIC_RELEASE);
}
#endif
static int bosc_ame_private_ddr_window_valid(void) {
  uintptr_t start = BOSC_AME_PRIVATE_DDR_START;
  uintptr_t end = BOSC_AME_PRIVATE_DDR_END;
  return start && !(start & 63u) && !(end & 63u) && end > start &&
         start >= BOSC_AME_PRIVATE_DDR_WINDOW_START &&
         end <= BOSC_AME_PRIVATE_DDR_WINDOW_END &&
         BOSC_AME_PRIVATE_DDR_WINDOW_END > BOSC_AME_PRIVATE_DDR_WINDOW_START;
}

uint64_t bosc_ame_private_ddr_acquire(uint64_t bytes) {
  uintptr_t owner = bosc_ame_private_ddr_enter(bytes);
#if BOSC_AME_PRIVATE_DDR_TRACE
  uintptr_t top_before = bosc_ame_private_ddr_top;
  uintptr_t active_before = bosc_ame_private_ddr_active;
#endif
  if (!bosc_ame_private_ddr_window_valid())
    bosc_ame_private_ddr_fail(BOSC_AME_PRIVATE_DDR_FAIL_WINDOW, 0u, bytes);
  uintptr_t start = BOSC_AME_PRIVATE_DDR_START;
  uintptr_t end = BOSC_AME_PRIVATE_DDR_END;
  uintptr_t top = bosc_ame_private_ddr_top ? bosc_ame_private_ddr_top : start;
  if (!bytes || (bytes & 63u) || bytes > BOSC_AME_PRIVATE_DDR_MAX_PAYLOAD ||
      top < start || top > end || end - top < BOSC_AME_PRIVATE_DDR_OVERHEAD ||
      bytes > end - top - BOSC_AME_PRIVATE_DDR_OVERHEAD)
    bosc_ame_private_ddr_fail(BOSC_AME_PRIVATE_DDR_FAIL_CAPACITY, top, bytes);
  BOSC_AME_DMA_SHARED_WRITES();
  volatile uint64_t *header = (volatile uint64_t *)top;
  header[0] = BOSC_AME_PRIVATE_DDR_MAGIC;
  header[1] = top;
  header[2] = bytes;
  header[3] = owner;
  header[4] = bosc_ame_private_ddr_active;
  header[5] = BOSC_AME_PRIVATE_DDR_GUARD ^ header[1] ^ header[2] ^ header[3] ^
              header[4];
  for (unsigned i = 6; i != 8; ++i)
    header[i] = BOSC_AME_PRIVATE_DDR_GUARD;
  uintptr_t payload = top + 64u;
  volatile uint64_t *tail = (volatile uint64_t *)(payload + bytes);
  for (unsigned i = 0; i != 8; ++i)
    tail[i] = BOSC_AME_PRIVATE_DDR_GUARD;
  bosc_ame_private_ddr_active = payload;
  bosc_ame_private_ddr_top = payload + bytes + 64u;
  uint64_t live = bosc_ame_private_ddr_top - start;
  if (live > bosc_ame_private_ddr_peak_bytes)
    bosc_ame_private_ddr_peak_bytes = live;
  ++bosc_ame_private_ddr_acquires;
  BOSC_AME_DMA_FENCE();
#if BOSC_AME_PRIVATE_DDR_TRACE
  bosc_ame_private_ddr_trace_record(1u, bytes, payload, top_before,
                                    active_before);
#endif
  bosc_ame_private_ddr_leave();
  return payload;
}

void bosc_ame_private_ddr_release(uint64_t address, uint64_t bytes) {
  uintptr_t owner = bosc_ame_private_ddr_enter(bytes);
#if BOSC_AME_PRIVATE_DDR_TRACE
  uintptr_t top_before = bosc_ame_private_ddr_top;
  uintptr_t active_before = bosc_ame_private_ddr_active;
#endif
  uintptr_t start = BOSC_AME_PRIVATE_DDR_START;
  uintptr_t end = BOSC_AME_PRIVATE_DDR_END;
  if (!bosc_ame_private_ddr_window_valid() ||
      address != bosc_ame_private_ddr_active || !bytes || (address & 63u) ||
      (bytes & 63u) || address < start + 64u || address > end ||
      end - address < 64u || bytes > end - address - 64u ||
      address + bytes + 64u != bosc_ame_private_ddr_top)
    bosc_ame_private_ddr_fail(BOSC_AME_PRIVATE_DDR_FAIL_LIFETIME, address,
                              bytes);
  // AME writes have completed before release. Read guards through the same
  // uncached region-5 policy used for CPU epilogues after native AME stores.
  BOSC_AME_DMA_SHARED_READS();
  volatile uint64_t *header = (volatile uint64_t *)(uintptr_t)(address - 64u);
  if (header[0] != BOSC_AME_PRIVATE_DDR_MAGIC || header[1] != address - 64u ||
      header[2] != bytes || header[3] != owner ||
      header[5] != (BOSC_AME_PRIVATE_DDR_GUARD ^ header[1] ^ header[2] ^
                    header[3] ^ header[4]))
    bosc_ame_private_ddr_fail(BOSC_AME_PRIVATE_DDR_FAIL_GUARD, address, bytes);
  for (unsigned i = 6; i != 8; ++i)
    if (header[i] != BOSC_AME_PRIVATE_DDR_GUARD)
      bosc_ame_private_ddr_fail(BOSC_AME_PRIVATE_DDR_FAIL_GUARD, address,
                                bytes);
  volatile uint64_t *tail = (volatile uint64_t *)(uintptr_t)(address + bytes);
  for (unsigned i = 0; i != 8; ++i)
    if (tail[i] != BOSC_AME_PRIVATE_DDR_GUARD)
      bosc_ame_private_ddr_fail(BOSC_AME_PRIVATE_DDR_FAIL_GUARD, address,
                                bytes);
  uintptr_t parent = header[4];
  // Finish all guard/parent loads before publishing allocator state. Stores
  // must use the write policy, just as acquire's metadata stores do.
  BOSC_AME_DMA_SHARED_WRITES();
  bosc_ame_private_ddr_top = address - 64u;
  bosc_ame_private_ddr_active = parent;
  ++bosc_ame_private_ddr_releases;
  header[0] = 0u;
  if (!parent) {
    bosc_ame_private_ddr_top = 0u;
#if BOSC_AME_PRIVATE_DDR_NR_SINGLE_E6
    bosc_ame_private_ddr_owner = 0u;
#else
    __atomic_store_n(&bosc_ame_private_ddr_owner, 0u, __ATOMIC_RELEASE);
#endif
  }
#if BOSC_AME_PRIVATE_DDR_TRACE
  bosc_ame_private_ddr_trace_record(2u, bytes, address, top_before,
                                    active_before);
#endif
  bosc_ame_private_ddr_leave();
}
