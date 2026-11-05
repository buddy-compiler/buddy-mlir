//===- nh_e6_platform.c - NH/E6 shared platform bring-up and report hooks -===//
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
// NH/E6 platform implementation.  See nh_e6_platform.h for the hardware
// contract and the application interface.
//
// The MMIO accessors, the CSR writes, the DDR staging window and the order
// barriers are overridable macros, the same pattern as bosc_ame_dma.c: a host
// test can predefine them (before including this file) to drive an in-memory
// fake register file, so the exact flow that runs on the board -- register
// offsets, PMP/satp setup order, CSR 0xbc0 policy values, write order, verdict
// mapping and message text -- is exercised on the host.  The UART is the
// common/uart driver; only the NH flow calls it.
//
// This file is intentionally free of application knowledge: it never mentions
// matrix sizes or application diagnostic fields.  The application supplies
// `e6_main()`, `nh_e6_app_tag()` and `nh_e6_app_report()`.
//

#include "nh_e6_platform.h"

#include "bosc_ame_dma.h"
#include "uart.h"

// ------------------------------------------------------------------
// Overridable MMIO and order-barrier hooks
// ------------------------------------------------------------------

#ifndef NH_E6_MMIO_READ32
#define NH_E6_MMIO_READ32(address) (*(volatile uint32_t *)(address))
#endif

#ifndef NH_E6_MMIO_WRITE32
#define NH_E6_MMIO_WRITE32(address, value)                                     \
  (*(volatile uint32_t *)(address) = (uint32_t)(value))
#endif

// Poll ordering load barrier (the launcher used `fence r, r`).
#ifndef NH_E6_LOAD_FENCE
#if defined(__riscv)
#define NH_E6_LOAD_FENCE() __asm__ volatile("fence r, r" ::: "memory")
#else
#define NH_E6_LOAD_FENCE() __atomic_thread_fence(__ATOMIC_SEQ_CST)
#endif
#endif

// Order barrier after a DDR diagnostic store (the launcher used `fence rw,
// rw`).
#ifndef NH_E6_STORE_FENCE
#if defined(__riscv)
#define NH_E6_STORE_FENCE() __asm__ volatile("fence rw, rw" ::: "memory")
#else
#define NH_E6_STORE_FENCE() __atomic_thread_fence(__ATOMIC_SEQ_CST)
#endif
#endif

// Order barrier after a SYSCTRL/mailbox publish (`fence iorw, iorw`).
#ifndef NH_E6_PUBLISH_FENCE
#if defined(__riscv)
#define NH_E6_PUBLISH_FENCE() __asm__ volatile("fence iorw, iorw" ::: "memory")
#else
#define NH_E6_PUBLISH_FENCE() __atomic_thread_fence(__ATOMIC_SEQ_CST)
#endif
#endif

//
// One `csrw <csr>, rs`, where <csr> is a compile-time CSR name passed as a
// string literal (`NH_E6_CSR_WRITE("pmpaddr0", value)`).  A host test replaces
// this with a logger; there is no meaningful host fallback, so an unoverridden
// non-RISC-V build stops at compile time instead of skipping the setup.
//
#ifndef NH_E6_CSR_WRITE
#if defined(__riscv)
#define NH_E6_CSR_WRITE(csr, value)                                            \
  __asm__ volatile("csrw " csr ", %0" ::"r"(value) : "memory")
#else
#error                                                                         \
    "NH_E6_CSR_WRITE must be defined on non-RISC-V builds (see test_nh_e6_platform.c)"
#endif
#endif

//
// `csrw satp, zero` plus the `sfence.vma` that must follow the bare-mode
// switch; kept as one step so the flush can not be separated from the write.
//
#ifndef NH_E6_SATP_WRITE_BARE
#if defined(__riscv)
#define NH_E6_SATP_WRITE_BARE()                                                \
  __asm__ volatile("csrw satp, zero\n\tsfence.vma zero, zero" ::: "memory")
#else
#error                                                                         \
    "NH_E6_SATP_WRITE_BARE must be defined on non-RISC-V builds (see test_nh_e6_platform.c)"
#endif
#endif

//
// DDR staging window of the PMP setup.  The example linker script exports the
// two boundaries as absolute addresses; `&symbol` is the value of an absolute
// linker symbol, hence the `extern char []` idiom.  A host test overrides both
// with the numbers it wants to assert.
//
#ifndef NH_E6_DDR_WINDOW_START
extern char __nh_e6_ddr_window_start[];
#define NH_E6_DDR_WINDOW_START ((uintptr_t)(void *)__nh_e6_ddr_window_start)
#endif

#ifndef NH_E6_DDR_WINDOW_END
extern char __nh_e6_ddr_window_end[];
#define NH_E6_DDR_WINDOW_END ((uintptr_t)(void *)__nh_e6_ddr_window_end)
#endif

static uint32_t nh_e6_read32(uintptr_t address) {
  return NH_E6_MMIO_READ32(address);
}

static void nh_e6_write32(uintptr_t address, uint32_t value) {
  NH_E6_MMIO_WRITE32(address, value);
}

// ------------------------------------------------------------------
// Diagnostic cells
// ------------------------------------------------------------------

// Trap context: written by the boot module's E6 trap entry, read by NH.
volatile uint64_t e6_diag_mcause;
volatile uint64_t e6_diag_mepc;
volatile uint64_t e6_diag_mtval;

// DMA failure context: written before the failure trap, read by NH.
volatile uint32_t e6_diag_dma_channel;
volatile uint32_t e6_diag_dma_reason;
volatile uint64_t e6_diag_dma_status;
volatile uint64_t e6_diag_dma_src_token;

// Last progress marker published by E6.
volatile uint32_t e6_diag_state;

// ------------------------------------------------------------------
// Report lines
// ------------------------------------------------------------------

//
// `[5.5296MHz]` is the board clock annotation of the validated image; the UART
// divisor itself is derived by the common driver from
// UART_DEFAULT_FREQ_HZ/UART_DEFAULT_BAUD when the NH flow initializes it.
//
void nh_e6_print_tag_prefix(void) {
  print_uart("[");
  print_uart(nh_e6_app_tag());
  print_uart("] ");
}

void nh_e6_print_mhz_prefix(void) {
  print_uart("[5.5296MHz]");
  nh_e6_print_tag_prefix();
}

static void nh_e6_print_progress(uint32_t marker) {
  nh_e6_print_mhz_prefix();
  print_uart("E6 progress=0x");
  print_uart_hex32(marker);
  print_uart("\r\n");
}

static void nh_e6_print_trap_context(void) {
  nh_e6_print_tag_prefix();
  print_uart("trap mcause=0x");
  print_uart_hex64(e6_diag_mcause);
  print_uart(" mepc=0x");
  print_uart_hex64(e6_diag_mepc);
  print_uart(" mtval=0x");
  print_uart_hex64(e6_diag_mtval);
  print_uart("\r\n");
}

static void nh_e6_print_dma_failure(void) {
  nh_e6_print_tag_prefix();
  print_uart("DMA failure channel=");
  print_uart_dec32(e6_diag_dma_channel);
  print_uart(" reason=");
  print_uart_dec32(e6_diag_dma_reason);
  print_uart(" status=0x");
  print_uart_hex64(e6_diag_dma_status);
  print_uart(" src_token=0x");
  print_uart_hex64(e6_diag_dma_src_token);
  print_uart("\r\n");
}

// ------------------------------------------------------------------
// Progress and DMA failure publication (E6 side)
// ------------------------------------------------------------------

void nh_e6_progress(uint32_t marker) {
  nh_e6_write32(NH_E6_MAILBOX_REG, marker);
  NH_E6_PUBLISH_FENCE();
  e6_diag_state = marker;
  NH_E6_STORE_FENCE();
}

__attribute__((noreturn)) void nh_e6_publish_dma_failure(uint32_t channel,
                                                         uint32_t reason,
                                                         uint64_t status,
                                                         uint64_t src_token) {
  //
  // Publish the reason last: NH treats a non-zero reason as "DMA context
  // present", so the other cells must already be visible.
  //
  e6_diag_dma_channel = channel;
  e6_diag_dma_status = status;
  e6_diag_dma_src_token = src_token;
  e6_diag_dma_reason = reason;
  NH_E6_STORE_FENCE();
  __builtin_trap();
}

// ------------------------------------------------------------------
// E6 address-space setup and CSR 0xbc0 cache policies
// ------------------------------------------------------------------

//
// Seven TOR PMP entries from the delivered reference's configure_pmp(), with
// the launcher's bare-mode switch appended.  Entry 3 already covers both DMA
// channel register files because of BOSC_AME_DMA_CHANNEL_STRIDE, and pmpaddr4/5
// bound entry 5, the DDR staging window exported by the example linker script.
// The order of the CSR writes, the `fence iorw, iorw` before the satp write and
// the `sfence.vma` after it are the board-tested sequence; do not reorder.
//
void nh_e6_platform_init(void) {
  NH_E6_CSR_WRITE("pmpaddr0", BOSC_AME_SPM0_BASE >> 2);
  NH_E6_CSR_WRITE("pmpaddr1",
                  (BOSC_AME_SPM1_BASE + BOSC_AME_SPM_BANK_BYTES) >> 2);
  NH_E6_CSR_WRITE("pmpaddr2", BOSC_AME_DMA0_BASE >> 2);
  NH_E6_CSR_WRITE("pmpaddr3",
                  (BOSC_AME_DMA1_BASE + BOSC_AME_DMA_CHANNEL_STRIDE) >> 2);
  NH_E6_CSR_WRITE("pmpaddr4", NH_E6_DDR_WINDOW_START >> 2);
  NH_E6_CSR_WRITE("pmpaddr5", NH_E6_DDR_WINDOW_END >> 2);
  // End of the last TOR entry: the rest of the 56-bit physical space.
  NH_E6_CSR_WRITE("pmpaddr6", UINT64_C(0x003fffffffffffff));
  // TOR addressing, R/W/X for all seven entries (the image and stacks fall in
  // entry 4, [0x60020000, 0x80100000)).
  NH_E6_CSR_WRITE("pmpcfg0", UINT64_C(0x000f0f0f0f0f0f0f));
  NH_E6_PUBLISH_FENCE();

  // Bare mode: the example hands physical addresses to DMA and AME.
  NH_E6_SATP_WRITE_BARE();
}

//
// Policy switches.  The reference drains the preceding accesses with a fence
// before the `csrw 0xbc0` (its shared_writes()/shared_reads()); the
// board-tested launcher had the same barrier before its uncached-load switch.
// Keeping both writes here means the two values and their order live in one
// place.
//
void nh_e6_cache_writes(void) {
  NH_E6_PUBLISH_FENCE();
  NH_E6_CSR_WRITE("0xbc0", UINT64_C(0x0f0f0f0f0f0f0f0f));
}

void nh_e6_cache_uncached_loads(void) {
  NH_E6_PUBLISH_FENCE();
  NH_E6_CSR_WRITE("0xbc0", UINT64_C(0x0f0f000f000f000f));
}

// ------------------------------------------------------------------
// E6 completion flow
// ------------------------------------------------------------------

void nh_e6_e6_main(void) {
  uint32_t result;
  int status;

  // Address-space setup first: the CSR 0xbc0 policies and every DMA/AME
  // access in e6_main() depend on these PMP regions.
  nh_e6_platform_init();

  status = e6_main();

  // e6_main() returning 0 is the application-level PASS.
  nh_e6_write32(NH_E6_DONE_REG,
                status == 0 ? NH_E6_DONE_PASS : NH_E6_DONE_FAIL);
  NH_E6_PUBLISH_FENCE();

  // NH owns the verdict; relay it to the VCS mailbox once it is published.
  do {
    NH_E6_LOAD_FENCE();
    result = nh_e6_read32(NH_E6_RESULT_REG);
  } while (result == 0u);
  nh_e6_write32(NH_E6_MAILBOX_REG, result);
  NH_E6_PUBLISH_FENCE();
}

// ------------------------------------------------------------------
// NH flow
// ------------------------------------------------------------------

static int nh_e6_done_is_terminal(uint32_t done) {
  return done == NH_E6_DONE_PASS || done == NH_E6_DONE_FAIL ||
         done == NH_E6_DONE_TRAP;
}

void nh_e6_nh_main(void) {
  uint32_t done = 0;
  uint32_t timed_out = 0;
  uint32_t last_progress = 0;
  uint32_t remaining = NH_E6_WAIT_ITERATIONS;
  uint32_t result;
  const char *verdict;

  init_uart(UART_DEFAULT_FREQ_HZ, UART_DEFAULT_BAUD);
  NH_E6_PUBLISH_FENCE();
  nh_e6_print_mhz_prefix();
  print_uart("start\r\n");

  // Clear any stale completion state left by a previous image.
  nh_e6_write32(NH_E6_DONE_REG, 0);
  nh_e6_write32(NH_E6_MAILBOX_REG, 0);
  nh_e6_write32(NH_E6_RESULT_REG, 0);
  NH_E6_PUBLISH_FENCE();

  // Program and release E6 through RA SYSCTRL.
  nh_e6_write32(NH_E6_E6_PC_REG, (uint32_t)(uintptr_t)e6_start);
  nh_e6_write32(NH_E6_E6_CONTROL_REG, 0);
  nh_e6_write32(NH_E6_E6_RELEASE_REG, NH_E6_E6_RELEASE_VALUE);
  nh_e6_write32(NH_E6_E6_CONTROL_REG, NH_E6_E6_RELEASE_VALUE);
  NH_E6_PUBLISH_FENCE();

  // Bounded wait for the terminal code; report each new progress marker.
  for (;;) {
    uint32_t marker;

    NH_E6_LOAD_FENCE();
    done = nh_e6_read32(NH_E6_DONE_REG);
    marker = nh_e6_read32(NH_E6_MAILBOX_REG);
    if (nh_e6_done_is_terminal(done))
      break;
    if (marker != 0u && marker != last_progress) {
      last_progress = marker;
      nh_e6_print_progress(marker);
    }
    if (remaining == 0u) {
      timed_out = 1u;
      break;
    }
    remaining--;
  }

  if (timed_out) {
    result = NH_E6_RESULT_FAIL;
    verdict = "TIMEOUT\r\n";
  } else if (done == NH_E6_DONE_PASS) {
    result = NH_E6_RESULT_PASS;
    verdict = "PASS\r\n";
  } else if (done == NH_E6_DONE_TRAP) {
    result = NH_E6_RESULT_FAIL;
    verdict = "E6 TRAP\r\n";
  } else {
    result = NH_E6_RESULT_FAIL;
    verdict = "FAIL\r\n";
  }

  nh_e6_write32(NH_E6_RESULT_REG, result);
  NH_E6_PUBLISH_FENCE();
  nh_e6_print_mhz_prefix();
  print_uart(verdict);

  //
  // The trap context and the DMA failure context belong to the TRAP verdict;
  // the DMA line appears only when the failure hook was actually reached.
  //
  if (done == NH_E6_DONE_TRAP) {
    nh_e6_print_trap_context();
    if (e6_diag_dma_reason != 0u)
      nh_e6_print_dma_failure();
  }

  // Application diagnostics, on every outcome.
  nh_e6_app_report();

  // Keep the VCS completion protocol in the same ELF as FPGA UART reporting.
  nh_e6_write32(NH_E6_MAILBOX_REG, result);
  NH_E6_PUBLISH_FENCE();
}

void nh_e6_nh_trap(void) {
  nh_e6_write32(NH_E6_RESULT_REG, NH_E6_RESULT_FAIL);
  NH_E6_PUBLISH_FENCE();
  nh_e6_print_mhz_prefix();
  print_uart("NH TRAP\r\n");
  nh_e6_write32(NH_E6_MAILBOX_REG, NH_E6_RESULT_FAIL);
  NH_E6_PUBLISH_FENCE();
}
