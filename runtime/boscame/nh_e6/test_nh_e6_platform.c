//===- test_nh_e6_platform.c - Host-side unit tests for NH/E6 platform helpers
//-===//
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
// Host-side test for the NH/E6 platform module.
//
// The test redirects the SYSCTRL/mailbox MMIO to an in-memory fake register
// file, redirects the common/uart register block to a fake and captures every
// character the driver emits, then includes uart.c and nh_e6_platform.c
// directly.  The exact flow that runs on the board is therefore exercised on
// the host: the same register offsets and write order, the same order
// barriers, the same verdict mapping and the same message text.
//
// Covered:
//   - the E6 hardware setup: the seven TOR PMP boundaries (SPM/DMA from
//     bosc_ame_dma.h, DDR from the linker-script window), pmpcfg0, the
//     `fence iorw, iorw` then `csrw satp, zero; sfence.vma` order, and that
//     `nh_e6_e6_main()` runs it before the application's `e6_main()`,
//   - the CSR 0xbc0 cache policies (write-through stores for phase switches,
//     uncached loads before reading DMA-updated data) and their fence,
//   - the NH release sequence through RA SYSCTRL (PC/CTRL/RELEASE/CTRL) with
//     the clear of DONE/MAILBOX/RESULT before it and the barrier placement,
//   - progress reporting from the patch6 mailbox, including the
//     "print only new markers" rule,
//   - the PASS / FAIL / E6 TRAP / TIMEOUT verdict mapping and the final
//     NH_RESULT + mailbox publication,
//   - the trap-CSR line and the DMA failure line (and that the DMA line only
//     appears when the failure hook was reached),
//   - the application report hook on every outcome,
//   - the E6 side: DONE publication, the wait for NH_RESULT and the mailbox
//     relay,
//   - the UART divisor/line-control programming produced by
//     init_uart(UART_DEFAULT_FREQ_HZ, UART_DEFAULT_BAUD),
//   - nh_e6_publish_dma_failure: the diagnostic cells are published before the
//     non-returning trap.
//
// The real MMIO character path (`UART_PUTC` default) is not exercised here;
// the RISC-V build's disassembly shows the LSR wait and THR store.
//
// Build and run (from the repository root):
//   cc -std=c11 -Wall -Wextra -Werror \
//      -I runtime/boscame/runtime \
//      -I runtime/boscame/uart \
//      -I runtime/boscame/nh_e6 \
//      runtime/boscame/nh_e6/test_nh_e6_platform.c \
//      -o /tmp/nh_e6_platform_test && /tmp/nh_e6_platform_test
//

#include <setjmp.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ------------------------------------------------------------------
// Fake UART: the common driver's register file and its byte sink
// ------------------------------------------------------------------

static volatile uint32_t fake_uart_regs[0x100 / 4u];
#define UART_BASE ((uintptr_t)fake_uart_regs)

static char uart_text[8192];
static size_t uart_text_length;

static void test_uart_putc(char byte) {
  if (uart_text_length + 1u < sizeof uart_text) {
    uart_text[uart_text_length++] = byte;
    uart_text[uart_text_length] = '\0';
  }
}

#define UART_PUTC(byte) test_uart_putc((char)(byte))
#include "uart.c"

// ------------------------------------------------------------------
// Fake SYSCTRL register file, event log and scripted E6 device
// ------------------------------------------------------------------

static uint32_t test_mmio_read32(uintptr_t address);
static void test_mmio_write32(uintptr_t address, uint32_t value);
static void test_load_fence(void);
static void test_store_fence(void);
static void test_publish_fence(void);
static void test_csr_write(const char *csr, uint64_t value);
static void test_satp_bare(void);

static volatile uint32_t fake_sysctrl[0x200u / 4u];

#define NH_E6_SYSCTRL_BASE ((uintptr_t)fake_sysctrl)
#define NH_E6_MMIO_READ32(address) test_mmio_read32(address)
#define NH_E6_MMIO_WRITE32(address, value) test_mmio_write32((address), (value))
#define NH_E6_LOAD_FENCE() test_load_fence()
#define NH_E6_STORE_FENCE() test_store_fence()
#define NH_E6_PUBLISH_FENCE() test_publish_fence()
// The host can not execute PMP/satp/sfence instructions: log the CSR name and
// the value that would be written, and the address-space step separately.
#define NH_E6_DDR_WINDOW_START UINT64_C(0x80100000)
#define NH_E6_DDR_WINDOW_END UINT64_C(0x80300000)
#define NH_E6_CSR_WRITE(csr, value) test_csr_write(csr, (uint64_t)(value))
#define NH_E6_SATP_WRITE_BARE() test_satp_bare()

#include "nh_e6_platform.c"

#define LOG_CAPACITY 8192
static char event_log[LOG_CAPACITY];
static size_t event_log_length;
static unsigned long load_fences;
static unsigned long store_fences;
static unsigned long publish_fences;

typedef struct {
  uint32_t done;
  uint32_t mailbox;
} e6_state;

static e6_state script[8];
static unsigned script_length;
static unsigned script_position;

static void log_event(const char *format, ...) {
  va_list args;
  int written;

  if (event_log_length + 1u >= sizeof event_log)
    return;
  va_start(args, format);
  written = vsnprintf(event_log + event_log_length,
                      sizeof event_log - event_log_length, format, args);
  va_end(args);
  if (written > 0) {
    event_log_length += (size_t)written;
    if (event_log_length >= sizeof event_log)
      event_log_length = sizeof event_log - 1u;
  }
}

static const char *register_name(uintptr_t address) {
  switch (address - (uintptr_t)fake_sysctrl) {
  case 0x050u:
    return "RELEASE";
  case 0x060u:
    return "CTRL";
  case 0x100u:
    return "PC";
  case 0x120u:
    return "DONE";
  case 0x140u:
    return "RESULT";
  case 0x160u:
    return "MAILBOX";
  default:
    return "?";
  }
}

// The fake E6 advances one scripted state per poll of the DONE register.
static uint32_t test_mmio_read32(uintptr_t address) {
  if (address - (uintptr_t)fake_sysctrl == 0x120u &&
      script_position < script_length) {
    fake_sysctrl[0x120u / 4u] = script[script_position].done;
    fake_sysctrl[0x160u / 4u] = script[script_position].mailbox;
    ++script_position;
  }
  return fake_sysctrl[(address - (uintptr_t)fake_sysctrl) / 4u];
}

static void test_mmio_write32(uintptr_t address, uint32_t value) {
  log_event("W:%s=0x%x|", register_name(address), value);
  fake_sysctrl[(address - (uintptr_t)fake_sysctrl) / 4u] = value;
}

static void test_load_fence(void) {
  ++load_fences;
  log_event("F:LOAD|");
}

static void test_store_fence(void) {
  ++store_fences;
  log_event("F:STORE|");
}

static void test_publish_fence(void) {
  ++publish_fences;
  log_event("F:PUBLISH|");
}

// CSR 0xbc0 / PMP / pmpcfg0 writes, in issue order.
static void test_csr_write(const char *csr, uint64_t value) {
  log_event("C:%s=0x%llx|", csr, (unsigned long long)value);
}

// `csrw satp, zero` plus the `sfence.vma` that must follow it.
static void test_satp_bare(void) { log_event("C:satp=0|T:SFENCE.VMA|"); }

// ------------------------------------------------------------------
// Application hooks
// ------------------------------------------------------------------

static int test_e6_status;
static unsigned app_report_calls;
static unsigned e6_main_calls;
static int e6_main_saw_platform_init;

static int log_contains(const char *needle) {
  return strstr(event_log, needle) != NULL;
}

int e6_main(void) {
  ++e6_main_calls;
  // The platform must have finished its address-space setup before this hook.
  e6_main_saw_platform_init = log_contains("T:SFENCE.VMA|");
  log_event("A:e6_main|");
  return test_e6_status;
}

const char *nh_e6_app_tag(void) { return "test-app"; }

void nh_e6_app_report(void) {
  ++app_report_calls;
  print_uart("<app-report>");
}

// Placeholder for the E6 entry the boot module defines at 0x80010000.
void e6_start(void) {}

// ------------------------------------------------------------------
// Tiny check harness
// ------------------------------------------------------------------

static int failures;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

static void check_text(const char *expected, const char *label) {
  if (strcmp(uart_text, expected) != 0) {
    fprintf(stderr, "FAIL %s: UART text\n  expected: %s\n  actual:   %s\n",
            label, expected, uart_text);
    ++failures;
  }
}

static void check_log(const char *expected, const char *label) {
  if (strcmp(event_log, expected) != 0) {
    fprintf(stderr,
            "FAIL %s: MMIO/barrier event log\n  expected: %s\n  actual:   %s\n",
            label, expected, event_log);
    ++failures;
  }
}

static jmp_buf test_trap_jump;

static void test_trap_handler(int signum) {
  (void)signum;
  longjmp(test_trap_jump, 1);
}

static void reset_all(void) {
  memset((void *)fake_uart_regs, 0, sizeof fake_uart_regs);
  memset((void *)fake_sysctrl, 0, sizeof fake_sysctrl);
  event_log[0] = '\0';
  event_log_length = 0;
  uart_text[0] = '\0';
  uart_text_length = 0;
  load_fences = 0;
  store_fences = 0;
  publish_fences = 0;
  script_length = 0;
  script_position = 0;
  test_e6_status = 0;
  app_report_calls = 0;
  e6_main_calls = 0;
  e6_main_saw_platform_init = 0;
  e6_diag_mcause = 0;
  e6_diag_mepc = 0;
  e6_diag_mtval = 0;
  e6_diag_dma_channel = 0;
  e6_diag_dma_reason = 0;
  e6_diag_dma_status = 0;
  e6_diag_dma_src_token = 0;
  e6_diag_state = 0;
}

// ------------------------------------------------------------------
// NH flow
// ------------------------------------------------------------------

static void test_nh_pass_with_progress(void) {
  char expected_log[1024];

  reset_all();
  script[0] = (e6_state){0u, 0u};
  script[1] = (e6_state){0u, 0x11u};
  script[2] = (e6_state){0u, 0x30u};
  script[3] = (e6_state){0u, 0x31u};
  script[4] = (e6_state){2u, 0x31u};
  script_length = 5;

  nh_e6_nh_main();

  check_text("[5.5296MHz][test-app] start\r\n"
             "[5.5296MHz][test-app] E6 progress=0x00000011\r\n"
             "[5.5296MHz][test-app] E6 progress=0x00000030\r\n"
             "[5.5296MHz][test-app] E6 progress=0x00000031\r\n"
             "[5.5296MHz][test-app] PASS\r\n"
             "<app-report>",
             "NH PASS");

  snprintf(expected_log, sizeof expected_log,
           "F:PUBLISH|"
           "W:DONE=0x0|W:MAILBOX=0x0|W:RESULT=0x0|F:PUBLISH|"
           "W:PC=0x%x|W:CTRL=0x0|W:RELEASE=0x1|W:CTRL=0x1|F:PUBLISH|"
           "F:LOAD|F:LOAD|F:LOAD|F:LOAD|F:LOAD|"
           "W:RESULT=0xff|F:PUBLISH|W:MAILBOX=0xff|F:PUBLISH|",
           (unsigned)(uintptr_t)e6_start);
  check_log(expected_log, "NH PASS");

  CHECK(app_report_calls == 1u);
  CHECK(load_fences == 5u);
  CHECK(store_fences == 0u);
  CHECK(publish_fences == 5u);

  // init_uart(5529600, 115200) -> divisor 3, 8N1, FIFO, DTR/RTS.
  CHECK(fake_uart_regs[0x00u / 4u] == 3u); // DLL
  CHECK(fake_uart_regs[0x04u / 4u] == 0u); // IER / DLH
  CHECK(fake_uart_regs[0x0cu / 4u] == 3u); // LCR
  CHECK(fake_uart_regs[0x08u / 4u] == 1u); // FCR
  CHECK(fake_uart_regs[0x10u / 4u] == 3u); // MCR
}

static void test_nh_fail(void) {
  char expected_log[1024];

  reset_all();
  script[0] = (e6_state){0xeeu, 0xeeu};
  script_length = 1;

  nh_e6_nh_main();

  check_text("[5.5296MHz][test-app] start\r\n"
             "[5.5296MHz][test-app] FAIL\r\n"
             "<app-report>",
             "NH FAIL");

  snprintf(expected_log, sizeof expected_log,
           "F:PUBLISH|"
           "W:DONE=0x0|W:MAILBOX=0x0|W:RESULT=0x0|F:PUBLISH|"
           "W:PC=0x%x|W:CTRL=0x0|W:RELEASE=0x1|W:CTRL=0x1|F:PUBLISH|"
           "F:LOAD|W:RESULT=0xee|F:PUBLISH|W:MAILBOX=0xee|F:PUBLISH|",
           (unsigned)(uintptr_t)e6_start);
  check_log(expected_log, "NH FAIL");
  CHECK(app_report_calls == 1u);
  CHECK(publish_fences == 5u);
}

static void test_nh_e6_trap_with_dma_context(void) {
  reset_all();
  script[0] = (e6_state){0xedu, 0xedu};
  script_length = 1;
  e6_diag_mcause = 2u;
  e6_diag_mepc = 0x80010008u;
  e6_diag_mtval = 0u;
  e6_diag_dma_channel = 1u;
  e6_diag_dma_reason = 4u;
  e6_diag_dma_status = 2u;
  e6_diag_dma_src_token = UINT64_C(0x80100000);

  nh_e6_nh_main();

  check_text("[5.5296MHz][test-app] start\r\n"
             "[5.5296MHz][test-app] E6 TRAP\r\n"
             "[test-app] trap mcause=0x0000000000000002"
             " mepc=0x0000000080010008 mtval=0x0000000000000000\r\n"
             "[test-app] DMA failure channel=1 reason=4"
             " status=0x0000000000000002 src_token=0x0000000080100000\r\n"
             "<app-report>",
             "NH E6 TRAP with DMA context");
  CHECK(app_report_calls == 1u);
}

static void test_nh_e6_trap_without_dma_context(void) {
  reset_all();
  script[0] = (e6_state){0xedu, 0xedu};
  script_length = 1;
  e6_diag_mcause = 5u;
  e6_diag_mepc = 0x8001099cu;
  e6_diag_mtval = 0x60000000u;

  nh_e6_nh_main();

  check_text("[5.5296MHz][test-app] start\r\n"
             "[5.5296MHz][test-app] E6 TRAP\r\n"
             "[test-app] trap mcause=0x0000000000000005"
             " mepc=0x000000008001099c mtval=0x0000000060000000\r\n"
             "<app-report>",
             "NH E6 TRAP without DMA context");
}

static void test_nh_timeout(void) {
  reset_all();
  script[0] = (e6_state){0u, 0u};
  script_length = 1;

  nh_e6_nh_main();

  check_text("[5.5296MHz][test-app] start\r\n"
             "[5.5296MHz][test-app] TIMEOUT\r\n"
             "<app-report>",
             "NH TIMEOUT");
  CHECK(load_fences == (unsigned long)NH_E6_WAIT_ITERATIONS + 1ul);
  CHECK(fake_sysctrl[0x140u / 4u] == NH_E6_RESULT_FAIL);
  CHECK(fake_sysctrl[0x160u / 4u] == NH_E6_RESULT_FAIL);
  CHECK(app_report_calls == 1u);
}

static void test_nh_trap(void) {
  reset_all();

  nh_e6_nh_trap();

  check_text("[5.5296MHz][test-app] NH TRAP\r\n", "NH TRAP");
  check_log("W:RESULT=0xee|F:PUBLISH|W:MAILBOX=0xee|F:PUBLISH|", "NH TRAP");
  CHECK(publish_fences == 2u);
}

// ------------------------------------------------------------------
// E6 flow
// ------------------------------------------------------------------

// The frozen TOR setup: SPM0/SPM1, the DMA MMIO window, the DDR staging
// window from the linker script, the rest of the space and pmpcfg0, then the
// fence/satp/sfence.vma address-space switch.
#define INIT_LOG                                                               \
  "C:pmpaddr0=0x8000000|C:pmpaddr1=0x8108000|"                                 \
  "C:pmpaddr2=0x18000000|C:pmpaddr3=0x18008000|"                               \
  "C:pmpaddr4=0x20040000|C:pmpaddr5=0x200c0000|"                               \
  "C:pmpaddr6=0x3fffffffffffff|C:pmpcfg0=0xf0f0f0f0f0f0f|"                     \
  "F:PUBLISH|C:satp=0|T:SFENCE.VMA|"

static void test_platform_init(void) {
  reset_all();

  nh_e6_platform_init();

  check_log(INIT_LOG, "platform init");
  CHECK(publish_fences == 1u);
}

static void test_cache_policies(void) {
  reset_all();

  nh_e6_cache_writes();

  check_log("F:PUBLISH|C:0xbc0=0xf0f0f0f0f0f0f0f|", "cache write policy");

  reset_all();

  nh_e6_cache_uncached_loads();

  check_log("F:PUBLISH|C:0xbc0=0xf0f000f000f000f|", "uncached-load policy");
}

static void test_e6_pass_and_fail(void) {
  reset_all();
  fake_sysctrl[0x140u / 4u] = NH_E6_RESULT_PASS;
  test_e6_status = 0;
  nh_e6_e6_main();

  check_text("", "E6 PASS");
  check_log(INIT_LOG "A:e6_main|W:DONE=0x2|F:PUBLISH|"
                     "F:LOAD|W:MAILBOX=0xff|F:PUBLISH|",
            "E6 PASS");
  CHECK(app_report_calls == 0u);
  CHECK(e6_main_calls == 1u);
  CHECK(e6_main_saw_platform_init == 1);

  reset_all();
  fake_sysctrl[0x140u / 4u] = NH_E6_RESULT_FAIL;
  test_e6_status = 1;
  nh_e6_e6_main();

  check_text("", "E6 FAIL");
  check_log(INIT_LOG "A:e6_main|W:DONE=0xee|F:PUBLISH|"
                     "F:LOAD|W:MAILBOX=0xee|F:PUBLISH|",
            "E6 FAIL");
  CHECK(e6_main_saw_platform_init == 1);
}

static void test_progress(void) {
  reset_all();

  nh_e6_progress(0x30u);

  check_log("W:MAILBOX=0x30|F:PUBLISH|F:STORE|", "progress");
  CHECK(e6_diag_state == 0x30u);
}

static void test_dma_failure_publishes_then_traps(void) {
  reset_all();
  signal(SIGILL, test_trap_handler);
  if (setjmp(test_trap_jump) == 0) {
    nh_e6_publish_dma_failure(1u, 4u, UINT64_C(2), UINT64_C(0x80100000));
    fprintf(stderr, "FAIL: nh_e6_publish_dma_failure returned\n");
    ++failures;
  }
  signal(SIGILL, SIG_DFL);

  CHECK(e6_diag_dma_channel == 1u);
  CHECK(e6_diag_dma_reason == 4u);
  CHECK(e6_diag_dma_status == UINT64_C(2));
  CHECK(e6_diag_dma_src_token == UINT64_C(0x80100000));
  check_log("F:STORE|", "DMA failure");
  CHECK(store_fences == 1u);
}

int main(void) {
  test_nh_pass_with_progress();
  test_nh_fail();
  test_nh_e6_trap_with_dma_context();
  test_nh_e6_trap_without_dma_context();
  test_nh_timeout();
  test_nh_trap();
  test_platform_init();
  test_cache_policies();
  test_e6_pass_and_fail();
  test_progress();
  test_dma_failure_publishes_then_traps();

  if (failures != 0) {
    fprintf(stderr, "nh_e6_platform: %d check(s) failed\n", failures);
    return 1;
  }
  printf("nh_e6_platform: all checks passed\n");
  return 0;
}
