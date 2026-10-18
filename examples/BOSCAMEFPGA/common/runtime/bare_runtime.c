//===- bare_runtime.c - Bare-metal C runtime for NR FPGA ------------------===//
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
// Bare-metal C runtime for the NH core on the NR FPGA platform.
// Provides _init (UART bring-up, lifecycle hooks around main, then wfi)
// and the default machine-mode trap handler used by the hello example.
//
//===----------------------------------------------------------------------===//

#include "bare_runtime.h"
#include "uart.h"

#include <stdint.h>

// Default NR UART clock (14.7456 MHz) and baud (115200).
// These match the board's fixed divisor of 8 in uart.c
// (freq / baud == 8). Override with -DUART_FREQ_HZ / -DUART_BAUD
// for documentation or future divisor programming; today's
// init_uart() still ignores both arguments for API compatibility.
#ifndef UART_FREQ_HZ
#define UART_FREQ_HZ 14745600u
#endif

#ifndef UART_BAUD
#define UART_BAUD 115200u
#endif

// Default weak no-op implementations of the bare-runtime lifecycle hooks.
// _init() calls them in order: print_banner -> before_main -> main ->
// after_main. An application may provide strong definitions of the same
// symbols (see hello.c) to print a banner or log around main(); if none
// are linked, these empty stubs keep the CRT path valid with no extra I/O.
__attribute__((weak)) void bare_runtime_print_banner(void) {}

__attribute__((weak)) void bare_runtime_before_main(void) {}

__attribute__((weak)) void bare_runtime_after_main(void) {}

// Default machine-mode trap handler, called from crt_uart.S trap_entry.
//
// Calling convention (set up by the assembly stub):
//   mcause - exception/interrupt cause (CSR mcause -> a0)
//   mepc   - faulting PC (CSR mepc -> a1)
//   regs   - pointer to the stacked x1-x31 frame on the trap stack (a2);
//            unused here, but available for a recovering override
//
// The CRT expects the return value to be written back into mepc before
// mret. This default implementation never returns: it dumps mcause/mepc/
// mtval over UART and parks the hart in wfi. Marked weak so an app can
// supply a strong handle_trap that resumes (return a new mepc) instead.
__attribute__((weak)) uintptr_t handle_trap(uintptr_t mcause, uintptr_t mepc,
                                            uintptr_t regs) {
  (void)regs;

  print_uart("\r\nTRAP mcause=0x");
  print_uart_addr((uint64_t)mcause);
  print_uart(" mepc=0x");
  print_uart_addr((uint64_t)mepc);
  uintptr_t mtval;
  __asm__ volatile("csrr %0, mtval" : "=r"(mtval));
  print_uart(" mtval=0x");
  print_uart_addr((uint64_t)mtval);
  print_uart("\r\n");

  // Fatal path: there is no OS/scheduler to resume into after a trap dump.
  // Spin in wfi so the hart stays idle and UART output is not flooded.
  while (1)
    __asm__ volatile("wfi");
}

// C entry point jumped to by crt_uart.S after BSS/TBSS clear and stack/gp
// setup. Initializes UART, runs the weak lifecycle hooks around main(), then
// parks the hart in wfi. Never returns: bare metal has no host process to
// exit back to, and returning would fall off into undefined code.
void _init(void) {
  init_uart(UART_FREQ_HZ, UART_BAUD);

  bare_runtime_print_banner();

  extern int main(void);
  bare_runtime_before_main();
  main();
  bare_runtime_after_main();

  while (1)
    __asm__ volatile("wfi");
}
