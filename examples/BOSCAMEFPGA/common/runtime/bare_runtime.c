//===- bare_runtime.c - Bare-metal C runtime for NR FPGA ------------------===//
//
// Provides _init (UART, banner, main, wfi) and trap handling for the NH
// hello example. Adapted from ModelZoo examples/tools/bare_runtime.c.
//
//===----------------------------------------------------------------------===//

#include "bare_runtime.h"
#include "uart.h"

#include <stdint.h>

#ifndef UART_FREQ_HZ
#define UART_FREQ_HZ 14745600u
#endif

#ifndef UART_BAUD
#define UART_BAUD 115200u
#endif

__attribute__((weak)) void bare_runtime_print_banner(void) {}

__attribute__((weak)) void bare_runtime_before_main(void) {}

__attribute__((weak)) void bare_runtime_after_main(void) {}

// crt_uart.S trap_entry jumps here. mcause/mepc are the trap cause and PC;
// regs is the stacked x1–x31 frame (unused by hello). Print mtval too, then
// halt: this path does not return a new mepc.
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

  while (1)
    __asm__ volatile("wfi");
}

// CRT jumps here after BSS/stack setup. Init UART, run hello's weak hooks
// and main(), then halt: there is no OS to return to.
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
