//===- hello.c - NH Hello World for NR FPGA -------------------------------===//
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
// Plain-C Hello World on the NH hart for NR FPGA. No MLIR, no RA, no AME.
// crt_uart.S jumps to _init() in bare_runtime.c, which calls the three
// strong hooks below around main(). UART MMIO is 0x310b0000 (uart.c).
// Load address is 0x80000000 (hello_nr.ld). Success on the serial console
// is the line "verify hello: PASS".
//
//===----------------------------------------------------------------------===//

#include "bare_runtime.h"
#include "uart.h"

#include <stdint.h>

// Fixed check used as a UART PASS/FAIL tag. -O2 constant-folds this to
// `li a0, 42`; the example verifies boot and print, not live arithmetic.
#define LHS 40
#define RHS 2
#define EXPECTED_ANSWER (LHS + RHS)

// Strong override of the weak no-op in bare_runtime.c.
// _init() calls this once after UART init, before before_main()/main().
void bare_runtime_print_banner(void) {
  print_uart("\r\n========================================\r\n");
  print_uart("  BUDDY-MLIR Hello World (plain C)\r\n");
  print_uart("  NR / NH RISC-V @ 0x80000000\r\n");
  print_uart("========================================\r\n\r\n");
}

// _init() calls this immediately before main().
void bare_runtime_before_main(void) { print_uart("rt: call main\r\n"); }

// _init() calls this after main() returns, then parks the hart in wfi.
void bare_runtime_after_main(void) { print_uart("\r\n=== Hello Done ===\r\n"); }

// Application entry from _init(). print_uart_int emits 8 hex digits, so
// 42 prints as 0000002A. Return 0 is ignored; _init() never exits to a host.
int main(void) {
  int answer = LHS + RHS;
  print_uart("Hello, World!\r\n");
  print_uart("computed: 40 + 2 = ");
  print_uart_int((uint32_t)answer);
  print_uart("\r\n");

  if (answer == EXPECTED_ANSWER) {
    print_uart("verify hello: PASS\r\n");
  } else {
    print_uart("verify hello: FAIL expected=");
    print_uart_int(EXPECTED_ANSWER);
    print_uart(" actual=");
    print_uart_int((uint32_t)answer);
    print_uart("\r\n");
  }
  return 0;
}
