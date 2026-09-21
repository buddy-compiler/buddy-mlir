//===- uart.h - NR UART public interface ----------------------------------===//
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
// Public UART API for the NH bare-metal runtime on NR FPGA.
// Implemented in uart.c: MMIO base 0x310b0000, TX at offset 0x20,
// fixed divisor 8 (14.7456 MHz / 115200), 8N1.
// bare_runtime.c calls init_uart() from _init(); print_* are used by
// the runtime trap dump and by applications such as hello.c.
//
//===----------------------------------------------------------------------===//

#ifndef EXAMPLES_BOSCAMEFPGA_COMMON_UART_UART_H
#define EXAMPLES_BOSCAMEFPGA_COMMON_UART_UART_H

#include <stdint.h>

// Called from _init() in bare_runtime.c before any print_*.
// freq and baud are kept for API compatibility; uart.c ignores them and
// programs the fixed divisor instead.
void init_uart(uint32_t freq, uint32_t baud);

// Write a NUL-terminated string. Does not append "\r\n"; callers supply it.
void print_uart(const char *text);

// Print value as exactly 8 uppercase hex digits, most significant nibble first.
// Despite the name, this is hexadecimal, not decimal.
void print_uart_int(uint32_t value);

// Print a 64-bit address as 16 hex digits: high 32 bits, then low 32 bits.
// Used by handle_trap() in bare_runtime.c for mcause / mepc / mtval.
void print_uart_addr(uint64_t value);

#endif // EXAMPLES_BOSCAMEFPGA_COMMON_UART_UART_H
