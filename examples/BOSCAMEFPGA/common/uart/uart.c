//===- uart.c - NR UART driver (base 0x310b0000) --------------------------===//
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
// 32-bit MMIO UART for the NH bare-metal runtime on NR FPGA.
// Base address is 0x310b0000. Transmit data is written at offset 0x20.
// The baud divisor is fixed at 8 (14.7456 MHz / 115200). init_uart()
// keeps freq/baud arguments for API compatibility and ignores them.
// Line format after init is 8N1. Register programming lives in uart.h
// so hello and the NR NH/RA runtime share one implementation.
//
//===----------------------------------------------------------------------===//

#include "uart.h"

// Public entry used by bare_runtime.c::_init. freq and baud are ignored:
// this board programs a fixed divisor inside uart_init().
void init_uart(uint32_t freq, uint32_t baud) {
  (void)freq;
  (void)baud;
  uart_init();
}

// Write a NUL-terminated string. Callers supply "\r\n"; this does not add them.
void print_uart(const char *text) {
  while (*text != '\0')
    uart_putc(*text++);
}

// Print value as exactly 8 uppercase hex digits, most significant nibble first.
void print_uart_int(uint32_t value) {
  static const char digits[] = "0123456789ABCDEF";
  for (int shift = 28; shift >= 0; shift -= 4)
    uart_putc(digits[(value >> shift) & 0xfu]);
}

// Print a 64-bit address as 16 hex digits: high 32 bits, then low 32 bits.
void print_uart_addr(uint64_t value) {
  print_uart_int((uint32_t)(value >> 32));
  print_uart_int((uint32_t)value);
}
