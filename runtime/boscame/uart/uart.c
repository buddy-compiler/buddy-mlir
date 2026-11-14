//===- uart.c - NR UART driver implementation -----------------------------===//
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

#include "uart.h"

void init_uart(uint32_t freq, uint32_t baud) {
  // Keep the NR word-wide MMIO map and use the upstream init contract.
  uart_configure(uart_divisor_for(freq, baud));
}

void write_serial(char value) { UART_PUTC(value); }

void print_uart(const char *text) {
  while (*text != '\0')
    UART_PUTC(*text++);
}

void print_uart_int(uint32_t value) {
  static const char digits[] = "0123456789ABCDEF";
  for (int shift = 28; shift >= 0; shift -= 4)
    UART_PUTC(digits[(value >> shift) & 0xfu]);
}

void print_uart_addr(uint64_t value) {
  print_uart_int((uint32_t)(value >> 32));
  print_uart_int((uint32_t)value);
}

void print_uart_hex32(uint32_t value) {
  static const char digits[] = "0123456789abcdef";
  for (int shift = 28; shift >= 0; shift -= 4)
    UART_PUTC(digits[(value >> shift) & 0xfu]);
}

void print_uart_hex64(uint64_t value) {
  print_uart_hex32((uint32_t)(value >> 32));
  print_uart_hex32((uint32_t)value);
}

void print_uart_dec32(uint32_t value) {
  char digits[10];
  int count = 0;

  if (value == 0u) {
    UART_PUTC('0');
    return;
  }
  while (value != 0u) {
    digits[count++] = (char)('0' + (value % 10u));
    value /= 10u;
  }
  while (count != 0)
    UART_PUTC(digits[--count]);
}
