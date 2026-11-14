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
// Line format after init is 8N1.
//
//===----------------------------------------------------------------------===//

#include "uart.h"

// DesignWare APB UART window. Offsets +0x00 and +0x04 change meaning when
// LCR.DLAB (bit 7) is set:
//   +0x00  RBR / THR / DLL    receive, transmit, or divisor low
//   +0x04  IER / DLH          interrupt enable, or divisor high
//   +0x08  FCR                FIFO control
//   +0x0c  LCR                line control; bit 7 is DLAB
//   +0x10  MCR                modem control
//   +0x14  LSR                line status; bit 5 is THR empty
//   +0x7c  USR                UART status; bit 0 is busy
//   +0x20  THR                TX register used by uart_putc on this board
#define UART_BASE 0x310b0000UL
#define UART_REG(offset) (*(volatile uint32_t *)(UART_BASE + (offset)))

#define UART_RBR_THR_DLL UART_REG(0x00)
#define UART_IER_DLH UART_REG(0x04)
#define UART_FCR UART_REG(0x08)
#define UART_LCR UART_REG(0x0c)
#define UART_MCR UART_REG(0x10)
#define UART_LSR UART_REG(0x14)
#define UART_USR UART_REG(0x7c)
#define UART_THR UART_REG(0x20)

static void uart_init(void) {
  // Interrupts off. DLAB is still 0, so +0x04 is the interrupt-enable register.
  UART_IER_DLH = 0;

  // DLAB=1 selects the divisor latches. Low bits 11b select 8-bit data.
  UART_LCR = 0x83;

  // USR bit 0 is UART Busy. Wait until the line is idle before touching
  // DLL/DLH.
  while (UART_USR & 0x01)
    ;

  // DLAB=1: +0x04 is DLH, +0x00 is DLL.
  // Fixed divisor 0x0008 matches 14.7456 MHz / 115200.
  UART_IER_DLH = 0;
  UART_RBR_THR_DLL = 8;

  // DLAB=0, 8 data bits, no parity, 1 stop bit (8N1).
  UART_LCR = 0x03;

  // Enable FIFOs. Bit 0 only; do not set the flush bits.
  UART_FCR = 0x01;

  // Assert DTR (bit 0) and RTS (bit 1) so the link partner sees the port ready.
  UART_MCR = 0x03;
}

// Spin until LSR bit 5 (THR empty), then write the byte to the board TX
// register.
static void uart_putc(char c) {
  while ((UART_LSR & 0x20) == 0)
    ;
  UART_THR = c;
}

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
