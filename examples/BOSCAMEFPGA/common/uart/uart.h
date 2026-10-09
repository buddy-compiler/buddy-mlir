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
//
// uart.c implements init_uart / print_uart / print_uart_int / print_uart_addr
// for hello (common/runtime). The MMIO helpers below are inline so the NR
// NH/RA runtime (common/nr) can program the same device without linking a
// second copy of uart.c. Only NH touches UART MMIO; RA forwards text through
// a DDR ring and calls these symbols only via NH.
//
// Hardware contract (DW APB UART on this board):
//   - Base address 0x310b0000
//   - TX data register at +0x20 (UART_THR), not the classic 16550 THR at +0x00
//   - Fixed baud divisor 8 (14.7456 MHz / 115200), 8N1
//   - init_uart(freq, baud) keeps the classical API but ignores both arguments
//
//===----------------------------------------------------------------------===//

#ifndef __UART_H
#define __UART_H

#include <stdint.h>

// Called from _init() in bare_runtime.c (hello) or nr_nh_main() (operators)
// before any console output. freq and baud are API compatibility stubs;
// uart_init() always programs the fixed board divisor.
void init_uart(uint32_t freq, uint32_t baud);

// Write a NUL-terminated string. Does not append "\r\n"; callers supply it.
void print_uart(const char *text);

// Print value as exactly 8 uppercase hex digits (MSB nibble first).
// Despite the name, this is hexadecimal, not decimal.
void print_uart_int(uint32_t value);

// Print a 64-bit value as 16 hex digits: high 32 bits, then low 32 bits.
// Used by trap handlers for mcause / mepc / mtval.
void print_uart_addr(uint64_t value);

// Physical MMIO base of the NR UART on this FPGA image.
#define UART_BASE 0x310b0000UL
#define UART_REG(offset) (*(volatile uint32_t *)(UART_BASE + (offset)))

// Register aliases. Several offsets share different meanings depending on DLAB
// (LCR bit 7): when DLAB=0, +0x00 is RBR/THR and +0x04 is IER; when DLAB=1,
// those same addresses are DLL and DLH.
#define UART_RBR_THR_DLL UART_REG(0x00) // RBR/THR (DLAB=0) or DLL (DLAB=1)
#define UART_IER_DLH UART_REG(0x04)     // IER (DLAB=0) or DLH (DLAB=1)
#define UART_FCR UART_REG(0x08)         // FIFO control
#define UART_LCR UART_REG(0x0c)         // Line control (DLAB, word length, ...)
#define UART_MCR UART_REG(0x10)         // Modem control (DTR/RTS)
#define UART_LSR UART_REG(0x14)         // Line status (TX empty, RX ready, ...)
#define UART_USR UART_REG(0x7c)         // DW APB UART status (Busy bit 0)
#define UART_THR UART_REG(0x20)         // Board-specific TX data register

// Program 8N1 and fixed divisor 8. Safe to call once at NH boot.
static inline void uart_init(void) {
  // Interrupts off. DLAB is still 0, so +0x04 is IER.
  UART_IER_DLH = 0;

  // DLAB=1 selects divisor latches. Low bits 11b select 8-bit data length.
  UART_LCR = 0x83;

  // USR bit 0 is UART Busy. Wait until the line is idle before touching
  // DLL/DLH.
  while (UART_USR & 0x01)
    ;

  // DLAB=1: +0x04 is DLH, +0x00 is DLL. Divisor 0x0008 => 115200 @ 14.7456 MHz.
  UART_IER_DLH = 0;
  UART_RBR_THR_DLL = 8;

  // DLAB=0 again: 8 data bits, no parity, 1 stop bit (8N1).
  UART_LCR = 0x03;

  // Enable FIFOs (bit 0). Do not set flush bits.
  UART_FCR = 0x01;

  // Assert DTR (bit 0) and RTS (bit 1) so the peer sees the port ready.
  UART_MCR = 0x03;
}

// Spin until LSR bit 5 (THR empty), then write one byte to UART_THR.
static inline void uart_putc(char c) {
  while ((UART_LSR & 0x20) == 0)
    ;
  UART_THR = c;
}

// Non-zero when LSR bit 0 reports a received byte waiting in RBR.
static inline int uart_rx_ready(void) { return (UART_LSR & 0x01) != 0; }

// Block until a byte is available, then read RBR (+0x00 with DLAB=0).
// Used by NH to forward host input into the NR RX ring for RA's nr_getchar().
static inline char uart_getc(void) {
  while (!uart_rx_ready())
    ;
  return (char)(UART_RBR_THR_DLL & 0xffu);
}

#endif // __UART_H
