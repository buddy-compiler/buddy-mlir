//===- uart.h - NR UART driver public interface ---------------------------===//
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
// DW APB UART helpers for Buddy bare-metal / NR / NH-E6 consoles.
//
// Board default base is 0x310b0000; host tests may predefine UART_BASE as an
// in-memory fake register file (same pattern as BOSC_AME_DMAx_BASE).  Character
// output goes through UART_PUTC so tests can capture the exact byte stream.
//
//===----------------------------------------------------------------------===//

#ifndef __UART_H
#define __UART_H

#include <stdint.h>

// ---- public print ABI (uart.c) ----

// Program divisor/FIFO for `freq` Hz clock and `baud` rate (8N1).
void init_uart(uint32_t freq, uint32_t baud);

// Blocking single-character TX via UART_PUTC.
void write_serial(char value);

// NUL-terminated string to the UART.
void print_uart(const char *text);

// Uppercase hex dump of a 32-bit value (legacy convention).
void print_uart_int(uint32_t value);

// Uppercase hex dump of a 64-bit address / value.
void print_uart_addr(uint64_t value);

// Lowercase fixed-width hex and unsigned decimal text used by NH/E6 reports.
// print_uart_int keeps the upstream uppercase convention.
void print_uart_hex32(uint32_t value);
void print_uart_hex64(uint64_t value);
void print_uart_dec32(uint32_t value);

#ifndef UART_DEFAULT_FREQ_HZ
#ifdef UART_FREQ_HZ
#define UART_DEFAULT_FREQ_HZ UART_FREQ_HZ
#else
#define UART_DEFAULT_FREQ_HZ 5529600u
#endif
#endif
#ifndef UART_DEFAULT_BAUD
#ifdef UART_BAUD
#define UART_DEFAULT_BAUD UART_BAUD
#else
#define UART_DEFAULT_BAUD 115200u
#endif
#endif

//
// The board value is fixed, but a host test may predefine UART_BASE as an
// in-memory fake register file, the same way the DMA helper lets tests
// override BOSC_AME_DMAx_BASE.
//
#ifndef UART_BASE
#define UART_BASE 0x310b0000UL
#endif
#define UART_REG(offset) (*(volatile uint32_t *)(UART_BASE + (offset)))

#define UART_RBR_THR_DLL UART_REG(0x00)
#define UART_IER_DLH UART_REG(0x04)
#define UART_FCR UART_REG(0x08)
#define UART_LCR UART_REG(0x0c)
#define UART_MCR UART_REG(0x10)
#define UART_LSR UART_REG(0x14)
#define UART_USR UART_REG(0x7c)
#define UART_THR UART_REG(0x20)

// Compute the 16-bit divisor latch for `freq` / `baud` (rounded, clamped).
static __attribute__((unused)) uint32_t uart_divisor_for(uint32_t freq,
                                                         uint32_t baud) {
  const uint64_t denominator =
      (uint64_t)(baud ? baud : UART_DEFAULT_BAUD) * 16u;
  uint64_t divisor;

  if (freq == 0u)
    freq = UART_DEFAULT_FREQ_HZ;
  divisor = ((uint64_t)freq + denominator / 2u) / denominator;
  if (divisor < 1u)
    divisor = 1u;
  if (divisor > 0xffffu)
    divisor = 0xffffu;
  return (uint32_t)divisor;
}

// Program IER/LCR/DLL/DLH/FCR/MCR for 8N1 with FIFO enabled.
static __attribute__((unused)) void uart_configure(uint32_t divisor) {
  // Disable UART interrupts while DLAB=0 (offset 0x04 is IER).
  UART_IER_DLH = 0;

  // DLAB=1 to unlock the divisor latch; low two bits set 8-bit data.
  UART_LCR = 0x83;

  // Wait until DW APB UART USR bit0 (UART Busy) clears.
  while (UART_USR & 0x01)
    ;

  // With DLAB=1, offset 0x04 is DLH and 0x00 is DLL.
  UART_IER_DLH = (divisor >> 8) & 0xffu;
  UART_RBR_THR_DLL = divisor & 0xffu;

  // DLAB=0, 8 data bits, no parity, 1 stop bit (8N1).
  UART_LCR = 0x03;

  // Enable FIFO without flushing RX/TX contents.
  UART_FCR = 0x01;

  // Assert DTR and RTS.
  UART_MCR = 0x03;
}

// Configure the UART with board-default frequency and baud.
static __attribute__((unused)) void uart_init(void) {
  uart_configure(uart_divisor_for(UART_DEFAULT_FREQ_HZ, UART_DEFAULT_BAUD));
}

//
// Character sink hook.  Defaults to the real MMIO path; a host test can
// predefine UART_PUTC(c) before including uart.h/uart.c to capture the exact
// byte stream that the print_uart* helpers produce.
//
#ifndef UART_PUTC
static void uart_putc(char c) {
  while ((UART_LSR & 0x20) == 0)
    ;
  UART_THR = c;
}
#define UART_PUTC(c) uart_putc(c)
#endif

// Non-zero when the RBR holds a received byte.
static __attribute__((unused)) int uart_rx_ready(void) {
  return (UART_LSR & 0x01) != 0;
}

// Blocking RX of one byte from RBR.
static __attribute__((unused)) char uart_getc(void) {
  while (!uart_rx_ready())
    ;
  return (char)(UART_RBR_THR_DLL & 0xffu);
}

// Blocking TX of a NUL-terminated C string.
static __attribute__((unused)) void uart_puts(char *s) {
  while (*s != '\0') {
    UART_PUTC(*(s++));
  }
}

#endif // __UART_H
