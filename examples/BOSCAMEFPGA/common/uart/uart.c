//===- uart.c - NR UART driver (base 0x310b0000) --------------------------===//
//
// 32-bit MMIO, TX at +0x20, fixed divisor 8 (14.7456 MHz / 115200). Adapted
// from ModelZoo thirdparty/nr UART sources.
//
//===----------------------------------------------------------------------===//

#include "uart.h"

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
  /* Interrupts off. DLAB is 0, so +0x04 is IER. */
  UART_IER_DLH = 0;

  /* DLAB=1 to program the divisor latches; low bits select 8-bit data. */
  UART_LCR = 0x83;

  /* DW APB UART USR bit0: UART Busy. */
  while (UART_USR & 0x01)
    ;

  /*
   * DLAB=1:
   *   +0x04 = DLH
   *   +0x00 = DLL
   *
   * Fixed baud-rate divisor 0x0008 (14.7456 MHz / 115200).
   */
  UART_IER_DLH = 0;
  UART_RBR_THR_DLL = 8;

  /* DLAB=0, 8 data bits, no parity, 1 stop bit (8N1). */
  UART_LCR = 0x03;

  /* Enable FIFOs without flushing them. */
  UART_FCR = 0x01;

  /* Assert DTR and RTS. */
  UART_MCR = 0x03;
}

static void uart_putc(char c) {
  while ((UART_LSR & 0x20) == 0)
    ;
  UART_THR = c;
}

void init_uart(uint32_t freq, uint32_t baud) {
  /* NR supplies a fixed divisor in uart_init; its input clock is not given. */
  (void)freq;
  (void)baud;
  uart_init();
}

void print_uart(const char *text) {
  while (*text != '\0')
    uart_putc(*text++);
}

void print_uart_int(uint32_t value) {
  static const char digits[] = "0123456789ABCDEF";
  for (int shift = 28; shift >= 0; shift -= 4)
    uart_putc(digits[(value >> shift) & 0xfu]);
}

void print_uart_addr(uint64_t value) {
  print_uart_int((uint32_t)(value >> 32));
  print_uart_int((uint32_t)value);
}
