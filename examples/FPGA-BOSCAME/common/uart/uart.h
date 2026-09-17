#ifndef __UART_H
#define __UART_H

#include <stdint.h>

/* Buddy bare-metal runtime UART interface. */
void init_uart(uint32_t freq, uint32_t baud);
void write_serial(char value);
void print_uart(const char *text);
void print_uart_int(uint32_t value);
void print_uart_addr(uint64_t value);

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
  /* 关闭串口中断，此时 DLAB=0，所以 +0x04 是 IER。 */
  UART_IER_DLH = 0;

  /*
   * DLAB=1，允许配置除数锁存器。
   * 低两位为 11，配置 8 位数据。
   */
  UART_LCR = 0x83;

  /* DW APB UART USR bit0：UART Busy。 */
  while (UART_USR & 0x01)
    ;

  /*
   * DLAB=1 时：
   *   +0x04 = DLH
   *   +0x00 = DLL
   *
   * 波特率除数为 0x0008。
   */
  UART_IER_DLH = 0;
  UART_RBR_THR_DLL = 8;

  /* DLAB=0，8 数据位、无校验、1 停止位，即 8N1。 */
  UART_LCR = 0x03;

  /* 使能 FIFO，但不主动清空收发 FIFO。 */
  UART_FCR = 0x01;

  /* 置位 DTR 和 RTS。 */
  UART_MCR = 0x03;
}

static void uart_putc(char c) {
  while ((UART_LSR & 0x20) == 0)
    ;
  UART_THR = c;
}

static __attribute__((unused)) int uart_rx_ready(void) { return (UART_LSR & 0x01) != 0; }
static __attribute__((unused)) char uart_getc(void) {
  while (!uart_rx_ready()) ;
  return (char)(UART_RBR_THR_DLL & 0xffu);
}

static void uart_puts(char *s) {
  while (*s != '\0') {
    uart_putc(*(s++));
  }
}

#endif // __UART_H
