//===- uart.h - NR UART public interface ----------------------------------===//
//
// print_uart / print_uart_int / print_uart_addr used by the bare-metal runtime.
// Adapted from ModelZoo thirdparty/nr UART sources.
//
//===----------------------------------------------------------------------===//

#ifndef EXAMPLES_BOSCAMEFPGA_COMMON_UART_UART_H
#define EXAMPLES_BOSCAMEFPGA_COMMON_UART_UART_H

#include <stdint.h>

void init_uart(uint32_t freq, uint32_t baud);
void print_uart(const char *text);
void print_uart_int(uint32_t value);
void print_uart_addr(uint64_t value);

#endif // EXAMPLES_BOSCAMEFPGA_COMMON_UART_UART_H
