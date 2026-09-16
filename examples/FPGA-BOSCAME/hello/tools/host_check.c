// Host-side output check for the FPGA-BOSCAME/hello example.
//
// The FPGA board is the only place the real UART exists, so this harness
// verifies the one thing that can be checked without it: the exact byte
// sequence hello.c sends to the serial port.
//
// It compiles hello.c for the *host* with -DHELLO_NO_MAIN (so the program's own
// main() is suppressed) and supplies capturing definitions of print_uart /
// print_uart_int in place of the NR UART driver, which waits
// on a real UART status register.  uart.c, crt_uart.S and the RISC-V execution
// path are untouched by this check.
//
// Build and run via:  make check

#include <stdint.h>
#include <stdio.h>
#include <string.h>

// --- capturing replacements for the NR UART driver --------------------------

static char captured[1024];
static size_t captured_len;

void print_uart(const char *str)
{
    while (*str != '\0' && captured_len < sizeof(captured) - 1)
        captured[captured_len++] = *str++;
    captured[captured_len] = '\0';
}

void print_uart_int(uint32_t value)
{
    char buf[9];
    for (int i = 0; i < 8; ++i) {
        uint8_t nibble = (value >> ((7 - i) * 4)) & 0xf;
        buf[i] = (char)(nibble < 10 ? '0' + nibble : 'A' + nibble - 10);
    }
    buf[8] = '\0';
    print_uart(buf);
}

// --- the code under test ----------------------------------------------------

void bare_runtime_print_banner(void);
void bare_runtime_before_main(void);
void bare_runtime_after_main(void);
void hello_run(void);

// Mirrors the order bare_runtime.c's _init uses on the board.  The expected
// text must match hello.c; LHS + RHS there is 40 + 2 = 42 = 0x2A.
static const char expected[] =
    "\r\n"
    "========================================\r\n"
    "  BUDDY-MLIR Hello World (plain C)\r\n"
    "  NR / NH RISC-V @ 0x80000000\r\n"
    "========================================\r\n"
    "\r\n"
    "rt: call main\r\n"
    "Hello, World!\r\n"
    "computed: 40 + 2 = 0000002A\r\n"
    "verify hello: PASS\r\n"
    "\r\n"
    "=== Hello Done ===\r\n";

int main(void)
{
    bare_runtime_print_banner();
    bare_runtime_before_main();
    hello_run();
    bare_runtime_after_main();

    printf("--- captured UART output ---\n%s\n----------------------------\n",
           captured);

    if (strcmp(captured, expected) != 0) {
        printf("host_check: FAIL\n");
        printf("  expected %zu bytes, got %zu bytes\n", strlen(expected),
               strlen(captured));
        return 1;
    }

    printf("host_check: PASS (UART byte sequence matches)\n");
    return 0;
}
