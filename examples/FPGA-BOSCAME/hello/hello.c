// Hello World on FPGA — plain C, no MLIR.
//
// The program runs on NH and prints a greeting and a fixed arithmetic tag to
// the NR UART at 0x310b0000, then returns.  There is no
// graph compiler in the path — clang compiles this straight to RISC-V and the
// linker lays it out for the board.
//
// The three bare_runtime_* functions below override the weak definitions in
// ../common/runtime/bare_runtime.c. That file supplies the strong _init that
// ../common/runtime/crt_uart.S jumps to, and it calls, in order:
//
//     bare_runtime_print_banner() → bare_runtime_before_main() → main()
//     → bare_runtime_after_main()
//
// The shared CRT provides generic single-hart setup; this example selects its
// own NR linker layout and UART driver. RA is not started.

#include "bare_runtime.h"
#include "uart.h"
#include <stdint.h>

// The greeting is followed by a value derived from these two operands.  Note
// that -O2 folds LHS + RHS into the immediate `li a0, 42` (see `make dump`),
// so this does not demonstrate run-time arithmetic; it is a fixed tag that
// makes the PASS/FAIL check below meaningful and confirms the compiled code
// ran and the print path works.
#define LHS 40
#define RHS 2
#define EXPECTED_ANSWER (LHS + RHS)

void bare_runtime_print_banner(void)
{
    print_uart("\r\n========================================\r\n");
    print_uart("  BUDDY-MLIR Hello World (plain C)\r\n");
    print_uart("  NR / NH RISC-V @ 0x80000000\r\n");
    print_uart("========================================\r\n\r\n");
}

void bare_runtime_before_main(void)
{
    print_uart("rt: call main\r\n");
}

void bare_runtime_after_main(void)
{
    print_uart("\r\n=== Hello Done ===\r\n");
}

void print_hello(int answer)
{
    print_uart("Hello, World!\r\n");
    print_uart("computed: 40 + 2 = ");
    print_uart_int((uint32_t)answer);
    print_uart("\r\n");

    // Keep a machine-readable PASS/FAIL marker in the serial output.
    if (answer == EXPECTED_ANSWER) {
        print_uart("verify hello: PASS\r\n");
    } else {
        print_uart("verify hello: FAIL expected=");
        print_uart_int(EXPECTED_ANSWER);
        print_uart(" actual=");
        print_uart_int((uint32_t)answer);
        print_uart("\r\n");
    }
}

// Everything the program does, minus the runtime hooks.  Kept separate from
// main() so tools/host_check.c can run it on the host under its own entry
// point (built with -DHELLO_NO_MAIN).
void hello_run(void)
{
    print_hello(LHS + RHS);
}

#ifndef HELLO_NO_MAIN
int main(void)
{
    hello_run();
    return 0;
}
#endif
