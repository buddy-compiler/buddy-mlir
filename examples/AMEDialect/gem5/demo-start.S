// Minimal RV64 Linux SE startup for the MLIR-defined main function.

    .section .text.start,"ax",@progbits
    .globl _start
    .type _start,@function
_start:
    andi sp, sp, -16
    call main
    li a7, 93
    ecall
1:  j 1b
    .size _start, .-_start

    .section .note.GNU-stack,"",@progbits
