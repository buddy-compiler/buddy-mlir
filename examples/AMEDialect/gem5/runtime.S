// Minimal RV64 Linux SE startup and adapter for the local XAME matrix ABI.
// This adapter is specific to ame_matmul's three i32 M arguments, one i32 ACC
// argument, two descriptors and three pointers. It is not a standard AME psABI.

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

    .text
    .globl ame_run
    .type ame_run,@function
// int ame_run(const int *a, const int *b, int *c)
// Acquire once, provide initialized matrix arguments, and release after the
// MLIR-generated kernel. Return 77 when ownership or an operation is unavailable.
ame_run:
    addi sp, sp, -48
    sd ra, 40(sp)
    sd a0, 0(sp)
    sd a1, 8(sp)
    sd a2, 16(sp)

    ame.acquire a0, zero
    andi a0, a0, 1
    beqz a0, .Lunavailable
    csrw amestatus, zero

    // The local calling convention assigns M arguments to m0/m1/m2 and the
    // ACC argument to acc0, without consuming integer argument registers.
    // Signed i32: bit 30 selects signed integers and the low field is 32.
    li a0, 0x40000020
    msettyp m0, a0
    msettyp m1, a0
    msettyp m2, a0
    asettyp acc0, a0
    mv a1, a0
    ld a2, 0(sp)
    ld a3, 8(sp)
    ld a4, 16(sp)
    call ame_matmul

    // Read unsupported-operation status before release can change AME state.
    csrr a0, amestatus
    andi a0, a0, 1
    sd a0, 24(sp)
    ame.release
    ld a0, 24(sp)
    beqz a0, .Ldone
.Lunavailable:
    li a0, 77
.Ldone:
    ld ra, 40(sp)
    addi sp, sp, 48
    ret
    .size ame_run, .-ame_run

    .section .note.GNU-stack,"",@progbits
