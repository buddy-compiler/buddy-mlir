// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// No ELF interpreter/libc: the board's dynamic loader cannot start with V OFF.
// This test observes the kernel's exec policy without executing any V code.
#include <asm/unistd.h>
// linux/prctl.h includes C declarations; values checked by the C++ driver.
#define PR_RISCV_V_GET_CONTROL 70
#define PR_RISCV_V_VSTATE_CTRL_CUR_MASK 3
#define PR_RISCV_V_VSTATE_CTRL_OFF 1

.text
.option norelax
.globl _start
_start:
  li a0, PR_RISCV_V_GET_CONTROL
  li a1, 0
  li a2, 0
  li a3, 0
  li a4, 0
  li a7, __NR_prctl
  ecall
  bltz a0, fail
  andi a0, a0, PR_RISCV_V_VSTATE_CTRL_CUR_MASK
  li t0, PR_RISCV_V_VSTATE_CTRL_OFF
  bne a0, t0, fail
  li a0, 1
  la a1, message
  li a2, message_end-message
  li a7, __NR_write
  ecall
  li a0, 0
  j finish
fail:
  li a0, 1
finish:
  li a7, __NR_exit
  ecall
.section .rodata
message:
  .ascii "PASS exec-off static image policy=off\n"
message_end:
.section .note.GNU-stack,"",@progbits
