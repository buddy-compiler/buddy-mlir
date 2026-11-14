#===- crt_uart.S - Single-hart bare-metal CRT for NR FPGA -----------------===#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#===----------------------------------------------------------------------===#
#
# Single-hart bare-metal CRT for the NH core on the NR FPGA platform.
# Boot path: zero GPRs, enable FPU state, install mtvec, set gp/sp, clear
# BSS/TBSS, then jump to C _init() in bare_runtime.c.
# Trap path: trap_entry saves x1-x31, calls handle_trap(mcause, mepc, regs),
# writes the return value back to mepc, restores state, and mret.
# RV64 only (hello builds with -march=rv64gc). Linker must provide
# __global_pointer$, __stack_top, and BSS/TBSS bounds.
#
#===----------------------------------------------------------------------===#

#include "encoding.h"

#if __riscv_xlen != 64
#error crt_uart.S supports RV64 only
#endif

#define LREG ld
#define SREG sd
#define REGBYTES 8

  .section ".text.init"
  .globl _start
# Reset / load entry. Linker script sets the reset vector to _start.
_start:
  # Zero all GPRs so early C code does not see residual register state.
  li  x1, 0
  li  x2, 0
  li  x3, 0
  li  x4, 0
  li  x5, 0
  li  x6, 0
  li  x7, 0
  li  x8, 0
  li  x9, 0
  li  x10,0
  li  x11,0
  li  x12,0
  li  x13,0
  li  x14,0
  li  x15,0
  li  x16,0
  li  x17,0
  li  x18,0
  li  x19,0
  li  x20,0
  li  x21,0
  li  x22,0
  li  x23,0
  li  x24,0
  li  x25,0
  li  x26,0
  li  x27,0
  li  x28,0
  li  x29,0
  li  x30,0
  li  x31,0

  # Enable floating-point state in mstatus (FS bits) before any FP use.
  li t0, MSTATUS_FS
  csrs mstatus, t0

  # Sanity check: on RV64, 1 << 31 is still positive in a 64-bit register.
  # If this image were somehow running as RV32, 1 << 31 would be negative
  # and we hang here instead of continuing with a broken ABI.
  li t0, 1
  slli t0, t0, 31
  bgez t0, 1f
2:
  j 2b
1:

#ifdef __riscv_flen
  # Temporarily point mtvec at the label after FPU init so an FP illegal
  # instruction during bring-up does not recurse into an unfinished trap path.
  la t0, 1f
  csrw mtvec, t0

  fssr    x0
  fmv.s.x f0, x0
  fmv.s.x f1, x0
  fmv.s.x f2, x0
  fmv.s.x f3, x0
  fmv.s.x f4, x0
  fmv.s.x f5, x0
  fmv.s.x f6, x0
  fmv.s.x f7, x0
  fmv.s.x f8, x0
  fmv.s.x f9, x0
  fmv.s.x f10,x0
  fmv.s.x f11,x0
  fmv.s.x f12,x0
  fmv.s.x f13,x0
  fmv.s.x f14,x0
  fmv.s.x f15,x0
  fmv.s.x f16,x0
  fmv.s.x f17,x0
  fmv.s.x f18,x0
  fmv.s.x f19,x0
  fmv.s.x f20,x0
  fmv.s.x f21,x0
  fmv.s.x f22,x0
  fmv.s.x f23,x0
  fmv.s.x f24,x0
  fmv.s.x f25,x0
  fmv.s.x f26,x0
  fmv.s.x f27,x0
  fmv.s.x f28,x0
  fmv.s.x f29,x0
  fmv.s.x f30,x0
  fmv.s.x f31,x0
1:
#endif

  # Install the real machine-mode trap vector used for the rest of boot/run.
  la t0, trap_entry
  csrw mtvec, t0

.option push
.option norelax
  la gp, __global_pointer$
.option pop

  # Hello is NH hart 0 only. Non-zero harts spin forever; stack top comes
  # from the linker script (__stack_top).
  csrr a0, mhartid
  li a1, 1
1:bgeu a0, a1, 1b

  la sp, __stack_top

  # C requires uninitialized static storage to start at zero. Clear BSS and
  # TBSS before _init; a CPU-only reset does not reload NOBITS sections.
  la t0, __bss_start
  la t1, __bss_end
3:
  bgeu t0, t1, 4f
  SREG zero, 0(t0)
  addi t0, t0, REGBYTES
  j 3b
4:
  la t0, __tbss_start
  la t1, __tbss_end
5:
  bgeu t0, t1, 6f
  SREG zero, 0(t0)
  addi t0, t0, REGBYTES
  j 5b
6:
  fence rw, rw

  # Hand off to C: bare_runtime.c::_init (UART, hooks, main, wfi).
  j _init
  .size    _start, . - _start

  .align 2
# Machine-mode trap stub. Calling convention into handle_trap (bare_runtime.c):
#   a0 = mcause, a1 = mepc, a2 = pointer to stacked x1-x31 frame.
# Return value in a0 is written back to mepc before mret. The default C
# handler never returns (prints then wfi); a strong override may resume.
trap_entry:
  addi sp, sp, -272

  SREG x1, 1*REGBYTES(sp)
  SREG x2, 2*REGBYTES(sp)
  SREG x3, 3*REGBYTES(sp)
  SREG x4, 4*REGBYTES(sp)
  SREG x5, 5*REGBYTES(sp)
  SREG x6, 6*REGBYTES(sp)
  SREG x7, 7*REGBYTES(sp)
  SREG x8, 8*REGBYTES(sp)
  SREG x9, 9*REGBYTES(sp)
  SREG x10, 10*REGBYTES(sp)
  SREG x11, 11*REGBYTES(sp)
  SREG x12, 12*REGBYTES(sp)
  SREG x13, 13*REGBYTES(sp)
  SREG x14, 14*REGBYTES(sp)
  SREG x15, 15*REGBYTES(sp)
  SREG x16, 16*REGBYTES(sp)
  SREG x17, 17*REGBYTES(sp)
  SREG x18, 18*REGBYTES(sp)
  SREG x19, 19*REGBYTES(sp)
  SREG x20, 20*REGBYTES(sp)
  SREG x21, 21*REGBYTES(sp)
  SREG x22, 22*REGBYTES(sp)
  SREG x23, 23*REGBYTES(sp)
  SREG x24, 24*REGBYTES(sp)
  SREG x25, 25*REGBYTES(sp)
  SREG x26, 26*REGBYTES(sp)
  SREG x27, 27*REGBYTES(sp)
  SREG x28, 28*REGBYTES(sp)
  SREG x29, 29*REGBYTES(sp)
  SREG x30, 30*REGBYTES(sp)
  SREG x31, 31*REGBYTES(sp)

  csrr a0, mcause
  csrr a1, mepc
  mv a2, sp
  jal handle_trap
  csrw mepc, a0

  # Keep privilege mode M after mret (set MPP in mstatus).
  li t0, MSTATUS_MPP
  csrs mstatus, t0

  LREG x1, 1*REGBYTES(sp)
  LREG x2, 2*REGBYTES(sp)
  LREG x3, 3*REGBYTES(sp)
  LREG x4, 4*REGBYTES(sp)
  LREG x5, 5*REGBYTES(sp)
  LREG x6, 6*REGBYTES(sp)
  LREG x7, 7*REGBYTES(sp)
  LREG x8, 8*REGBYTES(sp)
  LREG x9, 9*REGBYTES(sp)
  LREG x10, 10*REGBYTES(sp)
  LREG x11, 11*REGBYTES(sp)
  LREG x12, 12*REGBYTES(sp)
  LREG x13, 13*REGBYTES(sp)
  LREG x14, 14*REGBYTES(sp)
  LREG x15, 15*REGBYTES(sp)
  LREG x16, 16*REGBYTES(sp)
  LREG x17, 17*REGBYTES(sp)
  LREG x18, 18*REGBYTES(sp)
  LREG x19, 19*REGBYTES(sp)
  LREG x20, 20*REGBYTES(sp)
  LREG x21, 21*REGBYTES(sp)
  LREG x22, 22*REGBYTES(sp)
  LREG x23, 23*REGBYTES(sp)
  LREG x24, 24*REGBYTES(sp)
  LREG x25, 25*REGBYTES(sp)
  LREG x26, 26*REGBYTES(sp)
  LREG x27, 27*REGBYTES(sp)
  LREG x28, 28*REGBYTES(sp)
  LREG x29, 29*REGBYTES(sp)
  LREG x30, 30*REGBYTES(sp)
  LREG x31, 31*REGBYTES(sp)

  addi sp, sp, 272
  mret
