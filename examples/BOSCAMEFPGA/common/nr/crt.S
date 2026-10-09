#===- crt.S - NH/RA entry for NR operator images --------------------------===#
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
# Dual-hart CRT for NR operator binaries (not hello's crt_uart.S).
#
# NH path (_start):
#   Disable interrupts, enable FP state, set gp/sp/tp/mtvec, jump to nr_nh_main.
#   nr_nh_main owns UART, starts RA, and drains the console ring.
#
# RA path (_ra_start):
#   Enable VS/FS/XS/AME, set RA stack, announce entry, clear application BSS
#   and TBSS (not .workspace), then jump to nr_ra_main -> launch().
#
# Trap paths reset the matching stack and tail-call C reporters.
#
#===----------------------------------------------------------------------===#

  .section .text.init,"ax",@progbits
  .globl _start
  .type _start,@function
_start:
  # NH hart entry at 0x80000000 (nr.ld).
  csrw mie, zero
  # Enable FS (bits 14:13) so FPU state is usable after boot.
  li t0, 0x6000
  csrs mstatus, t0
.option push
.option norelax
  la gp, __global_pointer$
.option pop
  la sp, __stack_top
  la tp, __tbss_start
  la t0, .Lnh_trap
  csrw mtvec, t0
  csrw fcsr, zero
  tail nr_nh_main
  .size _start, .-_start

  .balign 16
  .globl _ra_start
  .type _ra_start,@function
_ra_start:
  # nr_nh_main stores this PC in the RA control block at 0x50000000+0x100.
  # 0x80010000 is only the completion byte: 0 running, 1 PASS, 2 FAIL.
  csrw mie, zero
  # VS[10:9], FS[14:13], XS[16:15], AME MS[26:25] — required for AME/RVV paths.
  li t0, 0x0601e600
  csrs mstatus, t0
.option push
.option norelax
  la gp, __global_pointer$
.option pop
  la sp, __ra_stack_top
  la tp, __tbss_start
  la t0, .Lra_trap
  csrw mtvec, t0
  csrw fcsr, zero
  # Log before clearing potentially large application BSS. The console ring is
  # initialized by NH and lives in a separate NOLOAD section (.nr_console).
  call nr_ra_entered
  la t0, __bss_start
  la t1, __bss_end
1:
  bgeu t0, t1, 2f
  sd zero, 0(t0)
  addi t0, t0, 8
  j 1b
2:
  la t0, __tbss_start
  la t1, __tbss_end
3:
  bgeu t0, t1, 4f
  sd zero, 0(t0)
  addi t0, t0, 8
  j 3b
4:
  # Neither core clears .workspace; launch() must initialize what it reads.
  fence rw, rw
  tail nr_ra_main
  .size _ra_start, .-_ra_start

  # Trap reporting is terminal: reset the stack (covers stack faults) and
  # hand mcause/mepc/mtval to the C reporter without restoring a frame.
  .balign 4
.Lnh_trap:
  la sp, __stack_top
  csrr a0, mcause
  csrr a1, mepc
  csrr a2, mtval
  tail nr_nh_trap
  .balign 4
.Lra_trap:
  la sp, __ra_stack_top
  csrr a0, mcause
  csrr a1, mepc
  csrr a2, mtval
  tail nr_ra_trap

  .section .note.GNU-stack,"",@progbits
