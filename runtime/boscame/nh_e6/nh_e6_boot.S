//===- nh_e6_boot.S - NH/E6 dual-core startup and trap stubs --------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

  // nh_e6_boot.S -- the NH/E6 startup module of the common platform.
  //
  // Everything that has to run before a C function can exist lives here, and
  // only here:
  //
  //   _start      NH entry at 0x80000000: NH stack, trap stub, C platform flow
  //   e6_start    E6 entry at 0x80010000: E6 stack, FP/vector/AME state, trap
  //               stub, image BSS clear, C E6 flow
  //   nh_trap_entry / e6_trap_entry  4-byte aligned direct mtvec targets
  //
  // The two cores have separate stacks (link script symbols) and separate
  // processors, so no per-hart bookkeeping is needed.  The image BSS belongs to
  // E6: e6_start clears __bss_start..__bss_end before e6_main() runs, so NH
  // state must not live in .bss.  The NH flow keeps everything on its stack and
  // the platform's DDR diagnostic cells are written by E6 only.
  //
  // The C flows (nh_e6_nh_main, nh_e6_e6_main, nh_e6_nh_trap) publish the
  // completion protocol and return; the halt loops below own the final stop.
  // The E6 trap entry captures the exception CSRs and publishes 0xed without
  // touching the stack, so it still reports when the fault is a bad stack.
  .option norvc

  // Fixed RA SYSCTRL contract, mirroring nh_e6_platform.h (a C header can not
  // be included from assembly).  Do not renumber.
  .equ E6_DONE,       0x50000120

  .section .text.nh_start, "ax", @progbits
  .globl _start
  .type _start, @function
_start:
  la sp, __nh_stack_top
  la t0, nh_trap_entry
  csrw mtvec, t0
  call nh_e6_nh_main
.Lnh_halt:
  wfi
  j .Lnh_halt
  .size _start, . - _start

  // Direct mtvec needs a 4-byte aligned base, so the stub stays in assembly
  // even though the handler body is C.
  .align 2
nh_trap_entry:
  j nh_e6_nh_trap
.Lnh_trap_halt:
  wfi
  j .Lnh_trap_halt

  .section .text.e6_start, "ax", @progbits
  .globl e6_start
  .type e6_start, @function
e6_start:
  la sp, __e6_stack_top
  // Enable FP/vector/AME state.  AME MS[26:25] are part of the patch6
  // execution contract; omitting 0x06000000 makes the first AME instruction
  // trap even though the core itself has started.
  li t0, 0x0601e600
  csrs mstatus, t0
  la t0, e6_trap_entry
  csrw mtvec, t0

  // E6 owns the image BSS: clear it before any C code observes a zero-init
  // cell.  Both bounds are 8-byte aligned and the size is rounded up by the
  // linker script, so the word loop can not overrun the range.
  la t0, __bss_start
  la t1, __bss_end
.Le6_bss_loop:
  bgeu t0, t1, .Le6_bss_done
  sd zero, 0(t0)
  addi t0, t0, 8
  j .Le6_bss_loop
.Le6_bss_done:
  fence rw, rw
  call nh_e6_e6_main
.Le6_halt:
  wfi
  j .Le6_halt
  .size e6_start, . - e6_start

  .align 2
e6_trap_entry:
  csrr t1, mcause
  la t0, e6_diag_mcause
  sd t1, 0(t0)
  csrr t1, mepc
  la t0, e6_diag_mepc
  sd t1, 0(t0)
  csrr t1, mtval
  la t0, e6_diag_mtval
  sd t1, 0(t0)
  fence rw, rw
  li t0, E6_DONE
  // Let NH emit a distinct UART TRAP verdict before publishing 0xee to VCS.
  li t1, 0xed
  sw t1, 0(t0)
  fence iorw, iorw
  j .Le6_halt
