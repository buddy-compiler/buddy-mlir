/* NH/RA entry contract adapted from ModelZoo thirdparty/nr/startup/crt_uart.S.
 * RA owns application BSS. Neither core clears the separate .workspace. */
  .section .text.init,"ax",@progbits
  .globl _start
  .type _start,@function
_start:
  csrw mie, zero
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
  csrw mie, zero
  # VS[10:9], FS[14:13], XS[16:15], AME MS[26:25].
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
  # Log before clearing potentially large application BSS. The console is
  # initialized by NH and resides in a separate, non-BSS output section.
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
  fence rw, rw
  tail nr_ra_main
  .size _ra_start, .-_ra_start

  # Trap reporting is terminal, so no interrupted register frame is needed.
  # Reset the stack as well: this also reports stack faults where possible.
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
