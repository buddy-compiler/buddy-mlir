/*===- hello_nr.ld - NH-only linker layout for NR FPGA --------------------===
 *
 * nanhu_xilinx.i_ddr[0] is 0x80000000. Adapted from ModelZoo
 * examples/tools/probe_compact.ld. RA is not started, so no NH-to-RA mailbox
 * or second stack is required.
 *
 *===----------------------------------------------------------------------===*/

OUTPUT_ARCH("riscv")
ENTRY(_start)
RAM_ORIGIN = 0x80000000;
RAM_LENGTH = 0x80000000;
STACK_SIZE = 0x100000;

PHDRS
{
  text PT_LOAD FLAGS(5);
  data PT_LOAD FLAGS(6);
}

MEMORY
{
  RAM (rwx) : ORIGIN = RAM_ORIGIN, LENGTH = RAM_LENGTH
}

SECTIONS
{
  . = ORIGIN(RAM);
  /DISCARD/ : { *(.note.gnu.build-id) }
  .text.init : { KEEP(*(.text.init)) } > RAM :text
  .text : { *(.text .text.*) } > RAM :text
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > RAM :text
  .data ALIGN(64) : { *(.data .data.*) } > RAM :data
  .sdata ALIGN(16) : {
    __global_pointer$ = . + 0x800;
    *(.sdata .sdata.*)
  } > RAM :data

  .bss ALIGN(64) (NOLOAD) : {
    __bss_start = .;
    *(.sbss .sbss.* .gnu.linkonce.sb.*)
    *(.bss .bss.*)
    *(COMMON)
    . = ALIGN(64);
    __bss_end = .;
  } > RAM :data

  .tdata ALIGN(64) : { *(.tdata .tdata.*) } > RAM :data
  .tbss ALIGN(64) (NOLOAD) : {
    __tbss_start = .;
    *(.tbss .tbss.*)
    . = ALIGN(64);
    __tbss_end = .;
  } > RAM :data

  /* CRT loads sp from __stack_top. Stack grows down toward __stack_base. */
  .stack ALIGN(64) (NOLOAD) : {
    __stack_base = .;
    . += STACK_SIZE;
    __stack_top = .;
  } > RAM :data

  ASSERT(_start == RAM_ORIGIN, "NH entry must be 0x80000000")
  ASSERT(__stack_top <= RAM_ORIGIN + RAM_LENGTH, "NH stack exceeds DDR")
}
