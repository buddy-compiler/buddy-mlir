OUTPUT_ARCH("riscv")
ENTRY(_start)

/* NH-only image for RAV0.5 / NR: nanhu_xilinx.i_ddr[0] is 0x80000000.
 * Adapted from ModelZoo examples/tools/probe_compact.ld; RA is not started, so no
 * NH-to-RA mailbox or second stack is required.
 */
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

  /* The CRT clears the optional runtime heap and ordinary BSS together. */
  .bare_runtime_heap ALIGN(64) (NOLOAD) : {
    __bss_start = .;
    *(.bare_runtime_heap)
  } > RAM :data
  .bss ALIGN(64) (NOLOAD) : {
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

  /* The shared CRT adds 1 MiB to __stack_base for hart 0. */
  .stack ALIGN(64) (NOLOAD) : {
    __stack_base = .;
    . += STACK_SIZE;
    __stack_top = .;
  } > RAM :data

  ASSERT(_start == RAM_ORIGIN, "NH entry must be 0x80000000")
  ASSERT(__stack_top <= RAM_ORIGIN + RAM_LENGTH, "NH stack exceeds DDR")
}
