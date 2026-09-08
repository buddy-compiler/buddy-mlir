OUTPUT_ARCH("riscv")
ENTRY(_start)

/* Shared bare-metal probe layout for the examples/ tests.
 *
 * This linker script describes the target platform, not any particular
 * example -- every example under examples/ links against this same copy via
 * its makefile's LD_SCRIPT. A tiny standalone test with no embedded model blob:
 * a single RAM segment from 0x86400000 (matches the Qwen3 probe/test layout).
 * writemem loads the flat binary to ariane_xilinx.i_ddr[0], which the core
 * sees at 0x86400000. */

RAM_ORIGIN = 0x86400000;
RAM_LENGTH = 0x10000000;   /* 256 MiB */

PHDRS
{
  text  PT_LOAD FLAGS(5);
  data  PT_LOAD FLAGS(6);
}

MEMORY
{
  RAM (rwx) : ORIGIN = RAM_ORIGIN, LENGTH = RAM_LENGTH
}

UART0_BASE = 0x10000000;

SECTIONS
{
  . = ORIGIN(RAM);

  /DISCARD/ : { *(.note.gnu.build-id) }
  .text.init : { *(.text.init) } > RAM :text
  .text      : { *(.text .text.*) } > RAM :text
  .rodata    : { *(.rodata .rodata.* .srodata .srodata.*) } > RAM :text
  .rodata.str1.1 : { *(.rodata.str1.1) } > RAM :text

  . = ALIGN(64);
  .data : { *(.data .data.*) } > RAM :data

  . = ALIGN(16);
  .sdata : {
    __global_pointer$ = . + 0x800;
    *(.sdata .sdata.*)
  } > RAM :data

  . = ALIGN(16);
  .bss (NOLOAD) : {
    . = ALIGN(64);
    *(.bare_runtime_heap)
    . = ALIGN(64);
    *(.matmul_ab)
    . = ALIGN(64);
    *(.sbss .sbss.* .gnu.linkonce.sb.*)
    *(.bss .bss.*)
    *(COMMON)
  } > RAM :data

  .tdata : { _tdata_begin = .; *(.tdata .tdata.*) _tdata_end = .; } > RAM :data
  .tbss  (NOLOAD) : { *(.tbss .tbss.*) _tbss_end = .; } > RAM :data

  . = ALIGN(64);
  _end = .;
}

/* Qwen3 crt_uart.S expects these for stack setup */
STACK_SIZE = 0x100000;
__stack_base = _end;
__stack_top = __stack_base + STACK_SIZE;
__heap_start = __stack_top;
__heap_end = __stack_top;
