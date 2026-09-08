OUTPUT_ARCH("riscv")
ENTRY(_start)

/*
 * Compact Buddy Qwen3 layer-wise image for the FPGA DDR window.
 *
 * The generated scheduler owns persistent hidden/KV/logit buffers and the
 * bare runtime owns a fixed per-layer arena.  They are NOLOAD state.  The
 * 757 MB packed parameter/input payload follows them in .model_blob, without
 * the large address hole produced by the old 256 MB probe layout.
 */
RAM_ORIGIN = 0x86400000;
RAM_END = 0xc0000000;
RAM_LENGTH = RAM_END - RAM_ORIGIN;
STACK_SIZE = 0x100000;

PHDRS
{
  text  PT_LOAD FLAGS(5);
  data  PT_LOAD FLAGS(6);
  model PT_LOAD FLAGS(4);
}

MEMORY
{
  RAM (rwx) : ORIGIN = RAM_ORIGIN, LENGTH = RAM_LENGTH
}

UART0_BASE = 0x10000000;

SECTIONS
{
  . = ORIGIN(RAM);
  __image_start = .;

  /DISCARD/ : { *(.note.gnu.build-id) }
  .text.init : { *(.text.init) } > RAM :text
  .text      : { *(.text .text.*) } > RAM :text
  .rodata    : {
    *(.rodata .rodata.* .srodata .srodata.*)
    *(.rodata.str1.1)
  } > RAM :text
  .rela.dyn  : { *(.rela.dyn) *(.rela.*) } > RAM :text

  . = ALIGN(64);
  .data : { *(.data .data.*) } > RAM :data

  . = ALIGN(16);
  .sdata : {
    __global_pointer$ = . + 0x800;
    *(.sdata .sdata.*)
  } > RAM :data

  . = ALIGN(64);
  __state_start = .;
  .bss (NOLOAD) : {
    *(.bare_runtime_heap)
    *(.sbss .sbss.* .gnu.linkonce.sb.*)
    *(.bss .bss.*)
    *(COMMON)
  } > RAM :data
  . = ALIGN(64);
  __state_end = .;

  .tdata : {
    _tdata_begin = .;
    *(.tdata .tdata.*)
    _tdata_end = .;
  } > RAM :data
  .tbss (NOLOAD) : {
    *(.tbss .tbss.*)
    _tbss_end = .;
  } > RAM :data

  . = ALIGN(64);
  _end = .;
  __model_start = .;
  .model_blob ALIGN(64) : { KEEP(*(.model_blob)) } > RAM :model
  . = ALIGN(64);
  __model_end = .;

  __stack_base = __model_end;
  __stack_top = __stack_base + STACK_SIZE;
  __heap_start = __stack_top;
  __heap_end = RAM_END;
  __image_end = __stack_top;

  ASSERT(SIZEOF(.model_blob) > 0, "Buddy Qwen3 model blob is empty")
  ASSERT(__stack_top <= RAM_END,
         "Buddy Qwen3 layer-wise image exceeds the FPGA DDR window")
}
