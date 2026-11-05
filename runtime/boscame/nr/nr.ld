OUTPUT_ARCH("riscv")
ENTRY(_start)

/* NR NH boots at 0x80000000. Leave the fixed RA signal outside code/BSS.
 * ModelZoo reports RA access faults in [0xb0000000,0xb8000000). The large
 * operator arena starts above that aperture and is deliberately NOLOAD. */
STACK_SIZE = 0x100000;
PHDRS {
  boot PT_LOAD FLAGS(5);
  mailbox PT_LOAD FLAGS(6);
  text PT_LOAD FLAGS(5);
  data PT_LOAD FLAGS(6);
  state PT_LOAD FLAGS(6);
  workspace PT_LOAD FLAGS(6);
}
MEMORY {
  LOW (rwx) : ORIGIN = 0x80000000, LENGTH = 0x30000000
  HIGH (rw) : ORIGIN = 0xb8000000, LENGTH = 0x48000000
}
SECTIONS {
  /DISCARD/ : { *(.note.gnu.build-id) *(.eh_frame .eh_frame.*) }
  .text.init 0x80000000 : { KEEP(*(.text.init)) } > LOW :boot
  ASSERT(. <= 0x80010000, "NR startup overlaps RA mailbox")
  .nr_mailbox 0x80010000 (NOLOAD) : {
    ra_sig = .;
    . += 0x1000;
  } > LOW :mailbox
  .text : { *(.text .text.*) } > LOW :text
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > LOW :text
  .data ALIGN(64) : { *(.data .data.*) } > LOW :data
  .sdata ALIGN(64) : {
    __global_pointer$ = . + 0x800;
    *(.sdata .sdata.*)
  } > LOW :data
  .tdata ALIGN(64) : { *(.tdata .tdata.*) } > LOW :data
  /* Match this before the ordinary .bss.* wildcard: NH initializes it. */
  .nr_console ALIGN(64) (NOLOAD) : {
    KEEP(*(.bss.nr_console))
    . = ALIGN(64);
  } > LOW :state
  .bss ALIGN(64) (NOLOAD) : {
    __bss_start = .;
    *(.sbss .sbss.* .bss .bss.*) *(COMMON)
    . = ALIGN(64);
    __bss_end = .;
  } > LOW :state
  .tbss ALIGN(64) (NOLOAD) : {
    __tbss_start = .;
    *(.tbss .tbss.*)
    . = ALIGN(64);
    __tbss_end = .;
  } > LOW :state
  .stacks ALIGN(64) (NOLOAD) : {
    __stack_base = .;
    . += STACK_SIZE;
    __stack_top = .;
    __ra_stack_base = .;
    . += STACK_SIZE;
    __ra_stack_top = .;
  } > LOW :state
  __heap_start = .;
  __heap_end = ORIGIN(LOW) + LENGTH(LOW);
  _end = .;
  .workspace 0xb8000000 (NOLOAD) : {
    __workspace_start = .;
    KEEP(*(.workspace .workspace.*))
    . = ALIGN(64);
    __workspace_end = .;
  } > HIGH :workspace
  ASSERT(_start == 0x80000000, "NH entry must be 0x80000000")
  ASSERT(__heap_start <= __heap_end, "NR ordinary state exceeds low DDR")
  ASSERT(__workspace_end <= 0x100000000, "NR workspace exceeds high DDR")
}
