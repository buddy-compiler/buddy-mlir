/*===- nr.ld - Dual-hart linker script for NR operators -------------------===
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *===----------------------------------------------------------------------===
 *
 * Memory map for NH/RA operator images:
 *   - NH boots at 0x80000000. 0x80010000 is the completion byte (NOLOAD):
 *     0 running, 1 PASS, 2 FAIL. The RA entry PC is 0x50000000+0x100.
 *   - Code, BSS, dual stacks, and bump heap stay in low DDR (< 0xb0000000).
 *   - RA faults have been observed in [0xb0000000, 0xb8000000); the large
 *     operator .workspace therefore starts at 0xb8000000 as NOLOAD so the
 *     uploaded .bin stays small. launch() must initialize every element it
 *     reads.
 *
 *===----------------------------------------------------------------------===*/

OUTPUT_ARCH("riscv")
ENTRY(_start)

/* 1 MiB stack per hart (NH then RA), placed after TBSS. */
STACK_SIZE = 0x100000;

/* FLAGS: 4 = read, 2 = write, 1 = execute. 5 = RX, 6 = RW. */
PHDRS {
  boot PT_LOAD FLAGS(5);      /* RX: .text.init, the NH entry page */
  mailbox PT_LOAD FLAGS(6);   /* RW: fixed RA completion page, not in the .bin */
  text PT_LOAD FLAGS(5);      /* RX: ordinary text + rodata */
  data PT_LOAD FLAGS(6);      /* RW: initialized .data / .sdata / .tdata */
  state PT_LOAD FLAGS(6);     /* RW: console, BSS, stacks; NOLOAD, zeroed by RA */
  workspace PT_LOAD FLAGS(6); /* RW: high-DDR arena; NOLOAD, not zeroed */
}

MEMORY {
  /* Low DDR: code + ordinary state. Length stops before the RA fault window. */
  LOW (rwx) : ORIGIN = 0x80000000, LENGTH = 0x30000000
  /* High DDR: operator workspace only. */
  HIGH (rw) : ORIGIN = 0xb8000000, LENGTH = 0x48000000
}

SECTIONS {
  /DISCARD/ : { *(.note.gnu.build-id) *(.eh_frame .eh_frame.*) }

  /* NH must enter at the DDR base used by UVHS loaders. */
  .text.init 0x80000000 : { KEEP(*(.text.init)) } > LOW :boot
  ASSERT(. <= 0x80010000, "NR startup overlaps RA mailbox")

  /* Completion byte only. RA's entry PC is not stored in this page. */
  .nr_mailbox 0x80010000 (NOLOAD) : {
    ra_sig = .;
    . += 0x1000;
  } > LOW :mailbox

  .text : { *(.text .text.*) } > LOW :text
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > LOW :text
  .data ALIGN(64) : { *(.data .data.*) } > LOW :data
  /* gp points 2 KiB into .sdata so +/-2 KiB gp-relative loads reach it. */
  .sdata ALIGN(64) : {
    __global_pointer$ = . + 0x800;
    *(.sdata .sdata.*)
  } > LOW :data
  .tdata ALIGN(64) : { *(.tdata .tdata.*) } > LOW :data

  /* Match before the ordinary .bss.* wildcard: NH initializes this ring. */
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

  /* Bump heap for malloc/aligned_alloc on RA; free() does not recycle.
     The cursor walks from here up to the end of LOW, still below 0xb0000000. */
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
