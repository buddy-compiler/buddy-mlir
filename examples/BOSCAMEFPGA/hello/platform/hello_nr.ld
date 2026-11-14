/*===- hello_nr.ld - NH-only linker layout for NR FPGA --------------------===
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
 * NH-only memory layout for the hello bare-metal image on NR FPGA.
 * DDR starts at nanhu_xilinx.i_ddr[0] = 0x80000000. ENTRY(_start) must
 * land at that address so the board reset/load path can jump into
 * crt_uart.S. RA is not started, so this script has no NH-to-RA mailbox
 * and no second stack. crt_uart.S expects __global_pointer$, __stack_top,
 * and the BSS / TBSS bounds defined below.
 *
 *===----------------------------------------------------------------------===*/

OUTPUT_ARCH("riscv")
ENTRY(_start)

/* DDR window placed at the NH load address. 2 GiB is the mapped size;
 * the image itself is much smaller and sits at the low end. */
RAM_ORIGIN = 0x80000000;
RAM_LENGTH = 0x80000000;
STACK_SIZE = 0x100000;   /* 1 MiB stack for the single NH hart */

/* ELF program headers: text is R+X (5), data is R+W (6). */
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

  /* Boot stub from crt_uart.S must be first so _start == RAM_ORIGIN. */
  .text.init : { KEEP(*(.text.init)) } > RAM :text
  .text : { *(.text .text.*) } > RAM :text
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > RAM :text

  .data ALIGN(64) : { *(.data .data.*) } > RAM :data

  /* GP-relative small data. __global_pointer$ is midway in the +/-2 KiB
   * window so one auipc/addi pair can reach the whole .sdata region. */
  .sdata ALIGN(16) : {
    __global_pointer$ = . + 0x800;
    *(.sdata .sdata.*)
  } > RAM :data

  /* NOLOAD: not stored in the ELF image. crt_uart.S zeros
   * [__bss_start, __bss_end) before jumping to _init. */
  .bss ALIGN(64) (NOLOAD) : {
    __bss_start = .;
    *(.sbss .sbss.* .gnu.linkonce.sb.*)
    *(.bss .bss.*)
    *(COMMON)
    . = ALIGN(64);
    __bss_end = .;
  } > RAM :data

  .tdata ALIGN(64) : { *(.tdata .tdata.*) } > RAM :data

  /* Thread-local BSS. crt_uart.S also clears [__tbss_start, __tbss_end). */
  .tbss ALIGN(64) (NOLOAD) : {
    __tbss_start = .;
    *(.tbss .tbss.*)
    . = ALIGN(64);
    __tbss_end = .;
  } > RAM :data

  /* CRT loads sp from __stack_top. The stack grows down toward
   * __stack_base. NOLOAD: reserve space only, do not emit zeros. */
  .stack ALIGN(64) (NOLOAD) : {
    __stack_base = .;
    . += STACK_SIZE;
    __stack_top = .;
  } > RAM :data

  ASSERT(_start == RAM_ORIGIN, "NH entry must be 0x80000000")
  ASSERT(__stack_top <= RAM_ORIGIN + RAM_LENGTH, "NH stack exceeds DDR")
}
