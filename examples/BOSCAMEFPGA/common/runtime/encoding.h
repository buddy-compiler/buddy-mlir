//===- encoding.h - RISC-V mstatus bits for crt_uart.S --------------------===//
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
//
// mstatus bit masks used by crt_uart.S on the NH bare-metal path.
// Prefer the toolchain header <riscv/encoding.h> when it is present.
// If that header is missing, or it does not define these macros, the
// fallbacks below match the RISC-V privileged spec:
//   MSTATUS_MPP (bits 12:11) = 11b, so mret returns to machine mode.
//   MSTATUS_FS  (bits 14:13) = 11b, so the FPU state is enabled.
//
//===----------------------------------------------------------------------===//

#ifndef EXAMPLES_BOSCAMEFPGA_COMMON_RUNTIME_ENCODING_H
#define EXAMPLES_BOSCAMEFPGA_COMMON_RUNTIME_ENCODING_H

// Pull in the toolchain definitions when the compiler can see them.
// crt_uart.S only needs MSTATUS_FS and MSTATUS_MPP from that header.
#if defined(__has_include)
#if __has_include(<riscv/encoding.h>)
#include <riscv/encoding.h>
#endif
#endif

// Previous privilege mode for mret. crt_uart.S sets this in trap_entry so
// the hart stays in machine mode after mret. 0x1800 is bits 12:11 = 11b (M).
#ifndef MSTATUS_MPP
#define MSTATUS_MPP 0x00001800UL
#endif

// Floating-point status. crt_uart.S sets this at _start before clearing the
// FP registers. 0x6000 is bits 14:13 = 11b (Dirty), which enables FP use.
#ifndef MSTATUS_FS
#define MSTATUS_FS 0x00006000UL
#endif

#endif // EXAMPLES_BOSCAMEFPGA_COMMON_RUNTIME_ENCODING_H
