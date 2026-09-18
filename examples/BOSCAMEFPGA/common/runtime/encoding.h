//===- encoding.h - RISC-V mstatus bits for crt_uart.S --------------------===//
//
// Fallback MSTATUS_FS / MSTATUS_MPP when the toolchain header is absent.
// Adapted from ModelZoo thirdparty/platform-v01 encoding headers.
//
//===----------------------------------------------------------------------===//

#ifndef EXAMPLES_BOSCAMEFPGA_COMMON_RUNTIME_ENCODING_H
#define EXAMPLES_BOSCAMEFPGA_COMMON_RUNTIME_ENCODING_H

#if defined(__has_include)
#if __has_include(<riscv/encoding.h>)
#include <riscv/encoding.h>
#endif
#endif

#ifndef MSTATUS_MPP
#define MSTATUS_MPP 0x00001800UL
#endif
#ifndef MSTATUS_FS
#define MSTATUS_FS 0x00006000UL
#endif

#endif // EXAMPLES_BOSCAMEFPGA_COMMON_RUNTIME_ENCODING_H
