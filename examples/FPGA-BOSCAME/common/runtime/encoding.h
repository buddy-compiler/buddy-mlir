#ifndef __LOCAL_RISCV_ENCODING_H__
#define __LOCAL_RISCV_ENCODING_H__

/* Prefer toolchain-provided header if available */
#if defined(__has_include)
#  if __has_include(<riscv/encoding.h>)
#    include <riscv/encoding.h>
#  endif
#endif

/* If toolchain header wasn't available, provide minimal fallbacks
 * sufficient for crt.S and util.h in riscv-dnn. */
#ifndef MSTATUS_UIE
#define MSTATUS_UIE   0x00000001UL
#endif
#ifndef MSTATUS_SIE
#define MSTATUS_SIE   0x00000002UL
#endif
#ifndef MSTATUS_MIE
#define MSTATUS_MIE   0x00000008UL
#endif
#ifndef MSTATUS_UPIE
#define MSTATUS_UPIE  0x00000010UL
#endif
#ifndef MSTATUS_SPIE
#define MSTATUS_SPIE  0x00000020UL
#endif
#ifndef MSTATUS_MPIE
#define MSTATUS_MPIE  0x00000080UL
#endif
#ifndef MSTATUS_SPP
#define MSTATUS_SPP   0x00000100UL
#endif
#ifndef MSTATUS_MPP
#define MSTATUS_MPP   0x00001800UL
#endif
#ifndef MSTATUS_FS
#define MSTATUS_FS    0x00006000UL
#endif
#ifndef MSTATUS_VS
#define MSTATUS_VS    0x00000600UL
#endif
#ifndef MSTATUS_XS
#define MSTATUS_XS    0x00018000UL
#endif

/* Only provide C helpers when not assembling */
#ifndef __ASSEMBLER__
#ifndef read_csr
#define read_csr(reg) ({ unsigned long __tmp; \
  __asm__ volatile ("csrr %0, " #reg : "=r"(__tmp)); \
  __tmp; })
#endif
#ifndef write_csr
#define write_csr(reg, val) do { unsigned long __v = (unsigned long)(val); \
  __asm__ volatile ("csrw " #reg ", %0" :: "rK"(__v)); \
} while (0)
#endif
#endif /* __ASSEMBLER__ */

#endif /* __LOCAL_RISCV_ENCODING_H__ */
