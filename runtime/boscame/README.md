# BOSCAME bare-metal runtime

Shared firmware pieces used by [`examples/BOSCAMEFPGA`](../../examples/BOSCAMEFPGA/README.md).
This directory is the **runtime index**; platform details live one level down.

| Path | Kind | Role |
| --- | --- | --- |
| `uart/` | Driver | NR UART MMIO (`0x310b0000`); host tests may override `UART_BASE` / `UART_PUTC` |
| `runtime/crt_uart.S` | Boot | Single-hart CRT → C `_init` |
| `runtime/encoding.h` | Boot | CSR definitions for CRT |
| `runtime/bare_runtime*.c` / `.h` | Runtime | Entry, bump heap, `memrefCopy`, trace, optional AME fence / W8A8 helper |
| `runtime/bosc_ame_dma.c` / `.h` | Driver | SPM/DMA register ABI (64 B granularity) |
| [`nr/`](nr/README.md) | Platform | NR NH→RA environment (mailbox, console) |
| [`nh_e6/`](nh_e6/README.md) | Platform | NH/E6 dual-hart via RA SYSCTRL (PMP / `satp` / DMA cache policy) |
| `toolchain.mk` | Build | LLVM tool selection for example makefiles |

Public headers stay stable: `bare_runtime.h`, `bosc_ame_dma.h`, `uart.h`,
`nr/nr_runtime.h`, `nh_e6/nh_e6_platform.h`. Splitting `.c` files does not
change the ABI.

**Do not link `nr` and `nh_e6` into the same image** — different boot, completion
codes, and link layouts.
