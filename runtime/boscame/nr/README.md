# NR NH/RA runtime

Firmware for NR FPGA operators that launch work on RA. NH boots at
`0x80000000`, owns the physical UART, starts RA; RA enables FS/VS/XS/AME and
calls the example’s `int launch(void)` (`0` = PASS).

Sibling of [`../nh_e6`](../nh_e6/README.md). Hello uses the single-hart bare
runtime instead of this directory.

| File | Role |
| --- | --- |
| `crt.S` / `nr.ld` | Dual-hart boot and DDR layout (mailbox `0x80010000`, consoles, stacks) |
| `nr_runtime.c` / `.h` | NH/RA platform: console forward, mailbox, traps, result report |
| `nr_memory.c` | RA heap, scalar memory helpers, `memrefCopy` |
| [`probes/`](probes/README.md) | Optional RVV / copy capability probes (diagnostics only) |

RA logs go to a DDR buffer that NH relays to UART (ring or append-only modes
via compile flags). `nr_getchar()` reads an NH-filled input ring; RA never
touches UART MMIO. Optional AME cache probe hooks stay off unless
`NR_RA_AME_CACHE_DIAGNOSTIC=1`.

Parent index: [`../README.md`](../README.md).
Board upload: [`../../../examples/BOSCAMEFPGA/README.md`](../../../examples/BOSCAMEFPGA/README.md).
