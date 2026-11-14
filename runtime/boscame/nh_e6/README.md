# NH/E6 platform (RA SYSCTRL)

Dual-hart firmware for BOSCAME SPM/DMA board examples: NH owns UART and the
final verdict; E6 runs the kernel and publishes completion through SYSCTRL.

- NH entry `0x80000000`, E6 entry `0x80010000`.
- Used by `layernorm_197x768` and the `matmul_i8_*` FPGA examples.
- Sibling of [`../nr`](../nr/README.md), **not** interchangeable.

| File | Role |
| --- | --- |
| `nh_e6_boot.S` | NH / E6 `_start`, traps, BSS clear |
| `nh_e6_platform.c` / `.h` | Release E6, bounded wait, PASS/FAIL/TRAP, PMP/`satp`, CSR `0xbc0` policy |
| `test_nh_e6_platform.c` | Host unit test with fake SYSCTRL/UART |

Apps supply `e6_main()`, `nh_e6_app_tag()`, and `nh_e6_app_report()` only —
no direct SYSCTRL, PMP, or `satp` programming.

Before `e6_main()`, the platform installs seven TOR PMP regions covering SPM,
DMA, the image/DDR window, and the rest of the address space, then `satp=0`
with `sfence.vma`. DMA helpers assume that setup and the CSR `0xbc0`
write-through / uncached-load policy described in `bosc_ame_dma.h`.

Parent index: [`../README.md`](../README.md).
