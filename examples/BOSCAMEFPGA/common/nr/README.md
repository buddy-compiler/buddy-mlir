# NR NH/RA operator runtime

NH/RA firmware for linalg operators under
[`../../qwen3-0.6b`](../../qwen3-0.6b/README.md). Hello does
**not** use this directory; it keeps [`../runtime`](../README.md).

NH starts at `0x80000000`, initializes the shared [`../uart`](../uart) driver,
and starts RA. RA enables FS/VS/XS/AME, clears ordinary BSS, and calls the
example's `int launch(void)`. Return `0` is a passing check; any other
value, a trap, or a heap exhaustion becomes `FAIL` on UART.

Do not also link `../runtime/bare_runtime.c` or compile a second copy of
`uart.c` into the operator image. This runtime already supplies startup,
memory, and UART-forwarding symbols. Launch includes `nr_runtime.h`.

## How examples build this

There is no `nr.mk` in this tree. Operator directories use one-line makefiles:

```make
include ../common.mk
```

[`../../qwen3-0.6b/common.mk`](../../qwen3-0.6b/common.mk) compiles the objects
listed below, links [`nr.ld`](nr.ld), runs `tools/check_nr_elf.py` on the ELF,
and produces `.bin`. Override `BUDDY_BIN` if `buddy-opt` is not in
`build/bin`.

```text
nr/
├── crt.S            # NH + RA entry (not hello's crt_uart.S)
├── nr.ld            # Dual-hart layout, `.workspace` NOLOAD
├── nr_runtime.h     # `launch`, print, heap, math, copy
├── nr_runtime.c     # Mailbox, UART ring, malloc, memrefCopy
├── nr_math.c        # Freestanding exp/log/... helpers
├── ame_sync.c       # `ame_fence()` (1x1 MMA, non-transposed B)
└── nr_copy.S        # Optional aligned RVV memcpy
```

General C (runtime + launch) must use `-march=rv64gc_zicbom` without `+v`.
Only audited kernel assembly and `nr_copy.S` may use vector target options.
Elementwise operators (for example `add_1x1024`) stay on the scalar llc path.

## Logging and UART

Only NH programs UART MMIO and cache maintenance. RA writes a 64 KiB DDR
ring; NH forwards it to UART. If the ring is full, RA waits for NH.

On RA, `nr_write` / `nr_puts` / `nr_hex32` / `nr_hex64` work, and the
`print_uart*` names are compatibility wrappers. Numbers are hex.
`nr_cycles()` is the RA cycle counter. `nr_getchar()` reads a 4 KiB input
ring filled by NH and returns `-1` when empty.

## Memory

Ordinary BSS is cleared by RA. NH and RA each have a 1 MiB stack. The
completion byte is `0x80010000` (0 running, 1 PASS, 2 FAIL). NH writes the
RA entry PC to `0x50000000+0x100`. Code, BSS, stacks, and the bump heap stay
in low DDR below `0xb0000000`.

Large arenas start at `0xb8000000` (ModelZoo notes faults in
`[0xb0000000, 0xb8000000)`):

```c
#include "nr_runtime.h"
static unsigned char arena[640u * 1024u * 1024u] NR_WORKSPACE;
```

`.workspace` is `NOLOAD`: it does not grow the uploaded `.bin` and is not
zeroed at reset. `launch()` must initialize every element the kernel reads.
Linker symbols `__workspace_start` / `__workspace_end` bound the section.

`malloc` / `aligned_alloc` use a low-DDR bump heap; `free` does not recycle.
Save `nr_heap_mark()` and call `nr_heap_reset(mark)` only after live results
are copied out. `memcpy` / `memmove` / `memset` / `memcmp` and MLIR
`memrefCopy` (rank ≤ 8) are provided. `nr_copy_bytes` may use RVV for
4-byte-aligned, non-overlapping copies and clobbers vector state.

## AME fence

`ame_fence()` in `ame_sync.c` is the ModelZoo NR 1×1 AME sync with a
non-transposed B load. NR does not support transposed B loads. The helper
changes AME configuration; call it at operator boundaries only. Generated
kernels that emit AME ops must still put `fence rw, rw` around each AME
instruction. Pure elementwise operators do not emit AME ops.
