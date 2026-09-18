# Shared NR FPGA Platform Sources

Files in this directory are reused by examples under `examples/BOSCAMEFPGA`.
They are vendored here; building does not require a local ModelZoo clone.

```text
common/
├── toolchain.mk          # LLVM tool selection (override with make NAME=value)
├── uart/                 # NR UART driver
│   ├── uart.h
│   └── uart.c
└── runtime/              # Single-hart CRT and C runtime
    ├── crt_uart.S
    ├── encoding.h
    ├── bare_runtime.h
    └── bare_runtime.c
```

| Path | Role |
| --- | --- |
| [`uart/uart.h`](uart/uart.h), [`uart/uart.c`](uart/uart.c) | NR UART: base `0x310b0000`, 32-bit MMIO, TX `+0x20`, LSR `+0x14`, fixed divisor 8 |
| [`runtime/crt_uart.S`](runtime/crt_uart.S) | Single-hart start: registers, `gp` / stack, BSS / TBSS clear, trap vector, jump to `_init` |
| [`runtime/encoding.h`](runtime/encoding.h) | `MSTATUS_FS` / `MSTATUS_MPP` fallbacks for the CRT |
| [`runtime/bare_runtime.c`](runtime/bare_runtime.c), [`.h`](runtime/bare_runtime.h) | C `_init`, UART init, weak banner / main hooks, trap print-and-halt |
| [`toolchain.mk`](toolchain.mk) | `RISCV_CC` / `RISCV_LD` / `RISCV_OBJCOPY` / `RISCV_OBJDUMP`; included by example makefiles |

## Toolchain

[`toolchain.mk`](toolchain.mk) is written for an example one directory below
`BOSCAMEFPGA` (`REPO_ROOT ?= ../../..`). It uses `llvm/build/bin/clang` when
that file exists, otherwise tools from `PATH`. A missing tool in the chosen
directory (for example `ld.lld`) also falls back to `PATH`.

```bash
# LLVM_BIN is relative to the example directory unless it is absolute.
make -C examples/BOSCAMEFPGA/hello LLVM_BIN=../../../llvm/build/bin all

# Use only tools on PATH.
make -C examples/BOSCAMEFPGA/hello LLVM_BIN= all
```

Required: clang with RV64, LLD, llvm-objcopy. llvm-objdump is used for
`make dump`. After changing the toolchain or `CFLAGS`, run `make clean`.

## UART and startup

- UART is 8N1 with a **fixed divisor of 8**, matching a 14.7456 MHz clock at
  115200 baud. `init_uart(freq, baud)` keeps those arguments for API
  compatibility and ignores them.
- `crt_uart.S` does not hard-code UART or DDR addresses. Memory layout comes
  from the example linker script.
- Flow: clear BSS / TBSS → `_init` → `init_uart` → banner → `main` → after-main
  → `wfi`. This is an NH-only path; it does not start RA.

An example linker script must define `_start`, `__global_pointer$`,
`__stack_top`, and the BSS / TBSS bounds. See
[`hello/platform/hello_nr.ld`](../hello/platform/hello_nr.ld).

## Runtime

`hello` is a plain-C program: `_init` initializes UART, calls the weak banner /
before-main / after-main hooks, runs `main()`, then waits in `wfi`. The CRT
still vectors traps to `handle_trap`, which prints `mcause` / `mepc` / `mtval`
and hangs. There is no heap, `malloc`, MLIR `memrefCopy`, AME, or cycle-trace
support in this runtime. New examples that need those must add them back and
check their own ABI and hardware.

## Provenance

Adapted from [ModelZoo](https://gitlink.org.cn/michaelcjl/ModelZoo) commit
`8815b74fb6d3cd6288c4d99ac6fd7c5d041a7cc3`:

| This tree | ModelZoo source |
| --- | --- |
| `uart/` | `thirdparty/nr/src/uart.c`, `thirdparty/nr/include/uart.h` |
| `runtime/crt_uart.S`, `encoding.h` | `thirdparty/platform-v01/` |
| `runtime/bare_runtime.c`, `.h` | `examples/tools/bare_runtime.c`, `bare_runtime.h` |

The CRT comments still refer to `riscv-dnn/include/common/crt.S`. File headers
record that origin; this tree does not invent license text for those sources.
Since import, the files have been formatted, UART implementation moved into
`uart.c`, and the assembly weak `_init` / gem5 `_exit` stubs removed so C
`_init` is the only entry after the CRT.

Hello bring-up and board upload are documented in [`../hello/README.md`](../hello/README.md)
and [`../README.md`](../README.md).
