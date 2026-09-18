# Hello World: NH Single-Hart Program on NR

A plain-C bare-metal example for `RAV0.5_FPGA_ALPHA_260908`. NH runs `main()`
directly, prints `Hello, World!` and a check result over UART, and does not
start RA or execute AME.

## Build and run

From the repository root:

```bash
make -C examples/BOSCAMEFPGA/hello all size
examples/BOSCAMEFPGA/fpga_run.sh \
  examples/BOSCAMEFPGA/hello/build/hello.bin --fpga=5
```

Or from this example directory:

```bash
cd examples/BOSCAMEFPGA/hello
make all
../fpga_run.sh build/hello.bin --fpga=5
```

The build produces `build/hello.elf` and an unpadded `build/hello.bin`.
The board script uploads into a private run directory, opens UART, and
invokes the server's `make uv_run5` to pad, load, and run, then streams
serial output back. A separate minicom session is not required.
Capture lasts 10 seconds by default; use `--capture-seconds=60` to extend
it. Toolchain, SSH, and server workdir are documented in the
[parent README](../README.md).

## Files

```text
hello/
├── hello.c                 # Program that runs on the board
├── makefile                # Build and disassembly
├── platform/hello_nr.ld    # NH-only linker layout
└── README.md               # This note
```

| File | Role |
| --- | --- |
| [`hello.c`](hello.c) | Overrides the `bare_runtime_*` hooks and prints the banner, `Hello, World!`, and the hex value of `40 + 2`. RA is not started. UART and CRT come from [`../common`](../common/README.md). |
| [`makefile`](makefile) | `hello.c` → RISC-V objects → `build/hello.elf` → `build/hello.bin` (raw, not padded). `make dump` disassembles; `make size` reports the image size. Includes [`../common/toolchain.mk`](../common/toolchain.mk). |
| [`platform/hello_nr.ld`](platform/hello_nr.ld) | Entry `0x80000000` (`nanhu_xilinx.i_ddr[0]`), BSS / TBSS bounds, 1 MiB stack. RA is not started; there is no NH-to-RA mailbox. |
| `build/hello.elf`, `build/hello.bin` | Build outputs, not committed. The server `uv_run5` 64-byte-pads the `.bin` before loading. |

## Expected output

```text

========================================
  BUDDY-MLIR Hello World (plain C)
  NR / NH RISC-V @ 0x80000000
========================================

rt: call main
Hello, World!
computed: 40 + 2 = 0000002A
verify hello: PASS

=== Hello Done ===
```

`print_uart_int` prints eight hex digits; `0000002A` is decimal 42.
`-O2` folds `40 + 2` into a constant. The example checks the boot and
print path, not run-time arithmetic.
