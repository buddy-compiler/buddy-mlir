# BOSCAME FPGA examples (NR / RAV0.5)

Bare-metal examples for the NR FPGA. This file is the single entry for layout,
upload/run, and links into runtime and compiler docs.

```text
BOSCAMEFPGA/
├── common/          # Shared UART, CRT, C runtime, toolchain.mk
├── hello/           # NH single-hart UART bring-up (start here)
├── layernorm_197x768/
├── matmul_i8_*/
├── fpga_run.sh      # Wrapper around tools/fpga_run.py
└── tools/           # Board launcher + encode / audit / sim helpers
```

| Path | Role |
| --- | --- |
| [`common/`](common/) | Shared platform sources |
| [`hello/`](hello/) | First working board example (NH UART only) |
| [`tools/`](tools/) | `fpga_run.py` / `fpga_remote.py` and audit/sim scripts |
| [`../../runtime/boscame/`](../../runtime/boscame/README.md) | Bare-metal / NH-E6 / NR runtime |

Contract → plan → emit pipeline:
[`docs/BOSCAMEKernelContractPipeline.md`](../../docs/BOSCAMEKernelContractPipeline.md).
Host-side dialect demos:
[`../BOSCAMEDialect/`](../BOSCAMEDialect/README.md).

## Examples

| Example | Status | Notes |
| --- | --- | --- |
| [`hello/`](hello/) | Working | NH UART; no RA / AME |
| [`layernorm_197x768/`](layernorm_197x768/) | Contract backend | Starts from checked-in `kernel.frontend.mlir` |
| [`matmul_i8_*`](./) | Contract backend | Same handoff pattern |

Add new operators as sibling directories with their own short README for build
and PASS criteria. Shared build/run stays in `common/` and `tools/`.

## Build notes

Toolchain selection is in [`common/toolchain.mk`](common/toolchain.mk)
(`llvm/build/bin` when present, else `PATH`). Override with `LLVM_BIN=...` or
`RISCV_CC` / `RISCV_LD` / `RISCV_OBJCOPY`. Run `make clean` after changing the
toolchain.

FPGA builds may enable temporary scalar FP32 reduction workarounds; see
[`tools/README.scalar-reductions.md`](tools/README.scalar-reductions.md).

## Upload and run

The launcher **does not compile**. Pass an existing `.bin`.

Needs: passwordless SSH to the FPGA server, a remote UVHS workdir with its
`Makefile`, a free board (`--fpga=<N>`), Python 3 on the host. Close minicom /
other UVHS sessions on that board first.

**Do not commit personal host names or workdir paths.**

```bash
export FPGA_SSH_HOST=<ssh-alias-or-host>
export FPGA_REMOTE_DIR=<uvhs-workdir>   # login-relative or absolute; no ~/
```

```bash
examples/BOSCAMEFPGA/fpga_run.sh path/to/image.bin --fpga=<N>
```

| Flag | Role |
| --- | --- |
| `--fpga=N` | Board and `/dev/FPGAN` (required) |
| `--ssh-host=...` | Override `FPGA_SSH_HOST` |
| `--remote-dir=...` | Override `FPGA_REMOTE_DIR` |
| `--capture-seconds=10` | UART window after startup |
| `--startup-timeout=180` | Max seconds waiting for load/start |
| `--baud=115200` | Must match the board image |

Flow: create a private remote `fpga-runs/run-<id>/`, upload `fpga_remote.py`
(as `runner.py`) and `image.bin`, detach one worker, relay UART to stdout,
fetch `result.json` and logs. Status → stderr; UART → stdout. Local logs:
`examples/BOSCAMEFPGA/build/fpga-runs/run-*/`.

Exit code 0 means load/capture succeeded; read the example’s own PASS/FAIL
line on UART.

Script roles and encode/audit helpers: [`tools/README.md`](tools/README.md).

## Quick start

1. Build and run [`hello`](hello/README.md).
2. For other examples, swap the `.bin` and follow that example’s PASS criteria.
