# BOSCAME FPGA examples (NR / RAV0.5)

Bare-metal examples for the NR FPGA. Shared platform code lives under
`common/`; board upload and UART relay live under `tools/` (via
`fpga_run.sh`). Each example has its own directory and README.

RA and AME usage depends on the example. The current bring-up path is
NH + UART only (see `hello`).

## Layout

```text
BOSCAMEFPGA/
├── common/          # Shared UART, CRT, C runtime, toolchain.mk
├── hello/           # Plain-C Hello World (working example)
├── fpga_run.sh      # Thin wrapper around tools/fpga_run.py
└── tools/           # Local launcher and remote UVHS worker
```

| Path | Role | Details |
| --- | --- | --- |
| [`common/`](common/) | Shared platform sources reused by all examples | [`common/README.md`](common/README.md) |
| [`fpga_run.sh`](fpga_run.sh) | Entry script: forwards all args to `tools/fpga_run.py` | [`tools/README.md`](tools/README.md) |
| [`tools/`](tools/) | Upload, load, UART relay, DDR readback check | [`tools/README.md`](tools/README.md) |

## Examples

| Example | Status | Notes |
| --- | --- | --- |
| [`hello/`](hello/) | Working | NH single-hart UART bring-up; no RA / AME |

Add new operators as sibling directories under `BOSCAMEFPGA/`, each with its
own README for build steps and expected output. Update this table when an
example is ready to use. Shared build/run mechanics stay in `common/` and
`tools/` unless the example needs extra runtime support.

## Shared build notes

Toolchain selection is in [`common/toolchain.mk`](common/toolchain.mk).
By default it uses `llvm/build/bin` when `clang` is there, otherwise
`PATH`. Override with `LLVM_BIN=...` or `RISCV_CC` / `RISCV_LD` /
`RISCV_OBJCOPY`. After changing the toolchain, run `make clean`.
See [`common/README.md`](common/README.md).

## Shared run notes

The launcher **does not compile**. Pass an existing `.bin` from any
example build.

You need: passwordless SSH to the FPGA server, a remote workdir that
already contains the UVHS `Makefile`, a free board (`--fpga=<N>`),
and Python 3 on the host. Close minicom / other UVHS sessions on that
board first.

**Do not commit personal host names or workdir paths.** Configure them
via environment variables or flags. For the lab SSH alias, UVHS install
path, and board map, see the internal documentation.

```bash
export FPGA_SSH_HOST=<ssh-alias-or-host>
export FPGA_REMOTE_DIR=<uvhs-workdir>   # login-relative or absolute; no ~/
```

`--ssh-host` and `--remote-dir` override those variables.

| Flag | Role |
| --- | --- |
| `--fpga=N` | Board and `/dev/FPGAN` (required; valid indices are lab-specific, see internal docs) |
| `--ssh-host=...` | Override `FPGA_SSH_HOST` |
| `--remote-dir=...` | Override `FPGA_REMOTE_DIR` |
| `--capture-seconds=10` | UART window after startup |

Status goes to stderr. Local logs:
`examples/BOSCAMEFPGA/build/fpga-runs/run-*/`.

Keep `fpga_run.sh` together with `tools/fpga_run.py` and
`tools/fpga_remote.py`. Full tool behavior:
[`tools/README.md`](tools/README.md).

## Quick start

The first working example is [`hello`](hello/README.md). Follow that
README to build and run on the FPGA. Other examples use the same upload
flow: swap the `.bin` and use that example’s README for PASS criteria.

## See also

- Platform sources: [`common/README.md`](common/README.md)
- Hello example (first bring-up): [`hello/README.md`](hello/README.md)
- Upload / run tooling: [`tools/README.md`](tools/README.md)
