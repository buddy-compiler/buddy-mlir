# BOSCAME FPGA: Hello on NR / RAV0.5

Bare-metal examples for the NR FPGA. **This tree currently brings up
[`hello`](hello/README.md) only**: compile a single-hart NH image, load it
into DDR, and print over UART. RA and AME are not used.

UART, CRT, and the C runtime live in [`common/`](common/README.md). Board
upload is [`fpga_run.sh`](fpga_run.sh) plus [`tools/`](tools/).

```text
BOSCAMEFPGA/
├── common/          # Shared UART, CRT, runtime, LLVM tool selection
├── hello/           # Plain-C Hello World (the working example)
├── fpga_run.sh      # Upload and run a .bin
└── tools/           # Python launcher and remote worker
```

## Build hello

From the repository root:

```bash
make -C examples/BOSCAMEFPGA/hello all size
```

This writes `examples/BOSCAMEFPGA/hello/build/hello.bin` (raw, not padded).

The makefile includes [`common/toolchain.mk`](common/toolchain.mk). By default
it uses `llvm/build/bin` when `clang` is there, otherwise `PATH`. Override
with `LLVM_BIN=...` or `RISCV_CC` / `RISCV_LD` / `RISCV_OBJCOPY`. After
changing the toolchain, run `make clean`. Details:
[`common/README.md`](common/README.md).

## Run on the FPGA

Needs: passwordless `ssh fpga` (or `--ssh-host`), a server workdir that
already has the UVHS `Makefile`, a free board (`--fpga=0` … `7`), and
Python 3 on the host. Close minicom / other UVHS sessions on that board first.

The script **does not compile**. Pass the `.bin` from the step above.

```bash
examples/BOSCAMEFPGA/fpga_run.sh \
  examples/BOSCAMEFPGA/hello/build/hello.bin \
  --fpga=5
```

The default SSH host is `fpga`. The default remote workdir is
`Desktop/fpga-tester-ISCAS` (relative to the SSH login directory; do not
write `~/`). That is the UVHS install path used in this lab. Other setups
should override with environment variables or flags; do not commit a
personal directory:

```bash
export FPGA_SSH_HOST=fpga
export FPGA_REMOTE_DIR=path/to/your-uvhs-workdir
```

`--ssh-host` and `--remote-dir` override those variables. Pick a free board
with `--fpga` (`0`–`7`).

Hello finishes well within the default 10 s capture window. Success is
`verify hello: PASS` on stdout (full text in [`hello/README.md`](hello/README.md)).
Exit code 0 means load and capture succeeded; look at the UART for PASS.

Status goes to stderr. Local logs:
`examples/BOSCAMEFPGA/build/fpga-runs/run-*/`.

Useful flags for hello:

| Flag | Role |
| --- | --- |
| `--fpga=N` | Board and `/dev/FPGAN` (required) |
| `--remote-dir=...` | Override default `Desktop/fpga-tester-ISCAS` |
| `--ssh-host=...` | Override default SSH alias `fpga` |
| `--capture-seconds=10` | UART window after startup |

Keep `fpga_run.sh` together with `tools/fpga_run.py` and `tools/fpga_remote.py`.
