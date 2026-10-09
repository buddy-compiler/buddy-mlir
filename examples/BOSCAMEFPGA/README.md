# BOSCAME FPGA examples (NR / RAV0.5)

Bare-metal examples for the NR FPGA. Shared platform code lives under
`common/`; board upload and UART relay live under `tools/` (via
`fpga_run.sh`).

Two bring-up paths:

- [`hello`](hello/README.md): plain C on a single NH hart (no RA / AME).
- [`qwen3-0.6b/`](qwen3-0.6b/README.md): NH starts RA; linalg operators built
  with `buddy-opt --lower-linalg-to-boscame=target=nr-fpga`.

UART is shared. Hello links [`common/runtime`](common/README.md); operators
link [`common/nr`](common/nr/README.md).

## Layout

```text
BOSCAMEFPGA/
├── common/          # UART, hello CRT, NR runtime, toolchain.mk
├── hello/           # Plain-C Hello World
├── qwen3-0.6b/      # Operator suite (see its README for the inventory)
├── fpga_run.sh      # Thin wrapper around tools/fpga_run.py
└── tools/           # Launcher, UVHS worker, AME encode, ELF audit
```

| Path | Role | Details |
| --- | --- | --- |
| [`common/`](common/) | Shared platform sources reused by all examples | [`common/README.md`](common/README.md) |
| [`fpga_run.sh`](fpga_run.sh) | Entry script: forwards all args to `tools/fpga_run.py` | [`tools/README.md`](tools/README.md) |
| [`tools/`](tools/) | Upload, load, UART relay, DDR readback; operator encode/audit | [`tools/README.md`](tools/README.md) |

## Examples

| Example | Status | Notes |
| --- | --- | --- |
| [`hello/`](hello/) | Working | NH single-hart UART bring-up; no RA / AME |
| [`qwen3-0.6b/`](qwen3-0.6b/README.md) | Working | NH→RA operator suite; inventory in that README |

## Shared build notes

Toolchain selection is in [`common/toolchain.mk`](common/toolchain.mk).
By default it uses `llvm/build/bin` when `clang` is there, otherwise
`PATH`. Override with `LLVM_BIN=...` or `RISCV_CC` / `RISCV_LD` /
`RISCV_OBJCOPY`. After changing the toolchain, run `make clean`.
See [`common/README.md`](common/README.md).

```bash
make -C examples/BOSCAMEFPGA/hello all size
make -C examples/BOSCAMEFPGA/qwen3-0.6b/<op> all check
```

Operators need `buddy-opt` / `buddy-translate` from this repository
(`build/bin`) with `--lower-linalg-to-boscame=target=nr-fpga`. Override with
`BUDDY_BIN=/path/to/bin` if needed. `check` is a host oracle only; it does
not replace a board run. Do not link `common/runtime` into an operator image.
Which `<op>` directories exist: [`qwen3-0.6b/README.md`](qwen3-0.6b/README.md).

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

```bash
# Hello (default capture window is enough)
examples/BOSCAMEFPGA/fpga_run.sh \
  examples/BOSCAMEFPGA/hello/build/hello.bin \
  --fpga=<N>

# Any operator under qwen3-0.6b/ (longer UART window is typical)
examples/BOSCAMEFPGA/fpga_run.sh \
  examples/BOSCAMEFPGA/qwen3-0.6b/<op>/build/<op>.bin \
  --fpga=<N> --capture-seconds=900
```

Hello finishes within the default 10 s capture window. Operators usually
need a longer window (`--capture-seconds=900` is a safe starting point).
Success on UART is `verify hello: PASS` or `verify <op>: PASS`.

| Flag | Role |
| --- | --- |
| `--fpga=N` | Board and `/dev/FPGAN` (required; valid indices are lab-specific, see internal docs) |
| `--ssh-host=...` | Override `FPGA_SSH_HOST` |
| `--remote-dir=...` | Override `FPGA_REMOTE_DIR` |
| `--capture-seconds=N` | UART window after startup (hello: 10; operators: try 900) |

Status goes to stderr. Local logs:
`examples/BOSCAMEFPGA/build/fpga-runs/run-*/`.

Keep `fpga_run.sh` together with `tools/fpga_run.py` and
`tools/fpga_remote.py`. Full tool behavior:
[`tools/README.md`](tools/README.md).

## See also

- Platform sources: [`common/README.md`](common/README.md)
- Hello example (first bring-up): [`hello/README.md`](hello/README.md)
- NR operator runtime: [`common/nr/README.md`](common/nr/README.md)
- Operator suite inventory: [`qwen3-0.6b/README.md`](qwen3-0.6b/README.md)
- Upload / run tooling: [`tools/README.md`](tools/README.md)
