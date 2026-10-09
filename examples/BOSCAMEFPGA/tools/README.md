# FPGA upload and run tools

Local and remote helpers used by [`../fpga_run.sh`](../fpga_run.sh) to load
an NR `.bin` onto a board, stream UART, and check DDR readback. Operator
builds also use the AME encode / restrict / ELF audit scripts below via
[`../qwen3-0.6b/common.mk`](../qwen3-0.6b/common.mk).

```text
tools/
├── fpga_run.py                 # Local half: SSH upload, detach worker, UART relay
├── fpga_remote.py              # Remote half: UVHS load, UART capture, result.json
├── ame_to_word.py              # Encode NR AME mnemonics to .word
├── restrict_fpga_assembly.py   # Validate forms; fence AME / vector memory
├── nr_isa.py                   # RVV allowlist and AME word checks
└── check_nr_elf.py             # Post-link ELF audit (SHF_EXECINSTR)
```

[`../fpga_run.sh`](../fpga_run.sh) is a thin wrapper that execs
`fpga_run.py` with the same arguments. Keep the shell script next to this
directory.

## Roles

| File | Where it runs | Role |
| --- | --- | --- |
| `fpga_run.py` | Developer machine | Creates a private remote run dir, uploads `fpga_remote.py` (as `runner.py`) and `image.bin`, starts the worker with `--detach`, relays UART to stdout, fetches logs |
| `fpga_remote.py` | FPGA server (UVHS workdir) | Claims the board, opens `/dev/FPGAN`, runs `make uv_runN`, captures UART, verifies DDR readback, writes `result.json` |
| `ame_to_word.py` | Build host | Encodes the verified NR AME mnemonic subset to `.word` (fixed or direct GPR mode) |
| `restrict_fpga_assembly.py` | Build host | Validates AME/RVV forms; inserts `fence rw, rw` before and after every AME / vector-memory op |
| `nr_isa.py` | Build host | Shared RVV allowlist and AME encoding checks used by the tools above |
| `check_nr_elf.py` | Build host | Walks every executable ELF section after link; confirms AME/RVV contract and fence adjacency |

`fpga_run*` only loads an existing `.bin`. The encode / restrict / ELF scripts
are invoked by `qwen3-0.6b/common.mk` during operator builds.

Status messages go to stderr; UART bytes go to stdout.

## Prerequisites

- Passwordless SSH to the FPGA server
- Remote workdir that already contains the UVHS `Makefile`
- Free board (`--fpga=<N>`); close minicom / other UVHS sessions first
- Python 3 on the host

SSH host names, UVHS install paths, and board maps are lab-specific.
**Do not commit personal values.** See the internal documentation.

## Configuration

```bash
export FPGA_SSH_HOST=<ssh-alias-or-host>
export FPGA_REMOTE_DIR=<uvhs-workdir>   # login-relative or absolute; no ~/
```

Flags `--ssh-host` and `--remote-dir` override those variables.

## Typical invocation

From the repository root (after building an image):

```bash
examples/BOSCAMEFPGA/fpga_run.sh \
  path/to/image.bin \
  --fpga=<N>
```

Useful flags:

| Flag | Role |
| --- | --- |
| `--fpga=N` | Board index and `/dev/FPGAN` (required; valid indices are lab-specific, see internal docs) |
| `--ssh-host=...` | Override `FPGA_SSH_HOST` |
| `--remote-dir=...` | Override `FPGA_REMOTE_DIR` |
| `--capture-seconds=10` | UART window after FPGA startup (operators often need `900`) |
| `--startup-timeout=180` | Max seconds waiting for load/start |
| `--baud=115200` | UART baud (must match the board image) |
| `--retries=5` | SSH reconnect attempts |

The script does **not** compile. Pass an existing `.bin`.

## Flow (summary)

1. Create `fpga-runs/run-<id>/` under the remote workdir (disk check first)
2. Upload `runner.py` and `image.bin` with SHA-256 verification
3. `--detach`: start at most one background worker for that run
4. `--relay`: stream `uart.raw.log` from a byte offset (survives SSH drops)
5. Fetch `result.json`, `uvhs.log`, and `worker.log` to the local run dir

Local logs: `examples/BOSCAMEFPGA/build/fpga-runs/run-*/`.
Remote logs: `<ssh-host>:<remote-dir>/fpga-runs/run-*/`.

Exit code 0 means load and capture succeeded; inspect UART for the
example’s own PASS/FAIL line (`verify hello: PASS` or
`verify <op>: PASS`; see [`../hello/README.md`](../hello/README.md)
and [`../qwen3-0.6b/README.md`](../qwen3-0.6b/README.md)).

## See also

- Tree overview and quick start: [`../README.md`](../README.md)
- Shared UART / CRT / NR runtime: [`../common/README.md`](../common/README.md),
  [`../common/nr/README.md`](../common/nr/README.md)
- Operator suite: [`../qwen3-0.6b/README.md`](../qwen3-0.6b/README.md)
