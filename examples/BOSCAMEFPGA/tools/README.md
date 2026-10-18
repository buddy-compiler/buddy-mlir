# FPGA upload and run tools

Local and remote helpers used by [`../fpga_run.sh`](../fpga_run.sh) to load
an NR `.bin` onto a board, stream UART, and check DDR readback.

```text
tools/
├── fpga_run.py       # Local half: SSH upload, detach worker, UART relay
└── fpga_remote.py    # Remote half: UVHS load, UART capture, result.json
```

[`../fpga_run.sh`](../fpga_run.sh) is a thin wrapper that execs
`fpga_run.py` with the same arguments. Keep the shell script next to this
directory.

## Roles

| File | Where it runs | Role |
| --- | --- | --- |
| `fpga_run.py` | Developer machine | Creates a private remote run dir, uploads `fpga_remote.py` (as `runner.py`) and `image.bin`, starts the worker with `--detach`, relays UART to stdout, fetches logs |
| `fpga_remote.py` | FPGA server (UVHS workdir) | Claims the board, opens `/dev/FPGAN`, runs `make uv_runN`, captures UART, verifies DDR readback, writes `result.json` |

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
| `--capture-seconds=10` | UART window after FPGA startup |
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
example’s own PASS/FAIL line (for hello, see
[`../hello/README.md`](../hello/README.md)).

## See also

- Tree overview and quick start: [`../README.md`](../README.md)
- Shared UART / CRT: [`../common/README.md`](../common/README.md)
