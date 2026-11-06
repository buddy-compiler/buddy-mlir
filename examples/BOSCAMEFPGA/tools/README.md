# BOSCAME FPGA tools

Board launcher and NR encode / audit / sim helpers. **How to upload and run**
lives in the parent [`../README.md`](../README.md); this file only lists scripts.

| Script | Role |
| --- | --- |
| `fpga_run.py` | Host: SSH upload, detach worker, UART relay |
| `fpga_remote.py` | Server: UVHS load, UART capture, DDR readback, `result.json` |
| `ame_to_word.py` | Encode verified NR AME mnemonics to raw words |
| `restrict_fpga_assembly.py` | Fence / reject illegal AME–RVV forms |
| `nr_isa.py` | NR instruction allowlist helpers |
| `boscame_sim.py` / `boscame_check.py` / `boscame_buffer_sim.py` | Functional sim and structural checks |
| `check_*.py` | ELF / stack / hash / boundary audits |

[`../fpga_run.sh`](../fpga_run.sh) execs `fpga_run.py` with the same arguments.
Each `.py` file header documents purpose, inputs/outputs, and when to run.

Temporary FP32 reduction workarounds (FPGA default):
[`README.scalar-reductions.md`](README.scalar-reductions.md).
