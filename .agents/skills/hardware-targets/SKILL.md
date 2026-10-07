---
name: hardware-targets
description: Measured facts about the hardware Buddy-MLIR targets (SpacemiT K3 A100/X100 cores with RVV, IME matrix engine, TCM and AI DMA; Tenstorrent) - core layout, bandwidths, instruction costs, how to run on the boards. Use before optimizing or estimating performance for a specific chip, or when characterizing a new one.
---

# Hardware targets

Facts here were measured on real boards. Each reference file starts with
when and on what it was measured; re-measure when the board, firmware or
kernel changes, and update the file.

## Which file to read

Files are named `<vendor>-<chip>-<topic>.md`.

| Target | Read |
| --- | --- |
| SpacemiT K3 (A100 AI cores, X100 cores) | `references/spacemit-k3-architecture.md` (cores, memory, IME, TCM, DMA); `references/spacemit-k3-instruction-costs.md` (RVV and IME costs); `references/spacemit-k3-board-setup.md` (running and profiling on a board) |
| Tenstorrent | No measurements yet. Environment: `docs/TenstorrentEnvironment.md`. |

Optimization history per platform (what was done, what was rejected) lives
with the workflow it belongs to, e.g.
`llm-inference-optimization/references/spacemit-k3-*.md`.

## Characterizing a new target

Measure before optimizing for it, and write
`references/<vendor>-<chip>-architecture.md` and
`-instruction-costs.md`:

1. Cores: which cores run the workload, how they are grouped (clusters,
   pairs sharing units), how the OS schedules them (special process modes).
2. Memory: read bandwidth with 1, half and all cores; cache sizes; any
   shared paths that saturate before DRAM (measure loads from cache with
   several cores per cluster).
3. Instructions: throughput and latency of the vector / matrix instructions
   the kernels use, at the element widths and register groupings they use
   (`scripts/spacemit-k3-insn-bench.py` is a template).
4. On-chip memories and DMA engines: size, who can reach them at what cost,
   how user space gets them.

Record numbers with units and the measurement method, and avoid board host
names, accounts and local paths.
