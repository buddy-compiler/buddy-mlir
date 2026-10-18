# SpacemiT K3: architecture as measured

Measured October 2026 on a K3 board (Linux 6.18 SpacemiT kernel), with
microbenchmarks and the DeepSeek R1 `w4g32` kernels. Instruction costs are
in `spacemit-k3-instruction-costs.md`.

## Cores

- 8 A100 AI cores (CPUs 8-15) with RVV 1.0 at VLEN 1024, Zfh/Zvfh, and the
  SpacemiT matrix extension (IME, `smt.vmadot*`). In-order: an instruction
  that waits for a long-latency result (a reduction, a load) stalls the
  core; independent work must be scheduled between.
- 2 clusters of 4 A100 cores: CPUs 8-11 and 12-15.
- The X100 general-purpose cores also exist; Buddy's K3 kernels do not use
  them.
- Only processes switched into "AI" mode run on the A100 cores:
  `echo 0 > /proc/set_ai_thread` in the process before it starts its
  threads (`spacemit-k3-board-setup.md`).
- LLVM: `-mcpu=spacemit-a100 -misched-prera-direction=topdown` schedules
  for the in-order pipeline (pipeline `kernels_a100`); bottom-up scheduling
  sinks loads next to their uses.

## Memory

- DRAM read bandwidth with 8 cores: ~26 GB/s. The RVV decode kernels reach
  24-25.7 GB/s on large matrices; 4 cores already reach ~24 GB/s.
- Cached vector loads go through a path shared by the cluster: a 1 KiB load
  takes ~14 ns alone and 36-41 ns when 3 or more cores of the cluster load.
  Kernels that re-read data from cache on every core of a cluster are bound
  by this, not by DRAM.
- `/tmp` on the boards is a tmpfs (RAM): large files there use memory.

## IME (matrix engine)

- One matrix unit per core pair: (8,9) (10,11) (12,13) (14,15). Two cores of
  a pair running `smt.vmadot` loops each get ~1.8x slower; other pairings do
  not interfere.
- Mixing fp16 (`.hp`) and int8 `vmadot` on the two cores of a pair costs
  ~2.5x each: keep a pair in one mode.
- `smt.vmadot` (XSMTVDot) and the int8 form (XSMTVDotII) are the same
  encodings; XSMTIME `vfmadot` raises SIGILL on the A100.
- `vmadot` computes 8 rows at a time: one row costs the same as eight.
- IME through Buddy: `ime.intr.vmadot.hp` (IME dialect, `-lower-ime
  target=k3`).

## TCM

- `/dev/tcm` (drivers/misc/tcm.c in the SpacemiT kernel): 8 blocks of
  384 KiB, two per core pair, 768 KiB per pair, mmap-able, readable and
  writable by all users.
- Pair-local TCM: 1 KiB load 9.3 ns, store 9.8 ns, no contention with 8
  cores loading. Another pair's TCM: ~3.3 us per KiB, never use it.
- Buddy: `runtime/spacemit/BuddySpacemitTcm.c` (`buddy_spacemit_tcm_pair` /
  `_here`, `BUDDY_SPACEMIT_TCM=0` disables it).

## AI DMA

- `drivers/dma/ai_dma_dev.c`: 8 controllers. User space fills a request ring
  (mmap `/dev/aidma_list`, 32 requests of 128 bytes: source / destination
  virtual address, pid, size, status SUBMIT) and rings `/dev/dma_msi`.
- The kernel translates only the first page of a request: source and
  destination must be physically contiguous (`/dev/ai_dma` kmalloc buffers up
  to 4 MiB, or TCM). Never pass transparent-hugepage addresses (the
  translation walks PTEs only).
- ~6 us per request. One channel DDR -> TCM ~6.9 GB/s at 256 KiB; 8 channels
  into one pair's TCM ~13.5 GB/s. Also a pack / transpose / pad mode.
- Driver bug: more than 8 SUBMIT requests can reuse a busy controller.

## SpacemiT software stack (for reference)

- spine-runtime (`libspert`, package spacemit-runtime) gives each core
  384 KiB of TCM through `spine_thread_tcm_malloc`.
- spine-triton has `smt.descriptor_load` / `smt.alloc` / mbarrier, lowered
  by the closed spine-mlir.
