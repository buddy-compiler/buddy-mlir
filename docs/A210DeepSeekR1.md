# DeepSeek R1 on the Zhihe A210 (variant `w4g32`, kernels `a210`)

The `w4g32` variant of DeepSeek-R1-Distill-Qwen-1.5B (int4 weights with one
f16 scale per 32 rows, see [K3DeepSeekR1.md](K3DeepSeekR1.md)) runs on the 8
cores of a Zhihe A210: 4 XuanTie C908 (CPUs 0-3) and 4 XuanTie C920 (CPUs
4-7), all riscv64 with RVV 1.0 at VLEN 128 (both support Zfh / Zvfh, which
the kernel does not list in `/proc/cpuinfo`). With `"kernels": "a210"` the
graph rewrite is that of the K3 kernels, but the kernels are those of
`frontend/Python/graph/transform/a210_w4.py`, written for VLEN 128 and for
cores of two kinds.

## Build

As in [K3DeepSeekR1.md](K3DeepSeekR1.md), with the A210 spec and VLEN 128:

```bash
python3 tools/buddy-codegen/build_model.py \
  --spec models/deepseek_r1/specs/a210_w4g32.json \
  --build-dir build \
  --target deepseek_r1_rax \
  --cmake-args=-DBUDDY_RISCV_VLEN=128 \
  --cmake-args=-DBUDDY_RISCV_ENABLE_ZFH_ZVFH=ON \
  --is-rvv-crosscompile \
  --riscv-gnu-toolchain ${BUILD_RISCV_GNU_TOOLCHAIN_DIR} \
  --riscv-mlir-c-runner-utils ${RISCV_MLIR_C_RUNNER_UTILS} \
  --riscv-llvm-build-dir ${RISCV_LLVM_BUILD_DIR} \
  --buddy-mlir-build-dir ${BUDDY_MLIR_BUILD_DIR}
```

The spec (`models/deepseek_r1/specs/a210_w4g32.json`):

```json
{
  "variant": "w4g32",
  "kernels": "a210",
  "max_token_len": 1024,
  "num_threads": 8,
  "thread_cores": ["c908", "c908", "c908", "c908", "c920", "c920", "c920", "c920"],
  "thread_weights": {
    "prefill": [1, 1, 1, 1, 3, 3, 3, 3],
    "decode": [1, 1, 1, 1, 0.75, 0.75, 0.75, 0.75]
  },
  "prefill_chunk": 64,
  "arena": true,
  "thread_pool": true
}
```

- `"kernels"`: `"k3"` (the default) or `"a210"`. It selects the kernel module
  (`k3_w4.py` / `a210_w4.py`), the generated file
  (`k3_kernels-w4g32.mlir` / `a210_kernels-w4g32.mlir`) and the object linked
  into the model library. `"prefill_ime"` needs `"k3"`.
- `"thread_weights"` (A210 kernels only, optional): one positive weight per
  thread for the kernels of the prefill chunk and one for decode; thread i
  gets about weights[i] / sum(weights) of each kernel's work. The thread pool
  (`thread_pool`) pins thread i to the i-th CPU the process may run on, so
  the lists above are in CPU order: run `buddy-cli` with all 8 CPUs allowed.
  Without a list, equal shares.
- `"thread_cores"` (A210 kernels only, optional): the core of each thread,
  `"c908"` or `"c920"`, in the same order. It selects the tile function a
  thread runs in the prefill kernels (`a210_w4.TILE_CONFIGS`: the block
  shape and the LLVM scheduling model, `tune-cpu`); without it, all threads
  run the C920 one.
- The weight files have the same size as with the K3 kernels but other
  bytes; `import_model.py` records the layout (`kernels` and the module's
  `W4G32_LAYOUT_VERSION`) in the weights manifest, so that a build never reuses
  weights packed for another layout.

## Run

```bash
buddy-cli --model deepseek_r1.rax --prompt "..."
```

`--max-tokens` counts the prompt and the generated tokens.

## How it works

| Piece | Where |
| --- | --- |
| The graph rewrite of the K3 kernels with another weight layout and other kernel names (`_Rewriter.PREFIX`, `pack_q4`, `pack_glu`, `kernel_spec`) | `k3_w4.py` (`_Rewriter`), `a210_w4.py` (`A210Rewriter`, `a210_w4_rewrite`) |
| Weight layout: tiles of 16 columns (one e8 register at VLEN 128), group by group 16 rows of 16 bytes (two int4 rows per byte) and the 16 f16 scales | `a210_w4.py` (`pack_tiles`) |
| The work of a kernel split by the thread weights: (tile, block of 4 rows) items in tile order, thread i a contiguous range | `a210_w4.py` (`split_bounds`, `_matmul_fn`) |
| The tiles: per group, 4 steps of 4 row pairs (one 64-byte load), the nibbles unpacked once for the 4 rows of a block, two int16 accumulators per row (`vwmacc.vx`), then int16 -> f32 (`vfwcvt`), the scales and an fma | `a210_w4.py` (`_group_sums`, `_accumulate`) |
| One prefill tile function per core config, a thread calling the one of its core; the C920 one scheduled for the SiFive U74 (`llvm.tune_cpu`, which `-convert-func-to-llvm` turns into the function's `tune-cpu`) | `a210_w4.py` (`TILE_CONFIGS`, `_tile_fn`, `_call_tile`) |
| Spec fields `kernels`, `thread_weights`, `thread_cores` | `gen_config.py`, `buddy_model.cmake`, `compile_pipeline.py` (`a210_kernels`) |
| Weight reuse keyed on the layout | `import_model.py` (`w4g32_layout`) |

The attention kernels are still those of `k3_w4.py` (vectors of a whole head,
128 floats, written for VLEN 1024); at 448 tokens they take a few percent of
the time.

Why these choices (measured on the A210, one core unless noted):

- Peak `vwmacc.vx` (e8 -> e16): C920 27.9 GMAC/s, C908 7.4 GMAC/s, the same
  at LMUL 1 and 2. Tiles of 32 columns (LMUL 2) were not faster.
- Read bandwidth: the C920 cluster reaches ~3 GB/s, the C908 cluster ~8 GB/s,
  all 8 cores ~11 GB/s (12-13 GB/s for the LM head kernel). Decode streams
  868 MB per token, so the C908 cores take the larger decode share.
- The K3 tiles of 128 columns are LMUL 8 at VLEN 128: their 32 unpacked rows
  do not fit in the registers and go through the stack.
- A fully unrolled group: llc unpacks all 16 row pairs first and spills them;
  a loop of 4 row pairs per step does not.
- One int16 accumulator per row makes each step a chain of 8 dependent
  `vwmacc` per row: 14.9 GMAC/s on a C920 in a C microbenchmark, two
  accumulators 18.5.
- The generated code against C: the full 64 x 1536 x 1536 o projection
  written in C with the same loops runs at 19-20.5 GOPS on one C920 and
  7.0 on one C908; the generated kernel at 18.2 and 7.7 (19.8 on the C920
  with the U74 schedule). A C microbenchmark that kept one tile and 4 rows
  in L1 had shown 28 GOPS, but the compiler had dropped part of its
  per-group epilogue (the scales were known constants); with the epilogue,
  the int16 -> f32 conversion and the two f32 operations per row and group
  cost ~30% next to the 32 `vwmacc` of a row and group, and four rows with
  two accumulators use all 32 vector registers. That is about the limit of
  this tile on the A210.
- On the in-order C908, small changes of the generated loops move the
  speed by 10-15% either way (llc's instruction order): a do-while form of
  the loops, which removes the loop-carried vector copies llc leaves in an
  `scf.for`, made the C908 slower (7.7 -> 6.9 GOPS); tiles of 2 rows and one
  accumulator were faster in C (8.2) but slower generated (6.7). The kernels
  keep what measured best, and an `opt -O3` stage before llc (the pipeline
  has none) did not help.

## Results

A210, `buddy-cli`, greedy, 8 threads on the 8 cores, a 448-token prompt
(the rows of 7 chunks); the text is the same as with the K3 kernels:

| | prefill, 448 tokens | decode after 448 tokens | decode from a short context |
| --- | --- | --- | --- |
| K3 kernels (`w4g32`, VLEN 128, 8 threads) | 26.6 s (16.8 tok/s) | 8.8 tok/s | 9.6 tok/s |
| A210 kernels | 10.0-10.4 s (44 tok/s) | 11.8 tok/s | 12.6 tok/s |
| llama.cpp (upstream 2026-10-01, RVV + Zvfh, Q4_0, Q4_0 LM head, 8 threads) | 67 s (pp448: 6.7 tok/s) | 4.7 tok/s | 5.0 tok/s |

What each part gave, measured while the kernels were developed (448-token
prefill / decode from a short context):

| | prefill | decode |
| --- | --- | --- |
| K3 kernels | 26.6 s | 9.6 tok/s |
| tiles of 16 columns, work split by thread weights | 18.5 s | 13.2 tok/s |
| groups as loops of 4 row pairs, two accumulators per row, int16 -> f32 with `vfwcvt` | 11.3 s | 12.7 tok/s |
| prefill tiles by core (`thread_cores`, U74 schedule for the C920), C920 weight 3 | 10.0-10.4 s | 12.6 tok/s |

Matmul kernels at the shapes of the model (8 threads, median of 20 calls):

| | K3 kernels | A210 kernels |
| --- | --- | --- |
| q / k / v, 64 rows | 7.6 ms (53 GOPS) | 3.0 ms (135 GOPS) |
| gate / up, 64 rows | 70 ms (50 GOPS) | 28.7 ms (123 GOPS) |
| down, 64 rows | 45 ms (39 GOPS) | 13.9 ms (127 GOPS) |
| LM head, 1 row | 13.2 ms (9.9 GB/s) | 10.6 ms (12.4 GB/s) |

## Tests

- `tests/Python/test_a210_w4_kernels.py`: the matmul kernels on the host
  against numpy references, one row and a block of rows, equal, weighted and
  mixed-core splits.
- `tests/Python/test_a210_w4_import.py`: the spec fields, the rewrite of a
  tiny Qwen2 model with the A210 kernels and their weight layout.
