# Buddy Compiler BGE-M3 Example

## Introduction

This example compiles the BGE-M3 dense embedding model (single-forward path,
~568M params, f32) to a RAX package and optimizes it for the SpacemiT K3
(issue #888). It uses the same self-contained build style as
`examples/BuddyDeepSeekR1`: the x86 pipeline generates architecture-
independent artifacts, and the K3 side builds natively with prebuilt LLVM
and runs the final benchmark.

Optimizations live in `midend/lib/Conversion/MatMulOptimization/`:

| Pass | What it does |
|---|---|
| `-bge-m3-batchmatmul-transpose-b-vec` | Vectorizes the 48 attention batch matmuls (Q@Kᵀ / scores@V) with guarded transpose-B mapping |
| `-linalg-transpose-tile` | Tiled, cache-friendly lowering of the 240 materialized `linalg.transpose` copies |
| `-bge-m3-matmul-a100` | Replaces 143/144 f32 `linalg.matmul` with a call into the hybrid X100/A100 GEMM kernel (`a100_kernel.cpp`) |

Every optimization is gated by the embedding cosine gate (cos > 0.999).

## Directory layout

```
├── CMakeLists.txt       # self-contained x86 build: import -> .o -> .so -> .rax
├── README.md            # this file (issue #888 answer)
├── env.sh               # K3 environment (all K3 scripts source it)
├── a100_kernel.cpp      # hybrid GEMM kernel: X100 pthread pool + A100 spine-runtime
├── build_runtime.sh     # K3: build runner/embedding plugins (once)
├── build_seq.sh         # K3: native build of one seq variant
├── bench.sh             # K3: benchmark, mode A (cold CLI) + mode B (steady server)
├── bench_all.sh         # K3: serial benchmark of several seq/tag pairs
├── cos.py               # embedding cosine gate (pure stdlib)
├── export/              # x86: export MLIR graphs + weights for K3
└── src/ out/ results/   # K3 workspace (generated, git-ignored)
```

## How to run on non-RISC-V device

0. Prepare Python deps (`torch`, `transformers`, buddy python packages) and
   a local HuggingFace-format BGE-M3 snapshot.

1. Configure and build:

```bash
$ cmake -S . -B build -DBUDDY_BGEM3_EXAMPLES=ON \
    -DBUDDY_BGE_M3_MODEL_PATH=/path/to/bge-m3-snapshot
$ ninja -C build bge-m3-optimized
# output: build/examples/BuddyBgeM3/bge_m3.rax
```

2. Export artifacts for K3 (`export/export_artifacts.sh` sources
   `export/env_x86.sh` first; create it from your own paths):

```bash
$ bash examples/BuddyBgeM3/export/export_artifacts.sh 128 256 512
$ scp -r examples/BuddyBgeM3/dist \
    k3:~/buddy-k3/buddy-mlir/examples/BuddyBgeM3/src/
```

3. Verify on x86 (cosine gate against a saved baseline embedding):

```bash
$ build/bin/buddy-cli --model build/examples/BuddyBgeM3/bge_m3.rax \
    --prompt "The quick brown fox jumps over the lazy dog." --no-stats \
    > /tmp/emb_opt.txt
$ python3 examples/BuddyBgeM3/cos.py /tmp/emb_baseline.txt /tmp/emb_opt.txt
```

## How to run on RISC-V machine (SpacemiT K3)

K3 builds natively with prebuilt LLVM + buddy tools (no python/CMake/torch):

```bash
$ cd ~/buddy-k3/buddy-mlir/examples/BuddyBgeM3
$ bash build_runtime.sh            # once
$ bash build_seq.sh 128 baseline   # reference chain (offload OFF)
$ bash build_seq.sh 128 exp_a100   # optimized chain + A100 offload (default)
$ python3 cos.py results/seq128-baseline/emb.txt \
    results/seq128-exp_a100/emb.txt
$ bash bench.sh 128 baseline && bash bench.sh 128 exp_a100
```

Notes:
- `build_seq.sh` enables the A100 pass by default; build a no-offload
  reference package with `A100_PASS_OPTS="-bge-m3-a100-limit=0"`.
- Kernel knobs (env): `BGE_M3_A100_DISABLE=1`, `BGE_M3_A100_NOPOOL=1`,
  `BGE_M3_A100_NOCACHE=1`, `BGE_M3_A100_TIMING=1`, `BGE_M3_A100_TS=1`,
  `BGE_M3_A100_DUMP=<n>[,<m>...]`.
- Environment: SpacemiT K3, Bianbu 4.0.1 (kernel 6.18.3), riscv64,
  8× Spacemit X100 cores (RVV 1.0, VLEN=256) + 8× A100 intelligent cores,
  LLVM 24.0.0git (RuyiAI riscv fork), buddy-mlir built natively in user mode.

---

## Issue #888 Answer: BGE-M3 RAX Optimization on SpacemiT K3

### 1. Baseline (f32, 16 OpenMP threads, steady-state mode B median)

| seq | latency (ms) | stdev | seq/s | token/s | cold-start (s) | peak RSS (MB) | cos |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 19926.6 | 120 | 0.050 | 6.4 | 29.1 | 4365 | 1.000000 |
| 256 | 39573 | ~800 | 0.025 | 6.5 | 42.9 | 5399 | 1.000000 |
| 512 | 115658.7 | 6490 | 0.0086 | 4.4 | 128.4 | 8568 | 1.000000 |

Measured efficiency: ~3.9 GFLOPS at seq 128/256, ~2.7 GFLOPS at seq 512.

### 2. Same-hardware reference comparison (HF transformers on K3, 16 threads)

| seq | RAX baseline (ms) | HF reference (ms) | RAX/HF |
|---:|---:|---:|---:|
| 128 | 19926.6 | 24681.63 | **0.81x** |
| 256 | 39573 | 52744.08 | **0.75x** |
| 512 | 115658.7 | 110122.98 | 1.05x |

The RAX baseline already reaches/exceeds the native reference at seq 128/256.

### 3. Profiling of the RAX execution path

- **RVV instruction census** (seq128 baseline, `llvm-objdump`): 144 `vsetvli`,
  576 `vle32.v`, 288 `vse32.v`, 288 `vfmacc.vf` — every matmul is vectorized
  on the N axis only; the K axis is a scalar loop (GEMV-style codegen).
- **Thread scaling** (seq128): 1/4/8/16 threads = 49.27/32.83/20.85/20.32 s —
  saturation at 8 threads (2.36x), i.e. memory/serial sections bound.
- **Graph census**: 144 matmul, 240 materialized `linalg.transpose`
  (315M elements, 302M from weight transposes), 1063 `linalg.generic`
  (elementwise/softmax/attention heads), 146 reduce ops, 586
  `memref.expand_shape` (dynamic strides).
- **Bottleneck ranking**: scalar-strided memory streams + un-parallelized
  sections > weak matmul vectorization > weight transpose copies (4KB
  strided, L3-bound) > OpenMP fork/join overhead (5197 parallel regions).

### 4. Optimizations (before/after)

Negative results (kept as deliverables, each narrowed the search space):

| # | Experiment | Result |
|---|---|---|
| 1 | `matmul-vectorization` scalable + layout fix-ups | verifier crash (`expand_shape` stride invariant broken) |
| 2/3 | scalable / fixed `vector-size=16` `matmul-vectorization` | zero/invalid CLS |
| 4/5 | `-mcpu=spacemit-x100` / `+zvl256b -riscv-v-vector-bits-min=256` (backend only) | cos = 0.340539329, bit-identical — upstream RuyiAI LLVM miscompile under "VLEN>=256" assumption |
| 6 | IME direct link (`ime.vmadot`) | SIGILL in user mode |
| 7 | all 144 matmuls offloaded to A100 | K3 llc miscompile (layer inputs zeroed, cos -> 0.27) |

Positive results:

| Optimization | Effect |
|---|---|
| x86 pipeline fix: `-mcpu=native` for llc | matmul core 0 -> 576 AVX-512 FMA (cos = 1.0) |
| `-bge-m3-batchmatmul-transpose-b-vec` | attention batch matmuls vectorized, -10% (x86), cos = 1.0 |
| `-linalg-transpose-tile=tile-size=8` | OTHER bucket 199.6G -> 31.5G cycles (-84%), -12~13% (x86) |
| A100 offload via spine-runtime (M/3 rows to A100) | see table below |

End-to-end on K3 (steady-state median, f32):

| seq | RAX before (ms) | optimized chain (ms) | + A100 (ms) | total gain |
|---:|---:|---:|---:|---:|
| 128 | 19926.6 | 18831.43 | **8154.37** | **-59.1%** |
| 256 | 39573 | 37231.74 | **14679.94** | **-62.9%** |
| 512 | 115658.7 | 35661.71 | **32997.34** | **-71.5%** |

A100 design notes:
- IME direct link is not available; the GEMM runs on the A100 cores through
  spine-runtime (`libspert`) tile kernels, in parallel with an X100 pthread
  pool (float-acc 4-way unrolled dot, ~18-19 GFLOPS). A100 double-acc dot
  reaches ~8.4 GFLOPS (float-acc is pathological at 1.4 GFLOPS), so the A100
  takes M/3 rows to balance finish times.
- A 128-bit content-hash LRU caches B transposes (per-request activations
  reuse malloc'd addresses; a pointer-keyed cache returned stale data).
- The first `spert::Stream` construction costs ~30s; it is pre-initialized
  on a background thread at dlopen, and one-shot CLI processes skip spert.
  Cold start: 17.80 / 25.31 / 48.76 s (+A100 packages).
- Gain per seq differs because the optimized chain reaches ~17 GFLOPS at
  M=512 (near the X100 ceiling) but only ~5-7 GFLOPS at M=128/256 where
  the transpose-B vectorization degrades; the A100 fills exactly that gap.
- Correctness: cos = 1.000000000 everywhere (CLI, server, request-to-request).

### 5. Re-benchmark: RAX before / after vs reference

| seq | RAX before (ms) | RAX after (ms) | HF reference (ms) | after/HF |
|---:|---:|---:|---:|---:|
| 128 | 19926.6 | **8154.37** | 24681.63 | **0.33x** |
| 256 | 39573 | **14679.94** | 52744.08 | **0.28x** |
| 512 | 115658.7 | **32997.34** | 110122.98 | **0.30x** |

### 6. Precision & correctness

- Precision: f32 throughout (baseline and optimized). No quantization used.
- Correctness gate: cosine similarity of the 1024-dim embedding against the
  baseline, required > 0.999 after every major optimization. All final
  packages pass at cos = 1.000000000.

### 7. Conclusion

The performance target is met: the optimized RAX is **59-71% faster** than
its own K3 baseline and **~3x faster than the native HF reference** on the
same hardware across seq 128/256/512, with cos = 1.000000000.

### 8. Remaining gap & follow-up work

- Remaining gap: optimized matmuls run at ~20-24 GFLOPS vs the x86 HF
  reference at ~190 GFLOPS; the gap is RVV vectorization depth (scalar K
  axis) and the un-vectorized generic/attention sections, not the A100.
- Follow-ups:
  1. Fix `matmul-vectorization` for large-K GEMV shapes (or move to upstream
     `linalg->vector` with `vector.contract` + `vfmacc.vv` K-axis kernel).
  2. Fix the RuyiAI LLVM RISC-V backend numeric bug under the
     "VLEN >= 256" assumption (`-mcpu=spacemit-x100` / `+zvl256b` both
     produce bit-identical wrong embeddings).
  3. Vectorize the 1063 strided `linalg.generic` ops (materialize transpose
     views / explicit MLIR vectorization).
  4. FP16/BF16 weights (halve memory traffic; must pass the cos gate).
  5. A100 IME fp16/int8 (`ime.vfmadot`) via spine-runtime (issue #916).
