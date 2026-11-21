# Buddy Compiler BGE-M3 Example

## Introduction

This example cross-compiles the BGE-M3 dense embedding model (single-forward
path, ~568M params, f32) to a self-contained riscv64 RAX package and
optimizes it for the SpacemiT K3 (issue #888). The whole lowering pipeline
lives in `CMakeLists.txt`: the build host cross-compiles the `.rax`, and the
K3 only executes it.

Optimizations live in `midend/lib/Conversion/MatMulOptimization/`:

| Pass | What it does |
|---|---|
| `-bge-m3-batchmatmul-transpose-b-vec` | Vectorizes the 48 attention batch matmuls (Q@Kᵀ / scores@V) with guarded transpose-B mapping |
| `-linalg-transpose-tile` | Tiled, cache-friendly lowering of the 240 materialized `linalg.transpose` copies |
| `-bge-m3-matmul-a100` | Replaces 143/144 f32 `linalg.matmul` with a call into the hybrid X100/A100 GEMM kernel (`a100_kernel.cpp`) |

Every optimization is gated by the embedding cosine gate (cos > 0.999).

## Directory layout

```
├── CMakeLists.txt         # cross-compile the riscv64 .rax (import → .o → .so → .rax)
├── cmake/rvv-cross.cmake  # cross toolchain + runtime-deps packing helpers
├── a100_kernel.cpp        # hybrid GEMM kernel: X100 pthread pool + A100 spine-runtime
├── README.md              # this file (issue #888 answer)
└── .gitignore             # generated artifacts + local test scripts
```

## How to run

The riscv64 `.rax` is cross-compiled on the build host; the K3 only executes
it. A local helper `cos.py` (git-ignored) runs the embedding cosine gate.

0. Prerequisites: Python `torch`/`transformers` + the buddy Python packages
   on the build host, a local HuggingFace-format BGE-M3 snapshot (the
   git-ignored `export/download_bge_m3.py` downloads it), and a riscv64 cross
   sysroot / cross MLIR-LLVM / libomp / spine-runtime SDK.

1. Cross-compile the riscv64 `.rax`:

```bash
$ cmake -S . -B build \
    -DBUDDY_BGEM3_CROSS=ON \
    -DBUDDY_BGEM3_A100=ON \
    -DBUDDY_BGE_M3_MODEL_PATH=/path/to/bge-m3-snapshot
$ ninja -C build bge-m3-optimized
# output: build/examples/BuddyBgeM3/bge_m3.rax (riscv64, self-contained:
# weights + libomp + libmlir_c_runner_utils + libspert)
```

`BUDDY_BGEM3_RISCV_SYSROOT` / `BUDDY_BGEM3_RISCV_GCC_TOOLCHAIN` default to
`~/k3-sysroot` / `~/k3-sysroot/usr`; `SPINE_RUNTIME_SDK` defaults to
`~/spine-sdk/spine-runtime.riscv64.0.7.1`. `BUDDY_BGEM3_SPEC` selects
`base`/`seq128`/`seq256`/`seq512`; `A100_PASS_OPTS` keeps the first matmul on
X100 (`-bge-m3-a100-start=1`) to dodge a K3 llc miscompile.

2. Run on the K3:

```bash
$ scp build/examples/BuddyBgeM3/bge_m3.rax k3:~/v2/bgem3/
$ ssh k3 'cd ~/v2/bgem3 && ~/v2/bin/buddy-cli \
    --model bge_m3.rax \
    --prompt "The quick brown fox jumps over the lazy dog." --no-stats'
```

Notes:
- Kernel knobs (env): `BGE_M3_A100_DISABLE=1`, `BGE_M3_A100_NOPOOL=1`,
  `BGE_M3_A100_NOCACHE=1`, `BGE_M3_A100_TIMING=1`, `BGE_M3_A100_TS=1`,
  `BGE_M3_A100_DUMP=<n>[,<m>...]`.
- Environment: SpacemiT K3, Bianbu 4.0.1 (kernel 6.18.3), riscv64,
  8× Spacemit X100 cores (RVV 1.0, VLEN=256) + 8× A100 intelligent cores,
  LLVM 24.0.0git (RuyiAI riscv fork).

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
- **Op-to-item mapping**: the 144 `linalg.matmul` carry the linear layers
  (Q/K/V, attention output, FFN); the 48 attention batch matmuls carry
  Q@K^T and scores@V; the 1063 `linalg.generic` carry softmax, the head
  reshapes and the elementwise FFN; the embedding lookup is a gather from
  the token-embedding table; pooling is the CLS slice (first token of
  `last_hidden_state`); L2 normalization runs in the runner after the
  forward; the 240 `linalg.transpose` + 302M transposed weight elements
  cover layout/memory movement; the 5197 OpenMP regions are the runtime
  overhead.

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

| seq | RAX before (ms) | RVV-only (ms) | + A100 (ms) | total gain |
|---:|---:|---:|---:|---:|
| 128 | 19926.6 | 18071 | **8187** | **-58.9%** |
| 256 | 39573 | 35934 | **15527** | **-60.8%** |
| 512 | 115658.7 | 102942 | **34141** | **-70.5%** |

All RAX numbers are re-measured from the CMake cross-compiled .rax
(host LLVM 24 + board sysroot). The +A100 column matches the native A100
results (8187 / 15527 / 34141 vs 8154 / 14680 / 32997 ms); the RVV-only
column matches the native seq128/256 (18071 / 35934 vs 18831 / 37232 ms)
but is ~3x slower at seq512 (102942 vs 35662 ms), which points at the
host LLVM 24 llc for the large subgraph rather than the passes.

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
| 128 | 19926.6 | **8187** | 24681.63 | **0.33x** |
| 256 | 39573 | **15527** | 52744.08 | **0.29x** |
| 512 | 115658.7 | **34141** | 110122.98 | **0.31x** |

### 6. Precision & correctness

- Precision: f32 throughout (baseline and optimized). No quantization used;
  the K3's A100 GEMM kernels accumulate in f32, so the A100 path keeps the
  same precision as the RVV path (cos = 1.000000000).
- Quantization is out of scope for this PR. The generic weight-only
  INT8/INT4 graph passes (`frontend/Python/graph/transform/quantization`)
  are not wired into the BGE-M3 importer, and the LLM-only `k3_w4` int4
  kernels do not apply to this BERT-style encoder. Adding an INT8/INT4
  BGE-M3 path (with a cos gate) is tracked as follow-up work.
- Batch: single request (batch = 1). The BGE-M3 dense path in this example
  runs one sequence per forward, so throughput is reported as sequences/s
  and tokens/s (§1) rather than batch throughput.
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
