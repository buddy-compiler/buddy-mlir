# EmbeddingGemma

Buddy compiles [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)
into a fixed-shape `.rax` for `buddy-cli`. One forward runs the Gemma3 encoder,
masked mean pooling, two dense layers, and L2 normalization. It returns one
768-dimensional embedding per input.

## Build and run

Start with a configured Buddy build with Python packages enabled. Use its Python
environment with `torch`, `transformers`, and `sentence_transformers`, and a
local Hugging Face snapshot of the model. See the [Buddy build instructions](../../README.md).

```bash
export EMBEDDINGGEMMA_MODEL_PATH=/path/to/embeddinggemma-300m
cmake -S . -B build \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DBUDDY_BUILD_EMBEDDINGGEMMA_MODEL=ON \
  -DBUDDY_EMBEDDINGGEMMA_MODEL_PATH="$EMBEDDINGGEMMA_MODEL_PATH"
cmake --build build --target buddy-cli embeddinggemma_rax

./build/bin/buddy-cli \
  --model ./build/models/embeddinggemma/embeddinggemma.rax \
  --prompt "The quick brown fox jumps over the lazy dog"
```

[The default F32 spec](specs/f32.json) compiles batch size 1 and sequence length
128. Set `BUDDY_EMBEDDINGGEMMA_SPEC` to an absolute path to select a different
spec, and rebuild for each shape. For a compiled batch of size B, pass
`--prompt-file` with exactly B input lines and `--batch-size B`. Use
`--no-stats` for JSON output only and keep stats enabled when benchmarking.
The RAX package includes the runner, compiled model, weights, and tokenizer.
Inference needs no Python.

## SpacemiT K3

From the repository root, build the K3 IME variant with the standard model target.

```bash
make riscv-model-embeddinggemma-k3_fp16_comp3 \
  RISCV_MODEL_LOCAL="$EMBEDDINGGEMMA_MODEL_PATH"
```

For a direct CMake build, follow the
[cross-compilation guide](../../docs/CrossCompilingRaxFile.md). For IME, set
`BUDDY_RISCV_ENABLE_ZFH_ZVFH=ON` and `BUDDY_RISCV_VLEN=1024`, and select
[`specs/k3_fp16_comp3.json`](specs/k3_fp16_comp3.json) or its shape-specific
equivalent. This path requires `batch_size * max_seq_len` to be divisible by 8.
For an F32 build, `"target_cpu": "spacemit-a100"` optionally selects A100 LLVM
scheduling.

### Optimizations

- **IME.** All 168 encoder attention and feed-forward Linear projections use
  compensated FP16 `high*high + 2^-11 * (low*high + high*low)` with F32
  accumulation and output. Offline weight packing and N-major tile traversal
  let consecutive tiles reuse packed weight blocks. Use the model library and
  weights from the same build because F32 and IME have different weight layouts.
- **RVV.** The pipeline vectorizes attention QK/PV BatchMatMul and GELU's
  polynomial `tanh`. The IME spec also enables fused encoder RMSNorm and
  vectorized Softmax `exp`. Mean pooling, both final dense layers, and final L2
  normalization use F32.
- **Lowering.** Pretranspose F32 Linear weights offline and use BLIS-style
  MatMul tiling. Canonicalize loops before OpenMP lowering. Fold Q/K transposes
  into elementwise operations, flatten elementwise loops, and release temporary
  buffers after their last use. Caller-allocated IME outputs expose buffer
  ownership to the compiler.
- **Runner.** Integer-ID BPE looks up symbol pairs while preserving vocabulary
  IDs and merge order. A read-only `mmap` of `arg0.data` avoids the heap copy.
  Keep the weight file intact during inference and publish replacements
  atomically, since truncating the mapped file in place can cause `SIGBUS`.

### Shared lowering and compensated IME

Math approximation reuses `polynomial-approximate-math=ops=tanh` and
`approximate-vector-exp`. `tanh-approx.mlir` only specifies shape-specific
tiling, vectorization, and loop flattening.

`embeddinggemma_ime.py` retains the high/low split, scaling, three-product
correction, and packed layout that `lower-linalg-to-ime` does not implement.
It reuses the existing `k3_w4` matrix/RMSNorm bodies and shared `kernels_ime`
lowering pipeline, including `lower-ime` and A100/VLEN validation. Its
dependency on internal `k3_w4` helpers remains.

## Benchmark and validate

The benchmark and validation scripts are kept in the reproducibility snapshot
[`720b6ecf`](https://github.com/buddy-compiler/buddy-mlir/tree/720b6ecf8c526208a623c502105bea8d5015af64/models/embeddinggemma).
The snapshot contains the scripts, workload corpus, specs, and the benchmark
instructions used for the results below. Check it out separately before
running the commands in this section.

```bash
git fetch https://github.com/buddy-compiler/buddy-mlir.git \
  pull/1070/head
git worktree add ../embeddinggemma-897-repro \
  720b6ecf8c526208a623c502105bea8d5015af64
cd ../embeddinggemma-897-repro
```

### Prepare inputs on the build host

[The workload generator](https://github.com/buddy-compiler/buddy-mlir/blob/720b6ecf8c526208a623c502105bea8d5015af64/models/embeddinggemma/prepare_workloads.py) creates separate specs and
full-length inputs for batches 1, 2, and 4 at 128, 256, and 512 tokens.
`matrix.json` lists the cases. This example selects B2/S128.

```bash
workloads="$PWD/build/embeddinggemma-workloads"
case_name=f32-b2-s128
python models/embeddinggemma/prepare_workloads.py \
  --model-path "$EMBEDDINGGEMMA_MODEL_PATH" --output-dir "$workloads"
python models/embeddinggemma/prepare_workloads.py \
  --model-path "$EMBEDDINGGEMMA_MODEL_PATH" --output-dir "$workloads/ime" \
  --spec models/embeddinggemma/specs/k3_fp16_comp3.json
python models/embeddinggemma/validate_accuracy.py --mode reference \
  --spec "$workloads/$case_name.json" \
  --texts-file "$workloads/$case_name-texts.json" \
  --output-dir "$workloads/reference-$case_name"
```

Build each RAX from its own revision and matching spec. The generator uses
the `f32-` filename prefix for both sets. Check the spec's `variant` field
to identify precision. For the IME build, use the spec in `$workloads/ime`.

```bash
riscv_build_dir=/path/to/configured-riscv-build
case_spec="$workloads/$case_name.json"
cmake -S . -B "$riscv_build_dir" \
  -DBUDDY_BUILD_EMBEDDINGGEMMA_MODEL=ON \
  -DBUDDY_EMBEDDINGGEMMA_MODEL_PATH="$EMBEDDINGGEMMA_MODEL_PATH" \
  -DBUDDY_EMBEDDINGGEMMA_SPEC="$case_spec"
cmake --build "$riscv_build_dir" --target embeddinggemma_rax
```

Save the packaged RAX from `models/embeddinggemma/embeddinggemma.rax` under
the build directory as `$case_name-baseline.rax` or `$case_name-optimized.rax`.

### Run on the K3

Copy the RAX files, matching specs, HF golden directory, native `buddy-cli`,
and the archived benchmark scripts to the board. The benchmark
requires Python 3.10+, NumPy, and GNU `time`. Use a filesystem with room for the
extracted RAX payloads.

```bash
workloads=/path/to/embeddinggemma-workloads
case_name=f32-b2-s128
cli=/path/to/riscv-buddy-cli

run_k3() {
  OMP_PROC_BIND=FALSE sh -c \
    'echo 0 > /proc/set_ai_thread && exec taskset -c 8,10,12,14 "$@"' sh "$@"
}

bench=(python3 models/embeddinggemma/benchmark.py
  --target 'SpacemiT K3 A100; VLEN=1024'
  --reference-dir "$workloads/reference-$case_name"
  --threads 4 --warmup 1 --repeat 3)

run_k3 "${bench[@]}" --configuration baseline --buddy-cli "$cli" \
  --model "$workloads/$case_name-baseline.rax" \
  --spec "$workloads/$case_name.json" --output-dir "$workloads/results/$case_name/baseline"

run_k3 "${bench[@]}" --configuration optimized --buddy-cli "$cli" \
  --model "$workloads/$case_name-optimized.rax" \
  --spec "$workloads/ime/$case_name.json" --output-dir "$workloads/results/$case_name/optimized"

run_k3 "${bench[@]}" --backend hf --configuration reference \
  --model /path/to/local-sentence-transformer \
  --hf-python /path/to/native-hf-python \
  --spec "$workloads/$case_name.json" --output-dir "$workloads/results/$case_name/reference"
```

Adjust the example to the assigned A100 cores and keep the same placement for
all variants. The HF backend uses the original SentenceTransformer in F32
with KV cache disabled. Its native Python needs PyTorch, Transformers,
SentenceTransformers, and `threadpoolctl`.

Each measurement starts a new process and times the first forward for the
whole batch. Tokenization and model setup occur before the timer. Accesses to
mapped weights can fault in pages during the forward. Process peak RSS covers
the entire request. The script derives batches/s and sequences/s from forward
latency and counts valid tokens including BOS/EOS for tokens/s.

Use matching inputs and separate output directories, and alternate the runs.
Keep the Git revision, spec, RAX checksum, CPU placement, and software
versions with the reports. Compare the reported medians and ranges and label
F32 versus `fp16_comp3` explicitly. Accept a benchmark only if
`report.json` has `validation_passed: true`. The default gate requires
cosine similarity > 0.99 and maximum absolute error < 1e-5 for every embedding
row. Raw logs and embeddings are in `runs/`.

## Final K3 results

The final comparison covered batches 1, 2, and 4 at 128, 256, and 512 tokens on
four A100 cores with four threads. Each configuration had three measured runs
per shape with warm weight pages and a fresh process per run. The optimized RAX
(`51b8ba5a`, `fp16_comp3`) was 23.24–38.24× faster on the first forward than
the F32 RAX baseline (`9c8e15c7`) and 7.11–19.03× faster than native HF F32.
Peak process RSS fell 71.6–93.5% relative to the RAX baseline. All 81 measured
runs passed the embedding gate. The optimized RAX reached a maximum absolute
error of 5.66e-7 and a minimum cosine similarity of 0.9999999999976 against HF F32.

## Selected K3 measurements

These stage comparisons used four A100 cores and four threads. Each row
compares adjacent implementations with its own measurement scope. The results
predate the latest upstream merge.

| Change | Precision | Workload | Median forward |
| --- | --- | --- | ---: |
| Use affine lowering for remaining Linalg ops | F32 | B1/S128, B4/S512 | 10.27%, 11.08% lower |
| Move Attention Linear projections to IME | FFN-only → all encoder Linear `fp16_comp3` | B1/S128 | 870.17 → 545.73 ms |
| Extend RVV RMSNorm from 97 to 145 norms | `fp16_comp3` | B1/S128 | 473.22 → 456.54 ms |
| Reuse weight blocks with N-major tile traversal | `fp16_comp3` | B1/S128 | 456.23 → 408.03 ms |
| Vectorize GELU `tanh` | `fp16_comp3` | B1/S128 | 411.03 → 382.36 ms |
| Flatten binary elementwise operations | `fp16_comp3` | B2/S128, B4/S512 | 1018.65 → 880.10 ms, 11776.95 → 10468.50 ms |
| Flatten scalar multiplications | `fp16_comp3` | B2/S128, B4/S512 | 886.28 → 842.08 ms, 10547.55 → 10337.70 ms |
| Vectorize Softmax `exp` | `fp16_comp3` | B1/S128, B2/S128, B4/S512 | 433.11 → 425.48 ms, 1026.45 → 1006.56 ms, 11476.38 → 10591.00 ms |

The scalar-multiply ranges overlapped at B4/S512. The Softmax comparison timed
the compiled forward on pretokenized inputs. Full-length and padded inputs
passed the HF gate.

Integer-ID BPE reduced tokenizer-only encoding from 12.16 s to 0.327 s for a
512-token input. In a separate B4/S512 CLI comparison with cached payloads,
total request time fell from 67.15 s to 16.36 s and peak RSS from 1354 to
1314 MiB. Exported embeddings were bit-identical and passed the HF gate.

The read-only weight mapping comparison used `fp16_comp3` packages that
differed only in the runner, with `21ed5d40` as the baseline. Each cell is the
median of two runs per variant in A-B-A-B order, using extracted RAX payloads.
Weight cache residency was verified before each request.

| Workload | Weight cache | CLI elapsed (s) | First forward (s) | Peak RSS (MiB) | Sampled peak PSS (MiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| B1/S128 | Warm | 5.725 → 4.365 | 0.387 → 0.581 | 1314 → 631 | 1311 → 476 |
| B1/S128 | Cold | 6.060 → 4.525 | 0.378 → 0.693 | 1314 → 551 | 1311 → 513 |
| B4/S512 | Warm | 16.125 → 14.640 | 9.474 → 9.499 | 1314 → 727 | 1312 → 720 |
| B4/S512 | Cold | 16.385 → 15.195 | 9.324 → 10.053 | 1315 → 570 | 1312 → 557 |

Peak RSS fell 45–58% and total CLI time fell 7–25%. Demand paging raised
first-forward latency. Exported embeddings stayed bit-identical, with
maximum absolute error of 4.99e-7 and minimum cosine similarity of
0.999999999998 against HF F32.
