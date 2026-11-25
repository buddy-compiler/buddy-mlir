# EmbeddingGemma

Buddy compiles [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)
into a fixed-shape `.rax` for `buddy-cli`. One forward runs the Gemma3 encoder,
masked mean pooling, two dense layers, and L2 normalization. It returns one
768-dimensional embedding per input.

## Build and run

Start with a configured Buddy build with Python packages enabled. Use its Python
environment with `torch`, `transformers`, and `sentence_transformers`, and a
local Hugging Face snapshot of the model. See the [Buddy build instructions](../../README.md).
Run the following commands from the repository root:

```bash
export EMBEDDINGGEMMA_MODEL_PATH=/path/to/embeddinggemma-300m
python3 tools/buddy-codegen/build_model.py \
  --spec models/embeddinggemma/specs/f32.json \
  --build-dir build --local-model "$EMBEDDINGGEMMA_MODEL_PATH"
cmake --build build --target buddy-cli

./build/bin/buddy-cli \
  --model ./build/models/embeddinggemma/embeddinggemma.rax \
  --prompt "The quick brown fox jumps over the lazy dog"
```

The RAX package includes the runner, compiled model, weights, and tokenizer.
Inference needs no Python.

## Configuration

[The default F32 spec](specs/f32.json) compiles batch size 1 and sequence length
128. Change `batch_size` / `max_seq_len` and rebuild with `--spec` for another
shape. For direct CMake builds, select the spec with `BUDDY_EMBEDDINGGEMMA_SPEC`;
the build target is `embeddinggemma_rax`.

For a compiled batch of size B, pass `--prompt-file` with exactly B input
lines and `--batch-size B`. Use `--no-stats` for JSON output only; keep stats
enabled when benchmarking.

Both checked-in specs set `num_threads` to 4; CMake defaults to 1 when the
field is absent. This count is compiled into the model's OpenMP regions.
Changing `OMP_NUM_THREADS` or benchmark `--threads` does not override it.
To change it, update the spec and rebuild. Match benchmark `--threads` and
the native HF thread count to this value.

## SpacemiT K3

Use the [cross-compilation setup](../../docs/CrossCompilingRaxFile.md) and the
standard model target:

```bash
make riscv-model-embeddinggemma-k3_fp16_comp3 \
  RISCV_MODEL_LOCAL="$EMBEDDINGGEMMA_MODEL_PATH"
make riscv-cli
```

The default outputs are
`build-riscv/embeddinggemma-k3_fp16_comp3/models/embeddinggemma/embeddinggemma.rax`
and `build-cross-rv/bin/buddy-cli`. Copy both to the board, then run:

```bash
OMP_PROC_BIND=FALSE sh -c \
  'echo 0 > /proc/set_ai_thread &&
   exec taskset -c 8,10,12,14 ./buddy-cli \
     --model embeddinggemma.rax --prompt "hello world"'
```

Adjust the CPU mask to the assigned A100 cores. For direct CMake builds,
select [the IME spec](specs/k3_fp16_comp3.json), set
`BUDDY_RISCV_ENABLE_ZFH_ZVFH=ON` and `BUDDY_RISCV_VLEN=1024`.
IME requires `batch_size * max_seq_len` divisible by 8. An F32 spec may set
`"target_cpu": "spacemit-a100"` to select A100 LLVM scheduling.

## Implementation

- **IME.** All 168 encoder attention and feed-forward Linear projections use
  compensated FP16 `high*high + 2^-11 * (low*high + high*low)` with F32
  accumulation and output. Weights are packed offline. Use the model library and
  weights from the same build because F32 and IME have different weight layouts.
- **RVV and lowering.** Use BLIS-style MatMul and RVV attention BatchMatMul;
  fold transposes, flatten elementwise loops, and release temporary buffers
  after their last use. GELU uses `polynomial-approximate-math=ops=tanh`. The IME
  spec adds fused encoder RMSNorm and Softmax vectorization with
  `approximate-vector-exp`. `tanh-approx.mlir` only provides shape-specific
  scheduling. Attention QK/PV, pooling, both final dense layers, and final L2
  normalization remain F32.
- **Runner.** Integer-ID BPE looks up symbol pairs while preserving vocabulary
  IDs and merge order. A read-only `mmap` of `arg0.data` avoids the heap copy.
  Keep the weight file intact during inference and publish replacements
  atomically, since truncating the mapped file in place can cause `SIGBUS`.

`embeddinggemma_ime.py` retains the high/low split, scaling, three-product
correction, and packed layout that `lower-linalg-to-ime` does not implement.
It reuses the existing `k3_w4` matrix/RMSNorm bodies and shared `kernels_ime`
lowering pipeline, including `lower-ime` and A100/VLEN validation. Its
dependency on internal `k3_w4` helpers remains.

## Benchmark and validate

Evaluation tools and the full procedure are archived in the
[reproducibility snapshot](https://github.com/buddy-compiler/buddy-mlir/blob/720b6ecf8c526208a623c502105bea8d5015af64/models/embeddinggemma/README.md#benchmark-and-validate).
When testing the current checkout, use those tools with the **current source
and specs**. Pass the current spec's absolute path through `prepare_workloads.py`
`--spec`; archived default specs predate compile-time thread selection.
Build the RAX from the current checkout, even if the tools run in a separate
worktree. To reproduce historical measurements, use their recorded revisions.

1. Generate workloads for B=1/2/4 and S=128/256/512 with `prepare_workloads.py`.
   Generate golden outputs on the build host using `validate_accuracy.py` with
   `--mode reference`, using the original HF SentenceTransformer in F32 and
   matching tokenization, padding, truncation, and fixed shapes.
2. Build a RAX for each spec, then run `benchmark.py` with native `buddy-cli`
   on K3. Match inputs, CPU placement, and thread counts across configurations;
   label F32 and `fp16_comp3` explicitly. The tools need Python 3.10+, NumPy,
   and GNU `time`; a native HF comparison additionally needs PyTorch,
   Transformers, SentenceTransformers, and `threadpoolctl`.
3. Accept results only when `report.json` has `validation_passed: true`:
   every embedding row must have cosine similarity > 0.99 and maximum
   absolute error < 1e-5 against the HF golden. Logs and embeddings are in `runs/`.

Each measured run starts a fresh process and times the first complete forward,
excluding tokenization and model setup. Mapped-weight page faults may occur
inside that interval. Peak RSS covers the whole process. Throughput derives
from forward latency; tokens/s counts valid tokens including BOS/EOS.

## Recorded K3 results

The final comparison covered batches 1, 2, and 4 at 128, 256, and 512 tokens on
four A100 cores with four threads. Each configuration had three measured runs
per shape with warm weight pages and a fresh process per run. The optimized RAX
(`51b8ba5a`, `fp16_comp3`) was 23.24–38.24× faster on the first forward than
the F32 RAX baseline (`9c8e15c7`) and 7.11–19.03× faster than native HF F32.
Peak process RSS fell 71.6–93.5% relative to the RAX baseline. All 81 measured
runs passed the embedding gate. The optimized RAX reached a maximum absolute
error of 5.66e-7 and a minimum cosine similarity of 0.9999999999976 against HF F32.
These speedups include a change in precision. Full workload tables and
profiling are in [PR #1070](https://github.com/buddy-compiler/buddy-mlir/pull/1070);
stage-by-stage measurements remain in the reproducibility snapshot above.
