# EmbeddingGemma

[google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)
sentence embeddings, served through the `buddy-cli` / `.rax` runtime. The build
imports the PyTorch SentenceTransformer pipeline, compiles the fixed-shape
forward graph to MLIR, links `embeddinggemma_model.so`, and packs an
`embeddinggemma.rax` manifest plus an `InferenceRunner` plugin that `buddy-cli`
loads at run time.

The compiled kernel is the full SentenceTransformer pipeline traced as a single
AOT graph: Gemma3TextModel encoder -> mean pooling over the real tokens -> dense
768->3072 -> dense 3072->768 -> L2 normalization. The runner tokenizes the input
text and emits the L2-normalized 768-dim embedding vector.

## Prerequisites

- A built LLVM/MLIR and `buddy-mlir` (see the top-level [README](../../README.md)),
  configured with `-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON`.
- The Python environment that `buddy-mlir` was built against, with `torch`,
  `transformers`, `sentence_transformers`, and the Buddy Python frontend
  available for model import and HF golden generation. The packaged `.rax`
  has no Python dependency for inference.
- A local HuggingFace `google/embeddinggemma-300m` snapshot directory.

## Build

`embeddinggemma` is wired through `buddy_add_model(MODEL_KIND single_forward)`
with its own importer/manifest generator
(`codegen/import-embeddinggemma.py` /
`codegen/gen_embeddinggemma_manifest.py`). It is gated behind the
`BUDDY_BUILD_EMBEDDINGGEMMA_MODEL` CMake option:

```bash
cd buddy-mlir
source buddy-env/bin/activate

cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DBUDDY_BUILD_EMBEDDINGGEMMA_MODEL=ON \
  -DBUDDY_EMBEDDINGGEMMA_MODEL_PATH=/path/to/embeddinggemma-300m
cmake --build build --target embeddinggemma_rax
```

`-DBUDDY_EMBEDDINGGEMMA_MODEL_PATH` must point to a local HuggingFace-format
snapshot (it is staged into the build and forwarded to the importer via the
`EMBEDDINGGEMMA_MODEL_PATH` environment variable).

`BUDDY_EMBEDDINGGEMMA_SPEC` selects the fixed-shape spec. The default
`specs/f32.json` uses F32, `max_seq_len = 128`, and `batch_size = 1` when that
field is absent. Changing batch size or sequence length requires rebuilding
the model. For RISC-V builds, configure the toolchain and runtime using
[CrossCompilingRaxFile.md](../../docs/CrossCompilingRaxFile.md), then build
EmbeddingGemma with the CMake target above.

The build emits these artifacts under `build/models/embeddinggemma/`:

| File | Description |
| --- | --- |
| `embeddinggemma.rax` | Model manifest (embeds weights, `embeddinggemma_model.so`, `tokenizer.json`, runner) |
| `embeddinggemma_model.so` | Compiled MLIR kernels (exports `_mlir_ciface_forward`) |
| `embeddinggemma_runner.so` | `InferenceRunner` plugin loaded by `buddy-cli` |
| `arg0.data` | Flattened model weights (307,581,953 f32 elements, ~1.2 GB) |
| `tokenizer.json` | Gemma byte-level BPE tokenizer, staged from the local snapshot |

## Run

```bash
./build/bin/buddy-cli \
  --model ./build/models/embeddinggemma/embeddinggemma.rax \
  --prompt "The quick brown fox jumps over the lazy dog"
```

- `--prompt "<text>"` selects the text to embed.
- `--prompt-file <file>` supplies one text per line. For a compiled batch of
  size B, supply exactly B lines and pass `--batch-size B`. A batch is executed
  by one forward call; `--batch-size` must match the compiled model.
- `--no-stats` suppresses runner logs and prints only the embedding vector.

The output is a JSON array of 768 values for B=1, or B arrays of 768 values
for a larger batch. Every row is an L2-normalized embedding.

## Baseline measurements

On Linux, `benchmark.py` uses the RISC-V runner and GNU `time` to collect
embedding latency, sequences/s, tokens/s, batches/s, and peak process RSS.
It requires NumPy. Use it to measure baseline and optimized Buddy builds with
the same settings. Original HF/SentenceTransformer outputs serve as the
correctness golden for both builds. HF preparation may run offline on the build host;
performance measurements run through RISC-V/QEMU or on a RISC-V host.

Prepare the workload matrix once, using a new output directory. The supplied
corpus generates distinct texts for batch sizes 1, 2 and 4, at effective
lengths of 128, 256 and 512 tokens including BOS/EOS, with no padding.
`matrix.json` lists the generated specs and text files. The following example
selects B2/S128; repeat it with each matrix entry for the full baseline:

```bash
export EMBEDDINGGEMMA_MODEL_PATH=/path/to/embeddinggemma-300m
EG_WORKLOAD_DIR="$PWD/build/embeddinggemma-workloads"
python models/embeddinggemma/prepare_workloads.py \
  --model-path "$EMBEDDINGGEMMA_MODEL_PATH" \
  --output-dir "$EG_WORKLOAD_DIR"

EG_CASE=f32-b2-s128
EG_SPEC="$EG_WORKLOAD_DIR/$EG_CASE.json"
EG_REFERENCE_DIR="$EG_WORKLOAD_DIR/reference-$EG_CASE"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
python models/embeddinggemma/validate_accuracy.py --mode reference \
  --spec "$EG_SPEC" \
  --texts-file "$EG_WORKLOAD_DIR/$EG_CASE-texts.json" \
  --output-dir "$EG_REFERENCE_DIR"
```

Retain these inputs and golden outputs for both baseline and optimization
runs. Configure an existing RISC-V CMake build for this spec, build the model,
and save the baseline package before applying optimizations:

```bash
EG_BUILD_DIR=/path/to/configured-riscv-build
EG_BUDDY_CLI=/path/to/riscv-buddy-cli-qemu-wrapper
cmake -S . -B "$EG_BUILD_DIR" \
  -DBUDDY_BUILD_EMBEDDINGGEMMA_MODEL=ON \
  -DBUDDY_EMBEDDINGGEMMA_MODEL_PATH="$EMBEDDINGGEMMA_MODEL_PATH" \
  -DBUDDY_EMBEDDINGGEMMA_SPEC="$EG_SPEC"
cmake --build "$EG_BUILD_DIR" --target embeddinggemma_rax

EG_CONFIGURATION=baseline
EG_RAX="$EG_WORKLOAD_DIR/$EG_CASE-baseline.rax"
cp --no-clobber "$EG_BUILD_DIR/models/embeddinggemma/embeddinggemma.rax" "$EG_RAX"
EG_RESULTS_DIR="$EG_WORKLOAD_DIR/results/$EG_CASE"
```

The QEMU wrapper must execute the RISC-V `buddy-cli` with the matching sysroot,
runtime libraries and CPU settings, forwarding all arguments. Keep runner
stats enabled so the script can read the forward timing and token count.
Measure the saved package with:

```bash
python models/embeddinggemma/benchmark.py \
  --model "$EG_RAX" \
  --buddy-cli "$EG_BUDDY_CLI" \
  --execution riscv-qemu \
  --target 'QEMU 10.0.2: max,vlen=1024,elen=64,zfh=true,zvfh=true' \
  --configuration "$EG_CONFIGURATION" \
  --spec "$EG_SPEC" \
  --reference-dir "$EG_REFERENCE_DIR" \
  --threads 4 --warmup 1 --repeat 3 --tolerance 1e-5 \
  --timeout 3600 \
  --output-dir "$EG_RESULTS_DIR/$EG_CONFIGURATION"
```

After applying an optimization, rebuild the affected compiler components and
the model. Confirm the relevant import/lowering stages reran. Save the new
package, then repeat the same benchmark command with these variables:

```bash
cmake --build "$EG_BUILD_DIR" --target embeddinggemma_rax
EG_CONFIGURATION=optimized
EG_RAX="$EG_WORKLOAD_DIR/$EG_CASE-optimized.rax"
cp --no-clobber "$EG_BUILD_DIR/models/embeddinggemma/embeddinggemma.rax" "$EG_RAX"
```

For frontend Python changes, rebuild the main `build` tree's Python packages
and remove the generated
`$EG_BUILD_DIR/models/embeddinggemma/.buddy_import_done` stamp before rebuilding
the model so the changed frontend is used for a fresh import.

Use fresh package filenames and result directories for each new candidate;
the baseline package remains fixed. For a same-precision comparison, keep
the weights, input texts, B/S, threads, CPU affinity, QEMU CPU configuration,
warmup and repeat counts identical. `--target` records the actual configuration
used by the caller. On a RISC-V host, use `--execution riscv-native` and pass
the native RISC-V `buddy-cli`. The script sets `OMP_NUM_THREADS` and disables
dynamic OpenMP thread counts; CPU affinity is inherited from the caller.

Warmup rounds prepare the extracted payload cache and are excluded from the
statistics. Each request starts a new process and reloads the model, so the
embedding latency includes first-call initialization and is not resident-model
steady-state latency. It measures the full compiled forward, including pooling
and normalization, and excludes loading and tokenization. Latency covers the
entire batch. Throughput uses the same forward timing:

```text
batches/s   = 1000 / embedding_latency_ms
sequences/s = batch_size * batches/s
tokens/s    = sum(valid_tokens) * batches/s
```

`valid_tokens` counts attention-mask entries equal to one, including BOS/EOS
and excluding padding. For the prepared full-length workloads, its sum is
`batch_size * sequence_length`.
Peak RSS covers the process's entire lifetime. Under QEMU this includes the
host emulator, weights and temporary data, and is not K3 hardware memory use.

`report.json` records the baseline/optimized configuration, precision or
quantization variant from the spec, RISC-V execution target, thread count,
batch size, sequence length, measurement scopes, and validation status. Each
sample includes raw measurements and latency/throughput mean, median, min,
max and population standard deviation, plus maximum peak RSS. QEMU times
describe emulator execution rather than A100 hardware performance.
Raw logs and embeddings are kept under
`runs/<round>/`; `warmup/` holds the unmeasured runs. The extracted payloads are
cached under `payload/`. After timing, every measured round is checked by
`validate_accuracy.compare_outputs`; a failed request, missing measurement,
timeout, or numerical mismatch returns a nonzero exit status. A timeout also
terminates the runner process group. No Git metadata or hashes are collected.

Accept a performance result only when `validation_passed` is true. The F32
commands above use cosine similarity > 0.99 and maximum absolute error < 1e-5
for every embedding row. A saved round can also be checked independently:

```bash
python models/embeddinggemma/validate_accuracy.py --mode compare \
  --reference-dir "$EG_REFERENCE_DIR" \
  --buddy-output-dir "$EG_RESULTS_DIR/$EG_CONFIGURATION/runs/000" \
  --tolerance 1e-5
```

Compare `samples[].embedding_latency_ms.median`, throughput medians and
`peak_rss_mib_max` in the baseline and optimized `report.json` files; inspect
the raw runs and standard deviation alongside them. Latency speedup is
`baseline_median_ms / optimized_median_ms`. This baseline uses F32. Future
precision changes need implementation support, matching spec metadata and
accuracy acceptance before performance is accepted; record and distinguish
different precisions explicitly.

## Notes

- The `.rax` and other artifacts live in the **build** directory
  (`build/models/embeddinggemma/`), not in the source tree.
- The tokenizer used by the runner is a pure-C++ reimplementation
  (`EmbeddinggemmaTokenizer.h`) of the checkpoint's Gemma byte-level BPE over
  the staged `tokenizer.json`; sequences are wrapped in `<bos>` / `<eos>` and
  padded to `max_seq_len` with `<pad>`.
