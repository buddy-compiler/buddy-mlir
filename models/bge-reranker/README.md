# BGE-Reranker-v2-M3

[BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) — a
cross-encoder reranker built on XLMRobertaForSequenceClassification — served
through the `buddy-cli` / `.rax` runtime. The build imports the PyTorch
encoder, compiles the fixed-shape forward graph to MLIR, links
`bge_reranker_model.so`, and packs a `bge_reranker.rax` manifest plus an
`InferenceRunner` plugin that `buddy-cli` loads at run time.

Given a query/document pair, the runner emits a single scalar relevance logit
(the score used to re-rank candidate documents). The pair is tokenized as one
wrapped sequence: `[<s>, query..., </s>, document..., </s>]`.

## Prerequisites

- A built LLVM/MLIR and `buddy-mlir` (see the top-level [README](../../README.md)),
  configured with `-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON`.
- The Python environment that `buddy-mlir` was built against, with `torch`,
  `transformers`, and the Buddy Python frontend available. This is only needed
  at **build time** to import the PyTorch model; the packaged `.rax` has no
  Python dependency at run time (see Notes).
- A local HuggingFace `BAAI/bge-reranker-v2-m3` snapshot directory. The
  repository intentionally does not provide a default local path.

## Build

`bge-reranker` is wired through `buddy_add_model(MODEL_KIND single_forward)` with
its own importer/manifest generator (`codegen/import-bge-reranker.py` /
`codegen/gen_bge_reranker_manifest.py`, the same pluggable
`IMPORT_SCRIPT`/`MANIFEST_SCRIPT` mechanism ColBERTv2 and BGE-M3 use). It is
gated behind the `BUDDY_BUILD_BGE_RERANKER_MODEL` CMake option, so build it
directly with CMake rather than `tools/buddy-codegen/build_model.py` (whose
`model_family` whitelist does not include `bge_reranker`):

```bash
cd buddy-mlir
conda activate buddy

cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DBUDDY_BUILD_BGE_RERANKER_MODEL=ON \
  -DBUDDY_BGE_RERANKER_MODEL_PATH=/path/to/bge-reranker-v2-m3
cmake --build build --target bge_reranker_rax
```

`-DBUDDY_BGE_RERANKER_MODEL_PATH` must point to a local HuggingFace-format
snapshot directory (it is staged into the build as
`BUDDY_BGE_RERANKER_MODEL_PATH` and forwarded to the importer via the
`BGE_RERANKER_MODEL_PATH` environment variable; the snapshot's `tokenizer.json`
is also staged as the run-time vocab asset).

The build emits these artifacts under `build/models/bge-reranker/`:

| File | Description |
| --- | --- |
| `bge_reranker.rax` | Model manifest (embeds weights, `bge_reranker_model.so`, `tokenizer.json`, runner) |
| `bge_reranker_model.so` | Compiled MLIR kernels (exports `_mlir_ciface_forward`) |
| `bge_reranker_runner.so` | `InferenceRunner` plugin loaded by `buddy-cli` |
| `arg0.data` | Flattened encoder weights |
| `tokenizer.json` | XLM-R SentencePiece-Unigram tokenizer, staged from the local snapshot |

## Run

```bash
# Single string, split on the last " <sep> ":
./build/bin/buddy-cli \
  --model ./build/models/bge-reranker/bge_reranker.rax \
  --prompt "what is panda? <sep> A panda is a bear native to China."

# Or a two-line prompt file (line 1 = query, line 2 = document):
printf 'what is panda?\nA panda is a bear native to China.\n' > pair.txt
./build/bin/buddy-cli \
  --model ./build/models/bge-reranker/bge_reranker.rax \
  --prompt-file pair.txt
```

The output is the scalar relevance logit (the score passed to `logits[0, 0]` of
the classification head).

## Notes

- The `.rax` and other artifacts live in the **build** directory
  (`build/models/bge-reranker/`), not in the source tree. Use the `build/`
  prefix in the `--model` path.
- Tokenization is a pure C++ reimplementation of XLM-R's SentencePiece-Unigram
  tokenizer (same encoder family as BGE-M3, so `BgeRerankerTokenizer.h` follows
  `BgeM3Tokenizer.h`), reading `tokenizer.json` directly. There is no
  Python/`transformers` dependency at run time.
- The imported graph uses `max_seq_len = 512` and
  `max_position_embeddings = 8194` from `models/bge-reranker/specs/f32.json`.
  Change the spec (or pass a different spec via `--spec`) to re-target a
  different fixed sequence length.
- The MLIR forward ABI is
  `forward(weights: memref<567755777xf32>, position_ids: memref<8194xi64>,
  input_ids: memref<1x512xi64>, attention_mask: memref<1x512xi64>) ->
  memref<1x1xf32>`. The runner fills `position_ids` with `arange(0, 8194)` and
  passes the pair as `[<s>, query, </s>, document, </s>]` right-padded to
  `max_seq_len`.
- For a quick accuracy check, compare the `buddy-cli` output against
  `AutoModelForSequenceClassification.from_pretrained(<local snapshot>)` using
  the same tokenizer settings: `tokenizer(query, document, truncation=True,
  max_length=512)` and score = `float(logits[0, 0])`.
