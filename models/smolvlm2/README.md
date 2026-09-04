# SmolVLM2 (text-LM)

SmolVLM2-500M-Instruct text decoder with
[HuggingFaceTB/SmolVLM2-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Instruct),
served through the `buddy-cli` / `.rax` runtime. The build imports the PyTorch
**text decoder** (`SmolVLMModel.text_model`, a 32-layer Llama3 decoder, hidden
= 960), compiles the fixed-shape forward graph to MLIR, links
`smolvlm2_model.so`, and packs a `smolvlm2.rax` manifest plus an
`InferenceRunner` plugin that `buddy-cli` loads at run time.

The runner byte-level-BPE-tokenizes the prompt (against the staged
`tokenizer.json`) and emits the last real token's 960-dimensional
`last_hidden_state` vector along with summary statistics.

## Scope / limitation

The full VLM forward takes `input_ids`, `attention_mask`, **and** `pixel_values`
and does **not** trace as a single TorchDynamo graph: the vision tower
(visuomotor projector) and the language model are separated by `torch.compile`
graph breaks, so the AOT importer would emit 12 sub-graphs. Per the model
integration policy, the importer therefore targets the **text-only path**:
`m.text_model(...)`, which traces as exactly one graph (291 parameters,
361,944,032 f32 elements). Image inputs are **not** supported; the `.rax` is a
text-decoder model.

## Prerequisites

- A built LLVM/MLIR and `buddy-mlir` (see the top-level [README](../../README.md)),
  configured with `-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON`.
- The Python environment that `buddy-mlir` was built against, with `torch`,
  `transformers`, and the Buddy Python frontend available. This is only needed
  at **build time** to import the PyTorch model; the packaged `.rax` has no
  Python dependency at run time (see Notes).
- A local HuggingFace `HuggingFaceTB/SmolVLM2-500M-Instruct` snapshot
  directory. The repository intentionally does not provide a default local
  path.

## Build

`smolvlm2` is wired through `buddy_add_model(MODEL_KIND single_forward)` with
its own importer/manifest generator (`codegen/import-smolvlm2.py` /
`codegen/gen_smolvlm2_manifest.py`, the same pluggable
`IMPORT_SCRIPT`/`MANIFEST_SCRIPT` mechanism ColBERTv2 and BGE-M3 use). It is
gated behind the `BUDDY_BUILD_SMOLVLM2_MODEL` CMake option (add the matching
`add_subdirectory(smolvlm2)` under `if(BUDDY_BUILD_SMOLVLM2_MODEL)` in
`models/CMakeLists.txt`), so build it directly with CMake rather than
`tools/buddy-codegen/build_model.py` (whose `model_family` whitelist does not
include `smolvlm2`):

```bash
cd buddy-mlir
conda activate buddy

cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DBUDDY_BUILD_SMOLVLM2_MODEL=ON \
  -DBUDDY_SMOLVLM2_MODEL_PATH=/path/to/HuggingFaceTB/SmolVLM2-500M-Instruct
cmake --build build --target smolvlm2_rax
```

`-DBUDDY_SMOLVLM2_MODEL_PATH` must point to a local HuggingFace-format
SmolVLM2 snapshot directory (it is staged into the build as
`BUDDY_SMOLVLM2_MODEL_PATH` and forwarded to the importer via the
`SMOLVLM2_MODEL_PATH` environment variable).

The build emits these artifacts under `build/models/smolvlm2/`:

| File | Description |
| --- | --- |
| `smolvlm2.rax` | Model manifest (embeds weights, `smolvlm2_model.so`, `tokenizer.json`, runner) |
| `smolvlm2_model.so` | Compiled MLIR kernels (exports `_mlir_ciface_forward`) |
| `smolvlm2_runner.so` | `InferenceRunner` plugin loaded by `buddy-cli` |
| `arg0.data` | Flattened text-decoder weights |
| `tokenizer.json` | GPT2 byte-level BPE tokenizer, staged from the local snapshot |

## Run

```bash
./build/bin/buddy-cli \
  --model ./build/models/smolvlm2/smolvlm2.rax \
  --prompt "Tell me about buddy-mlir"
```

- `--prompt "<text>"` selects the text to process.
- `--no-stats` suppresses runner logs and prints only the JSON result.

The output is a JSON object with the final real token's `last_hidden_state`
960-dimensional vector plus `sum` / `mean` / `l2_norm` / `argmax_dim`
statistics. The prompt is wrapped in the standard chat scaffold
`<|im_start|>user\n<prompt><|im_end|>\n<|im_start|>assistant\n` before
tokenization.

## Notes

- The `.rax` and other artifacts live in the **build** directory
  (`build/models/smolvlm2/`), not in the source tree. Use the `build/` prefix
  in the `--model` path.
- Tokenization is a pure C++ reimplementation of GPT-2 byte-level BPE
  (byte encoder, merge table, and added/special tokens parsed from
  `tokenizer.json` via `llvm::json`) built into `Smolvlm2Runner.cpp`. There is
  no Python/`transformers` dependency at run time, so a `.rax` built once can
  be copied to another machine (same architecture) and run with just `buddy-cli`
  and that single file — no source tree, no Python environment.
- The imported graph uses `max_seq_len = 64` and `hidden_size = 960` from
  `models/smolvlm2/specs/f32.json`. Change the spec (or pass a different spec
  via `--spec`) to re-target a different fixed sequence length; the runner
  reads `max_seq_len` / `hidden_size` / `params_size` from the manifest module
  attributes.
- The MLIR forward ABI is
  `forward(weights: memref<params_size x f32>, input_ids: memref<1 x 64 x i64>,
  attention_mask: memref<1 x 64 x i64>) -> (last_hidden_state:
  memref<1 x 64 x 960 x f32>)`. The runner fills `attention_mask` with 1 for
  real tokens and 0 for padding (`<|im_end|>`, id 2) and emits the last masked
  position's hidden-state vector.
