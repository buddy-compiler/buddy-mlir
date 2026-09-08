# PaddleOCR-VL

PaddleOCR-VL-0.9B OCR vision-language model with
[lvyufeng/PaddleOCR-VL-0.9B](https://huggingface.co/lvyufeng/PaddleOCR-VL-0.9B),
served through the `buddy-cli` / `.rax` runtime. The build imports the full
SigLIP vision encoder + projector + ERNIE-style decoder + LM head from the
HuggingFace remote code, compiles the fixed-shape OCR forward graph to MLIR,
links `paddleocr_model.so`, and packs a `paddleocr.rax` manifest plus an
`InferenceRunner` plugin that `buddy-cli` loads at run time.

The runner emits the **last-token logits** (a 103424-dim vector over the OCR
vocabulary) together with the top-5 token hypotheses, i.e. a single-shot OCR
forward over a deterministic fixed-shape input.

## Prerequisites

- A built LLVM/MLIR and `buddy-mlir` (see the top-level [README](../../README.md)),
  configured with `-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON`.
- The Python environment that `buddy-mlir` was built against, with `torch`,
  `transformers`, `numpy`, `einops`, and the Buddy Python frontend available.
  This is only needed at **build time** to import the PyTorch model; the
  packaged `.rax` has no Python dependency at run time (see Notes).
- A local HuggingFace `lvyufeng/PaddleOCR-VL-0.9B` snapshot directory
  (including the remote-code files `modeling_paddleocr_vl.py`,
  `configuration_paddleocr_vl.py`, `image_processing.py`,
  `processing_paddleocr_vl.py`). The repository intentionally does not provide
  a default local path.

## Build

`paddleocr` is wired through `buddy_add_model(MODEL_KIND single_forward)` with
its own importer/manifest generator (`codegen/import-paddleocr.py` /
`codegen/gen_paddleocr_manifest.py`, the same pluggable
`IMPORT_SCRIPT`/`MANIFEST_SCRIPT` mechanism ColBERTv2 and Whisper use). It is
gated behind the `BUDDY_BUILD_PADDLEOCR_MODEL` CMake option, so build it
directly with CMake rather than `tools/buddy-codegen/build_model.py` (whose
`model_family` whitelist does not include `paddleocr`):

```bash
cd buddy-mlir
conda activate buddy

cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DBUDDY_BUILD_PADDLEOCR_MODEL=ON \
  -DBUDDY_PADDLEOCR_MODEL_PATH=/path/to/PaddleOCR-VL-0.9B
cmake --build build --target paddleocr_rax
```

`-DBUDDY_PADDLEOCR_MODEL_PATH` must point to a local HuggingFace-format
PaddleOCR-VL-0.9B snapshot directory (it is staged into the build as
`BUDDY_PADDLEOCR_MODEL_PATH` and forwarded to the importer via the
`PADDLEOCR_MODEL_PATH` environment variable).

The build emits these artifacts under `build/models/paddleocr/`:

| File | Description |
| --- | --- |
| `paddleocr.rax` | Model manifest (embeds weights, `paddleocr_model.so`, `tokenizer.json`, runner) |
| `paddleocr_model.so` | Compiled MLIR kernels (exports `_mlir_ciface_forward`) |
| `paddleocr_runner.so` | `InferenceRunner` plugin loaded by `buddy-cli` |
| `arg0.data` | Flattened model weights (905,601,730 f32 elements) |
| `tokenizer.json` | Qwen-style tokenizer, staged from the local snapshot |

## Run

```bash
./build/bin/buddy-cli \
  --model ./build/models/paddleocr/paddleocr.rax \
  --prompt "identify the text in the image"
```

- `--prompt "<text>"` seeds the 10 text-token slots of the OCR sequence.
- `--no-stats` suppresses runner logs and prints only the JSON output.

The output is a JSON object with the logits shape, summary statistics of the
last-token logits vector, and the top-5 token hypotheses.

## Notes

- The `.rax` and other artifacts live in the **build** directory
  (`build/models/paddleocr/`), not in the source tree. Use the `build/` prefix
  in the `--model` path.
- **Fixed input, no image decoding.** The compiled forward takes a fixed
  982-token sequence: 972 image tokens followed by 10 text tokens. The runner
  fills `pixel_values` with a fixed-size zero buffer (3888x3x14x14) — it does
  NOT decode `--image`. This is deliberate: it keeps the integration
  deterministic and reproducible without a JPEG decoder in the runtime.
- **Text encoding is a simplified placeholder.** The 10 text slots are filled
  from `--prompt` with a byte→token-id mapping (falling back to pad id 1 when
  no prompt is given). This is NOT a full Qwen byte-level BPE tokenizer, and
  outputs for arbitrary text will not match a Python reference run that uses
  the real tokenizer. The model was traced with these fixed text positions so
  the graph, weights, and manifest are self-consistent.
- **Remote-code patching.** PaddleOCR-VL loads via `trust_remote_code`. The
  importer stages a patched copy of the snapshot under
  `/tmp/paddleocr_model_stage` (symlinks + a patched `modeling_paddleocr_vl.py`)
  so the vision+language path is Dynamo-fullgraph traceable; the original
  repo files are never modified.
- The MLIR forward ABI is
  `forward(weights, input_ids, pixel_values, attention_mask, position_ids) ->
  logits` with shapes `[905601730]`, `[1, 982]`, `[3888, 3, 14, 14]`,
  `[1, 982]`, `[3, 1, 982]` and `[1, 982, 103424]` respectively.
- For a quick sanity check, compare the last-token logits against
  `AutoModel.from_pretrained(...)` fed the same dummy inputs (972 image tokens
  + 10 pad tokens, zero `pixel_values`, zero 3-D `position_ids`). The
  validated reference path is the original PR's `validate_accuracy.py`.
