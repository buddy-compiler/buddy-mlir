# PaliGemma-3B-224

PaliGemma-3B-224 vision-language model
([google/paligemma-3b-mix-224](https://huggingface.co/google/paligemma-3b-mix-224)),
served through the `buddy-cli` / `.rax` runtime. The build imports the PyTorch
`PaliGemmaForConditionalGeneration` forward (SigLIP vision tower + Gemma
language model + 1152→2048 projector), compiles the fixed-shape forward graph
to MLIR, links `paligemma_model.so`, and packs a `paligemma.rax` manifest plus
an `InferenceRunner` plugin that `buddy-cli` loads at run time.

The runner emits the language-model logits for the last sequence position
(top-5 token ids) and a summary of the projected image features. The traced
forward is a full VLM step over a fixed 1 x 3 x 224 x 224 zero image and a
280-token text sequence (256 `<image>` tokens + 24 text tokens).

## Architecture

- Vision: SigLIP, 27 layers, hidden = 1152, 16 heads, 256 image patches.
- Text: Gemma, 18 layers, hidden = 2048, 8 heads, 1 KV head.
- Projector: `Linear(1152 -> 2048)`.
- Vocab: 257216; image token id: 257152.

## Prerequisites

- A built LLVM/MLIR and `buddy-mlir` (see the top-level [README](../../README.md)),
  configured with `-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON`.
- The Python environment that `buddy-mlir` was built against, with `torch`,
  `transformers`, and the Buddy Python frontend available. This is only needed
  at **build time** to import the PyTorch model; the packaged `.rax` has no
  Python dependency at run time (see Notes).
- A local HuggingFace `google/paligemma-3b-mix-224` snapshot directory. The
  repository intentionally does not provide a default local path.
- Enough RAM for the float32 model (~12 GB) plus the flattened weight dump
  (~11.7 GB) during import.

## Build

`paligemma` is wired through `buddy_add_model(MODEL_KIND single_forward)` with
its own importer/manifest generator (`codegen/import-paligemma.py` /
`codegen/gen_paligemma_manifest.py`, the same pluggable
`IMPORT_SCRIPT`/`MANIFEST_SCRIPT` mechanism ColBERTv2, BGE-M3 and Whisper use).
It is gated behind the `BUDDY_BUILD_PALIGEMMA_MODEL` CMake option, so build it
directly with CMake rather than `tools/buddy-codegen/build_model.py`:

```bash
cd buddy-mlir
conda activate buddy

cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DBUDDY_BUILD_PALIGEMMA_MODEL=ON \
  -DBUDDY_PALIGEMMA_MODEL_PATH=/path/to/paligemma-3b-mix-224
cmake --build build --target paligemma_rax
```

`-DBUDDY_PALIGEMMA_MODEL_PATH` must point to a local HuggingFace-format
`google/paligemma-3b-mix-224` snapshot directory (it is staged into the build
as `BUDDY_PALIGEMMA_MODEL_PATH` and forwarded to the importer via the
`PALIGEMMA_MODEL_PATH` environment variable).

The build emits these artifacts under `build/models/paligemma/`:

| File | Description |
| --- | --- |
| `paligemma.rax` | Model manifest (embeds weights, `paligemma_model.so`, tokenizer files, runner) |
| `paligemma_model.so` | Compiled MLIR kernels (exports `_mlir_ciface_forward`) |
| `paligemma_runner.so` | `InferenceRunner` plugin loaded by `buddy-cli` |
| `arg0.data` | Flattened f32 model weights |
| `tokenizer.json` | Gemma tokenizer, staged from the local snapshot |

## Run

```bash
./build/bin/buddy-cli \
  --model ./build/models/paligemma/paligemma.rax
```

- `--no-stats` suppresses runner logs and prints only the JSON output line.
- The runner builds a fixed, deterministic input batch (zero image, fixed
  token ids); `--prompt` / `--image` are not consumed by this runner.

The output is a JSON line with the logits shape and the top-5 token ids /
logit values at the last sequence position.

## Forward ABI

The compiled kernel is:

```
forward(weights        : memref<2923466608 x f32>,     // arg0.data
        position_ids   : memref<256 x i64>,            // 0..255 (vision patches)
        input_ids      : memref<1 x 280 x i64>,
        pixel_values   : memref<1 x 3 x 224 x 224 x f32>,
        attention_mask : memref<1 x 280 x i64>)
  -> (image_features  : memref<1 x 256 x 2048 x f32>,
      logits          : memref<1 x 280 x 257216 x f32>)
```

The two results are packed into one C struct, so
`_mlir_ciface_forward(ForwardResults*, weights*, position_ids*, input_ids*,
pixel_values*, attention_mask*)`. The `position_ids` memref is the packed i64
leaf of the trace (the SigLIP patch position buffer), filled `0..255` by the
runner; it is *not* part of `arg0.data` — only the f32 params are flattened
into `arg0.data`.

## Notes

- The `.rax` and other artifacts live in the **build** directory
  (`build/models/paligemma/`), not in the source tree. Use the `build/` prefix
  in the `--model` path.
- Input construction is deliberately simple and deterministic: `pixel_values`
  is a zero `1 x 3 x 224 x 224` image and `input_ids` is `256` copies of the
  image token (`257152`) followed by `24` copies of token `1`, with a
  full-ones attention mask. The kernel is shape-specialized, not
  value-specialized, so any token ids within the fixed shapes are valid. A
  real image input (`--image`) is not implemented because it would require a
  runtime SigLIP preprocessing pipeline (PIL/torchvision + resize/normalize);
  see the comment block in `PaligemmaRunner.cpp`.
- The importer traces the full VLM forward as one graph using the original
  PR's `get_placeholder_mask` monkey-patch (the stock method performs a
  data-dependent token-count check that Dynamo cannot trace) and fuses with
  `simply_fuse` only.
- For a quick accuracy check, compare the runner's last-token logits against
  `PaliGemmaForConditionalGeneration` reference logits from the same dummy
  inputs (`validate_accuracy.py` in the original PR sketches the comparison).
