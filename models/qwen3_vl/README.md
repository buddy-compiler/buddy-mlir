Vision-language OCR (image-text-to-text) with
[Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct),
served through the `buddy-cli` / `.rax` runtime. The build imports the PyTorch
model, compiles its vision encoder (ViT + DeepStack) and dense Qwen3 text decoder
(interleaved MRoPE) to MLIR, links them as shared libraries, and packs a
`qwen3_vl.rax` manifest plus an `InferenceRunner` plugin that `buddy-cli` loads at
run time. It uses the shared `buddy_add_model` entry with a Qwen3-VL-specific
multimodal build kind because the model emits separate vision and decoder shims.

## Prerequisites

- A built LLVM/MLIR and `buddy-mlir` (see the top-level [README](../../README.md)),
  configured through `tools/buddy-codegen/build_model.py`.
- The Python environment that `buddy-mlir` was built against (the conda `buddy`
  env: Python 3.10, `torch`, `transformers` with native `qwen3_vl` support,
  `pillow`, `numpy`).
- A local HuggingFace `Qwen3-VL-2B-Instruct` snapshot directory.

## Build

Use the same `tools/buddy-codegen/build_model.py` entry point as the other
packaged models. It configures CMake, imports the model, compiles the kernels,
builds the runner plugin, and stages `qwen3_vl.rax`:

```bash
cd buddy-mlir
conda activate buddy
export BUDDY_MLIR_BUILD_DIR=$PWD/build
export LLVM_MLIR_BUILD_DIR=$PWD/llvm/build
export PYTHONPATH=${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}

python3 tools/buddy-codegen/build_model.py \
  --spec models/qwen3_vl/specs/instruct_2b.json \
  --build-dir build \
  --local-model /path/to/Qwen3-VL-2B-Instruct
```

The import and the two MLIR compiles run as part of the target and are the slow
part (a few minutes each). The build emits these artifacts under
`build/models/qwen3_vl/`:

| File | Description |
| --- | --- |
| `qwen3_vl.rax` | Model manifest (`model_name`, runner library, vocab) |
| `qwen3_vl_runner.so` | `InferenceRunner` plugin loaded by `buddy-cli` |
| `artifacts/vision/vision_shim.so` | Compiled vision encoder kernels (flat-C ABI) |
| `artifacts/decoder_rt/decoder_shim.so` | Compiled Qwen3 decoder kernels (flat-C ABI) |
| `vision_weights.data` / `decoder_weights.data` | External weight blobs |
| `embed_table.bin` / `vocab.txt` | Tied embedding table / tokenizer vocab |

## Run

```bash
./build/bin/buddy-cli \
  --model ./build/models/qwen3_vl/qwen3_vl.rax \
  --image ./models/qwen3_vl/test_text.png \
  --prompt "Read all the text in the image."
```

- `--image <path>` selects any input image; it is resized to the pinned canonical
  resolution (grid `[1,14,28]`, 98 image tokens) before encoding.
- `--prompt "<text>"` is the instruction for the model (e.g. "Read all the text in
  the image."). The runner re-runs preprocessing for each query.

Expected output for the sample image:

```
[Qwen3-VL OCR] Buddy MLIR
Qwen-3-VL 0.0
2026
```

## Notes

- The `.rax` and other artifacts live in the **build** directory
  (`build/models/qwen3_vl/`), not in the source tree. Use the `build/` prefix in
  the `--model` path.
- Per-query preprocessing (image resize/patchify, prompt tokenization, MRoPE
  positions) is delegated to the HuggingFace processor via a small Python helper
  the runner invokes; the model forward itself runs entirely on the compiled
  kernels. A pure-C++ preprocessing path is future work.
- Greedy decode uses fixed-max-length recompute (no KV cache), and the image
  resolution is pinned at import time (no dynamic resolution / crop modes yet).
- The first run loads the HuggingFace processor once and memory-maps the weight
  blobs (~8 GB), so expect ~1–2 minutes end-to-end on CPU.
