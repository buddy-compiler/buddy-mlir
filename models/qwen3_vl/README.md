Vision-language OCR (image-text-to-text) with
[Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct),
served through the `buddy-cli` / `.rax` runtime. The build imports the PyTorch
model, compiles its vision encoder (ViT + DeepStack) and dense Qwen3 text decoder
(interleaved MRoPE) to MLIR, links them as shared libraries, and packs a
`qwen3_vl.rax` manifest plus an `InferenceRunner` plugin that `buddy-cli` loads at
run time. It uses the shared `buddy_add_model` entry with a Qwen3-VL-specific
multimodal build kind because the model emits separate vision and decoder shims.

By default the decoder is built as **KV prefill + single-token decode** with
**panel-packed decode GEMV** weights (`BUDDY_QWEN3_VL_KV_DECODE=ON`). Pass
`-DBUDDY_QWEN3_VL_KV_DECODE=OFF` for the legacy full-sequence recompute decoder.
KV mode is incompatible with `BUDDY_MODEL_LAYER_PARTITION=ON`.

## Prerequisites

- A built LLVM/MLIR and `buddy-mlir` (see the top-level [README](../../README.md)),
  configured through `tools/buddy-codegen/build_model.py`.
- The Python environment that `buddy-mlir` was built against (the conda `buddy`
  env: Python 3.10, `torch`, `transformers` with native `qwen3_vl` support,
  `pillow`, `numpy`).
- A local HuggingFace `Qwen3-VL-2B-Instruct` snapshot directory.
- For SpacemiT / RVV packages: a configured RISC-V cross tree
  (`IS_RVV_CROSSCOMPILE=ON`), host `buddy-opt`, and RISC-V `libomp` /
  `libmlir_c_runner_utils`.

## Build (host)

Use the same `tools/buddy-codegen/build_model.py` entry point as the other
packaged models (DeepSeek R1, Whisper). It imports the model, compiles the vision
and decoder kernels, builds the runner plugin, and stages `qwen3_vl.rax`. Qwen3-VL
requires a local HuggingFace snapshot, so `--local-model` is mandatory:

```bash
cd buddy-mlir
python3 tools/buddy-codegen/build_model.py \
  --spec models/qwen3_vl/specs/instruct_2b.json \
  --build-dir build \
  --local-model /path/to/Qwen3-VL-2B-Instruct
```

This assumes LLVM/MLIR and `buddy-mlir` are already built per the top-level
[README](../../README.md) (the `build/` directory configured against the in-tree
`llvm/build`).

The first KV import (vision + prefill + decode graphs) is the slow step. Later
rebuilds that only change lowering/shim can keep
`artifacts/.buddy_import_done_kv` and recompile objects / relink.

Artifacts under `<build-dir>/models/qwen3_vl/`:

| File | Description |
| --- | --- |
| `qwen3_vl.rax` | Model manifest (`model_name`, runner library, vocab) |
| `qwen3_vl_runner.so` | `InferenceRunner` plugin loaded by `buddy-cli` |
| `vision_shim.so` / `decoder_shim.so` | Staged shared libs (also under `artifacts/`) |
| `artifacts/vision/vision_shim.so` | Compiled vision encoder |
| `artifacts/decoder_rt/decoder_shim.so` | Decoder: KV prefill+decode (default) or legacy full forward |
| `vision_weights.data` / `decoder_weights.data` | Vision + prefill (or legacy) weights |
| `decoder_decode_weights.data` | Packed decode GEMV weights (KV mode only) |
| `embed_table.bin` / `vocab.txt` | Tied embedding table / tokenizer vocab |

## Build (RVV cross / SpacemiT)

Configure a cross build with `IS_RVV_CROSSCOMPILE=ON` (same toolchain / omp /
runner-utils setup as other RVV models). Then:

```bash
# If midend matmul passes changed, refresh host buddy-opt first:
cmake --build <host-build> --target buddy-opt -j$(nproc)

cmake -S . -B <cross-build> \
  -DBUDDY_QWEN3_VL_KV_DECODE=ON \
  -DBUDDY_QWEN3_VL_NUM_THREADS=8 \
  -DBUDDY_QWEN3_VL_MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct
  # plus your existing IS_RVV_CROSSCOMPILE / toolchain cache entries

cmake --build <cross-build> --target qwen3_vl_rax -j$(nproc)
```

Sync the package directory to the board (exclude `artifacts/`, CMake intermediates).
On SpacemiT X100, matmul stays on the **RVV FP16** path (`+zvl256b`); do not
enable XSMTIME `vfmadot` (SIGILL on board). Board-specific timings and package
names live under [`reports/`](reports/).

## Run

Host:

```bash
./build/bin/buddy-cli \
  --model ./build/models/qwen3_vl/qwen3_vl.rax \
  --image ./models/qwen3_vl/test_text.png \
  --prompt "Read all the text in the image."
```

On-device (after syncing the cross package next to a RISC-V `buddy-cli`):

```bash
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export LD_LIBRARY_PATH="$PWD:${LD_LIBRARY_PATH:-}"

/path/to/buddy-cli \
  --model ./qwen3_vl.rax \
  --runner-so ./qwen3_vl_runner.so \
  --image /path/to/test_text.png \
  --prompt "Read all the text in the image." \
  --max-tokens 32 \
  --temperature 0.0 \
  --cpus 0-7
```

With the default KV package, the runner auto-loads prefill/decode entry points
and `decoder_decode_weights.data` from the package / rax payload — no
`QWEN3_VL_DECODER_SO` / `QWEN3_VL_DECODE_WEIGHTS` overrides are required.
Logs should include `using KV prefill/decode decoder shim` and
`using packed decode weights from ...`.

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

Qwen3-VL packages also contain `qwen3_vl_serving.so`, referenced by the
manifest `serving_library` attribute. Start the resident HTTP model with:

```bash
./build/bin/buddy-server \
  --model ./build/models/qwen3_vl/qwen3_vl.rax \
  --host 127.0.0.1 --port 8080
```

The server accepts one local image through `image_path`, or an OpenAI-style
chat content array:

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Read all text in the image."},
      {"type": "image_url", "image_url": {"url": "file:/tmp/input.png"}}
    ]
  }],
  "stream": false
}
```

Only local paths/file URIs and one fixed-grid image are currently supported;
omitting the image uses the packaged test image.
Remote URLs, data URIs, in-memory image bytes, multiple images, video and
non-greedy sampling are rejected or unsupported. Decoder RoPE tables are
packaged for the build-time prompt length; requests with another tokenized
prompt length are rejected rather than producing silently incorrect results.
String stop sequences are not applied; `stop_token_ids` and the model EOS
ids are honored.
The HTTP resident path is implemented separately from `Qwen3VLRunner`, so
the existing buddy-cli interface and output behavior remain unchanged.

## Notes

- The `.rax` and other artifacts live in `<build-dir>/models/qwen3_vl/`.
- Runtime image decoding, resize and patchification use the pure-C++
  `ImagePreprocess.h` implementation; Python is only used while building the
  package and its fixed positional constants.
- Default greedy decode is **KV prefill once + per-token decode** (packed GEMV
  on decode linears). Legacy full-sequence recompute:
  `-DBUDDY_QWEN3_VL_KV_DECODE=OFF`. Image resolution is pinned at import time
  (no dynamic resolution / crop modes yet).
