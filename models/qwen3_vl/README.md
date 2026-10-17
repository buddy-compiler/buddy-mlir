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

## Run with buddy-server

Qwen3-VL packages also contain `qwen3_vl_server.so`, referenced by the
manifest `model_library` attribute. Start the resident HTTP model with:

```bash
./build/bin/buddy-server \
  --model ./build/models/qwen3_vl/qwen3_vl.rax \
  --host 127.0.0.1 --port 8080
```

The server exposes completion/chat, raw `/tokenize` and `/detokenize`, health,
and model listing. It accepts one local image through `image_path`, or an
OpenAI-style chat content array. Put the image before text and use the
canonical packaged instruction:

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "models/qwen3_vl/test_text.png"}},
      {"type": "text", "text": "Read all the text in the image."}
    ]
  }],
  "stream": false
}
```

Only local paths/hostless `file:` URIs and one fixed-grid image are supported.
Text-before-image, interleaved image/text, multiple image occurrences, remote
URLs, data URIs, in-memory image bytes, and video are rejected. A top-level
image, or the bundled fallback when no image is supplied, attaches to the first
user message. Default chat preserves system/user/assistant roles, empty message
contents, and multiple turns; a configured Buddy JSON template is cached when
loading and rendered once. Unsupported roles are rejected.

Decoder RoPE tables and image positions are packaged for the build-time prompt
layout. Both the complete encoded prompt length and all 98 image-pad positions
must match those resources. A differently shaped request requires rebuilt
positional constants. Completion expects an already rendered prompt with
exactly one `<|vision_start|>` + 98 `<|image_pad|>` + `<|vision_end|>` block;
it adds no template, image block, or BOS. Chat adds that block itself before
the selected user message's text. The example above uses the default build-time
instruction but still requires matching positional resources.

HTTP generation is greedy only: omit `temperature`, `top_k`, `top_p`, `min_p`,
`repeat_penalty`, `repeat_last_n`, and `seed`. Explicit non-null sampling fields
are rejected even at default values, including `temperature:0`. Budget aliases
use `n_predict > max_completion_tokens > max_tokens`; 0 evaluates the prompt,
-1 runs until stop/context, and positive values count newly selected tokens.
The shared serving controller applies string/token stops, usage accounting,
UTF-8 streaming, and cancellation with cleanup for subsequent requests.

Raw tokenization adds no chat/image scaffold or automatic BOS and never
truncates. It uses deterministic DP segmentation of the packaged byte
vocabulary; canonical BPE merge rules, regex splitting, and NFC normalization
are absent. It therefore does not certify canonical or legacy token ID
equivalence. Raw detokenization validates every ID, joins bytes including
special tokens and content after EOS, and repairs UTF-8 once.

See the [Qwen serving guide](../../tools/buddy-server/compatibility/qwen3-vl.md)
for codec options, fully rendered completion examples, template/image limits,
and validation boundaries. Build server/runtime/resident plugins from matching
sources for the v2 `ChatMessage.imagesAfterText` DTO layout. Default runtime
tests use synthetic RAX resources and dynamically loaded fixture kernels,
without exercising model weights or model arithmetic. The resident path uses
different tokenizer and request controls from `Qwen3VLRunner`, preserving
the existing buddy-cli interface and output behavior.

## Notes

- The `.rax` and other artifacts live in `build/models/qwen3_vl/`.
- Runtime image decoding, resize and patchification use the pure-C++
  `ImagePreprocess.h` implementation; Python is only used while building the
  package and its fixed positional constants.
- Greedy decode uses fixed-max-length recompute (no KV cache), and the image
  resolution is pinned at import time (no dynamic resolution / crop modes yet).
