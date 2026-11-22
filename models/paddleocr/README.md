# PaddleOCR-VL

PaddleOCR-VL-0.9B OCR vision-language model with
[lvyufeng/PaddleOCR-VL-0.9B](https://huggingface.co/lvyufeng/PaddleOCR-VL-0.9B),
served through the `buddy-cli` / `.rax` runtime. The build imports the full
SigLIP vision encoder + projector + ERNIE-style decoder + LM head from the
HuggingFace remote code, compiles the fixed-shape OCR forward graph to MLIR,
links `paddleocr_model.so`, and packs a `paddleocr.rax` manifest plus an
`InferenceRunner` plugin that `buddy-cli` loads at run time.

The FP16 K3 variant (`specs/f16_ocr_sample_last_token.json`) runs full
end-to-end OCR text generation (Prefill + static KV-cache Decode + C++
`tokenizer.json` decoding) directly from a local PNG/JPEG image. The original
F32 spec remains available for single-shot last-token logits.

`specs/f16_ocr_sample_last_token.json` is the K3 development variant. It
traces a real OCR sample and the official `OCR:` prompt with FP16 weights and
activations (plus 82 FP32 auxiliary parameters), exports Prefill and Decode
graphs that share the same weight buffers (`arg0.data` / `arg1.data`) with zero
duplication, and decodes the generated token sequence into text in C++.

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

Select the FP16 variant with
`-DBUDDY_PADDLEOCR_SPEC=$PWD/models/paddleocr/specs/f16_ocr_sample_last_token.json`.
It produces `arg0.data` (FP16) and `arg1.data` (FP32); both are needed by the
compiled forward ABI. K3 cross builds also require the RISC-V sysroot, target
OpenMP/MLIR runtime libraries, and `BUDDY_RISCV_VLEN=1024` for A100. Pass `-DBUDDY_PADDLEOCR_K3_IME=ON` (with `BUDDY_RISCV_VLEN=1024` and
`precision=f16`) to enable SpacemiT K3 A100 FP16 IME lowering
(`codegen/lower_to_obj.sh`): fused Flash-Attention with A100 FP16 IME,
zero-copy transposed-weight folding and unit-stride packing, transposed-weight
`M=1` FP32 vector GEMV, Vision patch-embedding conv-to-IME lowering, and
PR #997's elementwise/SiLU/VIR vector optimizations. On `K3-002` (A100 CPUs
8–15, 8 threads, after `echo 0 > /proc/set_ai_thread`) with direct PNG input
(`test_text.png`, `--prompt "OCR:"`), this reduces `paddleocr_model.so` from
`64M` to `38M` (including both Prefill and Decode), lowers Prefill latency from
`525.7s` (FP16 RVV) to `25.04s` (`~21x` speedup), and runs KV-cache Decode at
`3.11 tok/s` (`5.78s` for 18 decode steps, `30.82s` total), outputting the
exact OCR text matching PyTorch FP16 `model.generate`
(`Buddy MLIR\nQwen3-VL OCR\n2026`).

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
| `arg1.data` | FP32 auxiliary weights for the FP16 variant (82 elements) |
| `tokenizer.json` | Qwen-style tokenizer, staged from the local snapshot |

## Run

```bash
./build/bin/buddy-cli \
  --model ./build/models/paddleocr/paddleocr.rax \
  --prompt "identify the text in the image"
```

- `--prompt "<text>"` seeds the 10 text-token slots of the OCR sequence.
- `--no-stats` suppresses runner logs and prints only the JSON output.

With the FP16 sample variant, pass `--image <local PNG/JPEG> --prompt "OCR:"`.
The runner decodes and resizes the image in C++, runs Prefill + KV-cache Decode,
and decodes the generated token IDs via `tokenizer.json` to emit the recognized
OCR text (both human-readable summary on stderr and JSON on stdout).

## Notes

- The `.rax` and other artifacts live in the **build** directory
  (`build/models/paddleocr/`), not in the source tree. Use the `build/` prefix
  in the `--model` path.
- **Original F32 fixed input.** The original compiled forward takes a fixed
  982-token sequence: 972 image tokens followed by 10 text tokens. The runner
  fills `pixel_values` with a fixed-size zero buffer (3888x3x14x14). The
  FP16 sample variant supports real local images as described above.
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
  `[1, 982]`, `[3, 1, 982]` and `[1, 982, 103424]` respectively for the
  original F32 spec. The FP16 sample variant adds a second FP32 weight memref
  of 82 elements and emits `[1, 1, 103424]` FP16 logits.
- For a quick sanity check, compare the last-token logits against
  `AutoModel.from_pretrained(...)` fed the same dummy inputs (972 image tokens
  + 10 pad tokens, zero `pixel_values`, zero 3-D `position_ids`). For the
  sample variants, `benchmarks/validate_rax_logits.py` compares the full
  last-position logits against the matching PyTorch precision and input.
