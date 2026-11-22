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

`specs/f16_ocr_sample_last_token.json` is the K3 development variant. It
traces a real OCR sample and the official `OCR:` prompt, uses FP16 weights and
activations while retaining 82 FP32 auxiliary parameters, and computes only
the final position's LM-head logits. The original F32 spec remains available.

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
OpenMP/MLIR runtime libraries, and `BUDDY_RISCV_VLEN=1024` for A100. Pass
`-DBUDDY_PADDLEOCR_K3_IME=ON` (with `BUDDY_RISCV_VLEN=1024` and `precision=f16`)
to lower `M >= 8` `linalg.matmul` ops, Vision patch-embedding `linalg.conv_2d_nhwc_fhwc`
(padded `K=588->592`), `M=1` transposed `[N, K]` FP32 vector GEMV (`1x1024x103424`), and all
45 Attention `bmm1 -> scale -> softmax -> bmm2` blocks via K3 A100 FP16 IME (`codegen/lower_to_obj.sh`,
reusing PR #997's `-simplify-tosa-reshape`, `-silu-fusion`, `-fuse-prefill-elementwise`,
`target=k3 a100-fp16=true a100-gemm=true verify-fp16=true`, zero-copy `memref.transpose` weight
folding with unit-row-stride packing, L1-resident FP32 softmax + `-approximate-vector-exp`,
`-lower-linalg-to-vir=exp-only=true` for FP32 SiLU, `-matmul-vectorization-blis=vector-size=64 target-vlen-bits=1024`,
plus `M%8` / `N%32` tails and parallel `batch_matmul` vectorization). On `K3-002` (A100 CPUs 8–15, 8 threads,
after `echo 0 > /proc/set_ai_thread`), this reduces `paddleocr_model.so` from `64M` to `37M` and
lowers single-forward latency from `525.7s` (FP16 RVV) and `61.5s–61.9s` (non-fused IME) down to
`24.75s–25.96s` (`~20.3x–21.2x` over FP16 RVV) while preserving Top-5 token order
(`[35875, 93978, 13415, 85456, 93970]`), improving `max_abs_error` vs PyTorch FP16 reference from
`0.334` to `0.145` (pre-processed tensors, `rel_l2=0.94%`, `cos=0.999955`)
and `0.131` (direct PNG `--image` input, `24.99s`, `rel_l2=0.88%`, `cos=0.999960`).

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
The runner decodes and resizes the image in C++ to the compiled `[1,20,40]`
grid. `--image <directory>` instead reads the exact preprocessed tensors for
compiler correctness checks. The prompt and grid are fixed by this variant;
the runner currently returns first-token logits, not a full OCR transcript.

The output is a JSON object with the logits shape, summary statistics of the
last-token logits vector, and the top-5 token hypotheses.

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
