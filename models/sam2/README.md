# SAM2

SAM 2 ([facebook/sam2-hiera-tiny](https://huggingface.co/facebook/sam2-hiera-tiny))
image-segmentation **vision encoder**, served through the `buddy-cli` / `.rax`
runtime. The build imports the PyTorch `Sam2VisionModel` image encoder,
compiles the fixed-shape `1 x 3 x 256 x 256` forward graph to MLIR, links
`sam2_model.so`, and packs a `sam2.rax` manifest plus an `InferenceRunner`
plugin that `buddy-cli` loads at run time.

The runner feeds a fixed `1 x 3 x 256 x 256` image tensor through the encoder
and emits the resulting `1 x 8 x 8 x 768` image feature map (`last_hidden_state`)
plus the six FPN feature maps (`fpn_0`…`fpn_5`).

> Scope note: only the image **encoder** is compiled. The prompt-encoder +
> mask-decoder path of the full SAM 2 video model consumes interactive
> point/box prompts and a memory state that has no single fixed-shape forward,
> so it is out of scope for the AOT `single_forward` pipeline. The original PR
> traced the same `m.vision_encoder` sub-module, and this integration follows
> that trace target exactly.

## Prerequisites

- A built LLVM/MLIR and `buddy-mlir` (see the top-level [README](../../README.md)),
  configured with `-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON`.
- The Python environment that `buddy-mlir` was built against, with `torch`,
  `transformers`, and the Buddy Python frontend available. This is only needed
  at **build time** to import the PyTorch model; the packaged `.rax` has no
  Python dependency at run time (see Notes).
- A local HuggingFace `facebook/sam2-hiera-tiny` snapshot directory. The
  repository intentionally does not provide a default local path.

## Build

`SAM2` is wired through `buddy_add_model(MODEL_KIND single_forward)` with its
own importer/manifest generator (`codegen/import-sam2.py` /
`codegen/gen_sam2_manifest.py`, the same pluggable `IMPORT_SCRIPT` /
`MANIFEST_SCRIPT` mechanism ColBERTv2 uses). Build it directly with CMake
(its `model_family` is not in `tools/buddy-codegen/build_model.py`'s
whitelist). The `models/sam2/CMakeLists.txt` only fires once the model is
added to the top-level `models/CMakeLists.txt` gate (like ColBERTv2):

```cmake
# models/CMakeLists.txt (pending integrator change)
if(BUDDY_BUILD_SAM2_MODEL)
  add_subdirectory(sam2)
endif()
```

```bash
cd buddy-mlir
conda activate buddy

cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DBUDDY_BUILD_SAM2_MODEL=ON \
  -DBUDDY_SAM2_MODEL_PATH=/path/to/sam2-hiera-tiny
cmake --build build --target sam2_rax
```

`-DBUDDY_SAM2_MODEL_PATH` must point to a local HuggingFace-format
`facebook/sam2-hiera-tiny` snapshot directory. It is staged into the build and
forwarded to the importer via the `SAM2_MODEL_PATH` environment variable.

The build emits these artifacts under `build/models/sam2/`:

| File | Description |
| --- | --- |
| `sam2.rax` | Model manifest (embeds weights, `sam2_model.so`, runner) |
| `sam2_model.so` | Compiled MLIR kernels (exports `_mlir_ciface_forward`) |
| `sam2_runner.so` | `InferenceRunner` plugin loaded by `buddy-cli` |
| `arg0.data` | Flattened vision-encoder weights |
| `config.json` | HF model config, staged from the local snapshot |

## Run

```bash
./build/bin/buddy-cli \
  --model ./build/models/sam2/sam2.rax < /dev/null
```

- The runner encodes a deterministic all-zeros `1 x 3 x 256 x 256` image (the
  same reference input the original PR's `validate_accuracy.py` used). The
  `Sam2Runner` does not decode an actual image file; passing `--image` is
  rejected with a clear error, so `buddy-cli`'s stdin prompt is satisfied with
  `< /dev/null`.
- `--no-stats` suppresses runner logs and prints only the JSON output.

The output is a JSON object with the shape of every output tensor followed by
the flattened `last_hidden_state` image feature map (`1 x 8 x 8 x 768`
values in row-major order).

## Notes

- The `.rax` and other artifacts live in the **build** directory
  (`build/models/sam2/`), not in the source tree. Use the `build/` prefix in
  the `--model` path.
- There is no Python/`transformers` dependency at run time: a `.rax` built once
  can be copied to another machine (same architecture) and run with just
  `buddy-cli` and that single file — no source tree, no Python environment.
- The imported graph uses `image_size = 256` from
  `models/sam2/specs/f32.json`; change the spec (or pass a different spec via
  `--spec`) to re-target a different fixed input resolution. The traced ABI
  fixes the output to `1 x 8 x 8 x 768` plus the six FPN maps at
  `16/16/32/32/64/64`.
- The MLIR forward ABI is
  `forward(weights, pixel_values) -> (last_hidden_state, fpn_0..fpn_5)`; see
  the header comment in `Sam2Runner.cpp` for the exact
  `_mlir_ciface_forward` calling convention.
- For a quick accuracy check, compare the `buddy-cli` output against
  `AutoModel.from_pretrained(<local sam2 snapshot>).vision_encoder` on the same
  all-zeros `1 x 3 x 256 x 256` input.
