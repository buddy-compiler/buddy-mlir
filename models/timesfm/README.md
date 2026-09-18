# TimesFM 2.5 (200M)

Time-series foundation-model forecasting with
[google/timesfm-2.5-200m-pytorch](https://huggingface.co/google/timesfm-2.5-200m-pytorch),
served through the `buddy-cli` / `.rax` runtime. The build imports the PyTorch
TimesFM 2.5 encoder, compiles the fixed-shape point-forecast forward graph to
MLIR, links `timesfm_model.so`, and packs a `timesfm.rax` manifest plus an
`InferenceRunner` plugin that `buddy-cli` loads at run time.

The runner feeds a fixed-length context window (`num_patches x patch_length =
16 x 32 = 512` time points) into the AOT forward graph and emits the model's
raw point-forecast tensor, shaped
`(1, num_patches, output_patch_len * quantile_len) = (1, 16, 1280)`.

## Prerequisites

- A built LLVM/MLIR and `buddy-mlir` (see the top-level [README](../../README.md)),
  configured with `-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON`.
- The Python environment that `buddy-mlir` was built against, with `torch`,
  `timesfm`, and the Buddy Python frontend available. This is only needed at
  **build time** to import the PyTorch model; the packaged `.rax` has no Python
  dependency at run time (see Notes).
- A local HuggingFace `google/timesfm-2.5-200m-pytorch` snapshot directory
  (containing `config.json` + `model.safetensors`). The repository intentionally
  does not provide a default local path.

## Build

`timesfm` is wired through `buddy_add_model(MODEL_KIND single_forward)` with its
own importer/manifest generator (`codegen/import-timesfm.py` /
`codegen/gen_timesfm_manifest.py`, the same pluggable
`IMPORT_SCRIPT`/`MANIFEST_SCRIPT` mechanism ColBERTv2, BGE-M3, and Whisper use).
Add `add_subdirectory(models/timesfm)` (e.g. behind a `BUDDY_BUILD_TIMESFM_MODEL`
option) and build directly with CMake:

```bash
cd buddy-mlir
conda activate buddy

cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DBUDDY_TIMESFM_MODEL_PATH=/path/to/timesfm-2.5-200m-pytorch
cmake --build build --target timesfm_rax
```

`-DBUDDY_TIMESFM_MODEL_PATH` must point to a local HuggingFace-format TimesFM
snapshot directory (it is staged into the build as `BUDDY_TIMESFM_MODEL_PATH`
and forwarded to the importer via the `TIMESFM_MODEL_PATH` environment
variable).

The build emits these artifacts under `build/models/timesfm/`:

| File | Description |
| --- | --- |
| `timesfm.rax` | Model manifest (embeds weights, `timesfm_model.so`, runner) |
| `timesfm_model.so` | Compiled MLIR kernels (exports `_mlir_ciface_forward`) |
| `timesfm_runner.so` | `InferenceRunner` plugin loaded by `buddy-cli` |
| `arg0.data` | Flattened model weights (231,289,280 f32) |
| `f32.json` | Variant spec, staged alongside the `.rax` |

## Run

```bash
./build/bin/buddy-cli \
  --model ./build/models/timesfm/timesfm.rax \
  --prompt "0.1, 0.2, 0.4, 0.8, 1.6, ..."
```

- `--prompt "<floats>"` supplies the context time series (comma or whitespace
  separated). Values are aligned to the END of the fixed 512-point window: a
  short series is left-padded with zeros and a long series keeps its last 512
  values.
- With no `--prompt`, the runner uses a deterministic default context series
  (a sum of two sines), so every run reproduces the same forecast.
- `--no-stats` suppresses runner logs and prints only the forecast tensor.

The output is a JSON list of `num_patches` arrays, each with
`output_patch_len * quantile_len = 1280` values (per-patch raw point-projection
output).

## Forward ABI

The imported graph uses the fixed input shape `(1, num_patches, patch_length) =
(1, 16, 32)` from `models/timesfm/specs/f32.json`. The MLIR forward ABI is

```text
forward(weights: memref<231289280 x f32>,
        inputs:   memref<1 x 16 x 32 x f32>,
        masks:    memref<1 x 16 x 32 x f32>)
  -> (point_forecast: memref<1 x 16 x 1280 x f32>)
```

`_mlir_ciface_forward` takes one pointer per memref: the single result first,
then the inputs in declaration order
(`MemRef<float,3>*`, `MemRef<float,1>*`, `MemRef<float,3>*`,
`MemRef<float,3>*`). `arg0.data` holds the flattened `weights` memref.

## Notes

- The `.rax` and other artifacts live in the **build** directory
  (`build/models/timesfm/`), not in the source tree. Use the `build/` prefix in
  the `--model` path.
- There is no tokenizer / vocabulary for TimesFM (the input is raw numeric time
  series), so `tokenizer_file` is `"N/A"` and no vocab is staged. A `.rax` built
  once can be copied to another machine (same architecture) and run with just
  `buddy-cli` and that single file.
- The trace target replicates the original PR: it wraps the model so `forward`
  returns only the point-forecast tensor (`out[2]`), so the emitted result is
  the model's raw `output_ts`. The official TimesFM `decode()` additionally
  applies reversible instance normalization (`revin`) using running statistics
  of the input series and selects the point-forecast column `decode_index` (5);
  the AOT graph returns the raw point projection, so accuracy comparison should
  target the raw `output_ts` of `TimesFM_2p5_200M_torch.model`, reshaped to
  `(1, num_patches, output_patch_len, quantile_len)`.
- Change `num_patches` / `patch_length` in the spec (or pass a different spec
  via `--spec`) to re-target a different fixed context window. The model's
  official max context is 16384 points; the AOT graph must be re-imported for a
  different window.
- For a quick accuracy check, compare the `buddy-cli` output against
  `timesfm.TimesFM_2p5_200M_torch.from_pretrained(<local snapshot>, backend="cpu")`
  on the same `(1, 16, 32)` input and all-ones mask; the raw `output_ts` should
  match closely.
