# Kokoro-82M

Text-to-speech with [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M),
served through the `buddy-cli` / `.rax` runtime. The build imports the PyTorch
`KModel` (Albert phoneme encoder + duration predictor + ISTFTNet vocoder,
~81.8M params, 178-token phoneme vocab), compiles the fixed-shape
`forward_with_tokens` graph to MLIR, links `kokoro_model.so`, and packs a
`kokoro.rax` manifest plus an `InferenceRunner` plugin that `buddy-cli` loads at
run time.

The runner constructs the fixed-shape inputs (phoneme `input_ids`, a reference
speaker embedding `ref_s`, and `speed`), invokes the compiled forward, and emits
the generated waveform samples.

## Prerequisites

- A built LLVM/MLIR and `buddy-mlir` (see the top-level [README](../../README.md)),
  configured with `-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON`.
- The Python environment that `buddy-mlir` was built against, with `torch`,
  `transformers`, the Buddy Python frontend, and the `kokoro` package available.
  This is only needed at **build time** to import the PyTorch model; the packaged
  `.rax` has no Python dependency at run time.
- A local HuggingFace `hexgrad/Kokoro-82M` snapshot directory. The repository
  intentionally does not provide a default local path.

## Build

`kokoro` is wired through `buddy_add_model(MODEL_KIND single_forward)` with its
own importer/manifest generator (`codegen/import-kokoro.py` /
`codegen/gen_kokoro_manifest.py`, the same pluggable
`IMPORT_SCRIPT`/`MANIFEST_SCRIPT` mechanism BGE-M3 and ColBERTv2 use). This
directory is not yet hooked into `models/CMakeLists.txt` (which selects model
subdirectories via `BUDDY_BUILD_*` options), so add an
`if(BUDDY_BUILD_KOKORO_MODEL) add_subdirectory(kokoro) endif()` block there, then
build directly with CMake:

```bash
cd buddy-mlir
conda activate buddy

cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DBUDDY_BUILD_KOKORO_MODEL=ON \
  -DBUDDY_KOKORO_MODEL_PATH=/path/to/Kokoro-82M
cmake --build build --target kokoro_rax
```

`-DBUDDY_KOKORO_MODEL_PATH` must point to a local HuggingFace-format
`hexgrad/Kokoro-82M` snapshot directory (it is staged into the build as
`BUDDY_KOKORO_MODEL_PATH` and forwarded to the importer via the
`KOKORO_MODEL_PATH` environment variable; the importer loads `config.json` and
`kokoro-v1_0.pth` from that directory).

The build emits these artifacts under `build/models/kokoro/`:

| File | Description |
| --- | --- |
| `kokoro.rax` | Model manifest (embeds weights, `kokoro_model.so`, `config.json`, runner) |
| `kokoro_model.so` | Compiled MLIR kernels (exports `_mlir_ciface_forward`) |
| `kokoro_runner.so` | `InferenceRunner` plugin loaded by `buddy-cli` |
| `arg0.data` | Flattened model weights (~81.8M f32 values) |
| `config.json` | HF config staged from the local snapshot (also the phoneme vocab) |

## Run

```bash
./build/bin/buddy-cli \
  --model ./build/models/kokoro/kokoro.rax
```

The runner emits a small JSON summary of the generated waveform (sample count,
first samples, min/max/mean/RMS). `--no-stats` suppresses runner logs and prints
only that summary.

## Notes

- The `.rax` and other artifacts live in the **build** directory
  (`build/models/kokoro/`), not in the source tree. Use the `build/` prefix in
  the `--model` path.
- **Inputs are deterministic placeholders.** Phonemization of the prompt
  requires espeak-ng + the misaki G2P frontend, which is not implemented inside
  the C++ runner; the runner instead fills a fixed deterministic `input_ids`
  (a 30-token sequence in the 178-token phoneme vocab) and a fixed deterministic
  256-dim `ref_s` reference speaker embedding (a real value would come from a
  `voices/<voice>.pt` style vector). `speed` is fixed at 1.0, matching the trace.
- **Known limitation: the AOT import does not capture the TTS network.** The
  import target is `KModel.forward_with_tokens(input_ids, ref_s, speed)`, which
  builds a data-dependent alignment matrix from the predicted phoneme durations
  (`torch.repeat_interleave` over `pred_dur`), so `torch._dynamo` cannot emit a
  single static forward graph. The captured `subgraph0.mlir` is only the
  constant-length prefix (it returns the fixed sequence length 30), and the
  compiled `kokoro_model.so` exports `_mlir_ciface_forward` for a trivial
  constant function rather than the real TTS forward. The manifest and
  `KokoroRunner.cpp` encode the *intended* forward ABI —
  `forward(weights, input_ids, ref_s, speed) -> waveform` — but the runner will
  not produce intelligible speech until the duration/alignment computation is
  made traceable (e.g. a fixed-length alignment) or a segmented
  phonemize→predict-duration→synthesize pipeline is compiled per stage.
- The forward ABI is documented in full at the top of `KokoroRunner.cpp`.
  `audio_buffer_size` (450000 = 30 tokens × `max_dur` 50 × upsample factor 300)
  is the worst-case waveform length; the true voiced length is data-dependent.
