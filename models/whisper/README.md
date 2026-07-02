# Whisper

Automatic speech recognition with [openai/whisper-base](https://huggingface.co/openai/whisper-base),
served through the `buddy-cli` / `.rax` runtime. The build imports the PyTorch
model, compiles it to MLIR, links `whisper_model.so`, and packs a `whisper.rax`
manifest plus an `InferenceRunner` plugin that `buddy-cli` loads at run time.

## Prerequisites

- A built LLVM/MLIR and `buddy-mlir` (see the top-level [README](../../README.md)),
  configured with `-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON`.
- A Python environment with `requirements.txt` installed (`transformers`, `torch`).
- Optional: a local HuggingFace `whisper-base` snapshot directory (to avoid a download).

## Build

Use the same `tools/buddy-codegen/build_model.py` entry point as the other
packaged models:

```bash
cd buddy-mlir
python3 tools/buddy-codegen/build_model.py \
  --spec models/whisper/specs/base.json \
  --build-dir build
```

To import weights from a local HuggingFace snapshot directory, pass
`--local-model`:

```bash
python3 tools/buddy-codegen/build_model.py \
  --spec models/whisper/specs/base.json \
  --build-dir build \
  --local-model /path/to/whisper-base
```

If `--local-model` is omitted, the importer downloads `openai/whisper-base`.
The build emits these artifacts under
`build/models/whisper/`:

| File | Description |
| --- | --- |
| `whisper.rax` | Model manifest (embeds weights, `whisper_model.so`, vocab, runner) |
| `whisper_model.so` | Compiled MLIR kernels (exports `_mlir_ciface_forward`) |
| `whisper_runner.so` | `InferenceRunner` plugin loaded by `buddy-cli` |
| `vocab.txt` / `audio.wav` | Tokenizer vocab / sample audio |

## Run

```bash
./build/bin/buddy-cli \
  --model ./build/models/whisper/whisper.rax \
  --audio ./build/models/whisper/audio.wav
```

- `--audio <path.wav>` selects the input clip (16 kHz mono PCM). If omitted, it
  falls back to `audio.wav` next to the `.rax`.
- `--max-tokens N` caps the number of decode steps (~18 s per token on CPU); use
  `--max-tokens 16` for a quick partial result.
- `--no-stats` suppresses the per-token iteration log and prints only `[Output]`.

Expected output for the sample clip:

```
[Output]  Nor is Mr. Quilter's manner less interesting than his matter.
```

## Notes

- The `.rax` and other artifacts live in the **build** directory
  (`build/models/whisper/`), not in the source tree. Use the `build/` prefix in
  the `--model` / `--audio` paths.
- The `.rax` embeds the runner plugin, so after editing the runner you must
  re-run `ninja -C build whisper_rax` to repack. For quick iteration you can
  override it at run time with `--runner-so <absolute-path>`.
