# Whisper

Whisper speech recognition is available through both the existing `buddy-cli`
runner and the independent `buddy-server` audio transcription backend.

## Build

The recommended entry point is the Python build wrapper. It configures CMake
from the Whisper variant spec and builds the inferred `whisper_rax` target:

```bash
conda run -n buddy-mlir python tools/buddy-codegen/build_model.py \
  --spec models/whisper/specs/base.json \
  --build-dir build \
  --local-model /path/to/whisper-base
```

`build_model.py` is a CMake wrapper; the model import itself is performed by
`models/whisper/codegen/import-whisper.py`. The importer requires PyTorch,
Transformers, and the Buddy Python package, and generates the MLIR and parameter
data consumed by the native build.

The equivalent explicit CMake commands are:

```bash
conda run -n buddy-mlir cmake -S . -B build \
  -DBUDDY_BUILD_WHISPER_MODEL=ON \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DBUDDY_WHISPER_MODEL_PATH=/path/to/whisper-base
conda run -n buddy-mlir cmake --build build --target whisper_rax
```

Use a local Hugging Face model snapshot through `--local-model` or
`BUDDY_WHISPER_MODEL_PATH` to keep the import offline. Without a local model,
the importer may fetch the model configured by the spec (by default,
`openai/whisper-base`).

The package contains `whisper.rax`, `whisper_model.so`,
`whisper_runner.so`, `whisper_transcription.so`, `vocab.txt`, and the sample
`audio.wav`. The manifest keeps both runner and transcription plugin entries;
the default payload mode embeds all of them in the `.rax`.

## buddy-cli

```bash
./build/bin/buddy-cli \
  --model ./build/models/whisper/whisper.rax \
  --audio ./build/models/whisper/audio.wav \
  --max-tokens 16
```

`--audio` accepts a local WAV file and defaults to `audio.wav` beside
the `.rax` package. `--no-stats` preserves the concise `[Output]` form.

## buddy-server

Start the server with a Whisper package. The server discovers
`transcription_library` from the manifest; `--transcription-so` is an explicit
override for legacy or development layouts.

```bash
./build/bin/buddy-server --model ./build/models/whisper/whisper.rax \
  --host 127.0.0.1 --port 8080
```

The endpoint accepts JSON containing a path on the server machine:

```bash
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -H 'Content-Type: application/json' \
  -d '{"model":"whisper_base","file":"/absolute/path/to/audio.wav",'\
'"max_tokens":64}'
```

The response contains OpenAI-compatible `text` plus Buddy extensions:
`model`, `generated_tokens`, and `timings.preprocess_ms`,
`timings.inference_ms`, and `timings.total_ms`.

The initial API supports only local WAV paths (`file` or the compatibility
alias `audio_path`). It rejects multipart uploads, inline bytes/base64, remote
URLs, data URIs, non-WAV files, and streaming requests. One model owns one
runtime/session; inference is serialized while weights remain resident.
