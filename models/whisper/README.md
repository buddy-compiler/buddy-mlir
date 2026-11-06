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

The build directory contains `whisper.rax`, the compiled model and runtime
libraries, `vocab.txt`, and the sample `audio.wav`. The default payload mode
embeds the weights, model kernels, runtime libraries, vocabulary, runner, and
server plugin in the `.rax`. The sample `audio.wav` remains a separate input
file; copy it alongside the package when running the bundled example.

## buddy-cli

```bash
./build/bin/buddy-cli \
  --model ./build/models/whisper/whisper.rax \
  --audio ./build/models/whisper/audio.wav \
  --max-tokens 64
```

`--audio` accepts a local WAV file and defaults to `audio.wav` beside
the `.rax` package. `--no-stats` preserves the concise `[Output]` form.

## buddy-server

Start the server with a Whisper package. The server discovers
`model_library` from the manifest.

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

## Runtime tests

With `BUDDY_ENABLE_TESTS=ON` and `BUDDY_BUILD_WHISPER_MODEL=ON`, run:

```bash
cmake --build build --target buddy-whisper-runtime-test buddy-whisper-preprocess-test
ctest --test-dir build -R '^whisper-(runtime|preprocess)$' --output-on-failure
```

These tests use a tiny model kernel and do not download model weights.
`whisper-runtime` exercises the runtime with real audio preprocessing.
`whisper-preprocess` substitutes a counting preprocessing function to check
that each request preprocesses its audio once, while each decoding step
receives a fresh copy of the features. Both tests exercise repeated requests
and a kernel that overwrites its input features.
