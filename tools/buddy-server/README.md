# buddy-server

`buddy-server` is a minimal HTTP serving entry point for resident BuddyRuntime
models. The server binary is built by default and is decoupled from concrete
model implementations through resident model plugins.

## Build

`buddy-server` is built by default:

```bash
cmake -G Ninja -S . -B build \
  -DMLIR_DIR=$PWD/llvm/build/lib/cmake/mlir \
  -DLLVM_DIR=$PWD/llvm/build/lib/cmake/llvm

cmake --build build --target buddy-server
cmake --build build --target check-buddy-server
```

If the DeepSeek R1 model target is enabled, it also builds a resident serving
plugin that can be referenced from the `.rax` manifest or passed explicitly with
`--serving-so`:

```bash
cmake -G Ninja -S . -B build \
  -DMLIR_DIR=$PWD/llvm/build/lib/cmake/mlir \
  -DLLVM_DIR=$PWD/llvm/build/lib/cmake/llvm \
  -DBUDDY_BUILD_DEEPSEEK_R1_MODEL=ON \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON

cmake --build build --target buddy-server
cmake --build build --target check-buddy-server
```

DeepSeek R1 model builds produce:

```text
deepseek_r1_model.so      # compiled MLIR model kernels
deepseek_r1_runner.so     # buddy-cli InferenceRunner plugin
deepseek_r1_serving.so    # buddy-server ResidentModel plugin
deepseek_r1.rax           # manifest, with runner_library + serving_library
```

The smoke test runs `buddy-server --help` and does not require a real model.

## Implementation

The server is split into small layers:

```text
HTTP transport
  tools/buddy-server/SimpleHttpServer.*
  - minimal HTTP/1.1 socket server
  - JSON responses and SSE streaming writes

JSON adapter
  tools/buddy-server/JsonCodec.*
  - parses HTTP JSON into ServingTypes DTOs
  - formats completion, chat, tokenize, error, and SSE chunk responses

Server entry
  tools/buddy-server/buddy-server.cpp
  - parses command line arguments
  - starts HTTP routes
  - loads the model in a background thread

Resident model plugin loader
  runtime/include/buddy/runtime/core/ResidentModelPlugin.h
  tools/buddy-server/ResidentModelPluginHandle.*
  - defines the resident model plugin ABI
  - loads --serving-so with dlopen/dlsym
  - keeps the plugin loaded for the lifetime of the ResidentModel

Resident model boundary
  runtime/include/buddy/runtime/core/ServingTypes.h
  runtime/include/buddy/runtime/core/ResidentModel.h
  - model-serving DTOs and abstract interface

DeepSeek resident model
  models/deepseek_r1/DeepSeekR1ResidentModel.*
  models/deepseek_r1/DeepSeekR1ResidentModelPlugin.cpp
  - resident serving plugin entry point for deepseek_r1_serving.so
  - owns one long-lived ModelSession
  - loads weights once
  - serializes generation with an internal mutex
```

`TextGeneration` exposes a callback-based generation overload. The legacy
`runGeneration()` still prints to stdout for CLI use; the callback overload is
used by `buddy-server` to emit SSE chunks without coupling generation to HTTP.

## Start Server

`.rax` manifest resident plugin mode:

```mlir
rhal.module @deepseek_r1 attributes {
  model_name = "deepseek_r1_f32",
  runner_library = "file:deepseek_r1_runner.so",
  serving_library = "file:deepseek_r1_serving.so"
} {
  ...
}
```

```bash
./build/bin/buddy-server \
  --model ./build/models/deepseek_r1/deepseek_r1.rax \
  --chat-template examples/BuddyDeepSeekR1/deepseek-r1.json \
  --host 127.0.0.1 \
  --port 8080
```

Explicit resident plugin override:

```bash
./build/bin/buddy-server \
  --model ./build/models/deepseek_r1/deepseek_r1.rax \
  --serving-so /path/to/deepseek_r1_serving.so \
  --chat-template examples/BuddyDeepSeekR1/deepseek-r1.json \
  --host 127.0.0.1 \
  --port 8080
```

The serving plugin must export:

```cpp
extern "C" buddy::runtime::ResidentModel *
buddy_create_resident_model_v1();

extern "C" void
buddy_destroy_resident_model_v1(buddy::runtime::ResidentModel *);
```

It may also export:

```cpp
extern "C" const char *buddy_resident_model_type_v1();
```

Resident backend loading priority:

```text
1. --serving-so
2. .rax module attr serving_library
```

If neither source provides a resident serving plugin, startup fails before the
HTTP listener is created.

Legacy explicit artifact mode:

```bash
./build/bin/buddy-server \
  --model-so /path/to/deepseek_r1_model.so \
  --weights /path/to/arg0.data \
  --vocab /path/to/vocab.txt \
  --serving-so /path/to/deepseek_r1_serving.so \
  --model-type deepseek_r1 \
  --chat-template examples/BuddyDeepSeekR1/deepseek-r1.json
```

The HTTP listener starts first. Model loading runs in a background thread, so
`/health` can report `loading`, `ok`, or `error`.

## Endpoints

### GET `/health`

```bash
curl http://127.0.0.1:8080/health
```

Example response while loading:

```json
{
  "status": "loading",
  "model_loaded": false,
  "model": "deepseek_r1",
  "backend": "",
  "context_length": 0,
  "message": "model is loading"
}
```

### POST `/completion`

Receives an already-rendered prompt.

```bash
curl http://127.0.0.1:8080/completion \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "<|User|>Hello<|Assistant|>",
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.95
  }'
```

Streaming:

```bash
curl -N http://127.0.0.1:8080/completion \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "<|User|>Hello<|Assistant|>",
    "max_tokens": 128,
    "stream": true
  }'
```

### POST `/v1/chat/completions`

Receives chat messages and renders them with the configured chat template.
The response shape follows the OpenAI chat completion style.

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek_r1",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain buddy-mlir briefly."}
    ],
    "max_tokens": 256
  }'
```

Streaming:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "Tell me one sentence about MLIR."}
    ],
    "stream": true
  }'
```

### POST `/tokenize`

```bash
curl http://127.0.0.1:8080/tokenize \
  -H 'Content-Type: application/json' \
  -d '{
    "content": "Hello buddy-mlir",
    "count_only": false
  }'
```

Count only:

```bash
curl http://127.0.0.1:8080/tokenize \
  -H 'Content-Type: application/json' \
  -d '{
    "content": "Hello buddy-mlir",
    "count_only": true
  }'
```

## Request Options

Completion and chat requests support:

```json
{
  "max_tokens": 512,
  "temperature": 0.0,
  "top_k": 0,
  "top_p": 1.0,
  "min_p": 0.0,
  "repeat_penalty": 1.0,
  "repeat_last_n": 64,
  "seed": 0,
  "stop_token_ids": [151643, 151647],
  "stream": false
}
```

## Current MVP Limits

- The server binary is model-agnostic. Resident backends can be loaded with
  `--serving-so` or discovered from `.rax` `serving_library`.
- DeepSeek R1 resident serving is available through `deepseek_r1_serving.so`
  when the DeepSeek R1 model target is built.
- One process owns one `ModelSession`; generation is serialized by a mutex.
- `stop_token_ids` are supported. `stop` strings are parsed but not yet applied
  by the generation loop.
- The HTTP layer is a minimal POSIX socket implementation, not a full-featured
  production HTTP framework.
- SSE streaming uses `data: ...` chunks and finishes with `data: [DONE]`.
