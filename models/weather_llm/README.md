# Weather-LLM

Weather-LLM
([AuraWorxAI/weather-llm-sft](https://huggingface.co/AuraWorxAI/weather-llm-sft))
is a `LlamaForCausalLM` for weather forecasting, served through the
`buddy-cli` / `.rax` runtime. The build imports the PyTorch model with the
generic `tools/buddy-codegen/import_model.py` pipeline, compiles the
prefill/decode graphs to MLIR, links `weather_llm_model.so`, and packs a
`weather_llm.rax` manifest plus an `InferenceRunner` plugin that `buddy-cli`
loads at run time.

Model facts (from `config.json`):

| Key | Value |
| --- | --- |
| Architecture | `LlamaForCausalLM` (`model_type: llama`) |
| Params | 1.26B |
| Layers / heads / kv-heads | 24 / 16 / 8 |
| Hidden size / head dim | 2048 / 128 |
| Vocab size | 32000 |
| EOS | 3 (`</s>`) |
| Tokenizer | custom SentencePiece (`tokenizer.model`) |

The tokenizer is a custom SentencePiece model. The in-tree Llama codec
`tokenizeLlama()`/`revertLlama()` in
`frontend/Interfaces/buddy/LLM/TextContainer.h` consumes a plain token-per-line
vocab text file, so this directory ships `vocab.txt`, a token-per-line
conversion of the snapshot's `tokenizer.model` (one SentencePiece piece per
line, indexed by position — matching `id_to_piece(i)`). It is staged next to
the `.rax` via `ASSET_FILES` and referenced by the manifest's
`file:vocab.txt` URI.

## Prerequisites

- A built LLVM/MLIR and `buddy-mlir` (see the top-level
  [README](../../README.md)), configured with
  `-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON`.
- The Python environment that `buddy-mlir` was built against, with `torch`,
  `transformers`, and the Buddy Python frontend available. This is only needed
  at **build time** to import the PyTorch model; the packaged `.rax` has no
  Python dependency at run time (see Notes).
- A local HuggingFace `AuraWorxAI/weather-llm-sft` snapshot directory
  (config.json, tokenizer.model, model.safetensors). The repository
  intentionally does not provide a default local path.

## Build

`weather_llm` is wired through `buddy_add_model(NAME weather_llm ...)`
(SPEC `specs/f32.json`, `RUNNER_SRC WeatherRunner.cpp`,
`RUNNER_PLUGIN_SRC WeatherRunnerPlugin.cpp`) and gated behind the
`BUDDY_BUILD_WEATHER_LLM_MODEL` CMake option. Build it directly with CMake:

```bash
cd buddy-mlir
source buddy-env/bin/activate   # or conda activate buddy

cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DBUDDY_BUILD_WEATHER_LLM_MODEL=ON \
  -DBUDDY_WEATHER_LLM_LOCAL_MODEL=/path/to/weather-llm-sft
cmake --build build --target weather_llm_rax
```

`-DBUDDY_WEATHER_LLM_LOCAL_MODEL` must point to a local HuggingFace-format
`AuraWorxAI/weather-llm-sft` snapshot directory. It is forwarded to the
importer via the `DEEPSEEKR1_MODEL_PATH` environment variable (the generic
`tools/buddy-codegen/import_model.py` uses that env var to locate local
weights), and `config.json` is auto-detected from the snapshot. You can also
pass `-DBUDDY_WEATHER_LLM_HF_CONFIG=/path/to/config.json` explicitly.

The build emits these artifacts under `build/models/weather_llm/`:

| File | Description |
| --- | --- |
| `weather_llm.rax` | Model manifest (embeds weights, `weather_llm_model.so`, `vocab.txt`, runner) |
| `weather_llm_model.so` | Compiled MLIR kernels (prefill + decode) |
| `weather_llm_runner.so` | `InferenceRunner` plugin loaded by `buddy-cli` |
| `arg0.data` | Flattened weights (1,263,634,432 f32 elements) |
| `vocab.txt` | SentencePiece vocabulary, token-per-line, staged from this dir |

## Run

Single-shot generation:

```bash
./build/bin/buddy-cli \
  --model ./build/models/weather_llm/weather_llm.rax \
  --prompt "The weather forecast for tomorrow indicates"
```

Options:

- `--max-tokens <N>` sets the number of generated tokens (default 256).
- `--seed <N>` fixes the sampler seed for reproducibility.
- `--no-stats` suppresses runner logs.
- `--chat-template <path.json>` + `--interactive` starts an interactive REPL
  using the given chat template. Interactive mode requires a chat template;
  Weather-LLM itself is a plain causal LM and works fine single-shot.

## Notes

- The `.rax` and other artifacts live in the **build** directory
  (`build/models/weather_llm/`), not in the source tree. Use the `build/`
  prefix in the `--model` path.
- Generation stops on the Llama EOS token id `3` (`</s>`), configured in
  `WeatherRunner.cpp`.
- Tokenization is a pure C++ reimplementation of the SentencePiece-style
  tokenizer built into `WeatherRunner.cpp` (`tokenizeLlama`/`revertLlama`),
  loaded from the staged `vocab.txt`. There is no Python/`transformers`
  dependency at run time, so a `.rax` built once can be copied to another
  machine (same architecture) and run with just `buddy-cli` and that single
  file.
- The model is compiled as the `f32` variant at `max_token_len = 1024`
  (see `specs/f32.json`; `head_num = 8`, `hidden_size = 128` = head dim,
  `kv_layers = 48`, `vocab_size = 32000` derived by
  `tools/buddy-codegen/gen_config.py`).
- `import_model.py` and `validate_accuracy.py` are intentionally **not**
  vendored in this directory — the generic
  `tools/buddy-codegen/import_model.py` is used for import.
