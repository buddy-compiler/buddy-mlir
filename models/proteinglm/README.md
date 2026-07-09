# ProteinGLM buddy-cli Integration

This directory wires ProteinGLM masked language modeling into the unified
`buddy-cli` runtime flow:

- custom PyTorch-to-MLIR importer
- custom RHAL manifest generator
- `InferenceRunner` implementation
- runner plugin shared library
- self-contained `.rax` package

## Model Snapshot

Use a local HuggingFace-format ProteinGLM snapshot. The expected default path
for local testing is:

```bash
/home/gnhuang/models/proteinglm-1b-mlm
```

The directory must contain at least:

- `config.json`
- `model.safetensors`
- `tokenizer.model`
- `tokenizer_config.json`
- `special_tokens_map.json`
- remote-code files such as `modeling_proteinglm.py`

## Build

From the repository root:

```bash
python tools/buddy-codegen/build_model.py \
  --spec models/proteinglm/specs/base.json \
  --build-dir build \
  --local-model /home/gnhuang/models/proteinglm-1b-mlm
```

Equivalent CMake configuration:

```bash
cmake -S . -B build \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DBUDDY_BUILD_PROTEINGLM_MODEL=ON \
  -DBUDDY_PROTEINGLM_SPEC=models/proteinglm/specs/base.json \
  -DBUDDY_PROTEINGLM_MODEL_PATH=/home/gnhuang/models/proteinglm-1b-mlm

cmake --build build --target proteinglm_rax
```

Expected outputs:

- `build/models/proteinglm/proteinglm_model.so`
- `build/models/proteinglm/proteinglm_runner.so`
- `build/models/proteinglm/proteinglm.rax`

The `.rax` embeds the model shared library, runner plugin, weights, and
tokenizer assets.

## Run

Smoke test:

```bash
./build/bin/buddy-cli \
  --model build/models/proteinglm/proteinglm.rax \
  --prompt 'A <mask> C' \
  --no-stats
```

Expected output shape is a top-k list for the mask position, for example:

```text
position 1:
  <eos> (34): ...
  G (3): ...
  C (20): ...
```

## Implementation Notes

ProteinGLM is built as `MODEL_KIND single_forward`, not as an autoregressive
prefill/decode LLM. The runner performs one fixed-shape forward pass and prints
top-k token predictions at `<mask>` positions.

The importer applies small compatibility shims for the current local toolchain:

- keeps HuggingFace dynamic remote-code cache under the build directory
- supplies `config.max_length` when the snapshot only has `seq_length`
- handles newer `transformers` tied-weight metadata expectations
- primes and freezes rotary embedding caches to avoid Dynamo graph breaks for
  the fixed `max_seq_len`

The manifest uses `vocab_uri` for the tokenizer path so `rax-pack` can embed it
as a payload entry. Do not add a separate `tokenizer_uri=file:...` unless the
runner and packer are updated to resolve it portably.

## Key Files

- `specs/base.json`: model family, shape, vocab, and artifact names
- `codegen/import-proteinglm.py`: PyTorch import to `forward.mlir`,
  `subgraph0.mlir`, and `arg0.data`
- `codegen/gen_proteinglm_manifest.py`: RHAL manifest generation
- `ProteinGLMRunner.cpp`: runtime loading, tokenization, forward call, top-k
  printing
- `ProteinGLMRunnerPlugin.cpp`: `InferenceRunner` plugin C ABI exports
- `CMakeLists.txt`: `buddy_add_model` integration
