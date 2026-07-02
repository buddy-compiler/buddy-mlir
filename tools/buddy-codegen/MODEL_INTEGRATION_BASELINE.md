# Model Integration Baseline

This document defines the shared baseline for adding packaged models to
`buddy-mlir`. New model integrations should follow this path unless there is a
clear technical blocker.

The goal is one public build rule for all models:

```bash
python3 tools/buddy-codegen/build_model.py \
  --spec models/<model_family>/specs/<variant>.json \
  --build-dir build \
  --local-model /path/to/local/hf/snapshot
```

Do not document `cmake -S` / `cmake --build` as the primary user path. CMake
targets are implementation details behind the Python build entry.

## Required Shape

Every model integration must provide:

- `models/<model_family>/CMakeLists.txt`
- `models/<model_family>/README.md`
- `models/<model_family>/specs/<variant>.json`
- `models/<model_family>/<Model>Runner.cpp`
- `models/<model_family>/<Model>RunnerPlugin.cpp`
- `models/<model_family>/include/buddy/runtime/models/<Model>Runner.h`
- Model-specific importer or manifest helpers only when the shared tools cannot
  express the model yet.

Generated artifacts belong under `build/models/<model_family>/`, never in the
source tree.

## Build Entry

`tools/buddy-codegen/build_model.py` is the only top-level user entry.

It must:

- read `model_family` from the spec;
- enable the right CMake option, such as `BUDDY_BUILD_<MODEL>_MODEL=ON`;
- set the model spec cache variable;
- pass local HuggingFace snapshots through `--local-model`;
- infer the default CMake target, usually `<model_family>_rax`;
- reject unsupported model families explicitly.

Repository code must not hard-code developer-local model paths such as
`/home/<user>/...`. If a model needs a local snapshot, require `--local-model`.

## Spec Rules

Specs are the model contract. They should describe model identity and compile
time shape, not machine-local paths.

Required fields:

```json
{
  "hf_model_path": "org/model-name",
  "model_family": "<model_family>",
  "variant": "<variant>"
}
```

Add model-specific compile-time constants as needed, for example:

```json
{
  "max_seq_len": 512,
  "hidden_size": 1024,
  "vocab_size": 250002
}
```

Do not put local snapshot paths in specs. Use `hf_model_path` for the public
model identity and `--local-model` for local import.

## CMake Rules

Model CMake files should use `buddy_add_model(...)` from:

```cmake
include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)
```

Use `MODEL_KIND` to select the build shape:

- `llm_prefill_decode`: decoder-only LLMs with prefill/decode entrypoints.
- `single_forward`: encoder, embedding, classification, and other one-forward
  models.

Example for a single-forward model:

```cmake
buddy_add_model(
  NAME bge_m3
  MODEL_KIND single_forward
  SPEC "${BUDDY_BGE_M3_SPEC}"
  RUNNER_SRC BgeM3Runner.cpp
  RUNNER_PLUGIN_SRC BgeM3RunnerPlugin.cpp
  LOCAL_MODEL "${BUDDY_BGE_M3_MODEL_PATH}"
  LOCAL_MODEL_ASSETS
    tokenizer.json
    tokenizer_config.json
  EXTRA_PAYLOAD_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/codegen/model_tokenize.py"
)
```

Avoid writing model-specific import/compile/link/pack commands in
`models/<model_family>/CMakeLists.txt`. The model CMake file should mainly
declare the model.

## Import Rules

The shared import entry is:

```bash
tools/buddy-codegen/import_model.py
```

It should dispatch by `model_family`. Existing model-specific import logic may
live under `models/<model_family>/codegen/`, but it should be called from the
shared import entry rather than from a private build script.

Importer outputs should be predictable:

For `MODEL_KIND llm_prefill_decode`:

```text
forward_prefill.mlir
subgraph0_prefill.mlir
forward_decode.mlir
subgraph0_decode.mlir
weight files
```

For `MODEL_KIND single_forward`:

```text
forward.mlir
subgraph0.mlir
weight files
```

Use stable names so `compile_pipeline.py` and `buddy_model.cmake` can remain
model-agnostic.

## Compile Rules

The shared compile entry is:

```bash
tools/buddy-codegen/compile_pipeline.py
```

Do not hand-write `mlir-opt | buddy-opt | mlir-translate | llvm-as | llc`
pipelines in per-model build scripts.

Supported public modes:

- `--compile-all` for LLM prefill/decode graph sets.
- `--compile-partitioned` for partitioned LLM builds.
- `--compile-single-forward` for single-forward models.

If a new model needs a different lowering sequence, add a named mode or pipeline
type to `compile_pipeline.py` and keep the model CMake file declarative.

## Manifest Rules

The shared manifest entry is:

```bash
tools/buddy-codegen/gen_manifest.py
```

It should dispatch by `model_family` or `MODEL_KIND` when manifest structure
differs.

Manifests must describe:

- `model_name`;
- `runner_library`;
- model shared library code object;
- external weights;
- runtime buffers;
- tokenizer or preprocessing assets required at runtime.

Payload files should be copied to the build artifact directory and referenced
with relative `file:<name>` URIs so `.rax` packaging is relocatable.

## Runner Rules

Each model must provide an `InferenceRunner` implementation and plugin.

The runner should:

- load paths and attrs from the `.rax` manifest;
- load the model shared library and resolve `_mlir_ciface_*` symbols;
- load weights from packaged constants;
- handle model-specific preprocessing/postprocessing;
- keep generated payload extraction paths out of the source tree;
- support `--no-stats` clean output where practical.

For temporary Python preprocessing helpers, package the helper and required
assets into `.rax`. Document pure-C++ preprocessing as future work if it is not
implemented.

## README Rules

Model READMEs should follow this structure:

```text
# Model Name

Short model/runtime description.

## Prerequisites

## Build

## Run

## Notes
```

The Build section must use `tools/buddy-codegen/build_model.py` as the primary
command. Mention lower-level CMake targets only as implementation details or for
debug iteration.

Do not include developer-local paths. Use placeholders like:

```text
/path/to/<model-snapshot>
```

## Verification Rules

Every integration should verify at least:

- Python syntax for changed codegen scripts;
- `build_model.py --dry-run` emits the expected CMake options and target;
- full `build_model.py` build produces `<model_family>.rax`;
- `buddy-cli --model build/models/<model_family>/<model_family>.rax ...` runs;
- output numerics or text match a reference implementation where feasible.

For numeric models, report shape, norm if relevant, cosine similarity or max
absolute difference against HuggingFace/reference output.

## What Not To Do

- Do not add private per-model build scripts that duplicate
  `compile_pipeline.py`.
- Do not make users run raw CMake commands as the primary build path.
- Do not hard-code local model paths in source, specs, CMake defaults, or README.
- Do not write generated artifacts into `models/<model_family>/`.
- Do not change DeepSeek/LLM defaults when adding a new model kind.
- Do not add model-specific assumptions to shared tools without gating them by
  `model_family` or `MODEL_KIND`.
