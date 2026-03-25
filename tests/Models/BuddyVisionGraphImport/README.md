# Buddy Vision Graph Import

This directory contains model-specific import tests for vision and multimodal
models that verify a Hugging Face model can be imported by `DynamoCompiler`,
lowered to MLIR, and written to disk.

## What This Test Checks

Each `test_import_*.py` script follows the same workflow:

1. Load a model config from a local path or Hugging Face model ID.
2. Build a random-weight model from config.
3. Import the model with `DynamoCompiler`.
4. Assert that the import stays in a single graph.
5. Lower the imported graph to MLIR.
6. Save `subgraph0.mlir` and `forward.mlir`.

These tests are currently intended to be run directly with Python.

## Models

| Model | Test Script | Type |
|-------|-------------|------|
| MobileViT Small | `test_import_mobilevit_small.py` | Vision |
| CLIP ViT Base Patch32 | `test_import_clip_vit_base_patch32.py` | Vision-Language (Multimodal) |

## Prerequisites

Build Buddy MLIR with Python packages enabled, then export the runtime paths
from the build directory:

```bash
cd buddy-mlir/build
export BUDDY_MLIR_BUILD_DIR=$PWD
export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages
```

If you use local model snapshots, source the helper script once:

```bash
cd buddy-mlir
source tests/Models/BuddyVisionGraphImport/set_model_env.sh
```

You can also export a single model path directly without sourcing the helper
script. For example:

```bash
export CLIP_ViT_BASE_MODEL_PATH=/path-to-clip-vit-base-patch32
export MOBILEVIT_SMALL_MODEL_PATH=/path-to-mobilevit-small
```

## Run One Model Test

Example: run the CLIP ViT Base import test.

```bash
cd buddy-mlir
python3 tests/Models/BuddyVisionGraphImport/test_import_clip_vit_base_patch32.py
```

If `--output-dir` is not specified, the generated MLIR files are written to:

```text
build/tests/Models/BuddyVisionGraphImport/clip-vit-base/
```

You can also override the MLIR output location:

```bash
python3 tests/Models/BuddyVisionGraphImport/test_import_clip_vit_base_patch32.py \
  --output-dir /tmp/clip_vit_base_mlir
```

## Expected Success Signals

The script should print all of the following:

```text
Graph import completed: 1 graph(s) generated
✓ No graph break detected
MLIR generation completed
✓ MLIR structure verified
✓ ... graph construction test PASSED
```

`len(graphs) == 1` is the key import condition. If it is greater than `1`, the
model hit a graph break during Dynamo import.

## Output Files

By default, generated MLIR is written under:

```text
build/tests/Models/BuddyVisionGraphImport/<model_name>/
```

Each successful run emits:

```text
subgraph0.mlir
forward.mlir
```

## Adding a New Model Test

Use an existing script in this directory as the template and keep the same
structure:

1. Add a `test_import_<model>.py` file.
2. Read the model path from an environment variable first.
3. Support `--output-dir`.
4. Import with `DynamoCompiler`.
5. Assert a single imported graph.
6. Lower to MLIR and save `subgraph0.mlir` and `forward.mlir`.

When the model requires a local snapshot, add its environment variable to:

```text
tests/Models/BuddyVisionGraphImport/set_model_env.sh
```

## Notes

- `lit.local.cfg.py` currently excludes the Python files in this directory, so
  these tests are not picked up as normal `lit` tests.
- Some models may need model-specific import settings such as static cache or
  scalar output capture. Keep those settings inside the corresponding test
  script instead of adding shared fallback logic here.
