# Buddy Vision Graph Import

This directory contains model-specific import tests that verify a Hugging Face
vision model can be imported by `DynamoCompiler`, lowered to MLIR, and written
to disk.

The directory includes lightweight vision architecture coverage tests. Those
tests use tiny random-weight configs to cover graph structure without
downloading large checkpoints.

## Covered Models

| Model | Test file |
| --- | --- |
| CLIP | `test_import_clip.py` |
| DeiT | `test_import_deit.py` |
| DETR | `test_import_detr.py` |
| MobileViT | `test_import_mobilevit.py` |
| RegNet | `test_import_regnet.py` |
| ResNet | `test_import_resnet.py` |
| SegFormer | `test_import_segformer.py` |
| Swin Transformer | `test_import_swin.py` |
| ViT | `test_import_vit.py` |

## What This Test Checks

Each `test_import_*.py` script follows the same workflow:

1. Build a random-weight model from a tiny Hugging Face config.
2. Create a fixed-shape image input tensor.
3. Import the model with `DynamoCompiler`.
4. Assert that the import stays in a single graph.
5. Lower the imported graph to MLIR.
6. Save `subgraph0.mlir` and `forward.mlir`.

These tests are currently intended to be run directly with Python.

## Prerequisites

Build Buddy MLIR with Python packages enabled, then export the runtime paths
from the build directory:

```bash
cd buddy-mlir/build
export BUDDY_MLIR_BUILD_DIR=$PWD
export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages
```

## Run One Model Test

Example: run the ViT import test.

```bash
cd buddy-mlir
python3 tests/Models/BuddyVisionGraphImport/test_import_vit.py
```

If `BUDDY_MLIR_BUILD_DIR` is exported, the generated MLIR files are written to:

```text
build/tests/Models/BuddyVisionGraphImport/vit/
```

## Expected Success Signals

The script should print all of the following:

```text
Graph import completed: 1 graph(s)
✓ No graph break detected
✓ ... graph construction test PASSED
```

`len(graphs) == 1` is the key import condition. If it is greater than `1`, the
model hit a graph break during Dynamo import.

## Output Files

By default, generated MLIR is written under:

```text
build/tests/Models/BuddyVisionGraphImport/<model>/
```

Each successful run emits:

```text
subgraph0.mlir
forward.mlir
```

For ViT, the default output directory is:

```text
build/tests/Models/BuddyVisionGraphImport/vit/
```

## Adding a New Model Test

Use an existing script in this directory as the template and keep the same
structure:

1. Add a `test_import_<model>.py` file.
2. Build the model from a tiny random-weight config.
3. Use fixed-shape tensor inputs.
4. Import with `DynamoCompiler`.
5. Assert a single imported graph.
6. Lower to MLIR and save `subgraph0.mlir` and `forward.mlir`.

## Notes

- `lit.local.cfg.py` currently excludes the Python files in this directory, so
  these tests are not picked up as normal `lit` tests.
- These tests use tiny configs and do not download pretrained weights.
- Some models may need model-specific import settings such as scalar output
  capture. Keep those settings inside the corresponding test script instead of
  adding shared fallback logic here.
