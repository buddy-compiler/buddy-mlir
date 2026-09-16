# PyTorch Operator Coverage Tool (issue #911)

Automated coverage analysis for Buddy-MLIR's PyTorch frontend path:

`PyTorch / ATen → DynamoCompiler (_ops_map) → Buddy Graph → ops_registry lowering → (optional) compile / run`

## Quick start (static, no build required)

From the `buddy-mlir` repo root:

```bash
python tools/pytorch_op_coverage/run_coverage.py \
  --out-dir tools/pytorch_op_coverage/out
```

Artifacts:

- `pytorch_op_coverage.json` — machine-readable
- `pytorch_op_coverage.md` — human-readable

## Live mode (needs Buddy Python packages)

Build with `BUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON`, then:

```bash
export BUDDY_MLIR_BUILD_DIR=$PWD/build
export LLVM_MLIR_BUILD_DIR=$PWD/llvm/build
export PYTHONPATH=${BUDDY_MLIR_BUILD_DIR}/python_packages:${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${PYTHONPATH}

python tools/pytorch_op_coverage/run_coverage.py --mode live \
  --out-dir tools/pytorch_op_coverage/out
```

Live mode currently probes a small seed set (add/mm/topk/gather/scatter_add/silu/softmax) through `DynamoCompiler.importer` + `lower_to_top_level_ir`. Compile + numerical correctness hooks are intentionally stubbed for the next iteration.

## Windows note

A full LLVM/Buddy build on Windows is heavy and this repo historically has an NTFS-invalid path under `tests/Models/BuddyLeNet/images/1-28*28.png`. Prefer Linux or WSL for live mode. Static mode runs on Windows without that build.

Sparse-checkout tip if the `*` filename blocks clone:

```bash
git config core.protectNTFS false
git sparse-checkout init --no-cone
git sparse-checkout set '/*' '!/tests/Models/BuddyLeNet/images/1-28*28.png'
```

## Methodology

See [`docs/PytorchOpCoverage.md`](../../docs/PytorchOpCoverage.md).

## Target operator set (denominator)

[`data/target_ops_v0.json`](data/target_ops_v0.json) defines **Buddy Target Op Set v0**.

The 90%+ goal in issue #911 applies to this (or a successor) **clearly named** denominator after live compile+correctness measurement — not to “frontend map exists”.
