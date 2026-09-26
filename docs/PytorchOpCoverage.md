# PyTorch operator coverage

This internal evaluation tool checks which PyTorch operators Buddy-MLIR can
recognize, lower, compile and execute correctly. It reports source evidence
separately from measured results; registration alone does not establish support.

## Run

From the repository root, with Python 3.12 or later:

```bash
# Source inspection; no PyTorch or Buddy build required.
python scripts/pytorch_op_coverage/run_coverage.py

# PyTorch 2.10.0: eager/export checks and Transformer/MoE block traces.
python scripts/pytorch_op_coverage/run_coverage.py \
  --mode trace --workloads --out-dir scripts/pytorch_op_coverage/out/trace

# After building Buddy with BUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON:
export PYTHONPATH=$PWD/build/python_packages:$PWD/llvm/build/tools/mlir/python_packages/mlir_core:$PYTHONPATH
# For an out-of-tree LLVM build, also set LLVM_LIBS_DIR to its runtime library directory.
python scripts/pytorch_op_coverage/run_coverage.py \
  --mode live --workloads --min-coverage 90 --out-dir /tmp/buddy-coverage-live
```

Each run writes `pytorch_op_coverage.json` and `pytorch_op_coverage.md`.
See the [live CPU snapshot](../scripts/pytorch_op_coverage/out/live/pytorch_op_coverage.md)
for measured results and remaining failures.
Keep separate output directories for static, trace and live measurements.
Workers run in separate processes, with `--timeout 120` seconds per worker.
Failures and timeouts retain the last recorded stage.

Exit **0** means the requested checks and any specified threshold passed.
Exit **1** means a case/schema failure, timeout, source change during the run,
or failed coverage threshold. Exit **2** means missing dependencies or invalid
input. Skipped cases alone do not fail a run; they never count as validated.
A successful static or trace command is not a successful live coverage gate.

## Read the report

The target file `scripts/pytorch_op_coverage/data/target_ops_v1.json` defines
**106** unique `namespace::operator.overload` identities and their real PyTorch
2.10.0 schemas. Membership in multiple model families does not duplicate an
operator in the overall denominator.

The report keeps three kinds of evidence independent:

- **Source:** direct frontend-map lookup, registered lowering dialects, candidate
  aliases, and limitations/review flags. Registrations are a union of dialects,
  not a claim that every dialect or the selected CPU path works.
- **Cases:** export, import, lowering, compilation/JIT creation, execution and
  numerical comparison, with input shapes, dtypes, strides and failure reasons.
- **Workloads:** observed operators in representative blocks, including entries
  outside the fixed target set. Workload success does not automatically credit
  individual operators.

`validated_for_profile` counts an operator only when **all required cases pass
all execution stages**, with no retained limitation/review flag. Failed,
skipped, untested and limited operators remain in the denominator. Evidence
categories overlap; their counts should not be added together.

The `cpu-export-v5` profile configures all 106 operators with three cases each
(318 required cases, unchanged from `cpu-export-v4`):
two small float32 shapes and one float64 shape; integer-only operators retain
integer inputs. Factory cases have no tensor inputs and specify output dtype
and shape explicitly; `_to_copy` cases convert between f32 and f64. Vision cases
use small NCHW inputs with rectangular shapes. Seed is 0. Floating outputs use
rtol `1e-4` and atol `1e-5`;
integer/bool outputs require exact equality. Output arity, order and dtype
must match. Coverage applies only to the declared small CPU cases. The
`contiguous` fixture uses a strided input. New indexing and scatter fixtures
include duplicate indices where accumulation is defined; attention uses causal
rectangular sequences. Dynamic-output and unsupported cases remain in the
measurement and denominator.

Reshape, contiguous, argsort and softmax cases retain their previous result and
also check a boundary result: an internal strided slice, non-last sorting axis
with descending order, or a transposed softmax with dtype conversion. Each case
must match every returned result.

The live path uses strict `torch.export`, Buddy `_compile_fx`, TOSA-priority
registries and `dynamo_run()`, with external calls disabled and no user-supplied
decomposition table. Buddy AOT functionalization may still rewrite operators;
JSON records the resulting Buddy graph operation classes. An operator removed
or rewritten during export is skipped. No eager result substitutes for JIT
execution. Output flattening is reused from the existing ATen coverage runner;
metadata-only passes and its skip list are not inherited.

## Workloads and reproducibility

`--workloads` adds an attention/residual/layer-normalization/MLP block and a
four-expert, top-2 MoE block with dispatch, expert GEMMs and scatter-add combine.
These small fixed-shape blocks do not establish full-model support. Inventory
counts are maximum FX node counts across profiles, not runtime frequencies.
Review outside-target operators before adding them in a new target-set version.

Reports record source hashes, full Git revision and dirty state, runtime
versions and input settings. Live mode checks the loaded Buddy Python sources
against the inspected repository, ignoring line-ending differences and a
wheel-generated package initializer. Native builds must also match the intended
Buddy/LLVM revisions. Regenerate after source changes. The v1 migration
removes two Buddy-specific cache helpers from v0's 108 entries and corrects the
Prim namespace and softmax overload. The original v0 file is historical, not a
runner input; its percentages are not directly comparable with v1. Retained v0
review flags are conservative exclusions, not all confirmed restrictions.

The live snapshot uses Linux x86-64, Python 3.12.3, NumPy 2.4.2 and
PyTorch 2.10.0+cpu.
Its native binaries come from the official
[Buddy nightly v0.0.10.dev20260917](https://github.com/buddy-compiler/buddy-mlir/releases/tag/nightly/v0.0.10.dev20260917),
at Buddy revision `669977354e8e47dc3d084e40c9317ef4dfccbaa7` and LLVM revision
`2d26d272a0ff74b8c81eac0607b07f98b82ecc46`. The installed Python frontend uses
this checkout, including the lowerings and CPU layout handling described below;
its sources are checked against the repository. This is a prebuilt-runtime
measurement, not a clean native rebuild.

## Verify and accept

```bash
python -m unittest discover -s scripts/pytorch_op_coverage -p 'test_*.py' -v
```

Standard-library tests check accounting and failure reporting. With PyTorch,
the suite also checks export fixtures, an independent MoE reference and adapter
stage handling using a fake backend. Fake-backend tests are not Buddy coverage.
The core tests also run through the repository's Python lit suite. Apply the
repository's pinned Ruff checks before submission.

The CPU regressions in `tests/Python/JIT/` cover:

- `to_copy_bool.py`, `dtype_views.py`: casts, range precision, broadcasts and
  strided slice updates, including empty outputs and signed zero.
- `gelu_layer_norm.py`, `mean_stack_clamp.py`: activation/reduction results,
  auxiliary layer-norm outputs, default/negative axes and scalar clamp bounds.
- `index_updates.py`, `scatter_reductions.py`: duplicate-index accumulation,
  include-self modes, unchanged inputs and runtime bounds assertions.
- `slice_select.py`, `layout_sort_softmax.py`: offset/strided layouts,
  scalar/empty results, sort axes and stable ties, NaNs and softmax dtype changes.
- `causal_attention.py`: attention output and log-sum-exp for causal/noncausal
  f32/f64 inputs, unequal sequence lengths and custom scales.

For example, run `python tests/Python/JIT/layout_sort_softmax.py` in the same
Buddy environment as the live measurement. Existing FileCheck tests cover
the corresponding import IR; JIT tests provide numerical evidence.

These lowerings target static CPU inputs. Mean supports f32/f64 without dtype
conversion; layer norm requires positive static shapes. Stack requires matching
shapes and dtypes; scalar clamp supports f32/f64/i32/i64 and rejects overflowing
integer bounds. Scatter mean remains unsupported. Additional restrictions are
retained in the target file and excluded from the strict numerator.

The CPU pipeline preserves runtime offsets and strides. Matmul vectorization
runs only when all relevant operands have proven unit minor strides; otherwise
the module uses loop lowering. The Dynamo adapter copies inputs to contiguous
storage, while direct Graph callers retain general layouts unless they explicitly
promise contiguous inputs. Reshape/clone may add copies, and sorting uses
sequential adjacent swaps. Full-model performance has not been validated.

For a 90% gate on this 106-entry set, at least **96 operators** must qualify.
Resolve review flags with evidence; do not remove them merely to raise the score.
The gate is only part of issue #911 acceptance: representative model execution,
correctness regressions for prioritized missing operators, and reproduction on
a built Buddy runtime remain required. Inspect report source hashes before
comparing runs; configured or exported cases alone do not establish support.
