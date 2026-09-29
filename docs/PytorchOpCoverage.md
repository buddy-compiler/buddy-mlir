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
for measured results and scope. The checked-in `cpu-export-v13` snapshot passes
the 100% gate: **106/106** target operators, **318/318** operator cases and all
six block cases. The MoE subset is **47/47**. Full-model and clean-build
acceptance are separate, as described below.
The `cpu-export-v13` inference profile uses `buddy.compiler.export.export` for
one-hot and pixel-unshuffle. This opt-in wrapper preserves runtime label checks
and inferred class counts through AOT, and aligns empty unshuffle metadata with
PyTorch eager. Other operators use ordinary strict export. JSON records both
the requested ATen identity and the actual `buddy_export` custom target;
percentages apply to this adapted path, not the unadapted export path.

Applications using these operations can call
`from buddy.compiler.export import export`, then `export(module, args, strict=True)`.
The wrapper leaves other operations unchanged and does not enable training.
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

The `cpu-export-v10` profile configures all 106 operators with three cases each
(318 required cases, unchanged from `cpu-export-v9`):
two small float32 shapes and one float64 shape; integer-only operators retain
integer inputs. Factory cases have no tensor inputs and specify output dtype
and shape explicitly; `_to_copy` cases convert between f32 and f64. Vision cases
use small NCHW inputs with rectangular shapes. Seed is 0. Floating outputs use
rtol `1e-4` and atol `1e-5`;
integer/bool outputs require exact equality. Output arity, order and dtype
must match. Expected NaNs must appear in the same positions; missing or extra
NaNs fail comparison. Coverage applies only to the declared small CPU cases. The
`contiguous` fixture uses a strided input. New indexing and scatter fixtures
include duplicate indices where accumulation is defined; attention uses causal
rectangular sequences. Dynamic-output and unsupported cases remain in the
measurement and denominator.

Reshape, contiguous, argsort and softmax cases retain their previous result and
also check a boundary result: an internal strided slice, non-last sorting axis
with descending order, or a transposed softmax with dtype conversion. Each case
must match every returned result.

Scatter-reduce cases preserve the previous sum result and additionally check
mean with both include-self modes. CPU regressions cover duplicate indices,
untouched entries, empty indices, and integer mean rounding toward negative
infinity for f32/f64/i32/i64 inputs.

Pixel-shuffle cases retain their original output and also check an internal
strided slice. Pixel-unshuffle cases check both non-empty and empty outputs
through the declared export adapter. Regressions verify downstream transposes
and reject conflicting metadata from the unadapted export path.

Attention cases preserve the causal unmasked outputs and add rank-2/rank-4
additive masks, including a fully masked row. Both attention output and
log-sum-exp must match PyTorch; fully masked rows return zeros. Regressions
also cover mask broadcasting, strided masks and f32 masks with f64 queries.

Top-k cases retain the original result and add a non-last axis on a strided
view and a NaN input. The lowering selects distinct indices and supports static
shapes and k. Regression checks allow PyTorch's unspecified ordering for ties
and `sorted=False`, while requiring correct selected values and valid indices.

The live path uses strict `torch.export`, Buddy `_compile_fx`, TOSA-priority
registries and `dynamo_run()`, with external calls disabled and no user-supplied
decomposition table. Buddy AOT functionalization may still rewrite operators;
JSON records the resulting Buddy graph operation classes. An operator removed
or rewritten during export is skipped unless it matches one of the two declared
export-adapter targets. No eager result substitutes for JIT
execution. Output flattening is reused from the existing ATen coverage runner;
metadata-only passes and its skip list are not inherited.
The AOT tracing inputs retain the exported placeholder metadata and its symbolic
shape environment; real inputs still provide runtime parameter storage. This
allows data-dependent output lengths without specializing them to the example.

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

### Model checks

Model validation is separate from operator coverage and never changes its
numerator. With the same Buddy runtime and `transformers==4.57.1` installed:

```bash
python scripts/pytorch_op_coverage/run_model_validation.py \
  --model bert-tiny --out-dir /tmp/buddy-models
python scripts/pytorch_op_coverage/run_model_validation.py \
  --model mixtral-block --out-dir /tmp/buddy-models
```

The BERT check downloads the pinned `prajjwal1/bert-tiny` checkpoint and compares
hidden states and pooled outputs for two padded text batches, reusing one
compiled graph. Mixtral checks use the standard implementation with small random
weights and four experts/top-2 routing. Additional choices are
`mixtral-block-nonstrict`, `mixtral-decoder` and `mixtral-decoder-mask`; the last
passes an explicit four-dimensional causal mask to isolate mask construction
from expert dispatch. These checks do not establish task accuracy or performance.
`olmoe-block` tests a standard OLMoE block with a fixed expert loop, separating
data-dependent token selection from Mixtral's data-dependent expert loop.

Each check writes JSON and a worker log, records model/runtime/source versions,
and uses an isolated process with a configurable `--timeout` (300 seconds by
default). Exit 0 requires successful numerical comparison; failures and timeouts
exit 1 and retain the failing stage. In the tested stack, standard Mixtral
dispatch fails during export at the data-dependent expert loop; passing the
small fixed-shape MoE coverage block does not resolve that limitation.
Preserving export symbols lets OLMoE pass AOT import; it currently fails during
lowering of reshape with data-dependent output dimensions; dynamic advanced
indexing now lowers successfully.

### Regression checks

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
- `advanced_indexing.py`: broadcast placement, negative indices, strided views,
  scalar/broadcast update values, duplicate-index accumulation, dynamic boolean
  masks and mixed integer/boolean indices, with runtime bounds assertions.
- `index_updates.py`, `scatter_reductions.py`: duplicate-index accumulation,
  include-self modes, unchanged inputs and runtime bounds assertions.
- `slice_select.py`, `layout_sort_softmax.py`: offset/strided layouts,
  scalar/empty results, sort axes and stable ties, NaNs and softmax dtype changes.
- `causal_attention.py`: attention output and log-sum-exp for causal/noncausal
  f32/f64 inputs, unequal sequence lengths, custom scales and additive masks.
- `pixel_rearrange.py`: exact pixel shuffle/unshuffle results across ranks,
  factors and strided views; empty outputs and downstream transposes, with
  rejection of invalid dimensions and conflicting raw export metadata.
- `one_hot.py`: scalar/empty explicit outputs, inferred runtime class counts,
  reuse with changed labels, strided inputs and runtime invalid-label assertions.
- `topk.py`: NaNs, infinities, integer limits, ties, scalar/empty outputs,
  arbitrary axes and strided views; input preservation and index uniqueness.
- `numeric_copy.py`: conversions among bool/i8/i32/i64/f32/f64, scalar and
  empty inputs, strided views, NaNs in boolean conversion and runtime range
  checks before float-to-integer conversion.
- `bincount.py`: runtime-varying bucket counts, weighted/unweighted and empty
  inputs, minlength, strided views and invalid-index runtime assertions.
- `spatial_resampling.py`: batched/unbatched 1D reflection with negative pads,
  nearest/bilinear interpolation, explicit sizes, fractional scales, corner
  alignment and strided views; all four padding modes in 1D/2D/3D, including
  negative padding and batched/unbatched inputs.
- `integer_float_cast.py`: boolean/signed-integer conversion to f32/f64,
  extreme values, strided views and floating mask arithmetic.
- `moe_routing.py`: reuse one compiled coverage block across changed inputs,
  routing weights and expert weights, including unused and unevenly loaded
  experts. This does not establish dynamic-output dispatch support.
- `nonzero_dynamic.py`: nonzero and masked-select outputs with runtime-varying
  lengths, including empty and scalar inputs, using one compiled graph per
  input shape/dtype.

For example, run `python tests/Python/JIT/layout_sort_softmax.py` in the same
Buddy environment as the live measurement. Existing FileCheck tests cover
the corresponding import IR; JIT tests provide numerical evidence.

These lowerings target static CPU inputs. Bincount produces a runtime-sized
output; arbitrary downstream dynamic-shape operations are not established.
Float-to-integer copies reject nonfinite or out-of-range truncated values at
runtime, where native conversion results are not portable. Mean supports f32/f64 without dtype
conversion; layer norm requires positive static shapes. Stack requires matching
shapes and dtypes; scalar clamp supports f32/f64/i32/i64 and rejects overflowing
integer bounds. Scatter mean uses floor rounding for integer inputs. Any retained
target-file review flags exclude an operator from the profile numerator. In the
pinned stack, the opt-in export wrapper is required for inferred one-hot,
invalid-label checks and correct empty pixel-unshuffle metadata. Inferred
one-hot has a runtime-sized output; arbitrary downstream dynamic-shape
operations are not established. The unadapted export path retains these gaps.

The CPU pipeline preserves runtime offsets and strides. Matmul vectorization
runs only when all relevant operands have proven unit minor strides; otherwise
the module uses loop lowering. The Dynamo adapter copies inputs to contiguous
storage, while direct Graph callers retain general layouts unless they explicitly
promise contiguous inputs. Reshape/clone may add copies, and sorting uses
sequential adjacent swaps. Advanced index updates also use sequential loops
to preserve duplicate-index accumulation order. Full-model performance has not been validated.

For a 90% gate on this 106-entry set, at least **96 operators** must qualify.
A 100% claim requires all **106 operators** to qualify with no retained review
flags; passing all 318 configured cases alone is insufficient.
Resolve review flags with evidence; do not remove them merely to raise the score.
The gate is only part of issue #911 acceptance: representative model execution,
correctness regressions for prioritized missing operators, and reproduction on
a built Buddy runtime remain required. Inspect report source hashes before
comparing runs; configured or exported cases alone do not establish support.
