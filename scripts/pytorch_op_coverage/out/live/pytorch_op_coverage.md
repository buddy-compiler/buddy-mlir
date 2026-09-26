# PyTorch operator coverage

- Mode: **live**; run: **failed**; exit: **1**
- Generated (UTC): `2026-09-26T09:37:26.348458+00:00`
- Target: **Buddy Target Op Set v1** / `1.0.0`; **106** unique operators
- Source: `0cde280f44ee339554872d8a77ea27e6e4e166bd`; dirty: `True`
- Source SHA-256: `71e9828484b4bcde09435b3565cf2b0b6a11f71da412ea3ee14b2517d675fdf1`
- Profile: `cpu-export-v5`; seed 0; rtol 1e-4; atol 1e-5; external calls disabled

> Registration and export are not compile/correctness evidence. Untested, skipped, failed and limited operators stay in the denominator.

> v1 has 106 entries versus v0's 108: two Buddy cache helpers were removed; Prim namespace and softmax overload were corrected. Percentages are not directly comparable.

Python: `3.12.3`; measured torch: `2.10.0+cpu`; schema snapshot torch: `2.10.0+cpu`.

## Coverage

| Evidence | Count | % of fixed denominator |
| --- | ---: | ---: |
| frontend_recognized | 97 | 91.51% |
| registered_lowering | 97 | 91.51% |
| alias_candidate | 4 | 3.77% |
| unmapped | 5 | 4.72% |
| known_limited | 10 | 9.43% |
| validated_for_profile | 92 | 86.79% |

Live validation: **requested**. Confirmed end-to-end numerator: **92**.
Operators without an input contract: **0**.
A completed run is not the 90% gate; use `--mode live --min-coverage 90` for that gate.

MoE: **39/47** validated for profile (82.98%); **6** known limited.

## Execution evidence

| Stage | Passed cases |
| --- | ---: |
| exported | 318 |
| imported | 303 |
| lowered | 300 |
| compiled | 294 |
| executed | 294 |
| correctness | 294 |

Case outcomes: `{"blocked": 0, "failed": 24, "passed": 294, "skipped": 0, "timeout": 0}`

## Operator details

| Operator | Source evidence | Required/passed cases | Validated | Limitation / alias candidates |
| --- | --- | ---: | --- | --- |
| `aten::mm.default` | registered_lowering | 3/3 | True |  |
| `aten::bmm.default` | registered_lowering | 3/3 | True |  |
| `aten::addmm.default` | registered_lowering | 3/3 | True |  |
| `aten::baddbmm.default` | registered_lowering | 3/3 | True |  |
| `aten::add.Tensor` | registered_lowering | 3/3 | True |  |
| `aten::mul.Tensor` | registered_lowering | 3/3 | True |  |
| `aten::div.Tensor` | registered_lowering | 3/3 | True |  |
| `aten::sub.Tensor` | registered_lowering | 3/3 | True |  |
| `aten::neg.default` | registered_lowering | 3/3 | True |  |
| `aten::pow.Tensor_Scalar` | registered_lowering | 3/3 | True |  |
| `aten::rsqrt.default` | registered_lowering | 3/3 | True |  |
| `aten::sqrt.default` | registered_lowering | 3/3 | True |  |
| `aten::exp.default` | registered_lowering | 3/3 | True |  |
| `aten::silu.default` | registered_lowering | 3/3 | True |  |
| `aten::gelu.default` | registered_lowering | 3/3 | True |  |
| `aten::relu.default` | registered_lowering | 3/3 | True |  |
| `aten::sigmoid.default` | registered_lowering | 3/3 | True |  |
| `aten::tanh.default` | registered_lowering | 3/3 | True |  |
| `aten::_softmax.default` | registered_lowering | 3/3 | True |  |
| `aten::native_layer_norm.default` | registered_lowering | 3/3 | True |  |
| `aten::mean.dim` | registered_lowering | 3/3 | True |  |
| `aten::sum.dim_IntList` | registered_lowering | 3/3 | True |  |
| `aten::amax.default` | registered_lowering | 3/3 | True |  |
| `aten::embedding.default` | registered_lowering | 3/3 | True |  |
| `aten::cat.default` | registered_lowering | 3/3 | True |  |
| `aten::stack.default` | registered_lowering | 3/3 | True |  |
| `aten::slice.Tensor` | registered_lowering | 3/3 | True |  |
| `aten::select.int` | registered_lowering | 3/3 | True |  |
| `aten::view.default` | registered_lowering | 3/3 | True |  |
| `aten::reshape.default` | alias_candidate | 3/3 | True | aten::view.default, aten::_unsafe_view.default |
| `aten::transpose.int` | registered_lowering | 3/3 | True |  |
| `aten::permute.default` | registered_lowering | 3/3 | True |  |
| `aten::unsqueeze.default` | registered_lowering | 3/3 | True |  |
| `aten::squeeze.dim` | registered_lowering | 3/3 | True |  |
| `aten::expand.default` | registered_lowering | 3/3 | True |  |
| `aten::repeat.default` | registered_lowering | 3/3 | True |  |
| `aten::clone.default` | registered_lowering | 3/3 | True |  |
| `aten::_to_copy.default` | registered_lowering | 3/3 | True |  |
| `prims::convert_element_type.default` | registered_lowering | 3/3 | True |  |
| `aten::where.self` | registered_lowering | 3/3 | True |  |
| `aten::masked_fill.Scalar` | registered_lowering | 3/3 | True |  |
| `aten::arange.start` | registered_lowering | 3/3 | True |  |
| `aten::arange.start_step` | registered_lowering | 3/3 | True |  |
| `aten::ones.default` | registered_lowering | 3/3 | True |  |
| `aten::zeros.default` | registered_lowering | 3/3 | True |  |
| `aten::full.default` | registered_lowering | 3/3 | True |  |
| `aten::scalar_tensor.default` | registered_lowering | 3/3 | True |  |
| `aten::_scaled_dot_product_flash_attention_for_cpu.default` | registered_lowering | 3/3 | False | Causal/noncausal f32/f64 and custom scales have CPU regressions. Mask broadcasting and fully masked rows remain unresolved. |
| `aten::index.Tensor` | registered_lowering | 3/3 | False | Review pending: Advanced indexing / None / boolean masks may be partial. |
| `aten::index_select.default` | registered_lowering | 3/3 | True |  |
| `aten::gather.default` | registered_lowering | 3/3 | True |  |
| `aten::scatter_add.default` | registered_lowering | 3/3 | True |  |
| `aten::slice_scatter.default` | registered_lowering | 3/3 | True |  |
| `aten::cumsum.default` | registered_lowering | 3/3 | True |  |
| `aten::eq.Scalar` | registered_lowering | 3/3 | True |  |
| `aten::eq.Tensor` | registered_lowering | 3/3 | True |  |
| `aten::ne.Scalar` | registered_lowering | 3/3 | True |  |
| `aten::gt.Scalar` | registered_lowering | 3/3 | True |  |
| `aten::lt.Scalar` | registered_lowering | 3/3 | True |  |
| `aten::le.Scalar` | registered_lowering | 3/3 | True |  |
| `aten::ge.Scalar` | registered_lowering | 3/3 | True |  |
| `aten::maximum.default` | registered_lowering | 3/3 | True |  |
| `aten::minimum.default` | registered_lowering | 3/3 | True |  |
| `aten::clamp.default` | registered_lowering | 3/3 | True |  |
| `aten::split.Tensor` | registered_lowering | 3/3 | True |  |
| `aten::split_with_sizes.default` | registered_lowering | 3/3 | True |  |
| `aten::unbind.int` | registered_lowering | 3/3 | True |  |
| `aten::contiguous.default` | alias_candidate | 3/3 | True | aten::clone.default |
| `aten::copy.default` | registered_lowering | 3/3 | True |  |
| `aten::lift_fresh_copy.default` | registered_lowering | 3/3 | True |  |
| `aten::topk.default` | registered_lowering | 3/3 | False | linalg.py:topk_op requires static shapes and a static integer k; complex types are rejected. |
| `aten::softmax.int` | alias_candidate | 3/3 | True | aten::_softmax.default |
| `aten::argsort.default` | unmapped | 3/3 | True |  |
| `aten::sort.default` | registered_lowering | 3/3 | True |  |
| `aten::argmax.default` | registered_lowering | 3/3 | True |  |
| `aten::argmin.default` | registered_lowering | 3/3 | True |  |
| `aten::_unsafe_index.Tensor` | registered_lowering | 3/3 | True |  |
| `aten::index_put.default` | registered_lowering | 3/3 | False | Review pending: Accumulate / advanced indexing may be partial. |
| `aten::index_add.default` | registered_lowering | 3/3 | True |  |
| `aten::index_copy.default` | registered_lowering | 3/3 | True |  |
| `aten::scatter.src` | registered_lowering | 3/3 | True |  |
| `aten::scatter.value` | registered_lowering | 3/3 | True |  |
| `aten::scatter.reduce` | registered_lowering | 3/3 | True |  |
| `aten::scatter.value_reduce` | registered_lowering | 3/3 | True |  |
| `aten::scatter_reduce.two` | registered_lowering | 3/3 | False | Sum/product/min/max include-self semantics have CPU regressions; mean remains explicitly unsupported. |
| `aten::masked_scatter.default` | registered_lowering | 3/3 | True |  |
| `aten::masked_select.default` | registered_lowering | 3/0 | False |  |
| `aten::nonzero.default` | registered_lowering | 3/0 | False |  |
| `aten::nonzero_static.default` | registered_lowering | 3/3 | True |  |
| `aten::one_hot.default` | unmapped | 3/3 | False | Configured explicit-num_classes=4 cases pass through AOT decomposition after the bool-to-integer copy fix. Inferred class counts, empty inputs and invalid indices remain untested. |
| `aten::bincount.default` | unmapped | 3/0 | False | The cpu-export-v1 cases fail during AOT import with DynamicOutputShapeException; minlength does not fix the data-dependent output extent. |
| `aten::cumprod.default` | registered_lowering | 3/3 | True |  |
| `aten::repeat_interleave.self_int` | registered_lowering | 3/3 | True |  |
| `aten::repeat_interleave.Tensor` | registered_lowering | 3/3 | True |  |
| `aten::convolution.default` | registered_lowering | 3/3 | True |  |
| `aten::avg_pool2d.default` | registered_lowering | 3/3 | True |  |
| `aten::_adaptive_avg_pool2d.default` | registered_lowering | 3/3 | True |  |
| `aten::max_pool2d_with_indices.default` | registered_lowering | 3/3 | True |  |
| `aten::upsample_bilinear2d.vec` | registered_lowering | 3/0 | False |  |
| `aten::upsample_nearest2d.vec` | registered_lowering | 3/0 | False |  |
| `aten::grid_sampler_2d.default` | registered_lowering | 3/3 | True |  |
| `aten::pad.default` | alias_candidate | 3/0 | False | Review pending: General padding modes are not established by a constant_pad_nd mapping. |
| `aten::constant_pad_nd.default` | registered_lowering | 3/3 | True |  |
| `aten::reflection_pad2d.default` | registered_lowering | 3/3 | True |  |
| `aten::pixel_shuffle.default` | unmapped | 3/0 | False | Review pending: Vision path; may be unsupported. |
| `aten::pixel_unshuffle.default` | unmapped | 3/0 | False | Review pending: Vision path; may be unsupported. |

## Failed, blocked and untested cases

| Operator | Case | Status | Stage | Reason |
| --- | --- | --- | --- | --- |
| `aten::masked_select.default` | small-f32 | failed | imported | DynamicOutputShapeException: aten.masked_select.default<br><br>While executing %masked_select : [num_users=2] = call_function[target=torch.ops.aten.masked_select.default](args = (%args_0, %args_1), kwargs = {})<br>Original traceback:<br>  File "/opt/venv/lib/python3.12/site-packages/torch/_dynamo/functional_export.py", line 216, in forward<br>    res = self._export_root(*args, **kwargs)<br>  File "/workspace/scripts/pytorch_op_coverage/probes.py", line 410, in forward<br>    return fn(*args)<br><br>Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs) |
| `aten::masked_select.default` | rect-f32 | failed | imported | DynamicOutputShapeException: aten.masked_select.default<br><br>While executing %masked_select : [num_users=2] = call_function[target=torch.ops.aten.masked_select.default](args = (%args_0, %args_1), kwargs = {})<br>Original traceback:<br>  File "/opt/venv/lib/python3.12/site-packages/torch/_dynamo/functional_export.py", line 216, in forward<br>    res = self._export_root(*args, **kwargs)<br>  File "/workspace/scripts/pytorch_op_coverage/probes.py", line 410, in forward<br>    return fn(*args)<br><br>Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs) |
| `aten::masked_select.default` | small-f64 | failed | imported | DynamicOutputShapeException: aten.masked_select.default<br><br>While executing %masked_select : [num_users=2] = call_function[target=torch.ops.aten.masked_select.default](args = (%args_0, %args_1), kwargs = {})<br>Original traceback:<br>  File "/opt/venv/lib/python3.12/site-packages/torch/_dynamo/functional_export.py", line 216, in forward<br>    res = self._export_root(*args, **kwargs)<br>  File "/workspace/scripts/pytorch_op_coverage/probes.py", line 410, in forward<br>    return fn(*args)<br><br>Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs) |
| `aten::nonzero.default` | small-f32 | failed | imported | DynamicOutputShapeException: aten.nonzero.default<br><br>While executing %nonzero : [num_users=2] = call_function[target=torch.ops.aten.nonzero.default](args = (%args_0,), kwargs = {})<br>Original traceback:<br>  File "/opt/venv/lib/python3.12/site-packages/torch/_dynamo/functional_export.py", line 216, in forward<br>    res = self._export_root(*args, **kwargs)<br>  File "/workspace/scripts/pytorch_op_coverage/probes.py", line 410, in forward<br>    return fn(*args)<br><br>Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs) |
| `aten::nonzero.default` | rect-f32 | failed | imported | DynamicOutputShapeException: aten.nonzero.default<br><br>While executing %nonzero : [num_users=2] = call_function[target=torch.ops.aten.nonzero.default](args = (%args_0,), kwargs = {})<br>Original traceback:<br>  File "/opt/venv/lib/python3.12/site-packages/torch/_dynamo/functional_export.py", line 216, in forward<br>    res = self._export_root(*args, **kwargs)<br>  File "/workspace/scripts/pytorch_op_coverage/probes.py", line 410, in forward<br>    return fn(*args)<br><br>Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs) |
| `aten::nonzero.default` | small-f64 | failed | imported | DynamicOutputShapeException: aten.nonzero.default<br><br>While executing %nonzero : [num_users=2] = call_function[target=torch.ops.aten.nonzero.default](args = (%args_0,), kwargs = {})<br>Original traceback:<br>  File "/opt/venv/lib/python3.12/site-packages/torch/_dynamo/functional_export.py", line 216, in forward<br>    res = self._export_root(*args, **kwargs)<br>  File "/workspace/scripts/pytorch_op_coverage/probes.py", line 410, in forward<br>    return fn(*args)<br><br>Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs) |
| `aten::bincount.default` | small-f32 | failed | imported | DynamicOutputShapeException: aten.bincount.default<br><br>While executing %bincount : [num_users=2] = call_function[target=torch.ops.aten.bincount.default](args = (%args_0, None, 4), kwargs = {})<br>Original traceback:<br>  File "/opt/venv/lib/python3.12/site-packages/torch/_dynamo/functional_export.py", line 216, in forward<br>    res = self._export_root(*args, **kwargs)<br>  File "/workspace/scripts/pytorch_op_coverage/probes.py", line 410, in forward<br>    return fn(*args)<br>  File "/workspace/scripts/pytorch_op_coverage/probes.py", line 385, in <lambda><br>    fn, args = lambda a: op(a, minlength=4), (torch.tensor([0, 3, 1, 0]),)<br><br>Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs) |
| `aten::bincount.default` | rect-f32 | failed | imported | DynamicOutputShapeException: aten.bincount.default<br><br>While executing %bincount : [num_users=2] = call_function[target=torch.ops.aten.bincount.default](args = (%args_0, None, 4), kwargs = {})<br>Original traceback:<br>  File "/opt/venv/lib/python3.12/site-packages/torch/_dynamo/functional_export.py", line 216, in forward<br>    res = self._export_root(*args, **kwargs)<br>  File "/workspace/scripts/pytorch_op_coverage/probes.py", line 410, in forward<br>    return fn(*args)<br>  File "/workspace/scripts/pytorch_op_coverage/probes.py", line 385, in <lambda><br>    fn, args = lambda a: op(a, minlength=4), (torch.tensor([0, 3, 1, 0]),)<br><br>Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs) |
| `aten::bincount.default` | small-f64 | failed | imported | DynamicOutputShapeException: aten.bincount.default<br><br>While executing %bincount : [num_users=2] = call_function[target=torch.ops.aten.bincount.default](args = (%args_0, None, 4), kwargs = {})<br>Original traceback:<br>  File "/opt/venv/lib/python3.12/site-packages/torch/_dynamo/functional_export.py", line 216, in forward<br>    res = self._export_root(*args, **kwargs)<br>  File "/workspace/scripts/pytorch_op_coverage/probes.py", line 410, in forward<br>    return fn(*args)<br>  File "/workspace/scripts/pytorch_op_coverage/probes.py", line 385, in <lambda><br>    fn, args = lambda a: op(a, minlength=4), (torch.tensor([0, 3, 1, 0]),)<br><br>Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs) |
| `aten::upsample_bilinear2d.vec` | small-f32 | failed | compiled | MLIRError: Failure while executing pass pipeline:<br>error: unknown: 'arith.index_cast' op operand #0 must be signless-non-zero-bitwidth-integer-like or memref of signless-integer, but got 'f32'<br> note: unknown: see current operation: %101 = "arith.index_cast"(%arg14) : (f32) -> index |
| `aten::upsample_bilinear2d.vec` | rect-f32 | failed | compiled | MLIRError: Failure while executing pass pipeline:<br>error: unknown: 'arith.index_cast' op operand #0 must be signless-non-zero-bitwidth-integer-like or memref of signless-integer, but got 'f32'<br> note: unknown: see current operation: %101 = "arith.index_cast"(%arg14) : (f32) -> index |
| `aten::upsample_bilinear2d.vec` | small-f64 | failed | compiled | MLIRError: Failure while executing pass pipeline:<br>error: unknown: 'arith.index_cast' op operand #0 must be signless-non-zero-bitwidth-integer-like or memref of signless-integer, but got 'f64'<br> note: unknown: see current operation: %101 = "arith.index_cast"(%arg14) : (f64) -> index |
| `aten::upsample_nearest2d.vec` | small-f32 | failed | compiled | MLIRError: Failure while executing pass pipeline:<br>error: unknown: 'arith.index_cast' op operand #0 must be signless-non-zero-bitwidth-integer-like or memref of signless-integer, but got 'f32'<br> note: unknown: see current operation: %24 = "arith.index_cast"(%arg1) : (f32) -> index |
| `aten::upsample_nearest2d.vec` | rect-f32 | failed | compiled | MLIRError: Failure while executing pass pipeline:<br>error: unknown: 'arith.index_cast' op operand #0 must be signless-non-zero-bitwidth-integer-like or memref of signless-integer, but got 'f32'<br> note: unknown: see current operation: %24 = "arith.index_cast"(%arg1) : (f32) -> index |
| `aten::upsample_nearest2d.vec` | small-f64 | failed | compiled | MLIRError: Failure while executing pass pipeline:<br>error: unknown: 'arith.index_cast' op operand #0 must be signless-non-zero-bitwidth-integer-like or memref of signless-integer, but got 'f32'<br> note: unknown: see current operation: %24 = "arith.index_cast"(%arg1) : (f32) -> index |
| `aten::pad.default` | small-f32 | failed | lowered | ValueError: not enough values to unpack (expected 3, got 2) |
| `aten::pad.default` | rect-f32 | failed | lowered | ValueError: not enough values to unpack (expected 3, got 2) |
| `aten::pad.default` | small-f64 | failed | lowered | ValueError: not enough values to unpack (expected 3, got 2) |
| `aten::pixel_shuffle.default` | small-f32 | failed | imported | KeyError: 'pixel_shuffle.default' |
| `aten::pixel_shuffle.default` | rect-f32 | failed | imported | KeyError: 'pixel_shuffle.default' |
| `aten::pixel_shuffle.default` | small-f64 | failed | imported | KeyError: 'pixel_shuffle.default' |
| `aten::pixel_unshuffle.default` | small-f32 | failed | imported | KeyError: 'pixel_unshuffle.default' |
| `aten::pixel_unshuffle.default` | rect-f32 | failed | imported | KeyError: 'pixel_unshuffle.default' |
| `aten::pixel_unshuffle.default` | small-f64 | failed | imported | KeyError: 'pixel_unshuffle.default' |

## Representative block workloads

These are small fixed-shape blocks, not full-model acceptance tests. Trace success only validates eager execution and export.

| Workload | Case | Status | Stage results | Reason |
| --- | --- | --- | --- | --- |
| transformer_block | small-f32 | passed | exported=passed, imported=passed, lowered=passed, compiled=passed, executed=passed, correctness=passed |  |
| transformer_block | rect-f32 | passed | exported=passed, imported=passed, lowered=passed, compiled=passed, executed=passed, correctness=passed |  |
| transformer_block | small-f64 | passed | exported=passed, imported=passed, lowered=passed, compiled=passed, executed=passed, correctness=passed |  |
| moe_block | small-f32 | passed | exported=passed, imported=passed, lowered=passed, compiled=passed, executed=passed, correctness=passed |  |
| moe_block | rect-f32 | passed | exported=passed, imported=passed, lowered=passed, compiled=passed, executed=passed, correctness=passed |  |
| moe_block | small-f64 | passed | exported=passed, imported=passed, lowered=passed, compiled=passed, executed=passed, correctness=passed |  |

## Observed workload operators and gaps

Counts are maximum FX node counts per workload across profiles, not runtime invocation frequencies. MoE entries are listed first. Outside-target operators do not change the denominator.

| Operator | Workload: FX nodes | In target | Source evidence |
| --- | --- | --- | --- |
| `aten::matmul.default` | transformer_block: 8, moe_block: 1 | False | outside_target |
| `aten::reshape.default` | moe_block: 4 | True | alias_candidate |
| `aten::index_select.default` | moe_block: 3 | True | registered_lowering |
| `aten::unsqueeze.default` | moe_block: 3 | True | registered_lowering |
| `aten::bmm.default` | moe_block: 2 | True | registered_lowering |
| `aten::div.Tensor` | transformer_block: 1, moe_block: 1 | True | registered_lowering |
| `aten::expand.default` | moe_block: 2 | True | registered_lowering |
| `aten::softmax.int` | transformer_block: 1, moe_block: 1 | True | alias_candidate |
| `aten::arange.default` | moe_block: 1 | False | outside_target |
| `aten::mul.Tensor` | moe_block: 1 | True | registered_lowering |
| `aten::scatter_add.default` | moe_block: 1 | True | registered_lowering |
| `aten::silu.default` | moe_block: 1 | True | registered_lowering |
| `aten::squeeze.dim` | moe_block: 1 | True | registered_lowering |
| `aten::sum.dim_IntList` | moe_block: 1 | True | registered_lowering |
| `aten::topk.default` | moe_block: 1 | True | registered_lowering |
| `aten::zeros_like.default` | moe_block: 1 | False | outside_target |
| `aten::add.Tensor` | transformer_block: 2 | True | registered_lowering |
| `aten::gelu.default` | transformer_block: 1 | True | registered_lowering |
| `aten::layer_norm.default` | transformer_block: 1 | False | outside_target |
| `aten::transpose.int` | transformer_block: 1 | True | registered_lowering |

## Remaining work

- Validate all configured cases on a built Buddy CPU runtime; retain native failures and timeouts.
- Add input contracts for untested operators; expand shapes, dtypes and attributes with regression tests.
- Prioritize MoE workload blockers; review trace-derived additions as a new target-set version.
