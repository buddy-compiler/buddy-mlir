# PyTorch operator coverage

- Mode: **live**; run: **failed**; exit: **1**
- Generated (UTC): `2026-09-23T08:52:57.552731+00:00`
- Target: **Buddy Target Op Set v1** / `1.0.0`; **106** unique operators
- Source: `7e36bc29faa4e2c4ea3081a83c2b56b5c49f742b`; dirty: `True`
- Source SHA-256: `e2b36e21164104f63614d16e594d0a789f4a7a89409c3012fe0bdbd7b5263d15`
- Profile: `cpu-export-v2`; seed 0; rtol 1e-4; atol 1e-5; external calls disabled

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
| known_limited | 15 | 14.15% |
| validated_for_profile | 52 | 49.06% |

Live validation: **requested**. Confirmed end-to-end numerator: **52**.
Operators without an input contract: **42**.
A completed run is not the 90% gate; use `--mode live --min-coverage 90` for that gate.

MoE: **23/47** validated for profile (48.94%); **10** known limited.

## Execution evidence

| Stage | Passed cases |
| --- | ---: |
| exported | 192 |
| imported | 189 |
| lowered | 179 |
| compiled | 179 |
| executed | 179 |
| correctness | 173 |

Case outcomes: `{"blocked": 0, "failed": 19, "passed": 173, "skipped": 42, "timeout": 0}`

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
| `aten::mean.dim` | registered_lowering | 3/0 | False |  |
| `aten::sum.dim_IntList` | registered_lowering | 3/3 | True |  |
| `aten::amax.default` | registered_lowering | 3/3 | True |  |
| `aten::embedding.default` | registered_lowering | 0/0 | False |  |
| `aten::cat.default` | registered_lowering | 3/3 | True |  |
| `aten::stack.default` | registered_lowering | 3/0 | False |  |
| `aten::slice.Tensor` | registered_lowering | 3/0 | False |  |
| `aten::select.int` | registered_lowering | 3/0 | False |  |
| `aten::view.default` | registered_lowering | 3/3 | True |  |
| `aten::reshape.default` | alias_candidate | 3/3 | False | Configured contiguous cases pass through AOT ViewOp. Non-contiguous reshape semantics remain untested. |
| `aten::transpose.int` | registered_lowering | 3/3 | True |  |
| `aten::permute.default` | registered_lowering | 3/3 | True |  |
| `aten::unsqueeze.default` | registered_lowering | 3/3 | True |  |
| `aten::squeeze.dim` | registered_lowering | 3/3 | True |  |
| `aten::expand.default` | registered_lowering | 3/2 | False |  |
| `aten::repeat.default` | registered_lowering | 3/3 | True |  |
| `aten::clone.default` | registered_lowering | 3/3 | True |  |
| `aten::_to_copy.default` | registered_lowering | 0/0 | False |  |
| `prims::convert_element_type.default` | registered_lowering | 3/3 | True |  |
| `aten::where.self` | registered_lowering | 3/3 | True |  |
| `aten::masked_fill.Scalar` | registered_lowering | 3/3 | True |  |
| `aten::arange.start` | registered_lowering | 0/0 | False |  |
| `aten::arange.start_step` | registered_lowering | 0/0 | False |  |
| `aten::ones.default` | registered_lowering | 0/0 | False |  |
| `aten::zeros.default` | registered_lowering | 0/0 | False |  |
| `aten::full.default` | registered_lowering | 0/0 | False |  |
| `aten::scalar_tensor.default` | registered_lowering | 0/0 | False |  |
| `aten::_scaled_dot_product_flash_attention_for_cpu.default` | registered_lowering | 0/0 | False | tosa.py:scaled_dot_product_flash_attention_for_cpu_op asserts dropout_p == 0 for explicit dropout arguments. |
| `aten::index.Tensor` | registered_lowering | 0/0 | False | Review pending: Advanced indexing / None / boolean masks may be partial. |
| `aten::index_select.default` | registered_lowering | 3/3 | True |  |
| `aten::gather.default` | registered_lowering | 3/3 | True |  |
| `aten::scatter_add.default` | registered_lowering | 3/3 | True |  |
| `aten::slice_scatter.default` | registered_lowering | 0/0 | False |  |
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
| `aten::clamp.default` | registered_lowering | 3/0 | False |  |
| `aten::split.Tensor` | registered_lowering | 0/0 | False |  |
| `aten::split_with_sizes.default` | registered_lowering | 0/0 | False |  |
| `aten::unbind.int` | registered_lowering | 0/0 | False |  |
| `aten::contiguous.default` | alias_candidate | 0/0 | False | Review pending: May be elided or cloned depending on strides; requires layout-specific cases. |
| `aten::copy.default` | registered_lowering | 0/0 | False |  |
| `aten::lift_fresh_copy.default` | registered_lowering | 0/0 | False |  |
| `aten::topk.default` | registered_lowering | 3/3 | False | linalg.py:topk_op requires static shapes and a static integer k; complex types are rejected. |
| `aten::softmax.int` | alias_candidate | 3/3 | False | Configured last-dimension cases pass through AOT SoftmaxOp. Other dimensions and optional output dtype remain untested. |
| `aten::argsort.default` | unmapped | 3/3 | False | Configured default-attribute cases pass through AOT SortOp. Other dimensions, descending order and non-contiguous inputs remain untested. |
| `aten::sort.default` | registered_lowering | 3/3 | True |  |
| `aten::argmax.default` | registered_lowering | 3/3 | True |  |
| `aten::argmin.default` | registered_lowering | 3/3 | True |  |
| `aten::_unsafe_index.Tensor` | registered_lowering | 0/0 | False |  |
| `aten::index_put.default` | registered_lowering | 0/0 | False | Review pending: Accumulate / advanced indexing may be partial. |
| `aten::index_add.default` | registered_lowering | 3/3 | True |  |
| `aten::index_copy.default` | registered_lowering | 3/3 | True |  |
| `aten::scatter.src` | registered_lowering | 0/0 | False |  |
| `aten::scatter.value` | registered_lowering | 0/0 | False |  |
| `aten::scatter.reduce` | registered_lowering | 0/0 | False | Review pending: Reduce mode / dtype coverage may be partial. |
| `aten::scatter.value_reduce` | registered_lowering | 0/0 | False |  |
| `aten::scatter_reduce.two` | registered_lowering | 0/0 | False | Review pending: Reduce mode / dtype coverage may be partial. |
| `aten::masked_scatter.default` | registered_lowering | 0/0 | False |  |
| `aten::masked_select.default` | registered_lowering | 0/0 | False |  |
| `aten::nonzero.default` | registered_lowering | 0/0 | False |  |
| `aten::nonzero_static.default` | registered_lowering | 0/0 | False |  |
| `aten::one_hot.default` | unmapped | 3/3 | False | Configured explicit-num_classes=4 cases pass through AOT decomposition after the bool-to-integer copy fix. Inferred class counts, empty inputs and invalid indices remain untested. |
| `aten::bincount.default` | unmapped | 3/0 | False | The cpu-export-v1 cases fail during AOT import with DynamicOutputShapeException; minlength does not fix the data-dependent output extent. |
| `aten::cumprod.default` | registered_lowering | 3/3 | True |  |
| `aten::repeat_interleave.self_int` | registered_lowering | 0/0 | False |  |
| `aten::repeat_interleave.Tensor` | registered_lowering | 0/0 | False |  |
| `aten::convolution.default` | registered_lowering | 0/0 | False |  |
| `aten::avg_pool2d.default` | registered_lowering | 0/0 | False |  |
| `aten::_adaptive_avg_pool2d.default` | registered_lowering | 0/0 | False |  |
| `aten::max_pool2d_with_indices.default` | registered_lowering | 0/0 | False |  |
| `aten::upsample_bilinear2d.vec` | registered_lowering | 0/0 | False |  |
| `aten::upsample_nearest2d.vec` | registered_lowering | 0/0 | False |  |
| `aten::grid_sampler_2d.default` | registered_lowering | 0/0 | False |  |
| `aten::pad.default` | alias_candidate | 0/0 | False | Review pending: General padding modes are not established by a constant_pad_nd mapping. |
| `aten::constant_pad_nd.default` | registered_lowering | 0/0 | False |  |
| `aten::reflection_pad2d.default` | registered_lowering | 0/0 | False |  |
| `aten::pixel_shuffle.default` | unmapped | 0/0 | False | Review pending: Vision path; may be unsupported. |
| `aten::pixel_unshuffle.default` | unmapped | 0/0 | False | Review pending: Vision path; may be unsupported. |

## Failed, blocked and untested cases

| Operator | Case | Status | Stage | Reason |
| --- | --- | --- | --- | --- |
| `aten::mean.dim` | small-f32 | failed | lowered | IndexError: list index out of range |
| `aten::mean.dim` | rect-f32 | failed | lowered | IndexError: list index out of range |
| `aten::mean.dim` | small-f64 | failed | lowered | IndexError: list index out of range |
| `aten::embedding.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::stack.default` | small-f32 | failed | lowered | TypeError: ConcatOp.__init__() takes 3 positional arguments but 4 were given |
| `aten::stack.default` | rect-f32 | failed | lowered | TypeError: ConcatOp.__init__() takes 3 positional arguments but 4 were given |
| `aten::stack.default` | small-f64 | failed | lowered | TypeError: ConcatOp.__init__() takes 3 positional arguments but 4 were given |
| `aten::slice.Tensor` | small-f32 | failed | correctness | AssertionError: The values for attribute 'shape' do not match: torch.Size([4, 7]) != torch.Size([4, 4]). |
| `aten::slice.Tensor` | rect-f32 | failed | correctness | AssertionError: The values for attribute 'shape' do not match: torch.Size([3, 4]) != torch.Size([3, 2]). |
| `aten::slice.Tensor` | small-f64 | failed | correctness | AssertionError: The values for attribute 'shape' do not match: torch.Size([4, 7]) != torch.Size([4, 4]). |
| `aten::select.int` | small-f32 | failed | correctness | AssertionError: Tensor-likes are not close!<br><br>Mismatched elements: 4 / 4 (100.0%)<br>Greatest absolute difference: 2.1035847663879395 at index (2,) (up to 1e-05 allowed)<br>Greatest relative difference: 3.6601779460906982 at index (1,) (up to 0.0001 allowed) |
| `aten::select.int` | rect-f32 | failed | correctness | AssertionError: Tensor-likes are not close!<br><br>Mismatched elements: 3 / 3 (100.0%)<br>Greatest absolute difference: 2.625518321990967 at index (0,) (up to 1e-05 allowed)<br>Greatest relative difference: 2.420898199081421 at index (0,) (up to 0.0001 allowed) |
| `aten::select.int` | small-f64 | failed | correctness | AssertionError: Tensor-likes are not close!<br><br>Mismatched elements: 4 / 4 (100.0%)<br>Greatest absolute difference: 1.8905123899519136 at index (3,) (up to 1e-05 allowed)<br>Greatest relative difference: 2.1217799663158794 at index (3,) (up to 0.0001 allowed) |
| `aten::expand.default` | small-f64 | failed | lowered | NotImplementedError: Unsupported element type! |
| `aten::_to_copy.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::arange.start` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::arange.start_step` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::ones.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::zeros.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::full.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::scalar_tensor.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::_scaled_dot_product_flash_attention_for_cpu.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::index.Tensor` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::slice_scatter.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::clamp.default` | small-f32 | failed | lowered | TypeError: ClampOp.__init__() takes 5 positional arguments but 7 were given |
| `aten::clamp.default` | rect-f32 | failed | lowered | TypeError: ClampOp.__init__() takes 5 positional arguments but 7 were given |
| `aten::clamp.default` | small-f64 | failed | lowered | TypeError: ClampOp.__init__() takes 5 positional arguments but 7 were given |
| `aten::split.Tensor` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::split_with_sizes.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::unbind.int` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::contiguous.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::copy.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::lift_fresh_copy.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::_unsafe_index.Tensor` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::index_put.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::scatter.src` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::scatter.value` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::scatter.reduce` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::scatter.value_reduce` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::scatter_reduce.two` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::masked_scatter.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::masked_select.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::nonzero.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::nonzero_static.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::bincount.default` | small-f32 | failed | imported | DynamicOutputShapeException: aten.bincount.default<br><br>While executing %bincount : [num_users=2] = call_function[target=torch.ops.aten.bincount.default](args = (%args_0, None, 4), kwargs = {})<br>Original traceback:<br>  File "/opt/venv/lib/python3.12/site-packages/torch/_dynamo/functional_export.py", line 216, in forward<br>    res = self._export_root(*args, **kwargs)<br>  File "/workspace/scripts/pytorch_op_coverage/probes.py", line 220, in forward<br>    return fn(*args)<br>  File "/workspace/scripts/pytorch_op_coverage/probes.py", line 197, in <lambda><br>    fn, args = lambda a: op(a, minlength=4), (torch.tensor([0, 3, 1, 0]),)<br><br>Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs) |
| `aten::bincount.default` | rect-f32 | failed | imported | DynamicOutputShapeException: aten.bincount.default<br><br>While executing %bincount : [num_users=2] = call_function[target=torch.ops.aten.bincount.default](args = (%args_0, None, 4), kwargs = {})<br>Original traceback:<br>  File "/opt/venv/lib/python3.12/site-packages/torch/_dynamo/functional_export.py", line 216, in forward<br>    res = self._export_root(*args, **kwargs)<br>  File "/workspace/scripts/pytorch_op_coverage/probes.py", line 220, in forward<br>    return fn(*args)<br>  File "/workspace/scripts/pytorch_op_coverage/probes.py", line 197, in <lambda><br>    fn, args = lambda a: op(a, minlength=4), (torch.tensor([0, 3, 1, 0]),)<br><br>Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs) |
| `aten::bincount.default` | small-f64 | failed | imported | DynamicOutputShapeException: aten.bincount.default<br><br>While executing %bincount : [num_users=2] = call_function[target=torch.ops.aten.bincount.default](args = (%args_0, None, 4), kwargs = {})<br>Original traceback:<br>  File "/opt/venv/lib/python3.12/site-packages/torch/_dynamo/functional_export.py", line 216, in forward<br>    res = self._export_root(*args, **kwargs)<br>  File "/workspace/scripts/pytorch_op_coverage/probes.py", line 220, in forward<br>    return fn(*args)<br>  File "/workspace/scripts/pytorch_op_coverage/probes.py", line 197, in <lambda><br>    fn, args = lambda a: op(a, minlength=4), (torch.tensor([0, 3, 1, 0]),)<br><br>Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs) |
| `aten::repeat_interleave.self_int` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::repeat_interleave.Tensor` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::convolution.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::avg_pool2d.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::_adaptive_avg_pool2d.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::max_pool2d_with_indices.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::upsample_bilinear2d.vec` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::upsample_nearest2d.vec` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::grid_sampler_2d.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::pad.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::constant_pad_nd.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::reflection_pad2d.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::pixel_shuffle.default` | unconfigured | skipped | — | No explicit input contract yet |
| `aten::pixel_unshuffle.default` | unconfigured | skipped | — | No explicit input contract yet |

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
