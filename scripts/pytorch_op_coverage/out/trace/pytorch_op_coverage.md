# PyTorch operator coverage

- Mode: **trace**; run: **completed**; exit: **0**
- Generated (UTC): `2026-09-29T10:18:55.265216+00:00`
- Target: **Buddy Target Op Set v1** / `1.0.0`; **106** unique operators
- Source: `d173db9e65670f4eeca89107d025418742a16b57`; dirty: `True`
- Source SHA-256: `409b9e8345e0b47ff061d40550aff647e22e36c4581bd7ee3fd821accf59eb6f`
- Profile: `cpu-export-v13`; seed 0; rtol 1e-4; atol 1e-5; external calls disabled

> Registration and export are not compile/correctness evidence. Untested, skipped, failed and limited operators stay in the denominator.

> v1 has 106 entries versus v0's 108: two Buddy cache helpers were removed; Prim namespace and softmax overload were corrected. Percentages are not directly comparable.

Export uses `buddy.compiler.export` for checked one-hot and pixel-unshuffle semantics. Actual custom targets are recorded per case; this profile does not claim the unadapted export path supports these boundaries.

Python: `3.12.3`; measured torch: `2.10.0+cpu`; schema snapshot torch: `2.10.0+cpu`.

## Coverage

| Evidence | Count | % of fixed denominator |
| --- | ---: | ---: |
| frontend_recognized | 100 | 94.34% |
| registered_lowering | 100 | 94.34% |
| alias_candidate | 4 | 3.77% |
| unmapped | 2 | 1.89% |
| known_limited | 0 | 0.0% |
| validated_for_profile | 0 | 0.0% |

Live validation: **not measured**. Confirmed end-to-end numerator: **0**.
Operators without an input contract: **0**.
A completed run is not the 90% gate; use `--mode live --min-coverage 90` for that gate.

MoE: **0/47** validated for profile (0.0%); **0** known limited.

## Execution evidence

| Stage | Passed cases |
| --- | ---: |
| exported | 318 |
| imported | 0 |
| lowered | 0 |
| compiled | 0 |
| executed | 0 |
| correctness | 0 |

Case outcomes: `{"blocked": 0, "failed": 0, "passed": 318, "skipped": 0, "timeout": 0}`

## Operator details

| Operator | Source evidence | Required/passed cases | Validated | Limitation / alias candidates |
| --- | --- | ---: | --- | --- |
| `aten::mm.default` | registered_lowering | 3/0 | False |  |
| `aten::bmm.default` | registered_lowering | 3/0 | False |  |
| `aten::addmm.default` | registered_lowering | 3/0 | False |  |
| `aten::baddbmm.default` | registered_lowering | 3/0 | False |  |
| `aten::add.Tensor` | registered_lowering | 3/0 | False |  |
| `aten::mul.Tensor` | registered_lowering | 3/0 | False |  |
| `aten::div.Tensor` | registered_lowering | 3/0 | False |  |
| `aten::sub.Tensor` | registered_lowering | 3/0 | False |  |
| `aten::neg.default` | registered_lowering | 3/0 | False |  |
| `aten::pow.Tensor_Scalar` | registered_lowering | 3/0 | False |  |
| `aten::rsqrt.default` | registered_lowering | 3/0 | False |  |
| `aten::sqrt.default` | registered_lowering | 3/0 | False |  |
| `aten::exp.default` | registered_lowering | 3/0 | False |  |
| `aten::silu.default` | registered_lowering | 3/0 | False |  |
| `aten::gelu.default` | registered_lowering | 3/0 | False |  |
| `aten::relu.default` | registered_lowering | 3/0 | False |  |
| `aten::sigmoid.default` | registered_lowering | 3/0 | False |  |
| `aten::tanh.default` | registered_lowering | 3/0 | False |  |
| `aten::_softmax.default` | registered_lowering | 3/0 | False |  |
| `aten::native_layer_norm.default` | registered_lowering | 3/0 | False |  |
| `aten::mean.dim` | registered_lowering | 3/0 | False |  |
| `aten::sum.dim_IntList` | registered_lowering | 3/0 | False |  |
| `aten::amax.default` | registered_lowering | 3/0 | False |  |
| `aten::embedding.default` | registered_lowering | 3/0 | False |  |
| `aten::cat.default` | registered_lowering | 3/0 | False |  |
| `aten::stack.default` | registered_lowering | 3/0 | False |  |
| `aten::slice.Tensor` | registered_lowering | 3/0 | False |  |
| `aten::select.int` | registered_lowering | 3/0 | False |  |
| `aten::view.default` | registered_lowering | 3/0 | False |  |
| `aten::reshape.default` | alias_candidate | 3/0 | False | aten::view.default, aten::_unsafe_view.default |
| `aten::transpose.int` | registered_lowering | 3/0 | False |  |
| `aten::permute.default` | registered_lowering | 3/0 | False |  |
| `aten::unsqueeze.default` | registered_lowering | 3/0 | False |  |
| `aten::squeeze.dim` | registered_lowering | 3/0 | False |  |
| `aten::expand.default` | registered_lowering | 3/0 | False |  |
| `aten::repeat.default` | registered_lowering | 3/0 | False |  |
| `aten::clone.default` | registered_lowering | 3/0 | False |  |
| `aten::_to_copy.default` | registered_lowering | 3/0 | False |  |
| `prims::convert_element_type.default` | registered_lowering | 3/0 | False |  |
| `aten::where.self` | registered_lowering | 3/0 | False |  |
| `aten::masked_fill.Scalar` | registered_lowering | 3/0 | False |  |
| `aten::arange.start` | registered_lowering | 3/0 | False |  |
| `aten::arange.start_step` | registered_lowering | 3/0 | False |  |
| `aten::ones.default` | registered_lowering | 3/0 | False |  |
| `aten::zeros.default` | registered_lowering | 3/0 | False |  |
| `aten::full.default` | registered_lowering | 3/0 | False |  |
| `aten::scalar_tensor.default` | registered_lowering | 3/0 | False |  |
| `aten::_scaled_dot_product_flash_attention_for_cpu.default` | registered_lowering | 3/0 | False |  |
| `aten::index.Tensor` | registered_lowering | 3/0 | False |  |
| `aten::index_select.default` | registered_lowering | 3/0 | False |  |
| `aten::gather.default` | registered_lowering | 3/0 | False |  |
| `aten::scatter_add.default` | registered_lowering | 3/0 | False |  |
| `aten::slice_scatter.default` | registered_lowering | 3/0 | False |  |
| `aten::cumsum.default` | registered_lowering | 3/0 | False |  |
| `aten::eq.Scalar` | registered_lowering | 3/0 | False |  |
| `aten::eq.Tensor` | registered_lowering | 3/0 | False |  |
| `aten::ne.Scalar` | registered_lowering | 3/0 | False |  |
| `aten::gt.Scalar` | registered_lowering | 3/0 | False |  |
| `aten::lt.Scalar` | registered_lowering | 3/0 | False |  |
| `aten::le.Scalar` | registered_lowering | 3/0 | False |  |
| `aten::ge.Scalar` | registered_lowering | 3/0 | False |  |
| `aten::maximum.default` | registered_lowering | 3/0 | False |  |
| `aten::minimum.default` | registered_lowering | 3/0 | False |  |
| `aten::clamp.default` | registered_lowering | 3/0 | False |  |
| `aten::split.Tensor` | registered_lowering | 3/0 | False |  |
| `aten::split_with_sizes.default` | registered_lowering | 3/0 | False |  |
| `aten::unbind.int` | registered_lowering | 3/0 | False |  |
| `aten::contiguous.default` | alias_candidate | 3/0 | False | aten::clone.default |
| `aten::copy.default` | registered_lowering | 3/0 | False |  |
| `aten::lift_fresh_copy.default` | registered_lowering | 3/0 | False |  |
| `aten::topk.default` | registered_lowering | 3/0 | False |  |
| `aten::softmax.int` | alias_candidate | 3/0 | False | aten::_softmax.default |
| `aten::argsort.default` | unmapped | 3/0 | False |  |
| `aten::sort.default` | registered_lowering | 3/0 | False |  |
| `aten::argmax.default` | registered_lowering | 3/0 | False |  |
| `aten::argmin.default` | registered_lowering | 3/0 | False |  |
| `aten::_unsafe_index.Tensor` | registered_lowering | 3/0 | False |  |
| `aten::index_put.default` | registered_lowering | 3/0 | False |  |
| `aten::index_add.default` | registered_lowering | 3/0 | False |  |
| `aten::index_copy.default` | registered_lowering | 3/0 | False |  |
| `aten::scatter.src` | registered_lowering | 3/0 | False |  |
| `aten::scatter.value` | registered_lowering | 3/0 | False |  |
| `aten::scatter.reduce` | registered_lowering | 3/0 | False |  |
| `aten::scatter.value_reduce` | registered_lowering | 3/0 | False |  |
| `aten::scatter_reduce.two` | registered_lowering | 3/0 | False |  |
| `aten::masked_scatter.default` | registered_lowering | 3/0 | False |  |
| `aten::masked_select.default` | registered_lowering | 3/0 | False |  |
| `aten::nonzero.default` | registered_lowering | 3/0 | False |  |
| `aten::nonzero_static.default` | registered_lowering | 3/0 | False |  |
| `aten::one_hot.default` | unmapped | 3/0 | False |  |
| `aten::bincount.default` | registered_lowering | 3/0 | False |  |
| `aten::cumprod.default` | registered_lowering | 3/0 | False |  |
| `aten::repeat_interleave.self_int` | registered_lowering | 3/0 | False |  |
| `aten::repeat_interleave.Tensor` | registered_lowering | 3/0 | False |  |
| `aten::convolution.default` | registered_lowering | 3/0 | False |  |
| `aten::avg_pool2d.default` | registered_lowering | 3/0 | False |  |
| `aten::_adaptive_avg_pool2d.default` | registered_lowering | 3/0 | False |  |
| `aten::max_pool2d_with_indices.default` | registered_lowering | 3/0 | False |  |
| `aten::upsample_bilinear2d.vec` | registered_lowering | 3/0 | False |  |
| `aten::upsample_nearest2d.vec` | registered_lowering | 3/0 | False |  |
| `aten::grid_sampler_2d.default` | registered_lowering | 3/0 | False |  |
| `aten::pad.default` | alias_candidate | 3/0 | False | aten::constant_pad_nd.default |
| `aten::constant_pad_nd.default` | registered_lowering | 3/0 | False |  |
| `aten::reflection_pad2d.default` | registered_lowering | 3/0 | False |  |
| `aten::pixel_shuffle.default` | registered_lowering | 3/0 | False |  |
| `aten::pixel_unshuffle.default` | registered_lowering | 3/0 | False |  |

## Representative block workloads

These are small fixed-shape blocks, not full-model acceptance tests. Trace success only validates eager execution and export.

| Workload | Case | Status | Stage results | Reason |
| --- | --- | --- | --- | --- |
| transformer_block | small-f32 | passed | exported=passed, imported=not_run, lowered=not_run, compiled=not_run, executed=not_run, correctness=not_run |  |
| transformer_block | rect-f32 | passed | exported=passed, imported=not_run, lowered=not_run, compiled=not_run, executed=not_run, correctness=not_run |  |
| transformer_block | small-f64 | passed | exported=passed, imported=not_run, lowered=not_run, compiled=not_run, executed=not_run, correctness=not_run |  |
| moe_block | small-f32 | passed | exported=passed, imported=not_run, lowered=not_run, compiled=not_run, executed=not_run, correctness=not_run |  |
| moe_block | rect-f32 | passed | exported=passed, imported=not_run, lowered=not_run, compiled=not_run, executed=not_run, correctness=not_run |  |
| moe_block | small-f64 | passed | exported=passed, imported=not_run, lowered=not_run, compiled=not_run, executed=not_run, correctness=not_run |  |

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
