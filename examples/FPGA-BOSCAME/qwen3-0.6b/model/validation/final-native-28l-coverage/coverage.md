# Final native-K model operator coverage

Mapping status: PASS. Actual final graph: prefill 957 calls; decode 957 calls. This status covers operator classification and artifact/ABI provenance; it does not assert full-model FPGA numerical validation.

Sources and SHA256 values, every IR operation, semantic evidence, checkpoint weight, shape/dtype, C/LLVM ABI and call order are in `operator-mapping.json`. Per-kernel AME/RVV instructions are in `kernel-isa.json`.

| Semantic | Prefill operations | Decode operations |
|---|---:|---:|
| attention | 112 | 112 |
| attention_output_projection | 84 | 84 |
| down_projection | 84 | 84 |
| embedding | 1 | 1 |
| final_rmsnorm | 1 | 1 |
| gate_projection | 84 | 84 |
| input_rmsnorm | 28 | 28 |
| k_norm | 28 | 28 |
| k_projection | 84 | 84 |
| kv_cache | 112 | 112 |
| lm_head | 3 | 3 |
| post_attention_rmsnorm | 28 | 28 |
| q_norm | 28 | 28 |
| q_projection | 84 | 84 |
| silu | 28 | 28 |
| up_projection | 84 | 84 |
| v_projection | 84 | 84 |

Each projection has separate quantize, AME matmul and dequantize calls. Attention has QK, scale/mask, softmax and PV. KV includes layout and in-place cache writes.

## Remaining Buddy operations

| Operation group | Prefill | Decode |
|---|---:|---:|
| cache_position_or_mask_index_math | 92 | 92 |
| constant_or_storage | 1234 | 1234 |
| entry_return | 1 | 1 |
| gqa_expand_full_cache | 56 | 56 |
| layout_transpose | 113 | 113 |
| lm_head_last_token_slice | 1 | 0 |
| reshape_view | 688 | 688 |
| residual_add | 56 | 56 |
| rope_broadcast_multiply | 112 | 112 |
| rope_frequency_duplicate | 1 | 1 |
| rope_frequency_outer_product | 1 | 1 |
| rope_half_slice | 112 | 112 |
| rope_negated_half | 56 | 56 |
| rope_rotated_half_join | 112 | 112 |
| rope_rotated_sum | 56 | 56 |
| rope_table_scale | 2 | 2 |
| rope_trigonometric_table | 2 | 2 |
| swiglu_gate_multiply | 28 | 28 |
| unused_original_causal_mask | 28 | 28 |

No unclassified operations or large unreplaced dense operations are claimed only when the JSON status is PASS. The dense threshold is 1,000,000 MACs; it does not exempt memory movement from reporting.

GQA still expands both K and V over all 512 slots (4 MiB each, 56 expansions per entry = 224 MiB of logical output writes). Native QK removes a separate 4 MiB key transpose per layer; it does not remove GQA expansion.

RoPE frequency outer product is 1,024 MACs for prefill / 64 MACs for decode. Sin/cos, split-half rotations, residual adds, SwiGLU multiply, attention output transpose and last-token slice remain in Buddy. Their per-op ISA and timing have not been established by seeing RVV elsewhere in the ELF.

## Kernel-local ISA

| Kernel classification | Cases |
|---|---:|
| RVV floating-point dot | 4 |
| scalar arithmetic with RVV memory movement | 29 |
| scalar instructions | 2 |
| AME int8 matrix arithmetic | 11 |

Scalar arithmetic with RVV memory movement is explicitly distinct from RVV arithmetic. The archive contains kernel/ABI objects only; runtime, launch/test data and graph generic fallback are excluded.

Actual per-call descriptor offset/sizes/strides are resolved from final LLVM. Runtime fields retain entry-argument provenance, and unknown fields remain explicit. Logical tensor shapes are never substituted for unknown physical descriptors. The top-level JSON also states the exact W8A8 contract.

Reproduce from the repository root with the Python 3.11 environment matching the Buddy bindings:
```bash
PYTHONPATH=build-python/python_packages python3 -B examples/FPGA-BOSCAME/qwen3-0.6b/model/tools/final_operator_mapping.py --build examples/FPGA-BOSCAME/qwen3-0.6b/model/build/review-native-28l --layout examples/FPGA-BOSCAME/qwen3-0.6b/model/validation/weight-layout.json --output examples/FPGA-BOSCAME/qwen3-0.6b/model/validation/final-native-28l-coverage
```
