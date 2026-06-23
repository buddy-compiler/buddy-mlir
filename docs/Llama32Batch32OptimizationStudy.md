# Llama3.2 3B Batch32 Optimization Study

本文档记录 Buddy CLI Llama3.2 3B batch32 的性能优化实验。目标是用
可复现的单变量实验确认哪些优化应保留，哪些优化暂时关闭，并和
official StableHLO generation path 以及 tt-metal fallback 路径做同机对比。

## Benchmark Shape

- Model: `meta-llama/Llama-3.2-3B`
- Batch size: 32
- Prompt format: completion / plain prompt
- Prompt length: 32 tokens
- Max cache length: 128 tokens
- Decode length: 96 generated tokens
- Sampling: greedy, `--temperature 0`
- Main metric: tokens/s/user from wall time

Buddy optimization experiments should use the same runtime mode unless the row is
explicitly testing runtime behavior:

```bash
BUDDY_LLAMA31_IGNORE_EOS=1 cmake --build "$BUDDY_BUILD" --target llama32_tt_b32_rax

"$BUDDY_BUILD/bin/buddy-cli" \
  --model "$BUDDY_BUILD/models/llama32_tt_b32/llama32_tt_b32.rax" \
  --prompt-file "$BUDDY_REPO_ROOT/models/llama32_tt/llama32_3b_default_prompts.txt" \
  --prompt-length 32 \
  --batch-size 32 \
  --max-tokens 96 \
  --temperature 0 \
  --defer-decode-token-readback
```

`--defer-decode-token-readback` is treated as the benchmark runtime mode because
it removes per-step host token readback from the decode critical path. It
requires a fixed-length benchmark package built with `BUDDY_LLAMA31_IGNORE_EOS=1`.

## Reference Baselines

These are historical local P150A reference points. They should be rerun after
the final Buddy configuration is selected.

| Runtime | Scenario | tokens/s/user | Status |
| --- | --- | ---: | --- |
| official StableHLO generation path | Llama3.2 3B generation demo, batch32 | 1.328 | historical local run |
| tt-metal fallback | `tt_transformers` fallback Llama3.2 3B, batch32 | 27.98 | historical local run |
| Buddy CLI default | Default Llama3.2 b32 package, no deferred readback | 13.079 | historical local run |

## Historical Buddy Candidate Data

These rows were recovered from old trace JSON files. They are useful as a guide,
but the final table should be based on fresh runs from the current branch.

| Candidate | Build/runtime changes | full tokens/s/user | steady tokens/s/user | Decision |
| --- | --- | ---: | ---: | --- |
| `rmsnorm_defer` | `RMSNORM_FUSION` + deferred readback | 16.73 | 21.49 | rerun |
| `splitembed_keepweights_defer` | `SPLIT_EMBEDDING_WEIGHT`, `KEEP_STATIC_WEIGHT_INPUTS` + deferred readback | 22.21 | 23.60 | rerun |
| `packqkv_splitembed_keepweights_defer` | `PACK_QKV`, `SPLIT_EMBEDDING_WEIGHT`, `KEEP_STATIC_WEIGHT_INPUTS` + deferred readback | 23.17 | 24.45 | rerun |
| `packqkv_splitembed_keepweights_defer_smoke` | same as above, short smoke | 3.79 | 24.84 | smoke only |
| `lmhead_hifi2_defer_smoke` | `LMHEAD_HIFI2` + deferred readback | 10.69 | 21.51 | smoke only |
| `splitembed_keepweights_lmheadnofp32_defer_smoke` | no-fp32 LM-head candidate + deferred readback | 5.33 | 22.85 | smoke only |
| `splitembed_keepweights_argmaxtile_sc_defer_smoke` | `ARGMAX_TILE` with split embedding/static-weight candidate | 3.55 | 8.69 | likely reject |

## Single-Variable Experiment Order

Each step starts from the previous kept configuration and adds exactly one new
CMake optimization variable. If the new row is slower, unstable, or fails
correctness smoke, the variable is removed before moving on.

| Step | Added variable | Reason | Result | Decision |
| ---: | --- | --- | --- | --- |
| 0 | none | Current default package, benchmark-mode baseline | pending | pending |
| 1 | `BUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON` | Old trace showed a large gain with deferred readback | pending | pending |
| 2 | `BUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON` | Avoid using one tied weight layout for both embedding and LM-head | pending | pending |
| 3 | `BUDDY_LLAMA32_DECODE_KEEP_STATIC_WEIGHT_INPUTS=ON` | Avoid deallocating static decode weights each step | pending | pending |
| 4 | `BUDDY_LLAMA32_DECODE_PACK_QKV=ON` | Pre-pack Q/K/V static weights and remove runtime concat weight path | pending | pending |
| 5 | `BUDDY_LLAMA32_DECODE_FOLD_IDENTITY_MUL=ON` | Remove generated multiply-by-one artifacts | pending | pending |
| 6 | `BUDDY_LLAMA32_DECODE_LMHEAD_HIFI2=ON` | Try faster LM-head matmul accumulation mode | pending | pending |
| 7 | `BUDDY_LLAMA32_DECODE_ARGMAX_TILE=ON` | Let argmax consume tiled LM-head logits directly | pending | pending |
| 8 | `BUDDY_LLAMA32_DECODE_PACK_MLP_GATE_UP=ON` | Pre-pack MLP gate/up weights | pending | pending |
| 9 | `BUDDY_LLAMA32_DECODE_NATIVE_U32_TOKEN_IO=ON` | Avoid unnecessary token dtype/layout conversion | pending | pending |

Lower priority candidates, only after the table above is stable:

- `BUDDY_LLAMA32_DECODE_SPLIT_LM_HEAD=ON`
- `BUDDY_LLAMA32_DECODE_LM_HEAD_DRAM_PC=ON`
- `BUDDY_LLAMA32_DECODE_LM_HEAD_MCAST1D_PC=ON`
- `BUDDY_LLAMA32_DECODE_FUSE_CREATE_QKV_HEADS=ON`
- `BUDDY_LLAMA32_DECODE_FUSE_CONCAT_HEADS=ON`
- `BUDDY_LLAMA32_DECODE_FUSE_CONCAT_HEADS_SDPA_OUTPUT=ON`
- `BUDDY_LLAMA32_DECODE_PRECOMPUTE_ROPE=ON`

## Fresh Run Log

Fresh current-branch results go here.

| Step | Kept config | Added variable | Build status | Correctness smoke | full tokens/s/user | steady tokens/s/user | Notes |
| ---: | --- | --- | --- | --- | ---: | ---: | --- |
| 0 | none | none | pass | user0 text sane | 2.7089 | 5.2432 | `baseline_defer_20260622_143809.json`; all experimental decode flags off |
| 1 | none | `BUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON` | pass | user0 text sane | 5.3596 | 21.3568 | `rmsnorm_defer_20260622_145944.json`; keep |
| 2 | RMSNorm fusion | `BUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON` | pass on direct retry | user0 text sane | 3.5113 | 23.4155 | first scripted run hit runtime `Bus error`; direct retry of the same rax passed; keep for steady decode |
| 3 | RMSNorm fusion + split embedding | `BUDDY_LLAMA32_DECODE_KEEP_STATIC_WEIGHT_INPUTS=ON` | pass | user0 text sane | 4.5930 | 21.0980 | `rmsnorm_splitembed_keepweights_defer_20260622_154116.json`; reject for now |
| 4 | RMSNorm fusion + split embedding | `BUDDY_LLAMA32_DECODE_PACK_QKV=ON` | pass | user0 text sane | 4.4420 | 24.0845 | `rmsnorm_packqkv_splitembed_defer_20260622_155916.json`; keep |
| 5 | RMSNorm fusion + split embedding + packed QKV | `BUDDY_LLAMA32_DECODE_FOLD_IDENTITY_MUL=ON` | pass | user0 text sane | 6.2221 | 25.6916 | `rmsnorm_packqkv_splitembed_fold1_defer_20260622_161543.json`; postprocess reported `fold_identity_mul=0`, so treat as no-op/run variance |

## Repeated Run Log

For stable decisions, each candidate is rebuilt once and then run five times.
The table uses the mean of the five runtime traces.

| Candidate | Runs | Avg full tokens/s/user | Avg steady tokens/s/user | Steady min | Steady max | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `rmsnorm_packqkv_splitembed_defer` | 5 | 4.2372 | 25.4024 | 24.7144 | 26.1175 | effective best config so far |
| `rmsnorm_packqkv_splitembed_fold1_defer` | 0 | n/a | n/a | n/a | n/a | reject; postprocess reported `fold_identity_mul=0` and the first runtime run hung before producing timing output |
| `rmsnorm_packqkv_splitembed_lmhead_hifi2_defer` | 5 | 3.8732 | 24.2513 | 21.3154 | 25.6583 | reject; slower and less stable than effective best |
| `rmsnorm_packqkv_splitembed_argmax_tile_defer` | 0 | n/a | n/a | n/a | n/a | reject; runtime fails because TTNN multicore argmax requires `ROW_MAJOR` input layout but this graph feeds tiled logits |
| `rmsnorm_packqkv_splitembed_pack_mlp_defer` | 5 | 4.7820 | 24.8900 | 23.7698 | 25.4391 | reject; no average gain over effective best and package build is heavier |
| `rmsnorm_packqkv_splitembed_u32tok_defer` | 5 | 4.5489 | 25.3143 | 24.3343 | 26.1916 | reject for now; close to effective best but average is slightly lower |
| `rmsnorm_packqkv_splitembed_precompute_rope_defer` | 5 | 10.5043 | 25.4710 | 23.5737 | 26.2406 | keep cautiously; average is slightly above the previous effective best, but run-to-run variance is higher |
| `rmsnorm_packqkv_splitembed_splitlm_defer` | 5 | 5.6281 | 25.3188 | 24.2240 | 26.2715 | reject; no average gain over effective best |
| `rmsnorm_packqkv_splitembed_lmhead_dram_pc_defer` | 0 | n/a | n/a | n/a | n/a | reject; runtime fails after first decode because DRAM-sharded matmul program config requires sharded input tensor A |
| `rmsnorm_packqkv_splitembed_lmhead_mcast1d_pc_defer` | 5 | 7.2220 | 25.3187 | 24.2484 | 26.1213 | reject for now; average is slightly below effective best and run-to-run variance is higher |
| `rmsnorm_packqkv_splitembed_fuse_create_qkv_heads_defer` | 0 | n/a | n/a | n/a | n/a | reject; postprocess fails because the expected V reshape pattern is not present in the generated TTNN graph |
| `rmsnorm_packqkv_splitembed_fuse_concat_heads_defer` | 5 | 3.8648 | 23.2152 | 21.6165 | 24.4409 | reject; postprocess succeeds, but runtime is slower than effective best |
| `rmsnorm_packqkv_splitembed_fuse_concat_heads_sdpa_output_defer` | 0 | n/a | n/a | n/a | n/a | reject; runtime fails because sharded SDPA output is not supported for GQA |

## Ten-Run Recheck For LM-Head Candidates

The following two LM-head candidates were rerun with ten runtime repetitions.
The first three runs are treated as warmup/noisy startup samples and excluded
from the average. Both candidates were tested on top of the current kept
configuration:

- `BUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON`
- `BUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON`
- `BUDDY_LLAMA32_DECODE_PACK_QKV=ON`
- `BUDDY_LLAMA32_DECODE_PRECOMPUTE_ROPE=ON`
- `--defer-decode-token-readback`

| Candidate | Extra variable | Runs used | Avg full tokens/s/user | Avg steady tokens/s/user | Steady min | Steady max | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `rmsnorm_packqkv_splitembed_rope_splitlm_defer` | `BUDDY_LLAMA32_DECODE_SPLIT_LM_HEAD=ON` | 7/10 | 7.6100 | 25.3354 | 22.8546 | 26.1614 | keep off; close to the best kept config, but no clear average gain and full decode remains noisy |
| `rmsnorm_packqkv_splitembed_rope_lmhead_mcast1d_pc_defer` | `BUDDY_LLAMA32_DECODE_LM_HEAD_MCAST1D_PC=ON` | 7/10 | 6.3091 | 23.1296 | 19.1373 | 26.0372 | keep off; slower on average and less stable |

The split-LM-head run is interesting because several steady samples still reach
about 26 tokens/s/user, but the average does not beat the best kept configuration
from the five-run sweep. The mcast1d program-config run has larger slow tails,
with several steady samples around 19-23 tokens/s/user, so it is not a default
candidate.

## Default Decision

Enable the four optimizations that improved the Llama 3.2 3B batch32 path:

- `BUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON`
- `BUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON`
- `BUDDY_LLAMA32_DECODE_PACK_QKV=ON`
- `BUDDY_LLAMA32_DECODE_PRECOMPUTE_ROPE=ON`

Keep the other experimental flags off by default. The RoPE precompute result is
kept because its five-run average is the best measured result, but it should be
rechecked when the surrounding runtime variance is lower. `--defer-decode-token-readback`
is still an explicit benchmark/runtime option because it trades away per-token
readback for throughput.

## Decision Rules

- Keep an optimization if the fresh 96-token run improves steady tokens/s/user
  and does not regress generated text sanity.
- Prefer wall-time metrics over device-only counters.
- Treat the first decode step separately. The main optimization target is warmed
  steady decode, but final reports should also include full decode wall.
- If a variable only helps in short smoke but regresses the 96-token run, reject
  it.
- If a variable requires another variable to be meaningful, record it as a
  dependent candidate rather than enabling it alone by default.
