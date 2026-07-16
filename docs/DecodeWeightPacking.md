# Decode Weight Packing

This document describes decode weight panel-packing. It repacks the matmul
weights read by the decode phase into the layout its GEMV kernel walks, so the
inner loop reads memory contiguously instead of striding across pages.

The current implementation and validation use DeepSeek R1 as the worked example.
Other models can use the same flow, but each needs its weight shapes checked
against the packing constraints below before being enabled.

## Environment

Run from the repository root and activate the Python environment used for model
import:

```bash
cd buddy-mlir
conda activate buddy
```

## Background

Decode generates one token at a time, so every matmul is a GEMV (`m == 1`). Each
weight byte is read once and never reused, which means the phase is limited by
how the weights are laid out, not by how many of them there are.

`matmul-vectorization-decode` tiles `N` into `vecSize`-wide panels and reduces
over `K` inside each panel. For a row-major `B[K, N]`, consecutive `k` in that
inner loop are `N` elements apart:

| weight | shape (K x N) | inner-loop stride |
|--------|---------------|-------------------|
| q_proj, o_proj | 1536 x 1536 | 6 KB |
| gate_proj, up_proj | 1536 x 8960 | 35 KB |
| down_proj | 8960 x 1536 | 6 KB |
| lm_head | 1536 x 151936 | 594 KB |

Every iteration lands on a different page and the hardware prefetcher stops
working ahead. Panel-packing reorders each weight to

```
Bpacked[N/V][K][V]                        V = vecSize
offset(nt, k, v) = nt*K*V + k*V + v
```

so consecutive `k` are `V` elements apart: one contiguous stream. The same bytes
are moved, in a different order. This is the standard BLIS/oneDNN/ggml GEMV
prepack, and it can be done once at import time because decode weights are
static.

Packing permutes the bytes inside each weight. It does not change the weight's
shape, its offset, or the size of the parameter buffer, so the decode MLIR is
unchanged: the same function signature and the same subviews. The packing
produces a second parameter file, `arg0-decode.data`, alongside the plain
`arg0.data`. Prefill is handed the plain file and decode the packed one; that is
the whole of the difference.

Prefill keeps the plain weights because the two phases want opposite layouts.
Prefill is compute-bound at `m = 1024` -- its `matmul-vectorization-blis` kernel
reuses each resident tile of `B` across many rows of `A` -- and it reads
row-major `B`. Handing it packed bytes does not fail loudly: it reads them as
row-major and produces fluent but wrong output.

## Default Build Integration

Packing is off by default. It is enabled per spec, by setting
`decode_pack_vector_size` to the panel width, which must match the decode
`vector-size` in use:

```json
{
  "variant": "f32",
  "decode_pack_vector_size": 32
}
```

`models/deepseek_r1/specs/f32_packed_decode.json` is the f32 DeepSeek R1 spec
with packing enabled. Build it the usual way:

```bash
python3 tools/buddy-codegen/build_model.py \
  --spec models/deepseek_r1/specs/f32_packed_decode.json \
  --build-dir build
```

When the spec enables packing, the importer runs the
`pack_decode_matmul_weights` graph transform over the decode graph and writes
decode's parameters to `arg0-decode.data`; `compile_pipeline.py` compiles decode
with `-matmul-vectorization-decode-packed` in place of the plain kernel; and
`gen_session.py` loads both parameter files and gives each phase the one it
expects.

Two restrictions are enforced at config time:

- f32, f16 and bf16 only. Quantized variants lay their weights out through the
  dequant kernels and are not in scope.
- Not supported together with `tiered_kv_cache`. Only `gen_impl` was taught to
  hand decode a different weight buffer, so `gen_impl_tiered` would give decode
  the plain weights while compiling it with the packed kernel.

`examples/BuddyDeepSeekR1` has its own packed target, built the usual way:

```bash
ninja buddy-deepseek-r1-packed-run
```

It imports with `--pack-decode-weights --pack-vector-size 32` and writes its
artifacts to `packed_decode/` so they do not clobber the plain f32 outputs.

## Adapting To Other Models

The packed kernel selects per matmul, by shape. It cannot look at a memref and
tell packed bytes from plain ones, so a weight that was left unpacked but
compiled with the packed kernel is read through panel-layout addressing: a wrong
model, with no crash and no failing test.

`pack_decode_matmul_weights` therefore packs *every* matmul weight in the decode
graph and raises if it cannot pack one, rather than packing what it can. Because
there are then no exceptions, the pass can be given an empty `packed-shapes`,
which means "all", and there is no opt-in list to keep in step with the packer.

When enabling this for another model, check:

- **every** matmul weight can be packed. `N` must be divisible by the panel
  width for all of them, and each must be a parameter rather than a computed
  value. The transform refuses the graph otherwise, which is the intended
  behaviour; do not work around it by packing a subset.
- both matmul op types are covered. `MatmulOp`'s weight is `args[1]`,
  `AddMMOp`'s is `args[2]`. Projections with a bias import as `AddMMOp` -- in
  Qwen2 that is q/k/v -- and a version of the transform that handled only
  `MatmulOp` silently packed 113 of DeepSeek R1's 197 weights.
- weights that are not matmul operands are left alone. The embedding table in
  particular must not be packed: it is `[vocab, hidden]`, the transpose of
  `lm_head` and with an identical element count, and it is read by a gather. The
  transform selects by matmul operand rather than by size, so the embedding is
  excluded by construction, but any size-based heuristic added later would
  confuse the two.
- decode's parameter list is packed, not prefill's. The two graphs are traced
  separately and own separate parameter lists, though the tensors inside them
  are the same objects. The transform rebinds list entries
  (`_params_ref[i] = packed`) rather than mutating tensors in place. An in-place
  repack is visible through prefill's list as well and corrupts it.

Weight reuse (`BUDDY_MODEL_REUSE_WEIGHTS`) accounts for packing: the packed file
is a weight entry of its own, so a build in which it was never written is not
reused, and the panel width is recorded in `.buddy_weights_manifest.json`, so
changing it invalidates the reuse. This matters because the packed file is the
same size as the plain one, and a size check alone cannot tell one panel width
from another.

## Observed DeepSeek R1 Results

On the f32 DeepSeek R1 Distill Qwen 1.5B experiment, with 48 threads bound to
both NUMA nodes (`numactl --cpunodebind=0,1 --interleave=0,1 taskset -c 0-47`):

- Per-operator decode time, layer 0:

  | operator | plain | packed |
  |----------|-------|--------|
  | q_proj | 0.087 ms | 0.048 ms |
  | k_proj | 0.056 ms | 0.028 ms |
  | v_proj | 0.055 ms | 0.028 ms |
  | o_proj | 0.078 ms | 0.046 ms |
  | gate_proj | 0.519 ms | 0.315 ms |
  | up_proj | 0.446 ms | 0.241 ms |
  | down_proj | 0.411 ms | 0.234 ms |
  | lm_head (once per token) | 6.03 ms | about 1.5 ms |

- Decode throughput roughly doubles.
- Measured against a 267.9 GB/s STREAM read ceiling on the test machine, the
  plain kernel reached 40-58% of peak on these weights.
- Generated tokens are identical to the unpacked build.
- The packed parameter file is the same size as the plain one, and a packed build
  holds both: about 13.5 GB of weights in memory for this model instead of 6.8.

These numbers are machine-dependent. Note also that end-to-end decode tokens/s
is a poor measure here: on the test machine it varied by about 65% run to run
for the same binary, so the per-operator figures above were taken from
instrumented per-region timings averaged over a full generation.
