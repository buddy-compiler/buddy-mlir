# Graph/numerical review checkpoint

Paused at the user's request. Do not start further builds or hardware jobs until
the user resumes the task. This note covers the graph/numerical subagent only;
the root agent owns final board image construction and model FPGA results.

## Verified work

- Full-capacity reviewed model artifacts remain in `build/review-{1,4,28}l`.
  Host execution of the real compiled Buddy graph matches the independent
  `triton-host` W8A8 reference exactly for last-position full-vocabulary logits,
  eight decode steps, and all valid K/V snapshots (27 tensor comparisons).
  Reports: `review-host-{1,4,28}l.json`.
- Independent `nr-fpga` arithmetic references are in the corresponding
  `quant-nr-order` directories. These emulate ordered BK64 FMA attention and the
  common NR scalar math. The use of shared scalar math is an explicit oracle
  limitation, not independent validation of `nr_math.c` itself.
- Opt-in `--attention-position` derives the effective length from the existing
  `i32[S]` position workspace while retaining physical capacity/stride 512.
  QK skips unused column tiles; PV masks both operands and skips unused reduction
  blocks. Neither the default graph nor old artifacts silently change.
- Stage A FPGA run `run-68eee3e53cd247a2`: four kernels, 60 length checks, exact
  oracle agreement including NaN-poisoned unused cache. Immutable evidence is
  under `board/review/attention-position`, including UART, image hash, DDR
  readback, source snapshots, IR, manifests, and assembly.
- Independent opt-in model products are in `build/review-position-{1,4,28}l`:
  `replacement/`, `nr-prefill/`, `nr-decode/`, `model-lib/`, `host-bridge/`.
  All three layers' products were completed before this pause. Each static
  library contains 47 kernels; remaining runtime references are `expf`/`memcpy`.
- Opt-in one- and four-layer graph host runs passed with zero max/mean error for
  all 27 comparisons. Reports: `review-host-position-{1,4}l.json`.
- Eight graph contract unit tests passed. They include returning-output
  nonaliasing, structural matcher rejection, dynamic-kernel ABI/tile/grid
  rejection, finite/shape errors, trajectory divergence, and a counterexample
  distinguishing fused from unfused arithmetic.

## Final stop state

The opt-in 28-layer host check was still computing without a final result after
approximately seven minutes. To honor the requested stop, the verified owned
process PID 3890919 (tool session 37879) received SIGTERM and exited 143. There
are no active graph-subagent builds or checks. Its expected output locations are:

```
build/review-position-28l-host-run.log
build/review-position-28l/host-run/host-run.json
build/review-position-28l/host-run/arrays.npz
```

No `host-run.json` or `arrays.npz` was written. The explicit
`review-host-position-28l-paused.json` records `STOPPED_UNVERIFIED`. The completed
28-layer objects/archive remain usable. This does not invalidate the completed
full-capacity 28-layer host result, which is a different kernel configuration.

The stopped run was started with eight decode steps, the established 16-token prompt, full
vocabulary, zero error thresholds, and
`build/review-28l/quant-host-order` as its quantized reference. The JSON's
`numerical_validation` and saved full arrays determine PASS; process existence
or nonempty logs do not. No additional build or check was started after the stop.

## Ready but not hardware-validated

`build/attention-position-boundary-kernels/suite-all/suite.bin` contains a
separate small Stage A probe:

- M=1 lengths `[1,2,23,2,1]` for QK and PV.
- M=16 lengths `[23,16,23]` for QK and PV.
- 64-byte sentinels around every buffer, unchanged position check, NaN tails.

Scalar-host and vector-host checks and ELF audit passed. It has not been run
on FPGA by this subagent. Its cases live in
`build/attention-position-boundary-cases`; old Stage A artifacts are unchanged.

## Unbuilt direct-key-cache draft

The root agent's timing of the in-progress one-layer FPGA run found full-cache
`layout_k` transpose expensive. Immediately before the pause the following
source-only draft was written:

- `../triton/kernels_position_native.py`: QK reads original cache
  `[head, capacity, head_dim]` and transposes only the active tile during loads.
- `../triton/cases.py`: optional `native_key_layout` selects that module.
- `tools/attention_position_cases.py --native-key`: QK-only launcher generation
  matching the original physical cache layout.
- `../triton/build.py`: source provenance records the actual selected module.

**No direct-key case has been generated, compiled, numerically checked, or run
on FPGA. No model calls it.** Keep it classified as untested source. To resume,
first generate fresh isolated cases, then scalar/vector host validation, ELF
audit, Stage A FPGA, only then a separate graph variant removing `layout_k` and
the 4 MiB K-transpose workspace. Preserve the same BK64 accumulation order and
mask invalid cache entries. Do not overwrite the validated position variant.

## Supporting tools

- `tools/kernel_build_overlay.py`: fresh local, read-only-use manifest/object
  index joining shared kernels and optional position kernels. Do not use it as
  a kernel compiler output directory; its entries are local symlinks.
- `tools/archive_attention_position_run.py`: strict case/length/order/error/
  completion/image/readback validator and evidence snapshot.
- `ATTENTION_POSITION.md`: complete isolated kernel/graph/library/host commands
  and the distinction between kernel-only timing and model timing.

No git index operations, `examples/BuddyQwen3` edits or FPGA actions were made
by this subagent.
