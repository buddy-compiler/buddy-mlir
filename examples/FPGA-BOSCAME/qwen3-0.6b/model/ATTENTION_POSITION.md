# Optional attention using the valid KV length

The optional **native key layout** variant adds `--attention-native-key` to
`--attention --attention-position`. Its QK loads the original
`[head, capacity, head_dim]` key cache, so graph replacement omits the full-cache
`layout_k` call and its 4 MiB workspace per layer/entry. The BK64 dot arithmetic
and all other model operations remain unchanged. Stage A FPGA run
`run-1552379db1d74dd4` passed all 65 length/NaN/sentinel checks; the immutable
record is `validation/board/review/attention-native/validation.json`.
Native one-, four- and 28-layer host graph checks pass all 27 tensor comparisons
with zero error (`validation/review-host-native-{1,4,28}l.json`). Full-model FPGA
results must be read from their own board validation records.

This variant keeps the physical cache at 512 slots and uses the existing
`Position[S]` workspace (`i32`, contiguous, refreshed before every graph call)
to derive `valid = Position[S-1] + 1`. It changes only the QK/PV Triton kernels.
The existing Buddy graph, causal mask, softmax, cache update and NR runtime remain
in the execution chain. It is selected explicitly with `--attention-position`;
the default continues to use the full-capacity kernels.

QK skips inactive output tiles and writes zero to every inactive output element.
PV runs `ceil(valid / 64)` reduction blocks. Both operands of the last PV block
are masked; otherwise `0 * NaN` from an unused cache slot would poison the result.
Within each active block the RVV FMA order is unchanged. The caller must reject
`position < 0` or `position + S > capacity`. The kernel clamp is an additional
memory bound, not a definition of valid model output for an invalid request.

## Evidence and scope

- `validation/board/review/attention-position/validation.json`: FPGA5 run
  `run-68eee3e53cd247a2`, four kernels, 60 ordered length checks, exact numeric
  comparisons, NaN tails, image hash, DDR readback and runtime completion.
- `validation/attention-position-host.json`: scalar host, vectorized host, and
  NR ELF instruction audit. The ELF contains `vfmacc.vf` and RVV loads/stores.
- `validation/review-host-position-1l.json`: actual compiled one-layer graph,
  16-token prefill plus eight successive decode calls; 27 comparisons of full
  last-position vocabulary logits and valid KV snapshots, all max/mean error 0
  against the independent `triton-host` quantized reference.
- `build/attention-position-boundary-{cases,kernels}`: separate short-prompt
  probes for valid lengths 1, 2 and 23, plus 64-byte sentinels around every
  buffer. Consult the separate board record before claiming FPGA validation.

The first Stage A run measures the same dynamic kernel at different lengths.
The mean `cycles(valid512) / cycles(valid16)` is 9.85 for prefill PV, 11.74 for
decode PV, 13.70 for prefill QK and 26.05 for decode QK. These are kernel-only
ratios; they are neither a comparison against the old kernel nor a model speedup.

## Rebuild the isolated kernel suite

Run from the repository root. Use the Python environment in which triton-riscv
was built (currently `boscame`). `MODEL_PYTHON` below is separately the Python
matching the built Buddy bindings; these environments need not have the same
Python version.

```bash
MODEL=examples/FPGA-BOSCAME/qwen3-0.6b/model
TRITON=examples/FPGA-BOSCAME/qwen3-0.6b/triton
export TRITON_PYTHON="${TRITON_PYTHON:-python3}"
source examples/FPGA-BOSCAME/common/triton/triton-env.sh

python3 -B "$MODEL/tools/attention_position_cases.py" \
  --output "$MODEL/build/attention-position-cases"
export QWEN_CASE_ROOTS="$PWD/$MODEL/build/attention-position-cases"
export QWEN_TRITON_BUILD_ROOT="$PWD/$MODEL/build/attention-position-kernels"
cases=(attention_qk_position_16x1x512x128 attention_qk_position_16x16x512x128
       attention_pv_position_16x1x128x512 attention_pv_position_16x16x128x512)
case_args=()
for name in "${cases[@]}"; do case_args+=(--case "$name"); done
"$TRITON_PYTHON" -B "$TRITON/build.py" "${case_args[@]}" --host --suite --jobs 4
"$TRITON_PYTHON" -B "$TRITON/build.py" "${case_args[@]}" --suite --jobs 4
```

Do not rebuild directories belonging to a run being evaluated. For the short
boundary probe, generate into a fresh directory with `--boundary-probe`, select
the four names ending in `_bounds`, and set both output environment variables
to the corresponding fresh directories. All generated launchers now include
sentinels; the archived first run preserves its original launcher without them.

The existing shared runner handles upload/start/capture. A new hardware run must
wait for the platform's active job to complete:

```bash
examples/FPGA-BOSCAME/fpga_run.sh \
  "$MODEL/build/attention-position-kernels/suite-all/suite.bin" \
  --fpga=5 --capture-seconds=600 --completion-marker='[nr] RA returned:'

python3 -B "$MODEL/tools/archive_attention_position_run.py" \
  --run examples/FPGA-BOSCAME/build/fpga-runs/run-REPLACE_WITH_RETURNED_ID \
  --build "$MODEL/build/attention-position-kernels" \
  --cases "$MODEL/build/attention-position-cases" \
  --output "$MODEL/validation/board/review/attention-position-NEW_RUN"
```

The archive validator rejects missing/reordered/duplicate cases or lengths,
nonzero errors, invalid cycle counts, wrong image hashes and missing DDR/runtime
completion. It copies logs, manifests, generated sources, IR and assembly.

## Build a model against the verified kernels

The overlay below creates local symlinks for read-only import/archive/link use.
It neither rebuilds nor overwrites the existing shared kernel tree. Do not use
this overlay as a kernel compiler output directory.

```bash
python3 -B "$MODEL/tools/kernel_build_overlay.py" \
  --source "$TRITON/build" \
  --source "$MODEL/build/attention-position-kernels" \
  --output "$MODEL/build/review-position-kernels"

export MODEL_PYTHON="${MODEL_PYTHON:-python3}"
export PYTHONPATH="$PWD/build-python/python_packages${PYTHONPATH:+:$PYTHONPATH}"
LAYERS=1  # also 4 or 28
DEST="$MODEL/build/review-position-${LAYERS}l"
inputs=(--assets "$MODEL/assets/official" --checkpoint "$MODEL/assets/checkpoint"
        --triton-build "$MODEL/build/review-position-kernels" --layers "$LAYERS"
        --max-cache-len 512 --w8a8 --attention --attention-position)
"$MODEL_PYTHON" -B "$MODEL/tools/triton_call_replace.py" "${inputs[@]}" \
  --prefill-len 16 --output "$DEST/replacement"
for kind in prefill decode; do
  "$MODEL_PYTHON" -B "$MODEL/tools/lower_model_nr.py" "${inputs[@]}" \
    --prefill-len 16 --kind "$kind" --output "$DEST/nr-$kind"
done
TRITON_BUILD="$PWD/$MODEL/build/review-position-kernels" \
ADAPTERS="$PWD/$DEST/replacement/qwen_triton_adapters.c" \
OUT="$PWD/$DEST/model-lib" bash "$MODEL/tools/build_model_lib.sh"
TRITON_BUILD="$PWD/$MODEL/build/review-position-kernels" \
ADAPTERS="$PWD/$DEST/replacement/qwen_triton_adapters.c" \
OUT="$PWD/$DEST/host-bridge" bash "$MODEL/tools/build_host_bridge.sh"
```

Use the unchanged official checkpoint layout matching `LAYERS` for host scoring:
`build/import/probe-1layer/weight-layout.json`,
`build/import/probe-4layer/weight-layout.json`, or `validation/weight-layout.json`
for all 28 layers. The independent reference is the matching
`build/review-${LAYERS}l/quant-host-order` (host) or `quant-nr-order` (FPGA).

```bash
LAYOUT="$MODEL/build/import/probe-1layer/weight-layout.json"
"$MODEL_PYTHON" -B "$MODEL/tools/run_graph_host.py" "${inputs[@]}" \
  --replace --layout "$LAYOUT" --output "$DEST/host-run" \
  --external-lib "$DEST/host-bridge/libqwen_triton_host.so" --decode-steps 8 \
  --prompt-ids 151644,872,198,3838,374,9625,30,151645,198,151644,77091,198,151667,271,151668,271 \
  --quant-reference-dir "$MODEL/build/review-${LAYERS}l/quant-host-order" \
  --max-abs-error 0 --mean-abs-error 0
"$MODEL_PYTHON" -B "$MODEL/tools/score_graph_host.py" \
  --run "$DEST/host-run" \
  --quant-reference "$MODEL/build/review-${LAYERS}l/quant-host-order" \
  --output "$MODEL/validation/review-host-position-${LAYERS}l.json" \
  --max-abs-error 0 --mean-abs-error 0
```

For the final image, use the public image builder and shared runtime described
in `README.md`, pointing its graph/library inputs at this `DEST`. Keep the NR
quantized reference, board UART record, full-logit/KV comparison, and any cycle
profiling separate from the host result. A compiled 28-layer image alone is not
evidence that the 28-layer model completed on FPGA.

## Native key layout build

The commands above also apply to the native variant with these explicit changes:

1. Generate separate cases with
   `attention_position_cases.py --native-key --output "$MODEL/build/attention-native-cases"`.
2. Point `QWEN_CASE_ROOTS` at that directory and `QWEN_TRITON_BUILD_ROOT` at
   `"$PWD/$MODEL/build/attention-native-kernels"`. Select the two cases
   `attention_qk_position_native_16x1x512x128` and
   `attention_qk_position_native_16x16x512x128` for both host and NR suite builds.
   The generated length list covers 1/2, every 16+8 decode length, powers-of-two
   block edges through 512, and shrinking lengths. M=16 only uses lengths >=16.
3. Make a fresh `review-native-kernels` overlay with three sources:
   `triton/build`, `attention-position-kernels`, and `attention-native-kernels`.
   The existing position PV kernel is reused.
4. Use `DEST="$MODEL/build/review-native-${LAYERS}l"`, the native overlay as
   `--triton-build`, and append `--attention-native-key` to the model input
   arguments. Keep the original checkpoint, cache capacity and references.
5. Save host scores as `validation/review-host-native-${LAYERS}l.json`.

`lower_model_nr.py` now checks the actual post-bufferization key descriptors
before object generation. It rejects missing callsites or any storage other
than contiguous `[1,16,512,128]` with strides `[1048576,65536,128,1]`. This
guards against selecting a physical-layout kernel based only on tensor shape.
The evidence is saved as `native-key-abi.json` next to each object. For existing
objects generated immediately before adding the automatic audit, reproduce it
without rebuilding:

```bash
python3 -B "$MODEL/tools/audit_attention_native_ir.py" \
  --llvm "$DEST/nr-prefill/forward_prefill.ll" --layers "$LAYERS" \
  --output "$DEST/nr-prefill/native-key-abi.json"
```

Repeat with `decode` for the other entry. The audit parses the compiler's
unoptimized LLVM insert/extract chains and fails on unknown expression forms;
it is a narrowly scoped evidence checker, not a general LLVM parser.
