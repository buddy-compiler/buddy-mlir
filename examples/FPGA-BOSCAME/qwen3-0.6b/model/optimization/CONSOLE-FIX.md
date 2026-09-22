# Fixed-input NR console comparison

The reference is ModelZoo commit `cb9541c0e874136b327e6af19898320465a925ea`,
`examples/buddy-qwen35-fpga/platform/nr/qwen35_nr_console.c`. Its finite-input
mode appends without waiting for NH acknowledgement. Only dynamic input uses a
blocking ring. Ordinary fences and NH cache operations already agree with our
platform; this comparison changes the console protocol, not AME instructions.

## Implementation

The shared implementation is `common/nr/nr_console.inc`, used by
`nr_runtime.c`. Build options are `NR_CONSOLE_APPEND_ONLY=0|1` and
`NR_CONSOLE_CAPACITY`. Model builder options are `--console-mode` and
`--console-capacity`. Capacity is explicit in each image plan. Overflow retains
the first bytes and forces final NR FAIL; it cannot silently pass a truncated
numerical trace. Interactive input continues to require a ring.

The silent observer and its decoder are described in [HANG-WATCH.md](HANG-WATCH.md).
The archiver verifies the raw worker log before reconstructing RA lines, binds
the observer to the linked image, and preserves both logs. Diagnostic cycles
are not uninstrumented throughput results.

## Board Evidence

Configuration: FPGA5, official Qwen3-0.6B, 28 layers, capacity 128, fixed text
`What is France?`, on-board tokenizer, 16-token prefill and eight successive
decode calls. The graph, Triton archive, weights, tokenizer, independent oracle,
and graph-final AME resync are reused without changes. No hardware settings or
external symlink targets are modified.

| Variant | Console | Result |
| --- | --- | --- |
| Silent watch | 512 KiB append-only | Full numerical/text PASS, run `run-0496801ccb0344c3` |
| Silent control | 512 KiB ring | Built and ELF-audited; not yet run |
| Per-kernel progress/probe, AME resync | 4 MiB append-only | Stalled during prefill, then interrupted; run `run-5eb565e2b8264980` |
| Per-kernel progress/probe, ordinary fence | 4 MiB append-only | Same prefill boundary, then interrupted; run `run-912dfb70b06d4c03` |
| Exact prefill operand probe, ordinary fence | 4 MiB append-only | Descriptors match planned ranges; no return observed; run `run-e573730a84f74df1` |
| Exact prefill tile probe and NH watch | 4 MiB append-only | Earlier layer-0 boundary; NH alive, selected layer-19 call not reached; run `run-7e4d51f849eb4bae` |
| Deferred NH drain, ordinary fence profiling | 4 MiB append-only | Prefill and decode position 16 PASS; position 17 numerical FAIL; run `run-72a2e4724ff14fe4` |
| Deferred NH drain, AME resync profiling | 4 MiB append-only | No completion in 1500 seconds; run `run-ca6f2cf51eef4b88` |
| Deferred NH drain, selected layer 14/15 intermediate checks | 4 MiB append-only | Full numerical/text and 603 intermediate comparisons PASS; run `run-c6f5f08277804edb` |
| Same selected-intermediate image, repeat | 4 MiB append-only | Full numerical/text and 603 intermediate comparisons PASS; run `run-78b078e60c764e6b` |
| Same selected return boundaries, synchronization only | 4 MiB append-only | Prefill PASS, first decode validation reaches data-section illegal-instruction trap; run `run-aa2729001d4049d1` |

The first run's reproducible archive is
[`validation/board/console-fix/append-watch`](../validation/board/console-fix/append-watch/verification.json).
All 27 full-tensor checks pass; maximum absolute error is
`9.5367431640625e-7`, maximum reported mean absolute error is
`9.415223808928452e-12`. Predictions are
`49000, 374, 264, 3146, 7407, 304, 4787, 5159, 11`.
All three DDR segment readbacks match. The observed selected matmul returned
in every decode call, and no console wait or dropped byte was reported.

The local pair under `build/console-fix-20260920/{ring-watch,append-watch}` has
identical graph, adapter, model-entry, observer, and AME-sync objects. The
selected `triton_matmul_1x1024x2048` is 640 bytes with SHA256
`d0893340aad26f4fbaeca098c7ef984a7e43d848cf786d1d926bb84bfc46a024` in both.
Runtime and linked addresses differ. `comparison.json` in that build directory
records ELF hashes and per-object comparisons.

The second run stopped producing UART at 152326 bytes, with final boundary
`[kernel] begin _mlir_ciface_kernel_matmul_16x3072x1024 call=0000000000000026`.
This is occurrence 38 (zero-based), corresponding to layer index 19 in the
current two-MLP-projections-per-layer graph. Worker samples show unchanged
bytes from elapsed 185 through 335 seconds. A stop request then ended this run;
the result is INTERRUPTED, not a numerical PASS. All DDR readbacks match.
Evidence is in
[`append-profile-stalled`](../validation/board/console-fix/append-profile-stalled/summary.json).

This failure happened with no RA consumer wait and far below append capacity.
Therefore removing ring backpressure is insufficient to fix all stalls. The
last boundary is still not a hardware PC. The next comparison image changes only
the profiler's extra per-kernel resync to an ordinary fence; graph-final resync,
kernel objects, console mode and capacity stay the same. It is under
`build/console-fix-20260920/append-profile-fence/`.

After an SSH outage on 2026-09-20, FPGA5 availability was checked and the third
run started on 2026-09-21. It again stopped at exactly that prefill boundary,
with 152326 UART bytes unchanged at worker elapsed 201 through 411 seconds.
It was stopped through its own worker's stop file; all three DDR segment
readbacks match. The archive is
[`append-profile-fence-stalled`](../validation/board/console-fix/append-profile-fence-stalled/summary.json).
Removing per-kernel AME resync is therefore insufficient as well. Graph-final
resync has not yet been reached in either stalled run.

IR and adapter inspection maps occurrence 38 to layer index 19's gate
projection, `qwen_graph_w8a8_mm_137_matmul_16x3072x1024`. Its expected byte
ranges are A `[0xe0c5edc0,0xe0c62dc0)` (i8), B
`[0xc9398700,0xc9698700)` (i8), and C `[0xe0c62e00,0xe0c92e00)` (i32).
These are disjoint, 64-byte aligned and inside the linked workspace/weights.
The generated entry clears all 196608 C bytes before prefill. These static
checks do not substitute for observing the descriptors at runtime. The next
image moves `--profile-probe` to this exact symbol and occurrence, retaining
fence-only profiling and the 4 MiB append console.

That fourth run printed all three descriptors, matching these exact addresses,
zero offsets and expected contiguous strides. It then stopped at 153172 bytes,
unchanged from worker elapsed 189 through 459 seconds, and was interrupted.
No kernel return was observed. Its graph, model-entry, adapter and runtime
objects match the fence-profile control. All DDR readbacks match. See
[`prefill-probe-stalled`](../validation/board/console-fix/prefill-probe-stalled/summary.json).
This rules against an obvious argument-address/shape/stride mismatch at this
call, but cannot exclude internal machine access errors or stale visibility.

Adding `--profile-tile-probe --profile-watch` to that probe uses linker wrappers
around the original raw Triton call (all 12 ABI parameters forwarded), and NH
samples ENTER/RETURN records independently of RA UART. In that fifth run,
output stopped earlier at `matmul_16x1024x1024 call=1` in layer zero. All 80 NH
frames remain parseable; NH cycles/samples advance, while RA count and consumed
remain 2701, with no pending/wait/dropped bytes. The selected layer-19 call was
not reached. See
[`prefill-tile-watch-stalled`](../validation/board/console-fix/prefill-tile-watch-stalled/summary.json).
This confirms observation/instrumentation changes can move the boundary. It
does not prove cache freshness or locate a hardware PC. A subsequent comparison
will defer NH console consumption until RA completion, retaining the same
finite RA log but removing live console cache invalidation during computation.

## Limits

This removes the producer's dependency on receiving NH's consumer cursor. It
does not prove that this dependency caused the historical hangs. The silent
run emits less than 64 KiB of RA output, so it need not exercise a ring-full
wait even with the previous capacity. The per-kernel run checks the earlier
repeatable observation boundary with a finite log large enough for all stages;
its capacity/layout and cache traffic also differ from older images.

No startup AME prime, new cache instruction, or graph synchronization change
is mixed into these experiments. The earlier unprofiled RA trap and the
profiler stalls may have different causes. A successful run does not settle
that question.

The builder additionally exposes opt-in `--ame-startup=prime` and
`--graph-entry-fence`, both disabled by default. The former invokes existing
`ame_fence()` once before the first graph in an RA boot, after operand
preparation; it is not a correctness self-test. The latter places ordinary
`fence rw,rw` immediately before every graph call. Host execution tests verify
ordering and lifetime only; these options have not yet been validated on FPGA.

## Deferred Consumption Comparison

`--console-drain=after-completion` keeps RA's append-only producer unchanged,
but NH skips console and input service while waiting for RA_SIGNAL. On RA
completion it uses the same drain and overflow checks. This mode rejects
interactive input and NH watch/UART probes. It does not change cache or FPGA
configuration, and still polls the existing completion mailbox. If RA stops,
its buffered log will not appear over UART. Choose sufficient capture time for
computation followed by UART transmission of the entire log.

The prepared comparison is
`build/console-fix-20260921/deferred-profile/`. Against `prefill-probe`, the
following compiled objects match exactly:

| Object | SHA256 |
| --- | --- |
| model_main.o | `56b80d12daaac1a338a2e82a9ea6ea8f4435582fae4227c6cf1f495b01716ce4` |
| kernel-profile.o | `2321987e8e299f04793d3e6c43fe42c8855ce5c04e480d28f49a7dd557d1a9a8` |
| forward_prefill.nr.o | `3ac648d83e7fd304f232f051152c3991e30147c26b2e69484a908e5d9b01fa01` |
| forward_decode.nr.o | `90440ef55dd4e46e0886a06b35f67f68e1eda31f5c02c7b8083e2a4cc1f90e25` |
| adapters.o | `6a6d653c59253c2096b19ab1d2ed69771ff2b49005c54039dceb74e8b203ae99` |
| ame_sync.o | `81b7b1e9d5cd23b0ce22303b9a4d26814f58b23946a52e70248ce80a2a506913` |

Kernel archive, weights, tokenizer and oracle are reused. Runtime and linked
addresses change, so any observed improvement still needs repeat validation
and does not by itself prove a cache-coherence root cause.

This comparison completed as `run-72a2e4724ff14fe4`, with all three DDR
readbacks matching. The former layer-19 prefill matmul returned and synced;
prefill and decode position 16 pass the full logits/K/V checks. At position 17
all three checks fail, and the firmware returns FAIL. The next token still
matches (264), which is insufficient for numerical acceptance. See
[`deferred-profile-failed`](../validation/board/console-fix/deferred-profile-failed/summary.json).
Of 27 planned full-tensor checks, six passed, three failed and 18 were not
reached. Position 17 maximum/mean absolute errors are logits
`1.487941 / 0.248301`, K cache `2.339188 / 0.00221728`, and V cache
`2.890989 / 0.00890534`. The captured 662191 bytes fit the 4 MiB console.
Diagnostic compute cycles were 3695728218 for prefill, 1032409940 for decode
position 16 and 1035394394 for position 17; total launch cycles were
6076562661. These include profiler/buffered logging overhead and are not
accepted throughput measurements.

Against the passing `append-watch` log, position 17's sampled K/V values match
for layers 0 through 14. The first sampled difference is layer index 15: K
`403D7686` versus `40211ED0`, V `BF76BA37` versus `BF86721D`. These are only
head 0, dimension 0 samples, not complete intermediate checks. The useful
inspection interval includes layer 14 attention output/MLP and layer 15
input norm, quantization and QKV. This run did not hang; neither the console
change nor fence-only profiling is a verified fix.

The next controlled comparison, `build/console-fix-20260921/deferred-resync`,
retains deferred drain and restores the original profiler's `ame-resync`.
Startup prime and graph-entry fence remain disabled.
This comparison timed out as `run-ca6f2cf51eef4b88`: all DDR readbacks match,
but only the 71-byte NH banner was observed in 1500 seconds. See
[`deferred-resync-timeout`](../validation/board/console-fix/deferred-resync-timeout/summary.json).
Deferred output provides no intermediate RA progress in this case. This does
not identify a stopped PC or establish whether resync caused the timeout.

## Selected Numerical Boundaries

The intermediate probe now supports `--intermediate-layers=14,15` with a full
28-layer graph and independent intermediate reference. Layer ownership comes
from checkpoint-matched weights. RMSNorm boundaries, all seven linear outputs,
and int8/scale values at each linear consumer are checked. Shared quantization
is followed through its actual producer, including K/V/up consumers. Shared
activation and adjacent output/input reference aliases must be exactly equal
before packing. These aliases do not introduce duplicate runtime comparisons.

This selection has 67 boundaries per graph (134 manifest entries, 603 checks
over prefill plus eight decodes), occupying 7767360 reference bytes. The host
reference contains 10395 arrays; all 27 pre-existing logits/KV arrays match
the accepted independent reference exactly. `schema_version=2` records 28
model layers separately from the selected layer indices. The former one-layer
schema remains supported without changing its coverage contract.

The selected diagnostic image is
`build/console-fix-20260921/selected-intermediates/image/`. It retains deferred
console and graph-final resync, omits per-kernel profiling/progress, and performs
the selected full-tensor comparisons through linker wrappers calling the same
compiled graph adapters. Each selected wrapper adds AME resync after the real
call. This changes synchronization, traffic and layout, and is not a controlled
performance comparison or proof of a fix. Attention/RoPE/SiLU are observed only
through adjacent selected boundaries; integer accumulators are not checked.

This image completed as `run-c6f5f08277804edb`. Its
[`selected-intermediates` archive](../validation/board/console-fix/selected-intermediates/verification.json)
passes all 27 full last-position logits/effective-KV checks and all 603 selected
intermediate comparisons. Every selected intermediate max/mean error is zero.
The only nonzero aggregate error is position 16 logits: maximum
`9.5367431640625e-7`, mean `9.415223808928452e-12`. All nine predicted tokens
match the independent quantized reference. All three DDR readbacks match.
Prepared image SHA256 is
`97dc9b1bd42153c82ae460da559b965f016d4b097c16d95c8ce6e459c74f027e`.

The first launch took 12708090146 cycles (861.82 seconds at the documented
14.7456 MHz). Instrumented prefill compute took 3796415623 cycles (257.46
seconds), and mean decode compute took 1009515062.5 cycles (68.46 seconds).
These include selected wrapper comparisons and synchronization, so are not
uninstrumented throughput measurements. Removing detailed profiling, adding
selected synchronization, extra tensor reads and address-layout changes are
all possible contributors to this success; the root cause remains unresolved.

The probe generator now also verifies the descriptor order inside selected
adapter calls. Regenerating in a fresh directory after this check was added
reproduces the exact C and reference-blob hashes of the tested image. Coverage
reports with `coverage_includes_aliases=true` count the 63 independently verified
shared references as covered. Existing image manifests retain their original
primary-key-only accounting and remain numerically valid.

Repeating the exact same prepared image as `run-78b078e60c764e6b` also passes
the full archive checks, with the same token trajectory and errors. See
[`selected-intermediates-repeat`](../validation/board/console-fix/selected-intermediates-repeat/verification.json).
Launch cycles are 12707995634. SSH disconnected once; the existing worker was
resumed at UART byte 71, without a second upload or FPGA restart. Two successes
establish this diagnostic image's repeat result, not general stability of the
uninstrumented model.

Full-KV failures additionally print the first unequal element's layer, head,
cache position, dimension and values after the aggregate comparison fails. The
extra scan does not run on successful comparisons and is not a hardware PC
probe. It can distinguish a difference in an old cache slot from one in the
newly generated slot.

To reproduce the selected probe, first generate the independent intermediate
reference (host validation data, never supplied as model operands):

```bash
MODEL="$PWD/examples/FPGA-BOSCAME/qwen3-0.6b/model"
REFERENCE="$MODEL/build/console-fix-20260921/intermediate-reference-28l"
python -B "$MODEL/tools/quant_model_reference.py" \
  --assets "$MODEL/assets/official" --checkpoint "$MODEL/assets/checkpoint" \
  --layout "$MODEL/build/ame-v05/model-28l-cap128/import/weight-layout.json" \
  --output "$REFERENCE" --layers 28 --max-cache-len 128 --decode-steps 8 \
  --arithmetic-profile nr-fpga --capture-intermediates \
  --prompt-ids 151644,872,198,3838,374,9625,30,151645,198,151644,77091,198,151667,271,151668,271
```

Use the common image-build arguments from [HANG-WATCH.md](HANG-WATCH.md),
omitting `--hang-watch` and all profiler options, and adding:

```bash
--intermediate-layers=14,15 --intermediate-arrays "$REFERENCE/arrays.npz" \
--intermediate-layout "$MODEL/build/ame-v05/model-28l-cap128/import/weight-layout.json" \
--intermediate-graph-dir "$MODEL/build/quant-opt/shared/replacement" \
--graph-sync=ame-resync --console-mode=append-only --console-capacity=4194304 \
--console-drain=after-completion
```

Prepare/run with the common commands below. The complete trace must pass
`archive_model_run.py`, including full logits/KV, tokenizer/text, and schema-v2
intermediate checks. A partial or failed trace remains diagnostic evidence only.
The image plan embeds the exact intermediate manifest; generated C and oracle
blob are bound to build-time hashes. The generator additionally checks actual
typed SSA sharing and final LLVM parameter provenance, not only report labels.

## Selected Synchronization Without Comparisons

`--boundary-sync-manifest` derives the exact same selected adapter boundaries
from a prior schema-v2 intermediate manifest, then rechecks its checkpoint
identity, typed SSA, final LLVM provenance and adapter argument order against
the current graph. Each generated wrapper forwards all descriptors once to
the original adapter and calls the public `ame_fence()` after it returns.
There are no intermediate comparisons, reference tensors, per-kernel UART or
cycle reads in these wrappers. Normal full logits/KV validation remains.

Use the common build command with a fresh `OUT`, no profiler/intermediate/watch
options, and these additions:

```bash
--boundary-sync-manifest "$MODEL/build/console-fix-20260921/selected-intermediates/image/intermediate-probe.json" \
--intermediate-layout "$MODEL/build/ame-v05/model-28l-cap128/import/weight-layout.json" \
--intermediate-graph-dir "$MODEL/build/quant-opt/shared/replacement" \
--graph-sync=ame-resync --console-mode=append-only --console-capacity=4194304 \
--console-drain=after-completion
```

`boundary-sync.c/json` and the copied selection manifest record all 90 linker
wrappers (45 per graph). The exact plan and build-time source hashes are checked
by `archive_model_run.py`, as are the linked wrapper symbols. This mode is
mutually exclusive with intermediate probes, per-kernel profiling and NH watch.
Its layer selection is diagnostic, not a general synchronization policy.

The first build is `build/console-fix-20260921/boundary-sync/`. Both graph
objects, `adapters.o`, `nr_runtime.o` and `ame_sync.o` match the selected-probe
image byte for byte. Its ELF audit passes with no undefined symbols. Removing
the 7767360-byte intermediate oracle and comparison code changes final
addresses and memory traffic; even a PASS cannot isolate synchronization from
all layout/timing effects.

This version failed as `run-aa2729001d4049d1` after full prefill logits/K/V
passed exactly. Decode position 16 reached the partial validation line
`[compare] logits position=00000010 count=` before an RA trap. The
[`boundary-sync-trap` archive](../validation/board/console-fix/boundary-sync-trap/summary.json)
records `mcause=2`, `mepc=0x801e1bea`, `mtval=0x367fbfcf`. The exact ELF places
that PC inside `.rodata`/`model_reference_raw`; the four bytes there are
`cf bf 7f 36`, equal to the reported instruction value in little-endian order.
The compiled `.text` ends at `0x8017894c`. This is execution of data, not an
unsupported AME opcode at a legitimate kernel instruction. The preceding
transfer and cause remain unknown; the image did not save interrupted ra/sp.
All three load readbacks match. Decode has no completed numerical acceptance.

The public terminal trap path now captures the interrupted return address and
stack pointer before resetting the stack, then prints them through the same
existing console/host paths. It never dereferences the interrupted stack.
`tools/test_nr_trap.py` checks actual compiled capture order and the C argument
ABI. A fresh boundary-sync build with this trap-only diagnostic is the next
experiment. Normal execution gets no new per-kernel probe or stack read, but
linked addresses can still change.

## Reproduction

Start with the build command in [HANG-WATCH.md](HANG-WATCH.md), using a fresh
`OUT`, `--graph-sync=ame-resync`, and either:

```bash
# Silent comparison; same capacity for the ring control.
--hang-watch _mlir_ciface_kernel_matmul_1x1024x2048:14 \
--console-mode=append-only --console-capacity=524288

# Historical per-kernel output pattern; omit --hang-watch.
--profile-kernels --profile-progress \
--profile-probe _mlir_ciface_kernel_matmul_1x1024x2048:14 \
--profile-sync=ame-resync --console-mode=append-only --console-capacity=4194304
```

Prepare and run only when the previous worker has finished. Commands assume
the repository root and the Python environment documented by the model README.

```bash
MODEL="$PWD/examples/FPGA-BOSCAME/qwen3-0.6b/model"
SHARED="$MODEL/build/quant-opt/shared"
python -B "$MODEL/tools/prepare_model_run.py" \
  --image "$OUT/image/qwen_model.bin" --elf "$OUT/image/qwen_model.elf" \
  --weights "$SHARED/weights/weights-w8a8.bin" \
  --weight-manifest "$SHARED/weights/w8a8-segment.json" \
  --tokenizer "$MODEL/build/tokenizer.bin" --output "$OUT/prepared"
"$MODEL/tools/run_model.sh" "$OUT/prepared" \
  --fpga=5 --capture-seconds=1800 --startup-timeout=900
```

Use `--resume-run=RUN_ID` with the same image if SSH disconnects, as described
by `fpga_run.sh`. Do not start another worker to recover an existing run.
After completion, use the archive command in [KERNEL-TIMING.md](KERNEL-TIMING.md)
with the current `OUT`, `RUN_ID`, and a fresh archive directory. The archiver
automatically handles a bound silent observer; the raw UART remains unchanged.

## Startup prime comparison

The ModelZoo Qwen35 entry performs one `ame_fence()` before its first graph.
The builder exposes this as `--ame-startup=prime`; the default remains `none`.
On FPGA5, the same production image and fixed 28-layer trajectory were run
twice with `ame_startup=none` and both completed all logits/KV checks. A
graph-final `fence rw,rw` comparison also completed all checks. An image
changing only `ame_startup` to `prime` completed prefill and printed
`[model] decode begin position=00000010`, then produced no further UART for
1800 seconds. Its incomplete archive is
[`production-prime-timeout`](../validation/board/console-fix/production-prime-timeout/summary.json).

This is evidence against enabling startup prime as the current fix. It does
not identify the hardware PC or prove that the extra AME operation is
intrinsically unsupported, because the image layout and AME state both change.
The working default therefore stays `ame_startup=none`.
