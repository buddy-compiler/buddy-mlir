# Silent kernel observation and graph synchronization comparison

This is a local diagnostic build procedure for the 28-layer, capacity-128,
16-token prefill plus eight-step decode case. It is not a performance or
numerical acceptance result. The local build commands do not upload, reset,
start a job, or change FPGA configuration. The console comparison below was
authorized for FPGA5 on 2026-09-20; server output stays inside
`/home/hjuser/Desktop/fpga-tester-ISCAS`, without writing through its external
directory symlinks or changing hardware configuration.

## What is observed

`--hang-watch _mlir_ciface_kernel_matmul_1x1024x2048:14` wraps the zero-based
14th call to that symbol in each graph invocation. Computation stays in the
linked Triton/Buddy kernel. The wrapper snapshots its three rank-2 descriptors
and records ENTER/RETURN in shared DDR, without UART or additional AME work.
Graph boundaries record GRAPH_RETURN, SYNC_DONE and COLLECT_BEGIN/DONE.
NH emits low-frequency `NRWATCH1` frames directly to UART, independently of
the RA console ring.

`--hang-console=blocking` preserves console backpressure. `bounded` limits
wait iterations and counts dropped bytes; that variant is diagnostic only.
It must not be accepted as a complete numerical UART trace. Do not change
console mode while comparing graph synchronization, since that introduces a
second variable.

`--graph-sync=ame-resync` is the unchanged default: after each graph it calls
the public `ame_fence()`, which includes extra 1x1 AME operations.
`--graph-sync=fence` uses the developer's exact
`asm volatile("fence rw, rw" ::: "memory")` sequence instead. It does not
change AME/RVV fences in the kernel archive. `--profile-sync` is a separate
option for profiler wrappers; profiling is disabled in this comparison.

The supplied screenshots and existing fence locations are discussed in
[the cache contract note](../../validation/ame-v05-cache-contract.md).

## Fixed-input console transport

ModelZoo `examples/buddy-qwen35-fpga/platform/nr/qwen35_nr_console.c` uses an
append-only console for fixed input, and a blocking ring only for dynamic input.
The public runtime now supports the same finite-output policy with
`NR_CONSOLE_APPEND_ONLY=1` and `NR_CONSOLE_CAPACITY`. The model builder exposes
`--console-mode=append-only --console-capacity=524288`. Its existing default
remains a 65536-byte ring until this comparison establishes board behavior.

Append-only output never reads or waits for NH's consumer cursor. If full, it
keeps earlier bytes and sets a sticky overflow flag. NH reports console overflow
and final NR FAIL even when model arithmetic returned success. It is rejected
with interactive mode or bounded-ring diagnostics. The mode and capacity are
recorded in `w8a8-image-plan.json`; the runtime does not depend on `references`.

For transport comparisons, pass the SAME explicit capacity for both modes and
keep `--graph-sync`, profiler settings, graph, archive, weights and reference
identical. Use the silent watch to avoid per-kernel printing. Changing console
mode still changes runtime instructions, layout and cache traffic; a passing
append-only run alone is not proof that stale consumer acknowledgements caused
earlier stalls. A silent validation run may never fill even the smaller ring.

The first local pair is under `build/console-fix-20260920/{ring-watch,append-watch}`.
`comparison.json` records equal graph, adapter, model-entry, AME-sync and watch
objects, plus identical target matmul bytes. Both use capacity 524288 and the
original graph-final AME resync. No startup prime or graph-boundary changes are
mixed into this pair.

## Rebuild locally

Run from the repository root with the model build Python environment active.
The paths below reuse the existing compiled graph, archive, official resources
and independent numerical oracle. They do not rebuild kernels or download
weights. Choose a fresh `OUT` for each build to preserve previous evidence.

```bash
MODEL="$PWD/examples/FPGA-BOSCAME/qwen3-0.6b/model"
SHARED="$MODEL/build/quant-opt/shared"
SYNC=fence                     # Use ame-resync for the control.
OUT="$MODEL/build/hang-watch-reproduction/$SYNC"
[[ ! -e "$OUT" && ! -L "$OUT" ]] || exit 1
mkdir -p "$OUT"
python -B "$MODEL/tools/build_nr_w8a8_image.py" \
  --repo-root "$PWD" --linker /usr/bin/ld.lld-20 \
  --report "$SHARED/replacement/triton-call-replacement.json" \
  --segment "$SHARED/weights/w8a8-segment.json" \
  --graph-ir "$SHARED/nr-prefill/forward_prefill.ll" \
  --decode-ir "$SHARED/nr-decode/forward_decode.ll" \
  --archive "$SHARED/model-lib/libqwen_triton.a" \
  --adapters "$SHARED/replacement/qwen_triton_adapters.c" \
  --output "$OUT/image" --layers 28 --cache-len 128 \
  --prefill-len 16 --decode-steps 8 --prompt-text 'What is France?' \
  --tokenizer-blob "$MODEL/build/tokenizer.bin" \
  --reference-arrays "$MODEL/build/ame-v05/model-28l-cap128/quant-nr-fpga/arrays.npz" \
  --reference-metadata "$MODEL/build/ame-v05/model-28l-cap128/quant-nr-fpga/quant-reference.json" \
  --hang-watch _mlir_ciface_kernel_matmul_1x1024x2048:14 \
  --hang-console blocking --graph-sync "$SYNC" > "$OUT/build.log" 2>&1
```

Check `image/elf-audit.json`, `image/image.json`, `image/hang-watch.json` and
`image/w8a8-image-plan.json`. The latter must record
`profile_kernels=false`, `completion_sync=null`, the chosen
`graph_completion_sync`, and the same model and workspace dimensions in both
builds. Both graph LLVM inputs, adapters and static archive must have identical
input hashes. Compare the linked target kernel's bytes too: an unchanged
source alone is not machine-code evidence. Synchronization changes can still
alter overall image layout and timing.

The local builds prepared on 2026-09-20 are under
`build/hang-watch-20260920/blocking/` (control) and
`build/hang-watch-20260920/fence-graph/` (single graph-final fence).
A successful local build is not a board PASS. Neither is deployed by this
procedure.

The [local comparison record](../validation/hang-watch-graph-sync-local.json)
records ELF and target-function hashes. The target's 640 machine-code bytes,
both graph objects, adapters, runtime objects and observer object are identical
between the prepared images. The candidate contains no `ame_fence` or
`ame_resync` symbol; ELF audit counts drop from 131 to 121 AME instructions
because the unused resync code is garbage-collected. The target function moves
from `0x80176538` to `0x8017647c`, so overall code layout is not controlled by
this comparison.

## Decode observations

For a later authorized run, retain its original `uart.raw.log` and decode a
local copy:

```bash
python3 -B "$MODEL/tools/decode_hang_watch.py" \
  --uart /path/to/uart.raw.log --output "$OUT/decoded" \
  --console-capacity=65536
```

Use the actual `console.capacity_bytes` from the image plan, including 524288
for the console comparison. Older images without that field use 65536.

`uart.ra.log` removes complete NH frames byte-for-byte; `nh-watch.json` retains
samples, descriptors, raw hexadecimal fields, dropped-byte and truncation
information. It does not grant numerical acceptance or infer a root cause.

* An observed KERNEL_RETURN supports that the selected C call returned; it
  does not alone prove all AME/DDR activity completed.
* GRAPH_RETURN followed by no SYNC_DONE narrows the observed interval to
  graph-final synchronization or visibility/observation of subsequent marks.
* Only ENTER with ongoing NH heartbeats does not distinguish a kernel stall,
  a RA trap, or stale RA records. A repeated even sequence is not a guarantee
  of fresh or mutually consistent cache lines.
* Console wait flags and producer/consumer counters are separate samples.
  Progress only with bounded console is evidence to investigate the shared
  ring, not proof of a particular cache mechanism.
* Missing return UART in old profiler logs is weaker evidence: printing that
  marker can itself block. No instruction-failure conclusion follows from it.

Use the exact descriptor offset/strides and element sizes from the typed
adapter when checking buffer ranges. The generic `MemRef2` record does not
carry dtype; addresses and shapes alone cannot prove in-bounds access.
