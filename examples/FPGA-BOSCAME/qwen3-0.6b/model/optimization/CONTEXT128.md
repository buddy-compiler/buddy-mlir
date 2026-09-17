# Context capacity 128

Completed FPGA5 run `run-2c2dd9c210c1492a`: **MODEL_RUN_NUMERIC_PASS**.
Full 28-layer 16+8 execution, all 27 full logits/effective-KV checks, and fixed
board text processing passed. See
`../validation/board/ame-v05/model-28l-cap128/verification.json` and
`../validation/board/ame-v05/model-28l-cap128-performance.json`.
Prefill graph: 382.35 s; mean decode graph: 76.59 s/token; mean decode including
preparation, selection and KV retention: 80.01 s/token at configured 14.7456 MHz.
This image has no kernel profiler. Earlier SSH loss affected the local relay;
the original remote worker completed without a restart.

The model deployment default is now 128 total tokens, including prompt and
generation. Official checkpoint configuration, weights, 28 layers, vocabulary,
head dimension, and the fixed 16-token prefill plus eight decode steps are
unchanged. This is a deployment cache limit, not a change to model architecture.

All capacity-dependent artifacts are regenerated: Buddy graphs, memref cache
shapes/strides, attention and cache-update Triton kernels, workspace, independent
references, and the final ELF. Historical capacity512 artifacts remain intact.
Explicit `CONTEXT=512`, `--max-cache-len 512`, `--capacity 512`, and image
`--cache-len 512` can still select the historical capacity; every stage must agree.
The RVV target setting `zvl512b` means 512 vector bits and must remain unchanged.

## Artifacts and checks

Current build: `../build/ame-v05/model-28l-cap128/`.

- `import/`, `replacement/`, `nr-prefill/`, `nr-decode/`: actual 28-layer graph.
  Each graph has 957 external kernel calls, with no uncovered large computation.
- `model-lib/`: 46 kernels. Ten capacity-dependent kernels were rebuilt; the
  other 36 retain the previously accepted optimized objects. No test launch or
  NR runtime is included in the archive.
- `weights/`: regenerated from the new parameter layout. Its 598,230,784 bytes
  are identical to the accepted capacity512 weight segment (SHA256
  `0ddd3640545b26c09a3e2fa6829f7675a2d1a5b128490635b7037efd697e33a7`).
- `fp32-reference/`, `quant-triton-host/`, `quant-nr-fpga/`: newly executed
  capacity128 references. Prefill argmax 49000; decode IDs
  `[374,264,3146,7407,304,4787,5159,11]` agree with FP32.
- `image/`, `prepared/`: unprofiled NR image and exact DDR deployment plan.
  Per-kernel profiler fences/UART are absent; normal kernel and graph-end
  synchronization remain. Full per-step logits/effective-KV checks remain.

The ten new kernel specializations passed FPGA5 independently in
`run-28100ee541de485d`, including 83 varying-length attention checks with NaN
inactive entries and sentinel guards. Mask and softmax errors remain within
their existing oracle tolerances; attention and KV results were exact. Evidence:
`../validation/board/ame-v05/cap128-kernels/verification.json`.

## Memory

Actual linked arenas, excluding the separate low-address heap:

| Item | Capacity512 | Capacity128 |
|---|---:|---:|
| K + V cache | 112 MiB | 28 MiB |
| Persistent high-address total | 829,741,824 B | 706,566,912 B |
| Prefill constant malloc-site sum | 404,347,735 B | 140,106,583 B |
| Decode constant malloc-site sum | 356,201,515 B | 91,960,363 B |

Malloc-site sums are static estimates, not measured heap peaks. See the build's
`memory.json`; the runtime reports actual `scratch_bytes` after graph return.
Full-capacity cache expansion/copies shrink by four; linears and lm_head do not.

## Reproduction

From the repository root, the exact commands used are preserved in:

- `../build/ame-v05/cap128-kernels/BUILD.md` and `cap128-build.sh` beside it:
  case generation, Triton compilation, host/NR suites, kernel overlay/archive.
- `../build/ame-v05/model-28l-cap128/graph-command.sh`: import, replacement,
  prefill/decode lowering, host bridge, compiled-host execution.
- `../build/ame-v05/cap128-reference.sh`: FP32 and both quantized references.
- `../build/ame-v05/model-28l-cap128/build-image.sh`: image and deployment plan.

These commands use separate Python environments for Buddy bindings (3.11) and
Triton (boscame), plus LLD20. Keep recorded builds immutable; use fresh output
directories for new variants. Do not compile into the symlink kernel overlay.

To run the prepared image without creating a second UART owner:

```bash
MODEL=examples/FPGA-BOSCAME/qwen3-0.6b/model
"$MODEL/tools/run_model.sh" "$MODEL/build/ame-v05/model-28l-cap128/prepared" \
  --fpga=5 --capture-seconds=2400 --startup-timeout=900
```

Kernel PASS and a linked ELF are not complete 28-layer acceptance. A completed
run still requires `archive_model_run.py` to check all nine graph results and
27 complete logits/KV comparisons against `quant-nr-fpga`, plus text handling,
deployment, and compiler provenance. Changes to capacity and optional profiler
mean timing alone cannot isolate the cause of the historical capacity512 stalls.
