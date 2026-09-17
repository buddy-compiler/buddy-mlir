# Native-layout attention: one-layer FPGA model run

`run-390dfc362f234ec6` passed strict archive validation with the exact ELF, padded
boot image, two uploaded segment hashes, worker DDR readback results, independent
NR reference NPZ, repacked oracle and actual ELF-embedded oracle bytes.

Scope: real checkpoint, **one decoder layer**, vocabulary 151936, 16-token prefill
and eight sequential decode calls. All 27 last-position logits / effective K/V
checks have max and mean error zero. This is not hidden-state, 28-layer or UART
text-interaction acceptance. Fixed input IDs were used in this particular run.

`kernel-profile-verification.json` independently derives **39 kernel calls per
entry** from actual LLVM RAW call sites and generated C adapter bodies, excluding
bridge double counting. All nine measured call totals agree. Kernel cycle sums
fit within graph compute cycles. Native QK consumes the cache layout directly,
so the previous layout_k call is absent.

Instrumented prefill graph compute: **554,858,750 cycles**; complete measured
model phases: **565,132,864 cycles**. Decode graph compute is
**290,779,337–291,425,808 cycles/step**; complete measured model phases are
**300,493,501–301,139,829 cycles/step**. Model phases sum preparation + graph +
selection + cache retention, excluding numerical oracle and UART. Detailed
per-step phase/scratch metrics are in the profile verification. The remainder
outside kernels includes several types of work and is not pure memory time.

Reproduce archival from repository root with a NEW output directory:

```bash
MODEL=examples/FPGA-BOSCAME/qwen3-0.6b/model
python3 "$MODEL/tools/archive_model_run.py" \
  --run examples/FPGA-BOSCAME/build/fpga-runs/run-390dfc362f234ec6 \
  --build "$MODEL/build/review-native-1l" \
  --image-dir "$MODEL/build/review-native-1l/image-oracle-profile" \
  --prepared-dir "$MODEL/build/review-native-1l/prepared-oracle-profile" \
  --host-graph "$MODEL/build/review-1l/host-trace/arrays.npz" \
  --quant-reference "$MODEL/build/review-1l/quant-nr-order/arrays.npz" \
  --layers 1 --output <new-archive-directory>
```

The tool refuses existing output directories. `verification.json` binds archived
source/IR/object/library/map/ELF files by SHA256. Large weights and NPZ arrays are
identified by source path, size and hash instead of duplicated. Downloaded logs
do not include huge DDR readback files: the archive explicitly attributes those
checks to the completed remote worker, with the matching upload manifest.
