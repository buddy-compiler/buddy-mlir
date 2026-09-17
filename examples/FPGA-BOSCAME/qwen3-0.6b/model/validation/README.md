# Model validation evidence

Git tracks a compact record of the Qwen FPGA checkpoint: validation reports and
JSON summaries, raw UART and other logs, numeric/intermediate/profile/text
verification, deployment manifests, linker maps, and historical source and
validator snapshots. The per-run READMEs describe the scope of each result;
one-layer and four-layer results do not establish full 28-layer FPGA acceptance.

The complete local board archives remain on disk. `model/.gitignore` excludes
their generated objects, libraries, ELF images, binary arrays, LLVM/MLIR/Triton
IR and generated assembly from Git. Handwritten assembly under archived
`sources/` directories remains tracked. Logs, JSON reports and maps are retained.
References to archived ELF/IR/object files in per-run READMEs describe these
complete local archives, not all the files included in a fresh Git checkout.

`verification.json` retains the original artifact paths and SHA256 records,
including `archive_sha256` entries for omitted files. They record provenance;
checking every recorded hash requires the corresponding local artifacts.
Large weights and reference NPZ arrays may already be represented by path,
size and hash rather than copied into the local archive. No generated files
are deleted by these ignore rules.

For final native-K 28-layer coverage, Git retains
[`coverage.md`](final-native-28l-coverage/coverage.md),
[`kernel-isa.json`](final-native-28l-coverage/kernel-isa.json) and
[`report-validation.json`](final-native-28l-coverage/report-validation.json).
The approximately 28 MB `operator-mapping.json` remains local and ignored.
The validation report records its SHA256; coverage PASS concerns operator
classification and artifact/ABI provenance, not full-model FPGA numerics.

## Regenerate local detail

Run from the repository root. The existing graph/kernel build procedures and
toolchain prerequisites are in [`ATTENTION_POSITION.md`](../ATTENTION_POSITION.md)
and [`REVIEW.md`](../REVIEW.md). The coverage command below uses Python 3.11
matching the Buddy MLIR bindings and requires the existing native 28-layer build.
It writes the detailed inventory and coverage summaries to an ignored build
directory, leaving the recorded evidence unchanged:

```bash
MODEL=examples/FPGA-BOSCAME/qwen3-0.6b/model
PYTHONPATH=build-python/python_packages python3 -B "$MODEL/tools/final_operator_mapping.py" \
  --build "$MODEL/build/review-native-28l" \
  --layout "$MODEL/validation/weight-layout.json" \
  --output "$MODEL/build/reproduced-final-native-28l-coverage"
```

To reconstruct the four-layer local archive from its original run, build, image,
deployment and reference inputs, use the existing archival command with a new
output directory:

```bash
MODEL=examples/FPGA-BOSCAME/qwen3-0.6b/model
python3 "$MODEL/tools/archive_model_run.py" \
  --run examples/FPGA-BOSCAME/build/fpga-runs/run-b8d3ed3ead854a24 \
  --build "$MODEL/build/review-native-4l" \
  --image-dir "$MODEL/build/review-native-4l/image-fixed-text-progress" \
  --prepared-dir "$MODEL/build/review-native-4l/prepared-fixed-text-progress" \
  --host-graph "$MODEL/build/review-native-4l/host-run/arrays.npz" \
  --quant-reference "$MODEL/build/review-4l/quant-nr-order/arrays.npz" \
  --layers 4 --assets "$MODEL/assets/official" \
  --output "$MODEL/build/reproduced-archive-4l"
```

The archival tool refuses an existing output directory. Other accepted run
configurations have their commands in their own READMEs under `board/review/`.
A fresh checkout must first restore or regenerate the required local build and
reference inputs. New FPGA execution produces a new run ID and new evidence;
use that actual run ID when archiving. Rebuilding does not recreate historical
UART logs or guarantee the original artifact hashes after source/toolchain changes.
