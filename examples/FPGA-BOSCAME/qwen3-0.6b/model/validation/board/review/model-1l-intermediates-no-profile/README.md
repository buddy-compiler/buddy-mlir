# One-layer selected intermediate and fixed-text FPGA validation

Run `run-550440b96df24b2c` passed exact image/weight/tokenizer DDR readbacks and
strict archival with ELF/NPZ oracle provenance. The compiler-generated graph
executed one 16-token prefill and eight successive decode calls at positions
16..23. The board applied the official template and tokenizer to literal UTF-8
`What is France?` and incrementally decoded nine actual model predictions.

- 414 selected full-tensor intermediate comparisons: max and mean error **0**.
- 27 full last-position vocabulary logits / effective K/V comparisons: max and mean error **0**.
- `MODEL_RUN_NUMERIC_PASS`, `INTERMEDIATES_PASS`, `FULL_LOGITS_KV_PASS`, `FIXED_TEXT_PASS`.
- Partial hidden-state coverage: K RoPE is checked through KV, SwiGLU through
  down-projection input, final norm only at the used last row; int32 accumulators
  and unlisted tensors are not directly captured. See the exact manifest.
- Linker: LLD20. Shared AME msettype asm contract fixed. Kernel profiler disabled;
  intermediate checks/probe UART still add overhead, so these timings are not
  uninstrumented model performance.

The 1-layer truncation outputs ` internallyimizeimizeimizeimizeimizeimizeimizeimize`.
This is not full-model text quality or 28-layer acceptance. UART RX is deferred.

`build-command.sh` rebuilds this configuration from the already generated native
IR, adapter, Triton archive, independent NR reference and weights. Run from the
repository root. Prerequisite graph/kernel commands are in `ATTENTION_POSITION.md`
and `REVIEW.md` at model root. Keep source/artifact hashes for this accepted run;
a changed source requires a new build directory and board validation.

```bash
MODEL=examples/FPGA-BOSCAME/qwen3-0.6b/model
"$MODEL/tools/run_model.sh" "$MODEL/build/review-native-1l/prepared-intermediates-no-profile" \
  --fpga=5 --capture-seconds=600 --startup-timeout=900
python3 "$MODEL/tools/archive_model_run.py" \
  --run examples/FPGA-BOSCAME/build/fpga-runs/run-550440b96df24b2c \
  --build "$MODEL/build/review-native-1l" \
  --image-dir "$MODEL/build/review-native-1l/image-intermediates-no-profile" \
  --prepared-dir "$MODEL/build/review-native-1l/prepared-intermediates-no-profile" \
  --host-graph "$MODEL/build/review-native-1l/host-run/arrays.npz" \
  --quant-reference "$MODEL/build/review-native-1l/quant-nr-trace/arrays.npz" \
  --layers 1 --assets "$MODEL/assets/official" --output <new-archive-directory>
```

A new hardware run gets a different run ID: substitute that actual ID for archive.
