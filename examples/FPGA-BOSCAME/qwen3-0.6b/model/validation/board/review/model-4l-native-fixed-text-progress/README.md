# Four-layer native-K fixed-text FPGA validation

Run `run-b8d3ed3ead854a24` passed exact segment upload/readback, embedded numeric
oracle provenance, all 27 last-position full-vocabulary logits/effective KV checks
(max/mean error zero), typed-IR-derived kernel call counts, official template and
tokenizer IDs, and board incremental text decode.

Prompt is literal UTF-8 `What is France?`. FPGA template/encode produces 16 IDs;
those actual IDs feed the graph. One prefill then eight successive decode calls
retain KV and advance position 16..23. Nine predictions include the prefill output.
This truncated four-layer model emits `noimderalsdayliberatorally-`; that output
is not representative of the full 28-layer model. Full-model acceptance is pending.

The image has per-kernel progress UART inside graph timings. Do not report these
as uninstrumented model throughput. The strict numerical/profile/text reports,
ELF, maps, objects, Triton evidence, input hashes and raw UART are archived here.

Reproduce archival from repository root into a NEW directory:

```bash
MODEL=examples/FPGA-BOSCAME/qwen3-0.6b/model
python3 "$MODEL/tools/archive_model_run.py" \
  --run examples/FPGA-BOSCAME/build/fpga-runs/run-b8d3ed3ead854a24 \
  --build "$MODEL/build/review-native-4l" \
  --image-dir "$MODEL/build/review-native-4l/image-fixed-text-progress" \
  --prepared-dir "$MODEL/build/review-native-4l/prepared-fixed-text-progress" \
  --host-graph "$MODEL/build/review-native-4l/host-run/arrays.npz" \
  --quant-reference "$MODEL/build/review-4l/quant-nr-order/arrays.npz" \
  --layers 4 --assets "$MODEL/assets/official" --output <new-directory>
```
