# One-layer FPGA numerical acceptance (review run)

Run `run-015b4fabf4014edc` on FPGA5 completed with `result.json.status=OK` and
`[nr] RA returned: PASS`. Both uploaded segments matched DDR readback.
`numeric-verification.json` reports `FULL_LOGITS_KV_PASS`.

This is the real-checkpoint **one-layer** Buddy graph with W8A8 Triton linear,
embedding, RMSNorm, SiLU, attention and KV-write replacements, full vocabulary
151936, one 16-token prefill and eight sequential decode steps (positions 16–23).
Prefill graph start is position 0; its last-output comparison is position 15.
Prefill chooses token 33067 and each of the eight decode calls chooses 11853.
The graph chooses tokens; embedded independent-reference arrays are read only
by validation, after graph computation and argmax selection.

All 27 FPGA comparisons passed with max and mean absolute error **0** against
the independent `nr-fpga` arithmetic reference: every last-position vocabulary
logit plus every valid K/V element for each of nine calls. This does not capture
hidden states or logits at every intermediate prefill position. The compiled-host
comparison is separately sampled: its worst of nine selected logits is
9.5367431640625e-7 and worst argmax logit difference is 3.814697265625e-6;
the sampled K/V values match exactly.

Graph-only prefill took **1,183,615,186 cycles**; decode took
**784,100,292–784,135,605 cycles/step**. Total graph time is 7,456,607,050 cycles.
The NR launch total, including oracle/trace/setup overhead, is 7,591,649,001 cycles.
See `verification.json` for each step and all source hashes.

The executed legacy `numeric-reference.json` is preserved byte-for-byte and lacks
the schema/dimensions fields used by newer builders, so the checker's
`metadata_verified=false` is intentional. Separate archive crosschecks verified
prompt IDs, layer/cache dimensions, reference trajectory, the NPZ-to-blob packing,
and actual embedded blob bytes at `model_reference_raw=0x8002c580` in the exact
ELF whose hash matches the uploaded deployment. `verification.json` records
these supplemental checks without altering the original manifest.

Recheck from repository root (large original build artifacts remain local):

```bash
MODEL=examples/FPGA-BOSCAME/qwen3-0.6b/model
python3 "$MODEL/tools/check_board_trace.py" \
  --uart "$MODEL/validation/board/review/model-1l-full/uart.raw.log" \
  --host-graph "$MODEL/build/review-1l/host-trace/arrays.npz" \
  --quant-reference "$MODEL/build/review-1l/quant-nr-order/arrays.npz" \
  --embedded-reference "$MODEL/validation/board/review/model-1l-full/numeric-reference.json" \
  --output "$MODEL/build/review-1l/rechecked-numeric-verification.json"
```

Raw logs, deployment plan, image build commands/input hashes, link map, ELF audit,
static archive manifest/symbols and reference metadata are archived here. Large
weights, ELF/bin and NPZ arrays are referenced by path, size and SHA256 rather
than duplicated. The ELF audit found 38,526 instructions including 131 AME and
254 RVV, with zero undefined symbols.

This result does **not** establish 28-layer FPGA acceptance or UART-driven text
interaction. Fixed input IDs are intentional for this numerical regression;
board tokenizer and RX diagnostics have separate evidence.
