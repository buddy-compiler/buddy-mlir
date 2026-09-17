# Fixed-text trace verifier

`tools/check_fixed_text.py` performs post-run host validation only. It reads the
exact fixed-prompt manifest and official tokenizer assets, applies the official
single-user chat template with thinking disabled, and compares every board
prompt token ID. This host oracle does not supply inputs to the board run.

It requires one prefill at position 0, eight decode calls at positions 16–23,
nine model-selected prediction IDs, correct previous-token inputs, official EOS
flags, and exactly one completion marker of each kind. Incremental UTF-8 bytes
are checked against the official ByteLevel vocabulary with streaming replacement
semantics and against the official complete decoder. Incomplete UTF-8 prefixes
must remain buffered until completed or flushed at the end.

The final UART payload is read by the declared byte length and compared exactly
to its announced hex. Payload text resembling diagnostics or PASS/FAIL markers
is not parsed as control. Missing, duplicate, malformed, reordered, truncated or
inconsistent records cause a nonzero exit status.

```bash
MODEL=examples/FPGA-BOSCAME/qwen3-0.6b/model
python3 "$MODEL/tools/check_fixed_text.py" \
  --uart <completed-run>/uart.raw.log \
  --image-plan <exact-image>/w8a8-image-plan.json \
  --assets "$MODEL/assets/official" \
  --output <new-report.json>
```

Successful status is `FIXED_TEXT_PASS`; failures are `NOT_ACCEPTED`. The archive
tool automatically invokes this check when the image plan has `fixed_prompt`,
and requires `--assets` for that mode. Numerical logits/KV and kernel profiles
remain separate mandatory archive checks when present.

Ten host-only tests passed (see `fixed-text-checker-tests.log`), including real
official tokenizer/template checks, split UTF-8 tokens, malformed bytes, final
incomplete sequence flush, skipped special tokens, synthetic marker-like payload,
24 trace corruption cases and manifest tampering. These tests are not FPGA text
acceptance; an actual board trace is still required.
