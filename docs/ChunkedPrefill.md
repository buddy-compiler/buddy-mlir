# Chunked Prefill

This document describes chunked prefill for the models built with
`tools/buddy-codegen`: `forward_prefill` processes a fixed number of prompt
tokens per call, and the generated session walks the prompt chunk by chunk.
DeepSeek R1 is the worked example.

## Usage

Set `prefill_chunk` in the model spec:

```json
{
  "variant": "f32",
  "max_token_len": 1024,
  "prefill_chunk": 64
}
```

- a positive integer: the number of prompt tokens per call, at most
  `max_token_len`;
- `true`: the default chunk size, 64;
- `0`, `false` or absent: no chunks (one call over `max_token_len` positions,
  as before).

`models/deepseek_r1/specs/f32_chunked.json` is the f32 spec with chunks of 64.
Every variant is supported. Chunked prefill is not combined with
`tiered_kv_cache` (rejected by `gen_config.py`) or with the partitioned exports
(rejected by `import_model.py`).

```bash
python3 tools/buddy-codegen/build_model.py \
    --spec models/deepseek_r1/specs/f32_chunked.json --build-dir build
```

## How it works

Without chunks, `forward_prefill` takes `max_token_len` tokens and returns the
KV cache and the logits of all positions; whatever the prompt length, it
computes `max_token_len` positions.

With chunks, `import_model.py` traces `forward_prefill` like `forward_decode`,
with `prefill_chunk` tokens instead of one: the token ids `[1, C]`, the start
position, and the KV cache in and out. It returns the logits of the chunk's
last row only, `[1, 1, vocab]` (traced with `logits_to_keep=1`: the LM head
runs on that row only), as the session uses no other row. The two functions
have the same ABI, and the generated `ModelSession` calls `forward_prefill`
through the decode function type.

`ModelSession::prefill()`:

1. zeroes the KV cache: attention masks the positions not written yet by
   multiplying them with 0, so they must hold finite values;
2. runs the prompt in chunks of `C` tokens at positions `0, C, 2C, ...`. The
   last chunk is right-aligned, so that its last row is the last prompt token;
   the rows it recomputes get the same keys and values;
3. keeps the logits of the last prompt token only: those the last chunk
   returns (`logitsData()` ignores its `tokenOffset` in this mode). The other
   calls compute logits nobody reads; if the model library exports
   `buddy_set_prefill_logits(int32_t)`, the session calls it with 0 before
   each of them and 1 before the last one, and the model may skip its LM head
   while it is 0 (the logits of that call are then undefined). The w4g32
   kernels do (`docs/K3DeepSeekR1.md`); other models compute them as before.

A prompt shorter than `C` runs all its tokens but the last as one chunk, padded
with copies of the last of them, and its last token as a decode step. The
padding rows lie past the prompt: they are never attended to, and the next
decode steps overwrite their keys and values.

The import also skips `flash_attention_prefill` for the chunked graph, which
attends to the KV cache like decode, and the session allocates no
`[1, max_token_len, vocab]` prefill logits buffer (622 MB for f32 DeepSeek R1).

## Cost

Prefill takes as many calls as the prompt has chunks. Each call has a cost that
does not depend on the chunk size, so whether chunks pay off depends on the
prompt length and on the platform.

DeepSeek-R1-Distill-Qwen-1.5B, x86 (Xeon Platinum 8575C, 48 threads), prefill
time in seconds (`buddy-cli`, greedy; the generated text is the same in all
columns):

| prompt tokens | no chunks | `prefill_chunk` 64 | `prefill_chunk` 256 |
| --- | --- | --- | --- |
| 63 | 5.6 | 1.6 | 2.4 |
| 64 | 5.7 | 0.9 | |
| 129 | 5.7 | 2.7 | 3.1 |
| 300 | 5.7 | 4.4 | 4.8 |
| 458 | 6.1 | 8.1 | 4.7 |
| 900 | 5.8 | 16.7 | 9.2 |

From the two chunk sizes, a call costs about 0.43 s plus 7.5 ms per row on this
machine, while one call over all 1024 positions takes 5.7 s: chunks are faster
for prompts up to a few hundred tokens and slower for long ones. A prompt
shorter than the chunk size also pays for the first decode step (63 tokens:
1.6 s, 64 tokens: 0.9 s).

These numbers were measured when `forward_prefill` still computed the logits
of all the rows of a chunk. Computing the last row only does not change them
measurably on this machine: the f32 LM head of 64 rows is a few tens of
milliseconds here. It matters more where the LM head is a larger share of a
call: the int4 build for the SpacemiT K3 ([K3DeepSeekR1.md](K3DeepSeekR1.md))
prefills 8% to 18% faster (64 tokens: 3.05 -> 2.70 s, 458 tokens: 11.4 ->
10.2 s, 900 tokens: 21.6 -> 17.6 s).
