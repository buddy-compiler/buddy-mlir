# Comparing with llama.cpp

A speed comparison is meaningful only between equivalent computations.

## Align the formats

- Weights: llama.cpp `Q4_0` is int4 with one f16 scale per 32 weights, like
  Buddy's `w4g32`. But `llama-quantize` keeps `output.weight` (the LM head)
  in `Q6_K` by default, more bytes per decoded token than an int4 head.
  Quantize with `--output-tensor-type q4_0` for an aligned head.
- KV cache and activations: note their precision on both sides (f32 vs f16
  KV cache changes attention bandwidth at long contexts).
- Report the perplexity of both on the same tokens (below): equal speed at
  worse accuracy is not parity.

## Measure the same way

- `llama-bench -p <n>` measures prefill of n tokens, `-n 128` decode from an
  empty context. Measure Buddy's decode at the same context lengths
  (`buddy-cli` decodes after the prompt, so also measure from a short
  prompt).
- Run each framework the way it is meant to run on the platform (see the
  platform's board setup in the `hardware-targets` skill).
- Same threads, same board, alternating runs.

## Perplexity

- llama: `llama-perplexity -m model.gguf -f wiki.test.raw -c 512 --chunks 40`.
- Buddy: `buddy-cli --model m.rax --perplexity ids.txt --ppl-context 512
  --ppl-chunks 40` (`docs/Runtime.md`), with the ids of the same text from
  the model's Hugging Face tokenizer or `llama-tokenize --ids` (no BOS), so
  that both see the same tokens. Buddy scores through its decode path (each
  chunk starts from an empty context).
- llama.cpp's batched evaluation (`-b`/`-ub` > 1) gives a slightly different
  value from one-token-at-a-time evaluation (`-ub 1`); compare like with
  like.

Measured results per platform: `spacemit-k3-decode.md` ("Accuracy and the
llama.cpp baseline").
