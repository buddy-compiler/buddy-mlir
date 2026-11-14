# Numerics policy

Every performance change belongs to one of two classes. Decide which before
implementing, and state it in the commit message and PR.

## Bit-identical

Same operations, same operands, same order: only scheduling, data movement,
loop structure, parallel split of independent work, or memory placement
change. Examples: looping an unrolled block, computing several heads per
work item where each head does exactly what it did alone, moving a buffer
into TCM, skipping a computation whose result is unused.

Validate:

- Kernel level: write the outputs of baseline and candidate to files and
  `cmp` them.
- Model level: the generated text of several prompts (short, medium, long
  context) has the same hash as the baseline's. Greedy decoding diverges
  within a few dozen tokens after any rounding change, so equal text over
  100+ tokens is strong evidence.

## Numerics-changing

Anything that changes rounding: a different summation order (splitting a
reduction across threads, reassociating, folding vector halves before a
reduction), lower precision (f16 accumulation, f16 KV cache or scales), a
different quantization, a different kernel (IME fp16 products vs RVV f32).

Validate:

- Measure perplexity on a fixed text (`buddy-cli --perplexity`, wikitext-2
  test, `--ppl-context 512 --ppl-chunks 40`, scored like llama-perplexity) for
  baseline and candidate. Report both.
- Compare against the reference implementation (e.g. f16/f32 llama.cpp, or
  numpy for one kernel) and give the largest error relative to the largest
  output.
- Show that the text is still coherent; note after how many tokens it
  departs from the baseline's.
- Ask the maintainer before submitting: some numerics changes are not wanted
  even when they are faster.

Prefer to make a numerics-changing optimization opt-in (a spec option)
unless the maintainer agrees to make it the default.
