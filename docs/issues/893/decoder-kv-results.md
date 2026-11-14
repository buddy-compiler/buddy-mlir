# Whisper Base K3 A100: current-position logits and decoder KV cache

The cached encoder remains a separate one-time graph. A new optional 32-token
KV package uses a first-token graph to populate self- and cross-attention
state, then a one-token step graph with persistent static caches. Both return
only the current logits row, rather than all 32 positions. The implementation
follows the repository's DeepSeek static-cache import and session-state pattern
while retaining Whisper's encoder-decoder attention structure.

## K3-002 measurements

Whisper Base FP16; A100 CPUs 8–15; eight OpenMP threads; AI-thread
registration; VLEN 1024; LLVM pin `3845a205`. Both audio files, weights,
and CLI match the previous cached-encoder comparison. Weight loading and RAX
extraction are outside the stage timers. The final RAX SHA256 is
`27fc28db94c542547b453a393c8ec76d91b9a94427be4f360390ee09b73b1c3b`;
the packed three-stage weight file SHA256 is
`2a267515d26b6c06346a54a74986fff788a6f90604bd8bbc1fcca3d396a4f349`.

| Audio | Encoder once | Decoder total | Core total | End to end | RTF | Output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4.815 s, run 1 | 36.252 s | 7.949 s | 44.201 s | 46.287 s | 9.61 | Reference match, 19 forwards |
| 4.815 s, run 2 | 36.413 s | 7.933 s | 44.346 s | 46.431 s | 9.64 | Reference match, 19 forwards |
| 1.000 s | 36.384 s | 3.233 s | 39.617 s | 41.675 s | 41.67 | `Nor is`, six forwards |
| 4.815 s, final package | 36.206 s | 7.883 s | 44.089 s | 46.180 s | 9.59 | Reference match, 19 forwards |
| 1.000 s, final package | 36.148 s | 3.274 s | 39.421 s | 41.515 s | 41.52 | `Nor is`, six forwards |

The first decoder call takes about 1.48 s and subsequent calls about 0.36 s.
The previous selected-IME cached encoder package took 68.786–69.192 s on
the same long audio, so the KV package is about 1.49x faster end to end.
The long transcription has the same 19 token IDs and EOS position as that
package and produces `Nor is Mr. Quilter's manner less interesting than his
matter.`

## Numerical checks and scope

- A host build using identical encoder state, FP16 weights, and fixed decoder
  token IDs produced bitwise-identical full-vocabulary logits at positions
  0, 1, 18, and 31 when selecting one row after the decoder hidden state.
- A separate host KV-versus-full-decoder check with fixed token IDs gave
  max absolute logits differences of 0 at position 0, 0.117 at position 1,
  and 0.148 at position 18; the respective mean differences were 0,
  0.0208, and 0.0271. Across all first 19 positions, the largest absolute
  difference was 0.219. All compared rows were finite and had the same top-1.
- Both complete K3 transcriptions and the one-second sample matched the
  reference text. These checks do not establish corpus-level accuracy or
  bitwise equivalence between cached and full decoder attention.

The package is limited to 31 generated tokens. The KV step graph skips
`batchmatmul-optimize` because its current vector load requires unit inner
stride; the initial graph and encoder keep the existing selected optimizations.
