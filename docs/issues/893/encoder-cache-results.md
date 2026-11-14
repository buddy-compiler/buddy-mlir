# Whisper Base K3 A100: encoder reuse

Based on `perf/whisper-k3-a100` at `88b39121`. The new optional
`base_f16_k3_a100_cached32.json` package exports separate encoder and decoder
entrypoints. The runtime calls the encoder once and retains its F32 output
for all decoder forwards. The FP16 packed weights are byte-identical to the
previous 32-position package (SHA256
`272c89542e811ee9b30ae0bfea384292ab061eef609098ebcb29c193283a5096`).

## K3-002 results

Conditions: A100 CPUs 8–15, eight OpenMP threads, AI-thread registration,
1024-bit VLEN, Whisper Base FP16, the same 16 kHz example audio and
`buddy-cli`. Weight loading and RAX extraction are outside the stage timers.
The cached package uses the shared RVV/linalg lowering with
`BUDDY_WHISPER_K3_IME=OFF` and LLVM pin `3845a205`. The tested cached RAX
SHA256 is `0ca56d6489cfd71a142001c30ae9b57614f2fe361bf27887ff765ff9d578a521`.
The long audio SHA256 is
`473b1bb573e8f7cdcd639c6e318036236190107b66bfe6892286121a92d464c1`;
the one-second clip SHA256 is
`3ac6dc45be8b48e0377bbf67921cf0530b3579f9c293749854853a347dc2da42`.

| Audio | Encoder once | Decoder total | Core total | End to end | RTF | Text |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4.815 s, run 1 | 44.320 s | 32.287 s | 76.607 s | 78.699 s | 16.34 | Reference match |
| 4.815 s, run 2 | 44.274 s | 31.406 s | 75.680 s | 77.776 s | 16.15 | Reference match |
| 1.000 s | 44.261 s | 9.939 s | 54.200 s | 56.285 s | 56.29 | Reference match |

Both long runs made 19 decoder forwards and generated EOS. Their full text is
`Nor is Mr. Quilter's manner less interesting than his matter.` The short run
made six decoder forwards and produced `Nor is`. These are the same two
transcriptions used by the earlier issue #893 validation; they establish
sample-level correctness, not corpus-level accuracy.

The earlier 32-position package without encoder reuse took 678.098–679.645 s
end to end on the same board and audio. The new 77.776–78.699 s range is
about **8.6–8.7× faster**. These complete runs were measured separately,
not in an interleaved A/B sequence. The gap is much larger than the observed
repeat variation. Decoder work is now about 1.65–1.70 s per token, while the
one-time encoder still costs about 44.3 s.

## Correctness boundary

PyTorch FP16 gave bitwise-identical logits when passed a separately computed
encoder output. A host build of the split Buddy graph produced the expected
first token `<|en|>`. On K3, compiling the split encoder with the A100 IME
pass instead produced `<|vi|>` and an encoder hidden state that differed from
the previously validated full graph (mean absolute difference 0.88 on a
zero-feature probe). Disabling IME restored the correct first token and both
complete transcriptions. CMake rejects `encoder_cache=true` together with
The follow-up selective IME configuration retains correctness; see
`encoder-ime-selective-results.md`.

The cached package is limited to 31 generated tokens. For longer
transcriptions use the existing 448-position package. The next performance
work is reducing the still-fixed 32-position decoder computation and
investigating why enabling the remaining 512x512 IME group changes the split
encoder's output.
