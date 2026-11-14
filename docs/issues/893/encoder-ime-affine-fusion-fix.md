# Whisper cached KV encoder: affine fusion fix

When all 512x512, 512x2048, and 2048x512 encoder GEMMs were lowered to A100
IME, the cached graph's first token changed from `<|en|>` to `<|vi|>`.
Disabling `affine-loop-fusion` only for that encoder restores the correct
transcription while retaining all three IME shape groups.

The pass-level cause was verified on K3-002 with a first-block diagnostic.
Before fusion, the convolution padding buffer is zero-filled and then its
interior is filled with the audio input. After fusion, the IR contains a
zero store to the same padded buffer inside the convolution reduction loop,
immediately before its load. A pre-IME numeric assertion failed with fusion
enabled and passed without it. The no-IME graph was bitwise identical with
or without the IME layout pass. Adding generic memory-effect traits to the
IME pack/GEMM ops did not fix the fusion outcome; those ops have already
been lowered before the affine fusion pass runs.

The resulting package was built with LLVM pin `3845a205`, Whisper Base FP16,
K3-002 A100 CPUs 8–15, eight OpenMP threads, AI-thread registration, and
1024-bit VLEN. Final RAX SHA256:
`9df359f30253b9ad95ed7cf4c9eedd0b0fe814bdde999a7b7085f14903af0d56`.

| Audio | Encoder once | Decoder total | Core total | End to end | RTF | Output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4.815 s, run 1 | 34.924 s | 7.885 s | 42.809 s | 44.891 s | 9.32 | Reference match, 19 forwards |
| 4.815 s, run 2 | 34.825 s | 7.917 s | 42.742 s | 44.821 s | 9.31 | Reference match, 19 forwards |
| 1.000 s | 34.985 s | 3.279 s | 38.264 s | 40.354 s | 40.35 | `Nor is`, six forwards |

These complete ASR checks preserve the two existing reference transcripts.
They do not establish corpus-level quality or bitwise logits equivalence.
The prior selective-IME KV package took 46.180–46.431 s on the long audio;
the small additional latency improvement here is indicative because the
complete runs were measured separately rather than alternating A/B.
