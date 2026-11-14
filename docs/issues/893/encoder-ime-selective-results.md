# Whisper cached encoder: selective A100 IME

The cached 32-position Whisper Base FP16 graph has three groups of encoder
GEMM shapes: 24 calls with KxN 512x512, six with 512x2048, and six with
2048x512. On K3-002, the original split-graph setting that lowered all three
groups to IME produced an incorrect first token (`<|vi|>`). The no-IME package
produced the reference transcription.

Single-group and pair-group experiments kept the first token correct. The
fastest validated pair selected 512x2048 and 2048x512, leaving 512x512 on its
existing path. The two selected groups have 12 sets of each A100
`smt.vfwmadot` encoding in the final model library. The decoder graph stays
on the existing path. This selection is now applied when the cached spec is
built with `BUDDY_WHISPER_K3_IME=ON`.

Conditions: K3-002 A100 CPUs 8–15, eight OpenMP threads, AI-thread
registration, 1024-bit VLEN, LLVM pin `3845a205`, Whisper Base FP16 weights,
and the same audio and `buddy-cli` as the encoder-cache baseline. The final
RAX SHA256 is `dfa6d29dd3d854a4c73ed115af462ca237588b91d7654bb5a93fbb46e523329c`.

| Audio | Encoder once | Decoder total | Core total | End to end | RTF | Output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4.815 s, run 1 | 35.993 s | 30.710 s | 66.703 s | 68.786 s | 14.29 | Reference match, 19 forwards |
| 4.815 s, run 2 | 35.951 s | 31.156 s | 67.107 s | 69.192 s | 14.37 | Reference match, 19 forwards |
| 1.000 s | 35.845 s | 9.710 s | 45.555 s | 47.659 s | 47.66 | `Nor is`, six forwards |

Against the no-IME cached package's 77.776–78.699 s complete runs, this
selective IME package saves about nine seconds on the long audio. The result
is about 9.8–9.9x faster than the earlier 32-position package that recomputed
the encoder every token; those baseline runs were separate, not interleaved.

This is a correctness-preserving selection for the validated sample inputs,
not a repair of the underlying full-IME numerical mismatch. The 512x512
group needs a stage-level numeric bisect before it can be enabled for the
cached encoder. The package still supports at most 31 generated tokens.
