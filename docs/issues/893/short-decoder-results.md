# Whisper Base on K3 A100: 32-position decoder experiment

This continues issue #893 from `perf/whisper-k3-a100` at `7244c6d7`.
The importer now uses the spec's `max_token_len` for its traced decoder input.
The new FP16 spec sets that length to 32. The model weights, audio input,
compiler pipeline, and A100 IME setting are unchanged. This package accepts
`--max-tokens` from 1 through 31; the 448-position spec remains available for
longer transcriptions.

## Conditions

- K3-002, A100 CPUs 8–15, eight OpenMP threads, AI-thread registration,
  1024-bit VLEN, observed CPU frequency 1.8 GHz.
- LLVM pin `3845a205`; Whisper Base FP16 weights. The generated weight file
  has the same SHA256 in both packages:
  `272c89542e811ee9b30ae0bfea384292ab061eef609098ebcb29c193283a5096`.
- 4.815-second, 16 kHz example audio, SHA256
  `473b1bb573e8f7cdcd639c6e318036236190107b66bfe6892286121a92d464c1`.
- The 448-position RAX is the previously validated package, SHA256
  `75f3ef47a318c373cd1f5dc1a70186bd55d795a9c855dc329a24fcd0f1cf4d5d`.
  The 32-position RAX SHA256 is
  `760362e26106daf0361cb4cda9d5cf38d0f740bbfce20ff6c539b2fef1dbc9f4`.
- Both packages were run through the same `buddy-cli`, with the same affinity,
  environment, and audio. Weight loading and RAX extraction are excluded from
  the stage timers.

## Results

| Graph | First forward, alternating processes | Complete core forwards | Complete transcription | RTF |
| --- | ---: | ---: | ---: | ---: |
| 448 positions, FP16 | 41.407, 41.448 s | 782.922, 803.724 s (earlier run) | 793.907, 815.140 s (earlier run) | 164.88, 169.29 |
| 32 positions, FP16 | 36.074, 36.118 s | 673.055, 671.620 s | 679.645, 678.098 s | 141.15, 140.83 |

The same-day first-forward A/B/A/B comparison shows a **1.148×** speedup,
or about 5.33 seconds saved per forward. The complete 32-position runs each
made 19 forwards, generated EOS, and produced the exact reference text:

> Nor is Mr. Quilter's manner less interesting than his matter.

Their end-to-end times differ by 0.23%. Relative to the earlier 448-position
complete runs, the observed improvement is about 1.17–1.20×; those runs were
on a different day, so the alternating first-forward pair is the controlled
speedup estimate. During the second 32-position run, `VmHWM` stabilized at
3,291,092 kB (about 3.14 GiB). The earlier 448-position run reported about
4.00 GiB; this memory comparison is also across runs on different days.

## Remaining bottleneck

The 32-position graph still spends about 36 seconds on each full forward.
Every generated token still recomputes the 1500-position encoder. The
same-day A/B result establishes that reducing the fixed decoder and LM-head
extent removes only about 13% of first-forward wall time under these
conditions. The next structural experiment should export encoder output as a
separate graph result, reuse it across transcription steps, and measure the
encoder and decoder with distinct runtime timers. This experiment does not
separate those stages and makes no claim about their individual wall times.
