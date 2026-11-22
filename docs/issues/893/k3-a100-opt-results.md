# Whisper Base K3 A100: end-to-end optimization results

Building on the cached-encoder, static-KV-decoder baseline (`bb817bed`), this
series optimizes all three runtime stages of the Whisper Base FP16 package on
SpacemiT K3 A100 cores (encoder, KV decoder, and DAP audio preprocessing):

- **Encoder IME & attention lowering**:
  - Fuse `linalg.transpose` + `linalg.matmul` into `linalg.matmul_transpose_b`
    and lower `matmul_transpose_b`, encoder `Q @ K^T` (`batch_matmul_transpose_b`),
    and `conv_1d_ncw_fcw` (via `im2col`) to A100 FP16 IME kernels with contiguous
    `vector<8xf16>` B-packing and vectorized FP32-accumulating tails.
  - Relax `IMEGemmF16Lowering` from `N % 32 == 0` to `N >= 32` so 8x32 tiles
    handle arbitrary row strides such as `N = 1500` in `Q @ K^T`.
  - Lower encoder `P @ V` (`batch_matmul`, `8x1500x1500 @ 8x1500x64`) to a
    4-row M-unrolled FP32-accumulating RVV kernel, avoiding FP16 accumulation
    drift after switching softmax to FP32.
- **Decoder KV vectorization**:
  - Extend `-matmul-transpose-b-vectorization-decode` to FP16 with FP32
    accumulation and 1D `scf.parallel` for static `M == 1` GEMV, and add
    head-parallel RVV FP32-accumulating `batch_matmul` lowering for `M == 1`
    decode self/cross-attention.
  - Disable unsafe `affine-loop-fusion` on `kv_init` and `kv_step` (the same
    `reinterpret_cast` alias issue fixed for the encoder in `bb817bed`).
- **Fused softmax, GELU, and vector `exp`/`erf`**:
  - Add `-fuse-attention-softmax` to fold zero-tensor adds, fuse `log_softmax` +
    `exp` (including preceding FP16 clamp), and fuse 7-op `f16 -> f32 -> erf -> f16`
    GELU sequences.
  - Add `-lower-vector-softmax-erf` and extend `-approximate-vector-exp` with
    vectorized `math.erf` polynomial approximation so softmax and GELU run
    entirely on RVV without scalar `expf`/`erff` calls.
- **Audio preprocessing (`ExtendDAPPass`)**:
  - Hoist `make_rfftp_plan(400)` out of the 3,001-frame STFT loop and free all
    plan/scratch allocations (fixing ~1 GB per-call heap leak), reuse a stack
    frame buffer, skip silence frames, and vectorize the `80x201 @ 201x3001`
    mel filterbank matmul with `vector<16xf64>` FMA + `-O3` codegen.

## K3-002 measurements

Whisper Base FP16; K3-002 A100 CPUs 8–15; eight OpenMP threads; AI-thread
registration; 1024-bit VLEN; LLVM pin `3845a205`. Final RAX SHA256:
`1dd0f6edd7c70f1c36967f43ce07c5851a18cdc154b548f937980a5eb4d87801`.

| Package (`audio.wav`, 4.815 s) | Preprocess | Encoder | Decoder | Core total | End to end | RTF | Output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline (`bb817bed`, all-IME KV) | 2,070 ms | 35,040 ms | 7,910 ms | 42,950 ms | 45,060 ms | 9.36 | Reference match |
| Optimized (`perf/whisper-k3-a100`) | 101 ms | 1,622 ms | 272 ms | 1,894 ms | 2,031 ms | 0.42 | Reference match |
| Speedup | 20.5x | 21.6x | 29.1x | 22.7x | 22.2x | — | — |

## Validation scope

- Verified exact reference transcription on `audio.wav` (two runs:
  `Nor is Mr. Quilter's manner less interesting than his matter.`) and
  `audio1s.wav` (`Nor is`).
- These complete ASR checks do not establish corpus-level WER or bitwise logits
  equivalence, and baseline versus optimized runs were measured separately
  rather than alternating A/B.
