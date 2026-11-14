# Cached Whisper encoder: 512x512 IME numeric investigation

The earlier selectable A100 IME encoder path in
`encoder-ime-selective-results.md` kept 512x512 GEMMs on their existing path.
This note records how the full-IME error was narrowed down and resolved for
the cached KV graph.

On K3-002, a temporary diagnostic graph exported the six Whisper Base encoder
block outputs and final layer norm. Its final PyTorch eager output was bitwise
identical to the ordinary encoder wrapper, and its packed encoder weights had
the same SHA256. Inputs were zero audio features on A100 CPUs 8–15. The table
reports mean absolute error against the no-IME graph at each returned stage.

| IME selection | First block | Final layer norm | Result |
| --- | ---: | ---: | --- |
| None, but IME layout pass enabled | 0 | 0 | Bitwise identical |
| 512x512 only | 0.00305 | 0.02203 | First ASR token unchanged |
| 512x2048 and 2048x512 | 0.00123 | 0.01231 | Full ASR text matches reference |
| All three shapes, affine fusion enabled | 0.42831 | 0.87889 | First ASR token changes to `<|vi|>` |
| All three shapes, encoder affine fusion disabled | 0.00316 | 0.02326 | Full ASR text matches reference |

The all-shapes error is present in the first encoder block. A second graph
exposed that block's pre-attention layer norm, attention residual, and FFN
intermediates. The pre-attention layer norm's first element was about -0.200
in the no-IME, 512x512-only, and FFN-only variants, matching PyTorch. With
all three shapes enabled it became about +0.078, even though that value is
computed before the block's GEMMs. The subsequent affine loop fusion is the
pass-level cause, as shown by the IR and controlled compilation below.

A graph containing only the first encoder block reproduced the same +0.078
value, excluding later blocks as the cause. Its 512x512-only and FFN-only
variants retained the expected -0.200 value. The error therefore requires
the two IME groups to coexist inside the first block.

Controlled checks excluded several simpler explanations: the layout pass
without selected IME GEMMs was bitwise identical; all-shapes IME still failed
with one OpenMP thread; removing automatic buffer deallocation, later batch
matmul vectorization, or later ordinary matmul vectorization did not restore
the pre-attention value. A diagnostic override that zeroed the IME GEMM
accumulators also left it unchanged. Extending both packed A and packed B
scratch allocations by 4096 FP16 elements at the tail left the first-block
error unchanged. The diagnostic programs and binaries were kept outside the
source patch.

The faulty transformation is visible in the generated IR. Before fusion,
the first convolution's padding buffer is zero-filled, then its interior is
filled from audio features, and the convolution reads it. After fusion, an
additional zero store to that same buffer appears inside the convolution's
reduction loop immediately before its load. The same pattern occurs at the
second convolution. Skipping `affine-loop-fusion` only for the cached KV
encoder preserves the audio input and lets all three IME shape groups run
correctly. The other Whisper graphs retain their existing pipeline. See
`encoder-ime-affine-fusion-fix.md` for the integrated K3 result.
