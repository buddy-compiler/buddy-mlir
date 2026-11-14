# Cached Whisper encoder: 512x512 IME numeric investigation

The selectable A100 IME encoder path in `encoder-ime-selective-results.md`
keeps 512x512 GEMMs on their existing path. This note records why the full
IME combination remains disabled and where to continue the investigation.

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
| All three shapes | 0.42831 | 0.87889 | First ASR token changes to `<|vi|>` |

The all-shapes error is present in the first encoder block. A second graph
exposed that block's pre-attention layer norm, attention residual, and FFN
intermediates. The pre-attention layer norm's first element was about -0.200
in the no-IME, 512x512-only, and FFN-only variants, matching PyTorch. With
all three shapes enabled it became about +0.078, even though that value is
computed before the block's GEMMs. This suggests a graph-local storage or
lowering interaction; it does not by itself prove which buffer or instruction
is at fault.

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

The next lower-level check is an IME scratch/alias bounds probe around the
first block's packed A/B and output buffers. Continue using the selective
shape configuration for production evaluation until that check passes. The
full-IME variant is slower for this encoder as well as incorrect.
