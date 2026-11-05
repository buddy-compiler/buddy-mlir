# Temporary scalar FP32 reduction workarounds (FPGA)

FPGA builds default to C++ lowering that expands selected FP32 vector
reductions to ordered scalar loops so the final NR ELF does not keep the
failing RVV reduction forms. This is a **software workaround**, not a proven
RTL root cause.

Two independent switches (FPGA `make` enables both by default):

| Pass option | Makefile restore | Rejected opcode |
| --- | --- | --- |
| `scalar-f32-ordered-reductions=true` | `ALLOW_VECTOR_ORDERED_REDUCTIONS=1` | `vfredosum.vs` |
| `scalar-f32-max-reductions=true` | `ALLOW_VECTOR_MAX_REDUCTIONS=1` | `vfredmax.vs` |

Implementation sketch: bounded function-entry scratch, volatile vector
snapshot, per-lane scalar loads, scalar SSA accumulator; other vector ops
stay. Assembly filters and `check_nr_elf.py` reject residual encodings unless
the matching `--allow-vector-*-reductions` diagnostic flag is set.

```bash
# Restore original vector reductions for hardware diagnosis (no clean needed
# when flipping stamps; LLVM and later stages rebuild).
make ALLOW_VECTOR_ORDERED_REDUCTIONS=1
make ALLOW_VECTOR_MAX_REDUCTIONS=1

buddy-opt kernel.boscame.mlir \
  --lower-bosc-ame="scalar-f32-ordered-reductions=true scalar-f32-max-reductions=true" \
  -o kernel.llvm.mlir
```

Supported shapes are fixed 1-D f32 reductions in sequential regions (ordered
ADD with explicit seed; max forms `maxnumf` / `maximumf` / `maximumnumf` with
or without seed). Masked, scalable, or concurrent/unknown regions error
explicitly. Non-FPGA callers keep pass defaults `false`.

Evidence for LayerNorm board acceptance and known reduction failures belongs
with the example / validation logs; do not treat “no `vfredosum` in the ELF”
alone as numeric proof.
