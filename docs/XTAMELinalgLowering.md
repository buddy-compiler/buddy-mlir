# Linalg-to-XTAME Matmul Pass: Implementation and Validation Record

Status recorded on 2026-11-08. This document describes the current prototype,
the shapes it accepts, the code it generates, and the validation performed. All
runtime results come from the Xuantie QEMU user-mode model; no physical XTAME
hardware was available, so there is no hardware performance figure here.

The pass implementation is in
[`LowerLinalgToXTAME.cpp`](../midend/lib/Conversion/LowerLinalgToXTAME/LowerLinalgToXTAME.cpp).
The XTAME operation set and its LLVM export are in
[`XTAME.td`](../midend/include/Dialect/XTAME/XTAME.td) and
[`LegalizeForLLVMExport.cpp`](../midend/lib/Dialect/XTAME/Transforms/LegalizeForLLVMExport.cpp).

## Supported inputs

The pass `-lower-linalg-to-xtame` matches three buffer-semantics operations whose
operands are static, contiguous, identity-layout memrefs:

| Operation | Operands | Requirement |
| --- | --- | --- |
| `linalg.matmul` | `memref<MxKxi8> x memref<KxNxi8> -> memref<MxNxi32>` | `M >= 4`, `N >= 1`, `K >= 1` |
| `linalg.batch_matmul` | `memref<BxMxKxi8> x memref<BxKxNxi8> -> memref<BxMxNxi32>` | the same limits per batch; `B >= 1` |
| `linalg.quantized_matmul` | the matmul operands plus two `i32` zero points | the same limits |

`linalg.batch_matmul` walks the batch dimension with an `scf.for` and lowers each
iteration as one matmul on a rank-reduced subview, so a batch slice costs the
same sequence as a plain matmul.

`linalg.quantized_matmul` accumulates `(A - AZp) * (B - BZp)`. The matmul part
produces `sum_k A*B`; a correction then adds

```text
-AZp * sum_k B - BZp * sum_k A + K * AZp * BZp
```

per output element. Both sums are matrix products with a vector of ones, so they
run through the same AME sequence: `A` times a `Kx1` column of ones gives the row
sums, and a `4xK` block of ones times `B` gives the column sums in its first row
(a tile needs at least four rows). The two matrices of ones are private
`memref.global` constants of the module (`@xtame_ones_<rows>_<cols>`,
`alignment = 16`), one symbol per distinct shape, so the correction does not
build them at run time. Only the final update of `C` is scalar, one
subtract-and-add per output element.

The correction keeps three `i32` temporaries on the stack, `4*M + 20*N` bytes in
total, so the pass rejects a shape above `--max-temporary-bytes` (default 16384)
with a diagnostic that names the requirement instead of emitting a program that
overflows the stack at run time.

### Access footprint of a tile

Every AME load and store is configured with the byte width of the subview it
touches: `tilek` is the chunk width for `mlde8` and `mldte8`, and `4 * cols` for
the i32 tile. Nothing is rounded up to four columns or sixteen rows, so a tail
tile reads and writes exactly its subview: for `M = 4, K = 1, N = 1` the loads
use `mcfgki 1` and the store `mcfgki 4`, and for `5x3x2` the reduction tile is
3 bytes wide and the C block 8. `linalg-to-xtame-matmul.mlir` pins those widths
and the benchmark runs both shapes under QEMU against a Python product, so a
tail tile that touched a neighbour would show up as a wrong element.

Every other shape, element type or semantics is rejected, and the rejection is
checked by
[`linalg-to-xtame-unsupported.mlir`](../tests/Conversion/linalg-to-xtame-unsupported.mlir).

The blocking rules below follow from measurements on the QEMU model, not from a
hardware specification:

| Constraint | Value used by the pass | Observation |
| --- | --- | --- |
| Minimum tile height | 4 rows | `mtilem` of 2 or 3 produces incorrect elements; 4 and above are correct |
| Maximum tile height | 16 rows | `mtilem` of 20 or above raises `SIGILL` |
| Output columns per tile | 4 | `mcfgni 8` writes only the first four columns; the i32 accumulator row is one RLEN |
| `tilek` | byte width, at most 16 | `tilek = K` is correct for `K <= 16`; `tilek = 8` with `K = 16` accumulates only half; `tilek = 32` gives wrong values |

`tilek` is the byte width of the innermost dimension of the current operation,
not an element count: one i32 output element is 4 bytes, and an i8 reduction
tile of `K` elements is `K` bytes.

## Generated structure

For each row tile of 4 to 16 rows (a greedy partition of `M` that never leaves a
tile below 4 rows):

```text
subview A[row0, 0] [rows, K]                     # only when the tile is not all of A
mcfgmi rows
```

then, for each block of at most four output columns:

```text
mcfgki 4 * cols        ; C row width in bytes
mlde32 acc, cStride, cTile        ; preserve the initial C tile
<reduction over K>
mcfgki 4 * cols
mste32 acc, cStride, cTile
```

Reduction over `K`:

- `K <= 16`: one `mcfgki K` followed by `mlde8` (A), `mldte8` (B) and one
  `mmacc`. When there is more than one column block, A is loaded once before the
  column loop, because A does not depend on the columns and `mmacc` does not
  write `ms1`.
- `K > 16`: a loop over chunks, followed by a peeled chunk of `K % 4` elements
  when non-zero. The chunk is 16 elements when `(K - K % 4) % 16 == 0` and 4
  elements otherwise.

When the reduction needs more than one chunk and the output has at most six
column blocks, the emitter interchanges the loops:

```text
mlde32 m0, ...   ; block 0 accumulator
mlde32 m3, ...   ; block 1 accumulator   (m1 = A, m2 = B are reserved)
scf.for k in chunks:
    mlde8 m1, aStride, A[k]
    for each block b:
        mldte8 m2, bStride, B[k, block b]
        mmacc acc[b], m2, m1
mste32 m0, ...
mste32 m3, ...
```

This loads A once per chunk instead of once per chunk and column block. When
there are more than six column blocks the registers are exhausted, and each
column block keeps its own K loop.

Strides are derived from the operand shapes: A rows are `K` bytes apart, B rows
are `N` bytes apart, and C rows are `4 * N` bytes apart.

### Pointer lowering for subviews

`xt_ame` load and store operations take a memref plus a byte stride. The LLVM
export derives the address as the memref's aligned pointer plus the descriptor
offset, so a `memref.subview` addresses the tile it denotes instead of the start
of the base buffer. Before this fix both iterations of the reduction loop read
the same tile, which the K=8 runtime check exposed.
[`xtame-memref-subview.mlir`](../tests/Conversion/xtame-memref-subview.mlir)
checks that the offset operand of `memref.extract_strided_metadata` reaches the
`llvm.getelementptr`.

## Build and register the pass

Configure Buddy-MLIR as described in the repository build instructions, then
build the pass and `buddy-opt`:

```bash
ninja -C build LowerLinalgToXTAMEPass -v
ninja -C build buddy-opt -j8
build/bin/buddy-opt --help | grep lower-linalg-to-xtame
```

The help output contains `--lower-linalg-to-xtame`.

### Lower, translate and assemble by hand

The example makefile runs this sequence; the same commands work on any file that
satisfies the supported inputs:

```bash
set -o pipefail
build/bin/buddy-opt input.mlir \
  -lower-linalg-to-xtame \
  -lower-xt-ame \
  -convert-linalg-to-loops \
  -lower-affine \
  -convert-scf-to-cf \
  -convert-cf-to-llvm \
  -expand-strided-metadata \
  -lower-affine \
  -convert-arith-to-llvm \
  -convert-math-to-llvm \
  -convert-func-to-llvm \
  -finalize-memref-to-llvm \
  -reconcile-unrealized-casts \
  | build/bin/buddy-translate --buddy-to-llvmir \
  | build/bin/buddy-llc -filetype=asm \
      -mtriple=riscv64-unknown-linux-gnu \
      -mattr=+m,+a,+c,+d,+xtheadame \
      -o input.s
```

The second `-lower-affine` is required: `-expand-strided-metadata` turns the
descriptor offsets of the tile subviews into affine index arithmetic. Without
`-expand-strided-metadata` the `memref.subview` operations survive and
`buddy-translate` fails because it does not register the memref dialect.

## Tests

| Test | Covers |
| --- | --- |
| [`linalg-to-xtame-matmul.mlir`](../tests/Conversion/linalg-to-xtame-matmul.mlir) | IR for M=4/8/20, N=4/5/8, K=4/6/8/12/20; a second `RUN` line checks the assembly through `buddy-translate` and `buddy-llc` |
| [`linalg-to-xtame-unsupported.mlir`](../tests/Conversion/linalg-to-xtame-unsupported.mlir) | `M = 2` must fail legalisation instead of emitting a kernel |
| [`xtame-memref-subview.mlir`](../tests/Conversion/xtame-memref-subview.mlir) | Subview addressing: the descriptor offset must reach the address computation |
| [`xtame-register-asm.mlir`](../tests/Conversion/xtame-register-asm.mlir) | Dialect parsing, fixed-register lowering and assembly |
| [`xtame-ssa-intrinsics.mlir`](../tests/Conversion/xtame-ssa-intrinsics.mlir) | The SSA intrinsic interface |
| [`linalg-matmul.mlir`](../examples/XTAMEDialect/linalg-matmul.mlir) | End-to-end example: initialization, matmul, store, element-wise self-check, IR and assembly |

```bash
ninja -C build check-buddy
```

## Reproducible runtime validation

[`examples/XTAMEDialect/linalg-matmul.mlir`](../examples/XTAMEDialect/linalg-matmul.mlir)
computes an 8x20 by 20x8 matmul, checks every output element against the same
expression that produced the inputs, and returns the number of mismatches, so a
correct run exits with status 0. The shape exercises a row tile (8 rows), two
column blocks (8 columns) and a chunked reduction (K = 20, therefore the
interchanged form).

The example directory provides the whole path:

```bash
cd examples/XTAMEDialect
QEMU_RISCV64=/path/to/qemu-riscv64 make linalg-matmul-run
# matmul self-check passed: all output elements matched
```

The target lowers the MLIR, translates to LLVM IR, emits assembly with
`buddy-llc`, assembles both the kernel and a start stub with `llvm-mc`, links
statically with `ld.lld -m elf64lriscv`, and runs the result under QEMU. The
start stub is generated by the makefile, so no hand-written assembly is checked
in.

`QEMU_RISCV64` defaults to `qemu-riscv64` from `PATH`; the system emulator
package of a typical distribution does not implement the XTAME instructions and
terminates with `SIGILL` at `th.mcfgmi`. Use the Xuantie build below.

### Build the Xuantie QEMU user-mode emulator

```bash
curl --http1.1 -fL \
  https://codeload.github.com/T-head-Semi/qemu/tar.gz/refs/heads/xuantie-qemu-9.0 \
  -o /tmp/xuantie-qemu-9.tar.gz
mkdir -p /tmp/xuantie-qemu-work
tar -xzf /tmp/xuantie-qemu-9.tar.gz \
  -C /tmp/xuantie-qemu-work --strip-components=1
mkdir -p /tmp/xuantie-qemu-build
cd /tmp/xuantie-qemu-build
../xuantie-qemu-work/configure \
  --target-list=riscv64-linux-user \
  --disable-system \
  --disable-docs
ninja -j8
```

On the host used for this work, compilation initially failed because this QEMU
revision defines `struct sched_attr` locally while the host glibc headers
already provide it. For the temporary QEMU build only, the duplicate struct
definition in `linux-user/syscall.c` was removed before rerunning Ninja; this
edit is not part of Buddy-MLIR.

The emulator used for all results below is version 8.2.94 with the
`xt-c9501fdvmt` CPU model:

```bash
/tmp/xuantie-qemu-build/qemu-riscv64 -cpu xt-c9501fdvmt ./linalg-matmul
```

## Measured shapes

Each entry is a self-checking program: a correct run exits with status 0. The
inputs vary by row and column, so a wrong tile base, row offset or accumulator
register cannot cancel out.

| M | K | N | Result |
| --- | --- | --- | --- |
| 4 | 4 | 4 | 0 |
| 4 | 8 | 4 | 0 |
| 4 | 8 | 5 | 0 |
| 4 | 6 | 6 | 0 |
| 4 | 8 | 8 | 0 |
| 8 | 8 | 8 | 0 |
| 8 | 20 | 8 | 0 |
| 8 | 20 | 16 | 0 |
| 4 | 22 | 7 | 0 |
| 16 | 20 | 13 | 0 |
| 20 | 12 | 4 | 0 |
| 17 | 10 | 4 | 0 |
| 33 | 20 | 8 | 0 |
| 20 | 20 | 7 | 0 |

Reductions with `K` from 4 to 64 were also checked with a uniform right-hand
side.

## Benchmark

`benchmarks/XtameBench` holds one kernel per file in `ops/` and builds each of
them twice, through this pass and as scalar loops, then counts the instructions
both programs execute under QEMU:

```bash
QEMU_RISCV64=/path/to/qemu-riscv64 \
    python3 benchmarks/XtameBench/run_xtame_bench.py
```

| case | AME instructions | matmul instructions, AME | matmul instructions, scalar |
| --- | --- | --- | --- |
| `matmul_4x1x1` | 10 | 16 | 104 |
| `matmul_4x4x4` | 10 | 16 | 889 |
| `matmul_5x3x2` | 10 | 17 | 471 |
| `matmul_8x20x8` | 22 | 34 | 14730 |
| `matmul_16x32x16` | 37 | 68 | 92554 |
| `matmul_16x256x16` | 163 | 306 | 723338 |
| `matmul_20x20x20` | 91 | 120 | 91809 |
| `matmul_33x20x13` | 112 | 142 | 98517 |
| `matmul_32x64x32` | 275 | 783 | 730378 |
| `batch_matmul_4x8x16x8` | 72 | 170 | 47983 |
| `quantized_matmul_16x32x16` | 87 | 4066 | 108938 |
| `quantized_matmul_16x256x16` | 381 | 4902 | 854410 |

The columns are the case file, the AME instructions it executes, and the
executed instructions of the whole program minus the fill, zero and dump cost
for the AME and for the scalar version. Both programs therefore pay the same
non-kernel cost, and the last two columns compare only the kernel.

`quantized_matmul_16x32x16` executes 87 AME instructions and 4066 instructions
in total, against 68 for the plain matmul of the same shape. The extra AME work
is the two ones-vector matmuls; the rest is the per-element update of `C`, which
is `O(M*N)`. The same case needed 11890 instructions when the sums were scalar
reductions, and 4745 when the ones were filled at run time.

`quantized_matmul_16x256x16` shows the scaling: 4902 instructions against 306 for
the plain matmul. The per-element part is amortised as `K` grows, so the ratio of
the quantized path to the plain matmul falls from 60 times at `K = 32` to 16
times at `K = 256`.

These are instruction counts, not times. QEMU has no timing model for the
XuanTie core, so the ratio says how much less work the AME version executes, not
how much faster it runs; one `th.mmacc.w.b` also performs many more
multiply-accumulates than one scalar instruction. The benchmark prints cycles
from `rdcycle`, which is meaningful only on a board.

The table also shows the effect of the chunk selection. Reductions use
16-element chunks while they fit and only fall back to 4-element chunks for the
remainder, so `K = 20` is 16 + 4: `matmul_20x20x20` dropped from 284 to 120
executed instructions and `matmul_33x20x13` from 369 to 142.

## Known limitations

- Only `i8 x i8 -> i32`. There is no `i16`, `i4` or floating-point path yet, and
  the operand types are matched exactly, so a non-identity layout is rejected
  rather than lowered.
- `linalg.batch_matmul` requires a contiguous batch-major buffer: the batch loop
  derives the row stride of every slice from `K`, so a non-contiguous batch
  dimension is rejected.
- The zero-point correction of `linalg.quantized_matmul` still updates every
  output element with scalar code, and its three stack temporaries cap the shapes
  the pass accepts (`--max-temporary-bytes`, 16384 by default).
- Buffer semantics only. A tensor-semantics matmul must be bufferized before
  this pass.
- The tail handling is validated by value and by the tile widths the pass emits,
  not by an out-of-bounds analysis: no verifier or address sanitizer checks that
  a tail tile stays inside its operand.
- All runtime results come from the QEMU model. Nothing here is a hardware
  measurement, and the `mtilem`/`tilek` limits above are properties of the model
  that the pass respects; they are not confirmed on silicon.

## Future work

Recorded, not scheduled.

1. **Fold the zero-point correction into the AME sequence.** The per-row constant
   `-BZp * sum_k A` and the per-column constant `-AZp * sum_k B` would have to be
   operands of a matrix product, and the tile only holds `i8` operands, so the
   final update of `C` stays scalar. Folding it needs an `i32`-operand
   multiply-accumulate, which the XTAME instruction set does not expose.
2. **Share the ones constants across functions.** One `memref.global` is created
   per distinct `(rows, cols)` in a module; two functions with the same `K` share
   it, but the symbol search is a linear walk over the module.
3. **Measurements on hardware.** Every number in this document is an instruction
   count from the QEMU model. Cycles, throughput and the `mtilem`/`tilek` limits
   themselves still need a board.
