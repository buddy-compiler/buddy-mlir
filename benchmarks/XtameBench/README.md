# XtameBench

Instruction-count benchmark for the AME matrix-multiply family. Every file in
`ops/` holds one matrix kernel on `memref` operands, following the one-kernel-
per-file layout of `benchmarks/LinalgBench`.

The driver builds three programs from each case plus a generated `@main`
harness:

* `xtame` — lowered by `-lower-linalg-to-xtame`, so the kernel becomes
  `th.mcfg*`, `th.mlde*`, `th.mmacc.w.b` and `th.mste32`;
* `scalar` — the same kernel lowered to loops, i.e. the product without the AME
  extension;
* `overhead` — the same harness without calling the kernel, so the fill, zero
  and dump cost that the other two pay can be subtracted.

The harness fills A and B with index-derived values, runs the kernel `--reps`
times inside a region timed with `rdcycle`, and writes C, the cycle count and a
small keep-alive buffer to stdout. The keep-alive buffer makes the fills live in
the overhead program too; without it the compiler removes them and the
subtraction is inflated. The driver compares C with a Python product, then counts
the instructions each program executes under QEMU.

## What the numbers mean

Instructions, not time. QEMU has no timing model for the XuanTie core, so neither
the `rdcycle` value nor the wall clock of the QEMU process says anything about
hardware. The table compares how much work each version executes for the same
kernel; cycles are only meaningful when the binaries run on a board.

Measurement details:

* `total insns` comes from QEMU's `insn` plugin
  (`-d plugin -plugin .../libinsn.so`).
* `AME ops` counts AME instructions. QEMU's RISC-V disassembler does not know
  them and prints `illegal`, so the driver joins the `in_asm` disassembly with
  the per-block execution counts of the `exec` log. `-d nochain` is required,
  otherwise only the first block of a chained run is logged. As a check, the
  block sum is compared with the plugin total and a `note:` is printed when they
  disagree.
* The kernel is inlined into the harness (`-inline -symbol-dce`) so the numbers
  exclude call and prologue overhead.

## Cases

| case | operation | state |
| --- | --- | --- |
| `matmul_4x1x1` | `linalg.matmul`, the smallest accepted shape: K and N are both a tail | measured |
| `matmul_4x4x4` | `linalg.matmul`, one register row of K | measured |
| `matmul_5x3x2` | `linalg.matmul`, M, K and N are all remainders | measured |
| `matmul_8x20x8` | `linalg.matmul`, K = 16 + 4 chunked, two column blocks | measured |
| `matmul_16x32x16` | `linalg.matmul`, four column blocks | measured |
| `matmul_16x256x16` | `linalg.matmul`, eight 16-element K chunks per column block | measured |
| `matmul_20x20x20` | `linalg.matmul`, interchange path | measured |
| `matmul_33x20x13` | `linalg.matmul`, row, column and K remainders | measured |
| `matmul_32x64x32` | `linalg.matmul`, too many column blocks to interchange | measured |
| `batch_matmul_4x8x16x8` | `linalg.batch_matmul`, batch loop over slices | measured |
| `quantized_matmul_16x32x16` | `linalg.quantized_matmul`, matmul plus ones-vector sums and a scalar update | measured |
| `quantized_matmul_16x256x16` | the same at `K = 256`, where the `O(K)` part of the correction shows up | measured |

A case the pass cannot lower is reported as `no-xtame` and skipped, the same way
`benchmarks/LinalgBench` reports `no-vir`.

## Requirements

* Buddy-MLIR built (`build/bin/buddy-opt`, `buddy-translate`, `buddy-llc`) and
  the LLVM tools (`llvm/build/bin/llvm-mc`).
* `ld.lld`.
* A `qemu-riscv64` that implements the XuanTie matrix extension and is built with
  plugins (`--enable-plugins`, so that `tests/plugin/libinsn.so` exists). The
  stock distribution QEMU stops with SIGILL at `th.mcfgmi`.

## Run

```bash
QEMU_RISCV64=/path/to/qemu-riscv64 python3 \
    benchmarks/XtameBench/run_xtame_bench.py
```

Selected cases and a CSV:

```bash
python3 benchmarks/XtameBench/run_xtame_bench.py matmul_20x20x20 \
    --csv results.csv
```

Variables: `QEMU_RISCV64`, `QEMU_CPU` (default `xt-c9501fdvmt`),
`QEMU_PLUGIN_DIR` (default: the `tests/plugin` next to the QEMU binary).
Intermediate files go to `build/benchmarks/XtameBench/`.

## On hardware

Without `--qemu` the driver runs the binaries directly, so it needs a board or an
emulator that can execute them natively:

```bash
python3 benchmarks/XtameBench/run_xtame_bench.py --reps 1000 \
    --csv board.csv
```

The program times the `reps` loop with `rdcycle`, so `cycles/reps` is the cost of
one kernel plus the C zeroing. Use `--reps 1000` or more, run each case several
times and report the spread; a difference inside the spread is not a result.

## Reference result

QEMU `xt-c9501fdvmt`, `--reps 1`, one run per case, 2026-11-08:

| case | correct | AME ops | AME matmul | scalar matmul | work ratio |
| --- | --- | --- | --- | --- | --- |
| matmul_4x1x1 | yes | 10 | 16 | 104 | 6.5x |
| matmul_4x4x4 | yes | 10 | 16 | 889 | 55.6x |
| matmul_5x3x2 | yes | 10 | 17 | 471 | 27.7x |
| matmul_8x20x8 | yes | 22 | 34 | 14730 | 433.2x |
| matmul_16x32x16 | yes | 37 | 68 | 92554 | 1361.1x |
| matmul_16x256x16 | yes | 163 | 306 | 723338 | 2363.8x |
| matmul_20x20x20 | yes | 91 | 120 | 91809 | 765.1x |
| matmul_33x20x13 | yes | 112 | 142 | 98517 | 693.8x |
| matmul_32x64x32 | yes | 275 | 783 | 730378 | 932.8x |
| batch_matmul_4x8x16x8 | yes | 72 | 170 | 47983 | 282.3x |
| quantized_matmul_16x32x16 | yes | 87 | 4066 | 108938 | 26.8x |
| quantized_matmul_16x256x16 | yes | 381 | 4902 | 854410 | 174.3x |

### Column definitions

| column | definition |
| --- | --- |
| `case` | File stem in `ops/`. The file holds the kernel and fixes its operand shapes. |
| `status` | `ok` when the pass lowered the kernel and the program ran. `no-xtame` when the pass or the harness rejected the case; the reason is printed on the same line and the case is skipped. |
| `correct` | Element-by-element comparison of the C matrix the program dumped with the product computed in Python from the same fill formula. `yes` means every element matched; `WRONG` means at least one differed, and the driver then exits non-zero. |
| `AME ops` | AME instructions the program executed: `th.mcfgmi`, `th.mcfgni`, `th.mcfgki`, `th.mlde8`, `th.mlde32`, `th.mldte8`, `th.mmacc.w.b` and `th.mste32`, summed over the run. Counted per translation block and multiplied by the number of times that block executed. |
| `AME matmul` | Executed instructions of the whole `xtame` program minus those of the `overhead` program, which runs the same harness without calling the kernel. What remains is the work of the kernel: the AME instructions plus the address arithmetic that feeds them. |
| `scalar matmul` | The same subtraction for the `scalar` program, where the kernel is lowered to loops instead of AME instructions. |
| `work ratio` | `scalar matmul` divided by `AME matmul`. It compares instruction counts, not time. |
| `cycles` | Printed only without `--qemu`, i.e. for a native run: the `rdcycle` delta around the `--reps` loop, so `cycles/reps` is one kernel plus the C zeroing. |

`AME matmul` and `scalar matmul` are executed instructions with the shared
overhead subtracted. The ratio is not a speedup: one `th.mmacc.w.b` performs many
more multiply-accumulates than one scalar instruction, and its latency is unknown
here.

`matmul_20x20x20` executes 163 AME instructions while `matmul_16x32x16` executes
38 for more multiply-accumulates. K = 20 is not a multiple of 16, so the pass
chunks the reduction into five 4-element steps instead of one 16-element step
plus a 4-element tail. This is a known inefficiency of the current chunk
selection, not a property of the hardware.

`quantized_matmul_16x32x16` shows the cost of the zero-point correction: 87 AME
instructions and 4066 instructions in total, against 68 for the plain matmul of
the same shape. The row sums of A and the column sums of B are matrix products
with a vector of ones that lives in a `memref.global` constant, so no fill runs
at run time; the rest is the per-element update of C.

`quantized_matmul_16x256x16` shows how that cost scales: 4902 instructions
against 306 for the plain matmul. The O(M*N) update is amortised as K grows, so
the ratio of the quantized path to the plain matmul falls from 60 times at
K = 32 to 16 times at K = 256. What remains, and what would need more work, is
listed under "Future work" in
[docs/XTAMELinalgLowering.md](../../docs/XTAMELinalgLowering.md).
