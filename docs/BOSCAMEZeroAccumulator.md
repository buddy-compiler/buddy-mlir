# BOSCAME zero accumulators

`bosc_ame.tile.zero_acc : vector<MxNxi32>` produces an all-zero accumulator
with no memory operand and no input accumulator. It accepts fixed shapes with
`1 <= M,N <= 64`, matching `tile.mma` and `tile.store_acc`. At the tile SSA
level it is a pure value: no zero buffer, old C read, or old-content dependency
is introduced. Choosing it requires a proven zero initialization at the start
of the corresponding reduction. An existing `C += A*B` still starts with
`tile.load_acc`; later K blocks must preserve the accumulated result.

The NR lowering creates one side-effecting assembly bundle. It programs M/N
and `mtype=0x21` (16-bit widening mode), emits
`mwsub.h.mm Acc0,Acc0,Acc0`, and fences the bundle. Its independent output
constraint names the allocated accumulator without supplying an old value.
Every following MMA programs its own M/N/K and the existing signed-i8
compiler mode `mtype=0x10010`, while stores program the raw i32 accumulator
memory mode. No matrix register is kept live
across a DMA/runtime call. The SPM/DMA checker treats `zero_acc` as a new
accumulator lifetime and applies the same call/DMA restrictions as a load.

The production encoder and ISA validator accept only the two self-clear
forms `mwsub.mm` and `mwsub.h.mm`, with all three register fields equal and
in Acc0–Acc7. Generic `mwsub.mm` requires a 16-bit mtype. Unequal sources,
different destinations, other widths, and general accumulator subtraction
remain rejected. The assembly restriction pass and linked-ELF audit require
an adjacent `fence rw,rw` on both sides of every AME instruction.

The hardware evidence is the standalone
[clear_acc probe](../reference/examples/clear_acc/README.md), including its
[board validation](../reference/examples/clear_acc/VALIDATION.md). That probe
validated both forms for Acc0–Acc7 and 1×1, 2×3, and 8×8, including polluted
seeds, repeated clears, clearing after MMA, subsequent MMA, inactive elements,
padding, and guards. Its separate no-clear image fails as expected. This
compiler change does not extend those board measurements to larger tiles.

`tests/Conversion/BOSCAME/ame-zero-acc.py` checks the production encoder,
rejects unequal register fields, checks fence insertion, and executes exact
tile semantics with poisoned SPM and guarded output. It additionally covers
4×40, 32×64, and 64×64 by simulation. Dialect, lowering/LLVM/assembly, and
DMA/call lifetime regressions live beside that test. The production linked
ELF auditor also accepts the retained board-tested `clear_acc.elf` through
the same restricted instruction validator used for generated examples.

The planner records a reciprocal link between the private zero fill and its
matmul: `plan.fill` uses `engine = "ame", kind = "deferred-zero"`, and
`plan.route.initialization` identifies the fill task and uses `kind = "zero"`
or `"first-iteration-zero"`. The verifier and emitter independently re-prove
that the i32 fill is exactly zero, covers the complete private accumulator,
and has no observer before the product. The loop form additionally requires
a statically nonempty positive-step `scf.for` containing the matmul directly.
Only its first iteration clears the accumulator; subsequent iterations load
the stored partial sum. Each branch owns a complete clear/load, MMA, and store
lifetime, so no register value crosses the branch or the next DMA call.
Nonzero initial contents, a prior observer through any alias, incomplete
task coverage, and zero-trip loops retain the ordinary memory initialization.

`tests/Conversion/LowerLinalgToBOSCAME/PlanZeroAccumulator.py` runs the real
planner, both verification gates, lowering, and the strict simulator. It
checks a complete product, a K loop with a nonzero lower bound, a padded 4×40
tail, dirty SPM and repeated calls, signed i8 extremes, i32 wraparound for an
existing nonzero accumulator, input/output guards, and rejection of malformed
initialization/task links. The product reads no external C and writes each
valid output byte once.
