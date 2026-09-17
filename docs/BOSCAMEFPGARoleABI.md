# FPGA tile/accumulator role ABI: design proposal

Status: **owner selected role-level separation**; general role types remain
to be implemented. The current W8A8 adapter carries explicit load roles and
slots, with the carrier-width heuristic confined to `+xboscame-fpga`.
Related: `docs/BOSCAMEFPGAValueSemantics.md` (current prototype status),
`.plan/migration.md` (Q2 / uncertainty 1).

## 1. Problem

The RISC-V backend has two matrix register files, `TileReg` (`tr0..tr7`) and
`AccReg` (`acc0..acc7`). TableGen constrains every AME operand to the right
file, but a value that carries **no instruction constraint of its own** - the
loop-carried accumulator PHI, and the `COPY`s that PHI elimination inserts -
has only its LLVM type to go on. Today one type therefore maps to one file:

```
prototype (+xboscame-fpga):  i32 -> AccReg,  everything else -> TileReg
```

This is a heuristic for one datapath (8-bit A/B tiles, 32-bit accumulator). It
breaks as soon as an element type is used in both roles (an `i32` tile, or an
`f32`/`f16` accumulator) and it is currently contained by
`verifyFpgaAmeCapabilities`, which rejects anything outside the W8A8 datapath.

## 2. Constraint that decides the mechanism

The role cannot be recovered after instruction selection:

* `MachineRegisterInfo::constrainRegClass` only narrows to a **subclass**,
* `MachineRegisterInfo::setRegClass` only accepts a **superclass** of the
  current class,
* `TileReg` and `AccReg` are unrelated classes (disjoint register lists).

So a virtual register can never be moved between the two files once created, and
"seed the role from the instruction constraints and propagate it along PHI/COPY
edges in a pre-RA pass" is **not implementable**. The role has to exist where the
value is created: in the IR type, or in a role-carrying intrinsic/SDNode.

A second constraint: LLVM's `MVT` enum is fixed at build time; a target cannot
add value types. Options are therefore (a) an in-tree LLVM change, (b) target
extension types, or (c) keep one type and encode the role in the operation.

## 3. Options

### A. Role-carrying MLIR type, lowered to a target extension type

* MLIR: the BOSCAME dialect distinguishes the roles in its tile ABI, e.g.
  `!bosc_ame.tile<4x4xi8>` / `!bosc_ame.acc<4x4xi32>` (or an attribute on the
  existing matrix type). The FPGA adaptation layer already knows the role of
  every value it creates, so this is a local change to the emitter.
* LLVM dialect: the role lowers to `!llvm.target<"riscv.ame.tile", ...>` /
  `!llvm.target<"riscv.ame.acc", ...>` (MLIR has `LLVM::TargetExtType`).
* Backend: `RISCVTargetLowering` maps the extension types to `TileReg` /
  `AccReg` (register type and type-legalization hooks), and the intrinsics keep
  their current operand classes.
* Cost: touches the MLIR type definitions, the emitters, the type converter, the
  LLVM dialect translation and ISel. Highest effort, but it removes the
  heuristic completely and is the only variant that can be argued upstream.

### B. Role in the intrinsic, distinct SDNode results

Give accumulator-producing intrinsics a distinct result node (for example a
separate `Intrinsic` with a `SDNode` property, or a target-specific node) so the
*producer* is unambiguous. This is not sufficient on its own: the PHI result is
still typed like a tile, so the role is lost exactly where it matters. It only
works combined with a type distinction, i.e. with A.

### C. In-tree LLVM MVT (upstream change)

Adding a matrix value type to `MVT` makes the role explicit at the IR level with
no extension-type plumbing, but the change must land upstream before it can be
used. Reasonable as a follow-up once A has shown the shape of the requirement.

### D. Containment (implemented)

Keep one type, keep the width heuristic, and make every unsupported combination
a compile error:

* FPGA convention gated behind `+xboscame-fpga`, default `+xboscame` unchanged,
* `verifyFpgaAmeCapabilities` rejects matrix-carrying AME operations outside the
  W8A8 datapath,
* matrix register spills are a fatal diagnostic (no lossless spill exists),
* the profile travels from MLIR to the backend as a target feature.

This is what the repository does today. It is safe but it does not generalise:
any new datapath (f32 tiles, i8 accumulators, mixed schedules) needs A.

## 4. Implementation direction

1. Keep **D** as temporary containment for W8A8. The adapter fixes A to
   `tr0/tr2`, B to `tr4..tr7`, and accumulators to `acc0..acc7`, validates
   liveness, and emits no matrix moves or spills. The `mmve` questions do not
   block this instruction subset and do not change the owner's role decision.
2. Implement **A**, MLIR side first: introduce the role in the BOSCAME tile
   ABI, keep the emitters as the single place that decides it, and lower it to
   target extension types. Do it on a branch with the current tests as the
   regression net, and keep the capability verifier from D in place while the
   new path is incomplete.
3. Track the MLIR-side ABI separately from the backend: the dialect type change
   is reviewable on its own and is also what an upstream submission would need.

## 5. Verification plan for A

* Dialect: the role types round-trip through the parser/printer, and the
  existing FPGA lit tests keep passing with the role-carrying types.
* Export: the LLVM dialect carries the extension types, and `buddy-translate`
  emits them into LLVM IR (`%v = call ... !llvm.target<...>`).
* Backend: an IR-level test per role that checks the selected register file in
  the assembly (`tr` / `acc` operands), plus the existing capacity and spill
  tests.
* A negative test that a tile-typed value can never be used as an accumulator
  operand and vice versa (verifier-level), which is the property the width
  heuristic cannot express today.

## 6. Implementation checklist for option A

Each step is independently committable; steps 1-4 keep the tree buildable and the
FPGA path working through the existing (heuristic) convention until step 5
switches it over.  Rollback is "revert this step" in every case, and the
regression net is the suite that exists today
(`scripts/check-fpga-migration.sh`).

**Step 0 - freeze the net.**  No code change.  Record the current results
(23 + 279 tests, assembly smoke) as the baseline for the migration.

**Step 1 - role types in the dialect.**  `midend/include/Dialect/BOSCAME/BOSCAME.td`:
add the two role types (or the role attribute on the existing matrix type) plus a
verifier that refuses to mix them; keep the old spelling accepted during the
transition.  Verify: a dialect round-trip test (parse/print) and a negative test
for a tile used as an accumulator operand.  Risk: ODS churn; keep the change
additive.

**Step 2 - emitters produce roles.**  `AMEValueEmitter.{h,cpp}` and both FPGA
passes: the helpers return role-typed values and the ops declare role-typed
operands/results; the emitters are already the single place that knows the role.
Verify: the existing FPGA lit tests keep passing with role types, plus a per-role
dataflow test (accumulator chains stay in the accumulator role through
`iter_args`).  Risk: touch every call site; no semantic change.

**Step 3 - export maps roles.**  `LegalizeForLLVMExport.cpp`: lower the role
types to distinct LLVM dialect types (preferred: `LLVM::TargetExtType`, for
example `!llvm.target<"riscv.ame.acc", ...>`), keeping the existing intrinsics
and the `llvm.target_features` annotation.  Verify: an IR test asserting the
extension types, no `unrealized_conversion_cast`, and an unchanged target
feature.  Risk: type converter ordering (the SCF structural conversion must
carry the role types too).

**Step 4 - LLVM IR translation.**  `BOSCAMEToLLVMIRTranslation.cpp` /
`ModuleTranslation` path: emit the extension types.  Verify: `buddy-translate`
output shows them and `llc` still parses the module.

**Step 5 - backend maps roles to register files.**
`RISCVISelLowering.cpp` (register type / legalization for the extension types),
`RISCVInstrInfo.cpp` (copies and spill diagnosis for both files).  Then remove
the width heuristic from `RISCVISelLowering.cpp` and relax
`verifyFpgaAmeCapabilities` to the datapaths the RTL actually supports.  Verify:
`tests/Target/RISCV/*.ll` (capacity, spill) plus a new per-role asm test that
checks `tr*` / `acc*` operands; re-run the whole matrix. Risk: this step changes
codegen. Keep copies unsupported unless their ISA semantics are confirmed.

**Step 6 - cleanup and documentation.**  Drop the transition affordances, update
`docs/BOSCAMEFPGAValueSemantics.md` (the prototype section shrinks), and refresh
the verification matrix.  Verify: `scripts/check-fpga-migration.sh` and the
GEM5/RTL acceptance once those inputs exist.

## 7. Owner decision and current implementation

The owner selected role-level separation as the final direction, with the
width heuristic temporarily allowed only behind `+xboscame-fpga`. No new
approval of that direction is pending.

The 2026-09-15 fixes add role-specific FPGA load/MMA/final-store intrinsics and
explicit physical load slots. A dedicated pre-RA allocator propagates those
slots through the W8A8 SSA graph and verifies liveness without implementing
copies or spills. The MVT carrier mapping remains W8A8-only; this does not claim
to implement the general role types in option A or to enable i32 input tiles.
