# BOSC AME value semantics and the FPGA target

This document is the reference for the migration of the Qwen3 FPGA AME fast path
onto the value-semantics pathway that landed upstream in
[buddy-compiler/buddy-mlir#829](https://github.com/buddy-compiler/buddy-mlir/pull/829)
(`3c47e88`, "update pathway design for BOSC AME"). It records the two AME
software conventions that now coexist, how a module selects one of them, the
`mtype` encoding table, the invariants the FPGA schedule must keep, and the
parts of the migration that are still open.

Everything below is derived from source in this repository and from the
`upstream` remote; no hardware claim is made that is not traceable to a comment
or a commit in the tree.

## 1. Two conventions, one dialect

`BOSCAME_MatrixType` is `vector<4x4xT>` and the instruction set is shared, but
the *configuration* and the *accumulator* convention differ:

| Aspect | `upstream` (default) | `qwen3-fpga` |
| --- | --- | --- |
| Configuration op | `bosc_ame.msettypei <raw width>` (immediate) | `bosc_ame.msettype %csr` (register, bit-field) |
| Configuration value | `4 / 8 / 16 / 32 / 64` element width | `0x10010` (MMA) / `0x10042` (accumulator) |
| MMA datapath in use | `i32 x i32 -> i32`, `f16 -> f32`, `f32 -> f32`, ... | `i8 x i8 -> i32` (`mqma.b.mm`) |
| Accumulator memory side | same element type as the accumulator | `mlce32.m` / `msce32.m` pair: the load reads an fp32 buffer as the integer accumulator, the store converts i32 -> f32 |
| Tile shape contract | 4x4 SSA value, backend-defined capacity | 16x16 tile, K = 64, lanes 0..7 |
| Register placement | LLVM backend | explicit A/B/ACC slots, checked before ordinary RA |

Both encodings are legal `i64` constants, so using the wrong one is a silent
wrong-answer bug, not a compile error. That is why the profile is resolved once
and threaded through the emitters instead of being written at each call site.

## 2. Selecting a profile

```mlir
module attributes {bosc_ame.target = "upstream"}    // default
module attributes {bosc_ame.target = "qwen3-fpga"}
```

* Attribute name: `bosc_ame.target` (`buddy::boscame::kAmeTargetAttrName`).
* Pass option: `--lower-linalg-to-boscame=target=qwen3-fpga` and
  `--lower-qwen-w8a8-to-boscame=target=qwen3-fpga`.
* Precedence and conflicts (`buddy::boscame::resolveAmeTarget`): the option
  wins when it is set, the attribute wins otherwise, an unparsable value is an
  error, and an option/attribute disagreement is an error.
* Scope is the module. Mixing the two conventions inside one module is not
  supported by design.
* Default is `upstream`, so the default pipeline still produces GEM5-compatible
  output; FPGA semantics are opt-in.

The non-default profile is written back to the module by whichever pass resolves
it (`--lower-linalg-to-boscame`, `--lower-qwen-w8a8-to-boscame`). The default
upstream path adds no marker. A later FPGA pass rejects already-lowered upstream
configuration, so changing targets cannot mix the two contracts.

The FPGA profile is then carried into the backend automatically. The BOSCAME
export adds `+xboscame-fpga` to each function's existing features, using
`llvm.target_features` for `func.func` and `target_features` for `llvm.func`,
when the module says `qwen3-fpga`, so the RISC-V target selects the FPGA
register-file convention from the IR itself:

```bash
# no -mattr=+xboscame-fpga needed: it travels with the IR
buddy-opt ... --lower-boscame ... | buddy-translate --buddy-to-llvmir   | llc -mtriple=riscv64 -mattr=+m,+f,+d,+v,+xboscame
```

The default (upstream) profile adds no such annotation, so the GEM5 path is
byte-for-byte unchanged.

`triton-w8a8-fast-path` is *not* a target switch. It selects the Triton
pre-fusion and the FPGA schedule, and it now requires `target=qwen3-fpga`:
without the FPGA target there is no `i8 x i8 -> f32` AME datapath, and the flag
must not be able to silently enable the bit-field encoding or the FPGA final
store.

`--lower-qwen-w8a8-to-boscame` emits FPGA-only semantics (bit-field
configuration plus the i32 -> f32 accumulator store), so it requires
`target=qwen3-fpga` and diagnoses otherwise.

## 3. `mtype` CSR encoding (FPGA)

Layout (RISC-V Matrix Extension v0.5 / Qwen3 RTL, see
`kernel/src/backends/ame/core/ame_core.c` in the FPGA flow):

```
bit 16    : mma  (matrix multiply-accumulate enable)
bit 12    : mf64, bit 11: mf32, bit 10: mbf16, bit 9: mf16
bit  8    : mint4
bit  7    : mint64, bit 6: mint32, bit 5: mint16, bit 4: mint8
bits 1:0  : msew (element width: 0=e8, 1=e16, 2=e32, 3=e64)
```

| Phase | Element type | Value | Meaning |
| --- | --- | --- | --- |
| MMA | `i8` | `0x10010` = 65552 | mma=1, mint8=1, msew=0 |
| Accumulator | `i32` | `0x10042` = 65602 | mma=1, mint32=1, msew=2 |

The single definition lives in `buddy::boscame::FpgaMtype`
(`midend/include/Dialect/BOSCAME/Transforms/FPGAAMETarget.h`); the upstream
pathway gets its raw widths from the same header through
`ame::getUpstreamMsetTypeImm`, so the two tables cannot drift apart.

## 4. Value-semantics invariants the FPGA schedule must keep

1. **One accumulator chain per logical accumulator.** The accumulator is an SSA
   value (`vector<4x4xi32>`) threaded through the K reduction with
   `scf.for iter_args` / `scf.yield`. It is never stored and reloaded inside the
   K loop: on this hardware `msce32.m` converts i32 -> f32, so reloading the
   fp32 result would feed IEEE-754 bits back as an integer accumulator.
2. **Accumulator initialization before the reduction.** `mlce32.m` from a
   buffer that is provably `+0.0` is the only accepted zero seed (the fp32 bit
   pattern of `+0.0` is the integer zero accumulator). The direct-C matcher
   requires the zero `linalg.fill`, so this is checked, not assumed.
3. **Final write-back once per tile**, after the whole K reduction, followed by
   `llvm.fence seq_cst` before CPU/RVV consumers, including consumers of a
   retained temporary.
4. **Configuration order** stays: tiles/accumulator type, accumulator load,
   MMA type, K loop (`msettilek` + loads + MMA), accumulator type, store. The
   encoding is chosen by the resolved profile, never by the call site.
5. **Direct-C legality** stays strict: a fresh zero-filled temporary with
   exactly one matmul and one copy (plus an optional dealloc) may be lowered
   in place of the copy; anything else keeps the temporary for the later
   pipeline.
6. **The resync round trip is preserved.** The W8A8 pass begins with a
   `1x1` loadA/loadB/MMA/loadC/storeC round trip whose MMA result is
   deliberately overwritten. In SSA every MMA needs an accumulator operand, so
   the accumulator load that feeds it is hoisted above the loads. The net
   instruction sequence is the original one plus that one hoisted `mlce32.m`;
   no instruction is dropped. The MMAs and configuration ops are not `Pure`
   (no `MemoryEffectOpInterface` is declared in `BOSCAME.td`), so they are not
   removed as dead code.

## 5. Shared emitter layer

`midend/include/Dialect/BOSCAME/Transforms/AMEValueEmitter.h` (implementation in
`midend/lib/Dialect/BOSCAME/Transforms/AMEValueEmitter.cpp`) is the only place
that knows how to turn matrix tiles into BOSCAME operations:

```cpp
VectorType getMatrixTileType(MLIRContext *, Type elementType);
Value createByteStride(OpBuilder&, Location, Value memref, unsigned dim = 0);
FailureOr<Value> createLoadA / createLoadB / createLoadBTransposed(...);
FailureOr<Value> createLoadAccumulator(..., Type accElementType,
                                       Type memoryElementType, ...);
FailureOr<Value> createMma(OpBuilder&, Location, Value acc, Value lhs, Value rhs, ...);
LogicalResult createStoreAccumulator(...);
LogicalResult configureTiles / configureTileK(...);
FailureOr<int64_t> configureMmaType / configureAccumulatorType(..., AmeTargetProfile);
```

The emitter creates no loops: wiring an accumulator across a reduction is the
caller's job because only the caller knows how many chains a schedule has.

## 6. Generation chains

```
default target      -> upstream BOSCAME SSA -> msettypei (raw width) -> upstream LLVM IR -> GEM5
bosc_ame.target=    -> FPGA BOSCAME SSA      -> msettype (0x10010/0x10042)
  "qwen3-fpga"                               -> FPGA LLVM IR/asm -> RTL / board
```

Acceptance requires both chains to be produced and checked: the default chain
must not contain `65552`/`65602` or `bosc_ame.fpga` markers, and the FPGA chain
must bind those constants to the right phase.

## 7. Migration status

Done:

* Upstream main merged with upstream's implementation as the skeleton for
  `BOSCAME.td`, `LowerLinalgToBOSCAME.cpp` and `LegalizeForLLVMExport.cpp`.
* Offset-aware `extractPointerFromMemref` ported into the upstream export file
  (the aligned pointer alone ignores the descriptor offset produced by
  subviews/reinterpret_cast, which would address the wrong tile).
* Profile resolution, bit-field encoder and capability checks
  (`FPGAAMETarget.h/.cpp`), shared SSA emitter (`AMEValueEmitter.h/.cpp`).
* `LowerLinalgToBOSCAME`: FPGA dispatcher (`classifyMatmul`), strict direct-C
  lowering on SSA, Triton fusion, consumer fence, dynamic legality contract.
* `LowerQwenW8A8ToBOSCAME`: migrated to the SSA emitter with loop-carried
  accumulators and the hoisted resync accumulator.

Implemented after the 2026-09-15 code review:

* Wide 2A4B and 1A8B/N64-tail schedules with bounded column ranges and an
  accumulator configuration before the first seed.
* Direct-C requires a strict temporary user set and alias-analysis proof that
  final C is distinct from A and B (for example `memref.distinct_objects`).
  Unknown inner strides and ambiguous aliases stay on the CPU/VIR path.
* FPGA loads carry `bosc_ame.fpga.slot`; export produces dedicated
  `llvm.riscv.bosc.fpga.*` load/MMA/final-store intrinsics. The load slot is an
  immediate operand in LLVM IR, not an SSA naming convention.
* A dedicated matrix allocation pass validates CFG liveness, propagates slots
  through PHIs/COPYs/tied accumulator updates, and removes matrix PHIs and
  redundant copies before ordinary RA. FPGA matrix registers are reserved from
  ordinary allocation. Conflicting slots, excess pressure, missing slots and
  live matrix values across calls are diagnosed; no move/spill is synthesized.
* FPGA MMA state effects are isolated from upstream `IntrNoMem` operations.
  Target features are preserved for both `func.func` and `llvm.func`; already
  lowered upstream configuration is rejected before switching to FPGA.

Still pending: a general role type replacing the W8A8 carrier-width heuristic,
RTL/board numeric validation, and the hardware performance baseline. Upstream
Generic* lowerings remain disabled under FPGA; unclaimed Linalg is left for
CPU/VIR. Matrix spilling and hardware copies remain unsupported by design.

## 8. FPGA register ABI: prototype status

The FPGA code generation path is gated behind a dedicated subtarget feature so
that the default `+xboscame` target (the GEM5/upstream flow) keeps upstream's
behaviour byte for byte:

```
-mattr=+xboscame            # upstream mapping, upstream copy/spill behaviour
-mattr=+xboscame,+xboscame-fpga   # FPGA prototype convention
```

Under `+xboscame-fpga` only:

| Aspect | Prototype behaviour | Status |
| --- | --- | --- |
| MVT to register class | `nxv32i32` -> `AccReg`, other matrix MVTs -> `TileReg`, so a loop-carried accumulator PHI lands in the accumulator file | heuristic for the W8A8 datapath (8-bit A/B tiles, 32-bit accumulator); **not** an ABI |
| Same-class matrix copy | rejected with a fatal diagnostic under `+xboscame-fpga`; no `mmve` is synthesized | adaptation preserves the reference instruction set; no ISA assumption is introduced |
| Cross-class matrix copy | not synthesised | `mmve*.a.t` / `mmve*.t.a` carry a GPR row index, so they are not plain copies |
| Matrix register spill | rejected with a fatal diagnostic; covered by `qwen-w8a8-matrix-capacity.ll` and `qwen-w8a8-matrix-spill.ll` | spill/reload is not implemented and must not be relied on |
| Target feature plumbing | the MLIR profile writes `bosc_ame.target` on the module and the export annotates functions with `+xboscame-fpga` | `llc` needs no manual flag; the default profile adds nothing |

The reference FPGA assembly contains no `mmve` instructions.  The adaptation
therefore does not synthesize same-class matrix copies and does not make an ISA
claim about `mmve` `eew` behavior. Redundant same-slot COPYs are eliminated;
any COPY requiring a physical move is diagnosed. The adaptation must preserve
fixed matrix register slots instead of introducing a new move sequence. The
`RISCVFPGARegisterAllocation` pass now enforces those slots and verifies that
simultaneously live values cannot overwrite each other.  The
`eew` question only returns if this project later expands from reproducing the
reference workloads to supporting arbitrary matrix-register copies.

The planned replacement for the width heuristic is a role-carrying
representation (distinct LLVM types or dedicated intrinsics/SDNodes for tile and
accumulator values) so that PHIs, copies and instruction constraints carry the
role instead of inferring it from the MVT. That change spans the MLIR tile ABI,
the LLVM dialect and the backend, and is the prerequisite for submitting any of
this upstream.

### Capability verification under the FPGA profile

Because the prototype convention derives the register file from the element
width, the FPGA profile is only defined for the W8A8 datapath: 8-bit A/B tiles
with a 32-bit accumulator.  `verifyFpgaAmeCapabilities` walks every
`bosc_ame` operation that carries a matrix value and rejects anything outside
that datapath with a diagnostic naming the operation and the offending element
type. It also requires f32 accumulator memory. It runs in
`--lower-linalg-to-boscame`, `--lower-qwen-w8a8-to-boscame`
and the BOSCAME export, so an unsupported op cannot reach the backend and pick
the wrong register file.  Configuration instructions and the high-level W8A8
semantic ops (which work on memrefs) are unaffected, and the check is disabled
for the default upstream profile.

Covered by `tests/Conversion/QwenW8A8/qwen-w8a8-fpga-unsupported-op.mlir`
(operation outside the schedule) and `...-fpga-wrong-type.mlir` (operation
inside the schedule with the wrong element type).

### Why the role distinction has to happen before register allocation

The eventual replacement for the width heuristic (Q1(b)) cannot be a
backend-only pass, and this is worth recording because it rules out the cheapest
option:

* A virtual register's class is fixed when it is created: `MRI.constrainRegClass`
  can only narrow to a subclass, and `MRI.setRegClass` only accepts a superclass
  of the current class. `TileReg` and `AccReg` are unrelated classes, so a pass
  cannot move an already-created PHI/COPY value from one to the other.
* Therefore "start from the instruction constraints and propagate the role along
  PHI/COPY edges" is not implementable after instruction selection. The role must
  be attached where the value is created: the IR type, or a role-carrying
  intrinsic/SDNode.

Options, in increasing cost:

1. **Diagnose ambiguity (short term).** Keep the width heuristic, but make the
   FPGA profile reject a module that gives the same element type both roles
   (for example i32 tiles together with an i32 accumulator), so the unsupported
   combination is a compile error instead of a silently wrong register file.
2. **Role-carrying IR (target state).** Give accumulator and tile values distinct
   LLVM IR types (target extension types) or distinct intrinsic result roles, so
   PHIs, copies and instruction constraints all carry the role. LLVM has no way
   for a target to add new MVTs, so this means either target extension types
   (with the corresponding ISel plumbing) or an upstream change.
3. **MLIR-side role first.** The MLIR tile ABI can carry the role explicitly
   (distinct types/attributes for accumulator vs tile), which then lowers to
   role-specific LLVM intrinsics or extension types. This is also the cleaner
   place to express the hardware contract, and it is a prerequisite for
   submitting any of the backend convention upstream.

### Verification status

| Item | Status |
| --- | --- |
| Python bindings (`operator.index`) | pass |
| Python test suite | 279/279 pass |
| MLIR -> LLVM IR -> assembly | pass |
| W8A8 assembly structure | pass |
| FPGA bit-field constants (`0x10010` / `0x10042`) | pass |
| Tile/Acc semantic distinction | not proven |
| `mmve` `eew` semantics | needs hardware confirmation |
| Cross-class `mmve` index semantics | needs hardware confirmation |
| Spill/reload | explicitly unsupported |
| GEM5 / RTL numeric acceptance | not performed |

## 7b. Verification matrix

Requirement (plan reference) -> evidence.  `BUDDY` is the repository root,
`LLVMB` the target LLVM build (`llvm/build-2d26`), `BM` a buddy build directory
configured against it.

| Requirement | Evidence | Status |
| --- | --- | --- |
| Upstream BOSCAME examples still lower and compile (plan phase 1) | `$LLVMB/bin/llvm-lit -s $BM/examples --filter=BOSCAMEDialect` | 11/11 pass |
| Default pathway unchanged: operation set, raw-width config, no FPGA markers | `tests/Conversion/QwenW8A8/qwen-w8a8-default-upstream.mlir` (MLIR + LLVM IR + absence checks) | pass |
| Value-semantics migration of the FPGA fast path | `tests/Conversion/QwenW8A8/qwen-w8a8-direct-c.mlir` (direct-C, tail extents, byte strides, rejected shapes) | pass |
| Accumulator stays resident across the K reduction | `tests/Conversion/QwenW8A8/qwen-w8a8-accumulator-chain.mlir` (K=128 one chain, K=192 eight chains, K=1024 boundary) | pass |
| Wide 2A4B / N64 schedules keep their instruction order and bank sharing | `tests/Conversion/linalg-to-boscame-triton-w8a8.mlir` (dataflow checks: one activation feeding both half blocks, four shared B tiles) | pass |
| Export to assembly: resync round trip, `msettype` phases, fence, RVV accumulation | `tests/Conversion/QwenW8A8/qwen-w8a8-rvv-asm.mlir` (`llc` runs the FPGA IR with only `+xboscame`; the FPGA feature travels in the IR) | pass |
| Single source of truth for the AME contract | resolver passes record a non-default profile; the export turns it into `+xboscame-fpga`; `qwen-w8a8-default-upstream.mlir` proves the default is untouched | pass |
| Unsupported datapath combinations are compile errors | `qwen-w8a8-fpga-unsupported-op.mlir`, `qwen-w8a8-fpga-wrong-type.mlir` | pass |
| Matrix values stay function-local (role-ABI prerequisite) | `qwen-w8a8-matrix-stays-local.mlir` | pass |
| Register capacity is decidable; no silent spill | `tests/Target/RISCV/qwen-w8a8-matrix-capacity.ll` (8 chains compile), `qwen-w8a8-matrix-spill.ll` (9 chains rejected with a diagnostic) | pass |
| Python bindings and frontend tests (plan phase 1) | `$LLVMB/bin/llvm-lit -s $BM_PYTHON/tests --filter=Python` with `BUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON` | 279/279 pass |
| Numerical acceptance on GEM5 or RTL/board | needs the runner command and a golden baseline from the hardware owner | **blocked (external)** |
| `mmve` `eew` semantics, cross-class move index semantics | needs the ISA/RTL owner | **blocked (external)** |
| General role types instead of the width heuristic (plan Q2) | `docs/BOSCAMEFPGARoleABI.md`; owner selected role-level separation; W8A8 intrinsics now carry load roles and slots | **general type work remains** |

## 7c. One-command check

```bash
scripts/check-fpga-migration.sh            # build-migrate, build-python, llvm/build-2d26
scripts/check-fpga-migration.sh <buddy-build> <buddy-python-build> <llvm-build>
```

It runs the whole automated part of the matrix above (both lit subsets, the
default-path separation, and an FPGA assembly smoke test where `llc` receives no
`-mattr=+xboscame-fpga`), prints the pending external items, and exits non-zero
if any automated check fails.

## 8. Reproducing the checks

```bash
# Build the target LLVM once, in its own build directory.
cmake -S llvm/llvm -B llvm/build-2d26 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
  -DLLVM_CCACHE_BUILD=ON
ninja -C llvm/build-2d26

# Build buddy-mlir against it.
cmake -S . -B build-migrate -G Ninja \
  -DLLVM_DIR=$PWD/llvm/build-2d26/lib/cmake/llvm \
  -DMLIR_DIR=$PWD/llvm/build-2d26/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release -DBUDDY_ENABLE_TESTS=ON
ninja -C build-migrate buddy-opt buddy-translate

# Default pathway must stay FPGA-free.
build-migrate/bin/buddy-opt examples/BOSCAMEDialect/linalg-to-boscame-matmul.mlir \
  -lower-linalg-to-boscame | grep -c 'msettypei'
# FPGA pathway must bind the bit-field constants to the right phases.
build-migrate/bin/buddy-opt --lower-linalg-to-boscame='target=qwen3-fpga' \
  tests/Conversion/QwenW8A8/qwen-w8a8-direct-c.mlir
```

## Fixed-slot adapter ABI (2026-09-15)

The supported FPGA instruction subset is `mlae8.m`, `mlbe8.m`, `mlbte8.m`,
`mlce32.m`, signed `mqma.b.mm`, and final `msce32.m`. Other upstream matrix
operations are rejected under this profile. The matrix load slot is mandatory
at export: A uses 0/2, B uses 4/5/6/7, and accumulators use 0 through 7. An
unannotated hand-written matrix value does not receive an arbitrary slot.

2A4B advances by 64 output columns. Decode uses eight chains only when at least
128 columns remain and four chains for the 64-column tail. All final stores
precede a fence, including direct-C and consumers of a retained temporary.

`qwen-w8a8-wide-codegen.mlir` compiles actual M32 prefill and decode N64/N128/N192
programs at O0/O2/O3 with both pass managers, checks every output tile range and
phase, and inspects both MIR and assembly. The native W8A8 assembly test covers
decode and M32 prefill. Capacity and liveness tests accept compatible branch
joins and reject nine live chains, conflicting PHIs, live values across calls,
and destructive updates or dead definitions that overwrite resident values.

The example lowering script and Makefile have also been exercised with that
small native W8A8 input through LLVM IR, assembly, FPGA instruction-word
conversion, and RISC-V object assembly. This checks tool selection and the
component compilation chain; it does not validate a full model image or runtime
numerical results.

The example and verification tools default to `build-migrate` and
`llvm/build-2d26`. Set `BUDDY_BUILD_DIR` and `LLVM_BUILD_DIR` to select a matching
pair; individual `BUDDY_OPT`, `BUDDY_TRANSLATE` and `MLIR_OPT` overrides are also
available. The Makefile accepts `BUDDY_LLC`. Missing required test suites or
failed compilation make the verification script fail.
