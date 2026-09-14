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
| Register placement | LLVM backend | backend plus verified logical slots |

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
   `llvm.fence seq_cst` when a consumer needs the AME result (the
   `bosc_ame.triton_consumer_fence` marker).
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

Not migrated yet (tracked as follow-up work):

* The wide `2A4B` / `1A8B` schedules (eight accumulator chains). The tiled
  16x16x64 schedule is used as the correctness baseline for those shapes; it
  produces the same numbers with more instructions.
* The upstream `Generic*` lowerings are not registered under the FPGA profile,
  because they program the raw-width configuration. Generic ops therefore stay
  legal under that profile and are left to the CPU/VIR pipeline.
* The LLVM IR / backend fork: FPGA register-form configuration chain, the
  explicit i32 -> f32 final store, CSR `Uses/Defs`, spill/reload of matrix
  registers, and fixed-slot constraints.
* RTL/board numeric validation and the performance baseline.

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
  tests/Conversion/qwen-w8a8-direct-c.mlir
```
