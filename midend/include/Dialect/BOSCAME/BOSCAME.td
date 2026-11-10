//===- BOSCAME.td - BOSCAME dialect operation definitions -----------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

// This SSA-based version models BOSC AME matrix registers as MLIR matrix tile
// values. Physical TR/ACC register numbers are intentionally absent here; they
// are selected later by LLVM instruction selection and register allocation.
//
//===----------------------------------------------------------------------===//

#ifndef BOSCAME_DIALECT
#define BOSCAME_DIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

//===----------------------------------------------------------------------===//
// BOSCAME Dialect Definition
//===----------------------------------------------------------------------===//

def BOSCAME_Dialect : Dialect {
  let name = "bosc_ame";
  let cppNamespace = "::buddy::boscame";
  // The SPM/DMA operations carry their configuration as inherent attributes
  // (`channel`, `bank`, `offset`, ...) and expose them through `attr-dict`.
  // LLVM 3845a205 turned `useStrictPropertiesInAssemblyFormat` on by default,
  // which rejects that combination; keep the documented legacy behaviour so
  // the existing custom assembly formats (and their tests) stay valid until
  // they are migrated to `prop-dict`.
  let useStrictPropertiesInAssemblyFormat = 0;
  let description = [{
    The BOSCAME dialect provides operations for the RISC-V Matrix Extension (RVA23 profile).
    BOSCAME provides matrix multiply-accumulate instructions with tile-based computation,
    supporting various data types and widening modes for AI/ML workloads.
  }];
  // `#bosc_ame.spm<bank>` and `!bosc_ame.dma_token<channel>` use the generic
  // attribute/type printers driven by their own `assemblyFormat`.
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

// High-level BOSC AME tiles are ordinary MLIR vectors.  For example,
// vector<4x4xi32> is legal at the BOSCAME dialect boundary.
def BOSCAME_MatrixType : Type<
    CPred<"::mlir::isa<::mlir::VectorType>($_self)">,
    "BOSC AME matrix vector type">;

//===----------------------------------------------------------------------===//
// BOSCAME Operation Base Class
//===----------------------------------------------------------------------===//

class BOSCAME_Op<string mnemonic, list<Trait> traits = []> :
  Op<BOSCAME_Dialect, mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// SPM storage space and DMA request handle
//===----------------------------------------------------------------------===//

// Memory space attribute attached to a `memref` type.  It answers "where do
// these bytes live", which is orthogonal to "which DMA engine moves them".
// The bank is static type information, never a runtime SSA value: a value whose
// storage space could change at run time cannot be verified statically.
def BOSCAME_SPMAttr : AttrDef<BOSCAME_Dialect, "SPM"> {
  let mnemonic = "spm";
  let summary = "Storage space of one on-chip SPM bank";
  let description = [{
    Marks the storage space of a `memref` as one of the two SPM banks of the
    patch7 platform:

    ```mlir
    memref<4096xi8, #bosc_ame.spm<0>>   // 4096 bytes in SPM0
    memref<64x64xi8, #bosc_ame.spm<1>>  // the same storage, a 2-D logical view
    ```

    `bank` must be 0 or 1.  The attribute says nothing about whether the SSA
    addresses inside that memref are physical or virtual; that is a runtime
    address-mode property (see the SPM/DMA design notes).  It also says nothing
    about element type or layout, so it never replaces the memref shape.
  }];
  let parameters = (ins "int64_t":$bank);
  let assemblyFormat = [{
    `<` $bank `>`
  }];
  let genVerifyDecl = 1;
}

// Handle of one in-flight DMA request.  The channel is part of the type so that
// `bosc_ame.dma.wait` can recover the register block (and therefore the
// completion-barrier address) from its operand alone.
def BOSCAME_DMATokenType : TypeDef<BOSCAME_Dialect, "DMAToken"> {
  let mnemonic = "dma_token";
  let summary = "Handle of an in-flight DMA request on one channel";
  let description = [{
    `!bosc_ame.dma_token<channel>` is the result type of
    `bosc_ame.dma.start`.  `channel` must be 0 or 1 and matches the `channel`
    attribute of the request that produced it.

    A token is a single-use request handle, not a value that can be stored,
    duplicated or transported: it must be consumed by exactly one
    `bosc_ame.dma.wait` in the block that produced it.  The compiler-internal
    shape of the type is deliberately opaque so that no other operation can
    manufacture or inspect a live request.
  }];
  let parameters = (ins "int64_t":$channel);
  let assemblyFormat = [{
    `<` $channel `>`
  }];
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// SPM / DMA operations
//===----------------------------------------------------------------------===//

def SpmViewOp : BOSCAME_Op<"spm.view", [Pure]> {
  let summary = "View a byte range of an SPM bank as a memref";
  let description = [{
    Describes an SPM byte range as a standard `memref` so that ordinary
    memory operations (including the existing matrix load/store operations) can
    access it:

    ```mlir
    %b = bosc_ame.spm.view {bank = 0 : i32, offset = 4096 : i64}
        : memref<64x64xi8, #bosc_ame.spm<0>>
    ```

    `offset` is the byte offset of the range inside the bank and is applied
    exactly once.  The result type must carry `#bosc_ame.spm<bank>` with the
    same bank, and its total element size (`product(shape) * size(element)`)
    must keep `offset + size` inside the 128 KiB bank.

    Element types are restricted to `i8/i16/i32/i64/f16/bf16/f32/f64` so the
    byte size of the range is unambiguous.  The operation itself never
    addresses or moves data: `-lower-bosc-ame` compiles the fixed bank base
    address (`0x20000000` for bank 0, `0x20400000` for bank 1) plus `offset`
    into the pointer, so no workspace root argument is needed and the kernel
    ABI stays free of SPM parameters.
  }];
  let arguments = (ins I32Attr:$bank, I64Attr:$offset);
  let results = (outs AnyMemRef:$view);
  let assemblyFormat = "attr-dict `:` type($view)";
  let hasVerifier = 1;
}

// Managed invocation-local storage in the platform DDR operand window.
def DDRFrameOp : BOSCAME_Op<"ddr.frame", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Acquire a bounded invocation-private DDR frame";
  let results = (outs AnyMemRef:$frame);
  let assemblyFormat = "attr-dict `:` type($frame)";
  let hasVerifier = 1;
}

def DDRReleaseOp : BOSCAME_Op<"ddr.release", [MemoryEffects<[MemFree]>]> {
  let summary = "Release a private DDR frame after its final use";
  let arguments = (ins AnyMemRef:$frame);
  let assemblyFormat = "$frame attr-dict `:` type($frame)";
  let hasVerifier = 1;
}

def DmaStartOp : BOSCAME_Op<"dma.start",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Start an asynchronous contiguous DMA transfer";
  let description = [{
    Starts one contiguous transfer on a DMA channel and returns a request
    handle:

    ```mlir
    %t = bosc_ame.dma.start %src, %dst, %bytes {channel = 0 : i32}
        : (memref<4096xi8>, memref<4096xi8, #bosc_ame.spm<0>>, index)
          -> !bosc_ame.dma_token<0>
    ```

    `bytes` is a byte count, not an element count.  Both endpoints must be
    1-D typed memrefs with unit element stride and a static length. Supported
    elements are i8/i16/i32/i64/f16/bf16/f32/f64. Endpoint capacity is
    elementCount * elementBytes; at least one endpoint must be in an SPM bank.
    The transfer starts at the effective address of
    the memref (descriptor offset included) and is legal when:

      * `bytes` is a positive multiple of 64,
      * both effective addresses are 64 B aligned,
      * both endpoints are at least `bytes` long,
      * the SPM endpoint lies inside a single bank,
      * the two ranges do not overlap.

    *Where the alignment comes from.*  The effective address is the storage
    aligned pointer plus descriptorOffset * elementBytes, including every
    `memref.view` byte shift on the view chain. The root is anchored (a 64 B
    aligned SPM bank, an explicitly aligned allocation, or a DDR argument
    covered by `bosc_ame.dma.abi`). The descriptor offset and view shifts
    require alignment proofs: a static multiple of the granularity, or a
    runtime expression whose divisibility follows from SSA operations.
    A `memref.view` shift is accepted either as a
    static multiple of 64 or as a runtime value the compiler can *prove* to be
    one (a constant, a sum/product of provable factors, a widening cast, or a
    loop induction variable with provably aligned lower bound and step).  A
    shift nothing can be proven about is rejected. A subview adds its offsets
    times source strides; reinterpret_cast replaces the descriptor offset.

    The operation does not wait: the request completes when the matching
    `bosc_ame.dma.wait` executes the completion barrier.  It therefore has
    memory effects (a device write plus a conservative ordering barrier) and is
    never `Pure`, so it cannot be moved, duplicated or dead-code eliminated.
  }];
  let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst, Index:$bytes,
                       I32Attr:$channel);
  let results = (outs BOSCAME_DMATokenType:$token);
  let assemblyFormat =
      "$src `,` $dst `,` $bytes attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def DmaWaitOp : BOSCAME_Op<"dma.wait",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Wait for a DMA request and check its hardware status";
  let description = [{
    Completes the request started by the `bosc_ame.dma.start` that produced
    `token`:

    ```mlir
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    ```

    The operation executes the completion barrier (SRC write-back, fence, one
    volatile `STATUS` read, fence) and then *checks* the status.  A normal
    return means "the transfer completed and succeeded".  On failure the
    runtime takes its error path (the shipped runtime traps); the operation
    itself has no result, so a caller cannot accidentally ignore a failed
    transfer and consume the destination as if it were data.

    The channel and the completion barrier are recovered from the token type,
    which is why a token may not be stored, duplicated or transported.  The
    channel is busy exactly while it has an outstanding request; a new
    `bosc_ame.dma.start` on the same channel is only legal after the previous
    `bosc_ame.dma.wait` has completed it.

    `STATUS` is live data, not a sticky done flag: bit 0 is `busy` and bit 1 is
    `bus_error`.  The runtime rejects `bus_error` and an observable `busy`
    after the barrier, an invalid channel, the `start` "not launched" sentinel
    (token 0) and a token that does not match the channel's current `SRC`.
  }];
  let arguments = (ins BOSCAME_DMATokenType:$token);
  let assemblyFormat = "$token attr-dict `:` qualified(type($token))";
}

//===----------------------------------------------------------------------===//
// Tile operations: SPM <-> AME matrix register file
//===----------------------------------------------------------------------===//
//
// These five operations are the *only* interface between an SPM buffer and the
// AME matrix register file.  A kernel never names a physical matrix register,
// an `mtype` value or a tile size: the shapes and element types of the tile
// values carry that information, and `-lower-bosc-ame` derives the hardware
// configuration (`msettilem/m/n/k`, `msettype`) from them.
//
// Matrix values are ordinary MLIR vectors whose 2-D shape is the *logical*
// tile shape of the matrix multiply, e.g. `vector<64x64xi8>`.  The scalable
// register type the backend actually uses is an implementation detail of the
// lowering, so a kernel reads as matrix algebra and stays portable across
// AME/W8A8 schedules.

def TileLoadAOp : BOSCAME_Op<"tile.load_a",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Load the A (left) matrix tile from SPM into the tile register file";
  let description = [{
    Loads one `M x K` A tile from an SPM buffer and returns it as a matrix
    value:

    ```mlir
    %a = bosc_ame.tile.load_a %spm_a
        : memref<64x64xi8, #bosc_ame.spm<0>> -> vector<64x64xi8>
    ```

    The operation reads `source` (declared as a `Read` effect, so the
    SPM/DMA checker can reject a load of a buffer a DMA transfer is still
    filling) and has no other operands: stride, element type and tile shape all
    come from the operand and result types.

    The memref must be a ranked, statically shaped, 2-D memref in
    `#bosc_ame.spm<bank>` whose layout keeps K contiguous:

      * `strides[1] == 1` (the K dimension is the innermost one),
      * `strides[0] >= K` (rows may be further apart than K, e.g. a pitched
        subview).

    The byte stride the AME load instruction needs is the *physical row*
    stride, i.e. `strides[0] * elementBytes`.  `strides[0] == K` is the
    compact row-major A layout the reference kernel stages with DMA.

    First version: signed `i8` A tiles with `1 <= M, K <= 64`.
  }];
  let arguments = (ins AnyMemRef:$source);
  let results = (outs BOSCAME_MatrixType:$matrix);
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($matrix)";
  let hasVerifier = 1;
}

def TileLoadBOp : BOSCAME_Op<"tile.load_b",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Load the B (right) matrix tile from SPM into the tile register file";
  let description = [{
    Loads one `K x N` B tile from an SPM buffer and returns it as a matrix
    value:

    ```mlir
    %b = bosc_ame.tile.load_b %spm_b
        : memref<64x64xi8, #bosc_ame.spm<1>, strided<[1, 64]>> -> vector<64x64xi8>
    ```

    **Logical vs. physical layout.**  The AME hardware stores B as an
    `N x K` buffer with K contiguous, i.e. the *physical* row of the staged
    buffer is an N-indexed row of K bytes.  Matrix algebra wants `B[K, N]`, so
    the operand is the same storage described logically:

      * `shape  = [K, N]`   (logical shape, matching `vector<KxNxi8>`),
      * `strides[0] == 1`   (K is contiguous inside a physical row),
      * `strides[1] >= K`   (the pitch between two physical rows).

    The AME load instruction therefore uses `strides[1] * elementBytes` as its
    byte stride, *not* `strides[0] * elementBytes`.  A `strided<[N, 1]>`
    row-major `[K, N]` buffer is the *transposed* B layout this hardware load
    does not implement, and is rejected instead of being silently loaded with
    transposed results.  The transposed instruction (`mlbte8.m`) is not part of
    the first version; a kernel that needs it must transpose B before staging
    it, and DMA never transposes.

    First version: signed `i8` B tiles with `1 <= K, N <= 64`.
  }];
  let arguments = (ins AnyMemRef:$source);
  let results = (outs BOSCAME_MatrixType:$matrix);
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($matrix)";
  let hasVerifier = 1;
}

def TileLoadAccOp : BOSCAME_Op<"tile.load_acc",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Load the accumulator tile from SPM into the accumulator register file";
  let description = [{
    Seeds an `M x N` accumulator matrix value from SPM:

    ```mlir
    %acc = bosc_ame.tile.load_acc %spm_acc
        : memref<64x64xi32, #bosc_ame.spm<0>> -> vector<64x64xi32>
    ```

    The accumulator memory is raw `i32` on this target (no conversion, no
    packing): the value is the exact `M x N` INT32 tile that
    `bosc_ame.tile.store_acc` writes back.  The layout keeps N contiguous:

      * `strides[1] == 1`,
      * `strides[0] >= N`.

    The byte stride is the physical row stride `strides[0] * elementBytes`.

    First version: `i32` accumulators with `1 <= M, N <= 64`.
  }];
  let arguments = (ins AnyMemRef:$source);
  let results = (outs BOSCAME_MatrixType:$matrix);
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($matrix)";
  let hasVerifier = 1;
}

def TileZeroAccOp : BOSCAME_Op<"tile.zero_acc", [Pure]> {
  let summary = "Create an i32 zero accumulator without reading memory or old contents";
  let description = [{
    Returns an `M x N` all-zero i32 accumulator. The operation has no input
    accumulator or memory operand, so it introduces no old-content dependency.
    It initializes a reduction; it must not replace an accumulated value at
    a later K block.

    ```mlir
    %zero = bosc_ame.tile.zero_acc : vector<64x64xi32>
    ```

    The NR backend configures the effective M/N and 16-bit widening mode, then
    emits only the verified `mwsub.mm` or `mwsub.h.mm` self-clear with
    `dst == src1 == src2`. Register choice is a backend detail. This operation
    does not expose general accumulator subtraction and reads no zero buffer.
    Subsequent MMA and stores program their own mtype. First version:
    `i32`, `1 <= M, N <= 64`.
  }];
  let results = (outs BOSCAME_MatrixType:$matrix);
  let assemblyFormat = "attr-dict `:` type($matrix)";
  let hasVerifier = 1;
}

def TileMmaOp : BOSCAME_Op<"tile.mma", [Pure]> {
  let summary = "Accumulate A x B into C on the AME matrix multiply-accumulate unit";
  let description = [{
    Computes one `M x N x K` matrix multiply-accumulate:

    ```mlir
    %c1 = bosc_ame.tile.mma %c0, %a, %b
        : vector<64x64xi32>, vector<64x64xi8>, vector<64x64xi8>
          -> vector<64x64xi32>
    ```

    Mathematical semantics:

    ```
    C_out[m, n] = C_in[m, n] + sum_k (A[m, k] * B[k, n])
    ```

    The operand order is fixed: accumulator, A, B.  The accumulator is an SSA
    value, so a chain of reductions is expressed by feeding the result of one
    `tile.mma` into the next instead of relying on implicit hardware register
    state:

    ```mlir
    %c1 = bosc_ame.tile.mma %c0, %a0, %b0 : ...
    %c2 = bosc_ame.tile.mma %c1, %a1, %b1 : ...
    ```

    The accumulator element type is the *wider* one: inputs are signed `i8`,
    the product is accumulated in `i32` with wrapping two's-complement
    arithmetic (the AME accumulator is a plain integer adder, there is no
    saturation on this datapath).  Reinterpreting the result as `i8` is
    therefore never allowed.

    Shape rules: `A` is `M x K`, `B` is `K x N` and `C` is `M x N`; the inner
    dimension of `A` and the outer dimension of `B` must agree, and `C` must
    use the same tile shape before and after the operation.

    The op is `Pure` at this level: the accumulator dependency is carried by
    SSA.  The LLVM intrinsic the lowering emits is `IntrWriteMem`, so the
    hardware update survives even if a result is not used.

    First version: signed `i8 x i8 -> i32`, `1 <= M, N, K <= 64`.
  }];
  let arguments = (ins BOSCAME_MatrixType:$acc, BOSCAME_MatrixType:$lhs,
                       BOSCAME_MatrixType:$rhs);
  let results = (outs BOSCAME_MatrixType:$matrix);
  let assemblyFormat =
      "$acc `,` $lhs `,` $rhs attr-dict `:` type($acc) `,` type($lhs) `,` "
      "type($rhs) `->` type($matrix)";
  let hasVerifier = 1;
}

def TileStoreAccOp : BOSCAME_Op<"tile.store_acc",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Write an accumulator tile back from the register file to SPM";
  let description = [{
    Stores an `M x N` INT32 accumulator tile into SPM:

    ```mlir
    bosc_ame.tile.store_acc %acc, %spm_acc
        : vector<64x64xi32>, memref<64x64xi32, #bosc_ame.spm<0>>
    ```

    The accumulator memory is raw `i32`: this is the *same* representation
    `bosc_ame.tile.load_acc` reads, so a store/load round trip is bit-exact.
    Write-back paths that convert to another element type are a different
    datapath (e.g. the W8A8 `i32 -> f32` final store) and are deliberately not
    part of this operation.

    The operation writes `dest` (declared as a `Write` effect, so the SPM/DMA
    checker rejects a store that overlaps an in-flight transfer) and returns
    nothing: the result of the computation is the memory it writes.  The
    layout rules are those of `bosc_ame.tile.load_acc`:
    `strides[1] == 1`, `strides[0] >= N`, byte stride
    `strides[0] * elementBytes`.

    First version: `i32` accumulators with `1 <= M, N <= 64`.
  }];
  let arguments = (ins BOSCAME_MatrixType:$matrix, AnyMemRef:$dest);
  let assemblyFormat = "$matrix `,` $dest attr-dict `:` type($matrix) `,` "
                       "type($dest)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Exact matrix-register memory transport
//===----------------------------------------------------------------------===//

def MatrixCopyOp : BOSCAME_Op<"matrix.copy",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Copy an exact rectangle through one raw NR ACC32 register";
  let description = [{
    Copies an exact byte rectangle between DDR and SPM, or between independent
    nonoverlapping SPM windows, without arithmetic,
    conversion or transposition. Offsets are bytes from each original
    endpoint's effective pointer; pitches and row_bytes are also bytes.
    Logical i8/i16/i32/i64 and floating-point storage can share the raw e32
    memory path only when every address and row range is word aligned.

    The bounded NR subset requires instruction_bits=32, register_count=1,
    register_class="acc", mtype=65602, at most 64 rows and 256 bytes per row.
    It reserves ACC0 for a single ordered config/load/store bundle enclosed
    by fence rw,rw. No matrix value or runtime call crosses the bundle.
    The ordinary mlce32/msce32 pair accesses only its configured effective
    rectangle; whole-register operations and rounded-up tails are forbidden.
    Internal copies retain both logical buffers in SPM and use ACC0 only for
    the duration of this bundle. They allocate no additional SPM window.
    Complete endpoint ranges, rather than just each tile, must be disjoint.
    Physical TR/ACC SSA values are not memref endpoints; this operation does
    not implement register-class conversion or persistent register copies.
  }];
  let arguments = (ins AnyMemRef:$source, AnyMemRef:$target,
                       I64Attr:$source_offset, I64Attr:$target_offset,
                       I64Attr:$rows, I64Attr:$row_bytes,
                       I64Attr:$source_pitch, I64Attr:$target_pitch,
                       I64Attr:$instruction_bits, I64Attr:$register_count,
                       I64Attr:$mtype, StrAttr:$register_class);
  let assemblyFormat = "$source `,` $target attr-dict `:` type($source) `,` type($target)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Matrix Configuration Operations (with register)
//===----------------------------------------------------------------------===//
def MSettypeOp : BOSCAME_Op<"msettype"> {
  let summary = "Set the mtype CSR based on register value";
  let description = [{
    Sets the mtype CSR from a dynamic register value to configure
    the matrix data type and tile register format.
  }];
  let arguments = (ins I64:$rs1);
  let results = (outs I64:$res);
  let assemblyFormat = "$rs1 attr-dict `:` type($res)";
}

def MSettilemOp : BOSCAME_Op<"msettilem"> {
  let summary = "Set tile M dimension with register value";
  let description = [{
    Sets the tile M dimension (number of rows) from a register value.
    mtilem = min(rs1, TMMAX)
  }];
  let arguments = (ins I64:$rs1);
  let results = (outs I64:$res);
  let assemblyFormat = "$rs1 attr-dict `:` type($res)";
}

def MSettilenOp : BOSCAME_Op<"msettilen"> {
  let summary = "Set tile N dimension with register value";
  let description = [{
    Sets the tile N dimension (number of columns) from a register value.
    mtilen = min(rs1, TNMAX)
  }];
  let arguments = (ins I64:$rs1);
  let results = (outs I64:$res);
  let assemblyFormat = "$rs1 attr-dict `:` type($res)";
}

def MSettilekOp : BOSCAME_Op<"msettilek"> {
  let summary = "Set tile K dimension with register value";
  let description = [{
    Sets the tile K dimension (inner dimension for matrix multiply) from a register value.
    mtilek = min(rs1, TKMAX)
  }];
  let arguments = (ins I64:$rs1);
  let results = (outs I64:$res);
  let assemblyFormat = "$rs1 attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// Matrix Configuration Operations (with immediate)
//===----------------------------------------------------------------------===//
def MSettypeiOp : BOSCAME_Op<"msettypei"> {
  let summary = "Set the bits of mtype CSR based on immediate";
  let description = [{
    Sets the upper bits of the mtype CSR to configure the matrix data type and tile register format.
    The immediate value encodes the extended data type configurations.
  }];
  let arguments = (ins I64Attr:$timm);
  let results = (outs I64:$res);
  let assemblyFormat = "$timm attr-dict `:` type($res)";
}

def MSettypehiOp : BOSCAME_Op<"msettypehi"> {
  let summary = "Set the high bits of mtype CSR based on immediate";
  let description = [{
    Sets the upper bits of the mtype CSR to configure the matrix data type and tile register format.
    The immediate value encodes the extended data type configurations.
  }];
  let arguments = (ins I64Attr:$timm);
  let results = (outs I64:$res);
  let assemblyFormat = "$timm attr-dict `:` type($res)";
}

def MSettilemiOp : BOSCAME_Op<"msettilemi"> {
  let summary = "Set tile M dimension with immediate";
  let description = [{
    Sets the tile M dimension (number of rows) from an immediate value.
    mtilem = min(tilem, TMMAX)
  }];
  let arguments = (ins I64Attr:$tilem);
  let results = (outs I64:$res);
  let assemblyFormat = "$tilem attr-dict `:` type($res)";
}

def MSettileniOp : BOSCAME_Op<"msettileni"> {
  let summary = "Set tile N dimension with immediate";
  let description = [{
    Sets the tile N dimension (number of columns) from an immediate value.
    mtilen = min(tilen, TNMAX)
  }];
  let arguments = (ins I64Attr:$tilen);
  let results = (outs I64:$res);
  let assemblyFormat = "$tilen attr-dict `:` type($res)";
}

def MSettilekiOp : BOSCAME_Op<"msettileki"> {
  let summary = "Set tile K dimension with immediate";
  let description = [{
    Sets the tile K dimension (inner dimension for matrix multiply) from an immediate value.
    mtilek = min(tilek, TKMAX)
  }];
  let arguments = (ins I64Attr:$tilek);
  let results = (outs I64:$res);
  let assemblyFormat = "$tilek attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// Data Move / Broadcast / Transpose Operations Base Classes (MLIR ODS)
//===----------------------------------------------------------------------===//
// Base class for Pure Matrix-to-Matrix Operations (e.g., T-T, A-A, Transpose, Broadcast)
// Arguments: md (dest index), ms1 (source index)
class BOSCAME_PureMove_Op<string mnemonic, string summary_desc>
    : BOSCAME_Op<mnemonic, [AllTypesMatch<["src", "res"]>]> {
  let summary = summary_desc;
  let arguments = (ins BOSCAME_MatrixType:$src);
  let results = (outs BOSCAME_MatrixType:$res);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($res)";
}

// Base class for Matrix Move with GPR Index
// Arguments: md (dest index), ms1 (source index), rs2 (GPR row index)
class BOSCAME_IndexedMove_Op<string mnemonic, string summary_desc>
    : BOSCAME_Op<mnemonic, [AllTypesMatch<["src", "res"]>]> {
  let summary = summary_desc;
  let arguments = (ins BOSCAME_MatrixType:$src, I64:$rs2);
  let results = (outs BOSCAME_MatrixType:$res);
  let assemblyFormat =
      "$src `,` $rs2 attr-dict `:` type($src) `,` type($rs2) "
      "`->` type($res)";
}

// Base class for Matrix Move with Immediate Index
// Arguments: md (dest index), ms1 (source index), imm (Immediate row index)
class BOSCAME_ImmIndexedMove_Op<string mnemonic, string summary_desc>
    : BOSCAME_Op<mnemonic, [AllTypesMatch<["src", "res"]>]> {
  let summary = summary_desc;
  let arguments = (ins BOSCAME_MatrixType:$src, I64Attr:$imm);
  let results = (outs BOSCAME_MatrixType:$res);
  let assemblyFormat =
      "$src `,` $imm attr-dict `:` type($src) `->` type($res)";
}

// Base class for Scalar-to-Matrix Move
// Arguments: md_in (old matrix value), rs1 (scalar value),
// rs2 (element/row index). The result is tied to md_in.
class BOSCAME_ScalarToMatrix_Op<string mnemonic, string summary_desc,
                                Type scalarType>
    : BOSCAME_Op<mnemonic, [AllTypesMatch<["md_in", "res"]>]> {
  let summary = summary_desc;

  // md_in 是被更新前的 tile/acc；结果 res 是更新后的新 SSA 值。
  let arguments = (ins BOSCAME_MatrixType:$md_in,
                       scalarType:$rs1,
                       I64:$rs2);
  let results = (outs BOSCAME_MatrixType:$res);

  let assemblyFormat =
      "$md_in `,` $rs1 `,` $rs2 attr-dict `:` "
      "type($md_in) `,` type($rs1) `,` type($rs2) "
      "`->` type($res)";
}

// Base class for Matrix-to-Scalar Move
// Arguments: ms1 (source index), rs2 (GPR row index) -> Returns I64
class BOSCAME_MatrixToScalar_Op<string mnemonic, string summary_desc,
                                Type scalarType>
    : BOSCAME_Op<mnemonic> {
  let summary = summary_desc;
  let arguments = (ins BOSCAME_MatrixType:$src, I64:$rs2);
  let results = (outs scalarType:$res);
  let assemblyFormat =
      "$src `,` $rs2 attr-dict `:` type($src) `,` type($rs2) "
      "`->` type($res)";
}

//===----------------------------------------------------------------------===//
// Multiclasses for Bitwidth Auto-Generation
//===----------------------------------------------------------------------===//
multiclass BOSCAME_PureMove_Multiclass<string prefix, string suffix, string op_name> {
  def "8Op"  : BOSCAME_PureMove_Op<prefix # "8" # suffix,  "8-bit "  # op_name>;
  def "16Op" : BOSCAME_PureMove_Op<prefix # "16" # suffix, "16-bit " # op_name>;
  def "32Op" : BOSCAME_PureMove_Op<prefix # "32" # suffix, "32-bit " # op_name>;
  def "64Op" : BOSCAME_PureMove_Op<prefix # "64" # suffix, "64-bit " # op_name>;
}

multiclass BOSCAME_IndexedMove_Multiclass<string prefix, string suffix, string op_name> {
  def "8Op"  : BOSCAME_IndexedMove_Op<prefix # "8" # suffix,  "8-bit "  # op_name>;
  def "16Op" : BOSCAME_IndexedMove_Op<prefix # "16" # suffix, "16-bit " # op_name>;
  def "32Op" : BOSCAME_IndexedMove_Op<prefix # "32" # suffix, "32-bit " # op_name>;
  def "64Op" : BOSCAME_IndexedMove_Op<prefix # "64" # suffix, "64-bit " # op_name>;
}

multiclass BOSCAME_ImmIndexedMove_Multiclass<string prefix, string suffix, string op_name> {
  def "8Op"  : BOSCAME_ImmIndexedMove_Op<prefix # "8" # suffix,  "8-bit "  # op_name>;
  def "16Op" : BOSCAME_ImmIndexedMove_Op<prefix # "16" # suffix, "16-bit " # op_name>;
  def "32Op" : BOSCAME_ImmIndexedMove_Op<prefix # "32" # suffix, "32-bit " # op_name>;
  def "64Op" : BOSCAME_ImmIndexedMove_Op<prefix # "64" # suffix, "64-bit " # op_name>;
}

multiclass BOSCAME_IntScalarToMatrix_Multiclass<string prefix, string suffix, string op_name> {
  def "8Op"  : BOSCAME_ScalarToMatrix_Op<prefix # "8" # suffix,  "8-bit "  # op_name, I64>;
  def "16Op" : BOSCAME_ScalarToMatrix_Op<prefix # "16" # suffix, "16-bit " # op_name, I64>;
  def "32Op" : BOSCAME_ScalarToMatrix_Op<prefix # "32" # suffix, "32-bit " # op_name, I64>;
  def "64Op" : BOSCAME_ScalarToMatrix_Op<prefix # "64" # suffix, "64-bit " # op_name, I64>;
}

multiclass BOSCAME_IntMatrixToScalar_Multiclass<string prefix, string suffix, string op_name> {
  def "8Op"  : BOSCAME_MatrixToScalar_Op<prefix # "8" # suffix,  "8-bit "  # op_name, I64>;
  def "16Op" : BOSCAME_MatrixToScalar_Op<prefix # "16" # suffix, "16-bit " # op_name, I64>;
  def "32Op" : BOSCAME_MatrixToScalar_Op<prefix # "32" # suffix, "32-bit " # op_name, I64>;
  def "64Op" : BOSCAME_MatrixToScalar_Op<prefix # "64" # suffix, "64-bit " # op_name, I64>;
}

multiclass BOSCAME_FloatScalarToMatrix_Multiclass<string prefix, string suffix, string op_name> {
  def "8Op"  : BOSCAME_ScalarToMatrix_Op<prefix # "8"  # suffix, "8-bit "  # op_name, F32>;
  def "16Op" : BOSCAME_ScalarToMatrix_Op<prefix # "16" # suffix, "16-bit " # op_name, F16>;
  def "32Op" : BOSCAME_ScalarToMatrix_Op<prefix # "32" # suffix, "32-bit " # op_name, F32>;
  def "64Op" : BOSCAME_ScalarToMatrix_Op<prefix # "64" # suffix, "64-bit " # op_name, F64>;
}
multiclass BOSCAME_FloatMatrixToScalar_Multiclass<string prefix, string suffix, string op_name> {
  def "8Op"  : BOSCAME_MatrixToScalar_Op<prefix # "8"  # suffix, "8-bit "  # op_name, F32>;
  def "16Op" : BOSCAME_MatrixToScalar_Op<prefix # "16" # suffix, "16-bit " # op_name, F16>;
  def "32Op" : BOSCAME_MatrixToScalar_Op<prefix # "32" # suffix, "32-bit " # op_name, F32>;
  def "64Op" : BOSCAME_MatrixToScalar_Op<prefix # "64" # suffix, "64-bit " # op_name, F64>;
}

//===----------------------------------------------------------------------===//
// Data Move Operations Instantiations
//===----------------------------------------------------------------------===//
// 1. Matrix-Matrix Move (Pure Matrix)
defm MmveTt : BOSCAME_PureMove_Multiclass<"mmve", ".t.t", "Tile-to-Tile Move">;
defm MmveAa : BOSCAME_PureMove_Multiclass<"mmve", ".a.a", "Acc-to-Acc Move">;
// 2. Matrix-Matrix Move (with GPR Index)
defm MmveAt : BOSCAME_IndexedMove_Multiclass<"mmve", ".a.t", "Tile-to-Acc Move">;
defm MmveTa : BOSCAME_IndexedMove_Multiclass<"mmve", ".t.a", "Acc-to-Tile Move">;
// 3. Matrix-Matrix Move (with Immediate Index)
defm MmvieAt : BOSCAME_ImmIndexedMove_Multiclass<"mmvie", ".a.t", "Tile-to-Acc Move">;
defm MmvieTa : BOSCAME_ImmIndexedMove_Multiclass<"mmvie", ".t.a", "Acc-to-Tile Move">;
// 4. Matrix to Scalar Move (Returns GPR)
defm MmveXt  : BOSCAME_IntMatrixToScalar_Multiclass<"mmve", ".x.t", "Tile-to-GPR Move">;
defm MmveXa  : BOSCAME_IntMatrixToScalar_Multiclass<"mmve", ".x.a", "Acc-to-GPR Move">;
defm MfmveFt : BOSCAME_FloatMatrixToScalar_Multiclass<"mfmve", ".f.t", "Tile-to-FPR Move">;
defm MfmveFa : BOSCAME_FloatMatrixToScalar_Multiclass<"mfmve", ".f.a", "Acc-to-FPR Move">;
// 5. Scalar GPR to Matrix Move
defm MmveTx  : BOSCAME_IntScalarToMatrix_Multiclass<"mmve", ".t.x", "GPR-to-Tile Move">;
defm MmveAx  : BOSCAME_IntScalarToMatrix_Multiclass<"mmve", ".a.x", "GPR-to-Acc Move">;
defm MfmveTf : BOSCAME_FloatScalarToMatrix_Multiclass<"mfmve", ".t.f", "FPR-to-Tile Move">;
defm MfmveAf : BOSCAME_FloatScalarToMatrix_Multiclass<"mfmve", ".a.f", "FPR-to-Acc Move">;
//===----------------------------------------------------------------------===//
// Broadcast Operations Instantiations
//===----------------------------------------------------------------------===//
// Single Row/Col/Elem Broadcasts (Pure Move semantics)
def MbcarMOp : BOSCAME_PureMove_Op<"mbcar.m", "Broadcast Tile Row">;
def MbcbrMOp : BOSCAME_PureMove_Op<"mbcbr.m", "Broadcast Tile Row">;
def MbccrMOp : BOSCAME_PureMove_Op<"mbccr.m", "Broadcast Acc Row">;

defm Mbcace : BOSCAME_PureMove_Multiclass<"mbcace", ".m", "Broadcast Tile Col A">;
defm Mbcbce : BOSCAME_PureMove_Multiclass<"mbcbce", ".m", "Broadcast Tile Col B">;
defm Mbccce : BOSCAME_PureMove_Multiclass<"mbccce", ".m", "Broadcast Acc Col">;

defm Mbcaee : BOSCAME_PureMove_Multiclass<"mbcaee", ".m", "Broadcast Tile Elem A">;
defm Mbcbee : BOSCAME_PureMove_Multiclass<"mbcbee", ".m", "Broadcast Tile Elem B">;
defm Mbccee : BOSCAME_PureMove_Multiclass<"mbccee", ".m", "Broadcast Acc Elem">;
//===----------------------------------------------------------------------===//
// Transpose Operations Instantiations
//===----------------------------------------------------------------------===//
defm Mtae : BOSCAME_PureMove_Multiclass<"mtae", ".m", "Transpose Tile A">;
defm Mtbe : BOSCAME_PureMove_Multiclass<"mtbe", ".m", "Transpose Tile B">;
defm Mtce : BOSCAME_PureMove_Multiclass<"mtce", ".m", "Transpose Acc C">;

//===----------------------------------------------------------------------===//
// Element-Wise Operations Base Classes (MLIR ODS)
//===----------------------------------------------------------------------===//
// Base class for 3-operand element-wise operations
class BOSCAME_EleWise3_Op<string mnemonic, string summary_desc, string symbol>
    : BOSCAME_Op<mnemonic, [AllTypesMatch<["src1", "src2", "res"]>]> {
  let summary = summary_desc # " matrix operation";
  let arguments = (ins BOSCAME_MatrixType:$src1, BOSCAME_MatrixType:$src2);
  let results = (outs BOSCAME_MatrixType:$res);
  let assemblyFormat =
      "$src1 `,` $src2 attr-dict `:` type($src1) `,` type($src2) "
      "`->` type($res)";
}

class BOSCAME_WidenEleWise3_Op<string mnemonic, string summary_desc,
                               string symbol>
    : BOSCAME_Op<mnemonic> {
  let summary = summary_desc # " widening matrix operation";
  let arguments = (ins BOSCAME_MatrixType:$src1, BOSCAME_MatrixType:$src2);
  let results = (outs BOSCAME_MatrixType:$res);
  let assemblyFormat =
      "$src1 `,` $src2 attr-dict `:` type($src1) `,` type($src2) "
      "`->` type($res)";
}

class BOSCAME_EleWise2_Op<string mnemonic, string summary_desc,
                          string func_name>
    : BOSCAME_Op<mnemonic, [AllTypesMatch<["src", "res"]>]> {
  let summary = summary_desc # " unary matrix operation";
  let arguments = (ins BOSCAME_MatrixType:$src);
  let results = (outs BOSCAME_MatrixType:$res);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($res)";
}

//===----------------------------------------------------------------------===//
// Multiclasses for Bitwidth Auto-Generation
//===----------------------------------------------------------------------===//
// Multiclass for standard integer widths (No Widen)
multiclass BOSCAME_EleWise_Int_NoWiden<string prefix, string op_name, string symbol> {
  def MmOp   : BOSCAME_EleWise3_Op<prefix # ".mm",    op_name # " dynamic-width", symbol>;
  //def HbMmOp : BOSCAME_EleWise3_Op<prefix # ".hb.mm", op_name # " 4-bit", symbol>;
  def BMmOp  : BOSCAME_EleWise3_Op<prefix # ".b.mm",  op_name # " 8-bit", symbol>;
  def HMmOp  : BOSCAME_EleWise3_Op<prefix # ".h.mm",  op_name # " 16-bit", symbol>;
  def WMmOp  : BOSCAME_EleWise3_Op<prefix # ".w.mm",  op_name # " 32-bit", symbol>;
  def DwMmOp : BOSCAME_EleWise3_Op<prefix # ".dw.mm", op_name # " 64-bit", symbol>;
}
// Multiclass for floating point widths (No Widen)
multiclass BOSCAME_EleWise_Float_NoWiden<string prefix, string op_name, string symbol> {
  def MmOp   : BOSCAME_EleWise3_Op<prefix # ".mm",    op_name # " dynamic-width", symbol>;
  //def CfMmOp : BOSCAME_EleWise3_Op<prefix # ".cf.mm", op_name # " 8-bit (cf)", symbol>;
  def HfMmOp : BOSCAME_EleWise3_Op<prefix # ".hf.mm", op_name # " 16-bit (half)", symbol>;
  def FMmOp  : BOSCAME_EleWise3_Op<prefix # ".f.mm",  op_name # " 32-bit (single)", symbol>;
  def DMmOp  : BOSCAME_EleWise3_Op<prefix # ".d.mm",  op_name # " 64-bit (double)", symbol>;
}
// Multiclass for integer widths (Double Widen - stops at 32-bit input)
multiclass BOSCAME_EleWise_Int_DoubleWiden<string prefix, string op_name, string symbol> {
  def MmOp   : BOSCAME_WidenEleWise3_Op<prefix # ".mm",    op_name # " dynamic-width", symbol>;
  //def HbMmOp : BOSCAME_WidenEleWise3_Op<prefix # ".hb.mm", op_name # " 4-bit", symbol>;
  def BMmOp  : BOSCAME_WidenEleWise3_Op<prefix # ".b.mm",  op_name # " 8-bit", symbol>;
  def HMmOp  : BOSCAME_WidenEleWise3_Op<prefix # ".h.mm",  op_name # " 16-bit", symbol>;
  def WMmOp  : BOSCAME_WidenEleWise3_Op<prefix # ".w.mm",  op_name # " 32-bit", symbol>;
}
// Multiclass for floating point widths (Double Widen - stops at 32-bit input)
multiclass BOSCAME_EleWise_Float_DoubleWiden<string prefix, string op_name, string symbol> {
  def MmOp   : BOSCAME_WidenEleWise3_Op<prefix # ".mm",    op_name # " dynamic-width", symbol>;
  //def CfMmOp : BOSCAME_WidenEleWise3_Op<prefix # ".cf.mm", op_name # " 8-bit (cf)", symbol>;
  def HfMmOp : BOSCAME_WidenEleWise3_Op<prefix # ".hf.mm", op_name # " 16-bit (half)", symbol>;
  def FMmOp  : BOSCAME_WidenEleWise3_Op<prefix # ".f.mm",  op_name # " 32-bit (single)", symbol>;
}
// Multiclass for Unary Floating Point (SQRT)
multiclass BOSCAME_EleWise_Unary_Float<string prefix, string op_name, string func> {
  def MOp   : BOSCAME_EleWise2_Op<prefix # ".m",    op_name # " dynamic-width", func>;
  //def CfMOp : BOSCAME_EleWise2_Op<prefix # ".cf.m", op_name # " 8-bit (cf)", func>;
  def HfMOp : BOSCAME_EleWise2_Op<prefix # ".hf.m", op_name # " 16-bit (half)", func>;
  def FMOp  : BOSCAME_EleWise2_Op<prefix # ".f.m",  op_name # " 32-bit (single)", func>;
  def DMOp  : BOSCAME_EleWise2_Op<prefix # ".d.m",  op_name # " 64-bit (double)", func>;
}

//===----------------------------------------------------------------------===//
// Element-Wise Operations Instantiations
//===----------------------------------------------------------------------===//
// 1. ADD / SUB - No Widen
defm Maddu  : BOSCAME_EleWise_Int_NoWiden<"maddu", "Unsigned add", "+">;
defm Msaddu : BOSCAME_EleWise_Int_NoWiden<"msaddu", "Saturated unsigned add", "+">;
defm Madd   : BOSCAME_EleWise_Int_NoWiden<"madd", "Signed add", "+">;
defm Msadd  : BOSCAME_EleWise_Int_NoWiden<"msadd", "Saturated signed add", "+">;
defm Mfadd  : BOSCAME_EleWise_Float_NoWiden<"mfadd", "Floating-point add", "+">;

defm Msubu  : BOSCAME_EleWise_Int_NoWiden<"msubu", "Unsigned sub", "-">;
defm Mssubu : BOSCAME_EleWise_Int_NoWiden<"mssubu", "Saturated unsigned sub", "-">;
defm Msub   : BOSCAME_EleWise_Int_NoWiden<"msub", "Signed sub", "-">;
defm Mssub  : BOSCAME_EleWise_Int_NoWiden<"mssub", "Saturated signed sub", "-">;
defm Mfsub  : BOSCAME_EleWise_Float_NoWiden<"mfsub", "Floating-point sub", "-">;
// 2. ADD / SUB - Double Widen
defm Mwaddu : BOSCAME_EleWise_Int_DoubleWiden<"mwaddu", "Double-widening unsigned add", "+">;
defm Mwadd  : BOSCAME_EleWise_Int_DoubleWiden<"mwadd", "Double-widening signed add", "+">;
defm Mfwadd : BOSCAME_EleWise_Float_DoubleWiden<"mfwadd", "Double-widening floating-point add", "+">;

defm Mwsubu : BOSCAME_EleWise_Int_DoubleWiden<"mwsubu", "Double-widening unsigned sub", "-">;
defm Mwsub  : BOSCAME_EleWise_Int_DoubleWiden<"mwsub", "Double-widening signed sub", "-">;
defm Mfwsub : BOSCAME_EleWise_Float_DoubleWiden<"mfwsub", "Double-widening floating-point sub", "-">;
// 3. MIN / MAX - No Widen
defm Mminu : BOSCAME_EleWise_Int_NoWiden<"mminu", "Unsigned min", "min">;
defm Mmaxu : BOSCAME_EleWise_Int_NoWiden<"mmaxu", "Unsigned max", "max">;
defm Mmin  : BOSCAME_EleWise_Int_NoWiden<"mmin", "Signed min", "min">;
defm Mmax  : BOSCAME_EleWise_Int_NoWiden<"mmax", "Signed max", "max">;
defm Mfmin : BOSCAME_EleWise_Float_NoWiden<"mfmin", "Floating-point min", "min">;
defm Mfmax : BOSCAME_EleWise_Float_NoWiden<"mfmax", "Floating-point max", "max">;
// 4. MUL / MULH / DIV - No Widen
defm Msmulu  : BOSCAME_EleWise_Int_NoWiden<"msmulu", "Saturated unsigned mul", "*">;
defm Mmul    : BOSCAME_EleWise_Int_NoWiden<"mmul", "Signed mul", "*">;
defm Msmul   : BOSCAME_EleWise_Int_NoWiden<"msmul", "Saturated signed mul", "*">;
defm Mfmul   : BOSCAME_EleWise_Float_NoWiden<"mfmul", "Floating-point mul", "*">;
defm Mmulhu  : BOSCAME_EleWise_Int_NoWiden<"mmulhu", "Unsigned mul high", "mulh">;
defm Mmulh   : BOSCAME_EleWise_Int_NoWiden<"mmulh", "Signed mul high", "mulh">;
defm Mmulhsu : BOSCAME_EleWise_Int_NoWiden<"mmulhsu", "Signed-unsigned mul high", "mulhsu">;
defm Msmulsu : BOSCAME_EleWise_Int_NoWiden<"msmulsu", "Saturated signed-unsigned mul high", "mulhsu">;
defm Mfdiv   : BOSCAME_EleWise_Float_NoWiden<"mfdiv", "Floating-point div", "/">;
// 5. MUL - Double Widen
defm Mwmulu  : BOSCAME_EleWise_Int_DoubleWiden<"mwmulu", "Double-widening unsigned mul", "*">;
defm Mwmul   : BOSCAME_EleWise_Int_DoubleWiden<"mwmul", "Double-widening signed mul", "*">;
defm Mwmulsu : BOSCAME_EleWise_Int_DoubleWiden<"mwmulsu", "Double-widening signed-unsigned mul", "*">;
defm Mfwmul  : BOSCAME_EleWise_Float_DoubleWiden<"mfwmul", "Double-widening floating-point mul", "*">;
// 6. LOGIC & SHIFT - No Widen (Individual & Multiclass)
def MandMmOp : BOSCAME_EleWise3_Op<"mand.mm", "Bitwise AND", "&">;
def MorMmOp  : BOSCAME_EleWise3_Op<"mor.mm", "Bitwise OR", "|">;
def MxorMmOp : BOSCAME_EleWise3_Op<"mxor.mm", "Bitwise XOR", "^">;

defm Msll : BOSCAME_EleWise_Int_NoWiden<"msll", "Logical left shift", "<<">;
defm Msrl : BOSCAME_EleWise_Int_NoWiden<"msrl", "Logical right shift", ">>">;
defm Msra : BOSCAME_EleWise_Int_NoWiden<"msra", "Arithmetic right shift", ">>a">;
// 7. SQRT - No Widen
defm Mfsqrt : BOSCAME_EleWise_Unary_Float<"mfsqrt", "Floating-point square root", "sqrt">;

//===----------------------------------------------------------------------===//
// Matrix Load Instructions
// Load matrix tiles from memory to tile registers
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// BOSC AME Load/Store Operation Base Classes (MLIR ODS)
//===----------------------------------------------------------------------===//
// Base class for all Matrix Load Operations
class BOSCAME_Load_Op<string mnemonic, string summary_text, string shape_desc>
    : BOSCAME_Op<mnemonic> {
  let summary = summary_text;
  let arguments = (ins AnyMemRef:$base, I64:$stride);
  let results = (outs BOSCAME_MatrixType:$res);
  let assemblyFormat =
      "$base `,` $stride attr-dict `:` type($base) `->` type($res)";
}

class BOSCAME_Store_Op<string mnemonic, string summary_text, string shape_desc>
    : BOSCAME_Op<mnemonic> {
  let summary = summary_text;
  let arguments = (ins BOSCAME_MatrixType:$src, AnyMemRef:$base, I64:$stride);
  let assemblyFormat =
      "$src `,` $base `,` $stride attr-dict `:` type($src) `,` type($base)";
}

//===----------------------------------------------------------------------===//
// Multiclasses for Bitwidth Auto-Generation
//===----------------------------------------------------------------------===//
multiclass BOSCAME_Load_Multiclass<string prefix, string name_base, string shape_desc> {
  def "8mOp"  : BOSCAME_Load_Op<prefix # "8.m",  "Load 8-bit " # name_base, shape_desc>;
  def "16mOp" : BOSCAME_Load_Op<prefix # "16.m", "Load 16-bit " # name_base, shape_desc>;
  def "32mOp" : BOSCAME_Load_Op<prefix # "32.m", "Load 32-bit " # name_base, shape_desc>;
  def "64mOp" : BOSCAME_Load_Op<prefix # "64.m", "Load 64-bit " # name_base, shape_desc>;
}

multiclass BOSCAME_Store_Multiclass<string prefix, string name_base, string shape_desc> {
  def "8mOp"  : BOSCAME_Store_Op<prefix # "8.m",  "Store 8-bit " # name_base, shape_desc>;
  def "16mOp" : BOSCAME_Store_Op<prefix # "16.m", "Store 16-bit " # name_base, shape_desc>;
  def "32mOp" : BOSCAME_Store_Op<prefix # "32.m", "Store 32-bit " # name_base, shape_desc>;
  def "64mOp" : BOSCAME_Store_Op<prefix # "64.m", "Store 64-bit " # name_base, shape_desc>;
}

//===----------------------------------------------------------------------===//
// Matrix Load Instructions Instantiations
//===----------------------------------------------------------------------===//
// Matrix A (Shape: mtilem x mtilek)
defm Mlae  : BOSCAME_Load_Multiclass<"mlae",  "left matrix tile A", "mtilem x mtilek">;
defm Mlate : BOSCAME_Load_Multiclass<"mlate", "left matrix tile A (transposed)", "mtilek x mtilem">;
defm Mltre : BOSCAME_Load_Multiclass<"mltre", "whole tile matrix", "mtilem x mtilek">;

// Matrix B (Shape: mtilek x mtilen)
defm Mlbe  : BOSCAME_Load_Multiclass<"mlbe",  "right matrix tile B", "mtilek x mtilen">;
defm Mlbte : BOSCAME_Load_Multiclass<"mlbte", "right matrix tile B (transposed)", "mtilen x mtilek">;

// Matrix C / Accumulator (Shape: mtilem x mtilen)
defm Mlce   : BOSCAME_Load_Multiclass<"mlce",   "output matrix tile C", "mtilem x mtilen">;
defm Mlcte  : BOSCAME_Load_Multiclass<"mlcte",  "output matrix tile C (transposed)", "mtilen x mtilem">;
defm Mlacce : BOSCAME_Load_Multiclass<"mlacce", "whole accumulation matrix", "mtilem x mtilen">;

//===----------------------------------------------------------------------===//
// Matrix Store Instructions Instantiations
//===----------------------------------------------------------------------===//
// Matrix A (Shape: mtilem x mtilek)
defm Msae  : BOSCAME_Store_Multiclass<"msae",  "left matrix tile A", "mtilem x mtilek">;
defm Msate : BOSCAME_Store_Multiclass<"msate", "left matrix tile A (transposed)", "mtilek x mtilem">;
defm Mstre : BOSCAME_Store_Multiclass<"mstre", "whole tile matrix", "mtilem x mtilek">;

// Matrix B (Shape: mtilek x mtilen)
defm Msbe  : BOSCAME_Store_Multiclass<"msbe",  "right matrix tile B", "mtilek x mtilen">;
defm Msbte : BOSCAME_Store_Multiclass<"msbte", "right matrix tile B (transposed)", "mtilen x mtilek">;

// Matrix C / Accumulator (Shape: mtilem x mtilen)
defm Msce   : BOSCAME_Store_Multiclass<"msce",   "output matrix tile C", "mtilem x mtilen">;
defm Mscte  : BOSCAME_Store_Multiclass<"mscte",  "output matrix tile C (transposed)", "mtilen x mtilem">;
defm Msacce : BOSCAME_Store_Multiclass<"msacce", "whole accumulation matrix", "mtilem x mtilen">;

//===----------------------------------------------------------------------===//
// Matrix Multiply-Accumulate (MMA) Operations Base Class
//===----------------------------------------------------------------------===//
class BOSCAME_NoWidenMMA_Op<string mnemonic, string summary_desc>
    : BOSCAME_Op<mnemonic,
                 [AllTypesMatch<["acc", "lhs", "rhs", "res"]>]> {
  let summary = summary_desc # " matrix multiply-accumulate";
  let arguments = (ins
    BOSCAME_MatrixType:$acc,
    BOSCAME_MatrixType:$lhs,
    BOSCAME_MatrixType:$rhs
  );
  let results = (outs BOSCAME_MatrixType:$res);
  let assemblyFormat =
      "$acc `,` $lhs `,` $rhs attr-dict `:` "
      "type($acc) `,` type($lhs) `,` type($rhs) `->` type($res)";
}

class BOSCAME_WidenMMA_Op<string mnemonic, string summary_desc>
    : BOSCAME_Op<mnemonic,
                 [AllTypesMatch<["acc", "res"]>,
                  AllTypesMatch<["lhs", "rhs"]>]> {
  let summary = summary_desc # " matrix multiply-accumulate";
  let arguments = (ins
    BOSCAME_MatrixType:$acc,
    BOSCAME_MatrixType:$lhs,
    BOSCAME_MatrixType:$rhs
  );
  let results = (outs BOSCAME_MatrixType:$res);
  let assemblyFormat =
      "$acc `,` $lhs `,` $rhs attr-dict `:` "
      "type($acc) `,` type($lhs) `,` type($rhs) `->` type($res)";
}

//===----------------------------------------------------------------------===//
// MMA Operations Instantiations
//===----------------------------------------------------------------------===//
//===--------------------------------------------------------------------===//
// 1. No-widen matrix multiply-accumulate
//===--------------------------------------------------------------------===//
// int
def MmaMmOp    : BOSCAME_NoWidenMMA_Op<"mma.mm",    "Signed dynamic-width">;
def MmaHmmOp   : BOSCAME_NoWidenMMA_Op<"mma.h.mm",  "Signed 16-bit">;
def MmaWmmOp   : BOSCAME_NoWidenMMA_Op<"mma.w.mm",  "Signed 32-bit">;
def MmaDwmmOp  : BOSCAME_NoWidenMMA_Op<"mma.dw.mm", "Signed 64-bit">;

// int, saturated
def MsmaMmOp   : BOSCAME_NoWidenMMA_Op<"msma.mm",    "Saturated signed dynamic-width">;
def MsmaHmmOp  : BOSCAME_NoWidenMMA_Op<"msma.h.mm",  "Saturated signed 16-bit">;
def MsmaWmmOp  : BOSCAME_NoWidenMMA_Op<"msma.w.mm",  "Saturated signed 32-bit">;
def MsmaDwmmOp : BOSCAME_NoWidenMMA_Op<"msma.dw.mm", "Saturated signed 64-bit">;

// uint
def MmauMmOp   : BOSCAME_NoWidenMMA_Op<"mmau.mm",    "Unsigned dynamic-width">;
def MmauHmmOp  : BOSCAME_NoWidenMMA_Op<"mmau.h.mm",  "Unsigned 16-bit">;
def MmauWmmOp  : BOSCAME_NoWidenMMA_Op<"mmau.w.mm",  "Unsigned 32-bit">;
def MmauDwmmOp : BOSCAME_NoWidenMMA_Op<"mmau.dw.mm", "Unsigned 64-bit">;

// uint, saturated
def MsmauMmOp   : BOSCAME_NoWidenMMA_Op<"msmau.mm",    "Saturated unsigned dynamic-width">;
def MsmauHmmOp  : BOSCAME_NoWidenMMA_Op<"msmau.h.mm",  "Saturated unsigned 16-bit">;
def MsmauWmmOp  : BOSCAME_NoWidenMMA_Op<"msmau.w.mm",  "Saturated unsigned 32-bit">;
def MsmauDwmmOp : BOSCAME_NoWidenMMA_Op<"msmau.dw.mm", "Saturated unsigned 64-bit">;

// float point
def MfmaMmOp   : BOSCAME_NoWidenMMA_Op<"mfma.mm",    "Floating-point dynamic-width">;
def MfmaHfmmOp : BOSCAME_NoWidenMMA_Op<"mfma.hf.mm", "Floating-point 16-bit (half)">;
def MfmaFmmOp  : BOSCAME_NoWidenMMA_Op<"mfma.f.mm",  "Floating-point 32-bit (single)">;
def MfmaDmmOp  : BOSCAME_NoWidenMMA_Op<"mfma.d.mm",  "Floating-point 64-bit (double)">;
//===--------------------------------------------------------------------===//
// 2. Double-widen matrix multiply-accumulate
//===--------------------------------------------------------------------===//
// uint
def MwmauMmOp  : BOSCAME_WidenMMA_Op<"mwmau.mm",   "Double-widening unsigned dynamic-width">;
def MwmauHmmOp : BOSCAME_WidenMMA_Op<"mwmau.h.mm", "Double-widening unsigned 16-bit">;
def MwmauWmmOp : BOSCAME_WidenMMA_Op<"mwmau.w.mm", "Double-widening unsigned 32-bit">;

// uint, saturated
def MswmauMmOp  : BOSCAME_WidenMMA_Op<"mswmau.mm",   "Saturated double-widening unsigned dynamic-width">;
def MswmauHmmOp : BOSCAME_WidenMMA_Op<"mswmau.h.mm", "Saturated double-widening unsigned 16-bit">;
def MswmauWmmOp : BOSCAME_WidenMMA_Op<"mswmau.w.mm", "Saturated double-widening unsigned 32-bit">;

// int
def MwmaMmOp  : BOSCAME_WidenMMA_Op<"mwma.mm",   "Double-widening signed dynamic-width">;
def MwmaHmmOp : BOSCAME_WidenMMA_Op<"mwma.h.mm", "Double-widening signed 16-bit">;
def MwmaWmmOp : BOSCAME_WidenMMA_Op<"mwma.w.mm", "Double-widening signed 32-bit">;

// int, saturated
def MswmaMmOp  : BOSCAME_WidenMMA_Op<"mswma.mm",   "Saturated double-widening signed dynamic-width">;
def MswmaHmmOp : BOSCAME_WidenMMA_Op<"mswma.h.mm", "Saturated double-widening signed 16-bit">;
def MswmaWmmOp : BOSCAME_WidenMMA_Op<"mswma.w.mm", "Saturated double-widening signed 32-bit">;

// float point
def MfwmaMmOp   : BOSCAME_WidenMMA_Op<"mfwma.mm",    "Double-widening floating-point dynamic-width">;
//def MfwmaCfmmOp : BOSCAME_WidenMMA_Op<"mfwma.cf.mm", "Double-widening floating-point 8-bit (cf)">;
def MfwmaHfmmOp : BOSCAME_WidenMMA_Op<"mfwma.hf.mm", "Double-widening floating-point 16-bit (half)">;
def MfwmaFmmOp  : BOSCAME_WidenMMA_Op<"mfwma.f.mm",  "Double-widening floating-point 32-bit (single)">;
//===--------------------------------------------------------------------===//
// 3. Quad-widen matrix multiply-accumulate
//===--------------------------------------------------------------------===//
// uint
def MqmauMmOp  : BOSCAME_WidenMMA_Op<"mqmau.mm",   "Quad-widening unsigned dynamic-width">;
def MqmauBmmOp : BOSCAME_WidenMMA_Op<"mqmau.b.mm", "Quad-widening unsigned 8-bit">;

// uint, saturated
def MsqmauMmOp  : BOSCAME_WidenMMA_Op<"msqmau.mm",   "Saturated quad-widening unsigned dynamic-width">;
def MsqmauBmmOp : BOSCAME_WidenMMA_Op<"msqmau.b.mm", "Saturated quad-widening unsigned 8-bit">;

// int
def MqmaMmOp  : BOSCAME_WidenMMA_Op<"mqma.mm",   "Quad-widening signed dynamic-width">;
def MqmaBmmOp : BOSCAME_WidenMMA_Op<"mqma.b.mm", "Quad-widening signed 8-bit">;

// int, saturated
def MsqmaMmOp  : BOSCAME_WidenMMA_Op<"msqma.mm",   "Saturated quad-widening signed dynamic-width">;
def MsqmaBmmOp : BOSCAME_WidenMMA_Op<"msqma.b.mm", "Saturated quad-widening signed 8-bit">;

// float point
def MfqmaMmOp   : BOSCAME_WidenMMA_Op<"mfqma.mm",    "Quad-widening floating-point dynamic-width">;
//def MfqmaCfmmOp : BOSCAME_WidenMMA_Op<"mfqma.cf.mm", "Quad-widening floating-point 8-bit (cf)">;
//===--------------------------------------------------------------------===//
// 4. Oct-widen matrix multiply-accumulate
//===--------------------------------------------------------------------===//
// uint
def MomauMmOp   : BOSCAME_WidenMMA_Op<"momau.mm",    "Oct-widening unsigned dynamic-width">;
//def MomauHbmmOp : BOSCAME_WidenMMA_Op<"momau.hb.mm", "Oct-widening unsigned 4-bit (hb)">;

// uint, saturated
def MsomauMmOp   : BOSCAME_WidenMMA_Op<"msomau.mm",    "Saturated oct-widening unsigned dynamic-width">;
//def MsomauHbmmOp : BOSCAME_WidenMMA_Op<"msomau.hb.mm", "Saturated oct-widening unsigned 4-bit (hb)">;

// int
def MomaMmOp   : BOSCAME_WidenMMA_Op<"moma.mm",    "Oct-widening signed dynamic-width">;
//def MomaHbmmOp : BOSCAME_WidenMMA_Op<"moma.hb.mm", "Oct-widening signed 4-bit (hb)">;

// int, saturated
def MsomaMmOp   : BOSCAME_WidenMMA_Op<"msoma.mm",    "Saturated oct-widening signed dynamic-width">;
//def MsomaHbmmOp : BOSCAME_WidenMMA_Op<"msoma.hb.mm", "Saturated oct-widening signed 4-bit (hb)">;

//===----------------------------------------------------------------------===//
// BOSCAME Intrinsic Operation Definitions (for LLVM IR translation)
// These map directly to LLVM intrinsics defined in IntrinsicsRISCVBOSCExt.td
//===----------------------------------------------------------------------===//

// Base class for BOSCAME intrinsic ops
// Note: enumName must match the intrinsic name pattern: riscv_bosc_<name>
class BOSCAME_IntrOpBase<string mnemonic, int numRes = 0,
                         list<int> overloadedResults = [],
                         list<int> overloadedOperands = [],
                         list<Trait> traits = []> :
  LLVM_IntrOpBase<BOSCAME_Dialect,
                  "intr." # mnemonic,
                  "riscv_bosc_" # !subst(".", "_", mnemonic),
                  overloadedResults,
                  overloadedOperands,
                  traits,
                  numRes>;

// Configuration intrinsics (with immediate value)
def BOSCAME_Msettypei_IntrOp : BOSCAME_IntrOpBase<"msettypei", 1, [0]>,
  Arguments<(ins LLVM_Type:$timm)>;
def BOSCAME_Msettypehi_IntrOp : BOSCAME_IntrOpBase<"msettypehi", 1, [0]>,
  Arguments<(ins LLVM_Type:$timm)>;
def BOSCAME_Msettilemi_IntrOp : BOSCAME_IntrOpBase<"msettilemi", 1, [0]>,
  Arguments<(ins LLVM_Type:$tilem)>;
def BOSCAME_Msettileni_IntrOp : BOSCAME_IntrOpBase<"msettileni", 1, [0]>,
  Arguments<(ins LLVM_Type:$tilen)>;
def BOSCAME_Msettileki_IntrOp : BOSCAME_IntrOpBase<"msettileki", 1, [0]>,
  Arguments<(ins LLVM_Type:$tilek)>;

// Configuration intrinsics (with register value)
def BOSCAME_Msettype_IntrOp : BOSCAME_IntrOpBase<"msettype", 1, [0]>,
  Arguments<(ins LLVM_Type:$rs1)>;
def BOSCAME_Msettilem_IntrOp : BOSCAME_IntrOpBase<"msettilem", 1, [0]>,
  Arguments<(ins LLVM_Type:$rs1)>;
def BOSCAME_Msettilen_IntrOp : BOSCAME_IntrOpBase<"msettilen", 1, [0]>,
  Arguments<(ins LLVM_Type:$rs1)>;
def BOSCAME_Msettilek_IntrOp : BOSCAME_IntrOpBase<"msettilek", 1, [0]>,
  Arguments<(ins LLVM_Type:$rs1)>;

//===----------------------------------------------------------------------===//
// Data Move / Broadcast / Transpose Intrinsic Operations (LLVM IR Translation)
//===----------------------------------------------------------------------===//
class BOSCAME_PureMove_IntrOp<string mnemonic>
    : BOSCAME_IntrOpBase<mnemonic, 1, [0]>,
      Arguments<(ins LLVM_Type:$src)>;

class BOSCAME_IndexedMove_IntrOp<string mnemonic>
    : BOSCAME_IntrOpBase<mnemonic, 1, [0], [0, 1]>,
      Arguments<(ins LLVM_Type:$src, LLVM_Type:$rs2)>;

class BOSCAME_ImmIndexedMove_IntrOp<string mnemonic>
    : BOSCAME_IntrOpBase<mnemonic, 1, [0], [0, 1]>,
      Arguments<(ins LLVM_Type:$src, LLVM_Type:$imm)>;

class BOSCAME_MatrixToScalar_IntrOp<string mnemonic>
    : BOSCAME_IntrOpBase<mnemonic, 1, [0], [0, 1]>,
      Arguments<(ins LLVM_Type:$src, LLVM_Type:$rs2)>;

class BOSCAME_ScalarToMatrix_IntrOp<string mnemonic>
    : BOSCAME_IntrOpBase<mnemonic, 1, [0], [0, 1, 2]>,
      Arguments<(ins LLVM_Type:$md_in,
                     LLVM_Type:$value,
                     LLVM_Type:$rs2)>;

//===----------------------------------------------------------------------===//
// Multiclasses for IntrOp Bitwidth Auto-Generation
//===----------------------------------------------------------------------===//
multiclass BOSCAME_PureMove_IntrOp_Multiclass<string prefix, string suffix> {
  def "8_IntrOp"  : BOSCAME_PureMove_IntrOp<prefix # "8."  # suffix>;
  def "16_IntrOp" : BOSCAME_PureMove_IntrOp<prefix # "16." # suffix>;
  def "32_IntrOp" : BOSCAME_PureMove_IntrOp<prefix # "32." # suffix>;
  def "64_IntrOp" : BOSCAME_PureMove_IntrOp<prefix # "64." # suffix>;
}

multiclass BOSCAME_IndexedMove_IntrOp_Multiclass<string prefix, string suffix> {
  def "8_IntrOp"  : BOSCAME_IndexedMove_IntrOp<prefix # "8."  # suffix>;
  def "16_IntrOp" : BOSCAME_IndexedMove_IntrOp<prefix # "16." # suffix>;
  def "32_IntrOp" : BOSCAME_IndexedMove_IntrOp<prefix # "32." # suffix>;
  def "64_IntrOp" : BOSCAME_IndexedMove_IntrOp<prefix # "64." # suffix>;
}

multiclass BOSCAME_ImmIndexedMove_IntrOp_Multiclass<string prefix, string suffix> {
  def "8_IntrOp"  : BOSCAME_ImmIndexedMove_IntrOp<prefix # "8."  # suffix>;
  def "16_IntrOp" : BOSCAME_ImmIndexedMove_IntrOp<prefix # "16." # suffix>;
  def "32_IntrOp" : BOSCAME_ImmIndexedMove_IntrOp<prefix # "32." # suffix>;
  def "64_IntrOp" : BOSCAME_ImmIndexedMove_IntrOp<prefix # "64." # suffix>;
}

multiclass BOSCAME_MatrixToScalar_IntrOp_Multiclass<string prefix, string suffix> {
  def "8_IntrOp"  : BOSCAME_MatrixToScalar_IntrOp<prefix # "8."  # suffix>;
  def "16_IntrOp" : BOSCAME_MatrixToScalar_IntrOp<prefix # "16." # suffix>;
  def "32_IntrOp" : BOSCAME_MatrixToScalar_IntrOp<prefix # "32." # suffix>;
  def "64_IntrOp" : BOSCAME_MatrixToScalar_IntrOp<prefix # "64." # suffix>;
}

multiclass BOSCAME_ScalarToMatrix_IntrOp_Multiclass<string prefix, string suffix> {
  def "8_IntrOp"  : BOSCAME_ScalarToMatrix_IntrOp<prefix # "8."  # suffix>;
  def "16_IntrOp" : BOSCAME_ScalarToMatrix_IntrOp<prefix # "16." # suffix>;
  def "32_IntrOp" : BOSCAME_ScalarToMatrix_IntrOp<prefix # "32." # suffix>;
  def "64_IntrOp" : BOSCAME_ScalarToMatrix_IntrOp<prefix # "64." # suffix>;
}

//===----------------------------------------------------------------------===//
// Data Move Operations Instantiations
//===----------------------------------------------------------------------===//
// 1. Matrix-Matrix Move (Pure Matrix)
defm BOSCAME_MmveTt : BOSCAME_PureMove_IntrOp_Multiclass<"mmve", "t.t">;
defm BOSCAME_MmveAa : BOSCAME_PureMove_IntrOp_Multiclass<"mmve", "a.a">;
// 2. Matrix-Matrix Move (with GPR Index)
defm BOSCAME_MmveAt : BOSCAME_IndexedMove_IntrOp_Multiclass<"mmve", "a.t">;
defm BOSCAME_MmveTa : BOSCAME_IndexedMove_IntrOp_Multiclass<"mmve", "t.a">;
// 3. Matrix-Matrix Move (with Immediate Index)
defm BOSCAME_MmvieAt : BOSCAME_ImmIndexedMove_IntrOp_Multiclass<"mmvie", "a.t">;
defm BOSCAME_MmvieTa : BOSCAME_ImmIndexedMove_IntrOp_Multiclass<"mmvie", "t.a">;
// 4. Matrix to Scalar Move (Returns GPR)
defm BOSCAME_MmveXt  : BOSCAME_MatrixToScalar_IntrOp_Multiclass<"mmve", "x.t">;
defm BOSCAME_MmveXa  : BOSCAME_MatrixToScalar_IntrOp_Multiclass<"mmve", "x.a">;
defm BOSCAME_MfmveFt : BOSCAME_MatrixToScalar_IntrOp_Multiclass<"mfmve", "f.t">;
defm BOSCAME_MfmveFa : BOSCAME_MatrixToScalar_IntrOp_Multiclass<"mfmve", "f.a">;
// 5. Scalar GPR to Matrix Move
defm BOSCAME_MmveTx  : BOSCAME_ScalarToMatrix_IntrOp_Multiclass<"mmve", "t.x">;
defm BOSCAME_MmveAx  : BOSCAME_ScalarToMatrix_IntrOp_Multiclass<"mmve", "a.x">;
defm BOSCAME_MfmveTf : BOSCAME_ScalarToMatrix_IntrOp_Multiclass<"mfmve", "t.f">;
defm BOSCAME_MfmveAf : BOSCAME_ScalarToMatrix_IntrOp_Multiclass<"mfmve", "a.f">;

//===----------------------------------------------------------------------===//
// Broadcast Operations Instantiations
//===----------------------------------------------------------------------===//
// Single Row/Col/Elem Broadcasts (Pure Move semantics)
def BOSCAME_MbcarM_IntrOp : BOSCAME_PureMove_IntrOp<"mbcar.m">;
def BOSCAME_MbcbrM_IntrOp : BOSCAME_PureMove_IntrOp<"mbcbr.m">;
def BOSCAME_MbccrM_IntrOp : BOSCAME_PureMove_IntrOp<"mbccr.m">;

defm BOSCAME_Mbcace : BOSCAME_PureMove_IntrOp_Multiclass<"mbcace", "m">;
defm BOSCAME_Mbcbce : BOSCAME_PureMove_IntrOp_Multiclass<"mbcbce", "m">;
defm BOSCAME_Mbccce : BOSCAME_PureMove_IntrOp_Multiclass<"mbccce", "m">;

defm BOSCAME_Mbcaee : BOSCAME_PureMove_IntrOp_Multiclass<"mbcaee", "m">;
defm BOSCAME_Mbcbee : BOSCAME_PureMove_IntrOp_Multiclass<"mbcbee", "m">;
defm BOSCAME_Mbccee : BOSCAME_PureMove_IntrOp_Multiclass<"mbccee", "m">;

//===----------------------------------------------------------------------===//
// Transpose Operations Instantiations
//===----------------------------------------------------------------------===//
defm BOSCAME_Mtae : BOSCAME_PureMove_IntrOp_Multiclass<"mtae", "m">;
defm BOSCAME_Mtbe : BOSCAME_PureMove_IntrOp_Multiclass<"mtbe", "m">;
defm BOSCAME_Mtce : BOSCAME_PureMove_IntrOp_Multiclass<"mtce", "m">;

//===----------------------------------------------------------------------===//
// Element-Wise Intrinsic Operation Definitions (LLVM IR Translation)
//===----------------------------------------------------------------------===//
class BOSCAME_EleWise3_IntrOp<string mnemonic>
    : BOSCAME_IntrOpBase<mnemonic, 1, [0]>,
      Arguments<(ins LLVM_Type:$src1, LLVM_Type:$src2)>;

class BOSCAME_WidenEleWise3_IntrOp<string mnemonic>
    : BOSCAME_IntrOpBase<mnemonic, 1, [0], [0]>,
      Arguments<(ins LLVM_Type:$src1, LLVM_Type:$src2)>;

class BOSCAME_EleWise2_IntrOp<string mnemonic>
    : BOSCAME_IntrOpBase<mnemonic, 1, [0]>,
      Arguments<(ins LLVM_Type:$src)>;

//===----------------------------------------------------------------------===//
// Multiclasses for IntrOp Bitwidth Auto-Generation
//===----------------------------------------------------------------------===//
multiclass BOSCAME_EleWise_Int_NoWiden_IntrOp<string prefix> {
  def "Mm_IntrOp"   : BOSCAME_EleWise3_IntrOp<prefix # ".mm">;
  //def "Hbmm_IntrOp" : BOSCAME_EleWise3_IntrOp<prefix # ".hb.mm">;
  def "Bmm_IntrOp"  : BOSCAME_EleWise3_IntrOp<prefix # ".b.mm">;
  def "Hmm_IntrOp"  : BOSCAME_EleWise3_IntrOp<prefix # ".h.mm">;
  def "Wmm_IntrOp"  : BOSCAME_EleWise3_IntrOp<prefix # ".w.mm">;
  def "Dwmm_IntrOp" : BOSCAME_EleWise3_IntrOp<prefix # ".dw.mm">;
}

multiclass BOSCAME_EleWise_Float_NoWiden_IntrOp<string prefix> {
  def "Mm_IntrOp"   : BOSCAME_EleWise3_IntrOp<prefix # ".mm">;
  //def "Cfmm_IntrOp" : BOSCAME_EleWise3_IntrOp<prefix # ".cf.mm">;
  def "Hfmm_IntrOp" : BOSCAME_EleWise3_IntrOp<prefix # ".hf.mm">;
  def "Fmm_IntrOp"  : BOSCAME_EleWise3_IntrOp<prefix # ".f.mm">;
  def "Dmm_IntrOp"  : BOSCAME_EleWise3_IntrOp<prefix # ".d.mm">;
}

multiclass BOSCAME_EleWise_Int_DoubleWiden_IntrOp<string prefix> {
  def "Mm_IntrOp"   : BOSCAME_WidenEleWise3_IntrOp<prefix # ".mm">;
  //def "Hbmm_IntrOp" : BOSCAME_WidenEleWise3_IntrOp<prefix # ".hb.mm">;
  def "Bmm_IntrOp"  : BOSCAME_WidenEleWise3_IntrOp<prefix # ".b.mm">;
  def "Hmm_IntrOp"  : BOSCAME_WidenEleWise3_IntrOp<prefix # ".h.mm">;
  def "Wmm_IntrOp"  : BOSCAME_WidenEleWise3_IntrOp<prefix # ".w.mm">;
}

multiclass BOSCAME_EleWise_Float_DoubleWiden_IntrOp<string prefix> {
  def "Mm_IntrOp"   : BOSCAME_WidenEleWise3_IntrOp<prefix # ".mm">;
  //def "Cfmm_IntrOp" : BOSCAME_WidenEleWise3_IntrOp<prefix # ".cf.mm">;
  def "Hfmm_IntrOp" : BOSCAME_WidenEleWise3_IntrOp<prefix # ".hf.mm">;
  def "Fmm_IntrOp"  : BOSCAME_WidenEleWise3_IntrOp<prefix # ".f.mm">;
}

multiclass BOSCAME_EleWise_Unary_Float_IntrOp<string prefix> {
  def "M_IntrOp"   : BOSCAME_EleWise2_IntrOp<prefix # ".m">;
  //def "Cfm_IntrOp" : BOSCAME_EleWise2_IntrOp<prefix # ".cf.m">;
  def "Hfm_IntrOp" : BOSCAME_EleWise2_IntrOp<prefix # ".hf.m">;
  def "Fm_IntrOp"  : BOSCAME_EleWise2_IntrOp<prefix # ".f.m">;
  def "Dm_IntrOp"  : BOSCAME_EleWise2_IntrOp<prefix # ".d.m">;
}

//===----------------------------------------------------------------------===//
// Element-Wise IntrOp Instantiations
//===----------------------------------------------------------------------===//
// 1. ADD / SUB - No Widen
defm BOSCAME_Maddu  : BOSCAME_EleWise_Int_NoWiden_IntrOp<"maddu">;
defm BOSCAME_Msaddu : BOSCAME_EleWise_Int_NoWiden_IntrOp<"msaddu">;
defm BOSCAME_Madd   : BOSCAME_EleWise_Int_NoWiden_IntrOp<"madd">;
defm BOSCAME_Msadd  : BOSCAME_EleWise_Int_NoWiden_IntrOp<"msadd">;
defm BOSCAME_Mfadd  : BOSCAME_EleWise_Float_NoWiden_IntrOp<"mfadd">;

defm BOSCAME_Msubu  : BOSCAME_EleWise_Int_NoWiden_IntrOp<"msubu">;
defm BOSCAME_Mssubu : BOSCAME_EleWise_Int_NoWiden_IntrOp<"mssubu">;
defm BOSCAME_Msub   : BOSCAME_EleWise_Int_NoWiden_IntrOp<"msub">;
defm BOSCAME_Mssub  : BOSCAME_EleWise_Int_NoWiden_IntrOp<"mssub">;
defm BOSCAME_Mfsub  : BOSCAME_EleWise_Float_NoWiden_IntrOp<"mfsub">;
// 2. ADD / SUB - Double Widen
defm BOSCAME_Mwaddu : BOSCAME_EleWise_Int_DoubleWiden_IntrOp<"mwaddu">;
defm BOSCAME_Mwadd  : BOSCAME_EleWise_Int_DoubleWiden_IntrOp<"mwadd">;
defm BOSCAME_Mfwadd : BOSCAME_EleWise_Float_DoubleWiden_IntrOp<"mfwadd">;

defm BOSCAME_Mwsubu : BOSCAME_EleWise_Int_DoubleWiden_IntrOp<"mwsubu">;
defm BOSCAME_Mwsub  : BOSCAME_EleWise_Int_DoubleWiden_IntrOp<"mwsub">;
defm BOSCAME_Mfwsub : BOSCAME_EleWise_Float_DoubleWiden_IntrOp<"mfwsub">;
// 3. MIN / MAX - No Widen
defm BOSCAME_Mminu : BOSCAME_EleWise_Int_NoWiden_IntrOp<"mminu">;
defm BOSCAME_Mmaxu : BOSCAME_EleWise_Int_NoWiden_IntrOp<"mmaxu">;
defm BOSCAME_Mmin  : BOSCAME_EleWise_Int_NoWiden_IntrOp<"mmin">;
defm BOSCAME_Mmax  : BOSCAME_EleWise_Int_NoWiden_IntrOp<"mmax">;
defm BOSCAME_Mfmin : BOSCAME_EleWise_Float_NoWiden_IntrOp<"mfmin">;
defm BOSCAME_Mfmax : BOSCAME_EleWise_Float_NoWiden_IntrOp<"mfmax">;
// 4. MUL / MULH / DIV - No Widen
defm BOSCAME_Msmulu  : BOSCAME_EleWise_Int_NoWiden_IntrOp<"msmulu">;
defm BOSCAME_Mmul    : BOSCAME_EleWise_Int_NoWiden_IntrOp<"mmul">;
defm BOSCAME_Msmul   : BOSCAME_EleWise_Int_NoWiden_IntrOp<"msmul">;
defm BOSCAME_Mfmul   : BOSCAME_EleWise_Float_NoWiden_IntrOp<"mfmul">;
defm BOSCAME_Mmulhu  : BOSCAME_EleWise_Int_NoWiden_IntrOp<"mmulhu">;
defm BOSCAME_Mmulh   : BOSCAME_EleWise_Int_NoWiden_IntrOp<"mmulh">;
defm BOSCAME_Mmulhsu : BOSCAME_EleWise_Int_NoWiden_IntrOp<"mmulhsu">;
defm BOSCAME_Msmulsu : BOSCAME_EleWise_Int_NoWiden_IntrOp<"msmulsu">;
defm BOSCAME_Mfdiv   : BOSCAME_EleWise_Float_NoWiden_IntrOp<"mfdiv">;
// 5. MUL - Double Widen
defm BOSCAME_Mwmulu  : BOSCAME_EleWise_Int_DoubleWiden_IntrOp<"mwmulu">;
defm BOSCAME_Mwmul   : BOSCAME_EleWise_Int_DoubleWiden_IntrOp<"mwmul">;
defm BOSCAME_Mwmulsu : BOSCAME_EleWise_Int_DoubleWiden_IntrOp<"mwmulsu">;
defm BOSCAME_Mfwmul  : BOSCAME_EleWise_Float_DoubleWiden_IntrOp<"mfwmul">;
// 6. LOGIC & SHIFT - No Widen (Individual & Multiclass)
def BOSCAME_MandMm_IntrOp : BOSCAME_EleWise3_IntrOp<"mand.mm">;
def BOSCAME_MorMm_IntrOp  : BOSCAME_EleWise3_IntrOp<"mor.mm">;
def BOSCAME_MxorMm_IntrOp : BOSCAME_EleWise3_IntrOp<"mxor.mm">;

defm BOSCAME_Msll : BOSCAME_EleWise_Int_NoWiden_IntrOp<"msll">;
defm BOSCAME_Msrl : BOSCAME_EleWise_Int_NoWiden_IntrOp<"msrl">;
defm BOSCAME_Msra : BOSCAME_EleWise_Int_NoWiden_IntrOp<"msra">;
// 7. SQRT - No Widen (Unary)
// The unary MLIR/LLVM intrinsic and selected hardware instruction use the
// ``.m`` suffix.
def BOSCAME_MfsqrtM_IntrOp   : BOSCAME_EleWise2_IntrOp<"mfsqrt.m">;
def BOSCAME_MfsqrtHfM_IntrOp : BOSCAME_EleWise2_IntrOp<"mfsqrt.hf.m">;
def BOSCAME_MfsqrtFM_IntrOp  : BOSCAME_EleWise2_IntrOp<"mfsqrt.f.m">;
def BOSCAME_MfsqrtDM_IntrOp  : BOSCAME_EleWise2_IntrOp<"mfsqrt.d.m">;

//===----------------------------------------------------------------------===//
// BOSCAME Intrinsic Operation Definitions (LLVM IR Translation)
// Matrix Load/Store Intrinsics Generators
//===----------------------------------------------------------------------===//
class BOSCAME_Load_IntrOp<string mnemonic>
    : BOSCAME_IntrOpBase<mnemonic, 1, [0], [1]>,
      Arguments<(ins LLVM_Type:$base, LLVM_Type:$stride)>;

class BOSCAME_Store_IntrOp<string mnemonic>
    : BOSCAME_IntrOpBase<mnemonic, 0, [], [0, 2]>,
      Arguments<(ins LLVM_Type:$src, LLVM_Type:$base, LLVM_Type:$stride)>;

// Multiclass to auto-generate 8/16/32/64 bit variants for Loads
multiclass BOSCAME_Load_IntrOp_Multiclass<string prefix> {
  def "8m_IntrOp"  : BOSCAME_Load_IntrOp<prefix # "8.m">;
  def "16m_IntrOp" : BOSCAME_Load_IntrOp<prefix # "16.m">;
  def "32m_IntrOp" : BOSCAME_Load_IntrOp<prefix # "32.m">;
  def "64m_IntrOp" : BOSCAME_Load_IntrOp<prefix # "64.m">;
}

// Multiclass to auto-generate 8/16/32/64 bit variants for Stores
multiclass BOSCAME_Store_IntrOp_Multiclass<string prefix> {
  def "8m_IntrOp"  : BOSCAME_Store_IntrOp<prefix # "8.m">;
  def "16m_IntrOp" : BOSCAME_Store_IntrOp<prefix # "16.m">;
  def "32m_IntrOp" : BOSCAME_Store_IntrOp<prefix # "32.m">;
  def "64m_IntrOp" : BOSCAME_Store_IntrOp<prefix # "64.m">;
}

//===----------------------------------------------------------------------===//
// Matrix Load Intrinsics Instantiations
//===----------------------------------------------------------------------===//
// Matrix A
defm BOSCAME_Mlae   : BOSCAME_Load_IntrOp_Multiclass<"mlae">;
defm BOSCAME_Mlate  : BOSCAME_Load_IntrOp_Multiclass<"mlate">;
defm BOSCAME_Mltre  : BOSCAME_Load_IntrOp_Multiclass<"mltre">;
// Matrix B
defm BOSCAME_Mlbe   : BOSCAME_Load_IntrOp_Multiclass<"mlbe">;
defm BOSCAME_Mlbte  : BOSCAME_Load_IntrOp_Multiclass<"mlbte">;
// Matrix C (Accumulator)
defm BOSCAME_Mlce   : BOSCAME_Load_IntrOp_Multiclass<"mlce">;
defm BOSCAME_Mlcte  : BOSCAME_Load_IntrOp_Multiclass<"mlcte">;
defm BOSCAME_Mlacce : BOSCAME_Load_IntrOp_Multiclass<"mlacce">;
//===----------------------------------------------------------------------===//
// Matrix Store Intrinsics Instantiations
//===----------------------------------------------------------------------===//
// Matrix A
defm BOSCAME_Msae   : BOSCAME_Store_IntrOp_Multiclass<"msae">;
defm BOSCAME_Msate  : BOSCAME_Store_IntrOp_Multiclass<"msate">;
defm BOSCAME_Mstre  : BOSCAME_Store_IntrOp_Multiclass<"mstre">;
// Matrix B
defm BOSCAME_Msbe   : BOSCAME_Store_IntrOp_Multiclass<"msbe">;
defm BOSCAME_Msbte  : BOSCAME_Store_IntrOp_Multiclass<"msbte">;
// Matrix C (Accumulator)
defm BOSCAME_Msce   : BOSCAME_Store_IntrOp_Multiclass<"msce">;
defm BOSCAME_Mscte  : BOSCAME_Store_IntrOp_Multiclass<"mscte">;
defm BOSCAME_Msacce : BOSCAME_Store_IntrOp_Multiclass<"msacce">;

//===----------------------------------------------------------------------===//
// Tile register matrix multiplication intrinsics (LLVM IR Translation)
//===----------------------------------------------------------------------===//
class BOSCAME_MMA_IntrOp<string mnemonic>
    : BOSCAME_IntrOpBase<mnemonic, 1, [0]>,
      Arguments<(ins LLVM_Type:$acc, LLVM_Type:$lhs, LLVM_Type:$rhs)>;

class BOSCAME_WidenMMA_IntrOp<string mnemonic>
    : BOSCAME_IntrOpBase<mnemonic, 1, [0], [1]>,
      Arguments<(ins LLVM_Type:$acc, LLVM_Type:$lhs, LLVM_Type:$rhs)>;

//===--------------------------------------------------------------------===//
// 1. No-widen matrix multiply-accumulate
//===--------------------------------------------------------------------===//
// int
def BOSCAME_MmaMm_IntrOp   : BOSCAME_MMA_IntrOp<"mma.mm">;
def BOSCAME_MmaHmm_IntrOp  : BOSCAME_MMA_IntrOp<"mma.h.mm">;
def BOSCAME_MmaWmm_IntrOp  : BOSCAME_MMA_IntrOp<"mma.w.mm">;
def BOSCAME_MmaDwmm_IntrOp : BOSCAME_MMA_IntrOp<"mma.dw.mm">;
// int, saturated
def BOSCAME_MsmaMm_IntrOp   : BOSCAME_MMA_IntrOp<"msma.mm">;
def BOSCAME_MsmaHmm_IntrOp  : BOSCAME_MMA_IntrOp<"msma.h.mm">;
def BOSCAME_MsmaWmm_IntrOp  : BOSCAME_MMA_IntrOp<"msma.w.mm">;
def BOSCAME_MsmaDwmm_IntrOp : BOSCAME_MMA_IntrOp<"msma.dw.mm">;
// uint
def BOSCAME_MmauMm_IntrOp   : BOSCAME_MMA_IntrOp<"mmau.mm">;
def BOSCAME_MmauHmm_IntrOp  : BOSCAME_MMA_IntrOp<"mmau.h.mm">;
def BOSCAME_MmauWmm_IntrOp  : BOSCAME_MMA_IntrOp<"mmau.w.mm">;
def BOSCAME_MmauDwmm_IntrOp : BOSCAME_MMA_IntrOp<"mmau.dw.mm">;
// uint, saturated
def BOSCAME_MsmauMm_IntrOp   : BOSCAME_MMA_IntrOp<"msmau.mm">;
def BOSCAME_MsmauHmm_IntrOp  : BOSCAME_MMA_IntrOp<"msmau.h.mm">;
def BOSCAME_MsmauWmm_IntrOp  : BOSCAME_MMA_IntrOp<"msmau.w.mm">;
def BOSCAME_MsmauDwmm_IntrOp : BOSCAME_MMA_IntrOp<"msmau.dw.mm">;
// float point
def BOSCAME_MfmaMm_IntrOp   : BOSCAME_MMA_IntrOp<"mfma.mm">;
def BOSCAME_MfmaHfmm_IntrOp : BOSCAME_MMA_IntrOp<"mfma.hf.mm">;
def BOSCAME_MfmaFmm_IntrOp  : BOSCAME_MMA_IntrOp<"mfma.f.mm">;
def BOSCAME_MfmaDmm_IntrOp  : BOSCAME_MMA_IntrOp<"mfma.d.mm">;

//===--------------------------------------------------------------------===//
// 2. Double-widen matrix multiply-accumulate
//===--------------------------------------------------------------------===//
// uint
def BOSCAME_MwmauMm_IntrOp   : BOSCAME_WidenMMA_IntrOp<"mwmau.mm">;
def BOSCAME_MwmauHmm_IntrOp  : BOSCAME_WidenMMA_IntrOp<"mwmau.h.mm">;
def BOSCAME_MwmauWmm_IntrOp  : BOSCAME_WidenMMA_IntrOp<"mwmau.w.mm">;
// uint, saturated
def BOSCAME_MswmauMm_IntrOp  : BOSCAME_WidenMMA_IntrOp<"mswmau.mm">;
def BOSCAME_MswmauHmm_IntrOp : BOSCAME_WidenMMA_IntrOp<"mswmau.h.mm">;
def BOSCAME_MswmauWmm_IntrOp : BOSCAME_WidenMMA_IntrOp<"mswmau.w.mm">;
// int
def BOSCAME_MwmaMm_IntrOp   : BOSCAME_WidenMMA_IntrOp<"mwma.mm">;
def BOSCAME_MwmaHmm_IntrOp  : BOSCAME_WidenMMA_IntrOp<"mwma.h.mm">;
def BOSCAME_MwmaWmm_IntrOp  : BOSCAME_WidenMMA_IntrOp<"mwma.w.mm">;
// int, saturated
def BOSCAME_MswmaMm_IntrOp   : BOSCAME_WidenMMA_IntrOp<"mswma.mm">;
def BOSCAME_MswmaHmm_IntrOp  : BOSCAME_WidenMMA_IntrOp<"mswma.h.mm">;
def BOSCAME_MswmaWmm_IntrOp  : BOSCAME_WidenMMA_IntrOp<"mswma.w.mm">;
// float point
def BOSCAME_MfwmaMm_IntrOp   : BOSCAME_WidenMMA_IntrOp<"mfwma.mm">;
//def BOSCAME_MfwmaCfmm_IntrOp : BOSCAME_WidenMMA_IntrOp<"mfwma.cf.mm">;
def BOSCAME_MfwmaHfmm_IntrOp : BOSCAME_WidenMMA_IntrOp<"mfwma.hf.mm">;
def BOSCAME_MfwmaFmm_IntrOp  : BOSCAME_WidenMMA_IntrOp<"mfwma.f.mm">;

//===--------------------------------------------------------------------===//
// 3. Quad-widen matrix multiply-accumulate
//===--------------------------------------------------------------------===//
// uint
def BOSCAME_MqmauMm_IntrOp  : BOSCAME_WidenMMA_IntrOp<"mqmau.mm">;
def BOSCAME_MqmauBmm_IntrOp : BOSCAME_WidenMMA_IntrOp<"mqmau.b.mm">;
// uint, saturated
def BOSCAME_MsqmauMm_IntrOp  : BOSCAME_WidenMMA_IntrOp<"msqmau.mm">;
def BOSCAME_MsqmauBmm_IntrOp : BOSCAME_WidenMMA_IntrOp<"msqmau.b.mm">;
// int
def BOSCAME_MqmaMm_IntrOp  : BOSCAME_WidenMMA_IntrOp<"mqma.mm">;
def BOSCAME_MqmaBmm_IntrOp : BOSCAME_WidenMMA_IntrOp<"mqma.b.mm">;
// int, saturated
def BOSCAME_MsqmaMm_IntrOp  : BOSCAME_WidenMMA_IntrOp<"msqma.mm">;
def BOSCAME_MsqmaBmm_IntrOp : BOSCAME_WidenMMA_IntrOp<"msqma.b.mm">;
// float point
def BOSCAME_MfqmaMm_IntrOp   : BOSCAME_WidenMMA_IntrOp<"mfqma.mm">;
//def BOSCAME_MfqmaCfmm_IntrOp : BOSCAME_WidenMMA_IntrOp<"mfqma.cf.mm">;

//===--------------------------------------------------------------------===//
// 4. Oct-widen matrix multiply-accumulate
//===--------------------------------------------------------------------===//
// uint
def BOSCAME_MomauMm_IntrOp   : BOSCAME_WidenMMA_IntrOp<"momau.mm">;
//def BOSCAME_MomauHbmm_IntrOp : BOSCAME_WidenMMA_IntrOp<"momau.hb.mm">;
// uint, saturated
def BOSCAME_MsomauMm_IntrOp   : BOSCAME_WidenMMA_IntrOp<"msomau.mm">;
//def BOSCAME_MsomauHbmm_IntrOp : BOSCAME_WidenMMA_IntrOp<"msomau.hb.mm">;
// int
def BOSCAME_MomaMm_IntrOp   : BOSCAME_WidenMMA_IntrOp<"moma.mm">;
//def BOSCAME_MomaHbmm_IntrOp : BOSCAME_WidenMMA_IntrOp<"moma.hb.mm">;
// int, saturated
def BOSCAME_MsomaMm_IntrOp   : BOSCAME_WidenMMA_IntrOp<"msoma.mm">;
//def BOSCAME_MsomaHbmm_IntrOp : BOSCAME_WidenMMA_IntrOp<"msoma.hb.mm">;

#endif // BOSCAME_DIALECT
