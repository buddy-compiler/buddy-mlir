//===-- ZAMES.td - ZAMES dialect definitions ------------------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ZAMES_DIALECT
#define ZAMES_DIALECT

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ZAMES_Dialect : Dialect {
  let name = "zames";
  let cppNamespace = "::buddy::zames";
  let summary = "Typed tile IR for the RISC-V matrix extension";
  let description = [{
    ZAMES models matrix tiles as SSA values with explicit kind, shape, and
    element information. Physical tile configuration is introduced by lowering,
    rather than exposed through configuration operations in this dialect.
  }];
  let extraClassDeclaration = [{
    void registerTypes();
  }];
  let useDefaultTypePrinterParser = 1;
}

class ZAMES_Op<string mnemonic, list<Trait> traits = []> :
    Op<ZAMES_Dialect, mnemonic, traits>;

def ZAMES_TileType : TypeDef<ZAMES_Dialect, "Tile", [ValueSemantics]> {
  let mnemonic = "tile";
  let summary = "Typed matrix tile";
  let description = [{
    `!zames.tile<kind, MxN, elem>` has kind `a`, `b`, or `c`, two positive
    or dynamic (`?`) extents, and a ZAMES element token. Hardware tile shapes
    are A=MxK, B=NxK, and C=MxN. Historical packed tokens remain representable;
    a target lowering may support only a subset of element tokens.
  }];
  let parameters = (ins
    StringRefParameter<"tile kind">:$kind,
    ArrayRefParameter<"int64_t">:$shape,
    StringRefParameter<"element token">:$element
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    bool isA() const { return getKind() == "a"; }
    bool isB() const { return getKind() == "b"; }
    bool isC() const { return getKind() == "c"; }
  }];
}

def ZAMES_SyncType : TypeDef<ZAMES_Dialect, "Sync", [ValueSemantics]> {
  let mnemonic = "sync";
  let summary = "Matrix sync register SSA handle";
  let description = [{
    Reset produces a sync handle and release produces its successor SSA value.
    Acquire consumes the handle and an index target. Physical register IDs are
    deliberately not part of this interface.
  }];
}

def LoadOp : ZAMES_Op<"load"> {
  let summary = "Load a matrix tile from a rank-2 memory view";
  let arguments = (ins
    Arg<AnyMemRef, "tile view", [MemRead]>:$view,
    Variadic<I64>:$dynamic_sizes,
    DefaultValuedAttr<BoolAttr, "false">:$transpose
  );
  let results = (outs ZAMES_TileType:$result);
  let assemblyFormat = "$view (`[` $dynamic_sizes^ `]`)? prop-dict attr-dict"
      " `:` type($view) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def StoreOp : ZAMES_Op<"store"> {
  let summary = "Store a matrix tile to a rank-2 memory view";
  let arguments = (ins
    ZAMES_TileType:$tile,
    Arg<AnyMemRef, "tile view", [MemWrite]>:$view,
    Variadic<I64>:$dynamic_sizes,
    DefaultValuedAttr<BoolAttr, "false">:$transpose
  );
  let assemblyFormat = "$tile `,` $view (`[` $dynamic_sizes^ `]`)? prop-dict attr-dict"
      " `:` qualified(type($tile)) `,` type($view)";
  let hasVerifier = 1;
}

def MmaOp : ZAMES_Op<"mma", [AllTypesMatch<["c", "result"]>]> {
  let summary = "Matrix multiply-accumulate on A=MxK, B=NxK, C=MxN tiles";
  let description = [{
    Dynamic size operands are ordered M, N, K and are present only for dynamic
    dimensions of A.M, B.N, and A.K. The result type is the type of C.
  }];
  let arguments = (ins
    ZAMES_TileType:$a, ZAMES_TileType:$b, ZAMES_TileType:$c,
    Variadic<I64>:$dynamic_sizes
  );
  let results = (outs ZAMES_TileType:$result);
  let assemblyFormat = "$a `,` $b `,` $c (`[` $dynamic_sizes^ `]`)? attr-dict"
      " `:` qualified(type($a)) `,` qualified(type($b)) `,` qualified(type($c))";
  let hasVerifier = 1;
}

def MzeroOp : ZAMES_Op<"mzero"> {
  let summary = "Clear a whole physical tile register";
  let description = [{
    Dynamic sizes describe the logical result type. The physical whole-register
    clear does not require tile-dimension configuration.
  }];
  let arguments = (ins Variadic<I64>:$dynamic_sizes);
  let results = (outs ZAMES_TileType:$result);
  let assemblyFormat =
      "(`[` $dynamic_sizes^ `]`)? attr-dict `:` qualified(type($result))";
  let hasVerifier = 1;
}

def MsyncRegResetOp : ZAMES_Op<"msyncregreset",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Reset a matrix sync register and produce its SSA handle";
  let results = (outs ZAMES_SyncType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def MfenceOp : ZAMES_Op<"mfence", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Matrix memory fence";
  let assemblyFormat = "attr-dict";
}

def MreleaseOp : ZAMES_Op<"mrelease",
    [MemoryEffects<[MemRead, MemWrite]>, SameOperandsAndResultType]> {
  let summary = "Release a sync handle and produce its SSA successor";
  let arguments = (ins ZAMES_SyncType:$sync);
  let results = (outs ZAMES_SyncType:$result);
  let assemblyFormat = "$sync attr-dict `:` type($sync)";
}

def MacquireOp : ZAMES_Op<"macquire", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Acquire a sync handle at the specified index target";
  let arguments = (ins ZAMES_SyncType:$sync, Index:$target);
  let assemblyFormat = "$sync `,` $target attr-dict `:` type($sync) `,` type($target)";
}

#endif // ZAMES_DIALECT
