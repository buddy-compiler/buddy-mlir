//====-- BOSCAME.td - BOSCAME dialect operation definitions --- tablegen --===//
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
//
// This file defines the AME dialect for RISC-V Matrix Extension.
//
//===----------------------------------------------------------------------===//

#ifndef BOSCAME_DIALECT
#define BOSCAME_DIALECT

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
  let description = [{
    The BOSCAME dialect provides operations for the RISC-V Matrix Extension (RVA23 profile).
    BOSCAME provides matrix multiply-accumulate instructions with tile-based computation,
    supporting various data types and widening modes for AI/ML workloads.

    Key features:
    - Tile-based matrix registers with configurable dimensions
    - Support for int4/int8/int16/int32/int64 integer types
    - Support for fp8/fp16/fp32/fp64 floating-point types
    - Widening matrix multiplication (2x, 4x, 8x output width)
    - Saturating arithmetic options
  }];
}

//===----------------------------------------------------------------------===//
// BOSCAME Operation Base Class
//===----------------------------------------------------------------------===//

class BOSCAME_Op<string mnemonic, list<Trait> traits = []> :
  Op<BOSCAME_Dialect, mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// Matrix Configuration Operations (with immediate)
//===----------------------------------------------------------------------===//

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
// Matrix Zero Operations
//===----------------------------------------------------------------------===//

def MzeroOp : BOSCAME_Op<"mzero"> {
  let summary = "Zero out accumulation matrix tile";
  let description = [{
    Sets all elements in the specified accumulation matrix tile to zero.
  }];
  let arguments = (ins I64Attr:$md);
  let assemblyFormat = "$md attr-dict";
}

//===----------------------------------------------------------------------===//
// Matrix Load Instructions
// Load matrix tiles from memory to tile registers
//===----------------------------------------------------------------------===//

// Load left matrix A (mtilem x mtilek)
def Mlae32mOp : BOSCAME_Op<"mlae32.m"> {
  let summary = "Load 32-bit left matrix tile A";
  let description = [{
    Load a left matrix tile from memory to tile register.
    Shape: mtilem x mtilek
    - md: destination tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$md,
    AnyMemRef:$base,
    I64:$stride
  );
  let assemblyFormat = "$md `,` $base `,` $stride attr-dict `:` type($base)";
}

// Load right matrix B (mtilek x mtilen)
def Mlbe32mOp : BOSCAME_Op<"mlbe32.m"> {
  let summary = "Load 32-bit right matrix tile B";
  let description = [{
    Load a right matrix tile from memory to tile register.
    Shape: mtilek x mtilen
    - md: destination tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$md,
    AnyMemRef:$base,
    I64:$stride
  );
  let assemblyFormat = "$md `,` $base `,` $stride attr-dict `:` type($base)";
}

//===----------------------------------------------------------------------===//
// Matrix Store Instructions
// Store matrix tiles from tile registers to memory
//===----------------------------------------------------------------------===//

// Store output/accumulator matrix C (mtilem x mtilen)
def Msce32mOp : BOSCAME_Op<"msce32.m"> {
  let summary = "Store 32-bit output matrix tile C";
  let description = [{
    Store an output/accumulator matrix tile to memory.
    Shape: mtilem x mtilen
    - ms3: source tile register index
    - base: base address pointer
    - stride: row byte stride
  }];
  let arguments = (ins
    I64Attr:$ms3,
    AnyMemRef:$base,
    I64:$stride
  );
  let assemblyFormat = "$ms3 `,` $base `,` $stride attr-dict `:` type($base)";
}

//===----------------------------------------------------------------------===//
// Signed Integer Matrix Multiplication Operations (Tile Register Version)
// These operate on tile registers, not memory
//===----------------------------------------------------------------------===//

def MmaWmmOp : BOSCAME_Op<"mma.w.mm"> {
  let summary = "Signed int32 tile matrix multiply-accumulate";
  let description = [{
    Performs signed int32 matrix multiply-accumulate on tile registers:
      md = md + ms1 x ms2
    where ms1, ms2 are tile registers, and md is accumulation register.

    - md: accumulation register index (0-7)
    - ms1: left tile register index (0-7)
    - ms2: right tile register index (0-7)
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms1,
    I64Attr:$ms2
  );
  let assemblyFormat = "$md `,` $ms1 `,` $ms2 attr-dict";
}

//===----------------------------------------------------------------------===//
// BOSCAME Intrinsic Operation Definitions (for LLVM IR translation)
// These map directly to LLVM intrinsics defined in IntrinsicsRISCVBuddyBOSCExt.td
//===----------------------------------------------------------------------===//

// Base class for BOSCAME intrinsic ops
// Note: enumName must match the intrinsic name pattern: riscv_buddy_<name>
class BOSCAME_IntrOpBase<string mnemonic, int numRes = 0, list<Trait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/BOSCAME_Dialect,
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_buddy_bosc_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[],
                  /*list<int> overloadedOperands=*/[],
                  /*list<Trait> traits=*/traits,
                  /*int numResults=*/numRes>;

// Configuration intrinsics (with immediate value)
def BOSCAME_Msettilemi_IntrOp : BOSCAME_IntrOpBase<"msettilemi", 1>,
  Arguments<(ins LLVM_Type:$tilem)>;

def BOSCAME_Msettileni_IntrOp : BOSCAME_IntrOpBase<"msettileni", 1>,
  Arguments<(ins LLVM_Type:$tilen)>;

def BOSCAME_Msettileki_IntrOp : BOSCAME_IntrOpBase<"msettileki", 1>,
  Arguments<(ins LLVM_Type:$tilek)>;

// Matrix zero intrinsic
def BOSCAME_Mzero_IntrOp : BOSCAME_IntrOpBase<"mzero">,
  Arguments<(ins LLVM_Type:$md)>;

// Matrix load intrinsics (to tile registers)
def BOSCAME_Mlae32m_IntrOp : BOSCAME_IntrOpBase<"mlae32.m">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$base, LLVM_Type:$stride)>;

def BOSCAME_Mlbe32m_IntrOp : BOSCAME_IntrOpBase<"mlbe32.m">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$base, LLVM_Type:$stride)>;

// Matrix store intrinsics (from tile registers)
def BOSCAME_Msce32m_IntrOp : BOSCAME_IntrOpBase<"msce32.m">,
  Arguments<(ins LLVM_Type:$ms3, LLVM_Type:$base, LLVM_Type:$stride)>;

// Tile register matrix multiplication intrinsics
def BOSCAME_MmaWmm_IntrOp : BOSCAME_IntrOpBase<"mma.w.mm">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

#endif // BOSCAME_DIALECT
