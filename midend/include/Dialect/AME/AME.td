//====------ AME.td - AME dialect operation definitions --- tablegen ------===//
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

#ifndef AME_DIALECT
#define AME_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

//===----------------------------------------------------------------------===//
// AME Dialect Definition
//===----------------------------------------------------------------------===//

def AME_Dialect : Dialect {
  let name = "ame";
  let cppNamespace = "::buddy::ame";
  let description = [{
    The AME dialect provides operations for the RISC-V Matrix Extension (RVA23 profile).
    AME provides matrix multiply-accumulate instructions with tile-based computation,
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
// AME Operation Base Class
//===----------------------------------------------------------------------===//

class AME_Op<string mnemonic, list<Trait> traits = []> :
  Op<AME_Dialect, mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// Matrix Configuration Operations
//===----------------------------------------------------------------------===//

def MSettypeOp : AME_Op<"msettype"> {
  let summary = "Set matrix type configuration";
  let description = [{
    Sets the matrix type configuration from a GPR register.
    Returns the previous mtype value.
  }];
  let arguments = (ins I64:$mtype);
  let results = (outs I64:$result);
  let assemblyFormat = "$mtype attr-dict";
}

def MSettypeiOp : AME_Op<"msettypei"> {
  let summary = "Set matrix type configuration with immediate";
  let description = [{
    Sets the matrix type configuration from an immediate value.
    Returns the previous mtype value.
  }];
  let arguments = (ins I64Attr:$mtype);
  let results = (outs I64:$result);
  let assemblyFormat = "$mtype attr-dict";
}

def MSettilemOp : AME_Op<"msettilem"> {
  let summary = "Set tile M dimension";
  let description = [{
    Sets the tile M dimension (number of rows).
    Returns the actual mtilem value set.
  }];
  let arguments = (ins I64:$tilem);
  let results = (outs I64:$result);
  let assemblyFormat = "$tilem attr-dict";
}

def MSettilenOp : AME_Op<"msettilen"> {
  let summary = "Set tile N dimension";
  let description = [{
    Sets the tile N dimension (number of columns).
    Returns the actual mtilen value set.
  }];
  let arguments = (ins I64:$tilen);
  let results = (outs I64:$result);
  let assemblyFormat = "$tilen attr-dict";
}

def MSettilekOp : AME_Op<"msettilek"> {
  let summary = "Set tile K dimension";
  let description = [{
    Sets the tile K dimension (inner dimension for matrix multiply).
    Returns the actual mtilek value set.
  }];
  let arguments = (ins I64:$tilek);
  let results = (outs I64:$result);
  let assemblyFormat = "$tilek attr-dict";
}

//===----------------------------------------------------------------------===//
// Matrix Configuration Operations (with immediate)
//===----------------------------------------------------------------------===//

def MSettilemiOp : AME_Op<"msettilemi"> {
  let summary = "Set tile M dimension with immediate";
  let description = [{
    Sets the tile M dimension (number of rows) from an immediate value.
    mtilem = min(tilem, TMMAX)
  }];
  let arguments = (ins I64Attr:$tilem);
  let assemblyFormat = "$tilem attr-dict";
}

def MSettileniOp : AME_Op<"msettileni"> {
  let summary = "Set tile N dimension with immediate";
  let description = [{
    Sets the tile N dimension (number of columns) from an immediate value.
    mtilen = min(tilen, TNMAX)
  }];
  let arguments = (ins I64Attr:$tilen);
  let assemblyFormat = "$tilen attr-dict";
}

def MSettilekiOp : AME_Op<"msettileki"> {
  let summary = "Set tile K dimension with immediate";
  let description = [{
    Sets the tile K dimension (inner dimension for matrix multiply) from an immediate value.
    mtilek = min(tilek, TKMAX)
  }];
  let arguments = (ins I64Attr:$tilek);
  let assemblyFormat = "$tilek attr-dict";
}

//===----------------------------------------------------------------------===//
// Matrix Zero Operations
//===----------------------------------------------------------------------===//

def MzeroOp : AME_Op<"mzero"> {
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
def Mlae32mOp : AME_Op<"mlae32.m"> {
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

def Mlae64mOp : AME_Op<"mlae64.m"> {
  let summary = "Load 64-bit left matrix tile A";
  let description = [{
    Load a 64-bit left matrix tile from memory to tile register.
    Shape: mtilem x mtilek
  }];
  let arguments = (ins
    I64Attr:$md,
    AnyMemRef:$base,
    I64:$stride
  );
  let assemblyFormat = "$md `,` $base `,` $stride attr-dict `:` type($base)";
}

// Load right matrix B (mtilek x mtilen)
def Mlbe32mOp : AME_Op<"mlbe32.m"> {
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

def Mlbe64mOp : AME_Op<"mlbe64.m"> {
  let summary = "Load 64-bit right matrix tile B";
  let description = [{
    Load a 64-bit right matrix tile from memory to tile register.
    Shape: mtilek x mtilen
  }];
  let arguments = (ins
    I64Attr:$md,
    AnyMemRef:$base,
    I64:$stride
  );
  let assemblyFormat = "$md `,` $base `,` $stride attr-dict `:` type($base)";
}

// Load output/accumulator matrix C (mtilem x mtilen)
def Mlce32mOp : AME_Op<"mlce32.m"> {
  let summary = "Load 32-bit output matrix tile C";
  let description = [{
    Load an output/accumulator matrix tile from memory.
    Shape: mtilem x mtilen
  }];
  let arguments = (ins
    I64Attr:$md,
    AnyMemRef:$base,
    I64:$stride
  );
  let assemblyFormat = "$md `,` $base `,` $stride attr-dict `:` type($base)";
}

def Mlce64mOp : AME_Op<"mlce64.m"> {
  let summary = "Load 64-bit output matrix tile C";
  let description = [{
    Load a 64-bit output/accumulator matrix tile from memory.
    Shape: mtilem x mtilen
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
def Msce32mOp : AME_Op<"msce32.m"> {
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

def Msce64mOp : AME_Op<"msce64.m"> {
  let summary = "Store 64-bit output matrix tile C";
  let description = [{
    Store a 64-bit output/accumulator matrix tile to memory.
    Shape: mtilem x mtilen
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

def MmaWmmTileOp : AME_Op<"mma.w.mm.tile"> {
  let summary = "Signed int32 tile matrix multiply-accumulate";
  let description = [{
    Performs signed int32 matrix multiply-accumulate on tile registers:
      md = md + ms1 × ms2
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

def MmaDwmmTileOp : AME_Op<"mma.dw.mm.tile"> {
  let summary = "Signed int64 tile matrix multiply-accumulate";
  let description = [{
    Performs signed int64 matrix multiply-accumulate on tile registers:
      md = md + ms1 × ms2
    where ms1, ms2 are tile registers, and md is accumulation register.
  }];
  let arguments = (ins
    I64Attr:$md,
    I64Attr:$ms1,
    I64Attr:$ms2
  );
  let assemblyFormat = "$md `,` $ms1 `,` $ms2 attr-dict";
}

//===----------------------------------------------------------------------===//
// High-level Matrix Multiplication Operations (MemRef Version)
// These are high-level ops that will be lowered to load/compute/store sequence
//===----------------------------------------------------------------------===//

def MqmaBmmOp : AME_Op<"mqma.b.mm"> {
  let summary = "Signed int8 matrix multiply-accumulate with quad-widen output";
  let description = [{
    Performs signed int8 matrix multiply-accumulate with quad-widen output:
      md = md + ms1 × ms2
    where ms1 and ms2 are int8 matrices, and md is int32 accumulator.
    
    Output is 4x widened: int8 × int8 -> int32.
    This is the primary instruction for int8 AI workloads.
  }];
  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$md,
    MemRefRankOf<[I8], [2]>:$ms1,
    MemRefRankOf<[I8], [2]>:$ms2
  );
  let assemblyFormat = "$md `,` $ms1 `,` $ms2 attr-dict `:` type($md) `,` type($ms1) `,` type($ms2)";
}

def MmaHmmOp : AME_Op<"mma.h.mm"> {
  let summary = "Signed int16 matrix multiply-accumulate";
  let description = [{
    Performs signed int16 matrix multiply-accumulate:
      md = md + ms1 × ms2
    where ms1, ms2, and md are all int16 matrices.
  }];
  let arguments = (ins
    MemRefRankOf<[I16], [2]>:$md,
    MemRefRankOf<[I16], [2]>:$ms1,
    MemRefRankOf<[I16], [2]>:$ms2
  );
  let assemblyFormat = "$md `,` $ms1 `,` $ms2 attr-dict `:` type($md) `,` type($ms1) `,` type($ms2)";
}

def MmaWmmOp : AME_Op<"mma.w.mm"> {
  let summary = "Signed int32 matrix multiply-accumulate";
  let description = [{
    Performs signed int32 matrix multiply-accumulate:
      md = md + ms1 × ms2
    where ms1, ms2, and md are all int32 matrices.
  }];
  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$md,
    MemRefRankOf<[I32], [2]>:$ms1,
    MemRefRankOf<[I32], [2]>:$ms2
  );
  let assemblyFormat = "$md `,` $ms1 `,` $ms2 attr-dict `:` type($md) `,` type($ms1) `,` type($ms2)";
}

def MmaDwmmOp : AME_Op<"mma.dw.mm"> {
  let summary = "Signed int64 matrix multiply-accumulate";
  let description = [{
    Performs signed int64 matrix multiply-accumulate:
      md = md + ms1 × ms2
    where ms1, ms2, and md are all int64 matrices.
    
    This is useful for high-precision matrix computations.
  }];
  let arguments = (ins
    MemRefRankOf<[I64], [2]>:$md,
    MemRefRankOf<[I64], [2]>:$ms1,
    MemRefRankOf<[I64], [2]>:$ms2
  );
  let assemblyFormat = "$md `,` $ms1 `,` $ms2 attr-dict `:` type($md) `,` type($ms1) `,` type($ms2)";
}

def MwmaHmmOp : AME_Op<"mwma.h.mm"> {
  let summary = "Signed int16 matrix multiply-accumulate with double-widen output";
  let description = [{
    Performs signed int16 matrix multiply-accumulate with double-widen output:
      md = md + ms1 × ms2
    where ms1 and ms2 are int16 matrices, and md is int32 accumulator.
  }];
  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$md,
    MemRefRankOf<[I16], [2]>:$ms1,
    MemRefRankOf<[I16], [2]>:$ms2
  );
  let assemblyFormat = "$md `,` $ms1 `,` $ms2 attr-dict `:` type($md) `,` type($ms1) `,` type($ms2)";
}

//===----------------------------------------------------------------------===//
// Unsigned Integer Matrix Multiplication Operations
//===----------------------------------------------------------------------===//

def MqmauBmmOp : AME_Op<"mqmau.b.mm"> {
  let summary = "Unsigned int8 matrix multiply-accumulate with quad-widen output";
  let description = [{
    Performs unsigned int8 matrix multiply-accumulate with quad-widen output:
      md = md + ms1 × ms2
    where ms1 and ms2 are uint8 matrices, and md is uint32 accumulator.
  }];
  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$md,
    MemRefRankOf<[I8], [2]>:$ms1,
    MemRefRankOf<[I8], [2]>:$ms2
  );
  let assemblyFormat = "$md `,` $ms1 `,` $ms2 attr-dict `:` type($md) `,` type($ms1) `,` type($ms2)";
}

def MmauHmmOp : AME_Op<"mmau.h.mm"> {
  let summary = "Unsigned int16 matrix multiply-accumulate";
  let description = [{
    Performs unsigned int16 matrix multiply-accumulate:
      md = md + ms1 × ms2
    where ms1, ms2, and md are all uint16 matrices.
  }];
  let arguments = (ins
    MemRefRankOf<[I16], [2]>:$md,
    MemRefRankOf<[I16], [2]>:$ms1,
    MemRefRankOf<[I16], [2]>:$ms2
  );
  let assemblyFormat = "$md `,` $ms1 `,` $ms2 attr-dict `:` type($md) `,` type($ms1) `,` type($ms2)";
}

//===----------------------------------------------------------------------===//
// Floating-Point Matrix Multiplication Operations
//===----------------------------------------------------------------------===//

def MfmaFmmOp : AME_Op<"mfma.f.mm"> {
  let summary = "FP32 matrix multiply-accumulate";
  let description = [{
    Performs FP32 matrix multiply-accumulate:
      md = md + ms1 × ms2
    where ms1, ms2, and md are all fp32 matrices.
  }];
  let arguments = (ins
    MemRefRankOf<[F32], [2]>:$md,
    MemRefRankOf<[F32], [2]>:$ms1,
    MemRefRankOf<[F32], [2]>:$ms2
  );
  let assemblyFormat = "$md `,` $ms1 `,` $ms2 attr-dict `:` type($md) `,` type($ms1) `,` type($ms2)";
}

def MfmaHfmmOp : AME_Op<"mfma.hf.mm"> {
  let summary = "FP16 matrix multiply-accumulate";
  let description = [{
    Performs FP16 matrix multiply-accumulate:
      md = md + ms1 × ms2
    where ms1, ms2, and md are all fp16 matrices.
  }];
  let arguments = (ins
    MemRefRankOf<[F16], [2]>:$md,
    MemRefRankOf<[F16], [2]>:$ms1,
    MemRefRankOf<[F16], [2]>:$ms2
  );
  let assemblyFormat = "$md `,` $ms1 `,` $ms2 attr-dict `:` type($md) `,` type($ms1) `,` type($ms2)";
}

def MfwmaHfmmOp : AME_Op<"mfwma.hf.mm"> {
  let summary = "FP16 matrix multiply-accumulate with double-widen output";
  let description = [{
    Performs FP16 matrix multiply-accumulate with FP32 accumulator:
      md = md + ms1 × ms2
    where ms1 and ms2 are fp16 matrices, and md is fp32 accumulator.
  }];
  let arguments = (ins
    MemRefRankOf<[F32], [2]>:$md,
    MemRefRankOf<[F16], [2]>:$ms1,
    MemRefRankOf<[F16], [2]>:$ms2
  );
  let assemblyFormat = "$md `,` $ms1 `,` $ms2 attr-dict `:` type($md) `,` type($ms1) `,` type($ms2)";
}

//===----------------------------------------------------------------------===//
// AME Intrinsic Operation Definitions (for LLVM IR translation)
// These map directly to LLVM intrinsics defined in IntrinsicsRISCVBuddyExt.td
//===----------------------------------------------------------------------===//

// Base class for AME intrinsic ops
// Note: enumName must match the intrinsic name pattern: riscv_buddy_<name>
class AME_IntrOpBase<string mnemonic, list<Trait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/AME_Dialect,
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_buddy_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[],
                  /*list<int> overloadedOperands=*/[],
                  /*list<Trait> traits=*/traits,
                  /*int numResults=*/0>;

// Configuration intrinsics (with immediate value)
def AME_Msettilemi_IntrOp : AME_IntrOpBase<"msettilemi">,
  Arguments<(ins LLVM_Type:$tilem)>;

def AME_Msettileni_IntrOp : AME_IntrOpBase<"msettileni">,
  Arguments<(ins LLVM_Type:$tilen)>;

def AME_Msettileki_IntrOp : AME_IntrOpBase<"msettileki">,
  Arguments<(ins LLVM_Type:$tilek)>;

// Matrix zero intrinsic
def AME_Mzero_IntrOp : AME_IntrOpBase<"mzero">,
  Arguments<(ins LLVM_Type:$md)>;

// Matrix load intrinsics (to tile registers)
def AME_Mlae32m_IntrOp : AME_IntrOpBase<"mlae32.m">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$base, LLVM_Type:$stride)>;

def AME_Mlae64m_IntrOp : AME_IntrOpBase<"mlae64.m">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$base, LLVM_Type:$stride)>;

def AME_Mlbe32m_IntrOp : AME_IntrOpBase<"mlbe32.m">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$base, LLVM_Type:$stride)>;

def AME_Mlbe64m_IntrOp : AME_IntrOpBase<"mlbe64.m">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$base, LLVM_Type:$stride)>;

def AME_Mlce32m_IntrOp : AME_IntrOpBase<"mlce32.m">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$base, LLVM_Type:$stride)>;

def AME_Mlce64m_IntrOp : AME_IntrOpBase<"mlce64.m">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$base, LLVM_Type:$stride)>;

// Matrix store intrinsics (from tile registers)
def AME_Msce32m_IntrOp : AME_IntrOpBase<"msce32.m">,
  Arguments<(ins LLVM_Type:$ms3, LLVM_Type:$base, LLVM_Type:$stride)>;

def AME_Msce64m_IntrOp : AME_IntrOpBase<"msce64.m">,
  Arguments<(ins LLVM_Type:$ms3, LLVM_Type:$base, LLVM_Type:$stride)>;

// Tile register matrix multiplication intrinsics
def AME_MmaWmmTile_IntrOp : AME_IntrOpBase<"mma.w.mm.tile">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

def AME_MmaDwmmTile_IntrOp : AME_IntrOpBase<"mma.dw.mm.tile">,
  Arguments<(ins LLVM_Type:$md, LLVM_Type:$ms1, LLVM_Type:$ms2)>;

#endif // AME_DIALECT
