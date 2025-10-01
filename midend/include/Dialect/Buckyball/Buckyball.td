//===-- Buckyball.td - Buckyball dialect operation definitions --- tablegen --===//
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
// This file defines the basic operations for the Buckyball dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BUCKYBALL_OPS
#define BUCKYBALL_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/PatternBase.td"

//===----------------------------------------------------------------------===//
// Buckyball dialect definition
//===----------------------------------------------------------------------===//

def Buckyball_Dialect : Dialect {
  let name = "buckyball";
  let cppNamespace = "::buddy::buckyball";
  let summary = "Basic dialect to target RISC-V Buckyball extension";
  let description = [{
    Buckyball is an NPU Design Framework with scaling and flexible 
    features. For more details about Buckyball, please see the 
    [Buckyball repo](https://github.com/DangoSys/buckyball).
  }];
}

//===----------------------------------------------------------------------===//
// Buckyball operation definitions
//===----------------------------------------------------------------------===//

class Buckyball_Op<string mnemonic, list<Trait> traits = []> :
  Op<Buckyball_Dialect, mnemonic, traits> {}

// instruction level operation
def MvinOp : Buckyball_Op<"bb_mvin"> {
  let summary = "Load operation";
  let description = [{
    Move data from main memory to scratchpad
    - MemRef to load in (DRAM address)
    - Local scratchpad address.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input, I64:$addr);
  let assemblyFormat = "$input $addr attr-dict `:` type($input) type($addr)";
}

// instruction level operation
def MvoutOp : Buckyball_Op<"bb_mvout"> {
  let summary = "Store operation";
  let description = [{
    Move data from scratchpad to main memory
    - MemRef to store from scratchpad.
    - Local scratchpad address.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$output, I64:$addr);
  let assemblyFormat = "$output $addr attr-dict `:` type($output) type($addr)";
}

// instruction level operation
def FenceOp : Buckyball_Op<"bb_fence"> {
  let summary = "Fence operation";
  let description = [{
    Fence operation is used to synchronize the execution of the current core.
  }];
  let arguments = (ins);
  let assemblyFormat = "attr-dict";
}

/* Multi-Level Tiling Design
 *  -> VecTileMatMulOp
 *  aMemArray, bMemArray, cMemArray - (Value) abc matrix's main memory address
 *  warpNum - use how many warps (choose the exec vector unit)  
 *
 *  -> MergeTileMatMulOp (divide <mergeM x mergeN x mergeK> into <mergeM x mergeN x 1>) Loop over mergeKLen
 *  aMemArray, bMemArray - (MemRef) MergeTile's main memory memref
 *  aMemAddrStart, bMemAddrStart, cMemAddrStart - abc matrix's relative main memory address ptr
 *  aSpAddrStart, bSpAddrStart, cSpAddrStart - abc matrix's scrachpad/accumulator ptr
 *  m, n, k - abc matrix's shape/size (used to calculate abc matrix's address stride) 
 *  mergeMLen, mergeNLen, mergeKLen - mergeTile's shape/size
 *
 *  -> MetaTileMatMulOp (divide <mergeM x mergeN x 1> into <1 x mergeN x 1>) Loop over mergeMLen
 *  aMemArray - (MemRef) MetaTile's main memory memref
 *  aSpAddrStart, bSpAddrStart, cSpAddrStart - abc matrix's relative start address of scrachpad/accumulator ptr
 *  mergeMLen, mergeNLen - metaTile's shape/size
 *  
 *  -> VecMulWarp16Op
 *  aSpAddrStart, bSpAddrStart, cSpAddrStart - abc matrix's relative start address of scrachpad/accumulator ptr
 *  nLen - VecMul's lane size (sum of mergeNLen)
 *
 *  -> mvin
 *  input - MemRef
 *  addr - I64
 *
 *  -> mvout
 *  output - MemRef
 *  addr - I64
 *
*/

// Tile level operation
def VecTileMatMulOp : Buckyball_Op<"bb_tile_matmul"> {
  let summary = "Vector-matrix multiplication";
  let description = [{
    Perform matrix multiplication operation in vector unit.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$aMemArray,
                       MemRefRankOf<[AnyType], [2]>:$bMemArray,
                       MemRefRankOf<[AnyType], [2]>:$cMemArray,
                       DefaultValuedAttr<I64Attr, "16">:$warpNum);
  let assemblyFormat = [{$aMemArray $bMemArray $cMemArray attr-dict `:` 
    type($aMemArray) type($bMemArray) type($cMemArray)}];
}

def MergeTileMatMulOp : Buckyball_Op<"bb_mergetile_matmul"> {
  let summary = "Merge tile matrix multiplication";
  let description = [{
    Perform matrix multiplication operation in vector unit.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$aMergeTileArray,
                       MemRefRankOf<[AnyType], [2]>:$aMemArray,
                       MemRefRankOf<[AnyType], [2]>:$bMergeTileArray,
                       MemRefRankOf<[AnyType], [2]>:$bMemArray,
                       I64:$aSpAddrStart, I64:$bSpAddrStart, I64:$cSpAddrStart,
                       I64:$metaMNum, I64:$metaNNum, I64:$metaKNum,
                       I64:$metaMLen, I64:$metaNLen, I64:$metaKLen);
  let assemblyFormat = [{$aMergeTileArray $aMemArray $bMergeTileArray $bMemArray $aSpAddrStart $bSpAddrStart $cSpAddrStart $metaMNum $metaNNum $metaKNum $metaMLen $metaNLen $metaKLen attr-dict `:` 
    type($aMergeTileArray) type($aMemArray) type($bMergeTileArray) type($bMemArray) type($aSpAddrStart) type($bSpAddrStart) type($cSpAddrStart) type($metaMNum) type($metaNNum) type($metaKNum) type($metaMLen) type($metaNLen) type($metaKLen)}];
}

def MetaTileMatMulOp : Buckyball_Op<"bb_metatile_matmul"> {
  let summary = "Meta tile matrix multiplication";
  let description = [{
    Perform matrix multiplication operation in vector unit.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$aMetaTileArray,
                       MemRefRankOf<[AnyType], [2]>:$aMemArray,
                       I64:$aSpAddrStart, I64:$bSpAddrStart, I64:$cSpAddrStart,
                       I64:$metaMNum, I64:$metaNNum,
                       I64:$metaMLen, I64:$metaNLen);
  let assemblyFormat = [{$aMetaTileArray $aMemArray $aSpAddrStart $bSpAddrStart $cSpAddrStart $metaMNum $metaNNum $metaMLen $metaNLen attr-dict `:` 
     type($aMetaTileArray) type($aMemArray) type($aSpAddrStart) type($bSpAddrStart) type($cSpAddrStart) type($metaMNum) type($metaNNum) type($metaMLen) type($metaNLen)}];
}

// instruction level operation
def VecMulWarp16Op : Buckyball_Op<"bb_mul_warp16"> {
  let summary = "Vector multiply with 16 warp parallelism";
  let description = [{
    Perform vector multiply operation in vector unit 
    with warp number is 16.
  }];
  let arguments = (ins I64:$aSpAddr, 
                       I64:$bSpAddr, 
                       I64:$cSpAddr, 
                       I64:$nLen);
  let assemblyFormat = [{$aSpAddr $bSpAddr $cSpAddr $nLen attr-dict `:` 
    type($aSpAddr) type($bSpAddr) type($cSpAddr) type($nLen)}];
}

// TODO: add really detailed deployment
// instruction level operation
def TransposeOp : Buckyball_Op<"bb_transpose"> {
  let summary = "Vector transpose";
  let description = [{
    Perform transpose operation on vector.
  }];
  let arguments = (ins);
  let assemblyFormat = "attr-dict";
}

// TODO: add really detailed deployment
// instruction level operation
def Im2colOp : Buckyball_Op<"bb_im2col"> {
  let summary = "Im2col operation";
  let description = [{
    Perform im2col operation on vector.
  }];
  let arguments = (ins);
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Buckyball Test Operation 
//===----------------------------------------------------------------------===//
def PrintMemRefOp : Buckyball_Op<"bb_print_memref"> {
  let summary = "Print memref value.";
  let arguments = (ins AnyTypeOf<[I8MemRef, I32MemRef, F32MemRef, F64MemRef]>:$input); 
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def PrintScalarOp : Buckyball_Op<"bb_print_scalar"> {
  let summary = "Print a scalar value.";
  let arguments = (ins AnyTypeOf<[I64, I32, F32, F64]>:$input); 
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def CounterStartOp : Buckyball_Op<"bb_counter_start"> {
  let summary = "Start counter";
  let description = [{
    Start counter.
  }];
  let arguments = (ins I64:$counterId);
  let assemblyFormat = "$counterId attr-dict `:` type($counterId)";
}

def CounterStopOp : Buckyball_Op<"bb_counter_stop"> {
  let summary = "Stop counter";
  let description = [{
    Stop counter and return the elapsed cycles.
  }];
  let arguments = (ins I64:$counterId);
  let results = (outs I64:$cycles);
  let assemblyFormat = "$counterId attr-dict `:` type($counterId) `->` type($cycles)";
}

//===----------------------------------------------------------------------===//
// Buckyball intrinsic operation definitions
//===----------------------------------------------------------------------===//

class Buckyball_IntrOpBase<string mnemonic, list<Trait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/Buckyball_Dialect,
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[],
                  /*list<int> overloadedOperands=*/[],
                  /*list<Trait> traits=*/traits,
                  /*int numResults=*/0>;

def Buckyball_Mvin_IntrOp : Buckyball_IntrOpBase<"bb_mvin">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;
  
def Buckyball_Mvout_IntrOp : Buckyball_IntrOpBase<"bb_mvout">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Buckyball_BBFP_MUL_IntrOp : Buckyball_IntrOpBase<"bb_bbfp_mul">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Buckyball_MATMUL_WS_IntrOp : Buckyball_IntrOpBase<"bb_matmul_ws">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

// No arguments needed for fence operation
def Buckyball_FENCE_IntrOp : Buckyball_IntrOpBase<"bb_fence">;

def Buckyball_MUL_WARP16_IntrOp : Buckyball_IntrOpBase<"bb_mul_warp16">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Buckyball_IM2COL_IntrOp : Buckyball_IntrOpBase<"bb_im2col">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Buckyball_TRANSPOSE_IntrOp : Buckyball_IntrOpBase<"bb_transpose">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

#endif // BUCKYBALL_OPS
