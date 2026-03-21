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
    Move data from main memory to a scratchpad bank (see `bb-tests/workloads/lib/bbhw/isa/33_mvin.c`).
    - `input`: source memref in DRAM; base pointer is encoded in the instruction.
    - `addr`: virtual bank id (operand historically named `addr`; maps to `BB_BANK0` in rs1).
    - `stride`: line stride field in rs2 (same units as hardware `FIELD(stride, 10, 28)`).
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input, I64:$addr, I64:$stride);
  let assemblyFormat = "$input $addr $stride attr-dict `:` type($input) type($addr) type($stride)";
}

// instruction level operation
def MvoutOp : Buckyball_Op<"bb_mvout"> {
  let summary = "Store operation";
  let description = [{
    Move data from a scratchpad bank to main memory (see `bb-tests/workloads/lib/bbhw/isa` mvout).
    - `output`: destination memref in DRAM.
    - `addr`: virtual bank id for the source bank.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$output, I64:$addr);
  let assemblyFormat = "$output $addr attr-dict `:` type($output) type($addr)";
}

// instruction level operation
def MsetOp : Buckyball_Op<"bb_mset"> {
  let summary = "Bank memory allocation / release";
  let description = [{
    Bind or free a scratchpad virtual bank (`32_mset.c`): `alloc = true` allocates
    (default), `alloc = false` releases the mapping after mvin/mvout.
  }];
  let arguments =
      (ins I64:$bankId, DefaultValuedAttr<BoolAttr, "true">:$alloc);
  let assemblyFormat = "$bankId attr-dict `:` type($bankId)";
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

/* Buckyball Hardware-Level Operations:
 *  MatMulOp - Fixed-size matrix multiplication that maps to hardware execution
 *  - Handles a fixed-size tile that fits in SPAD/ACC
 *  - Generates mvin/mvout for data movement
 *  - Executes computation using warp-level parallelism
 *  - Will be lowered to intrinsic operations (mvin/mvout/fence)
 */

// Hardware-level matrix multiplication operation
def MatMulOp : Buckyball_Op<"bb_matmul"> {
  let summary = "Hardware-level matrix multiplication";
  let description = [{
    Perform fixed-size matrix multiplication that fits in accelerator resources.
    This operation represents a complete matrix multiplication tile that:
    - Fits within the scratchpad and accumulator constraints
    - Performs data movement (mvin) for input tiles
    - Executes computation using warp-level parallelism
    - Writes back results (mvout) to main memory
    
    The operation will be lowered to:
    - Mvin operations for loading A and B matrices
    - Compute operations using warp parallelism
    - Mvout operations for storing C matrix
    - Fence operations for synchronization
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$aMemArray,
                       MemRefRankOf<[AnyType], [2]>:$bMemArray,
                       MemRefRankOf<[AnyType], [2]>:$cMemArray);
  let assemblyFormat = [{$aMemArray $bMemArray $cMemArray attr-dict `:` 
    type($aMemArray) type($bMemArray) type($cMemArray)}];
}

// TODO: add really detailed deployment
// instruction level operation
def TransposeOp : Buckyball_Op<"bb_transpose"> {
  let summary = "Vector transpose";
  let description = [{
    Perform transpose operation on vector.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input,
                    MemRefRankOf<[AnyType], [2]>:$output);
  let assemblyFormat = [{$input $output attr-dict `:` type($input) type($output)}];
}

// TODO: add really detailed deployment
// instruction level operation
def Im2colOp : Buckyball_Op<"bb_im2col"> {
  let summary = "Im2col operation";
  let description = [{
    Perform im2col data rearrangement for convolution.
    Extracts patches from input and rearranges them into columns.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input,
                       MemRefRankOf<[AnyType], [2]>:$output,
                       I64:$kRow, I64:$kCol,
                       I64:$inRow, I64:$inCol,
                       I64:$startRow, I64:$startCol);
  let assemblyFormat = [{$input $output $kRow $kCol $inRow $inCol $startRow $startCol
    attr-dict `:` type($input) type($output)}];
}

def QuantOp : Buckyball_Op<"bb_quant"> {
  let summary = "Quantize FP32 to INT8";
  let description = [{
    Quantize floating-point data to int8 using the given scale factor.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input,
                       MemRefRankOf<[AnyType], [2]>:$output,
                       F32:$scale);
  let assemblyFormat = [{$input $output $scale attr-dict `:` type($input) type($output) type($scale)}];
}

def DequantOp : Buckyball_Op<"bb_dequant"> {
  let summary = "Dequantize INT8 to FP32";
  let description = [{
    Dequantize int8 data to floating-point using the given scale factor.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input,
                       MemRefRankOf<[AnyType], [2]>:$output,
                       F32:$scale);
  let assemblyFormat = [{$input $output $scale attr-dict `:` type($input) type($output) type($scale)}];
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

def Buckyball_BFP_Mul_IntrOp : Buckyball_IntrOpBase<"bb_bbfp_mul">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

// No arguments needed for fence operation
def Buckyball_Fence_IntrOp : Buckyball_IntrOpBase<"bb_fence">;

def Buckyball_Mul_Warp16_IntrOp : Buckyball_IntrOpBase<"bb_mul_warp16">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Buckyball_Im2col_IntrOp : Buckyball_IntrOpBase<"bb_im2col">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Buckyball_Transpose_IntrOp : Buckyball_IntrOpBase<"bb_transpose">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Buckyball_Relu_IntrOp : Buckyball_IntrOpBase<"bb_relu">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Buckyball_Quant_IntrOp : Buckyball_IntrOpBase<"bb_quant">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Buckyball_Dequant_IntrOp : Buckyball_IntrOpBase<"bb_dequant">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Buckyball_Mset_IntrOp : Buckyball_IntrOpBase<"bb_mset">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

#endif // BUCKYBALL_OPS
