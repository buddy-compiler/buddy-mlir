//===-- Gemmini.td - Gemmini dialect operation definitions --- tablegen --===//
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
// This file defines the basic operations for the Gemmini dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GEMMINI_OPS
#define GEMMINI_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/PatternBase.td"

//===----------------------------------------------------------------------===//
// Gemmini dialect definition
//===----------------------------------------------------------------------===//

def Gemmini_Dialect : Dialect {
  let name = "gemmini";
  let cppNamespace = "::buddy::gemmini";
  let summary = "Basic dialect to target RISC-V Gemmini extension";
  let description = [{
    Gemmini is an accelerator based on systolic array.For more details about Gemmini,
    please see the [Gemmini](https://github.com/ucb-bar/gemmini).

  }];
}

//===----------------------------------------------------------------------===//
// Gemmini operation definitions
//===----------------------------------------------------------------------===//
class Gemmini_Op<string mnemonic, list<Trait> traits = []> :
  Op<Gemmini_Dialect, mnemonic, traits> {}

def  FlushOp : Gemmini_Op<"flush"> {
  let summary = "FLush op";
  let arguments = (ins I64:$skip);
  let assemblyFormat = "$skip attr-dict `:` type($skip)";
}

def ConfigStOp : Gemmini_Op<"configSt"> {
  let summary = "Config store operation";
  let arguments = (ins I64:$stride, DefaultValuedAttr<I64Attr, "0">:$activation, DefaultValuedAttr<F32Attr, "1.0">:$scale);  
  let assemblyFormat = "$stride attr-dict `:` type($stride)";
}

def ConfigLdOp : Gemmini_Op<"configLd"> {
  let summary = "Config load operation";
  let arguments = (ins I64:$stride, DefaultValuedAttr<F32Attr, "1.0">:$scale, DefaultValuedAttr<BoolAttr, "false">:$shrunk,
                   DefaultValuedAttr<I64Attr, "0">:$id);
  let assemblyFormat = "$stride attr-dict `:` type($stride)";
}

def ConfigExOp : Gemmini_Op<"configEx"> {
  let summary = "configExOp configures the execute pipeline";
  let arguments = (ins DefaultValuedAttr<I64Attr, "0">:$dataflow, DefaultValuedAttr<I64Attr, "0">:$sysAct, 
                       DefaultValuedAttr<I64Attr, "0">:$sysShift, DefaultValuedAttr<F32Attr, "1.0">:$sysAccScale,
                       DefaultValuedAttr<I64Attr, "1">:$aStride, DefaultValuedAttr<BoolAttr, "false">:$aTranspose,
                       DefaultValuedAttr<BoolAttr, "false">:$bTranspose);
  let assemblyFormat = "attr-dict";
}

def MvinOp : Gemmini_Op<"mvin"> {
  let summary = "Load operation";
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input, I64:$addr);
  let assemblyFormat = "$input $addr attr-dict `:` type($input) type($addr)";
}

def MvoutOp : Gemmini_Op<"mvout"> {
  let summary = "Store operation";
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$output, I64:$addr);
  let assemblyFormat = "$output $addr attr-dict `:` type($output) type($addr)";
}

def PrintOp : Gemmini_Op<"print"> {
  let summary = "Print operation";
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def PreloadZerosOp : Gemmini_Op<"preloadZeros"> {
  let summary = "Preload 0 matrix";
  let arguments = (ins I64:$addr, I64:$cRows, I64:$cCols);
  let assemblyFormat = "$addr $cRows $cCols attr-dict `:` type($addr) type($cRows) type($cCols)";
}

def PreloadOp : Gemmini_Op<"preload"> {
  let summary = "Preload matrixs operation";
  let arguments = (ins I64:$bdAddr, I64:$cAddr, I64:$bdRows, I64:$bdCols, I64:$cRows, I64:$cCols);
  let assemblyFormat = "$bdAddr $cAddr $bdRows $bdCols $cRows $cCols  attr-dict `:` type($bdAddr) type($cAddr) type($bdRows) type($bdCols) type($cRows) type($cCols) ";
}

def ComputePreloadedOp : Gemmini_Op<"computePreloaded"> {
  let summary = "compute operation";
  let arguments = (ins I64:$aAddr, I64:$bdAddr, I64:$aRows, I64:$aCols, I64:$bdRows, I64:$bdCols);
  let assemblyFormat = "$aAddr $bdAddr $aRows $aCols $bdRows $bdCols attr-dict `:` type($aAddr) type($bdAddr) type($aRows) type($aCols) type($bdRows) type($bdCols)";
}

def ComputeAccumulatedOp : Gemmini_Op<"computeAccumulated"> {
  let summary = "compute accumulated opertion";
  let arguments = (ins I64:$aAddr, I64:$bdAddr, I64:$aRows, I64:$aCols, I64:$bdRows, I64:$bdCols);
  let assemblyFormat = "$aAddr $bdAddr $aRows $aCols $bdRows $bdCols attr-dict `:` type($aAddr) type($bdAddr) type($aRows) type($aCols) type($bdRows) type($bdCols)";
}

def TileMatMulOp : Gemmini_Op<"tileMatMul"> {
  let summary = "loop instruction.";
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$aArray, MemRefRankOf<[AnyType], [2]>:$bArray,
                       MemRefRankOf<[AnyType], [2]>:$cArray, MemRefRankOf<[AnyType], [2]>:$dArray,
                       DefaultValuedAttr<F32Attr, "1.0">:$aScaleFactor, DefaultValuedAttr<F32Attr, "1.0">:$bScaleFactor,
                       DefaultValuedAttr<F32Attr, "1.0">:$dScaleFactor, DefaultValuedAttr<I64Attr, "0">:$act,
                       DefaultValuedAttr<F32Attr, "1.0">:$accScale, DefaultValuedAttr<F32Attr, "0.0">:$bertScale,
                       DefaultValuedAttr<BoolAttr, "false">:$repeatingBias, DefaultValuedAttr<BoolAttr, "false">:$aTranspose,
                       DefaultValuedAttr<BoolAttr, "false">:$bTranspose, DefaultValuedAttr<BoolAttr, "false">:$fullC,
                       DefaultValuedAttr<BoolAttr, "false">:$lowD, DefaultValuedAttr<I64Attr, "0">:$weightA);
  let assemblyFormat = "$aArray $bArray $cArray $dArray attr-dict `:` type($aArray) type($bArray) type($cArray) type($dArray)";  
}

//===----------------------------------------------------------------------===//
// Gemmini intrinsic operation definitions
//===----------------------------------------------------------------------===//

class Gemmini_IntrOpBase<string mnemonic, list<Trait> traits = []> : 
  LLVM_IntrOpBase</*Dialect dialect=*/Gemmini_Dialect, 
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[], 
                  /*list<int> overloadedOperands=*/[], 
                  /*list<Trait> traits=*/traits, 
                  /*int numResults=*/0>;
def Gemmini_Flush_IntrOp : Gemmini_IntrOpBase<"flush">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_ConfigSt_IntrOp : Gemmini_IntrOpBase<"configSt">, 
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_ConifgLd_IntrOp : Gemmini_IntrOpBase<"configLd">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_ConfigEX_IntrOp : Gemmini_IntrOpBase<"configEx">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_Mvin_IntrOp : Gemmini_IntrOpBase<"mvin">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_Mvout_IntrOp : Gemmini_IntrOpBase<"mvout">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_Preload_IntrOp : Gemmini_IntrOpBase<"preload">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_ComputePreloaded_IntrOp: Gemmini_IntrOpBase<"computePreloaded">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_ComputeAccumulated_IntrOp : Gemmini_IntrOpBase<"computeAccumulated">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_LoopWsConfigBounds_IntrOp : Gemmini_IntrOpBase<"loopWsConfigBounds">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_LoopWsConfigAddrsAB_IntrOp : Gemmini_IntrOpBase<"loopWsConfigAddrsAB">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_LoopWsConfigAddrsDC_IntrOp : Gemmini_IntrOpBase<"loopWsConfigAddrsDC">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_LoopWsConfigStridesAB_IntrOp : Gemmini_IntrOpBase<"loopWsConfigStridesAB">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_LoopWsConfigStridesDC_IntrOp : Gemmini_IntrOpBase<"loopWsConfigStridesDC">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_LoopWs_IntrOp : Gemmini_IntrOpBase<"loopWs">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

#endif 
