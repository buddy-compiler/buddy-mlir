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

def ConfigStOp : Gemmini_Op<"configSt"> {
  let summary = "Config store operation";
  let arguments = (ins I64:$stride, DefaultValuedAttr<I64Attr, "0">:$activation);  
  let assemblyFormat = "$stride attr-dict `:` type($stride)";
}

def ConfigLdOp : Gemmini_Op<"configLd"> {
  let summary = "Config load operation";
  let arguments = (ins I64:$stride);
  let assemblyFormat = "$stride attr-dict `:` type($stride)";
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

//===----------------------------------------------------------------------===//
// RVV intrinsic operation definitions
//===----------------------------------------------------------------------===//

class Gemmini_IntrOpBase<string mnemonic, list<Trait> traits = []> : 
  LLVM_IntrOpBase</*Dialect dialect=*/Gemmini_Dialect, 
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[], 
                  /*list<int> overloadedOperands=*/[], 
                  /*list<Trait> traits=*/traits, 
                  /*int numResults=*/0>;

def Gemmini_ConfigSt_IntrOp : Gemmini_IntrOpBase<"configSt">, 
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_ConifgLd_IntrOp : Gemmini_IntrOpBase<"configLd">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_Mvin_IntrOp : Gemmini_IntrOpBase<"mvin">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

def Gemmini_Mvout_IntrOp : Gemmini_IntrOpBase<"mvout">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

#endif 
