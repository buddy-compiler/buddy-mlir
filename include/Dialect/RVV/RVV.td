//===-- RVV.td - RVV dialect operation definitions ---------*- tablegen -*-===//
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
// This file defines the basic operations for the RVV dialect.
//
//===----------------------------------------------------------------------===//

#ifndef RVV_OPS
#define RVV_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/PatternBase.td"

//===----------------------------------------------------------------------===//
// Scalable Vector Type
//===----------------------------------------------------------------------===//

class ScalableVector_Type<Dialect dialect,string name> 
    : TypeDef<dialect, name> {
  let mnemonic = "vector";

  let summary = "Scalable vector type";
}

//===----------------------------------------------------------------------===//
// RVV dialect definition
//===----------------------------------------------------------------------===//

def RVV_Dialect : Dialect {
  let name = "rvv";
  let cppNamespace = "::buddy::rvv";
  let summary = "Basic dialect to target RISC-V Vector extension";
  let description = [{
    RISC-V vector extension (RVV) is the vector instruction set with scalable
    vector types, and the RVV instructions are vector length agnostic (VLA).
    For more details about RVV, please see the 
    [RVV specification](https://github.com/riscv/riscv-v-spec).
    This dialect contains the definitions of RVV operations and RVV 
    intrinsic operations. The former is used to interoperate with higher-level 
    dialects, and the latter is responsible for mapping to LLVM IR intrinsic.
  }];
}

//===----------------------------------------------------------------------===//
// RVV operation definitions
//===----------------------------------------------------------------------===//

class RVV_Op<string mnemonic, list<Trait> traits = []> :
  Op<RVV_Dialect, mnemonic, traits> {}

def RVVSetVlOp :
    RVV_Op<"setvl",
              !listconcat([], [AllTypesMatch<["avl", "sew", "lmul", "vl"]>])>,
    Arguments<(ins Index:$avl, Index:$sew, Index:$lmul)>,
    Results<(outs Index:$vl)> {
  let summary = "Set vector length according to AVL, SEW, and LMUL";
  let description = [{
    SetVl operation sets the vector length according to AVL, SEW, and LMUL.
    RISC-V vector extension uses this to achieve a direct and portable
    strip-mining approach, which is purposed to handle a large number of
    elements. The return value of this operation is the number of elements
    for a single iteration.
  }];
  let assemblyFormat = "$avl `,` $sew `,` $lmul attr-dict `:` type($avl)";
}

//===----------------------------------------------------------------------===//
// RVV intrinsic operation definitions
//===----------------------------------------------------------------------===//

class RVV_VSetVlI_IntrOp<string mnemonic, list<Trait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/RVV_Dialect, 
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[0], 
                  /*list<int> overloadedOperands=*/[], 
                  /*list<Trait> traits=*/traits, 
                  /*int numResults=*/1>;

def RVVIntrSetVlIOp : RVV_VSetVlI_IntrOp<"vsetvli">,
  Arguments<(ins AnyInteger, AnyInteger, AnyInteger)>;

#endif // RVV_OPS
