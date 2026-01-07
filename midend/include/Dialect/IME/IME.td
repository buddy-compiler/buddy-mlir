//====------ IME.td - IME dialect operation definitions --- tablegen ------===//
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
// This file defines the IME dialect.
//
//===----------------------------------------------------------------------===//

#ifndef IME_DIALECT
#define IME_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

def IME_Dialect : Dialect {
  let name = "ime";
  let cppNamespace = "::buddy::ime";
  let description = [{
    The IME dialect provides operations for the SpacemiT Intelligent Matrix Extension (IME).
    IME is an extension to RISC-V Vector Extension (RVV) that provides matrix multiply-accumulate
    instructions optimized for AI/ML workloads.
  }];
}

class IME_Op<string mnemonic, list<Trait> traits = []> :
  Op<IME_Dialect, mnemonic, traits> {}

def VmadotOp : IME_Op<"vmadot"> {
  let summary = "Integer matrix multiply-accumulate (signed × signed)";
  let description = [{
    Performs matrix multiply-accumulate: vd += vs1 × vs2
    where vs1 and vs2 are int8 matrices, and vd is int32 accumulator.
  }];
  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$vd,
    MemRefRankOf<[I8], [2]>:$vs1,
    MemRefRankOf<[I8], [2]>:$vs2
  );
  let assemblyFormat = "$vd `,` $vs1 `,` $vs2 attr-dict `:` type($vd) `,` type($vs1) `,` type($vs2)";
}

def VmadotuOp : IME_Op<"vmadotu"> {
  let summary = "Integer matrix multiply-accumulate (unsigned × unsigned)";
  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$vd,
    MemRefRankOf<[UI8], [2]>:$vs1,
    MemRefRankOf<[UI8], [2]>:$vs2
  );
  let assemblyFormat = "$vd `,` $vs1 `,` $vs2 attr-dict `:` type($vd) `,` type($vs1) `,` type($vs2)";
}

def VmadotsuOp : IME_Op<"vmadotsu"> {
  let summary = "Integer matrix multiply-accumulate (signed × unsigned)";
  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$vd,
    MemRefRankOf<[I8], [2]>:$vs1,
    MemRefRankOf<[UI8], [2]>:$vs2
  );
  let assemblyFormat = "$vd `,` $vs1 `,` $vs2 attr-dict `:` type($vd) `,` type($vs1) `,` type($vs2)";
}

def VmadotusOp : IME_Op<"vmadotus"> {
  let summary = "Integer matrix multiply-accumulate (unsigned × signed)";
  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$vd,
    MemRefRankOf<[UI8], [2]>:$vs1,
    MemRefRankOf<[I8], [2]>:$vs2
  );
  let assemblyFormat = "$vd `,` $vs1 `,` $vs2 attr-dict `:` type($vd) `,` type($vs1) `,` type($vs2)";
}

def VfmadotOp : IME_Op<"vfmadot"> {
  let summary = "Floating-point matrix multiply-accumulate (float16)";
  let arguments = (ins
    MemRefRankOf<[F16], [2]>:$vd,
    MemRefRankOf<[F16], [2]>:$vs1,
    MemRefRankOf<[F16], [2]>:$vs2
  );
  let assemblyFormat = "$vd `,` $vs1 `,` $vs2 attr-dict `:` type($vd) `,` type($vs1) `,` type($vs2)";
}

//===----------------------------------------------------------------------===//
// IME Intrinsic operation definitions
//===----------------------------------------------------------------------===//
class IME_IntrOpBase<string mnemonic, list<Trait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/IME_Dialect,
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_ime_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[],
                  /*list<int> overloadedOperands=*/[],
                  /*list<Trait> traits=*/traits,
                  /*int numResults=*/0>;

// Integer matrix multiply-accumulate intrinsics
def IME_Vmadot_IntrOp : IME_IntrOpBase<"vmadot">,
  Arguments<(ins LLVM_Type:$vd, LLVM_Type:$vs1, LLVM_Type:$vs2)>;

def IME_Vmadotu_IntrOp : IME_IntrOpBase<"vmadotu">,
  Arguments<(ins LLVM_Type:$vd, LLVM_Type:$vs1, LLVM_Type:$vs2)>;

def IME_Vmadotsu_IntrOp : IME_IntrOpBase<"vmadotsu">,
  Arguments<(ins LLVM_Type:$vd, LLVM_Type:$vs1, LLVM_Type:$vs2)>;

def IME_Vmadotus_IntrOp : IME_IntrOpBase<"vmadotus">,
  Arguments<(ins LLVM_Type:$vd, LLVM_Type:$vs1, LLVM_Type:$vs2)>;

// Floating-point matrix multiply-accumulate intrinsic
def IME_Vfmadot_IntrOp : IME_IntrOpBase<"vfmadot">,
  Arguments<(ins LLVM_Type:$vd, LLVM_Type:$vs1, LLVM_Type:$vs2)>;

#endif // IME_DIALECT
