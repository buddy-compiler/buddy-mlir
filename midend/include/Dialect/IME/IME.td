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
    where vs1 and vs2 are int8 or int16 matrices, and vd is int32 accumulator.
    
    For int8 (SEW=e8): MAC unit 4x4x8, matrices A(4x8), B(8x4), C(4x4)
    For int16 (SEW=e16): MAC unit 4x4x4, matrices A(4x4), B(4x4), C(4x4)
  }];
  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$vd,
    MemRefRankOf<[I8, I16], [2]>:$vs1,
    MemRefRankOf<[I8, I16], [2]>:$vs2
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
// Sliding-window Integer Matrix Multiply-Accumulate Operations
//===----------------------------------------------------------------------===//
// These operations select values from VS1 and VS1+1 using a sliding window.
// slide-1/2/3 use fixed slide values, while slide-n uses a dynamic value.

// slide-1 operations
def Vmadot1Op : IME_Op<"vmadot1"> {
  let summary = "Sliding-window integer MMA (signed × signed, slide=1)";
  let description = [{
    Performs sliding-window matrix multiply-accumulate with slide=1.
    Input A is selected from VS1 and VS1+1 with a slide offset of 1*K elements.
  }];
  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$vd,
    MemRefRankOf<[I8], [2]>:$vs1,
    MemRefRankOf<[I8], [2]>:$vs2
  );
  let assemblyFormat = "$vd `,` $vs1 `,` $vs2 attr-dict `:` type($vd) `,` type($vs1) `,` type($vs2)";
}

def Vmadot1uOp : IME_Op<"vmadot1u"> {
  let summary = "Sliding-window integer MMA (unsigned × unsigned, slide=1)";
  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$vd,
    MemRefRankOf<[UI8], [2]>:$vs1,
    MemRefRankOf<[UI8], [2]>:$vs2
  );
  let assemblyFormat = "$vd `,` $vs1 `,` $vs2 attr-dict `:` type($vd) `,` type($vs1) `,` type($vs2)";
}

def Vmadot1suOp : IME_Op<"vmadot1su"> {
  let summary = "Sliding-window integer MMA (signed × unsigned, slide=1)";
  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$vd,
    MemRefRankOf<[I8], [2]>:$vs1,
    MemRefRankOf<[UI8], [2]>:$vs2
  );
  let assemblyFormat = "$vd `,` $vs1 `,` $vs2 attr-dict `:` type($vd) `,` type($vs1) `,` type($vs2)";
}

def Vmadot1usOp : IME_Op<"vmadot1us"> {
  let summary = "Sliding-window integer MMA (unsigned × signed, slide=1)";
  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$vd,
    MemRefRankOf<[UI8], [2]>:$vs1,
    MemRefRankOf<[I8], [2]>:$vs2
  );
  let assemblyFormat = "$vd `,` $vs1 `,` $vs2 attr-dict `:` type($vd) `,` type($vs1) `,` type($vs2)";
}

// slide-2 operations
def Vmadot2Op : IME_Op<"vmadot2"> {
  let summary = "Sliding-window integer MMA (signed × signed, slide=2)";
  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$vd,
    MemRefRankOf<[I8], [2]>:$vs1,
    MemRefRankOf<[I8], [2]>:$vs2
  );
  let assemblyFormat = "$vd `,` $vs1 `,` $vs2 attr-dict `:` type($vd) `,` type($vs1) `,` type($vs2)";
}

def Vmadot2uOp : IME_Op<"vmadot2u"> {
  let summary = "Sliding-window integer MMA (unsigned × unsigned, slide=2)";
  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$vd,
    MemRefRankOf<[UI8], [2]>:$vs1,
    MemRefRankOf<[UI8], [2]>:$vs2
  );
  let assemblyFormat = "$vd `,` $vs1 `,` $vs2 attr-dict `:` type($vd) `,` type($vs1) `,` type($vs2)";
}

def Vmadot2suOp : IME_Op<"vmadot2su"> {
  let summary = "Sliding-window integer MMA (signed × unsigned, slide=2)";
  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$vd,
    MemRefRankOf<[I8], [2]>:$vs1,
    MemRefRankOf<[UI8], [2]>:$vs2
  );
  let assemblyFormat = "$vd `,` $vs1 `,` $vs2 attr-dict `:` type($vd) `,` type($vs1) `,` type($vs2)";
}

def Vmadot2usOp : IME_Op<"vmadot2us"> {
  let summary = "Sliding-window integer MMA (unsigned × signed, slide=2)";
  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$vd,
    MemRefRankOf<[UI8], [2]>:$vs1,
    MemRefRankOf<[I8], [2]>:$vs2
  );
  let assemblyFormat = "$vd `,` $vs1 `,` $vs2 attr-dict `:` type($vd) `,` type($vs1) `,` type($vs2)";
}

// slide-3 operations
def Vmadot3Op : IME_Op<"vmadot3"> {
  let summary = "Sliding-window integer MMA (signed × signed, slide=3)";
  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$vd,
    MemRefRankOf<[I8], [2]>:$vs1,
    MemRefRankOf<[I8], [2]>:$vs2
  );
  let assemblyFormat = "$vd `,` $vs1 `,` $vs2 attr-dict `:` type($vd) `,` type($vs1) `,` type($vs2)";
}

def Vmadot3uOp : IME_Op<"vmadot3u"> {
  let summary = "Sliding-window integer MMA (unsigned × unsigned, slide=3)";
  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$vd,
    MemRefRankOf<[UI8], [2]>:$vs1,
    MemRefRankOf<[UI8], [2]>:$vs2
  );
  let assemblyFormat = "$vd `,` $vs1 `,` $vs2 attr-dict `:` type($vd) `,` type($vs1) `,` type($vs2)";
}

def Vmadot3suOp : IME_Op<"vmadot3su"> {
  let summary = "Sliding-window integer MMA (signed × unsigned, slide=3)";
  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$vd,
    MemRefRankOf<[I8], [2]>:$vs1,
    MemRefRankOf<[UI8], [2]>:$vs2
  );
  let assemblyFormat = "$vd `,` $vs1 `,` $vs2 attr-dict `:` type($vd) `,` type($vs1) `,` type($vs2)";
}

def Vmadot3usOp : IME_Op<"vmadot3us"> {
  let summary = "Sliding-window integer MMA (unsigned × signed, slide=3)";
  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$vd,
    MemRefRankOf<[UI8], [2]>:$vs1,
    MemRefRankOf<[I8], [2]>:$vs2
  );
  let assemblyFormat = "$vd `,` $vs1 `,` $vs2 attr-dict `:` type($vd) `,` type($vs1) `,` type($vs2)";
}

// slide-n operations (dynamic slide value)
def VmadotnOp : IME_Op<"vmadotn"> {
  let summary = "Sliding-window integer MMA (signed × signed, slide=n)";
  let description = [{
    Performs sliding-window matrix multiply-accumulate with dynamic slide value.
    The slide value is passed as an additional operand.
  }];
  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$vd,
    MemRefRankOf<[I8], [2]>:$vs1,
    MemRefRankOf<[I8], [2]>:$vs2,
    I64:$slide
  );
  let assemblyFormat = "$vd `,` $vs1 `,` $vs2 `,` $slide attr-dict `:` type($vd) `,` type($vs1) `,` type($vs2)";
}

def VmadotnuOp : IME_Op<"vmadotnu"> {
  let summary = "Sliding-window integer MMA (unsigned × unsigned, slide=n)";
  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$vd,
    MemRefRankOf<[UI8], [2]>:$vs1,
    MemRefRankOf<[UI8], [2]>:$vs2,
    I64:$slide
  );
  let assemblyFormat = "$vd `,` $vs1 `,` $vs2 `,` $slide attr-dict `:` type($vd) `,` type($vs1) `,` type($vs2)";
}

def VmadotnsuOp : IME_Op<"vmadotnsu"> {
  let summary = "Sliding-window integer MMA (signed × unsigned, slide=n)";
  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$vd,
    MemRefRankOf<[I8], [2]>:$vs1,
    MemRefRankOf<[UI8], [2]>:$vs2,
    I64:$slide
  );
  let assemblyFormat = "$vd `,` $vs1 `,` $vs2 `,` $slide attr-dict `:` type($vd) `,` type($vs1) `,` type($vs2)";
}

def VmadotnusOp : IME_Op<"vmadotnus"> {
  let summary = "Sliding-window integer MMA (unsigned × signed, slide=n)";
  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$vd,
    MemRefRankOf<[UI8], [2]>:$vs1,
    MemRefRankOf<[I8], [2]>:$vs2,
    I64:$slide
  );
  let assemblyFormat = "$vd `,` $vs1 `,` $vs2 `,` $slide attr-dict `:` type($vd) `,` type($vs1) `,` type($vs2)";
}

//===----------------------------------------------------------------------===//
// Sliding-window Floating-Point Matrix Multiply-Accumulate Operations
//===----------------------------------------------------------------------===//

def Vfmadot1Op : IME_Op<"vfmadot1"> {
  let summary = "Sliding-window floating-point MMA (slide=1)";
  let arguments = (ins
    MemRefRankOf<[F16], [2]>:$vd,
    MemRefRankOf<[F16], [2]>:$vs1,
    MemRefRankOf<[F16], [2]>:$vs2
  );
  let assemblyFormat = "$vd `,` $vs1 `,` $vs2 attr-dict `:` type($vd) `,` type($vs1) `,` type($vs2)";
}

def Vfmadot2Op : IME_Op<"vfmadot2"> {
  let summary = "Sliding-window floating-point MMA (slide=2)";
  let arguments = (ins
    MemRefRankOf<[F16], [2]>:$vd,
    MemRefRankOf<[F16], [2]>:$vs1,
    MemRefRankOf<[F16], [2]>:$vs2
  );
  let assemblyFormat = "$vd `,` $vs1 `,` $vs2 attr-dict `:` type($vd) `,` type($vs1) `,` type($vs2)";
}

def Vfmadot3Op : IME_Op<"vfmadot3"> {
  let summary = "Sliding-window floating-point MMA (slide=3)";
  let arguments = (ins
    MemRefRankOf<[F16], [2]>:$vd,
    MemRefRankOf<[F16], [2]>:$vs1,
    MemRefRankOf<[F16], [2]>:$vs2
  );
  let assemblyFormat = "$vd `,` $vs1 `,` $vs2 attr-dict `:` type($vd) `,` type($vs1) `,` type($vs2)";
}

def VfmadotnOp : IME_Op<"vfmadotn"> {
  let summary = "Sliding-window floating-point MMA (slide=n)";
  let arguments = (ins
    MemRefRankOf<[F16], [2]>:$vd,
    MemRefRankOf<[F16], [2]>:$vs1,
    MemRefRankOf<[F16], [2]>:$vs2,
    I64:$slide
  );
  let assemblyFormat = "$vd `,` $vs1 `,` $vs2 `,` $slide attr-dict `:` type($vd) `,` type($vs1) `,` type($vs2)";
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

//===----------------------------------------------------------------------===//
// Sliding-window Integer Matrix Multiply-Accumulate Intrinsics
//===----------------------------------------------------------------------===//

// slide-1 intrinsics
def IME_Vmadot1_IntrOp : IME_IntrOpBase<"vmadot1">,
  Arguments<(ins LLVM_Type:$vd, LLVM_Type:$vs1, LLVM_Type:$vs2)>;

def IME_Vmadot1u_IntrOp : IME_IntrOpBase<"vmadot1u">,
  Arguments<(ins LLVM_Type:$vd, LLVM_Type:$vs1, LLVM_Type:$vs2)>;

def IME_Vmadot1su_IntrOp : IME_IntrOpBase<"vmadot1su">,
  Arguments<(ins LLVM_Type:$vd, LLVM_Type:$vs1, LLVM_Type:$vs2)>;

def IME_Vmadot1us_IntrOp : IME_IntrOpBase<"vmadot1us">,
  Arguments<(ins LLVM_Type:$vd, LLVM_Type:$vs1, LLVM_Type:$vs2)>;

// slide-2 intrinsics
def IME_Vmadot2_IntrOp : IME_IntrOpBase<"vmadot2">,
  Arguments<(ins LLVM_Type:$vd, LLVM_Type:$vs1, LLVM_Type:$vs2)>;

def IME_Vmadot2u_IntrOp : IME_IntrOpBase<"vmadot2u">,
  Arguments<(ins LLVM_Type:$vd, LLVM_Type:$vs1, LLVM_Type:$vs2)>;

def IME_Vmadot2su_IntrOp : IME_IntrOpBase<"vmadot2su">,
  Arguments<(ins LLVM_Type:$vd, LLVM_Type:$vs1, LLVM_Type:$vs2)>;

def IME_Vmadot2us_IntrOp : IME_IntrOpBase<"vmadot2us">,
  Arguments<(ins LLVM_Type:$vd, LLVM_Type:$vs1, LLVM_Type:$vs2)>;

// slide-3 intrinsics
def IME_Vmadot3_IntrOp : IME_IntrOpBase<"vmadot3">,
  Arguments<(ins LLVM_Type:$vd, LLVM_Type:$vs1, LLVM_Type:$vs2)>;

def IME_Vmadot3u_IntrOp : IME_IntrOpBase<"vmadot3u">,
  Arguments<(ins LLVM_Type:$vd, LLVM_Type:$vs1, LLVM_Type:$vs2)>;

def IME_Vmadot3su_IntrOp : IME_IntrOpBase<"vmadot3su">,
  Arguments<(ins LLVM_Type:$vd, LLVM_Type:$vs1, LLVM_Type:$vs2)>;

def IME_Vmadot3us_IntrOp : IME_IntrOpBase<"vmadot3us">,
  Arguments<(ins LLVM_Type:$vd, LLVM_Type:$vs1, LLVM_Type:$vs2)>;

// slide-n intrinsics (with dynamic slide value in GPR)
def IME_Vmadotn_IntrOp : IME_IntrOpBase<"vmadotn">,
  Arguments<(ins LLVM_Type:$vd, LLVM_Type:$vs1, LLVM_Type:$vs2, LLVM_Type:$slide)>;

def IME_Vmadotnu_IntrOp : IME_IntrOpBase<"vmadotnu">,
  Arguments<(ins LLVM_Type:$vd, LLVM_Type:$vs1, LLVM_Type:$vs2, LLVM_Type:$slide)>;

def IME_Vmadotnsu_IntrOp : IME_IntrOpBase<"vmadotnsu">,
  Arguments<(ins LLVM_Type:$vd, LLVM_Type:$vs1, LLVM_Type:$vs2, LLVM_Type:$slide)>;

def IME_Vmadotnus_IntrOp : IME_IntrOpBase<"vmadotnus">,
  Arguments<(ins LLVM_Type:$vd, LLVM_Type:$vs1, LLVM_Type:$vs2, LLVM_Type:$slide)>;

//===----------------------------------------------------------------------===//
// Sliding-window Floating-Point Matrix Multiply-Accumulate Intrinsics
//===----------------------------------------------------------------------===//

def IME_Vfmadot1_IntrOp : IME_IntrOpBase<"vfmadot1">,
  Arguments<(ins LLVM_Type:$vd, LLVM_Type:$vs1, LLVM_Type:$vs2)>;

def IME_Vfmadot2_IntrOp : IME_IntrOpBase<"vfmadot2">,
  Arguments<(ins LLVM_Type:$vd, LLVM_Type:$vs1, LLVM_Type:$vs2)>;

def IME_Vfmadot3_IntrOp : IME_IntrOpBase<"vfmadot3">,
  Arguments<(ins LLVM_Type:$vd, LLVM_Type:$vs1, LLVM_Type:$vs2)>;

def IME_Vfmadotn_IntrOp : IME_IntrOpBase<"vfmadotn">,
  Arguments<(ins LLVM_Type:$vd, LLVM_Type:$vs1, LLVM_Type:$vs2, LLVM_Type:$slide)>;

#endif // IME_DIALECT
