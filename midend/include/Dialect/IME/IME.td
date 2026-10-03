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
  let useStrictPropertiesInAssemblyFormat = 0;
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
// SSA vectors model values, never physical registers. The LLVM backend chooses
// register groups and enforces tied accumulators / parameter-register classes.
class IME_IntrOpBase<string mnemonic, list<Trait> traits = []> :
  LLVM_IntrOpBase<IME_Dialect, "intr." # mnemonic,
                  "riscv_ime_" # !subst(".", "_", mnemonic),
                  [0], [1, 2], !listconcat([Pure], traits), 1> {
  let hasVerifier = 1;
}

// Integer matrix multiply-accumulate intrinsics
def IME_Vmadot_IntrOp : IME_IntrOpBase<"vmadot">,
  Arguments<(ins LLVM_AnyScalableVector:$vd, LLVM_AnyScalableVector:$vs1, LLVM_AnyScalableVector:$vs2)>;

def IME_Vmadotu_IntrOp : IME_IntrOpBase<"vmadotu">,
  Arguments<(ins LLVM_AnyScalableVector:$vd, LLVM_AnyScalableVector:$vs1, LLVM_AnyScalableVector:$vs2)>;

def IME_Vmadotsu_IntrOp : IME_IntrOpBase<"vmadotsu">,
  Arguments<(ins LLVM_AnyScalableVector:$vd, LLVM_AnyScalableVector:$vs1, LLVM_AnyScalableVector:$vs2)>;

def IME_Vmadotus_IntrOp : IME_IntrOpBase<"vmadotus">,
  Arguments<(ins LLVM_AnyScalableVector:$vd, LLVM_AnyScalableVector:$vs1, LLVM_AnyScalableVector:$vs2)>;

// Floating-point matrix multiply-accumulate intrinsic
def IME_Vfmadot_IntrOp : IME_IntrOpBase<"vfmadot">,
  Arguments<(ins LLVM_AnyScalableVector:$vd, LLVM_AnyScalableVector:$vs1, LLVM_AnyScalableVector:$vs2)>;

// Native A100 fixed signatures require VLEN=1024 (vscale=16). zvl1024b
// guarantees only a minimum; the runtime must check vlenb=128 on an A100 core.
class IME_A100Vector<int lanes, string element> : Type<CPred<
  "::llvm::isa<::mlir::VectorType>($_self) && "
  "::llvm::cast<::mlir::VectorType>($_self) == "
  "::mlir::VectorType::get({" # lanes # "}, " # element # ", true)">,
  "A100 scalable tile vector">;
def IME_A100I8 : IME_A100Vector<8, "::mlir::IntegerType::get($_self.getContext(), 8)">;
def IME_A100I8Pair : IME_A100Vector<16, "::mlir::IntegerType::get($_self.getContext(), 8)">;
def IME_A100I32Pair : IME_A100Vector<4, "::mlir::IntegerType::get($_self.getContext(), 32)">;
def IME_A100F16 : IME_A100Vector<4, "::mlir::Float16Type::get($_self.getContext())">;

class IME_A100IntrOp<string mnemonic, list<int> results = [],
                    list<int> operands = [], list<int> immPositions = [],
                    list<string> immNames = []> :
  LLVM_IntrOpBase<IME_Dialect, "intr." # mnemonic,
                  "riscv_ime_" # !subst(".", "_", mnemonic),
                  results, operands, [Pure], 1, 0, 0, 0, 0, 0,
                  immPositions, immNames>;

class IME_HPIntrOp<string mnemonic> :
    IME_A100IntrOp<mnemonic # ".hp", [], [], [4], ["group"]> {
  let summary = "A100 int8 dot with per-column FP16 scaling and FP16 accumulation";
  let description = [{
    C += dot(A, transposed B) * scales[group*8 + column].
    C and scales each occupy one register. The runtime must establish
    MCPM.BF16=0; LLVM allocates scales to v0/v1. group is 0..7.
  }];
  let arguments = (ins IME_A100F16:$vd, IME_A100I8:$vs1,
                       IME_A100I8:$vs2, IME_A100F16:$scales,
                       ConfinedAttr<I32Attr, [IntMinValue<0>, IntMaxValue<7>]>:$group);
  let results = (outs IME_A100F16:$res);
}
class IME_SPIntrOp<string mnemonic> :
    IME_A100IntrOp<mnemonic # ".sp", [], [], [4], ["group"]> {
  let summary = "A100 4:2 sparse int8 dot with int32 accumulation";
  let arguments = (ins IME_A100I32Pair:$vd, IME_A100I8Pair:$vs1,
                       IME_A100I8:$vs2, IME_A100I8:$parameters,
                       ConfinedAttr<I32Attr, [IntMinValue<0>, IntMaxValue<3>]>:$group);
  let results = (outs IME_A100I32Pair:$res);
}
foreach suffix = ["", "u", "su", "us"] in {
  def IME_Vmadot # suffix # _HP_IntrOp : IME_HPIntrOp<"vmadot" # suffix>;
  def IME_Vmadot # suffix # _SP_IntrOp : IME_SPIntrOp<"vmadot" # suffix>;
}

class IME_PackIntrOp<string mnemonic, list<int> results, list<int> operands> :
    IME_A100IntrOp<mnemonic, results, operands, [2], ["block"]> {
  let hasVerifier = 1;
  let arguments = (ins AnyVectorOfAnyRank:$vs1, AnyVectorOfAnyRank:$vs2,
                       ConfinedAttr<I32Attr, [IntMinValue<0>, IntMaxValue<3>]>:$block);
  let results = (outs AnyVectorOfAnyRank:$res);
}
def IME_Vpack_IntrOp : IME_PackIntrOp<"vpack", [0], []>;
def IME_Vupack_IntrOp : IME_PackIntrOp<"vupack", [0], []>;
def IME_Vnpack_IntrOp : IME_PackIntrOp<"vnpack", [], [0]>;
def IME_Vnspack_IntrOp : IME_PackIntrOp<"vnspack", [], [0]>;
def IME_Vnpack4_IntrOp : IME_PackIntrOp<"vnpack4", [], []>;
def IME_Vnspack4_IntrOp : IME_PackIntrOp<"vnspack4", [], []>;

//===----------------------------------------------------------------------===//
// Sliding-window Integer Matrix Multiply-Accumulate Intrinsics
//===----------------------------------------------------------------------===//

// slide-1 intrinsics
def IME_Vmadot1_IntrOp : IME_IntrOpBase<"vmadot1">,
  Arguments<(ins LLVM_AnyScalableVector:$vd, LLVM_AnyScalableVector:$vs1, LLVM_AnyScalableVector:$vs2)>;

def IME_Vmadot1u_IntrOp : IME_IntrOpBase<"vmadot1u">,
  Arguments<(ins LLVM_AnyScalableVector:$vd, LLVM_AnyScalableVector:$vs1, LLVM_AnyScalableVector:$vs2)>;

def IME_Vmadot1su_IntrOp : IME_IntrOpBase<"vmadot1su">,
  Arguments<(ins LLVM_AnyScalableVector:$vd, LLVM_AnyScalableVector:$vs1, LLVM_AnyScalableVector:$vs2)>;

def IME_Vmadot1us_IntrOp : IME_IntrOpBase<"vmadot1us">,
  Arguments<(ins LLVM_AnyScalableVector:$vd, LLVM_AnyScalableVector:$vs1, LLVM_AnyScalableVector:$vs2)>;

// slide-2 intrinsics
def IME_Vmadot2_IntrOp : IME_IntrOpBase<"vmadot2">,
  Arguments<(ins LLVM_AnyScalableVector:$vd, LLVM_AnyScalableVector:$vs1, LLVM_AnyScalableVector:$vs2)>;

def IME_Vmadot2u_IntrOp : IME_IntrOpBase<"vmadot2u">,
  Arguments<(ins LLVM_AnyScalableVector:$vd, LLVM_AnyScalableVector:$vs1, LLVM_AnyScalableVector:$vs2)>;

def IME_Vmadot2su_IntrOp : IME_IntrOpBase<"vmadot2su">,
  Arguments<(ins LLVM_AnyScalableVector:$vd, LLVM_AnyScalableVector:$vs1, LLVM_AnyScalableVector:$vs2)>;

def IME_Vmadot2us_IntrOp : IME_IntrOpBase<"vmadot2us">,
  Arguments<(ins LLVM_AnyScalableVector:$vd, LLVM_AnyScalableVector:$vs1, LLVM_AnyScalableVector:$vs2)>;

// slide-3 intrinsics
def IME_Vmadot3_IntrOp : IME_IntrOpBase<"vmadot3">,
  Arguments<(ins LLVM_AnyScalableVector:$vd, LLVM_AnyScalableVector:$vs1, LLVM_AnyScalableVector:$vs2)>;

def IME_Vmadot3u_IntrOp : IME_IntrOpBase<"vmadot3u">,
  Arguments<(ins LLVM_AnyScalableVector:$vd, LLVM_AnyScalableVector:$vs1, LLVM_AnyScalableVector:$vs2)>;

def IME_Vmadot3su_IntrOp : IME_IntrOpBase<"vmadot3su">,
  Arguments<(ins LLVM_AnyScalableVector:$vd, LLVM_AnyScalableVector:$vs1, LLVM_AnyScalableVector:$vs2)>;

def IME_Vmadot3us_IntrOp : IME_IntrOpBase<"vmadot3us">,
  Arguments<(ins LLVM_AnyScalableVector:$vd, LLVM_AnyScalableVector:$vs1, LLVM_AnyScalableVector:$vs2)>;

// slide-n intrinsics (with dynamic slide value in GPR)
def IME_Vmadotn_IntrOp : IME_IntrOpBase<"vmadotn">,
  Arguments<(ins LLVM_AnyScalableVector:$vd, LLVM_AnyScalableVector:$vs1, LLVM_AnyScalableVector:$vs2, I64:$slide)>;

def IME_Vmadotnu_IntrOp : IME_IntrOpBase<"vmadotnu">,
  Arguments<(ins LLVM_AnyScalableVector:$vd, LLVM_AnyScalableVector:$vs1, LLVM_AnyScalableVector:$vs2, I64:$slide)>;

def IME_Vmadotnsu_IntrOp : IME_IntrOpBase<"vmadotnsu">,
  Arguments<(ins LLVM_AnyScalableVector:$vd, LLVM_AnyScalableVector:$vs1, LLVM_AnyScalableVector:$vs2, I64:$slide)>;

def IME_Vmadotnus_IntrOp : IME_IntrOpBase<"vmadotnus">,
  Arguments<(ins LLVM_AnyScalableVector:$vd, LLVM_AnyScalableVector:$vs1, LLVM_AnyScalableVector:$vs2, I64:$slide)>;

//===----------------------------------------------------------------------===//
// Sliding-window Floating-Point Matrix Multiply-Accumulate Intrinsics
//===----------------------------------------------------------------------===//

def IME_Vfmadot1_IntrOp : IME_IntrOpBase<"vfmadot1">,
  Arguments<(ins LLVM_AnyScalableVector:$vd, LLVM_AnyScalableVector:$vs1, LLVM_AnyScalableVector:$vs2)>;

def IME_Vfmadot2_IntrOp : IME_IntrOpBase<"vfmadot2">,
  Arguments<(ins LLVM_AnyScalableVector:$vd, LLVM_AnyScalableVector:$vs1, LLVM_AnyScalableVector:$vs2)>;

def IME_Vfmadot3_IntrOp : IME_IntrOpBase<"vfmadot3">,
  Arguments<(ins LLVM_AnyScalableVector:$vd, LLVM_AnyScalableVector:$vs1, LLVM_AnyScalableVector:$vs2)>;

def IME_Vfmadotn_IntrOp : IME_IntrOpBase<"vfmadotn">,
  Arguments<(ins LLVM_AnyScalableVector:$vd, LLVM_AnyScalableVector:$vs1, LLVM_AnyScalableVector:$vs2, I64:$slide)>;

#endif // IME_DIALECT
