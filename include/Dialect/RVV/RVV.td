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
// RVV LMUL type definitions
//===----------------------------------------------------------------------===//

class RVV_LMULType<string name, string typeMnemonic, string multiplier, 
                   list<Trait> traits = []>
    : TypeDef<RVV_Dialect, name, traits, "::mlir::RVVLMULType"> { 
  let mnemonic = typeMnemonic;
  let summary = "The vector register group multiplier (" # multiplier # ").";
  let parser = [{return get}]# name #[{($_ctxt);}];
}

def RVV_LMUL_MF8 : RVV_LMULType<"MF8", "mf8", "1/8"> {}
def RVV_LMUL_MF4 : RVV_LMULType<"MF4", "mf4", "1/4"> {}
def RVV_LMUL_MF2 : RVV_LMULType<"MF2", "mf2", "1/2"> {}
def RVV_LMUL_M1 : RVV_LMULType<"M1", "m1", "1"> {}
def RVV_LMUL_M2 : RVV_LMULType<"M2", "m2", "2"> {}
def RVV_LMUL_M4 : RVV_LMULType<"M4", "m4", "4"> {}
def RVV_LMUL_M8 : RVV_LMULType<"M8", "m8", "8"> {}

//===----------------------------------------------------------------------===//
// RVV mask type definitions
//===----------------------------------------------------------------------===//

class RVV_MaskType<string name, string typeMnemonic, string maskLength,
                   list<Trait> traits = []>
    : TypeDef<RVV_Dialect, name, traits, "::mlir::RVVMaskType"> { 
  let mnemonic = typeMnemonic;
  let summary = "The mask length (SEW/LMUL = " # maskLength # ") type.";
  let parser = [{return get}]# name #[{($_ctxt);}];
}

def RVV_Mask1 : RVV_MaskType<"Mask1", "mask1", "1"> {}
def RVV_Mask2 : RVV_MaskType<"Mask2", "mask2", "2"> {}
def RVV_Mask4 : RVV_MaskType<"Mask4", "mask4", "4"> {}
def RVV_Mask8 : RVV_MaskType<"Mask8", "mask8", "8"> {}
def RVV_Mask16 : RVV_MaskType<"Mask16", "mask16", "16"> {}
def RVV_Mask32 : RVV_MaskType<"Mask32", "mask32", "32"> {}
def RVV_Mask64 : RVV_MaskType<"Mask64", "mask64", "64"> {}

//===----------------------------------------------------------------------===//
// RVV scalable vector type definitions
//===----------------------------------------------------------------------===//

def ScalableVectorType : ScalableVector_Type<RVV_Dialect, "ScalableVector"> {
  let description = [{
    RVV scalable vector type takes two parameters. The first one is vector
    register group multiplier (LMUL) type or mask type. The second one is the
    element type, which indicates the selected element width (SEW) setting. 
    The LMUL and SEW are used to configure scalable vector length at runtime.
  }];

  let parameters = (ins "Type":$sizeType, "Type":$elementType);

  let printer = [{
    $_printer << "<" << getImpl()->sizeType << ',';
    $_printer << getImpl()->elementType << '>';
  }];

  let parser = [{
    if ($_parser.parseLess()) return Type();
    Type sizeType;
    if ($_parser.parseType(sizeType)) return Type();
    if ($_parser.parseComma()) return Type();
    Type elementType;
    if ($_parser.parseType(elementType)) return Type();
    if ($_parser.parseGreater()) return Type();
    return get($_ctxt, sizeType, elementType);
  }];
}
//===----------------------------------------------------------------------===//
// Additional LLVM type constraints
//===----------------------------------------------------------------------===//

def LLVMScalableVectorType :
  Type<CPred<"$_self.isa<::mlir::LLVM::LLVMScalableVectorType>()">,
       "LLVM dialect scalable vector type">;

def LLVMPointerType :
  Type<CPred<"$_self.isa<::mlir::LLVM::LLVMPointerType>()">,
       "LLVM pointer type">;

//===----------------------------------------------------------------------===//
// RVV scalable vector type constraints
//===----------------------------------------------------------------------===//

def IsRVVScalableVectorTypePred :
    CPred<"$_self.isa<::buddy::rvv::ScalableVectorType>()">;

class RVVScalableVectorOf<list<Type> allowedTypes> :
    ContainerType<AnyTypeOf<allowedTypes>, IsRVVScalableVectorTypePred,
          "$_self.cast<::buddy::rvv::ScalableVectorType>().getElementType()",
          "RVV scalable vector">;

//===----------------------------------------------------------------------===//
// RVV operation definitions
//===----------------------------------------------------------------------===//

class RVV_Op<string mnemonic, list<OpTrait> traits = []> :
  Op<RVV_Dialect, mnemonic, traits> {}

def RVVLoadOp : RVV_Op<"load">,
    Arguments<(ins Arg<AnyMemRef, "", [MemRead]>:$base, Index:$index,
                       AnyInteger:$length)>,
    Results<(outs RVVScalableVectorOf<[AnyType]>:$result)> {
  let summary = "Load scalable vector from memory";
  let description = [{
    Load a slice of memory into scalable vector with the given element length.
  }];
  let extraClassDeclaration = [{
    mlir::MemRefType getMemRefType() {
      return base().getType().cast<mlir::MemRefType>();
    }
  }];
  let assemblyFormat = "$base `[` $index `]` `,` $length attr-dict `:` "
    "type($base) `,` type($result) `,`  type($length)";
}

def RVVStoreOp : RVV_Op<"store">,
    Arguments<(ins RVVScalableVectorOf<[AnyType]>:$value, Arg<AnyMemRef, "",
                   [MemWrite]>:$base, Index:$index, AnyInteger:$length)> {
  let summary = "Store scalable vector into memory";
  let description = [{
    Store the given element length of a scalable vector on a slice of memory.
  }];
  let extraClassDeclaration = [{
    mlir::MemRefType getMemRefType() {
      return base().getType().cast<mlir::MemRefType>();
    }
  }];
  let assemblyFormat = "$value `,` $base `[` $index `]` `,` $length attr-dict "
    "`:` type($value) `,` type($base) `,`  type($length)";
}

class RVV_BinaryAAXNoMask_Op<string mnemonic, string op_description,
                             list<OpTrait> traits = []> :
  RVV_Op<mnemonic, !listconcat(traits, [AllTypesMatch<["src1", "dst"]>])> {
  let summary = op_description # "for RVV scalable vectors";
  let description = [{ The `rvv.}] # mnemonic # [{` operation can be of 
    vector-vector form or vector-scalar form. It also takes the vector length 
    value and returns a scalable vector with the result of the }]
    # op_description # [{.}];
  let arguments = (ins 
          RVVScalableVectorOf<[AnyInteger]>:$src1,
          AnyType:$src2,
          AnyInteger:$length
  );
  let results = (outs RVVScalableVectorOf<[AnyInteger]>:$dst);
  let assemblyFormat = "$src1 `,` $src2 `,` $length attr-dict `:` type($src1) "
    "`,` type($src2) `,` type($length)";
}

class RVV_BinaryAAXMask_Op<string mnemonic, string op_description,
                              list<OpTrait> traits = []> :
  RVV_Op<mnemonic, !listconcat(traits,
                       [AllTypesMatch<["maskedoff", "src1", "dst"]>,
                        AllTypesMatch<["length", "vta"]>])> {
  let summary = op_description # "for RVV scalable vectors";
  let description = [{ The `rvv.}] # mnemonic # [{` operation can be of 
    vector-vector form or vector-scalar form. It also takes the mask vector, 
    maskedoff vector, vector length value and returns a scalable vector with 
    the result of the }] # op_description # [{.}];
  let arguments = (ins 
          RVVScalableVectorOf<[AnyInteger]>:$maskedoff,
          RVVScalableVectorOf<[AnyInteger]>:$src1,
          AnyType:$src2,
          RVVScalableVectorOf<[I1]>:$mask,
          AnyInteger:$length,
          AnyInteger:$vta
  );
  let results = (outs RVVScalableVectorOf<[AnyInteger]>:$dst);
  let assemblyFormat = 
    "$maskedoff `,` $src1 `,` $src2 `,` $mask `,` $length `,` $vta attr-dict "
    "`:` type($maskedoff) `,` type($src2) `,` type($mask) `,` type($length)";
}

def RVVAddOp : RVV_BinaryAAXNoMask_Op<"add", "addition">;
def RVVSubOp : RVV_BinaryAAXNoMask_Op<"sub", "subtraction">;
def RVVMulOp : RVV_BinaryAAXNoMask_Op<"mul", "multiplication">;
def RVVDivOp : RVV_BinaryAAXNoMask_Op<"div", "division">;

def RVVMaskedAddOp : RVV_BinaryAAXMask_Op<"masked.add", "masked addition">;
def RVVMaskedSubOp : RVV_BinaryAAXMask_Op<"masked.sub", "masked subtraction">;
def RVVMaskedMulOp : RVV_BinaryAAXMask_Op<"masked.mul",
                                          "masked multiplication">;
def RVVMaskedDivOp : RVV_BinaryAAXMask_Op<"masked.div", "masked division">;

//===----------------------------------------------------------------------===//
// RVV intrinsic operation definitions
//===----------------------------------------------------------------------===//

class RVV_USLoad_IntrOp<string mnemonic, list<OpTrait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/RVV_Dialect, 
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[0], 
                  /*list<int> overloadedOperands=*/[1], 
                  /*list<OpTrait> traits=*/traits, 
                  /*int numResults=*/1>;

class RVV_USStore_IntrOp<string mnemonic, list<OpTrait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/RVV_Dialect, 
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[], 
                  /*list<int> overloadedOperands=*/[0, 2], 
                  /*list<OpTrait> traits=*/traits, 
                  /*int numResults=*/0>;

class RVV_BinaryAAXNoMask_IntrOp<string mnemonic,
                                    list<OpTrait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/RVV_Dialect, 
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[0], 
                  /*list<int> overloadedOperands=*/[1, 2], 
                  /*list<OpTrait> traits=*/traits, 
                  /*int numResults=*/1>;

class RVV_BinaryAAXMask_IntrOp<string mnemonic, list<OpTrait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/RVV_Dialect, 
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[0], 
                  /*list<int> overloadedOperands=*/[2, 4], 
                  /*list<OpTrait> traits=*/traits, 
                  /*int numResults=*/1>;

def RVVIntrLoadEleOp : RVV_USLoad_IntrOp<"vle">,
  Arguments<(ins LLVMPointerType, AnyInteger)>;
def RVVIntrStoreEleOp : RVV_USStore_IntrOp<"vse">,
  Arguments<(ins LLVMScalableVectorType, LLVMPointerType, AnyInteger)>;

def RVVIntrAddOp : RVV_BinaryAAXNoMask_IntrOp<"vadd">, 
  Arguments<(ins LLVMScalableVectorType, AnyType, AnyInteger)>;
def RVVIntrSubOp : RVV_BinaryAAXNoMask_IntrOp<"vsub">, 
  Arguments<(ins LLVMScalableVectorType, AnyType, AnyInteger)>;
def RVVIntrMulOp : RVV_BinaryAAXNoMask_IntrOp<"vmul">, 
  Arguments<(ins LLVMScalableVectorType, AnyType, AnyInteger)>;
def RVVIntrDivOp : RVV_BinaryAAXNoMask_IntrOp<"vdiv">, 
  Arguments<(ins LLVMScalableVectorType, AnyType, AnyInteger)>;

def RVVMaskedIntrAddOp : RVV_BinaryAAXMask_IntrOp<"vadd_mask">, 
  Arguments<(ins LLVMScalableVectorType, LLVMScalableVectorType, AnyType,
                 LLVMScalableVectorType, AnyInteger, AnyInteger)>;
def RVVMaskedIntrSubOp : RVV_BinaryAAXMask_IntrOp<"vsub_mask">, 
  Arguments<(ins LLVMScalableVectorType, LLVMScalableVectorType, AnyType, 
                 LLVMScalableVectorType, AnyInteger, AnyInteger)>;
def RVVMaskedIntrMulOp : RVV_BinaryAAXMask_IntrOp<"vmul_mask">, 
  Arguments<(ins LLVMScalableVectorType, LLVMScalableVectorType, AnyType, 
                 LLVMScalableVectorType, AnyInteger, AnyInteger)>;
def RVVMaskedIntrDivOp : RVV_BinaryAAXMask_IntrOp<"vdiv_mask">, 
  Arguments<(ins LLVMScalableVectorType, LLVMScalableVectorType, AnyType, 
                 LLVMScalableVectorType, AnyInteger, AnyInteger)>;

#endif // RVV_OPS
