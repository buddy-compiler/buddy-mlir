#ifndef GEMMINI_OPS
#define GEMMINI_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/PatternBase.td"

def Gemmini_Dialect : Dialect {
  let name = "gemmini";
  let cppNamespace = "::buddy::gemmini";
  let summary = "Basic dialect to target RISC-V gemmini extension";
  let description = [{gemmini}];
}

class Gemmini_Op<string mnemonic, list<Trait> traits = []> :
  Op<Gemmini_Dialect, mnemonic, traits> {}

def ConfigStOp : Gemmini_Op<"configSt"> {
  let summary = "ConfigOp operation";
  let arguments = (ins I64:$stride, DefaultValuedAttr<I64Attr, "0">:$activation);  
  let assemblyFormat = "$stride attr-dict `:` type($stride)";
}

def ConfigLdOp : Gemmini_Op<"configLd"> {
  let summary = "ConfigLd operation";
  let arguments = (ins I64:$stride);
  let assemblyFormat = "$stride attr-dict `:` type($stride)";
}

def MvinOp : Gemmini_Op<"mvin"> {
  let summary = "Mvin operation";
  let arguments = (ins I64:$input, I64:$addr);
  let assemblyFormat = "$input $addr attr-dict `:` type($input) type($addr)";
}

def MvoutOp : Gemmini_Op<"mvout"> {
  let summary = "Mvout operation";
  let arguments = (ins I64:$output, I64:$addr);
  let assemblyFormat = "$output $addr attr-dict `:` type($output) type($addr)";
}

def PrintOp : Gemmini_Op<"print"> {
  let summary = "print operation";
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

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