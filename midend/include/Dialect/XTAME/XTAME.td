//====--------- XTAME.td - XTAME dialect operation definitions ------------===//
//
// Matrix values are SSA vectors. MatrixReg physical-register selection happens
// in the RISC-V backend, so XTAME operations never carry m0-m7 indices.
//
//===----------------------------------------------------------------------===//

#ifndef XTAME_DIALECT
#define XTAME_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

def XTAME_Dialect : Dialect {
  let name = "xt_ame";
  let cppNamespace = "::buddy::xtame";
  let description = [{
    SSA-based operations for the XuanTie AME matrix extension. Matrix values
    lower to scalable LLVM vectors that are assigned to MatrixReg.
  }];
}

def XTAME_MatrixType : Type<
    CPred<"::mlir::isa<::mlir::VectorType>($_self)">,
    "XTHead AME matrix vector type">;

class XTAME_Op<string mnemonic, list<Trait> traits = []> :
    Op<XTAME_Dialect, mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// Matrix configuration
//===----------------------------------------------------------------------===//

def ThMcfgOp : XTAME_Op<"th.mcfg"> {
  let arguments = (ins I64:$mtype);
  let assemblyFormat = "$mtype attr-dict `:` type($mtype)";
}
def ThMcfgmOp : XTAME_Op<"th.mcfgm"> {
  let arguments = (ins I64:$tilem);
  let assemblyFormat = "$tilem attr-dict `:` type($tilem)";
}
def ThMcfgnOp : XTAME_Op<"th.mcfgn"> {
  let arguments = (ins I64:$tilen);
  let assemblyFormat = "$tilen attr-dict `:` type($tilen)";
}
def ThMcfgkOp : XTAME_Op<"th.mcfgk"> {
  let arguments = (ins I64:$tilek);
  let assemblyFormat = "$tilek attr-dict `:` type($tilek)";
}
def ThMcfgmiOp : XTAME_Op<"th.mcfgmi"> {
  let arguments = (ins I64Attr:$tilem);
  let assemblyFormat = "$tilem attr-dict";
}
def ThMcfgniOp : XTAME_Op<"th.mcfgni"> {
  let arguments = (ins I64Attr:$tilen);
  let assemblyFormat = "$tilen attr-dict";
}
def ThMcfgkiOp : XTAME_Op<"th.mcfgki"> {
  let arguments = (ins I64Attr:$tilek);
  let assemblyFormat = "$tilek attr-dict";
}

//===----------------------------------------------------------------------===//
// Matrix data movement
//===----------------------------------------------------------------------===//

class XTAME_Zero_Op<string mnemonic> : XTAME_Op<mnemonic> {
  let results = (outs XTAME_MatrixType:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}
class XTAME_MatrixUnary_Op<string mnemonic>
    : XTAME_Op<mnemonic, [AllTypesMatch<["src", "res"]>]> {
  let arguments = (ins XTAME_MatrixType:$src);
  let results = (outs XTAME_MatrixType:$res);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($res)";
}
class XTAME_MatrixScalar_Op<string mnemonic> : XTAME_Op<mnemonic> {
  let arguments = (ins I64:$scalar);
  let results = (outs XTAME_MatrixType:$res);
  let assemblyFormat = "$scalar attr-dict `:` type($scalar) `->` type($res)";
}
class XTAME_MatrixScalarIndexed_Op<string mnemonic> : XTAME_Op<mnemonic> {
  let arguments = (ins I64:$scalar, I64:$index);
  let results = (outs XTAME_MatrixType:$res);
  let assemblyFormat = "$scalar `,` $index attr-dict `:` type($scalar) `,` type($index) `->` type($res)";
}
class XTAME_ScalarMatrixIndexed_Op<string mnemonic> : XTAME_Op<mnemonic> {
  let arguments = (ins XTAME_MatrixType:$src, I64:$index);
  let results = (outs I64:$res);
  let assemblyFormat = "$src `,` $index attr-dict `:` type($src) `,` type($index) `->` type($res)";
}
class XTAME_MatrixBroadcast_Op<string mnemonic>
    : XTAME_Op<mnemonic, [AllTypesMatch<["src", "res"]>]> {
  let arguments = (ins XTAME_MatrixType:$src, I64Attr:$uimm3);
  let results = (outs XTAME_MatrixType:$res);
  let assemblyFormat = "$src `[` $uimm3 `]` attr-dict `:` type($src) `->` type($res)";
}
class XTAME_MatrixBinary_Op<string mnemonic>
    : XTAME_Op<mnemonic, [AllTypesMatch<["lhs", "rhs", "res"]>]> {
  let arguments = (ins XTAME_MatrixType:$lhs, XTAME_MatrixType:$rhs);
  let results = (outs XTAME_MatrixType:$res);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($res)";
}

def ThMzeroOp : XTAME_Zero_Op<"th.mzero">;
def ThMzero2rOp : XTAME_Zero_Op<"th.mzero2r">;
def ThMzero4rOp : XTAME_Zero_Op<"th.mzero4r">;
def ThMzero8rOp : XTAME_Zero_Op<"th.mzero8r">;
def ThMmovMmOp : XTAME_MatrixUnary_Op<"th.mmov.mm">;
def ThMdupbMXOp : XTAME_MatrixScalar_Op<"th.mdupb.m.x">;
def ThMduphMXOp : XTAME_MatrixScalar_Op<"th.mduph.m.x">;
def ThMdupwMXOp : XTAME_MatrixScalar_Op<"th.mdupw.m.x">;
def ThMdupdMXOp : XTAME_MatrixScalar_Op<"th.mdupd.m.x">;
def ThMmovbMXOp : XTAME_MatrixScalarIndexed_Op<"th.mmovb.m.x">;
def ThMmovhMXOp : XTAME_MatrixScalarIndexed_Op<"th.mmovh.m.x">;
def ThMmovwMXOp : XTAME_MatrixScalarIndexed_Op<"th.mmovw.m.x">;
def ThMmovdMXOp : XTAME_MatrixScalarIndexed_Op<"th.mmovd.m.x">;
def ThMmovbXMOp : XTAME_ScalarMatrixIndexed_Op<"th.mmovb.x.m">;
def ThMmovhXMOp : XTAME_ScalarMatrixIndexed_Op<"th.mmovh.x.m">;
def ThMmovwXMOp : XTAME_ScalarMatrixIndexed_Op<"th.mmovw.x.m">;
def ThMmovdXMOp : XTAME_ScalarMatrixIndexed_Op<"th.mmovd.x.m">;
def ThMmovMvIOp : XTAME_MatrixBroadcast_Op<"th.mmov.mv.i">;
def ThMcmovbMvIOp : XTAME_MatrixBroadcast_Op<"th.mcmovb.mv.i">;
def ThMcmovhMvIOp : XTAME_MatrixBroadcast_Op<"th.mcmovh.mv.i">;
def ThMcmovwMvIOp : XTAME_MatrixBroadcast_Op<"th.mcmovw.mv.i">;
def ThMcmovdMvIOp : XTAME_MatrixBroadcast_Op<"th.mcmovd.mv.i">;
def ThMpackMmOp : XTAME_MatrixBinary_Op<"th.mpack.mm">;
def ThMpackhlMmOp : XTAME_MatrixBinary_Op<"th.mpackhl.mm">;
def ThMpackhhMmOp : XTAME_MatrixBinary_Op<"th.mpackhh.mm">;

//===----------------------------------------------------------------------===//
// Memory operations
//===----------------------------------------------------------------------===//

class XTAME_Load_Op<string mnemonic> : XTAME_Op<mnemonic> {
  let arguments = (ins I64:$stride, AnyMemRef:$base);
  let results = (outs XTAME_MatrixType:$res);
  let assemblyFormat = "$stride `,` $base attr-dict `:` type($base) `->` type($res)";
}
class XTAME_Prefetch_Op<string mnemonic> : XTAME_Op<mnemonic> {
  let arguments = (ins I64:$stride, AnyMemRef:$base);
  let assemblyFormat = "$stride `,` $base attr-dict `:` type($base)";
}
class XTAME_Store_Op<string mnemonic> : XTAME_Op<mnemonic> {
  let arguments = (ins XTAME_MatrixType:$src, I64:$stride, AnyMemRef:$base);
  let assemblyFormat = "$src `,` $stride `,` $base attr-dict `:` type($src) `,` type($base)";
}
multiclass XTAME_Load_Ops<string prefix> {
  def "8Op" : XTAME_Load_Op<prefix # "8">;
  def "16Op" : XTAME_Load_Op<prefix # "16">;
  def "32Op" : XTAME_Load_Op<prefix # "32">;
  def "64Op" : XTAME_Load_Op<prefix # "64">;
}
multiclass XTAME_Prefetch_Ops<string prefix> {
  def "8Op" : XTAME_Prefetch_Op<prefix # "8">;
  def "16Op" : XTAME_Prefetch_Op<prefix # "16">;
  def "32Op" : XTAME_Prefetch_Op<prefix # "32">;
  def "64Op" : XTAME_Prefetch_Op<prefix # "64">;
}
multiclass XTAME_Store_Ops<string prefix> {
  def "8Op" : XTAME_Store_Op<prefix # "8">;
  def "16Op" : XTAME_Store_Op<prefix # "16">;
  def "32Op" : XTAME_Store_Op<prefix # "32">;
  def "64Op" : XTAME_Store_Op<prefix # "64">;
}
defm ThMlde : XTAME_Load_Ops<"th.mlde">;
defm ThMldte : XTAME_Load_Ops<"th.mldte">;
defm ThMslde : XTAME_Load_Ops<"th.mslde">;
defm ThMsldte : XTAME_Load_Ops<"th.msldte">;
defm ThMplde : XTAME_Prefetch_Ops<"th.mplde">;
defm ThMpldte : XTAME_Prefetch_Ops<"th.mpldte">;
defm ThMste : XTAME_Store_Ops<"th.mste">;
defm ThMstte : XTAME_Store_Ops<"th.mstte">;
defm ThMsste : XTAME_Store_Ops<"th.msste">;
defm ThMsstte : XTAME_Store_Ops<"th.msstte">;

//===----------------------------------------------------------------------===//
// Matrix multiply-accumulate
//===----------------------------------------------------------------------===//

class XTAME_MMA_Op<string mnemonic>
    : XTAME_Op<mnemonic> {
  let arguments = (ins XTAME_MatrixType:$acc, XTAME_MatrixType:$rhs,
                       XTAME_MatrixType:$lhs);
  let results = (outs XTAME_MatrixType:$res);
  let assemblyFormat = "$acc `,` $rhs `,` $lhs attr-dict `:` type($acc) `,` type($rhs) `,` type($lhs) `->` type($res)";
}
class XTAME_WidenMMA_Op<string mnemonic>
    : XTAME_Op<mnemonic> {
  let arguments = (ins XTAME_MatrixType:$acc, XTAME_MatrixType:$rhs,
                       XTAME_MatrixType:$lhs);
  let results = (outs XTAME_MatrixType:$res);
  let assemblyFormat = "$acc `,` $rhs `,` $lhs attr-dict `:` type($acc) `,` type($rhs) `,` type($lhs) `->` type($res)";
}
def ThMmaccWBOp : XTAME_WidenMMA_Op<"th.mmacc.w.b">;
def ThMmaccuWBOp : XTAME_WidenMMA_Op<"th.mmaccu.w.b">;
def ThMmaccusWBOp : XTAME_WidenMMA_Op<"th.mmaccus.w.b">;
def ThMmaccsuWBOp : XTAME_WidenMMA_Op<"th.mmaccsu.w.b">;
def ThMfmaccHOp : XTAME_MMA_Op<"th.mfmacc.h">;
def ThMfmaccBf16Op : XTAME_MMA_Op<"th.mfmacc.bf16">;
def ThMfmaccSOp : XTAME_MMA_Op<"th.mfmacc.s">;
def ThMfmaccDOp : XTAME_MMA_Op<"th.mfmacc.d">;
def ThMfmaccHE4m3Op : XTAME_WidenMMA_Op<"th.mfmacc.h.e4m3">;
def ThMfmaccHE5m2Op : XTAME_WidenMMA_Op<"th.mfmacc.h.e5m2">;
def ThMfmaccBf16E4m3Op : XTAME_WidenMMA_Op<"th.mfmacc.bf16.e4m3">;
def ThMfmaccBf16E5m2Op : XTAME_WidenMMA_Op<"th.mfmacc.bf16.e5m2">;
def ThMfmaccSHOp : XTAME_WidenMMA_Op<"th.mfmacc.s.h">;
def ThMfmaccSBf16Op : XTAME_WidenMMA_Op<"th.mfmacc.s.bf16">;
def ThMfmaccDSOp : XTAME_WidenMMA_Op<"th.mfmacc.d.s">;
def ThMfmaccSE4m3Op : XTAME_WidenMMA_Op<"th.mfmacc.s.e4m3">;
def ThMfmaccSE5m2Op : XTAME_WidenMMA_Op<"th.mfmacc.s.e5m2">;

//===----------------------------------------------------------------------===//
// LLVM intrinsic bridge operations
//===----------------------------------------------------------------------===//

class XTAME_IntrOpBase<string mnemonic, int numResults = 0,
                       list<int> overloadedResults = [],
                       list<int> overloadedOperands = [],
                       list<Trait> traits = []> :
    LLVM_IntrOpBase</*Dialect dialect=*/XTAME_Dialect,
                   /*string opName=*/"intr." # mnemonic,
                   /*string enumName=*/"riscv_th_" # !subst(".", "_", !subst("th.", "", mnemonic)),
                   /*list<int> overloadedResults=*/overloadedResults,
                   /*list<int> overloadedOperands=*/overloadedOperands,
                   /*list<Trait> traits=*/traits,
                   /*int numResults=*/numResults>;
class XTAME_Config_IntrOp<string mnemonic>
    : XTAME_IntrOpBase<mnemonic, 0, [], [0]>, Arguments<(ins LLVM_Type:$operand)>;
def XTAME_ThMcfg_IntrOp : XTAME_Config_IntrOp<"th.mcfg">;
def XTAME_ThMcfgm_IntrOp : XTAME_Config_IntrOp<"th.mcfgm">;
def XTAME_ThMcfgn_IntrOp : XTAME_Config_IntrOp<"th.mcfgn">;
def XTAME_ThMcfgk_IntrOp : XTAME_Config_IntrOp<"th.mcfgk">;
def XTAME_ThMcfgmi_IntrOp : XTAME_Config_IntrOp<"th.mcfgmi">;
def XTAME_ThMcfgni_IntrOp : XTAME_Config_IntrOp<"th.mcfgni">;
def XTAME_ThMcfgki_IntrOp : XTAME_Config_IntrOp<"th.mcfgki">;

class XTAME_Zero_IntrOp<string mnemonic> : XTAME_IntrOpBase<mnemonic, 1, [0]>;
class XTAME_MatrixUnary_IntrOp<string mnemonic> : XTAME_IntrOpBase<mnemonic, 1, [0], [0]>, Arguments<(ins LLVM_Type:$src)>;
class XTAME_MatrixScalar_IntrOp<string mnemonic> : XTAME_IntrOpBase<mnemonic, 1, [0], [0]>, Arguments<(ins LLVM_Type:$scalar)>;
class XTAME_MatrixScalarIndexed_IntrOp<string mnemonic> : XTAME_IntrOpBase<mnemonic, 1, [0], [0, 1]>, Arguments<(ins LLVM_Type:$scalar, LLVM_Type:$index)>;
class XTAME_ScalarMatrixIndexed_IntrOp<string mnemonic> : XTAME_IntrOpBase<mnemonic, 1, [0], [0, 1]>, Arguments<(ins LLVM_Type:$src, LLVM_Type:$index)>;
class XTAME_MatrixBroadcast_IntrOp<string mnemonic> : XTAME_IntrOpBase<mnemonic, 1, [0], [1]>, Arguments<(ins LLVM_Type:$src, LLVM_Type:$uimm3)>;
class XTAME_MatrixBinary_IntrOp<string mnemonic> : XTAME_IntrOpBase<mnemonic, 1, [0], [0]>, Arguments<(ins LLVM_Type:$lhs, LLVM_Type:$rhs)>;

def XTAME_ThMzero_IntrOp : XTAME_Zero_IntrOp<"th.mzero">;
def XTAME_ThMzero2r_IntrOp : XTAME_Zero_IntrOp<"th.mzero2r">;
def XTAME_ThMzero4r_IntrOp : XTAME_Zero_IntrOp<"th.mzero4r">;
def XTAME_ThMzero8r_IntrOp : XTAME_Zero_IntrOp<"th.mzero8r">;
def XTAME_ThMmovMm_IntrOp : XTAME_MatrixUnary_IntrOp<"th.mmov.mm">;
def XTAME_ThMdupbMX_IntrOp : XTAME_MatrixScalar_IntrOp<"th.mdupb.m.x">;
def XTAME_ThMduphMX_IntrOp : XTAME_MatrixScalar_IntrOp<"th.mduph.m.x">;
def XTAME_ThMdupwMX_IntrOp : XTAME_MatrixScalar_IntrOp<"th.mdupw.m.x">;
def XTAME_ThMdupdMX_IntrOp : XTAME_MatrixScalar_IntrOp<"th.mdupd.m.x">;
def XTAME_ThMmovbMX_IntrOp : XTAME_MatrixScalarIndexed_IntrOp<"th.mmovb.m.x">;
def XTAME_ThMmovhMX_IntrOp : XTAME_MatrixScalarIndexed_IntrOp<"th.mmovh.m.x">;
def XTAME_ThMmovwMX_IntrOp : XTAME_MatrixScalarIndexed_IntrOp<"th.mmovw.m.x">;
def XTAME_ThMmovdMX_IntrOp : XTAME_MatrixScalarIndexed_IntrOp<"th.mmovd.m.x">;
def XTAME_ThMmovbXM_IntrOp : XTAME_ScalarMatrixIndexed_IntrOp<"th.mmovb.x.m">;
def XTAME_ThMmovhXM_IntrOp : XTAME_ScalarMatrixIndexed_IntrOp<"th.mmovh.x.m">;
def XTAME_ThMmovwXM_IntrOp : XTAME_ScalarMatrixIndexed_IntrOp<"th.mmovw.x.m">;
def XTAME_ThMmovdXM_IntrOp : XTAME_ScalarMatrixIndexed_IntrOp<"th.mmovd.x.m">;
def XTAME_ThMmovMvI_IntrOp : XTAME_MatrixBroadcast_IntrOp<"th.mmov.mv.i">;
def XTAME_ThMcmovbMvI_IntrOp : XTAME_MatrixBroadcast_IntrOp<"th.mcmovb.mv.i">;
def XTAME_ThMcmovhMvI_IntrOp : XTAME_MatrixBroadcast_IntrOp<"th.mcmovh.mv.i">;
def XTAME_ThMcmovwMvI_IntrOp : XTAME_MatrixBroadcast_IntrOp<"th.mcmovw.mv.i">;
def XTAME_ThMcmovdMvI_IntrOp : XTAME_MatrixBroadcast_IntrOp<"th.mcmovd.mv.i">;
def XTAME_ThMpackMm_IntrOp : XTAME_MatrixBinary_IntrOp<"th.mpack.mm">;
def XTAME_ThMpackhlMm_IntrOp : XTAME_MatrixBinary_IntrOp<"th.mpackhl.mm">;
def XTAME_ThMpackhhMm_IntrOp : XTAME_MatrixBinary_IntrOp<"th.mpackhh.mm">;

class XTAME_Load_IntrOp<string mnemonic> : XTAME_IntrOpBase<mnemonic, 1, [0], [0]>, Arguments<(ins LLVM_Type:$stride, LLVM_Type:$base)>;
class XTAME_Prefetch_IntrOp<string mnemonic> : XTAME_IntrOpBase<mnemonic, 0, [], [0]>, Arguments<(ins LLVM_Type:$stride, LLVM_Type:$base)>;
class XTAME_Store_IntrOp<string mnemonic> : XTAME_IntrOpBase<mnemonic, 0, [], [0, 1]>, Arguments<(ins LLVM_Type:$src, LLVM_Type:$stride, LLVM_Type:$base)>;
multiclass XTAME_Load_IntrOps<string prefix> {
  def "8_IntrOp" : XTAME_Load_IntrOp<prefix # "8">;
  def "16_IntrOp" : XTAME_Load_IntrOp<prefix # "16">;
  def "32_IntrOp" : XTAME_Load_IntrOp<prefix # "32">;
  def "64_IntrOp" : XTAME_Load_IntrOp<prefix # "64">;
}
multiclass XTAME_Prefetch_IntrOps<string prefix> {
  def "8_IntrOp" : XTAME_Prefetch_IntrOp<prefix # "8">;
  def "16_IntrOp" : XTAME_Prefetch_IntrOp<prefix # "16">;
  def "32_IntrOp" : XTAME_Prefetch_IntrOp<prefix # "32">;
  def "64_IntrOp" : XTAME_Prefetch_IntrOp<prefix # "64">;
}
multiclass XTAME_Store_IntrOps<string prefix> {
  def "8_IntrOp" : XTAME_Store_IntrOp<prefix # "8">;
  def "16_IntrOp" : XTAME_Store_IntrOp<prefix # "16">;
  def "32_IntrOp" : XTAME_Store_IntrOp<prefix # "32">;
  def "64_IntrOp" : XTAME_Store_IntrOp<prefix # "64">;
}
defm XTAME_ThMlde : XTAME_Load_IntrOps<"th.mlde">;
defm XTAME_ThMldte : XTAME_Load_IntrOps<"th.mldte">;
defm XTAME_ThMslde : XTAME_Load_IntrOps<"th.mslde">;
defm XTAME_ThMsldte : XTAME_Load_IntrOps<"th.msldte">;
defm XTAME_ThMplde : XTAME_Prefetch_IntrOps<"th.mplde">;
defm XTAME_ThMpldte : XTAME_Prefetch_IntrOps<"th.mpldte">;
defm XTAME_ThMste : XTAME_Store_IntrOps<"th.mste">;
defm XTAME_ThMstte : XTAME_Store_IntrOps<"th.mstte">;
defm XTAME_ThMsste : XTAME_Store_IntrOps<"th.msste">;
defm XTAME_ThMsstte : XTAME_Store_IntrOps<"th.msstte">;

class XTAME_MMA_IntrOp<string mnemonic, string intrinsic>
    : LLVM_IntrOpBase</*Dialect dialect=*/XTAME_Dialect,
                     /*string opName=*/"intr." # mnemonic,
                     /*string enumName=*/"riscv_th_" # intrinsic,
                     /*list<int> overloadedResults=*/[0],
                     /*list<int> overloadedOperands=*/[],
                     /*list<Trait> traits=*/[],
                     /*int numResults=*/1>,
      Arguments<(ins LLVM_Type:$acc, LLVM_Type:$rhs, LLVM_Type:$lhs)>;
class XTAME_WidenMMA_IntrOp<string mnemonic, string intrinsic>
    : LLVM_IntrOpBase</*Dialect dialect=*/XTAME_Dialect,
                     /*string opName=*/"intr." # mnemonic,
                     /*string enumName=*/"riscv_th_" # intrinsic,
                     /*list<int> overloadedResults=*/[0],
                     /*list<int> overloadedOperands=*/[1],
                     /*list<Trait> traits=*/[],
                     /*int numResults=*/1>,
      Arguments<(ins LLVM_Type:$acc, LLVM_Type:$rhs, LLVM_Type:$lhs)>;
def XTAME_ThMmaccWB_IntrOp : XTAME_WidenMMA_IntrOp<"th.mmacc.w.b", "mmacc_w_b">;
def XTAME_ThMmaccuWB_IntrOp : XTAME_WidenMMA_IntrOp<"th.mmaccu.w.b", "mmaccu_w_b">;
def XTAME_ThMmaccusWB_IntrOp : XTAME_WidenMMA_IntrOp<"th.mmaccus.w.b", "mmaccus_w_b">;
def XTAME_ThMmaccsuWB_IntrOp : XTAME_WidenMMA_IntrOp<"th.mmaccsu.w.b", "mmaccsu_w_b">;
def XTAME_ThMfmaccH_IntrOp : XTAME_MMA_IntrOp<"th.mfmacc.h", "mfmacc_h_m">;
def XTAME_ThMfmaccBf16_IntrOp : XTAME_MMA_IntrOp<"th.mfmacc.bf16", "mfmacc_bf16_m">;
def XTAME_ThMfmaccS_IntrOp : XTAME_MMA_IntrOp<"th.mfmacc.s", "mfmacc_s_m">;
def XTAME_ThMfmaccD_IntrOp : XTAME_MMA_IntrOp<"th.mfmacc.d", "mfmacc_d_m">;
def XTAME_ThMfmaccHE4m3_IntrOp : XTAME_WidenMMA_IntrOp<"th.mfmacc.h.e4m3", "mfmacc_h_e4m3_m">;
def XTAME_ThMfmaccHE5m2_IntrOp : XTAME_WidenMMA_IntrOp<"th.mfmacc.h.e5m2", "mfmacc_h_e5m2_m">;
def XTAME_ThMfmaccBf16E4m3_IntrOp : XTAME_WidenMMA_IntrOp<"th.mfmacc.bf16.e4m3", "mfmacc_bf16_e4m3_m">;
def XTAME_ThMfmaccBf16E5m2_IntrOp : XTAME_WidenMMA_IntrOp<"th.mfmacc.bf16.e5m2", "mfmacc_bf16_e5m2_m">;
def XTAME_ThMfmaccSH_IntrOp : XTAME_WidenMMA_IntrOp<"th.mfmacc.s.h", "mfmacc_s_h_m">;
def XTAME_ThMfmaccSBf16_IntrOp : XTAME_WidenMMA_IntrOp<"th.mfmacc.s.bf16", "mfmacc_s_bf16_m">;
def XTAME_ThMfmaccDS_IntrOp : XTAME_WidenMMA_IntrOp<"th.mfmacc.d.s", "mfmacc_d_s_m">;
def XTAME_ThMfmaccSE4m3_IntrOp : XTAME_WidenMMA_IntrOp<"th.mfmacc.s.e4m3", "mfmacc_s_e4m3_m">;
def XTAME_ThMfmaccSE5m2_IntrOp : XTAME_WidenMMA_IntrOp<"th.mfmacc.s.e5m2", "mfmacc_s_e5m2_m">;

#endif // XTAME_DIALECT
