//====-- BOSCAME.td - BOSCAME dialect operation definitions --- tablegen --===//
//
// This SSA-based version models BOSC AME matrix registers as MLIR matrix tile
// values. Physical TR/ACC register numbers are intentionally absent here; they
// are selected later by LLVM instruction selection and register allocation.
//
//===----------------------------------------------------------------------===//

#ifndef BOSCAME_DIALECT
#define BOSCAME_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

//===----------------------------------------------------------------------===//
// BOSCAME Dialect Definition
//===----------------------------------------------------------------------===//

def BOSCAME_Dialect : Dialect {
  let name = "bosc_ame";
  let cppNamespace = "::buddy::boscame";
  let description = [{
    The BOSCAME dialect provides operations for the RISC-V Matrix Extension (RVA23 profile).
    BOSCAME provides matrix multiply-accumulate instructions with tile-based computation,
    supporting various data types and widening modes for AI/ML workloads.
  }];
}

// High-level BOSC AME tiles are ordinary MLIR vectors.  For example,
// vector<4x4xi32> is legal at the BOSCAME dialect boundary.
def BOSCAME_MatrixType : Type<
    CPred<"::mlir::isa<::mlir::VectorType>($_self)">,
    "BOSC AME matrix vector type">;

//===----------------------------------------------------------------------===//
// BOSCAME Operation Base Class
//===----------------------------------------------------------------------===//

class BOSCAME_Op<string mnemonic, list<Trait> traits = []> :
  Op<BOSCAME_Dialect, mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// Matrix Configuration Operations (with register)
//===----------------------------------------------------------------------===//
def MSettypeOp : BOSCAME_Op<"msettype"> {
  let summary = "Set the mtype CSR based on register value";
  let description = [{
    Sets the mtype CSR from a dynamic register value to configure
    the matrix data type and tile register format.
  }];
  let arguments = (ins I64:$rs1);
  let results = (outs I64:$res);
  let assemblyFormat = "$rs1 attr-dict `:` type($res)";
}

def MSettilemOp : BOSCAME_Op<"msettilem"> {
  let summary = "Set tile M dimension with register value";
  let description = [{
    Sets the tile M dimension (number of rows) from a register value.
    mtilem = min(rs1, TMMAX)
  }];
  let arguments = (ins I64:$rs1);
  let results = (outs I64:$res);
  let assemblyFormat = "$rs1 attr-dict `:` type($res)";
}

def MSettilenOp : BOSCAME_Op<"msettilen"> {
  let summary = "Set tile N dimension with register value";
  let description = [{
    Sets the tile N dimension (number of columns) from a register value.
    mtilen = min(rs1, TNMAX)
  }];
  let arguments = (ins I64:$rs1);
  let results = (outs I64:$res);
  let assemblyFormat = "$rs1 attr-dict `:` type($res)";
}

def MSettilekOp : BOSCAME_Op<"msettilek"> {
  let summary = "Set tile K dimension with register value";
  let description = [{
    Sets the tile K dimension (inner dimension for matrix multiply) from a register value.
    mtilek = min(rs1, TKMAX)
  }];
  let arguments = (ins I64:$rs1);
  let results = (outs I64:$res);
  let assemblyFormat = "$rs1 attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// Matrix Configuration Operations (with immediate)
//===----------------------------------------------------------------------===//
def MSettypeiOp : BOSCAME_Op<"msettypei"> {
  let summary = "Set the bits of mtype CSR based on immediate";
  let description = [{
    Sets the upper bits of the mtype CSR to configure the matrix data type and tile register format.
    The immediate value encodes the extended data type configurations.
  }];
  let arguments = (ins I64Attr:$timm);
  let results = (outs I64:$res);
  let assemblyFormat = "$timm attr-dict `:` type($res)";
}

def MSettypehiOp : BOSCAME_Op<"msettypehi"> {
  let summary = "Set the high bits of mtype CSR based on immediate";
  let description = [{
    Sets the upper bits of the mtype CSR to configure the matrix data type and tile register format.
    The immediate value encodes the extended data type configurations.
  }];
  let arguments = (ins I64Attr:$timm);
  let results = (outs I64:$res);
  let assemblyFormat = "$timm attr-dict `:` type($res)";
}

def MSettilemiOp : BOSCAME_Op<"msettilemi"> {
  let summary = "Set tile M dimension with immediate";
  let description = [{
    Sets the tile M dimension (number of rows) from an immediate value.
    mtilem = min(tilem, TMMAX)
  }];
  let arguments = (ins I64Attr:$tilem);
  let results = (outs I64:$res);
  let assemblyFormat = "$tilem attr-dict `:` type($res)";
}

def MSettileniOp : BOSCAME_Op<"msettileni"> {
  let summary = "Set tile N dimension with immediate";
  let description = [{
    Sets the tile N dimension (number of columns) from an immediate value.
    mtilen = min(tilen, TNMAX)
  }];
  let arguments = (ins I64Attr:$tilen);
  let results = (outs I64:$res);
  let assemblyFormat = "$tilen attr-dict `:` type($res)";
}

def MSettilekiOp : BOSCAME_Op<"msettileki"> {
  let summary = "Set tile K dimension with immediate";
  let description = [{
    Sets the tile K dimension (inner dimension for matrix multiply) from an immediate value.
    mtilek = min(tilek, TKMAX)
  }];
  let arguments = (ins I64Attr:$tilek);
  let results = (outs I64:$res);
  let assemblyFormat = "$tilek attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// Data Move / Broadcast / Transpose Operations Base Classes (MLIR ODS)
//===----------------------------------------------------------------------===//
// Base class for Pure Matrix-to-Matrix Operations (e.g., T-T, A-A, Transpose, Broadcast)
// Arguments: md (dest index), ms1 (source index)
class BOSCAME_PureMove_Op<string mnemonic, string summary_desc>
    : BOSCAME_Op<mnemonic, [AllTypesMatch<["src", "res"]>]> {
  let summary = summary_desc;
  let arguments = (ins BOSCAME_MatrixType:$src);
  let results = (outs BOSCAME_MatrixType:$res);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($res)";
}

// Base class for Matrix Move with GPR Index
// Arguments: md (dest index), ms1 (source index), rs2 (GPR row index)
class BOSCAME_IndexedMove_Op<string mnemonic, string summary_desc>
    : BOSCAME_Op<mnemonic, [AllTypesMatch<["src", "res"]>]> {
  let summary = summary_desc;
  let arguments = (ins BOSCAME_MatrixType:$src, I64:$rs2);
  let results = (outs BOSCAME_MatrixType:$res);
  let assemblyFormat =
      "$src `,` $rs2 attr-dict `:` type($src) `,` type($rs2) "
      "`->` type($res)";
}

// Base class for Matrix Move with Immediate Index
// Arguments: md (dest index), ms1 (source index), imm (Immediate row index)
class BOSCAME_ImmIndexedMove_Op<string mnemonic, string summary_desc>
    : BOSCAME_Op<mnemonic, [AllTypesMatch<["src", "res"]>]> {
  let summary = summary_desc;
  let arguments = (ins BOSCAME_MatrixType:$src, I64Attr:$imm);
  let results = (outs BOSCAME_MatrixType:$res);
  let assemblyFormat =
      "$src `,` $imm attr-dict `:` type($src) `->` type($res)";
}

// Base class for Scalar-to-Matrix Move
// Arguments: md_in (old matrix value), rs1 (scalar value),
// rs2 (element/row index). The result is tied to md_in.
class BOSCAME_ScalarToMatrix_Op<string mnemonic, string summary_desc,
                                Type scalarType>
    : BOSCAME_Op<mnemonic, [AllTypesMatch<["md_in", "res"]>]> {
  let summary = summary_desc;

  // md_in 是被更新前的 tile/acc；结果 res 是更新后的新 SSA 值。
  let arguments = (ins BOSCAME_MatrixType:$md_in,
                       scalarType:$rs1,
                       I64:$rs2);
  let results = (outs BOSCAME_MatrixType:$res);

  let assemblyFormat =
      "$md_in `,` $rs1 `,` $rs2 attr-dict `:` "
      "type($md_in) `,` type($rs1) `,` type($rs2) "
      "`->` type($res)";
}

// Base class for Matrix-to-Scalar Move
// Arguments: ms1 (source index), rs2 (GPR row index) -> Returns I64
class BOSCAME_MatrixToScalar_Op<string mnemonic, string summary_desc,
                                Type scalarType>
    : BOSCAME_Op<mnemonic> {
  let summary = summary_desc;
  let arguments = (ins BOSCAME_MatrixType:$src, I64:$rs2);
  let results = (outs scalarType:$res);
  let assemblyFormat =
      "$src `,` $rs2 attr-dict `:` type($src) `,` type($rs2) "
      "`->` type($res)";
}

//===----------------------------------------------------------------------===//
// Multiclasses for Bitwidth Auto-Generation
//===----------------------------------------------------------------------===//
multiclass BOSCAME_PureMove_Multiclass<string prefix, string suffix, string op_name> {
  def "8Op"  : BOSCAME_PureMove_Op<prefix # "8" # suffix,  "8-bit "  # op_name>;
  def "16Op" : BOSCAME_PureMove_Op<prefix # "16" # suffix, "16-bit " # op_name>;
  def "32Op" : BOSCAME_PureMove_Op<prefix # "32" # suffix, "32-bit " # op_name>;
  def "64Op" : BOSCAME_PureMove_Op<prefix # "64" # suffix, "64-bit " # op_name>;
}

multiclass BOSCAME_IndexedMove_Multiclass<string prefix, string suffix, string op_name> {
  def "8Op"  : BOSCAME_IndexedMove_Op<prefix # "8" # suffix,  "8-bit "  # op_name>;
  def "16Op" : BOSCAME_IndexedMove_Op<prefix # "16" # suffix, "16-bit " # op_name>;
  def "32Op" : BOSCAME_IndexedMove_Op<prefix # "32" # suffix, "32-bit " # op_name>;
  def "64Op" : BOSCAME_IndexedMove_Op<prefix # "64" # suffix, "64-bit " # op_name>;
}

multiclass BOSCAME_ImmIndexedMove_Multiclass<string prefix, string suffix, string op_name> {
  def "8Op"  : BOSCAME_ImmIndexedMove_Op<prefix # "8" # suffix,  "8-bit "  # op_name>;
  def "16Op" : BOSCAME_ImmIndexedMove_Op<prefix # "16" # suffix, "16-bit " # op_name>;
  def "32Op" : BOSCAME_ImmIndexedMove_Op<prefix # "32" # suffix, "32-bit " # op_name>;
  def "64Op" : BOSCAME_ImmIndexedMove_Op<prefix # "64" # suffix, "64-bit " # op_name>;
}

multiclass BOSCAME_IntScalarToMatrix_Multiclass<string prefix, string suffix, string op_name> {
  def "8Op"  : BOSCAME_ScalarToMatrix_Op<prefix # "8" # suffix,  "8-bit "  # op_name, I64>;
  def "16Op" : BOSCAME_ScalarToMatrix_Op<prefix # "16" # suffix, "16-bit " # op_name, I64>;
  def "32Op" : BOSCAME_ScalarToMatrix_Op<prefix # "32" # suffix, "32-bit " # op_name, I64>;
  def "64Op" : BOSCAME_ScalarToMatrix_Op<prefix # "64" # suffix, "64-bit " # op_name, I64>;
}

multiclass BOSCAME_IntMatrixToScalar_Multiclass<string prefix, string suffix, string op_name> {
  def "8Op"  : BOSCAME_MatrixToScalar_Op<prefix # "8" # suffix,  "8-bit "  # op_name, I64>;
  def "16Op" : BOSCAME_MatrixToScalar_Op<prefix # "16" # suffix, "16-bit " # op_name, I64>;
  def "32Op" : BOSCAME_MatrixToScalar_Op<prefix # "32" # suffix, "32-bit " # op_name, I64>;
  def "64Op" : BOSCAME_MatrixToScalar_Op<prefix # "64" # suffix, "64-bit " # op_name, I64>;
}

multiclass BOSCAME_FloatScalarToMatrix_Multiclass<string prefix, string suffix, string op_name> {
  def "8Op"  : BOSCAME_ScalarToMatrix_Op<prefix # "8"  # suffix, "8-bit "  # op_name, F32>;
  def "16Op" : BOSCAME_ScalarToMatrix_Op<prefix # "16" # suffix, "16-bit " # op_name, F16>;
  def "32Op" : BOSCAME_ScalarToMatrix_Op<prefix # "32" # suffix, "32-bit " # op_name, F32>;
  def "64Op" : BOSCAME_ScalarToMatrix_Op<prefix # "64" # suffix, "64-bit " # op_name, F64>;
}
multiclass BOSCAME_FloatMatrixToScalar_Multiclass<string prefix, string suffix, string op_name> {
  def "8Op"  : BOSCAME_MatrixToScalar_Op<prefix # "8"  # suffix, "8-bit "  # op_name, F32>;
  def "16Op" : BOSCAME_MatrixToScalar_Op<prefix # "16" # suffix, "16-bit " # op_name, F16>;
  def "32Op" : BOSCAME_MatrixToScalar_Op<prefix # "32" # suffix, "32-bit " # op_name, F32>;
  def "64Op" : BOSCAME_MatrixToScalar_Op<prefix # "64" # suffix, "64-bit " # op_name, F64>;
}

//===----------------------------------------------------------------------===//
// Data Move Operations Instantiations
//===----------------------------------------------------------------------===//
// 1. Matrix-Matrix Move (Pure Matrix)
defm MmveTt : BOSCAME_PureMove_Multiclass<"mmve", ".t.t", "Tile-to-Tile Move">;
defm MmveAa : BOSCAME_PureMove_Multiclass<"mmve", ".a.a", "Acc-to-Acc Move">;
// 2. Matrix-Matrix Move (with GPR Index)
defm MmveAt : BOSCAME_IndexedMove_Multiclass<"mmve", ".a.t", "Tile-to-Acc Move">;
defm MmveTa : BOSCAME_IndexedMove_Multiclass<"mmve", ".t.a", "Acc-to-Tile Move">;
// 3. Matrix-Matrix Move (with Immediate Index)
defm MmvieAt : BOSCAME_ImmIndexedMove_Multiclass<"mmvie", ".a.t", "Tile-to-Acc Move">;
defm MmvieTa : BOSCAME_ImmIndexedMove_Multiclass<"mmvie", ".t.a", "Acc-to-Tile Move">;
// 4. Matrix to Scalar Move (Returns GPR)
defm MmveXt  : BOSCAME_IntMatrixToScalar_Multiclass<"mmve", ".x.t", "Tile-to-GPR Move">;
defm MmveXa  : BOSCAME_IntMatrixToScalar_Multiclass<"mmve", ".x.a", "Acc-to-GPR Move">;
defm MfmveFt : BOSCAME_FloatMatrixToScalar_Multiclass<"mfmve", ".f.t", "Tile-to-FPR Move">;
defm MfmveFa : BOSCAME_FloatMatrixToScalar_Multiclass<"mfmve", ".f.a", "Acc-to-FPR Move">;
// 5. Scalar GPR to Matrix Move
defm MmveTx  : BOSCAME_IntScalarToMatrix_Multiclass<"mmve", ".t.x", "GPR-to-Tile Move">;
defm MmveAx  : BOSCAME_IntScalarToMatrix_Multiclass<"mmve", ".a.x", "GPR-to-Acc Move">;
defm MfmveTf : BOSCAME_FloatScalarToMatrix_Multiclass<"mfmve", ".t.f", "FPR-to-Tile Move">;
defm MfmveAf : BOSCAME_FloatScalarToMatrix_Multiclass<"mfmve", ".a.f", "FPR-to-Acc Move">;
//===----------------------------------------------------------------------===//
// Broadcast Operations Instantiations
//===----------------------------------------------------------------------===//
// Single Row/Col/Elem Broadcasts (Pure Move semantics)
def MbcarMOp : BOSCAME_PureMove_Op<"mbcar.m", "Broadcast Tile Row">;
def MbcbrMOp : BOSCAME_PureMove_Op<"mbcbr.m", "Broadcast Tile Row">;
def MbccrMOp : BOSCAME_PureMove_Op<"mbccr.m", "Broadcast Acc Row">;

defm Mbcace : BOSCAME_PureMove_Multiclass<"mbcace", ".m", "Broadcast Tile Col A">;
defm Mbcbce : BOSCAME_PureMove_Multiclass<"mbcbce", ".m", "Broadcast Tile Col B">;
defm Mbccce : BOSCAME_PureMove_Multiclass<"mbccce", ".m", "Broadcast Acc Col">;

defm Mbcaee : BOSCAME_PureMove_Multiclass<"mbcaee", ".m", "Broadcast Tile Elem A">;
defm Mbcbee : BOSCAME_PureMove_Multiclass<"mbcbee", ".m", "Broadcast Tile Elem B">;
defm Mbccee : BOSCAME_PureMove_Multiclass<"mbccee", ".m", "Broadcast Acc Elem">;
//===----------------------------------------------------------------------===//
// Transpose Operations Instantiations
//===----------------------------------------------------------------------===//
defm Mtae : BOSCAME_PureMove_Multiclass<"mtae", ".m", "Transpose Tile A">;
defm Mtbe : BOSCAME_PureMove_Multiclass<"mtbe", ".m", "Transpose Tile B">;
defm Mtce : BOSCAME_PureMove_Multiclass<"mtce", ".m", "Transpose Acc C">;

//===----------------------------------------------------------------------===//
// Element-Wise Operations Base Classes (MLIR ODS)
//===----------------------------------------------------------------------===//
// Base class for 3-operand element-wise operations
class BOSCAME_EleWise3_Op<string mnemonic, string summary_desc, string symbol>
    : BOSCAME_Op<mnemonic, [AllTypesMatch<["src1", "src2", "res"]>]> {
  let summary = summary_desc # " matrix operation";
  let arguments = (ins BOSCAME_MatrixType:$src1, BOSCAME_MatrixType:$src2);
  let results = (outs BOSCAME_MatrixType:$res);
  let assemblyFormat =
      "$src1 `,` $src2 attr-dict `:` type($src1) `,` type($src2) "
      "`->` type($res)";
}

class BOSCAME_WidenEleWise3_Op<string mnemonic, string summary_desc,
                               string symbol>
    : BOSCAME_Op<mnemonic> {
  let summary = summary_desc # " widening matrix operation";
  let arguments = (ins BOSCAME_MatrixType:$src1, BOSCAME_MatrixType:$src2);
  let results = (outs BOSCAME_MatrixType:$res);
  let assemblyFormat =
      "$src1 `,` $src2 attr-dict `:` type($src1) `,` type($src2) "
      "`->` type($res)";
}

class BOSCAME_EleWise2_Op<string mnemonic, string summary_desc,
                          string func_name>
    : BOSCAME_Op<mnemonic, [AllTypesMatch<["src", "res"]>]> {
  let summary = summary_desc # " unary matrix operation";
  let arguments = (ins BOSCAME_MatrixType:$src);
  let results = (outs BOSCAME_MatrixType:$res);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($res)";
}

//===----------------------------------------------------------------------===//
// Multiclasses for Bitwidth Auto-Generation
//===----------------------------------------------------------------------===//
// Multiclass for standard integer widths (No Widen)
multiclass BOSCAME_EleWise_Int_NoWiden<string prefix, string op_name, string symbol> {
  def MmOp   : BOSCAME_EleWise3_Op<prefix # ".mm",    op_name # " dynamic-width", symbol>;
  //def HbMmOp : BOSCAME_EleWise3_Op<prefix # ".hb.mm", op_name # " 4-bit", symbol>;
  def BMmOp  : BOSCAME_EleWise3_Op<prefix # ".b.mm",  op_name # " 8-bit", symbol>;
  def HMmOp  : BOSCAME_EleWise3_Op<prefix # ".h.mm",  op_name # " 16-bit", symbol>;
  def WMmOp  : BOSCAME_EleWise3_Op<prefix # ".w.mm",  op_name # " 32-bit", symbol>;
  def DwMmOp : BOSCAME_EleWise3_Op<prefix # ".dw.mm", op_name # " 64-bit", symbol>;
}
// Multiclass for floating point widths (No Widen)
multiclass BOSCAME_EleWise_Float_NoWiden<string prefix, string op_name, string symbol> {
  def MmOp   : BOSCAME_EleWise3_Op<prefix # ".mm",    op_name # " dynamic-width", symbol>;
  //def CfMmOp : BOSCAME_EleWise3_Op<prefix # ".cf.mm", op_name # " 8-bit (cf)", symbol>;
  def HfMmOp : BOSCAME_EleWise3_Op<prefix # ".hf.mm", op_name # " 16-bit (half)", symbol>;
  def FMmOp  : BOSCAME_EleWise3_Op<prefix # ".f.mm",  op_name # " 32-bit (single)", symbol>;
  def DMmOp  : BOSCAME_EleWise3_Op<prefix # ".d.mm",  op_name # " 64-bit (double)", symbol>;
}
// Multiclass for integer widths (Double Widen - stops at 32-bit input)
multiclass BOSCAME_EleWise_Int_DoubleWiden<string prefix, string op_name, string symbol> {
  def MmOp   : BOSCAME_WidenEleWise3_Op<prefix # ".mm",    op_name # " dynamic-width", symbol>;
  //def HbMmOp : BOSCAME_WidenEleWise3_Op<prefix # ".hb.mm", op_name # " 4-bit", symbol>;
  def BMmOp  : BOSCAME_WidenEleWise3_Op<prefix # ".b.mm",  op_name # " 8-bit", symbol>;
  def HMmOp  : BOSCAME_WidenEleWise3_Op<prefix # ".h.mm",  op_name # " 16-bit", symbol>;
  def WMmOp  : BOSCAME_WidenEleWise3_Op<prefix # ".w.mm",  op_name # " 32-bit", symbol>;
}
// Multiclass for floating point widths (Double Widen - stops at 32-bit input)
multiclass BOSCAME_EleWise_Float_DoubleWiden<string prefix, string op_name, string symbol> {
  def MmOp   : BOSCAME_WidenEleWise3_Op<prefix # ".mm",    op_name # " dynamic-width", symbol>;
  //def CfMmOp : BOSCAME_WidenEleWise3_Op<prefix # ".cf.mm", op_name # " 8-bit (cf)", symbol>;
  def HfMmOp : BOSCAME_WidenEleWise3_Op<prefix # ".hf.mm", op_name # " 16-bit (half)", symbol>;
  def FMmOp  : BOSCAME_WidenEleWise3_Op<prefix # ".f.mm",  op_name # " 32-bit (single)", symbol>;
}
// Multiclass for Unary Floating Point (SQRT)
multiclass BOSCAME_EleWise_Unary_Float<string prefix, string op_name, string func> {
  def MOp   : BOSCAME_EleWise2_Op<prefix # ".m",    op_name # " dynamic-width", func>;
  //def CfMOp : BOSCAME_EleWise2_Op<prefix # ".cf.m", op_name # " 8-bit (cf)", func>;
  def HfMOp : BOSCAME_EleWise2_Op<prefix # ".hf.m", op_name # " 16-bit (half)", func>;
  def FMOp  : BOSCAME_EleWise2_Op<prefix # ".f.m",  op_name # " 32-bit (single)", func>;
  def DMOp  : BOSCAME_EleWise2_Op<prefix # ".d.m",  op_name # " 64-bit (double)", func>;
}

//===----------------------------------------------------------------------===//
// Element-Wise Operations Instantiations
//===----------------------------------------------------------------------===//
// 1. ADD / SUB - No Widen
defm Maddu  : BOSCAME_EleWise_Int_NoWiden<"maddu", "Unsigned add", "+">;
defm Msaddu : BOSCAME_EleWise_Int_NoWiden<"msaddu", "Saturated unsigned add", "+">;
defm Madd   : BOSCAME_EleWise_Int_NoWiden<"madd", "Signed add", "+">;
defm Msadd  : BOSCAME_EleWise_Int_NoWiden<"msadd", "Saturated signed add", "+">;
defm Mfadd  : BOSCAME_EleWise_Float_NoWiden<"mfadd", "Floating-point add", "+">;

defm Msubu  : BOSCAME_EleWise_Int_NoWiden<"msubu", "Unsigned sub", "-">;
defm Mssubu : BOSCAME_EleWise_Int_NoWiden<"mssubu", "Saturated unsigned sub", "-">;
defm Msub   : BOSCAME_EleWise_Int_NoWiden<"msub", "Signed sub", "-">;
defm Mssub  : BOSCAME_EleWise_Int_NoWiden<"mssub", "Saturated signed sub", "-">;
defm Mfsub  : BOSCAME_EleWise_Float_NoWiden<"mfsub", "Floating-point sub", "-">;
// 2. ADD / SUB - Double Widen
defm Mwaddu : BOSCAME_EleWise_Int_DoubleWiden<"mwaddu", "Double-widening unsigned add", "+">;
defm Mwadd  : BOSCAME_EleWise_Int_DoubleWiden<"mwadd", "Double-widening signed add", "+">;
defm Mfwadd : BOSCAME_EleWise_Float_DoubleWiden<"mfwadd", "Double-widening floating-point add", "+">;

defm Mwsubu : BOSCAME_EleWise_Int_DoubleWiden<"mwsubu", "Double-widening unsigned sub", "-">;
defm Mwsub  : BOSCAME_EleWise_Int_DoubleWiden<"mwsub", "Double-widening signed sub", "-">;
defm Mfwsub : BOSCAME_EleWise_Float_DoubleWiden<"mfwsub", "Double-widening floating-point sub", "-">;
// 3. MIN / MAX - No Widen
defm Mminu : BOSCAME_EleWise_Int_NoWiden<"mminu", "Unsigned min", "min">;
defm Mmaxu : BOSCAME_EleWise_Int_NoWiden<"mmaxu", "Unsigned max", "max">;
defm Mmin  : BOSCAME_EleWise_Int_NoWiden<"mmin", "Signed min", "min">;
defm Mmax  : BOSCAME_EleWise_Int_NoWiden<"mmax", "Signed max", "max">;
defm Mfmin : BOSCAME_EleWise_Float_NoWiden<"mfmin", "Floating-point min", "min">;
defm Mfmax : BOSCAME_EleWise_Float_NoWiden<"mfmax", "Floating-point max", "max">;
// 4. MUL / MULH / DIV - No Widen
defm Msmulu  : BOSCAME_EleWise_Int_NoWiden<"msmulu", "Saturated unsigned mul", "*">;
defm Mmul    : BOSCAME_EleWise_Int_NoWiden<"mmul", "Signed mul", "*">;
defm Msmul   : BOSCAME_EleWise_Int_NoWiden<"msmul", "Saturated signed mul", "*">;
defm Mfmul   : BOSCAME_EleWise_Float_NoWiden<"mfmul", "Floating-point mul", "*">;
defm Mmulhu  : BOSCAME_EleWise_Int_NoWiden<"mmulhu", "Unsigned mul high", "mulh">;
defm Mmulh   : BOSCAME_EleWise_Int_NoWiden<"mmulh", "Signed mul high", "mulh">;
defm Mmulhsu : BOSCAME_EleWise_Int_NoWiden<"mmulhsu", "Signed-unsigned mul high", "mulhsu">;
defm Msmulsu : BOSCAME_EleWise_Int_NoWiden<"msmulsu", "Saturated signed-unsigned mul high", "mulhsu">;
defm Mfdiv   : BOSCAME_EleWise_Float_NoWiden<"mfdiv", "Floating-point div", "/">;
// 5. MUL - Double Widen
defm Mwmulu  : BOSCAME_EleWise_Int_DoubleWiden<"mwmulu", "Double-widening unsigned mul", "*">;
defm Mwmul   : BOSCAME_EleWise_Int_DoubleWiden<"mwmul", "Double-widening signed mul", "*">;
defm Mwmulsu : BOSCAME_EleWise_Int_DoubleWiden<"mwmulsu", "Double-widening signed-unsigned mul", "*">;
defm Mfwmul  : BOSCAME_EleWise_Float_DoubleWiden<"mfwmul", "Double-widening floating-point mul", "*">;
// 6. LOGIC & SHIFT - No Widen (Individual & Multiclass)
def MandMmOp : BOSCAME_EleWise3_Op<"mand.mm", "Bitwise AND", "&">;
def MorMmOp  : BOSCAME_EleWise3_Op<"mor.mm", "Bitwise OR", "|">;
def MxorMmOp : BOSCAME_EleWise3_Op<"mxor.mm", "Bitwise XOR", "^">;

defm Msll : BOSCAME_EleWise_Int_NoWiden<"msll", "Logical left shift", "<<">;
defm Msrl : BOSCAME_EleWise_Int_NoWiden<"msrl", "Logical right shift", ">>">;
defm Msra : BOSCAME_EleWise_Int_NoWiden<"msra", "Arithmetic right shift", ">>a">;
// 7. SQRT - No Widen
defm Mfsqrt : BOSCAME_EleWise_Unary_Float<"mfsqrt", "Floating-point square root", "sqrt">;

//===----------------------------------------------------------------------===//
// Matrix Load Instructions
// Load matrix tiles from memory to tile registers
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// BOSC AME Load/Store Operation Base Classes (MLIR ODS)
//===----------------------------------------------------------------------===//
// Base class for all Matrix Load Operations
class BOSCAME_Load_Op<string mnemonic, string summary_text, string shape_desc>
    : BOSCAME_Op<mnemonic> {
  let summary = summary_text;
  let arguments = (ins AnyMemRef:$base, I64:$stride);
  let results = (outs BOSCAME_MatrixType:$res);
  let assemblyFormat =
      "$base `,` $stride attr-dict `:` type($base) `->` type($res)";
}

class BOSCAME_Store_Op<string mnemonic, string summary_text, string shape_desc>
    : BOSCAME_Op<mnemonic> {
  let summary = summary_text;
  let arguments = (ins BOSCAME_MatrixType:$src, AnyMemRef:$base, I64:$stride);
  let assemblyFormat =
      "$src `,` $base `,` $stride attr-dict `:` type($src) `,` type($base)";
}

//===----------------------------------------------------------------------===//
// Multiclasses for Bitwidth Auto-Generation
//===----------------------------------------------------------------------===//
multiclass BOSCAME_Load_Multiclass<string prefix, string name_base, string shape_desc> {
  def "8mOp"  : BOSCAME_Load_Op<prefix # "8.m",  "Load 8-bit " # name_base, shape_desc>;
  def "16mOp" : BOSCAME_Load_Op<prefix # "16.m", "Load 16-bit " # name_base, shape_desc>;
  def "32mOp" : BOSCAME_Load_Op<prefix # "32.m", "Load 32-bit " # name_base, shape_desc>;
  def "64mOp" : BOSCAME_Load_Op<prefix # "64.m", "Load 64-bit " # name_base, shape_desc>;
}

multiclass BOSCAME_Store_Multiclass<string prefix, string name_base, string shape_desc> {
  def "8mOp"  : BOSCAME_Store_Op<prefix # "8.m",  "Store 8-bit " # name_base, shape_desc>;
  def "16mOp" : BOSCAME_Store_Op<prefix # "16.m", "Store 16-bit " # name_base, shape_desc>;
  def "32mOp" : BOSCAME_Store_Op<prefix # "32.m", "Store 32-bit " # name_base, shape_desc>;
  def "64mOp" : BOSCAME_Store_Op<prefix # "64.m", "Store 64-bit " # name_base, shape_desc>;
}

//===----------------------------------------------------------------------===//
// Matrix Load Instructions Instantiations
//===----------------------------------------------------------------------===//
// Matrix A (Shape: mtilem x mtilek)
defm Mlae  : BOSCAME_Load_Multiclass<"mlae",  "left matrix tile A", "mtilem x mtilek">;
defm Mlate : BOSCAME_Load_Multiclass<"mlate", "left matrix tile A (transposed)", "mtilek x mtilem">;
defm Mltre : BOSCAME_Load_Multiclass<"mltre", "whole tile matrix", "mtilem x mtilek">;

// Matrix B (Shape: mtilek x mtilen)
defm Mlbe  : BOSCAME_Load_Multiclass<"mlbe",  "right matrix tile B", "mtilek x mtilen">;
defm Mlbte : BOSCAME_Load_Multiclass<"mlbte", "right matrix tile B (transposed)", "mtilen x mtilek">;

// Matrix C / Accumulator (Shape: mtilem x mtilen)
defm Mlce   : BOSCAME_Load_Multiclass<"mlce",   "output matrix tile C", "mtilem x mtilen">;
defm Mlcte  : BOSCAME_Load_Multiclass<"mlcte",  "output matrix tile C (transposed)", "mtilen x mtilem">;
defm Mlacce : BOSCAME_Load_Multiclass<"mlacce", "whole accumulation matrix", "mtilem x mtilen">;

//===----------------------------------------------------------------------===//
// Matrix Store Instructions Instantiations
//===----------------------------------------------------------------------===//
// Matrix A (Shape: mtilem x mtilek)
defm Msae  : BOSCAME_Store_Multiclass<"msae",  "left matrix tile A", "mtilem x mtilek">;
defm Msate : BOSCAME_Store_Multiclass<"msate", "left matrix tile A (transposed)", "mtilek x mtilem">;
defm Mstre : BOSCAME_Store_Multiclass<"mstre", "whole tile matrix", "mtilem x mtilek">;

// Matrix B (Shape: mtilek x mtilen)
defm Msbe  : BOSCAME_Store_Multiclass<"msbe",  "right matrix tile B", "mtilek x mtilen">;
defm Msbte : BOSCAME_Store_Multiclass<"msbte", "right matrix tile B (transposed)", "mtilen x mtilek">;

// Matrix C / Accumulator (Shape: mtilem x mtilen)
defm Msce   : BOSCAME_Store_Multiclass<"msce",   "output matrix tile C", "mtilem x mtilen">;
defm Mscte  : BOSCAME_Store_Multiclass<"mscte",  "output matrix tile C (transposed)", "mtilen x mtilem">;
defm Msacce : BOSCAME_Store_Multiclass<"msacce", "whole accumulation matrix", "mtilem x mtilen">;

//===----------------------------------------------------------------------===//
// Matrix Multiply-Accumulate (MMA) Operations Base Class
//===----------------------------------------------------------------------===//
class BOSCAME_NoWidenMMA_Op<string mnemonic, string summary_desc>
    : BOSCAME_Op<mnemonic,
                 [AllTypesMatch<["acc", "lhs", "rhs", "res"]>]> {
  let summary = summary_desc # " matrix multiply-accumulate";
  let arguments = (ins
    BOSCAME_MatrixType:$acc,
    BOSCAME_MatrixType:$lhs,
    BOSCAME_MatrixType:$rhs
  );
  let results = (outs BOSCAME_MatrixType:$res);
  let assemblyFormat =
      "$acc `,` $lhs `,` $rhs attr-dict `:` "
      "type($acc) `,` type($lhs) `,` type($rhs) `->` type($res)";
}

class BOSCAME_WidenMMA_Op<string mnemonic, string summary_desc>
    : BOSCAME_Op<mnemonic,
                 [AllTypesMatch<["acc", "res"]>,
                  AllTypesMatch<["lhs", "rhs"]>]> {
  let summary = summary_desc # " matrix multiply-accumulate";
  let arguments = (ins
    BOSCAME_MatrixType:$acc,
    BOSCAME_MatrixType:$lhs,
    BOSCAME_MatrixType:$rhs
  );
  let results = (outs BOSCAME_MatrixType:$res);
  let assemblyFormat =
      "$acc `,` $lhs `,` $rhs attr-dict `:` "
      "type($acc) `,` type($lhs) `,` type($rhs) `->` type($res)";
}

//===----------------------------------------------------------------------===//
// MMA Operations Instantiations
//===----------------------------------------------------------------------===//
//===--------------------------------------------------------------------===//
// 1. No-widen matrix multiply-accumulate
//===--------------------------------------------------------------------===//
// int
def MmaMmOp    : BOSCAME_NoWidenMMA_Op<"mma.mm",    "Signed dynamic-width">;
def MmaHmmOp   : BOSCAME_NoWidenMMA_Op<"mma.h.mm",  "Signed 16-bit">;
def MmaWmmOp   : BOSCAME_NoWidenMMA_Op<"mma.w.mm",  "Signed 32-bit">;
def MmaDwmmOp  : BOSCAME_NoWidenMMA_Op<"mma.dw.mm", "Signed 64-bit">;

// int, saturated
def MsmaMmOp   : BOSCAME_NoWidenMMA_Op<"msma.mm",    "Saturated signed dynamic-width">;
def MsmaHmmOp  : BOSCAME_NoWidenMMA_Op<"msma.h.mm",  "Saturated signed 16-bit">;
def MsmaWmmOp  : BOSCAME_NoWidenMMA_Op<"msma.w.mm",  "Saturated signed 32-bit">;
def MsmaDwmmOp : BOSCAME_NoWidenMMA_Op<"msma.dw.mm", "Saturated signed 64-bit">;

// uint
def MmauMmOp   : BOSCAME_NoWidenMMA_Op<"mmau.mm",    "Unsigned dynamic-width">;
def MmauHmmOp  : BOSCAME_NoWidenMMA_Op<"mmau.h.mm",  "Unsigned 16-bit">;
def MmauWmmOp  : BOSCAME_NoWidenMMA_Op<"mmau.w.mm",  "Unsigned 32-bit">;
def MmauDwmmOp : BOSCAME_NoWidenMMA_Op<"mmau.dw.mm", "Unsigned 64-bit">;

// uint, saturated
def MsmauMmOp   : BOSCAME_NoWidenMMA_Op<"msmau.mm",    "Saturated unsigned dynamic-width">;
def MsmauHmmOp  : BOSCAME_NoWidenMMA_Op<"msmau.h.mm",  "Saturated unsigned 16-bit">;
def MsmauWmmOp  : BOSCAME_NoWidenMMA_Op<"msmau.w.mm",  "Saturated unsigned 32-bit">;
def MsmauDwmmOp : BOSCAME_NoWidenMMA_Op<"msmau.dw.mm", "Saturated unsigned 64-bit">;

// float point
def MfmaMmOp   : BOSCAME_NoWidenMMA_Op<"mfma.mm",    "Floating-point dynamic-width">;
def MfmaHfmmOp : BOSCAME_NoWidenMMA_Op<"mfma.hf.mm", "Floating-point 16-bit (half)">;
def MfmaFmmOp  : BOSCAME_NoWidenMMA_Op<"mfma.f.mm",  "Floating-point 32-bit (single)">;
def MfmaDmmOp  : BOSCAME_NoWidenMMA_Op<"mfma.d.mm",  "Floating-point 64-bit (double)">;
//===--------------------------------------------------------------------===//
// 2. Double-widen matrix multiply-accumulate
//===--------------------------------------------------------------------===//
// uint
def MwmauMmOp  : BOSCAME_WidenMMA_Op<"mwmau.mm",   "Double-widening unsigned dynamic-width">;
def MwmauHmmOp : BOSCAME_WidenMMA_Op<"mwmau.h.mm", "Double-widening unsigned 16-bit">;
def MwmauWmmOp : BOSCAME_WidenMMA_Op<"mwmau.w.mm", "Double-widening unsigned 32-bit">;

// uint, saturated
def MswmauMmOp  : BOSCAME_WidenMMA_Op<"mswmau.mm",   "Saturated double-widening unsigned dynamic-width">;
def MswmauHmmOp : BOSCAME_WidenMMA_Op<"mswmau.h.mm", "Saturated double-widening unsigned 16-bit">;
def MswmauWmmOp : BOSCAME_WidenMMA_Op<"mswmau.w.mm", "Saturated double-widening unsigned 32-bit">;

// int
def MwmaMmOp  : BOSCAME_WidenMMA_Op<"mwma.mm",   "Double-widening signed dynamic-width">;
def MwmaHmmOp : BOSCAME_WidenMMA_Op<"mwma.h.mm", "Double-widening signed 16-bit">;
def MwmaWmmOp : BOSCAME_WidenMMA_Op<"mwma.w.mm", "Double-widening signed 32-bit">;

// int, saturated
def MswmaMmOp  : BOSCAME_WidenMMA_Op<"mswma.mm",   "Saturated double-widening signed dynamic-width">;
def MswmaHmmOp : BOSCAME_WidenMMA_Op<"mswma.h.mm", "Saturated double-widening signed 16-bit">;
def MswmaWmmOp : BOSCAME_WidenMMA_Op<"mswma.w.mm", "Saturated double-widening signed 32-bit">;

// float point
def MfwmaMmOp   : BOSCAME_WidenMMA_Op<"mfwma.mm",    "Double-widening floating-point dynamic-width">;
//def MfwmaCfmmOp : BOSCAME_WidenMMA_Op<"mfwma.cf.mm", "Double-widening floating-point 8-bit (cf)">;
def MfwmaHfmmOp : BOSCAME_WidenMMA_Op<"mfwma.hf.mm", "Double-widening floating-point 16-bit (half)">;
def MfwmaFmmOp  : BOSCAME_WidenMMA_Op<"mfwma.f.mm",  "Double-widening floating-point 32-bit (single)">;
//===--------------------------------------------------------------------===//
// 3. Quad-widen matrix multiply-accumulate
//===--------------------------------------------------------------------===//
// uint
def MqmauMmOp  : BOSCAME_WidenMMA_Op<"mqmau.mm",   "Quad-widening unsigned dynamic-width">;
def MqmauBmmOp : BOSCAME_WidenMMA_Op<"mqmau.b.mm", "Quad-widening unsigned 8-bit">;

// uint, saturated
def MsqmauMmOp  : BOSCAME_WidenMMA_Op<"msqmau.mm",   "Saturated quad-widening unsigned dynamic-width">;
def MsqmauBmmOp : BOSCAME_WidenMMA_Op<"msqmau.b.mm", "Saturated quad-widening unsigned 8-bit">;

// int
def MqmaMmOp  : BOSCAME_WidenMMA_Op<"mqma.mm",   "Quad-widening signed dynamic-width">;
def MqmaBmmOp : BOSCAME_WidenMMA_Op<"mqma.b.mm", "Quad-widening signed 8-bit">;

// int, saturated
def MsqmaMmOp  : BOSCAME_WidenMMA_Op<"msqma.mm",   "Saturated quad-widening signed dynamic-width">;
def MsqmaBmmOp : BOSCAME_WidenMMA_Op<"msqma.b.mm", "Saturated quad-widening signed 8-bit">;

// float point
def MfqmaMmOp   : BOSCAME_WidenMMA_Op<"mfqma.mm",    "Quad-widening floating-point dynamic-width">;
//def MfqmaCfmmOp : BOSCAME_WidenMMA_Op<"mfqma.cf.mm", "Quad-widening floating-point 8-bit (cf)">;
//===--------------------------------------------------------------------===//
// 4. Oct-widen matrix multiply-accumulate
//===--------------------------------------------------------------------===//
// uint
def MomauMmOp   : BOSCAME_WidenMMA_Op<"momau.mm",    "Oct-widening unsigned dynamic-width">;
//def MomauHbmmOp : BOSCAME_WidenMMA_Op<"momau.hb.mm", "Oct-widening unsigned 4-bit (hb)">;

// uint, saturated
def MsomauMmOp   : BOSCAME_WidenMMA_Op<"msomau.mm",    "Saturated oct-widening unsigned dynamic-width">;
//def MsomauHbmmOp : BOSCAME_WidenMMA_Op<"msomau.hb.mm", "Saturated oct-widening unsigned 4-bit (hb)">;

// int
def MomaMmOp   : BOSCAME_WidenMMA_Op<"moma.mm",    "Oct-widening signed dynamic-width">;
//def MomaHbmmOp : BOSCAME_WidenMMA_Op<"moma.hb.mm", "Oct-widening signed 4-bit (hb)">;

// int, saturated
def MsomaMmOp   : BOSCAME_WidenMMA_Op<"msoma.mm",    "Saturated oct-widening signed dynamic-width">;
//def MsomaHbmmOp : BOSCAME_WidenMMA_Op<"msoma.hb.mm", "Saturated oct-widening signed 4-bit (hb)">;

//===----------------------------------------------------------------------===//
// BOSCAME Intrinsic Operation Definitions (for LLVM IR translation)
// These map directly to LLVM intrinsics defined in IntrinsicsRISCVBOSCExt.td
//===----------------------------------------------------------------------===//

// Base class for BOSCAME intrinsic ops
// Note: enumName must match the intrinsic name pattern: riscv_bosc_<name>
class BOSCAME_IntrOpBase<string mnemonic, int numRes = 0,
                         list<int> overloadedResults = [],
                         list<int> overloadedOperands = [],
                         list<Trait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/BOSCAME_Dialect,
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_bosc_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/overloadedResults,
                  /*list<int> overloadedOperands=*/overloadedOperands,
                  /*list<Trait> traits=*/traits,
                  /*int numResults=*/numRes>;

// Configuration intrinsics (with immediate value)
def BOSCAME_Msettypei_IntrOp : BOSCAME_IntrOpBase<"msettypei", 1, [0]>,
  Arguments<(ins LLVM_Type:$timm)>;
def BOSCAME_Msettypehi_IntrOp : BOSCAME_IntrOpBase<"msettypehi", 1, [0]>,
  Arguments<(ins LLVM_Type:$timm)>;
def BOSCAME_Msettilemi_IntrOp : BOSCAME_IntrOpBase<"msettilemi", 1, [0]>,
  Arguments<(ins LLVM_Type:$tilem)>;
def BOSCAME_Msettileni_IntrOp : BOSCAME_IntrOpBase<"msettileni", 1, [0]>,
  Arguments<(ins LLVM_Type:$tilen)>;
def BOSCAME_Msettileki_IntrOp : BOSCAME_IntrOpBase<"msettileki", 1, [0]>,
  Arguments<(ins LLVM_Type:$tilek)>;

// Configuration intrinsics (with register value)
def BOSCAME_Msettype_IntrOp : BOSCAME_IntrOpBase<"msettype", 1, [0]>,
  Arguments<(ins LLVM_Type:$rs1)>;
def BOSCAME_Msettilem_IntrOp : BOSCAME_IntrOpBase<"msettilem", 1, [0]>,
  Arguments<(ins LLVM_Type:$rs1)>;
def BOSCAME_Msettilen_IntrOp : BOSCAME_IntrOpBase<"msettilen", 1, [0]>,
  Arguments<(ins LLVM_Type:$rs1)>;
def BOSCAME_Msettilek_IntrOp : BOSCAME_IntrOpBase<"msettilek", 1, [0]>,
  Arguments<(ins LLVM_Type:$rs1)>;

//===----------------------------------------------------------------------===//
// Data Move / Broadcast / Transpose Intrinsic Operations (LLVM IR Translation)
//===----------------------------------------------------------------------===//
class BOSCAME_PureMove_IntrOp<string mnemonic>
    : BOSCAME_IntrOpBase<mnemonic, 1, [0]>,
      Arguments<(ins LLVM_Type:$src)>;

class BOSCAME_IndexedMove_IntrOp<string mnemonic>
    : BOSCAME_IntrOpBase<mnemonic, 1, [0], [0, 1]>,
      Arguments<(ins LLVM_Type:$src, LLVM_Type:$rs2)>;

class BOSCAME_ImmIndexedMove_IntrOp<string mnemonic>
    : BOSCAME_IntrOpBase<mnemonic, 1, [0], [0, 1]>,
      Arguments<(ins LLVM_Type:$src, LLVM_Type:$imm)>;

class BOSCAME_MatrixToScalar_IntrOp<string mnemonic>
    : BOSCAME_IntrOpBase<mnemonic, 1, [0], [0, 1]>,
      Arguments<(ins LLVM_Type:$src, LLVM_Type:$rs2)>;

class BOSCAME_ScalarToMatrix_IntrOp<string mnemonic>
    : BOSCAME_IntrOpBase<mnemonic, 1, [0], [0, 1, 2]>,
      Arguments<(ins LLVM_Type:$md_in,
                     LLVM_Type:$value,
                     LLVM_Type:$rs2)>;

//===----------------------------------------------------------------------===//
// Multiclasses for IntrOp Bitwidth Auto-Generation
//===----------------------------------------------------------------------===//
multiclass BOSCAME_PureMove_IntrOp_Multiclass<string prefix, string suffix> {
  def "8_IntrOp"  : BOSCAME_PureMove_IntrOp<prefix # "8."  # suffix>;
  def "16_IntrOp" : BOSCAME_PureMove_IntrOp<prefix # "16." # suffix>;
  def "32_IntrOp" : BOSCAME_PureMove_IntrOp<prefix # "32." # suffix>;
  def "64_IntrOp" : BOSCAME_PureMove_IntrOp<prefix # "64." # suffix>;
}

multiclass BOSCAME_IndexedMove_IntrOp_Multiclass<string prefix, string suffix> {
  def "8_IntrOp"  : BOSCAME_IndexedMove_IntrOp<prefix # "8."  # suffix>;
  def "16_IntrOp" : BOSCAME_IndexedMove_IntrOp<prefix # "16." # suffix>;
  def "32_IntrOp" : BOSCAME_IndexedMove_IntrOp<prefix # "32." # suffix>;
  def "64_IntrOp" : BOSCAME_IndexedMove_IntrOp<prefix # "64." # suffix>;
}

multiclass BOSCAME_ImmIndexedMove_IntrOp_Multiclass<string prefix, string suffix> {
  def "8_IntrOp"  : BOSCAME_ImmIndexedMove_IntrOp<prefix # "8."  # suffix>;
  def "16_IntrOp" : BOSCAME_ImmIndexedMove_IntrOp<prefix # "16." # suffix>;
  def "32_IntrOp" : BOSCAME_ImmIndexedMove_IntrOp<prefix # "32." # suffix>;
  def "64_IntrOp" : BOSCAME_ImmIndexedMove_IntrOp<prefix # "64." # suffix>;
}

multiclass BOSCAME_MatrixToScalar_IntrOp_Multiclass<string prefix, string suffix> {
  def "8_IntrOp"  : BOSCAME_MatrixToScalar_IntrOp<prefix # "8."  # suffix>;
  def "16_IntrOp" : BOSCAME_MatrixToScalar_IntrOp<prefix # "16." # suffix>;
  def "32_IntrOp" : BOSCAME_MatrixToScalar_IntrOp<prefix # "32." # suffix>;
  def "64_IntrOp" : BOSCAME_MatrixToScalar_IntrOp<prefix # "64." # suffix>;
}

multiclass BOSCAME_ScalarToMatrix_IntrOp_Multiclass<string prefix, string suffix> {
  def "8_IntrOp"  : BOSCAME_ScalarToMatrix_IntrOp<prefix # "8."  # suffix>;
  def "16_IntrOp" : BOSCAME_ScalarToMatrix_IntrOp<prefix # "16." # suffix>;
  def "32_IntrOp" : BOSCAME_ScalarToMatrix_IntrOp<prefix # "32." # suffix>;
  def "64_IntrOp" : BOSCAME_ScalarToMatrix_IntrOp<prefix # "64." # suffix>;
}

//===----------------------------------------------------------------------===//
// Data Move Operations Instantiations
//===----------------------------------------------------------------------===//
// 1. Matrix-Matrix Move (Pure Matrix)
defm BOSCAME_MmveTt : BOSCAME_PureMove_IntrOp_Multiclass<"mmve", "t.t">;
defm BOSCAME_MmveAa : BOSCAME_PureMove_IntrOp_Multiclass<"mmve", "a.a">;
// 2. Matrix-Matrix Move (with GPR Index)
defm BOSCAME_MmveAt : BOSCAME_IndexedMove_IntrOp_Multiclass<"mmve", "a.t">;
defm BOSCAME_MmveTa : BOSCAME_IndexedMove_IntrOp_Multiclass<"mmve", "t.a">;
// 3. Matrix-Matrix Move (with Immediate Index)
defm BOSCAME_MmvieAt : BOSCAME_ImmIndexedMove_IntrOp_Multiclass<"mmvie", "a.t">;
defm BOSCAME_MmvieTa : BOSCAME_ImmIndexedMove_IntrOp_Multiclass<"mmvie", "t.a">;
// 4. Matrix to Scalar Move (Returns GPR)
defm BOSCAME_MmveXt  : BOSCAME_MatrixToScalar_IntrOp_Multiclass<"mmve", "x.t">;
defm BOSCAME_MmveXa  : BOSCAME_MatrixToScalar_IntrOp_Multiclass<"mmve", "x.a">;
defm BOSCAME_MfmveFt : BOSCAME_MatrixToScalar_IntrOp_Multiclass<"mfmve", "f.t">;
defm BOSCAME_MfmveFa : BOSCAME_MatrixToScalar_IntrOp_Multiclass<"mfmve", "f.a">;
// 5. Scalar GPR to Matrix Move
defm BOSCAME_MmveTx  : BOSCAME_ScalarToMatrix_IntrOp_Multiclass<"mmve", "t.x">;
defm BOSCAME_MmveAx  : BOSCAME_ScalarToMatrix_IntrOp_Multiclass<"mmve", "a.x">;
defm BOSCAME_MfmveTf : BOSCAME_ScalarToMatrix_IntrOp_Multiclass<"mfmve", "t.f">;
defm BOSCAME_MfmveAf : BOSCAME_ScalarToMatrix_IntrOp_Multiclass<"mfmve", "a.f">;

//===----------------------------------------------------------------------===//
// Broadcast Operations Instantiations
//===----------------------------------------------------------------------===//
// Single Row/Col/Elem Broadcasts (Pure Move semantics)
def BOSCAME_MbcarM_IntrOp : BOSCAME_PureMove_IntrOp<"mbcar.m">;
def BOSCAME_MbcbrM_IntrOp : BOSCAME_PureMove_IntrOp<"mbcbr.m">;
def BOSCAME_MbccrM_IntrOp : BOSCAME_PureMove_IntrOp<"mbccr.m">;

defm BOSCAME_Mbcace : BOSCAME_PureMove_IntrOp_Multiclass<"mbcace", "m">;
defm BOSCAME_Mbcbce : BOSCAME_PureMove_IntrOp_Multiclass<"mbcbce", "m">;
defm BOSCAME_Mbccce : BOSCAME_PureMove_IntrOp_Multiclass<"mbccce", "m">;

defm BOSCAME_Mbcaee : BOSCAME_PureMove_IntrOp_Multiclass<"mbcaee", "m">;
defm BOSCAME_Mbcbee : BOSCAME_PureMove_IntrOp_Multiclass<"mbcbee", "m">;
defm BOSCAME_Mbccee : BOSCAME_PureMove_IntrOp_Multiclass<"mbccee", "m">;

//===----------------------------------------------------------------------===//
// Transpose Operations Instantiations
//===----------------------------------------------------------------------===//
defm BOSCAME_Mtae : BOSCAME_PureMove_IntrOp_Multiclass<"mtae", "m">;
defm BOSCAME_Mtbe : BOSCAME_PureMove_IntrOp_Multiclass<"mtbe", "m">;
defm BOSCAME_Mtce : BOSCAME_PureMove_IntrOp_Multiclass<"mtce", "m">;

//===----------------------------------------------------------------------===//
// Element-Wise Intrinsic Operation Definitions (LLVM IR Translation)
//===----------------------------------------------------------------------===//
class BOSCAME_EleWise3_IntrOp<string mnemonic>
    : BOSCAME_IntrOpBase<mnemonic, 1, [0]>,
      Arguments<(ins LLVM_Type:$src1, LLVM_Type:$src2)>;

class BOSCAME_WidenEleWise3_IntrOp<string mnemonic>
    : BOSCAME_IntrOpBase<mnemonic, 1, [0], [0]>,
      Arguments<(ins LLVM_Type:$src1, LLVM_Type:$src2)>;

class BOSCAME_EleWise2_IntrOp<string mnemonic>
    : BOSCAME_IntrOpBase<mnemonic, 1, [0]>,
      Arguments<(ins LLVM_Type:$src)>;

//===----------------------------------------------------------------------===//
// Multiclasses for IntrOp Bitwidth Auto-Generation
//===----------------------------------------------------------------------===//
multiclass BOSCAME_EleWise_Int_NoWiden_IntrOp<string prefix> {
  def "Mm_IntrOp"   : BOSCAME_EleWise3_IntrOp<prefix # ".mm">;
  def "Bmm_IntrOp"  : BOSCAME_EleWise3_IntrOp<prefix # ".b.mm">;
  def "Hmm_IntrOp"  : BOSCAME_EleWise3_IntrOp<prefix # ".h.mm">;
  def "Wmm_IntrOp"  : BOSCAME_EleWise3_IntrOp<prefix # ".w.mm">;
  def "Dwmm_IntrOp" : BOSCAME_EleWise3_IntrOp<prefix # ".dw.mm">;
}

multiclass BOSCAME_EleWise_Float_NoWiden_IntrOp<string prefix> {
  def "Mm_IntrOp"   : BOSCAME_EleWise3_IntrOp<prefix # ".mm">;
  def "Hfmm_IntrOp" : BOSCAME_EleWise3_IntrOp<prefix # ".hf.mm">;
  def "Fmm_IntrOp"  : BOSCAME_EleWise3_IntrOp<prefix # ".f.mm">;
  def "Dmm_IntrOp"  : BOSCAME_EleWise3_IntrOp<prefix # ".d.mm">;
}

multiclass BOSCAME_EleWise_Int_DoubleWiden_IntrOp<string prefix> {
  def "Mm_IntrOp"   : BOSCAME_WidenEleWise3_IntrOp<prefix # ".mm">;
  def "Bmm_IntrOp"  : BOSCAME_WidenEleWise3_IntrOp<prefix # ".b.mm">;
  def "Hmm_IntrOp"  : BOSCAME_WidenEleWise3_IntrOp<prefix # ".h.mm">;
  def "Wmm_IntrOp"  : BOSCAME_WidenEleWise3_IntrOp<prefix # ".w.mm">;
}

multiclass BOSCAME_EleWise_Float_DoubleWiden_IntrOp<string prefix> {
  def "Mm_IntrOp"   : BOSCAME_WidenEleWise3_IntrOp<prefix # ".mm">;
  def "Hfmm_IntrOp" : BOSCAME_WidenEleWise3_IntrOp<prefix # ".hf.mm">;
  def "Fmm_IntrOp"  : BOSCAME_WidenEleWise3_IntrOp<prefix # ".f.mm">;
}

multiclass BOSCAME_EleWise_Unary_Float_IntrOp<string prefix> {
  def "M_IntrOp"   : BOSCAME_EleWise2_IntrOp<prefix # ".m">;
  def "HfM_IntrOp" : BOSCAME_EleWise2_IntrOp<prefix # ".hf.m">;
  def "FM_IntrOp"  : BOSCAME_EleWise2_IntrOp<prefix # ".f.m">;
  def "DM_IntrOp"  : BOSCAME_EleWise2_IntrOp<prefix # ".d.m">;
}

//===----------------------------------------------------------------------===//
// Element-Wise IntrOp Instantiations
//===----------------------------------------------------------------------===//
// 1. ADD / SUB - No Widen
defm BOSCAME_Maddu  : BOSCAME_EleWise_Int_NoWiden_IntrOp<"maddu">;
defm BOSCAME_Msaddu : BOSCAME_EleWise_Int_NoWiden_IntrOp<"msaddu">;
defm BOSCAME_Madd   : BOSCAME_EleWise_Int_NoWiden_IntrOp<"madd">;
defm BOSCAME_Msadd  : BOSCAME_EleWise_Int_NoWiden_IntrOp<"msadd">;
defm BOSCAME_Mfadd  : BOSCAME_EleWise_Float_NoWiden_IntrOp<"mfadd">;

defm BOSCAME_Msubu  : BOSCAME_EleWise_Int_NoWiden_IntrOp<"msubu">;
defm BOSCAME_Mssubu : BOSCAME_EleWise_Int_NoWiden_IntrOp<"mssubu">;
defm BOSCAME_Msub   : BOSCAME_EleWise_Int_NoWiden_IntrOp<"msub">;
defm BOSCAME_Mssub  : BOSCAME_EleWise_Int_NoWiden_IntrOp<"mssub">;
defm BOSCAME_Mfsub  : BOSCAME_EleWise_Float_NoWiden_IntrOp<"mfsub">;
// 2. ADD / SUB - Double Widen
defm BOSCAME_Mwaddu : BOSCAME_EleWise_Int_DoubleWiden_IntrOp<"mwaddu">;
defm BOSCAME_Mwadd  : BOSCAME_EleWise_Int_DoubleWiden_IntrOp<"mwadd">;
defm BOSCAME_Mfwadd : BOSCAME_EleWise_Float_DoubleWiden_IntrOp<"mfwadd">;

defm BOSCAME_Mwsubu : BOSCAME_EleWise_Int_DoubleWiden_IntrOp<"mwsubu">;
defm BOSCAME_Mwsub  : BOSCAME_EleWise_Int_DoubleWiden_IntrOp<"mwsub">;
defm BOSCAME_Mfwsub : BOSCAME_EleWise_Float_DoubleWiden_IntrOp<"mfwsub">;
// 3. MIN / MAX - No Widen
defm BOSCAME_Mminu : BOSCAME_EleWise_Int_NoWiden_IntrOp<"mminu">;
defm BOSCAME_Mmaxu : BOSCAME_EleWise_Int_NoWiden_IntrOp<"mmaxu">;
defm BOSCAME_Mmin  : BOSCAME_EleWise_Int_NoWiden_IntrOp<"mmin">;
defm BOSCAME_Mmax  : BOSCAME_EleWise_Int_NoWiden_IntrOp<"mmax">;
defm BOSCAME_Mfmin : BOSCAME_EleWise_Float_NoWiden_IntrOp<"mfmin">;
defm BOSCAME_Mfmax : BOSCAME_EleWise_Float_NoWiden_IntrOp<"mfmax">;
// 4. MUL / MULH / DIV - No Widen
defm BOSCAME_Msmulu  : BOSCAME_EleWise_Int_NoWiden_IntrOp<"msmulu">;
defm BOSCAME_Mmul    : BOSCAME_EleWise_Int_NoWiden_IntrOp<"mmul">;
defm BOSCAME_Msmul   : BOSCAME_EleWise_Int_NoWiden_IntrOp<"msmul">;
defm BOSCAME_Mfmul   : BOSCAME_EleWise_Float_NoWiden_IntrOp<"mfmul">;
defm BOSCAME_Mmulhu  : BOSCAME_EleWise_Int_NoWiden_IntrOp<"mmulhu">;
defm BOSCAME_Mmulh   : BOSCAME_EleWise_Int_NoWiden_IntrOp<"mmulh">;
defm BOSCAME_Mmulhsu : BOSCAME_EleWise_Int_NoWiden_IntrOp<"mmulhsu">;
defm BOSCAME_Msmulsu : BOSCAME_EleWise_Int_NoWiden_IntrOp<"msmulsu">;
defm BOSCAME_Mfdiv   : BOSCAME_EleWise_Float_NoWiden_IntrOp<"mfdiv">;
// 5. MUL - Double Widen
defm BOSCAME_Mwmulu  : BOSCAME_EleWise_Int_DoubleWiden_IntrOp<"mwmulu">;
defm BOSCAME_Mwmul   : BOSCAME_EleWise_Int_DoubleWiden_IntrOp<"mwmul">;
defm BOSCAME_Mwmulsu : BOSCAME_EleWise_Int_DoubleWiden_IntrOp<"mwmulsu">;
defm BOSCAME_Mfwmul  : BOSCAME_EleWise_Float_DoubleWiden_IntrOp<"mfwmul">;
// 6. LOGIC & SHIFT - No Widen (Individual & Multiclass)
def BOSCAME_MandMm_IntrOp : BOSCAME_EleWise3_IntrOp<"mand.mm">;
def BOSCAME_MorMm_IntrOp  : BOSCAME_EleWise3_IntrOp<"mor.mm">;
def BOSCAME_MxorMm_IntrOp : BOSCAME_EleWise3_IntrOp<"mxor.mm">;

defm BOSCAME_Msll : BOSCAME_EleWise_Int_NoWiden_IntrOp<"msll">;
defm BOSCAME_Msrl : BOSCAME_EleWise_Int_NoWiden_IntrOp<"msrl">;
defm BOSCAME_Msra : BOSCAME_EleWise_Int_NoWiden_IntrOp<"msra">;
// 7. SQRT - No Widen (Unary)
// The unary MLIR/LLVM intrinsic and selected hardware instruction use the
// ``.m`` suffix.
def BOSCAME_MfsqrtM_IntrOp   : BOSCAME_EleWise2_IntrOp<"mfsqrt.m">;
def BOSCAME_MfsqrtHfM_IntrOp : BOSCAME_EleWise2_IntrOp<"mfsqrt.hf.m">;
def BOSCAME_MfsqrtFM_IntrOp  : BOSCAME_EleWise2_IntrOp<"mfsqrt.f.m">;
def BOSCAME_MfsqrtDM_IntrOp  : BOSCAME_EleWise2_IntrOp<"mfsqrt.d.m">;

//===----------------------------------------------------------------------===//
// BOSCAME Intrinsic Operation Definitions (LLVM IR Translation)
// Matrix Load/Store Intrinsics Generators
//===----------------------------------------------------------------------===//
class BOSCAME_Load_IntrOp<string mnemonic>
    : BOSCAME_IntrOpBase<mnemonic, 1, [0], [1]>,
      Arguments<(ins LLVM_Type:$base, LLVM_Type:$stride)>;

class BOSCAME_Store_IntrOp<string mnemonic>
    : BOSCAME_IntrOpBase<mnemonic, 0, [], [0, 2]>,
      Arguments<(ins LLVM_Type:$src, LLVM_Type:$base, LLVM_Type:$stride)>;

// Multiclass to auto-generate 8/16/32/64 bit variants for Loads
multiclass BOSCAME_Load_IntrOp_Multiclass<string prefix> {
  def "8m_IntrOp"  : BOSCAME_Load_IntrOp<prefix # "8.m">;
  def "16m_IntrOp" : BOSCAME_Load_IntrOp<prefix # "16.m">;
  def "32m_IntrOp" : BOSCAME_Load_IntrOp<prefix # "32.m">;
  def "64m_IntrOp" : BOSCAME_Load_IntrOp<prefix # "64.m">;
}

// Multiclass to auto-generate 8/16/32/64 bit variants for Stores
multiclass BOSCAME_Store_IntrOp_Multiclass<string prefix> {
  def "8m_IntrOp"  : BOSCAME_Store_IntrOp<prefix # "8.m">;
  def "16m_IntrOp" : BOSCAME_Store_IntrOp<prefix # "16.m">;
  def "32m_IntrOp" : BOSCAME_Store_IntrOp<prefix # "32.m">;
  def "64m_IntrOp" : BOSCAME_Store_IntrOp<prefix # "64.m">;
}

//===----------------------------------------------------------------------===//
// Matrix Load Intrinsics Instantiations
//===----------------------------------------------------------------------===//
// Matrix A
defm BOSCAME_Mlae   : BOSCAME_Load_IntrOp_Multiclass<"mlae">;
defm BOSCAME_Mlate  : BOSCAME_Load_IntrOp_Multiclass<"mlate">;
defm BOSCAME_Mltre  : BOSCAME_Load_IntrOp_Multiclass<"mltre">;
// Matrix B
defm BOSCAME_Mlbe   : BOSCAME_Load_IntrOp_Multiclass<"mlbe">;
defm BOSCAME_Mlbte  : BOSCAME_Load_IntrOp_Multiclass<"mlbte">;
// Matrix C (Accumulator)
defm BOSCAME_Mlce   : BOSCAME_Load_IntrOp_Multiclass<"mlce">;
defm BOSCAME_Mlcte  : BOSCAME_Load_IntrOp_Multiclass<"mlcte">;
defm BOSCAME_Mlacce : BOSCAME_Load_IntrOp_Multiclass<"mlacce">;
//===----------------------------------------------------------------------===//
// Matrix Store Intrinsics Instantiations
//===----------------------------------------------------------------------===//
// Matrix A
defm BOSCAME_Msae   : BOSCAME_Store_IntrOp_Multiclass<"msae">;
defm BOSCAME_Msate  : BOSCAME_Store_IntrOp_Multiclass<"msate">;
defm BOSCAME_Mstre  : BOSCAME_Store_IntrOp_Multiclass<"mstre">;
// Matrix B
defm BOSCAME_Msbe   : BOSCAME_Store_IntrOp_Multiclass<"msbe">;
defm BOSCAME_Msbte  : BOSCAME_Store_IntrOp_Multiclass<"msbte">;
// Matrix C (Accumulator)
defm BOSCAME_Msce   : BOSCAME_Store_IntrOp_Multiclass<"msce">;
defm BOSCAME_Mscte  : BOSCAME_Store_IntrOp_Multiclass<"mscte">;
defm BOSCAME_Msacce : BOSCAME_Store_IntrOp_Multiclass<"msacce">;

//===----------------------------------------------------------------------===//
// Tile register matrix multiplication intrinsics (LLVM IR Translation)
//===----------------------------------------------------------------------===//
class BOSCAME_MMA_IntrOp<string mnemonic>
    : BOSCAME_IntrOpBase<mnemonic, 1, [0]>,
      Arguments<(ins LLVM_Type:$acc, LLVM_Type:$lhs, LLVM_Type:$rhs)>;

class BOSCAME_WidenMMA_IntrOp<string mnemonic>
    : BOSCAME_IntrOpBase<mnemonic, 1, [0], [1]>,
      Arguments<(ins LLVM_Type:$acc, LLVM_Type:$lhs, LLVM_Type:$rhs)>;

//===--------------------------------------------------------------------===//
// 1. No-widen matrix multiply-accumulate
//===--------------------------------------------------------------------===//
// int
def BOSCAME_MmaMm_IntrOp   : BOSCAME_MMA_IntrOp<"mma.mm">;
def BOSCAME_MmaHmm_IntrOp  : BOSCAME_MMA_IntrOp<"mma.h.mm">;
def BOSCAME_MmaWmm_IntrOp  : BOSCAME_MMA_IntrOp<"mma.w.mm">;
def BOSCAME_MmaDwmm_IntrOp : BOSCAME_MMA_IntrOp<"mma.dw.mm">;
// int, saturated
def BOSCAME_MsmaMm_IntrOp   : BOSCAME_MMA_IntrOp<"msma.mm">;
def BOSCAME_MsmaHmm_IntrOp  : BOSCAME_MMA_IntrOp<"msma.h.mm">;
def BOSCAME_MsmaWmm_IntrOp  : BOSCAME_MMA_IntrOp<"msma.w.mm">;
def BOSCAME_MsmaDwmm_IntrOp : BOSCAME_MMA_IntrOp<"msma.dw.mm">;
// uint
def BOSCAME_MmauMm_IntrOp   : BOSCAME_MMA_IntrOp<"mmau.mm">;
def BOSCAME_MmauHmm_IntrOp  : BOSCAME_MMA_IntrOp<"mmau.h.mm">;
def BOSCAME_MmauWmm_IntrOp  : BOSCAME_MMA_IntrOp<"mmau.w.mm">;
def BOSCAME_MmauDwmm_IntrOp : BOSCAME_MMA_IntrOp<"mmau.dw.mm">;
// uint, saturated
def BOSCAME_MsmauMm_IntrOp   : BOSCAME_MMA_IntrOp<"msmau.mm">;
def BOSCAME_MsmauHmm_IntrOp  : BOSCAME_MMA_IntrOp<"msmau.h.mm">;
def BOSCAME_MsmauWmm_IntrOp  : BOSCAME_MMA_IntrOp<"msmau.w.mm">;
def BOSCAME_MsmauDwmm_IntrOp : BOSCAME_MMA_IntrOp<"msmau.dw.mm">;
// float point
def BOSCAME_MfmaMm_IntrOp   : BOSCAME_MMA_IntrOp<"mfma.mm">;
def BOSCAME_MfmaHfmm_IntrOp : BOSCAME_MMA_IntrOp<"mfma.hf.mm">;
def BOSCAME_MfmaFmm_IntrOp  : BOSCAME_MMA_IntrOp<"mfma.f.mm">;
def BOSCAME_MfmaDmm_IntrOp  : BOSCAME_MMA_IntrOp<"mfma.d.mm">;

//===--------------------------------------------------------------------===//
// 2. Double-widen matrix multiply-accumulate
//===--------------------------------------------------------------------===//
// uint
def BOSCAME_MwmauMm_IntrOp   : BOSCAME_WidenMMA_IntrOp<"mwmau.mm">;
def BOSCAME_MwmauHmm_IntrOp  : BOSCAME_WidenMMA_IntrOp<"mwmau.h.mm">;
def BOSCAME_MwmauWmm_IntrOp  : BOSCAME_WidenMMA_IntrOp<"mwmau.w.mm">;
// uint, saturated
def BOSCAME_MswmauMm_IntrOp  : BOSCAME_WidenMMA_IntrOp<"mswmau.mm">;
def BOSCAME_MswmauHmm_IntrOp : BOSCAME_WidenMMA_IntrOp<"mswmau.h.mm">;
def BOSCAME_MswmauWmm_IntrOp : BOSCAME_WidenMMA_IntrOp<"mswmau.w.mm">;
// int
def BOSCAME_MwmaMm_IntrOp   : BOSCAME_WidenMMA_IntrOp<"mwma.mm">;
def BOSCAME_MwmaHmm_IntrOp  : BOSCAME_WidenMMA_IntrOp<"mwma.h.mm">;
def BOSCAME_MwmaWmm_IntrOp  : BOSCAME_WidenMMA_IntrOp<"mwma.w.mm">;
// int, saturated
def BOSCAME_MswmaMm_IntrOp   : BOSCAME_WidenMMA_IntrOp<"mswma.mm">;
def BOSCAME_MswmaHmm_IntrOp  : BOSCAME_WidenMMA_IntrOp<"mswma.h.mm">;
def BOSCAME_MswmaWmm_IntrOp  : BOSCAME_WidenMMA_IntrOp<"mswma.w.mm">;
// float point
def BOSCAME_MfwmaMm_IntrOp   : BOSCAME_WidenMMA_IntrOp<"mfwma.mm">;
def BOSCAME_MfwmaHfmm_IntrOp : BOSCAME_WidenMMA_IntrOp<"mfwma.hf.mm">;
def BOSCAME_MfwmaFmm_IntrOp  : BOSCAME_WidenMMA_IntrOp<"mfwma.f.mm">;

//===--------------------------------------------------------------------===//
// 3. Quad-widen matrix multiply-accumulate
//===--------------------------------------------------------------------===//
// uint
def BOSCAME_MqmauMm_IntrOp  : BOSCAME_WidenMMA_IntrOp<"mqmau.mm">;
def BOSCAME_MqmauBmm_IntrOp : BOSCAME_WidenMMA_IntrOp<"mqmau.b.mm">;
// uint, saturated
def BOSCAME_MsqmauMm_IntrOp  : BOSCAME_WidenMMA_IntrOp<"msqmau.mm">;
def BOSCAME_MsqmauBmm_IntrOp : BOSCAME_WidenMMA_IntrOp<"msqmau.b.mm">;
// int
def BOSCAME_MqmaMm_IntrOp  : BOSCAME_WidenMMA_IntrOp<"mqma.mm">;
def BOSCAME_MqmaBmm_IntrOp : BOSCAME_WidenMMA_IntrOp<"mqma.b.mm">;
// int, saturated
def BOSCAME_MsqmaMm_IntrOp  : BOSCAME_WidenMMA_IntrOp<"msqma.mm">;
def BOSCAME_MsqmaBmm_IntrOp : BOSCAME_WidenMMA_IntrOp<"msqma.b.mm">;
// float point
def BOSCAME_MfqmaMm_IntrOp   : BOSCAME_WidenMMA_IntrOp<"mfqma.mm">;

//===--------------------------------------------------------------------===//
// 4. Oct-widen matrix multiply-accumulate
//===--------------------------------------------------------------------===//
// uint
def BOSCAME_MomauMm_IntrOp   : BOSCAME_WidenMMA_IntrOp<"momau.mm">;
// uint, saturated
def BOSCAME_MsomauMm_IntrOp   : BOSCAME_WidenMMA_IntrOp<"msomau.mm">;
// int
def BOSCAME_MomaMm_IntrOp   : BOSCAME_WidenMMA_IntrOp<"moma.mm">;
// int, saturated
def BOSCAME_MsomaMm_IntrOp   : BOSCAME_WidenMMA_IntrOp<"msoma.mm">;

#endif // BOSCAME_DIALECT
