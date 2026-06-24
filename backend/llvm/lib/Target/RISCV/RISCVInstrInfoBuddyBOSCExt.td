//===- RISCVInstrInfoBuddyBOSCExt.td --------------------------------------===//
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
// This is the instruction information file of RISC-V buddy BOSC AME extension.
//
//===----------------------------------------------------------------------===//

include "llvm/IR/IntrinsicsRISCVBuddyBOSCExt.td"

//===----------------------------------------------------------------------===//
// BOSC AME (RISC-V Matrix Extension) 32-bit Instructions
//===----------------------------------------------------------------------===//
// Reference: BOSC RISC-V Matrix Extension Specification
// 32-bit encoding format with prefix 1110111 at bits [6:0]
//
// Register Model:
//   - TileReg (tr0-tr7): Input matrices A and B, MLEN bits each
//   - AccReg (acc0-acc7): Accumulator matrix C, MLEN×AMUL bits each
//
// Data Flow:
//   Memory --[mlae/mlbe]--> TileReg --[mma]--> AccReg --[msce]--> Memory
//
// Matrix Multiplication Instruction Format (32-bit):
// | 31:26  | 25 | 24 | 23:20 | 19 | 18:15 | 14:12 | 11 | 10:7 | 6:0    |
// | funct6 | fp | sa | ms2   | sn | ms1   | eew   | ma | md   | OP-M32 |
//
// suffix = 1110111 (BOSC AME prefix)
//
// Widening Instructions (required for AMUL > 1):
//   - mqma.b.mm: int8 → int32 (quad-widen, AMUL ≥ 4, mmi8i32 MANDATORY)
//   - mwma.h.mm: int16 → int32 (double-widen, AMUL ≥ 2)
//   - mwma.w.mm: int32 → int64 (double-widen, AMUL ≥ 2)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AME Custom Operand Types for Tile Register Indices
//===----------------------------------------------------------------------===//
// These operand types allow intrinsics to pass immediate indices (0-7)
// for tile and accumulator registers. The AsmString of pseudo instructions
// hardcodes "acc" and "tr" prefixes so that indices 0-7 are printed as
// acc0-acc7 and tr0-tr7 respectively, without modifying LLVM submodule.

// AsmOperandClass for tile register index (0-7)
def BOSCAMETileIndexAsmOperand : AsmOperandClass {
  let Name = "BOSCAMETileIndex";
  let RenderMethod = "addImmOperands";
  let PredicateMethod = "isUImm3";
  let DiagnosticType = "InvalidBOSCAMETileIndex";
}

// AsmOperandClass for accumulator register index (0-7)
def BOSCAMEAccIndexAsmOperand : AsmOperandClass {
  let Name = "BOSCAMEAccIndex";
  let RenderMethod = "addImmOperands";
  let PredicateMethod = "isUImm3";
  let DiagnosticType = "InvalidBOSCAMEAccIndex";
}

// Operand type for TileReg index (0-7), printed with "tr" prefix in AsmString
def BOSCAMETileIndex : RISCVOp<i64> {
  let ParserMatchClass = BOSCAMETileIndexAsmOperand;
  let DecoderMethod = "decodeUImmOperand<3>";
  let OperandType = "OPERAND_UIMM3";
}

// Operand type for AccReg index (0-7), printed with "acc" prefix in AsmString
def BOSCAMEAccIndex : RISCVOp<i64> {
  let ParserMatchClass = BOSCAMEAccIndexAsmOperand;
  let DecoderMethod = "decodeUImmOperand<3>";
  let OperandType = "OPERAND_UIMM3";
}

//===----------------------------------------------------------------------===//
// BOSC AME 32-bit Instruction Format Base Class
//===----------------------------------------------------------------------===//

// Base class for BOSC AME 32-bit matrix multiplication instructions
// Uses MatrixReg for inputs (ms1, ms2) and MatrixReg for output (md)
class RVInstBOSCAME32<dag outs, dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<6> funct6;
  bit     fp;     // 0 for integer, 1 for floating-point (if supported in future)
  bit     sa;     // 0 for unsaturated, 1 for saturated
  bits<4> ms2;
  bit     sn;     // Whether the source operand is signed (for integers), 0 for floating-point
  bits<4> ms1;
  bits<3> eew;    // 0=int8, 1=int16, 2=int32, 3=int64, 7=int4
  bit     ma = 0b1;
  bits<4> md;
  bits<7> opcode = 0b1110111;

  let Inst{31-26} = funct6;
  let Inst{25}    = fp;
  let Inst{24}    = sa;
  let Inst{23-20} = ms2;
  let Inst{19}    = sn;
  let Inst{18-15} = ms1;
  let Inst{14-12} = eew;
  let Inst{11}    = ma;
  let Inst{10-7}  = md;
  let Inst{6-0} = 0b1110111;
}

//===----------------------------------------------------------------------===//
// BOSC AME 32-bit Matrix Multiplication Instructions
//===----------------------------------------------------------------------===//
// Format: mma.{h|w|dw}.mm acc, tr, tr
// Semantics: acc = acc + tr1 * tr2
//
// Data Flow: TileReg × TileReg → AccReg (accumulate)
//   - ms1: TileReg for matrix A
//   - ms2: TileReg for matrix B
//   - md:  AccReg for accumulation result C
//
// eew encoding:
//   000 = int8 (b), 001 = int16 (h), 010 = int32 (w), 011 = int64 (dw)
//   100 = use mtype.msew, 111 = int4
//
// funct6 encoding:
//   001000 = no-widen
//   001001 = double-widen
//   001010 = quad-widen
//   001011 = oct-widen
//===----------------------------------------------------------------------===//

// No-widen matrix multiply-accumulate: acc = acc + tr1 * tr2
// Input and output have the same element width
class BOSC_AME_MMA_NO_WIDEN<bit fp_val, bit sa_val, bit sn_val,
                   bits<3> eew_val, string opcodestr>
    : RVInstBOSCAME32<(outs AccReg:$md),
                  (ins AccReg:$md_in, TileReg:$ms1, TileReg:$ms2),
                  opcodestr, "$md, $ms1, $ms2"> {
  let Constraints = "$md = $md_in";  // Accumulator constraint
  let funct6 = 0b001000;     // No widening
  let fp = fp_val;
  let sa = sa_val;        // Matrix multiplication
  let sn = sn_val;        // (signed for integers, 0 for floating-point)
  let eew = eew_val;
  let ma = 0b1;       // Accumulate
  let opcode = 0b1110111;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  // int 16 32 64 , output no-widen
  def BOSC_AME_MMA_MM    : BOSC_AME_MMA_NO_WIDEN<0b0, 0b0, 0b1, 0b100, "mma.mm">;
  def BOSC_AME_MMA_H_MM  : BOSC_AME_MMA_NO_WIDEN<0b0, 0b0, 0b1, 0b001, "mma.h.mm">;
  def BOSC_AME_MMA_W_MM  : BOSC_AME_MMA_NO_WIDEN<0b0, 0b0, 0b1, 0b010, "mma.w.mm">;
  def BOSC_AME_MMA_DW_MM : BOSC_AME_MMA_NO_WIDEN<0b0, 0b0, 0b1, 0b011, "mma.dw.mm">;

  // int 16 32 64 , output  no-widen and saturated
  def BOSC_AME_MSMA_MM    : BOSC_AME_MMA_NO_WIDEN<0b0, 0b1, 0b1, 0b100, "msma.mm">;
  def BOSC_AME_MSMA_H_MM  : BOSC_AME_MMA_NO_WIDEN<0b0, 0b1, 0b1, 0b001, "msma.h.mm">;
  def BOSC_AME_MSMA_W_MM  : BOSC_AME_MMA_NO_WIDEN<0b0, 0b1, 0b1, 0b010, "msma.w.mm">;
  def BOSC_AME_MSMA_DW_MM : BOSC_AME_MMA_NO_WIDEN<0b0, 0b1, 0b1, 0b011, "msma.dw.mm">;

  // uint 16 32 64 , output no-widen
  def BOSC_AME_MMAU_MM    : BOSC_AME_MMA_NO_WIDEN<0b0, 0b0, 0b0, 0b100, "mmau.mm">;
  def BOSC_AME_MMAU_H_MM  : BOSC_AME_MMA_NO_WIDEN<0b0, 0b0, 0b0, 0b001, "mmau.h.mm">;
  def BOSC_AME_MMAU_W_MM  : BOSC_AME_MMA_NO_WIDEN<0b0, 0b0, 0b0, 0b010, "mmau.w.mm">;
  def BOSC_AME_MMAU_DW_MM : BOSC_AME_MMA_NO_WIDEN<0b0, 0b0, 0b0, 0b011, "mmau.dw.mm">;

  // uint 16 32 64 , output no-widen and saturated
  def BOSC_AME_MSMAU_MM    : BOSC_AME_MMA_NO_WIDEN<0b0, 0b1, 0b0, 0b100, "msmau.mm">;
  def BOSC_AME_MSMAU_H_MM  : BOSC_AME_MMA_NO_WIDEN<0b0, 0b1, 0b0, 0b001, "msmau.h.mm">;
  def BOSC_AME_MSMAU_W_MM  : BOSC_AME_MMA_NO_WIDEN<0b0, 0b1, 0b0, 0b010, "msmau.w.mm">;
  def BOSC_AME_MSMAU_DW_MM : BOSC_AME_MMA_NO_WIDEN<0b0, 0b1, 0b0, 0b011, "msmau.dw.mm">;

  // float point 16 32 64 , output no-widen
  def BOSC_AME_MFMA_MM    : BOSC_AME_MMA_NO_WIDEN<0b1, 0b0, 0b0, 0b100, "mfma.mm">;
  def BOSC_AME_MFMA_HF_MM : BOSC_AME_MMA_NO_WIDEN<0b1, 0b0, 0b0, 0b001, "mfma.hf.mm">;
  def BOSC_AME_MFMA_F_MM  : BOSC_AME_MMA_NO_WIDEN<0b1, 0b0, 0b0, 0b010, "mfma.f.mm">;
  def BOSC_AME_MFMA_D_MM  : BOSC_AME_MMA_NO_WIDEN<0b1, 0b0, 0b0, 0b011, "mfma.d.mm">;
}

// double-widen matrix multiply-accumulate: acc = acc + tr1 * tr2
class BOSC_AME_MMA_DOUBLE_WIDEN<bit fp_val, bit sa_val, bit sn_val,
                                bits<3> eew_val, string opcodestr>
    : RVInstBOSCAME32<(outs AccReg:$md),
                      (ins AccReg:$md_in, TileReg:$ms1, TileReg:$ms2),
                      opcodestr, "$md, $ms1, $ms2"> {
  let Constraints = "$md = $md_in";
  let funct6 = 0b001001;
  let fp     = fp_val;
  let sa     = sa_val;
  let sn     = sn_val;
  let eew    = eew_val;
  let ma     = 0b1;
  let opcode = 0b1110111;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  // uint 16→32, 32→64 with double widening
  def BOSC_AME_MWMAU_MM    : BOSC_AME_MMA_DOUBLE_WIDEN<0b0, 0b0, 0b0, 0b100, "mwmau.mm">;
  def BOSC_AME_MWMAU_H_MM  : BOSC_AME_MMA_DOUBLE_WIDEN<0b0, 0b0, 0b0, 0b001, "mwmau.h.mm">;
  def BOSC_AME_MWMAU_W_MM  : BOSC_AME_MMA_DOUBLE_WIDEN<0b0, 0b0, 0b0, 0b010, "mwmau.w.mm">;

  // uint 16→32, 32→64 with double widening and saturation
  def BOSC_AME_MSWMAU_MM   : BOSC_AME_MMA_DOUBLE_WIDEN<0b0, 0b1, 0b0, 0b100, "mswmau.mm">;
  def BOSC_AME_MSWMAU_H_MM : BOSC_AME_MMA_DOUBLE_WIDEN<0b0, 0b1, 0b0, 0b001, "mswmau.h.mm">;
  def BOSC_AME_MSWMAU_W_MM : BOSC_AME_MMA_DOUBLE_WIDEN<0b0, 0b1, 0b0, 0b010, "mswmau.w.mm">;

  // int 16→32, 32→64 with double widening
  def BOSC_AME_MWMA_MM     : BOSC_AME_MMA_DOUBLE_WIDEN<0b0, 0b0, 0b1, 0b100, "mwma.mm">;
  def BOSC_AME_MWMA_H_MM   : BOSC_AME_MMA_DOUBLE_WIDEN<0b0, 0b0, 0b1, 0b001, "mwma.h.mm">;
  def BOSC_AME_MWMA_W_MM   : BOSC_AME_MMA_DOUBLE_WIDEN<0b0, 0b0, 0b1, 0b010, "mwma.w.mm">;

  // int 16→32, 32→64 with double widening and saturation
  def BOSC_AME_MSWMA_MM    : BOSC_AME_MMA_DOUBLE_WIDEN<0b0, 0b1, 0b1, 0b100, "mswma.mm">;
  def BOSC_AME_MSWMA_H_MM  : BOSC_AME_MMA_DOUBLE_WIDEN<0b0, 0b1, 0b1, 0b001, "mswma.h.mm">;
  def BOSC_AME_MSWMA_W_MM  : BOSC_AME_MMA_DOUBLE_WIDEN<0b0, 0b1, 0b1, 0b010, "mswma.w.mm">;

  // float point 8→16, 16→32, 32→64 with double widening
  def BOSC_AME_MFWMA_MM    : BOSC_AME_MMA_DOUBLE_WIDEN<0b1, 0b0, 0b0, 0b100, "mfwma.mm">;
  def BOSC_AME_MFWMA_CF_MM : BOSC_AME_MMA_DOUBLE_WIDEN<0b1, 0b0, 0b0, 0b000, "mfwma.cf.mm">;
  def BOSC_AME_MFWMA_HF_MM : BOSC_AME_MMA_DOUBLE_WIDEN<0b1, 0b0, 0b0, 0b001, "mfwma.hf.mm">;
  def BOSC_AME_MFWMA_F_MM  : BOSC_AME_MMA_DOUBLE_WIDEN<0b1, 0b0, 0b0, 0b010, "mfwma.f.mm">;
}

// quad-widen matrix multiply-accumulate: acc = acc + tr1 * tr2
class BOSC_AME_MMA_QUAD_WIDEN<bit fp_val, bit sa_val, bit sn_val,
                               bits<3> eew_val, string opcodestr>
    : RVInstBOSCAME32<(outs AccReg:$md),
                      (ins AccReg:$md_in, TileReg:$ms1, TileReg:$ms2),
                      opcodestr, "$md, $ms1, $ms2"> {
  let Constraints = "$md = $md_in";
  let funct6 = 0b001010;
  let fp     = fp_val;
  let sa     = sa_val;
  let sn     = sn_val;
  let eew    = eew_val;
  let ma     = 0b1;
  let opcode = 0b1110111;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  // uint 8→32 with quad widening
  def BOSC_AME_MQMAU_MM   : BOSC_AME_MMA_QUAD_WIDEN<0b0, 0b0, 0b0, 0b100, "mqmau.mm">;
  def BOSC_AME_MQMAU_B_MM : BOSC_AME_MMA_QUAD_WIDEN<0b0, 0b0, 0b0, 0b000, "mqmau.b.mm">;
  // uint 8→32 with quad widening and saturation
  def BOSC_AME_MSQMAU_MM   : BOSC_AME_MMA_QUAD_WIDEN<0b0, 0b1, 0b0, 0b100, "msqmau.mm">;
  def BOSC_AME_MSQMAU_B_MM : BOSC_AME_MMA_QUAD_WIDEN<0b0, 0b1, 0b0, 0b000, "msqmau.b.mm">;
  // int 8→32 with quad widening
  def BOSC_AME_MQMA_MM   : BOSC_AME_MMA_QUAD_WIDEN<0b0, 0b0, 0b1, 0b100, "mqma.mm">;
  def BOSC_AME_MQMA_B_MM : BOSC_AME_MMA_QUAD_WIDEN<0b0, 0b0, 0b1, 0b000, "mqma.b.mm">;
  // int 8→32 with quad widening and saturation
  def BOSC_AME_MSQMA_MM   : BOSC_AME_MMA_QUAD_WIDEN<0b0, 0b1, 0b1, 0b100, "msqma.mm">;
  def BOSC_AME_MSQMA_B_MM : BOSC_AME_MMA_QUAD_WIDEN<0b0, 0b1, 0b1, 0b000, "msqma.b.mm">;
  // float point 8→32 with quad widening
  def BOSC_AME_MFQMA_MM    : BOSC_AME_MMA_QUAD_WIDEN<0b1, 0b0, 0b1, 0b100, "mfqma.mm">;
  def BOSC_AME_MFQMA_CF_MM : BOSC_AME_MMA_QUAD_WIDEN<0b1, 0b0, 0b1, 0b000, "mfqma.cf.mm">;
}

// oct-widen matrix multiply-accumulate: acc = acc + tr1 * tr2
class BOSC_AME_MMA_OCT_WIDEN<bit fp_val, bit sa_val, bit sn_val,
                              bits<3> eew_val, string opcodestr>
    : RVInstBOSCAME32<(outs AccReg:$md),
                      (ins AccReg:$md_in, TileReg:$ms1, TileReg:$ms2),
                      opcodestr, "$md, $ms1, $ms2"> {
  let Constraints = "$md = $md_in";
  let funct6 = 0b001011;
  let fp     = fp_val;
  let sa     = sa_val;
  let sn     = sn_val;
  let eew    = eew_val;
  let ma     = 0b1;
  let opcode = 0b1110111;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  // uint 4→32 with oct widening
  def BOSC_AME_MOMAU_MM    : BOSC_AME_MMA_OCT_WIDEN<0b0, 0b0, 0b0, 0b100, "momau.mm">;
  def BOSC_AME_MOMAU_HB_MM : BOSC_AME_MMA_OCT_WIDEN<0b0, 0b0, 0b0, 0b111, "momau.hb.mm">;
  // uint 4→32 with oct widening and saturation
  def BOSC_AME_MSOMAU_MM    : BOSC_AME_MMA_OCT_WIDEN<0b0, 0b1, 0b0, 0b100, "msomau.mm">;
  def BOSC_AME_MSOMAU_HB_MM : BOSC_AME_MMA_OCT_WIDEN<0b0, 0b1, 0b0, 0b111, "msomau.hb.mm">;
  // int 4→32 with oct widening
  def BOSC_AME_MOMA_MM    : BOSC_AME_MMA_OCT_WIDEN<0b0, 0b0, 0b1, 0b100, "moma.mm">;
  def BOSC_AME_MOMA_HB_MM : BOSC_AME_MMA_OCT_WIDEN<0b0, 0b0, 0b1, 0b111, "moma.hb.mm">;
  // int 4→32 with oct widening and saturation
  def BOSC_AME_MSOMA_MM    : BOSC_AME_MMA_OCT_WIDEN<0b0, 0b1, 0b1, 0b100, "msoma.mm">;
  def BOSC_AME_MSOMA_HB_MM : BOSC_AME_MMA_OCT_WIDEN<0b0, 0b1, 0b1, 0b111, "msoma.hb.mm">;
}

//===----------------------------------------------------------------------===//
// BOSC AME Element-Wise Instructions
//===----------------------------------------------------------------------===//
// Matrix element-wise add/sub/multiply instructions.
class BOSC_AME_MADD_NO_WIDEN<bit fp_val, bit sa_val, bit sn_val,
                              bits<3> eew_val, string opcodestr>
    : RVInstBOSCAME32<(outs AccReg:$md),
                      (ins AccReg:$ms1, AccReg:$ms2),
                      opcodestr, "$md, $ms1, $ms2"> {
  let funct6 = 0b001000;
  let fp     = fp_val;
  let sa     = sa_val;
  let sn     = sn_val;
  let eew    = eew_val;
  let ma     = 0b0;
  let opcode = 0b1110111;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  // uint 4 8 16 32 64 , output no-widen
  def BOSC_AME_MADDU_MM    : BOSC_AME_MADD_NO_WIDEN<0b0, 0b0, 0b0, 0b100, "maddu.mm">;
  def BOSC_AME_MADDU_HB_MM : BOSC_AME_MADD_NO_WIDEN<0b0, 0b0, 0b0, 0b111, "maddu.hb.mm">;
  def BOSC_AME_MADDU_B_MM  : BOSC_AME_MADD_NO_WIDEN<0b0, 0b0, 0b0, 0b000, "maddu.b.mm">;
  def BOSC_AME_MADDU_H_MM  : BOSC_AME_MADD_NO_WIDEN<0b0, 0b0, 0b0, 0b001, "maddu.h.mm">;
  def BOSC_AME_MADDU_W_MM  : BOSC_AME_MADD_NO_WIDEN<0b0, 0b0, 0b0, 0b010, "maddu.w.mm">;
  def BOSC_AME_MADDU_DW_MM : BOSC_AME_MADD_NO_WIDEN<0b0, 0b0, 0b0, 0b011, "maddu.dw.mm">;
  // uint 4 8 16 32 64 , output no-widen and saturated
  def BOSC_AME_MSADDU_MM    : BOSC_AME_MADD_NO_WIDEN<0b0, 0b1, 0b0, 0b100, "msaddu.mm">;
  def BOSC_AME_MSADDU_HB_MM : BOSC_AME_MADD_NO_WIDEN<0b0, 0b1, 0b0, 0b111, "msaddu.hb.mm">;
  def BOSC_AME_MSADDU_B_MM  : BOSC_AME_MADD_NO_WIDEN<0b0, 0b1, 0b0, 0b000, "msaddu.b.mm">;
  def BOSC_AME_MSADDU_H_MM  : BOSC_AME_MADD_NO_WIDEN<0b0, 0b1, 0b0, 0b001, "msaddu.h.mm">;
  def BOSC_AME_MSADDU_W_MM  : BOSC_AME_MADD_NO_WIDEN<0b0, 0b1, 0b0, 0b010, "msaddu.w.mm">;
  def BOSC_AME_MSADDU_DW_MM : BOSC_AME_MADD_NO_WIDEN<0b0, 0b1, 0b0, 0b011, "msaddu.dw.mm">;
  // int 4 8 16 32 64 , output no-widen
  def BOSC_AME_MADD_MM    : BOSC_AME_MADD_NO_WIDEN<0b0, 0b0, 0b1, 0b100, "madd.mm">;
  def BOSC_AME_MADD_HB_MM : BOSC_AME_MADD_NO_WIDEN<0b0, 0b0, 0b1, 0b111, "madd.hb.mm">;
  def BOSC_AME_MADD_B_MM  : BOSC_AME_MADD_NO_WIDEN<0b0, 0b0, 0b1, 0b000, "madd.b.mm">;
  def BOSC_AME_MADD_H_MM  : BOSC_AME_MADD_NO_WIDEN<0b0, 0b0, 0b1, 0b001, "madd.h.mm">;
  def BOSC_AME_MADD_W_MM  : BOSC_AME_MADD_NO_WIDEN<0b0, 0b0, 0b1, 0b010, "madd.w.mm">;
  def BOSC_AME_MADD_DW_MM : BOSC_AME_MADD_NO_WIDEN<0b0, 0b0, 0b1, 0b011, "madd.dw.mm">;
  // int 4 8 16 32 64 , output no-widen and saturated
  def BOSC_AME_MSADD_MM    : BOSC_AME_MADD_NO_WIDEN<0b0, 0b1, 0b1, 0b100, "msadd.mm">;
  def BOSC_AME_MSADD_HB_MM : BOSC_AME_MADD_NO_WIDEN<0b0, 0b1, 0b1, 0b111, "msadd.hb.mm">;
  def BOSC_AME_MSADD_B_MM  : BOSC_AME_MADD_NO_WIDEN<0b0, 0b1, 0b1, 0b000, "msadd.b.mm">;
  def BOSC_AME_MSADD_H_MM  : BOSC_AME_MADD_NO_WIDEN<0b0, 0b1, 0b1, 0b001, "msadd.h.mm">;
  def BOSC_AME_MSADD_W_MM  : BOSC_AME_MADD_NO_WIDEN<0b0, 0b1, 0b1, 0b010, "msadd.w.mm">;
  def BOSC_AME_MSADD_DW_MM : BOSC_AME_MADD_NO_WIDEN<0b0, 0b1, 0b1, 0b011, "msadd.dw.mm">;
  // float point 8 16 32 64 , output no-widen
  def BOSC_AME_MFADD_MM    : BOSC_AME_MADD_NO_WIDEN<0b1, 0b0, 0b1, 0b100, "mfadd.mm">;
  def BOSC_AME_MFADD_CF_MM : BOSC_AME_MADD_NO_WIDEN<0b1, 0b0, 0b1, 0b000, "mfadd.cf.mm">;
  def BOSC_AME_MFADD_HF_MM : BOSC_AME_MADD_NO_WIDEN<0b1, 0b0, 0b1, 0b001, "mfadd.hf.mm">;
  def BOSC_AME_MFADD_F_MM  : BOSC_AME_MADD_NO_WIDEN<0b1, 0b0, 0b1, 0b010, "mfadd.f.mm">;
  def BOSC_AME_MFADD_D_MM  : BOSC_AME_MADD_NO_WIDEN<0b1, 0b0, 0b1, 0b011, "mfadd.d.mm">;
}

class BOSC_AME_MADD_DOUBLE_WIDEN<bit fp_val, bit sa_val, bit sn_val,
                                  bits<3> eew_val, string opcodestr>
    : RVInstBOSCAME32<(outs AccReg:$md),
                      (ins AccReg:$ms1, AccReg:$ms2),
                      opcodestr, "$md, $ms1, $ms2"> {
  let funct6 = 0b001001;
  let fp     = fp_val;
  let sa     = sa_val;
  let sn     = sn_val;
  let eew    = eew_val;
  let ma     = 0b0;
  let opcode = 0b1110111;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  // uint 4 8 16 32 , output double-widen
  def BOSC_AME_MWADDU_MM    : BOSC_AME_MADD_DOUBLE_WIDEN<0b0, 0, 0, 0b100, "mwaddu.mm">;
  def BOSC_AME_MWADDU_HB_MM : BOSC_AME_MADD_DOUBLE_WIDEN<0b0, 0, 0, 0b111, "mwaddu.hb.mm">;
  def BOSC_AME_MWADDU_B_MM  : BOSC_AME_MADD_DOUBLE_WIDEN<0b0, 0, 0, 0b000, "mwaddu.b.mm">;
  def BOSC_AME_MWADDU_H_MM  : BOSC_AME_MADD_DOUBLE_WIDEN<0b0, 0, 0, 0b001, "mwaddu.h.mm">;
  def BOSC_AME_MWADDU_W_MM  : BOSC_AME_MADD_DOUBLE_WIDEN<0b0, 0, 0, 0b010, "mwaddu.w.mm">;
  // int 4 8 16 32 , output double-widen
  def BOSC_AME_MWADD_MM    : BOSC_AME_MADD_DOUBLE_WIDEN<0b0, 0, 0b1, 0b100, "mwadd.mm">;
  def BOSC_AME_MWADD_HB_MM : BOSC_AME_MADD_DOUBLE_WIDEN<0b0, 0, 0b1, 0b111, "mwadd.hb.mm">;
  def BOSC_AME_MWADD_B_MM  : BOSC_AME_MADD_DOUBLE_WIDEN<0b0, 0, 0b1, 0b000, "mwadd.b.mm">;
  def BOSC_AME_MWADD_H_MM  : BOSC_AME_MADD_DOUBLE_WIDEN<0b0, 0, 0b1, 0b001, "mwadd.h.mm">;
  def BOSC_AME_MWADD_W_MM  : BOSC_AME_MADD_DOUBLE_WIDEN<0b0, 0, 0b1, 0b010, "mwadd.w.mm">;
  // float point 8 16 32 , output double-widen
  def BOSC_AME_MFWADD_MM    : BOSC_AME_MADD_DOUBLE_WIDEN<0b1, 0b0, 0b1, 0b100, "mfwadd.mm">;
  def BOSC_AME_MFWADD_CF_MM : BOSC_AME_MADD_DOUBLE_WIDEN<0b1, 0b0, 0b1, 0b000, "mfwadd.cf.mm">;
  def BOSC_AME_MFWADD_HF_MM : BOSC_AME_MADD_DOUBLE_WIDEN<0b1, 0b0, 0b1, 0b001, "mfwadd.hf.mm">;
  def BOSC_AME_MFWADD_F_MM  : BOSC_AME_MADD_DOUBLE_WIDEN<0b1, 0b0, 0b1, 0b010, "mfwadd.f.mm">;
}

class BOSC_AME_MSUB_NO_WIDEN<bit fp_val, bit sa_val, bit sn_val,
                                  bits<3> eew_val, string opcodestr>
    : RVInstBOSCAME32<(outs AccReg:$md),
                      (ins AccReg:$ms1, AccReg:$ms2),
                      opcodestr, "$md, $ms1, $ms2"> {
  let funct6 = 0b001010;
  let fp     = fp_val;
  let sa     = sa_val;
  let sn     = sn_val;
  let eew    = eew_val;
  let ma     = 0b0;
  let opcode = 0b1110111;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {

  def BOSC_AME_MSUBU_MM    : BOSC_AME_MSUB_NO_WIDEN<0, 0, 0, 0b100, "msubu.mm">;
  def BOSC_AME_MSUBU_HB_MM : BOSC_AME_MSUB_NO_WIDEN<0, 0, 0, 0b111, "msubu.hb.mm">;
  def BOSC_AME_MSUBU_B_MM  : BOSC_AME_MSUB_NO_WIDEN<0, 0, 0, 0b000, "msubu.b.mm">;
  def BOSC_AME_MSUBU_H_MM  : BOSC_AME_MSUB_NO_WIDEN<0, 0, 0, 0b001, "msubu.h.mm">;
  def BOSC_AME_MSUBU_W_MM  : BOSC_AME_MSUB_NO_WIDEN<0, 0, 0, 0b010, "msubu.w.mm">;
  def BOSC_AME_MSUBU_DW_MM : BOSC_AME_MSUB_NO_WIDEN<0, 0, 0, 0b011, "msubu.dw.mm">;

  def BOSC_AME_MSSUBU_MM    : BOSC_AME_MSUB_NO_WIDEN<0, 1, 0, 0b100, "mssubu.mm">;
  def BOSC_AME_MSSUBU_HB_MM : BOSC_AME_MSUB_NO_WIDEN<0, 1, 0, 0b111, "mssubu.hb.mm">;
  def BOSC_AME_MSSUBU_B_MM  : BOSC_AME_MSUB_NO_WIDEN<0, 1, 0, 0b000, "mssubu.b.mm">;
  def BOSC_AME_MSSUBU_H_MM  : BOSC_AME_MSUB_NO_WIDEN<0, 1, 0, 0b001, "mssubu.h.mm">;
  def BOSC_AME_MSSUBU_W_MM  : BOSC_AME_MSUB_NO_WIDEN<0, 1, 0, 0b010, "mssubu.w.mm">;
  def BOSC_AME_MSSUBU_DW_MM : BOSC_AME_MSUB_NO_WIDEN<0, 1, 0, 0b011, "mssubu.dw.mm">;

  def BOSC_AME_MSUB_MM    : BOSC_AME_MSUB_NO_WIDEN<0, 0, 1, 0b100, "msub.mm">;
  def BOSC_AME_MSUB_HB_MM : BOSC_AME_MSUB_NO_WIDEN<0, 0, 1, 0b111, "msub.hb.mm">;
  def BOSC_AME_MSUB_B_MM  : BOSC_AME_MSUB_NO_WIDEN<0, 0, 1, 0b000, "msub.b.mm">;
  def BOSC_AME_MSUB_H_MM  : BOSC_AME_MSUB_NO_WIDEN<0, 0, 1, 0b001, "msub.h.mm">;
  def BOSC_AME_MSUB_W_MM  : BOSC_AME_MSUB_NO_WIDEN<0, 0, 1, 0b010, "msub.w.mm">;
  def BOSC_AME_MSUB_DW_MM : BOSC_AME_MSUB_NO_WIDEN<0, 0, 1, 0b011, "msub.dw.mm">;

  def BOSC_AME_MSSUB_MM    : BOSC_AME_MSUB_NO_WIDEN<0, 1, 1, 0b100, "mssub.mm">;
  def BOSC_AME_MSSUB_HB_MM : BOSC_AME_MSUB_NO_WIDEN<0, 1, 1, 0b111, "mssub.hb.mm">;
  def BOSC_AME_MSSUB_B_MM  : BOSC_AME_MSUB_NO_WIDEN<0, 1, 1, 0b000, "mssub.b.mm">;
  def BOSC_AME_MSSUB_H_MM  : BOSC_AME_MSUB_NO_WIDEN<0, 1, 1, 0b001, "mssub.h.mm">;
  def BOSC_AME_MSSUB_W_MM  : BOSC_AME_MSUB_NO_WIDEN<0, 1, 1, 0b010, "mssub.w.mm">;
  def BOSC_AME_MSSUB_DW_MM : BOSC_AME_MSUB_NO_WIDEN<0, 1, 1, 0b011, "mssub.dw.mm">;

  def BOSC_AME_MFSUB_MM    : BOSC_AME_MSUB_NO_WIDEN<1, 0, 1, 0b100, "mfsub.mm">;
  def BOSC_AME_MFSUB_CF_MM : BOSC_AME_MSUB_NO_WIDEN<1, 0, 1, 0b000, "mfsub.cf.mm">;
  def BOSC_AME_MFSUB_HF_MM : BOSC_AME_MSUB_NO_WIDEN<1, 0, 1, 0b001, "mfsub.hf.mm">;
  def BOSC_AME_MFSUB_F_MM  : BOSC_AME_MSUB_NO_WIDEN<1, 0, 1, 0b010, "mfsub.f.mm">;
  def BOSC_AME_MFSUB_D_MM  : BOSC_AME_MSUB_NO_WIDEN<1, 0, 1, 0b011, "mfsub.d.mm">;
}

class BOSC_AME_MSUB_DOUBLE_WIDEN<bit fp_val, bit sa_val, bit sn_val,
                                bits<3> eew_val, string opcodestr>
    : RVInstBOSCAME32<(outs AccReg:$md),
                      (ins AccReg:$ms1, AccReg:$ms2),
                      opcodestr, "$md, $ms1, $ms2"> {
  let funct6 = 0b001011;
  let fp     = fp_val;
  let sa     = sa_val;
  let sn     = sn_val;
  let eew    = eew_val;
  let ma     = 0b0;
  let opcode = 0b1110111;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {

  def BOSC_AME_MWSUBU_MM    : BOSC_AME_MSUB_DOUBLE_WIDEN<0, 0, 0, 0b100, "mwsubu.mm">;
  def BOSC_AME_MWSUBU_HB_MM : BOSC_AME_MSUB_DOUBLE_WIDEN<0, 0, 0, 0b111, "mwsubu.hb.mm">;
  def BOSC_AME_MWSUBU_B_MM  : BOSC_AME_MSUB_DOUBLE_WIDEN<0, 0, 0, 0b000, "mwsubu.b.mm">;
  def BOSC_AME_MWSUBU_H_MM  : BOSC_AME_MSUB_DOUBLE_WIDEN<0, 0, 0, 0b001, "mwsubu.h.mm">;
  def BOSC_AME_MWSUBU_W_MM  : BOSC_AME_MSUB_DOUBLE_WIDEN<0, 0, 0, 0b010, "mwsubu.w.mm">;

  def BOSC_AME_MWSUB_MM    : BOSC_AME_MSUB_DOUBLE_WIDEN<0, 0, 1, 0b100, "mwsub.mm">;
  def BOSC_AME_MWSUB_HB_MM : BOSC_AME_MSUB_DOUBLE_WIDEN<0, 0, 1, 0b111, "mwsub.hb.mm">;
  def BOSC_AME_MWSUB_B_MM  : BOSC_AME_MSUB_DOUBLE_WIDEN<0, 0, 1, 0b000, "mwsub.b.mm">;
  def BOSC_AME_MWSUB_H_MM  : BOSC_AME_MSUB_DOUBLE_WIDEN<0, 0, 1, 0b001, "mwsub.h.mm">;
  def BOSC_AME_MWSUB_W_MM  : BOSC_AME_MSUB_DOUBLE_WIDEN<0, 0, 1, 0b010, "mwsub.w.mm">;

  def BOSC_AME_MFWSUB_MM    : BOSC_AME_MSUB_DOUBLE_WIDEN<1, 0, 1, 0b100, "mfwsub.mm">;
  def BOSC_AME_MFWSUB_CF_MM : BOSC_AME_MSUB_DOUBLE_WIDEN<1, 0, 1, 0b000, "mfwsub.cf.mm">;
  def BOSC_AME_MFWSUB_HF_MM : BOSC_AME_MSUB_DOUBLE_WIDEN<1, 0, 1, 0b001, "mfwsub.hf.mm">;
  def BOSC_AME_MFWSUB_F_MM  : BOSC_AME_MSUB_DOUBLE_WIDEN<1, 0, 1, 0b010, "mfwsub.f.mm">;
}

class BOSC_AME_MMINMAX_NO_WIDEN<bit fp_val, bit sa_val, bit sn_val,
                                bits<3> eew_val, string opcodestr>
    : RVInstBOSCAME32<(outs AccReg:$md),
                      (ins AccReg:$ms1, AccReg:$ms2),
                      opcodestr, "$md, $ms1, $ms2"> {
  let funct6 = 0b001100;
  let fp     = fp_val;
  let sa     = sa_val;
  let sn     = sn_val;
  let eew    = eew_val;
  let ma     = 0b0;
  let opcode = 0b1110111;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {

  def BOSC_AME_MMINU_MM    : BOSC_AME_MMINMAX_NO_WIDEN<0, 0, 0, 0b100, "mminu.mm">;
  def BOSC_AME_MMINU_HB_MM : BOSC_AME_MMINMAX_NO_WIDEN<0, 0, 0, 0b111, "mminu.hb.mm">;
  def BOSC_AME_MMINU_B_MM  : BOSC_AME_MMINMAX_NO_WIDEN<0, 0, 0, 0b000, "mminu.b.mm">;
  def BOSC_AME_MMINU_H_MM  : BOSC_AME_MMINMAX_NO_WIDEN<0, 0, 0, 0b001, "mminu.h.mm">;
  def BOSC_AME_MMINU_W_MM  : BOSC_AME_MMINMAX_NO_WIDEN<0, 0, 0, 0b010, "mminu.w.mm">;
  def BOSC_AME_MMINU_DW_MM : BOSC_AME_MMINMAX_NO_WIDEN<0, 0, 0, 0b011, "mminu.dw.mm">;

  def BOSC_AME_MMAXU_MM    : BOSC_AME_MMINMAX_NO_WIDEN<0, 1, 0, 0b100, "mmaxu.mm">;
  def BOSC_AME_MMAXU_HB_MM : BOSC_AME_MMINMAX_NO_WIDEN<0, 1, 0, 0b111, "mmaxu.hb.mm">;
  def BOSC_AME_MMAXU_B_MM  : BOSC_AME_MMINMAX_NO_WIDEN<0, 1, 0, 0b000, "mmaxu.b.mm">;
  def BOSC_AME_MMAXU_H_MM  : BOSC_AME_MMINMAX_NO_WIDEN<0, 1, 0, 0b001, "mmaxu.h.mm">;
  def BOSC_AME_MMAXU_W_MM  : BOSC_AME_MMINMAX_NO_WIDEN<0, 1, 0, 0b010, "mmaxu.w.mm">;
  def BOSC_AME_MMAXU_DW_MM : BOSC_AME_MMINMAX_NO_WIDEN<0, 1, 0, 0b011, "mmaxu.dw.mm">;

  def BOSC_AME_MMIN_MM    : BOSC_AME_MMINMAX_NO_WIDEN<0, 0, 1, 0b100, "mmin.mm">;
  def BOSC_AME_MMIN_HB_MM : BOSC_AME_MMINMAX_NO_WIDEN<0, 0, 1, 0b111, "mmin.hb.mm">;
  def BOSC_AME_MMIN_B_MM  : BOSC_AME_MMINMAX_NO_WIDEN<0, 0, 1, 0b000, "mmin.b.mm">;
  def BOSC_AME_MMIN_H_MM  : BOSC_AME_MMINMAX_NO_WIDEN<0, 0, 1, 0b001, "mmin.h.mm">;
  def BOSC_AME_MMIN_W_MM  : BOSC_AME_MMINMAX_NO_WIDEN<0, 0, 1, 0b010, "mmin.w.mm">;
  def BOSC_AME_MMIN_DW_MM : BOSC_AME_MMINMAX_NO_WIDEN<0, 0, 1, 0b011, "mmin.dw.mm">;

  def BOSC_AME_MMAX_MM    : BOSC_AME_MMINMAX_NO_WIDEN<0, 1, 1, 0b100, "mmax.mm">;
  def BOSC_AME_MMAX_HB_MM : BOSC_AME_MMINMAX_NO_WIDEN<0, 1, 1, 0b111, "mmax.hb.mm">;
  def BOSC_AME_MMAX_B_MM  : BOSC_AME_MMINMAX_NO_WIDEN<0, 1, 1, 0b000, "mmax.b.mm">;
  def BOSC_AME_MMAX_H_MM  : BOSC_AME_MMINMAX_NO_WIDEN<0, 1, 1, 0b001, "mmax.h.mm">;
  def BOSC_AME_MMAX_W_MM  : BOSC_AME_MMINMAX_NO_WIDEN<0, 1, 1, 0b010, "mmax.w.mm">;
  def BOSC_AME_MMAX_DW_MM : BOSC_AME_MMINMAX_NO_WIDEN<0, 1, 1, 0b011, "mmax.dw.mm">;

  def BOSC_AME_MFMIN_MM    : BOSC_AME_MMINMAX_NO_WIDEN<1, 0, 1, 0b100, "mfmin.mm">;
  def BOSC_AME_MFMIN_CF_MM : BOSC_AME_MMINMAX_NO_WIDEN<1, 0, 1, 0b000, "mfmin.cf.mm">;
  def BOSC_AME_MFMIN_HF_MM : BOSC_AME_MMINMAX_NO_WIDEN<1, 0, 1, 0b001, "mfmin.hf.mm">;
  def BOSC_AME_MFMIN_F_MM  : BOSC_AME_MMINMAX_NO_WIDEN<1, 0, 1, 0b010, "mfmin.f.mm">;
  def BOSC_AME_MFMIN_D_MM  : BOSC_AME_MMINMAX_NO_WIDEN<1, 0, 1, 0b011, "mfmin.d.mm">;

  def BOSC_AME_MFMAX_MM    : BOSC_AME_MMINMAX_NO_WIDEN<1, 1, 1, 0b100, "mfmax.mm">;
  def BOSC_AME_MFMAX_CF_MM : BOSC_AME_MMINMAX_NO_WIDEN<1, 1, 1, 0b000, "mfmax.cf.mm">;
  def BOSC_AME_MFMAX_HF_MM : BOSC_AME_MMINMAX_NO_WIDEN<1, 1, 1, 0b001, "mfmax.hf.mm">;
  def BOSC_AME_MFMAX_F_MM  : BOSC_AME_MMINMAX_NO_WIDEN<1, 1, 1, 0b010, "mfmax.f.mm">;
  def BOSC_AME_MFMAX_D_MM  : BOSC_AME_MMINMAX_NO_WIDEN<1, 1, 1, 0b011, "mfmax.d.mm">;
}

class BOSC_AME_MMUL_NO_WIDEN<bit fp_val, bit sa_val, bit sn_val,
                                  bits<3> eew_val, string opcodestr>
    : RVInstBOSCAME32<(outs AccReg:$md),
                      (ins AccReg:$ms1, AccReg:$ms2),
                      opcodestr, "$md, $ms1, $ms2"> {
  let funct6 = 0b001101;
  let fp     = fp_val;
  let sa     = sa_val;
  let sn     = sn_val;
  let eew    = eew_val;
  let ma     = 0b0;
  let opcode = 0b1110111;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {

  def BOSC_AME_MSMULU_MM    : BOSC_AME_MMUL_NO_WIDEN<0, 1, 0, 0b100, "msmulu.mm">;
  def BOSC_AME_MSMULU_HB_MM : BOSC_AME_MMUL_NO_WIDEN<0, 1, 0, 0b111, "msmulu.hb.mm">;
  def BOSC_AME_MSMULU_B_MM  : BOSC_AME_MMUL_NO_WIDEN<0, 1, 0, 0b000, "msmulu.b.mm">;
  def BOSC_AME_MSMULU_H_MM  : BOSC_AME_MMUL_NO_WIDEN<0, 1, 0, 0b001, "msmulu.h.mm">;
  def BOSC_AME_MSMULU_W_MM  : BOSC_AME_MMUL_NO_WIDEN<0, 1, 0, 0b010, "msmulu.w.mm">;
  def BOSC_AME_MSMULU_DW_MM : BOSC_AME_MMUL_NO_WIDEN<0, 1, 0, 0b011, "msmulu.dw.mm">;

  def BOSC_AME_MMUL_MM    : BOSC_AME_MMUL_NO_WIDEN<0, 0, 1, 0b100, "mmul.mm">;
  def BOSC_AME_MMUL_HB_MM : BOSC_AME_MMUL_NO_WIDEN<0, 0, 1, 0b111, "mmul.hb.mm">;
  def BOSC_AME_MMUL_B_MM  : BOSC_AME_MMUL_NO_WIDEN<0, 0, 1, 0b000, "mmul.b.mm">;
  def BOSC_AME_MMUL_H_MM  : BOSC_AME_MMUL_NO_WIDEN<0, 0, 1, 0b001, "mmul.h.mm">;
  def BOSC_AME_MMUL_W_MM  : BOSC_AME_MMUL_NO_WIDEN<0, 0, 1, 0b010, "mmul.w.mm">;
  def BOSC_AME_MMUL_DW_MM : BOSC_AME_MMUL_NO_WIDEN<0, 0, 1, 0b011, "mmul.dw.mm">;

  def BOSC_AME_MSMUL_MM    : BOSC_AME_MMUL_NO_WIDEN<0, 1, 1, 0b100, "msmul.mm">;
  def BOSC_AME_MSMUL_HB_MM : BOSC_AME_MMUL_NO_WIDEN<0, 1, 1, 0b111, "msmul.hb.mm">;
  def BOSC_AME_MSMUL_B_MM  : BOSC_AME_MMUL_NO_WIDEN<0, 1, 1, 0b000, "msmul.b.mm">;
  def BOSC_AME_MSMUL_H_MM  : BOSC_AME_MMUL_NO_WIDEN<0, 1, 1, 0b001, "msmul.h.mm">;
  def BOSC_AME_MSMUL_W_MM  : BOSC_AME_MMUL_NO_WIDEN<0, 1, 1, 0b010, "msmul.w.mm">;
  def BOSC_AME_MSMUL_DW_MM : BOSC_AME_MMUL_NO_WIDEN<0, 1, 1, 0b011, "msmul.dw.mm">;

  def BOSC_AME_MFMUL_MM    : BOSC_AME_MMUL_NO_WIDEN<1, 0, 1, 0b100, "mfmul.mm">;
  def BOSC_AME_MFMUL_CF_MM : BOSC_AME_MMUL_NO_WIDEN<1, 0, 1, 0b000, "mfmul.cf.mm">;
  def BOSC_AME_MFMUL_HF_MM : BOSC_AME_MMUL_NO_WIDEN<1, 0, 1, 0b001, "mfmul.hf.mm">;
  def BOSC_AME_MFMUL_F_MM  : BOSC_AME_MMUL_NO_WIDEN<1, 0, 1, 0b010, "mfmul.f.mm">;
  def BOSC_AME_MFMUL_D_MM  : BOSC_AME_MMUL_NO_WIDEN<1, 0, 1, 0b011, "mfmul.d.mm">;
}

class BOSC_AME_MMULH_DIV_NO_WIDEN<bit fp_val, bit sa_val, bit sn_val,
                                  bits<3> eew_val, string opcodestr>
    : RVInstBOSCAME32<(outs AccReg:$md),
                      (ins AccReg:$ms1, AccReg:$ms2),
                      opcodestr, "$md, $ms1, $ms2"> {
  let funct6 = 0b001110;
  let fp     = fp_val;
  let sa     = sa_val;
  let sn     = sn_val;
  let eew    = eew_val;
  let ma     = 0b0;
  let opcode = 0b1110111;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {

  def BOSC_AME_MMULHU_MM    : BOSC_AME_MMULH_DIV_NO_WIDEN<0, 0, 0, 0b100, "mmulhu.mm">;
  def BOSC_AME_MMULHU_HB_MM : BOSC_AME_MMULH_DIV_NO_WIDEN<0, 0, 0, 0b111, "mmulhu.hb.mm">;
  def BOSC_AME_MMULHU_B_MM  : BOSC_AME_MMULH_DIV_NO_WIDEN<0, 0, 0, 0b000, "mmulhu.b.mm">;
  def BOSC_AME_MMULHU_H_MM  : BOSC_AME_MMULH_DIV_NO_WIDEN<0, 0, 0, 0b001, "mmulhu.h.mm">;
  def BOSC_AME_MMULHU_W_MM  : BOSC_AME_MMULH_DIV_NO_WIDEN<0, 0, 0, 0b010, "mmulhu.w.mm">;
  def BOSC_AME_MMULHU_DW_MM : BOSC_AME_MMULH_DIV_NO_WIDEN<0, 0, 0, 0b011, "mmulhu.dw.mm">;

  def BOSC_AME_MMULH_MM    : BOSC_AME_MMULH_DIV_NO_WIDEN<0, 0, 1, 0b100, "mmulh.mm">;
  def BOSC_AME_MMULH_HB_MM : BOSC_AME_MMULH_DIV_NO_WIDEN<0, 0, 1, 0b111, "mmulh.hb.mm">;
  def BOSC_AME_MMULH_B_MM  : BOSC_AME_MMULH_DIV_NO_WIDEN<0, 0, 1, 0b000, "mmulh.b.mm">;
  def BOSC_AME_MMULH_H_MM  : BOSC_AME_MMULH_DIV_NO_WIDEN<0, 0, 1, 0b001, "mmulh.h.mm">;
  def BOSC_AME_MMULH_W_MM  : BOSC_AME_MMULH_DIV_NO_WIDEN<0, 0, 1, 0b010, "mmulh.w.mm">;
  def BOSC_AME_MMULH_DW_MM : BOSC_AME_MMULH_DIV_NO_WIDEN<0, 0, 1, 0b011, "mmulh.dw.mm">;

  def BOSC_AME_MMULHSU_MM    : BOSC_AME_MMULH_DIV_NO_WIDEN<0, 1, 0, 0b100, "mmulhsu.mm">;
  def BOSC_AME_MMULHSU_HB_MM : BOSC_AME_MMULH_DIV_NO_WIDEN<0, 1, 0, 0b111, "mmulhsu.hb.mm">;
  def BOSC_AME_MMULHSU_B_MM  : BOSC_AME_MMULH_DIV_NO_WIDEN<0, 1, 0, 0b000, "mmulhsu.b.mm">;
  def BOSC_AME_MMULHSU_H_MM  : BOSC_AME_MMULH_DIV_NO_WIDEN<0, 1, 0, 0b001, "mmulhsu.h.mm">;
  def BOSC_AME_MMULHSU_W_MM  : BOSC_AME_MMULH_DIV_NO_WIDEN<0, 1, 0, 0b010, "mmulhsu.w.mm">;
  def BOSC_AME_MMULHSU_DW_MM : BOSC_AME_MMULH_DIV_NO_WIDEN<0, 1, 0, 0b011, "mmulhsu.dw.mm">;

  def BOSC_AME_MSMULSU_MM    : BOSC_AME_MMULH_DIV_NO_WIDEN<0, 1, 1, 0b100, "msmulsu.mm">;
  def BOSC_AME_MSMULSU_HB_MM : BOSC_AME_MMULH_DIV_NO_WIDEN<0, 1, 1, 0b111, "msmulsu.hb.mm">;
  def BOSC_AME_MSMULSU_B_MM  : BOSC_AME_MMULH_DIV_NO_WIDEN<0, 1, 1, 0b000, "msmulsu.b.mm">;
  def BOSC_AME_MSMULSU_H_MM  : BOSC_AME_MMULH_DIV_NO_WIDEN<0, 1, 1, 0b001, "msmulsu.h.mm">;
  def BOSC_AME_MSMULSU_W_MM  : BOSC_AME_MMULH_DIV_NO_WIDEN<0, 1, 1, 0b010, "msmulsu.w.mm">;
  def BOSC_AME_MSMULSU_DW_MM : BOSC_AME_MMULH_DIV_NO_WIDEN<0, 1, 1, 0b011, "msmulsu.dw.mm">;

  def BOSC_AME_MFDIV_MM    : BOSC_AME_MMULH_DIV_NO_WIDEN<1, 0, 1, 0b100, "mfdiv.mm">;
  def BOSC_AME_MFDIV_CF_MM : BOSC_AME_MMULH_DIV_NO_WIDEN<1, 0, 1, 0b000, "mfdiv.cf.mm">;
  def BOSC_AME_MFDIV_HF_MM : BOSC_AME_MMULH_DIV_NO_WIDEN<1, 0, 1, 0b001, "mfdiv.hf.mm">;
  def BOSC_AME_MFDIV_F_MM  : BOSC_AME_MMULH_DIV_NO_WIDEN<1, 0, 1, 0b010, "mfdiv.f.mm">;
  def BOSC_AME_MFDIV_D_MM  : BOSC_AME_MMULH_DIV_NO_WIDEN<1, 0, 1, 0b011, "mfdiv.d.mm">;
}

class BOSC_AME_MMUL_DOUBLE_WIDEN<bit fp_val, bit sa_val, bit sn_val,
                                  bits<3> eew_val, string opcodestr>
    : RVInstBOSCAME32<(outs AccReg:$md),
                      (ins AccReg:$ms1, AccReg:$ms2),
                      opcodestr, "$md, $ms1, $ms2"> {
  let funct6 = 0b001111;
  let fp     = fp_val;
  let sa     = sa_val;
  let sn     = sn_val;
  let eew    = eew_val;
  let ma     = 0b0;
  let opcode = 0b1110111;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {

  def BOSC_AME_MWMULU_MM    : BOSC_AME_MMUL_DOUBLE_WIDEN<0, 0, 0, 0b100, "mwmulu.mm">;
  def BOSC_AME_MWMULU_HB_MM : BOSC_AME_MMUL_DOUBLE_WIDEN<0, 0, 0, 0b111, "mwmulu.hb.mm">;
  def BOSC_AME_MWMULU_B_MM  : BOSC_AME_MMUL_DOUBLE_WIDEN<0, 0, 0, 0b000, "mwmulu.b.mm">;
  def BOSC_AME_MWMULU_H_MM  : BOSC_AME_MMUL_DOUBLE_WIDEN<0, 0, 0, 0b001, "mwmulu.h.mm">;
  def BOSC_AME_MWMULU_W_MM  : BOSC_AME_MMUL_DOUBLE_WIDEN<0, 0, 0, 0b010, "mwmulu.w.mm">;

  def BOSC_AME_MWMUL_MM    : BOSC_AME_MMUL_DOUBLE_WIDEN<0, 0, 1, 0b100, "mwmul.mm">;
  def BOSC_AME_MWMUL_HB_MM : BOSC_AME_MMUL_DOUBLE_WIDEN<0, 0, 1, 0b111, "mwmul.hb.mm">;
  def BOSC_AME_MWMUL_B_MM  : BOSC_AME_MMUL_DOUBLE_WIDEN<0, 0, 1, 0b000, "mwmul.b.mm">;
  def BOSC_AME_MWMUL_H_MM  : BOSC_AME_MMUL_DOUBLE_WIDEN<0, 0, 1, 0b001, "mwmul.h.mm">;
  def BOSC_AME_MWMUL_W_MM  : BOSC_AME_MMUL_DOUBLE_WIDEN<0, 0, 1, 0b010, "mwmul.w.mm">;

  def BOSC_AME_MWMULSU_MM    : BOSC_AME_MMUL_DOUBLE_WIDEN<0, 1, 1, 0b100, "mwmulsu.mm">;
  def BOSC_AME_MWMULSU_HB_MM : BOSC_AME_MMUL_DOUBLE_WIDEN<0, 1, 1, 0b111, "mwmulsu.hb.mm">;
  def BOSC_AME_MWMULSU_B_MM  : BOSC_AME_MMUL_DOUBLE_WIDEN<0, 1, 1, 0b000, "mwmulsu.b.mm">;
  def BOSC_AME_MWMULSU_H_MM  : BOSC_AME_MMUL_DOUBLE_WIDEN<0, 1, 1, 0b001, "mwmulsu.h.mm">;
  def BOSC_AME_MWMULSU_W_MM  : BOSC_AME_MMUL_DOUBLE_WIDEN<0, 1, 1, 0b010, "mwmulsu.w.mm">;

  def BOSC_AME_MFWMUL_MM    : BOSC_AME_MMUL_DOUBLE_WIDEN<1, 0, 1, 0b100, "mfwmul.mm">;
  def BOSC_AME_MFWMUL_CF_MM : BOSC_AME_MMUL_DOUBLE_WIDEN<1, 0, 1, 0b000, "mfwmul.cf.mm">;
  def BOSC_AME_MFWMUL_HF_MM : BOSC_AME_MMUL_DOUBLE_WIDEN<1, 0, 1, 0b001, "mfwmul.hf.mm">;
  def BOSC_AME_MFWMUL_F_MM  : BOSC_AME_MMUL_DOUBLE_WIDEN<1, 0, 1, 0b010, "mfwmul.f.mm">;
}

class BOSC_AME_LOGIC_NO_WIDEN<bit sa_val, bit sn_val, string opcodestr>
    : RVInstBOSCAME32<(outs AccReg:$md),
                      (ins AccReg:$ms1, AccReg:$ms2),
                      opcodestr, "$md, $ms1, $ms2"> {
  let funct6 = 0b010000;
  let fp     = 0b0;
  let sa     = sa_val;
  let sn     = sn_val;
  let eew    = 0b100;
  let ma     = 0b0;
  let opcode = 0b1110111;
}

class BOSC_AME_SQRT_NO_WIDEN<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAME32<(outs AccReg:$md),
                      (ins AccReg:$ms1),
                      opcodestr, "$md, $ms1"> {
  let funct6 = 0b010000;
  let fp     = 0b1;
  let sa     = 0b0;
  let ms2    = 0b0000;
  let sn     = 0b1;
  let eew    = eew_val;
  let ma     = 0b0;
  let opcode = 0b1110111;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {

  def BOSC_AME_MAND_MM : BOSC_AME_LOGIC_NO_WIDEN<0, 0, "mand.mm">;
  def BOSC_AME_MOR_MM  : BOSC_AME_LOGIC_NO_WIDEN<1, 0, "mor.mm">;
  def BOSC_AME_MXOR_MM : BOSC_AME_LOGIC_NO_WIDEN<1, 1, "mxor.mm">;

  def BOSC_AME_MFSQRT_MM    : BOSC_AME_SQRT_NO_WIDEN<0b100, "mfsqrt.mm">;
  def BOSC_AME_MFSQRT_CF_MM : BOSC_AME_SQRT_NO_WIDEN<0b000, "mfsqrt.cf.mm">;
  def BOSC_AME_MFSQRT_HF_MM : BOSC_AME_SQRT_NO_WIDEN<0b001, "mfsqrt.hf.mm">;
  def BOSC_AME_MFSQRT_F_MM  : BOSC_AME_SQRT_NO_WIDEN<0b010, "mfsqrt.f.mm">;
  def BOSC_AME_MFSQRT_D_MM  : BOSC_AME_SQRT_NO_WIDEN<0b011, "mfsqrt.d.mm">;
}

class BOSC_AME_SHIFT_NO_WIDEN<bit sa_val, bit sn_val,
                              bits<3> eew_val, string opcodestr>
    : RVInstBOSCAME32<(outs AccReg:$md),
                      (ins AccReg:$ms1, AccReg:$ms2),
                      opcodestr, "$md, $ms1, $ms2"> {
  let funct6 = 0b010001;
  let fp     = 0b0;
  let sa     = sa_val;
  let sn     = sn_val;
  let eew    = eew_val;
  let ma     = 0b0;
  let opcode = 0b1110111;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {

  def BOSC_AME_MSLL_MM    : BOSC_AME_SHIFT_NO_WIDEN<0, 0, 0b100, "msll.mm">;
  def BOSC_AME_MSLL_HB_MM : BOSC_AME_SHIFT_NO_WIDEN<0, 0, 0b111, "msll.hb.mm">;
  def BOSC_AME_MSLL_B_MM  : BOSC_AME_SHIFT_NO_WIDEN<0, 0, 0b000, "msll.b.mm">;
  def BOSC_AME_MSLL_H_MM  : BOSC_AME_SHIFT_NO_WIDEN<0, 0, 0b001, "msll.h.mm">;
  def BOSC_AME_MSLL_W_MM  : BOSC_AME_SHIFT_NO_WIDEN<0, 0, 0b010, "msll.w.mm">;
  def BOSC_AME_MSLL_DW_MM : BOSC_AME_SHIFT_NO_WIDEN<0, 0, 0b011, "msll.dw.mm">;

  def BOSC_AME_MSRL_MM    : BOSC_AME_SHIFT_NO_WIDEN<1, 0, 0b100, "msrl.mm">;
  def BOSC_AME_MSRL_HB_MM : BOSC_AME_SHIFT_NO_WIDEN<1, 0, 0b111, "msrl.hb.mm">;
  def BOSC_AME_MSRL_B_MM  : BOSC_AME_SHIFT_NO_WIDEN<1, 0, 0b000, "msrl.b.mm">;
  def BOSC_AME_MSRL_H_MM  : BOSC_AME_SHIFT_NO_WIDEN<1, 0, 0b001, "msrl.h.mm">;
  def BOSC_AME_MSRL_W_MM  : BOSC_AME_SHIFT_NO_WIDEN<1, 0, 0b010, "msrl.w.mm">;
  def BOSC_AME_MSRL_DW_MM : BOSC_AME_SHIFT_NO_WIDEN<1, 0, 0b011, "msrl.dw.mm">;

  def BOSC_AME_MSRA_MM    : BOSC_AME_SHIFT_NO_WIDEN<1, 1, 0b100, "msra.mm">;
  def BOSC_AME_MSRA_HB_MM : BOSC_AME_SHIFT_NO_WIDEN<1, 1, 0b111, "msra.hb.mm">;
  def BOSC_AME_MSRA_B_MM  : BOSC_AME_SHIFT_NO_WIDEN<1, 1, 0b000, "msra.b.mm">;
  def BOSC_AME_MSRA_H_MM  : BOSC_AME_SHIFT_NO_WIDEN<1, 1, 0b001, "msra.h.mm">;
  def BOSC_AME_MSRA_W_MM  : BOSC_AME_SHIFT_NO_WIDEN<1, 1, 0b010, "msra.w.mm">;
  def BOSC_AME_MSRA_DW_MM : BOSC_AME_SHIFT_NO_WIDEN<1, 1, 0b011, "msra.dw.mm">;
}

//===----------------------------------------------------------------------===//
// BOSC AME Configuration Instructions
//===----------------------------------------------------------------------===//
// Configuration instruction format (32-bit):
// Configuration Immediate Instructions:
// | 31:26  | 25 | 24:15 | 14:12  | 11:7 | 6:0    |
// | funct6 | im | imm10 | funct3 | rd   | OP-M32 |
// Configuration Instructions:
// | 31:26  | 25 | 24:20 | 19:15 | 14:12  | 11:7 | 6:0    |
// | funct6 | im | 00000 | rs1   | funct3 | rd   | OP-M32 |
//===----------------------------------------------------------------------===//

class RVInstBOSCAMEConfig32<bits<3> funct3_val, string opcodestr>
    : RVInst<(outs GPR:$rd), (ins GPR:$rs1),
               opcodestr, "$rd, $rs1", [], InstFormatOther> {
  bits<5> rs1;
  bits<5> rd;

  let Inst{6-0} = 0b1110111;
  let Inst{11-7} = rd;
  let Inst{14-12} = funct3_val;
  let Inst{19-15} = rs1;
  let Inst{24-20} = 0b00000;
  let Inst{25}    = 0b0;  // 0 for register source
  let Inst{31-26} = 0b000001;
}

class RVInstBOSCAMEConfigImm32<bits<3> funct3_val, string opcodestr>
    : RVInst<(outs GPR:$rd), (ins uimm32:$imm),
               opcodestr, "$rd, $imm", [], InstFormatOther> {
  bits<5> rd;
  bits<10> imm;    // 10-bit immediate

  let Inst{6-0} = 0b1110111;
  let Inst{11-7} = rd;
  let Inst{14-12} = funct3_val;
  let Inst{24-15} = imm;
  let Inst{25}    = 0b1;  // 1 for immediate source
  let Inst{31-26} = funct6_val;  // To be defined in subclasses
}

let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 0, mayStore = 0 in {
  // msettype - set the mtype CSR based on the value in a register
  def BOSC_AME_MSETTYPE : RVInstBOSCAMEConfig32<0b000000, 0b100, "msettype">;
  // msettilem - set tile M dimension from register
  def BOSC_AME_MSETTILEM : RVInstBOSCAMEConfig32<0b101, "msettilem">;
  // msettilen - set tile N dimension from register
  def BOSC_AME_MSETTILEN : RVInstBOSCAMEConfig32<0b100, "msettilen">;
  // msettilek - set tile K dimension from register
  def BOSC_AME_MSETTILEK : RVInstBOSCAMEConfig32<0b110, "msettilek">;
}

let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 0, mayStore = 0 in {
  // msettypei - set the mtype CSR based on their arguments
  def BOSC_AME_MSETTYPEI : RVInstBOSCAMEConfigImm32<0b000000, 0b100, "msettypei">;
  // msettypehi
  def BOSC_AME_MSETTYPEHI : RVInstBOSCAMEConfigImm32<0b000000, 0b101, "msettypehi">;
  // msettilemi - set tile M dimension from immediate
  def BOSC_AME_MSETTILEMI : RVInstBOSCAMEConfigImm32<0b101, "msettilemi">;
  // msettileni - set tile N dimension from immediate
  def BOSC_AME_MSETTILENI : RVInstBOSCAMEConfigImm32<0b100, "msettileni">;
  // msettileki - set tile K dimension from immediate
  def BOSC_AME_MSETTILEKI : RVInstBOSCAMEConfigImm32<0b110, "msettileki">;
}

//===----------------------------------------------------------------------===//
// BOSC AME Load/Store Instructions
//===----------------------------------------------------------------------===//
// Load/Store instruction format (32-bit):
// | 31:26  | 25 | 24:20 | 19:15 | 14:12 | 11 | 10:7    | 6:0    |
// | funct6 | ls | rs2   | rs1   | eew   | tr | ms3/td  | OP-M32 |
//
// ls: 0 for load (mlae/mlbe), 1 for store (msce)
// tr: whether the operand in register is transposed
// eew: 000=int8, 001=int16, 010=int32, 011=int64, 111=int4
//
// Data Flow Examples:
//   mlae32.m tr0, (a0), a1    # Load matrix A into TileReg
//   mlbe32.m tr1, (a0), a1    # Load matrix B into TileReg
//   mqma.b.mm acc0, tr0, tr1  # Compute: AccReg = AccReg + TileReg × TileReg
//   msce32.m acc0, (a0), a1   # Store AccReg to memory
//===----------------------------------------------------------------------===//

// Load into TileReg (for matrix A and B)
class RVInstBOSCAMELoadTile32<bits<6> funct6_val, bits<3> eew_val, bit tr_val, string opcodestr>
    : RVInst<(outs TileReg:$md), (ins GPR:$rs1, GPR:$rs2),
               opcodestr, "$md, $rs1, $rs2", [], InstFormatOther> {
  bits<5> rs2;
  bits<5> rs1;
  bits<4> md;  // TileReg index for load

  let Inst{6-0}   = 0b1110111;   // AME prefix
  let Inst{10-7}  = md;
  let Inst{11}    = tr_val;
  let Inst{14-12} = eew_val;
  let Inst{19-15} = rs1;
  let Inst{24-20} = rs2;
  let Inst{25}    = 0;           // ls = 0 for load
  let Inst{31-26} = funct6_val;  // funct6
}

// Load into AccReg (for accumulator C)
class RVInstBOSCAMELoadAcc32<bits<6> funct6_val, bits<3> eew_val, bit tr_val, string opcodestr>
    : RVInst<(outs AccReg:$md), (ins GPR:$rs1, GPR:$rs2),
               opcodestr, "$md, $rs1, $rs2", [], InstFormatOther> {
  bits<5> rs2;
  bits<5> rs1;
  bits<4> md;  // AccReg index for load

  let Inst{6-0}   = 0b1110111;   // AME prefix
  let Inst{10-7}  = md;
  let Inst{11}    = tr_val;
  let Inst{14-12} = eew_val;
  let Inst{19-15} = rs1;
  let Inst{24-20} = rs2;
  let Inst{25}    = 0;           // ls = 0 for load
  let Inst{31-26} = funct6_val;  // funct6
}

// Store from TileReg (for matrix A and B)
class RVInstBOSCAMEStoreTile32<bits<6> funct6_val, bits<3> eew_val, bit tr_val, string opcodestr>
    : RVInst<(outs), (ins TileReg:$ms3, GPR:$rs1, GPR:$rs2),
               opcodestr, "$ms3, $rs1, $rs2", [], InstFormatOther> {
  bits<5> rs2;
  bits<5> rs1;
  bits<4> ms3;  // TileReg index for store

  let Inst{6-0}   = 0b1110111;   // AME prefix
  let Inst{10-7}  = ms3;
  let Inst{11}    = tr_val;
  let Inst{14-12} = eew_val;
  let Inst{19-15} = rs1;
  let Inst{24-20} = rs2;
  let Inst{25}    = 1;           // ls = 1 for store
  let Inst{31-26} = funct6_val;  // funct6
}

// Store from AccReg (for accumulator C)
class RVInstBOSCAMEStoreAcc32<bits<6> funct6_val, bits<3> eew_val, bit tr_val, string opcodestr>
    : RVInst<(outs), (ins AccReg:$ms3, GPR:$rs1, GPR:$rs2),
               opcodestr, "$ms3, $rs1, $rs2", [], InstFormatOther> {
  bits<5> rs2;
  bits<5> rs1;
  bits<4> ms3;  // AccReg index for store

  let Inst{6-0}   = 0b1110111;   // AME prefix
  let Inst{10-7}  = ms3;
  let Inst{11}    = tr_val;
  let Inst{14-12} = eew_val;
  let Inst{19-15} = rs1;
  let Inst{24-20} = rs2;
  let Inst{25}    = 1;           // ls = 1 for store
  let Inst{31-26} = funct6_val;  // funct6
}

//===----------------------------------------------------------------------===//
// Load matrix A into TileReg - mlae*.m
// Syntax: mlae{8|16|32|64}.m md, (rs1), rs2
//   md:  Destination TileReg
//   rs1: Base address (GPR)
//   rs2: Row stride in bytes (GPR)
//===----------------------------------------------------------------------===//
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 1, mayStore = 0 in {
  def BOSC_AME_MLAE8_M  : RVInstBOSCAMELoadTile32<0b000001, 0b000, 0b0, "mlae8.m">;
  def BOSC_AME_MLAE16_M : RVInstBOSCAMELoadTile32<0b000001, 0b001, 0b0, "mlae16.m">;
  def BOSC_AME_MLAE32_M : RVInstBOSCAMELoadTile32<0b000001, 0b010, 0b0, "mlae32.m">;
  def BOSC_AME_MLAE64_M : RVInstBOSCAMELoadTile32<0b000001, 0b011, 0b0, "mlae64.m">;

  def BOSC_AME_MLATE8_M  : RVInstBOSCAMELoadTile32<0b000001, 0b000, 0b1, "mlate8.m">;
  def BOSC_AME_MLATE16_M : RVInstBOSCAMELoadTile32<0b000001, 0b001, 0b1, "mlate16.m">;
  def BOSC_AME_MLATE32_M : RVInstBOSCAMELoadTile32<0b000001, 0b010, 0b1, "mlate32.m">;
  def BOSC_AME_MLATE64_M : RVInstBOSCAMELoadTile32<0b000001, 0b011, 0b1, "mlate64.m">;

  def BOSC_AME_MLTRE8_M  : RVInstBOSCAMELoadTile32<0b000011, 0b000, 0b0, "mltre8.m">;
  def BOSC_AME_MLTRE16_M : RVInstBOSCAMELoadTile32<0b000011, 0b001, 0b0, "mltre16.m">;
  def BOSC_AME_MLTRE32_M : RVInstBOSCAMELoadTile32<0b000011, 0b010, 0b0, "mltre32.m">;
  def BOSC_AME_MLTRE64_M : RVInstBOSCAMELoadTile32<0b000011, 0b011, 0b0, "mltre64.m">;
}

//===----------------------------------------------------------------------===//
// Load matrix B into TileReg - mlbe*.m
// Syntax: mlbe{8|16|32|64}.m md, (rs1), rs2
//===----------------------------------------------------------------------===//
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 1, mayStore = 0 in {
  def BOSC_AME_MLBE8_M  : RVInstBOSCAMELoadTile32<0b000010, 0b000, 0b0, "mlbe8.m">;
  def BOSC_AME_MLBE16_M : RVInstBOSCAMELoadTile32<0b000010, 0b001, 0b0, "mlbe16.m">;
  def BOSC_AME_MLBE32_M : RVInstBOSCAMELoadTile32<0b000010, 0b010, 0b0, "mlbe32.m">;
  def BOSC_AME_MLBE64_M : RVInstBOSCAMELoadTile32<0b000010, 0b011, 0b0, "mlbe64.m">;

  def BOSC_AME_MLBTE8_M  : RVInstBOSCAMELoadTile32<0b000010, 0b000, 0b1, "mlbte8.m">;
  def BOSC_AME_MLBTE16_M : RVInstBOSCAMELoadTile32<0b000010, 0b001, 0b1, "mlbte16.m">;
  def BOSC_AME_MLBTE32_M : RVInstBOSCAMELoadTile32<0b000010, 0b010, 0b1, "mlbte32.m">;
  def BOSC_AME_MLBTE64_M : RVInstBOSCAMELoadTile32<0b000010, 0b011, 0b1, "mlbte64.m">;
}

//===----------------------------------------------------------------------===//
// Load matrix C (accumulator) into AccReg - mlce*.m
// Syntax: mlce{8|16|32|64}.m md, (rs1), rs2
//   md: Destination AccReg (MLEN×AMUL bits)
// Note: eew here refers to the output element width after widening
//===----------------------------------------------------------------------===//
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 1, mayStore = 0 in {
  def BOSC_AME_MLCE8_M  : RVInstBOSCAMELoadAcc32<0b000000, 0b000, 0b0, "mlce8.m">;
  def BOSC_AME_MLCE16_M : RVInstBOSCAMELoadAcc32<0b000000, 0b001, 0b0, "mlce16.m">;
  def BOSC_AME_MLCE32_M : RVInstBOSCAMELoadAcc32<0b000000, 0b010, 0b0, "mlce32.m">;
  def BOSC_AME_MLCE64_M : RVInstBOSCAMELoadAcc32<0b000000, 0b011, 0b0, "mlce64.m">;

  def BOSC_AME_MLCTE8_M  : RVInstBOSCAMELoadAcc32<0b000000, 0b000, 0b1, "mlcte8.m">;
  def BOSC_AME_MLCTE16_M : RVInstBOSCAMELoadAcc32<0b000000, 0b001, 0b1, "mlcte16.m">;
  def BOSC_AME_MLCTE32_M : RVInstBOSCAMELoadAcc32<0b000000, 0b010, 0b1, "mlcte32.m">;
  def BOSC_AME_MLCTE64_M : RVInstBOSCAMELoadAcc32<0b000000, 0b011, 0b1, "mlcte64.m">;

  def BOSC_AME_MLACCE8_M  : RVInstBOSCAMELoadAcc32<0b000011, 0b000, 0b1, "mlacce8.m">;
  def BOSC_AME_MLACCE16_M : RVInstBOSCAMELoadAcc32<0b000011, 0b001, 0b1, "mlacce16.m">;
  def BOSC_AME_MLACCE32_M : RVInstBOSCAMELoadAcc32<0b000011, 0b010, 0b1, "mlacce32.m">;
  def BOSC_AME_MLACCE64_M : RVInstBOSCAMELoadAcc32<0b000011, 0b011, 0b1, "mlacce64.m">;
}

//===----------------------------------------------------------------------===//
// Store matrix A from TileReg - msae*.m
// Syntax: msae{8|16|32|64}.m ms3, (rs1), rs2
//===----------------------------------------------------------------------===//
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 1 in {
  def BOSC_AME_MSAE8_M  : RVInstBOSCAMEStoreTile32<0b000001, 0b000, 0b0, "msae8.m">;
  def BOSC_AME_MSAE16_M : RVInstBOSCAMEStoreTile32<0b000001, 0b001, 0b0, "msae16.m">;
  def BOSC_AME_MSAE32_M : RVInstBOSCAMEStoreTile32<0b000001, 0b010, 0b0, "msae32.m">;
  def BOSC_AME_MSAE64_M : RVInstBOSCAMEStoreTile32<0b000001, 0b011, 0b0, "msae64.m">;

  def BOSC_AME_MSATE8_M  : RVInstBOSCAMEStoreTile32<0b000001, 0b000, 0b1, "msate8.m">;
  def BOSC_AME_MSATE16_M : RVInstBOSCAMEStoreTile32<0b000001, 0b001, 0b1, "msate16.m">;
  def BOSC_AME_MSATE32_M : RVInstBOSCAMEStoreTile32<0b000001, 0b010, 0b1, "msate32.m">;
  def BOSC_AME_MSATE64_M : RVInstBOSCAMEStoreTile32<0b000001, 0b011, 0b1, "msate64.m">;

  def BOSC_AME_MSTRE8_M  : RVInstBOSCAMEStoreTile32<0b000011, 0b000, 0b0, "mstre8.m">;
  def BOSC_AME_MSTRE16_M : RVInstBOSCAMEStoreTile32<0b000011, 0b001, 0b0, "mstre16.m">;
  def BOSC_AME_MSTRE32_M : RVInstBOSCAMEStoreTile32<0b000011, 0b010, 0b0, "mstre32.m">;
  def BOSC_AME_MSTRE64_M : RVInstBOSCAMEStoreTile32<0b000011, 0b011, 0b0, "mstre64.m">;
}

//===----------------------------------------------------------------------===//
// Store matrix B from TileReg - msbe*.m
// Syntax: msbe{8|16|32|64}.m ms3, (rs1), rs2
//===----------------------------------------------------------------------===//
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 1 in {
  def BOSC_AME_MSBE8_M  : RVInstBOSCAMEStoreTile32<0b000010, 0b000, 0b0, "msbe8.m">;
  def BOSC_AME_MSBE16_M : RVInstBOSCAMEStoreTile32<0b000010, 0b001, 0b0, "msbe16.m">;
  def BOSC_AME_MSBE32_M : RVInstBOSCAMEStoreTile32<0b000010, 0b010, 0b0, "msbe32.m">;
  def BOSC_AME_MSBE64_M : RVInstBOSCAMEStoreTile32<0b000010, 0b011, 0b0, "msbe64.m">;

  def BOSC_AME_MSBTE8_M  : RVInstBOSCAMEStoreTile32<0b000010, 0b000, 0b1, "msbte8.m">;
  def BOSC_AME_MSBTE16_M : RVInstBOSCAMEStoreTile32<0b000010, 0b001, 0b1, "msbte16.m">;
  def BOSC_AME_MSBTE32_M : RVInstBOSCAMEStoreTile32<0b000010, 0b010, 0b1, "msbte32.m">;
  def BOSC_AME_MSBTE64_M : RVInstBOSCAMEStoreTile32<0b000010, 0b011, 0b1, "msbte64.m">;
}

//===----------------------------------------------------------------------===//
// Store matrix C (accumulator) from AccReg - msce*.m
// Syntax: msce{8|16|32|64}.m ms3, (rs1), rs2
//   ms3: Source AccReg (MLEN×AMUL bits)
// This is the primary store for computation results
//===----------------------------------------------------------------------===//
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 1 in {
  def BOSC_AME_MSCE8_M  : RVInstBOSCAMEStoreAcc32<0b000000, 0b000, 0b0, "msce8.m">;
  def BOSC_AME_MSCE16_M : RVInstBOSCAMEStoreAcc32<0b000000, 0b001, 0b0, "msce16.m">;
  def BOSC_AME_MSCE32_M : RVInstBOSCAMEStoreAcc32<0b000000, 0b010, 0b0, "msce32.m">;
  def BOSC_AME_MSCE64_M : RVInstBOSCAMEStoreAcc32<0b000000, 0b011, 0b0, "msce64.m">;

  def BOSC_AME_MSCTE8_M  : RVInstBOSCAMEStoreAcc32<0b000000, 0b000, 0b1, "mscte8.m">;
  def BOSC_AME_MSCTE16_M : RVInstBOSCAMEStoreAcc32<0b000000, 0b001, 0b1, "mscte16.m">;
  def BOSC_AME_MSCTE32_M : RVInstBOSCAMEStoreAcc32<0b000000, 0b010, 0b1, "mscte32.m">;
  def BOSC_AME_MSCTE64_M : RVInstBOSCAMEStoreAcc32<0b000000, 0b011, 0b1, "mscte64.m">;

  def BOSC_AME_MSACCE8_M  : RVInstBOSCAMEStoreAcc32<0b000011, 0b000, 0b1, "msacce8.m">;
  def BOSC_AME_MSACCE16_M : RVInstBOSCAMEStoreAcc32<0b000011, 0b001, 0b1, "msacce16.m">;
  def BOSC_AME_MSACCE32_M : RVInstBOSCAMEStoreAcc32<0b000011, 0b010, 0b1, "msacce32.m">;
  def BOSC_AME_MSACCE64_M : RVInstBOSCAMEStoreAcc32<0b000011, 0b011, 0b1, "msacce64.m">;
}

//===----------------------------------------------------------------------===//
// BOSC AME Data Move Instructions
//===----------------------------------------------------------------------===//
// Data Move Instruction format (32-bit):
// | 31:26  | 25 | 24:20 | 19:15 | 14:12 | 11:7 | 6:0    |
// | funct6 | di | rs2   | rs1   | eew   | rd   | OP-M32 |
//===----------------------------------------------------------------------===//
class RVInstBOSCAMEMove<dag outs, dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<6> funct6;
  bit     di;
  bits<5> rs2;
  bits<5> rs1;
  bits<3> eew;    // 0=int8, 1=int16, 2=int32, 3=int64, 7=int4
  bits<5> rd;
  bits<7> opcode = 0b1110111;

  let Inst{31-26} = funct6;
  let Inst{25}    = di;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = eew;
  let Inst{11-7}  = rd;
  let Inst{6-0} = 0b1110111;
}

// Data Move Instructions between Matrix Registers
class RVInstBOSCAMEMove_T_T<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs TileReg:$md), (ins TileReg:$ms1),
                        opcodestr, "$md, $ms1"> {
  bits<5> md;   // TileReg index for store
  bits<5> ms1;  // TileReg index for load

  let funct6 = 0b000111;
  let di     = 0b0;
  let rs2    = 0b00000;
  let rs1    = ms1;
  let eew    = eew_val;
  let rd     = md;
  let opcode = 0b1110111;
}

class RVInstBOSCAMEMove_A_A<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs AccReg:$md), (ins AccReg:$ms1),
                        opcodestr, "$md, $ms1"> {
  bits<5> md;   // AccReg index for store
  bits<5> ms1;  // AccReg index for load

  let funct6 = 0b000111;
  let di     = 0b0;
  let rs2    = 0b00001;
  let rs1    = ms1;
  let eew    = eew_val;
  let rd     = md;
  let opcode = 0b1110111;
}

class RVInstBOSCAMEMove_A_T<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs AccReg:$md), (ins TileReg:$ms1, GPR:$rs2),
                        opcodestr, "$md, $ms1, $rs2"> {
  bits<5> md;   // AccReg index for store
  bits<5> ms1;  // TileReg index for load

  let funct6 = 0b000100;
  let di     = 0b0;
  let rs1    = ms1;
  let eew    = eew_val;
  let rd     = md;
  let opcode = 0b1110111;
}

class RVInstBOSCAMEMove_T_A<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs TileReg:$md), (ins AccReg:$ms1, GPR:$rs2),
                        opcodestr, "$md, $ms1, $rs2"> {
  bits<5> md;   // TileReg index for load
  bits<5> ms1;  // AccReg index for store

  let funct6 = 0b000100;
  let di     = 0b1;
  let rs1    = ms1;
  let eew    = eew_val;
  let rd     = md;
  let opcode = 0b1110111;
}

class RVInstBOSCAMEMoveImm_A_T<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs AccReg:$md), (ins TileReg:$ms1, uimm5:$imm),
                         opcodestr, "$md, $ms1, $imm"> {
  bits<5> imm;
  bits<5> ms1;
  bits<5> md;

  let funct6 = 0b000100;
  let di     = 0b0;
  let rs2    = imm;
  let rs1    = ms1;
  let eew    = eew_val;
  let rd     = md;
  let opcode = 0b1110111;
}

class RVInstBOSCAMEMoveImm_T_A<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs TileReg:$md), (ins AccReg:$ms1, uimm5:$imm),
                         opcodestr, "$md, $ms1, $imm"> {
  bits<5> imm;
  bits<5> ms1;
  bits<5> md;

  let funct6 = 0b000100;
  let di     = 0b1;
  let rs2    = imm;
  let rs1    = ms1;
  let eew    = eew_val;
  let rd     = md;
  let opcode = 0b1110111;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {

  def BOSC_AME_MMVE8_T_T  : RVInstBOSCAMEMove_T_T<0b000, "mmve8.t.t">;
  def BOSC_AME_MMVE16_T_T : RVInstBOSCAMEMove_T_T<0b001, "mmve16.t.t">;
  def BOSC_AME_MMVE32_T_T : RVInstBOSCAMEMove_T_T<0b010, "mmve32.t.t">;
  def BOSC_AME_MMVE64_T_T : RVInstBOSCAMEMove_T_T<0b011, "mmve64.t.t">;

  def BOSC_AME_MMVE8_A_A  : RVInstBOSCAMEMove_A_A<0b000, "mmve8.a.a">;
  def BOSC_AME_MMVE16_A_A : RVInstBOSCAMEMove_A_A<0b001, "mmve16.a.a">;
  def BOSC_AME_MMVE32_A_A : RVInstBOSCAMEMove_A_A<0b010, "mmve32.a.a">;
  def BOSC_AME_MMVE64_A_A : RVInstBOSCAMEMove_A_A<0b011, "mmve64.a.a">;

  def BOSC_AME_MMVE8_A_T  : RVInstBOSCAMEMove_A_T<0b000, "mmve8.a.t">;
  def BOSC_AME_MMVE16_A_T : RVInstBOSCAMEMove_A_T<0b001, "mmve16.a.t">;
  def BOSC_AME_MMVE32_A_T : RVInstBOSCAMEMove_A_T<0b010, "mmve32.a.t">;
  def BOSC_AME_MMVE64_A_T : RVInstBOSCAMEMove_A_T<0b011, "mmve64.a.t">;

  def BOSC_AME_MMVE8_T_A  : RVInstBOSCAMEMove_T_A<0b000, "mmve8.t.a">;
  def BOSC_AME_MMVE16_T_A : RVInstBOSCAMEMove_T_A<0b001, "mmve16.t.a">;
  def BOSC_AME_MMVE32_T_A : RVInstBOSCAMEMove_T_A<0b010, "mmve32.t.a">;
  def BOSC_AME_MMVE64_T_A : RVInstBOSCAMEMove_T_A<0b011, "mmve64.t.a">;

  def BOSC_AME_MMVIE8_A_T  : RVInstBOSCAMEMoveImm_A_T<0b100, "mmvie8.a.t">;
  def BOSC_AME_MMVIE16_A_T : RVInstBOSCAMEMoveImm_A_T<0b101, "mmvie16.a.t">;
  def BOSC_AME_MMVIE32_A_T : RVInstBOSCAMEMoveImm_A_T<0b110, "mmvie32.a.t">;
  def BOSC_AME_MMVIE64_A_T : RVInstBOSCAMEMoveImm_A_T<0b111, "mmvie64.a.t">;

  def BOSC_AME_MMVIE8_T_A  : RVInstBOSCAMEMoveImm_T_A<0b100, "mmvie8.t.a">;
  def BOSC_AME_MMVIE16_T_A : RVInstBOSCAMEMoveImm_T_A<0b101, "mmvie16.t.a">;
  def BOSC_AME_MMVIE32_T_A : RVInstBOSCAMEMoveImm_T_A<0b110, "mmvie32.t.a">;
  def BOSC_AME_MMVIE64_T_A : RVInstBOSCAMEMoveImm_T_A<0b111, "mmvie64.t.a">;
}

// Data Move Instructions between Matrix and Integer
class RVInstBOSCAMEMove_X_T<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs GPR:$rd), (ins TileReg:$ms1, GPR:$rs2),
                        opcodestr, "$rd, $ms1, $rs2"> {
  bits<5> rs2;
  bits<5> ms1;
  bits<5> rd;

  let funct6 = 0b000101;
  let di     = 0b0;
  let rs1    = ms1;
  let eew    = eew_val;
}

class RVInstBOSCAMEMove_T_X<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs TileReg:$md), (ins GPR:$rs1, GPR:$rs2),
                        opcodestr, "$md, $rs1, $rs2"> {
  bits<5> rs2;
  bits<5> rs1;
  bits<5> md;

  let funct6 = 0b000101;
  let di     = 0b1;
  let eew    = eew_val;
  let rd     = md;
}

class RVInstBOSCAMEMove_X_A<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs GPR:$rd), (ins AccReg:$ms1, GPR:$rs2),
                        opcodestr, "$rd, $ms1, $rs2"> {
  bits<5> rs2;
  bits<5> ms1;
  bits<5> rd;

  let funct6 = 0b000101;
  let di     = 0b0;
  let rs1    = ms1;
  let eew    = eew_val;
}

class RVInstBOSCAMEMove_A_X<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs AccReg:$md), (ins GPR:$rs1, GPR:$rs2),
                        opcodestr, "$md, $rs1, $rs2"> {
  bits<5> rs2;
  bits<5> rs1;
  bits<5> md;

  let funct6 = 0b000101;
  let di     = 0b1;
  let eew    = eew_val;
  let rd     = md;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {

  def BOSC_AME_MMVE8_X_T  : RVInstBOSCAMEMove_X_T<0b000, "mmve8.x.t">;
  def BOSC_AME_MMVE16_X_T : RVInstBOSCAMEMove_X_T<0b001, "mmve16.x.t">;
  def BOSC_AME_MMVE32_X_T : RVInstBOSCAMEMove_X_T<0b010, "mmve32.x.t">;
  def BOSC_AME_MMVE64_X_T : RVInstBOSCAMEMove_X_T<0b011, "mmve64.x.t">;

  def BOSC_AME_MMVE8_T_X  : RVInstBOSCAMEMove_T_X<0b000, "mmve8.t.x">;
  def BOSC_AME_MMVE16_T_X : RVInstBOSCAMEMove_T_X<0b001, "mmve16.t.x">;
  def BOSC_AME_MMVE32_T_X : RVInstBOSCAMEMove_T_X<0b010, "mmve32.t.x">;
  def BOSC_AME_MMVE64_T_X : RVInstBOSCAMEMove_T_X<0b011, "mmve64.t.x">;

  def BOSC_AME_MMVE8_X_A  : RVInstBOSCAMEMove_X_A<0b100, "mmve8.x.a">;
  def BOSC_AME_MMVE16_X_A : RVInstBOSCAMEMove_X_A<0b101, "mmve16.x.a">;
  def BOSC_AME_MMVE32_X_A : RVInstBOSCAMEMove_X_A<0b110, "mmve32.x.a">;
  def BOSC_AME_MMVE64_X_A : RVInstBOSCAMEMove_X_A<0b111, "mmve64.x.a">;

  def BOSC_AME_MMVE8_A_X  : RVInstBOSCAMEMove_A_X<0b100, "mmve8.a.x">;
  def BOSC_AME_MMVE16_A_X : RVInstBOSCAMEMove_A_X<0b101, "mmve16.a.x">;
  def BOSC_AME_MMVE32_A_X : RVInstBOSCAMEMove_A_X<0b110, "mmve32.a.x">;
  def BOSC_AME_MMVE64_A_X : RVInstBOSCAMEMove_A_X<0b111, "mmve64.a.x">;
}

// Data Move Instructions between Matrix and Float-point
class RVInstBOSCAMEFMove_F_T<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs GPR:$rd), (ins TileReg:$ms1, GPR:$rs2),
                        opcodestr, "$rd, $ms1, $rs2"> {
  bits<5> rs2;
  bits<5> ms1;
  bits<5> rd;

  let funct6 = 0b000110;
  let di     = 0b0;
  let rs1    = ms1;
  let eew    = eew_val;
}

class RVInstBOSCAMEFMove_T_F<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs TileReg:$md), (ins GPR:$rs1, GPR:$rs2),
                        opcodestr, "$md, $rs1, $rs2"> {
  bits<5> rs2;
  bits<5> rs1;
  bits<5> md;

  let funct6 = 0b000110;
  let di     = 0b1;
  let eew    = eew_val;
  let rd     = md;
}

class RVInstBOSCAMEFMove_F_A<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs GPR:$rd), (ins AccReg:$ms1, GPR:$rs2),
                        opcodestr, "$rd, $ms1, $rs2"> {
  bits<5> rs2;
  bits<5> ms1;
  bits<5> rd;

  let funct6 = 0b000110;
  let di     = 0b0;
  let rs1    = ms1;
  let eew    = eew_val;
}

class RVInstBOSCAMEFMove_A_F<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs AccReg:$md), (ins GPR:$rs1, GPR:$rs2),
                        opcodestr, "$md, $rs1, $rs2"> {
  bits<5> rs2;
  bits<5> rs1;
  bits<5> md;

  let funct6 = 0b000110;
  let di     = 0b1;
  let eew    = eew_val;
  let rd     = md;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {

  def BOSC_AME_MFMVE8_F_T  : RVInstBOSCAMEFMove_F_T<0b000, "mfmve8.f.t">;
  def BOSC_AME_MFMVE16_F_T : RVInstBOSCAMEFMove_F_T<0b001, "mfmve16.f.t">;
  def BOSC_AME_MFMVE32_F_T : RVInstBOSCAMEFMove_F_T<0b010, "mfmve32.f.t">;
  def BOSC_AME_MFMVE64_F_T : RVInstBOSCAMEFMove_F_T<0b011, "mfmve64.f.t">;

  def BOSC_AME_MFMVE8_T_F  : RVInstBOSCAMEFMove_T_F<0b000, "mfmve8.t.f">;
  def BOSC_AME_MFMVE16_T_F : RVInstBOSCAMEFMove_T_F<0b001, "mfmve16.t.f">;
  def BOSC_AME_MFMVE32_T_F : RVInstBOSCAMEFMove_T_F<0b010, "mfmve32.t.f">;
  def BOSC_AME_MFMVE64_T_F : RVInstBOSCAMEFMove_T_F<0b011, "mfmve64.t.f">;

  def BOSC_AME_MFMVE8_F_A  : RVInstBOSCAMEFMove_F_A<0b100, "mfmve8.f.a">;
  def BOSC_AME_MFMVE16_F_A : RVInstBOSCAMEFMove_F_A<0b101, "mfmve16.f.a">;
  def BOSC_AME_MFMVE32_F_A : RVInstBOSCAMEFMove_F_A<0b110, "mfmve32.f.a">;
  def BOSC_AME_MFMVE64_F_A : RVInstBOSCAMEFMove_F_A<0b111, "mfmve64.f.a">;

  def BOSC_AME_MFMVE8_A_F  : RVInstBOSCAMEFMove_A_F<0b100, "mfmve8.a.f">;
  def BOSC_AME_MFMVE16_A_F : RVInstBOSCAMEFMove_A_F<0b101, "mfmve16.a.f">;
  def BOSC_AME_MFMVE32_A_F : RVInstBOSCAMEFMove_A_F<0b110, "mfmve32.a.f">;
  def BOSC_AME_MFMVE64_A_F : RVInstBOSCAMEFMove_A_F<0b111, "mfmve64.a.f">;
}

// Data Broadcast Instructions
class RVInstBOSCAMEBroadcastTile<bits<5> rs2_val, bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs TileReg:$md), (ins TileReg:$ms1),
                        opcodestr, "$md, $ms1"> {
  bits<5> md;
  bits<5> ms1;

  let funct6 = 0b000111;
  let di     = 0b0;
  let rs2    = rs2_val;
  let rs1    = ms1;
  let eew    = eew_val;
  let rd     = md;
  let opcode = 0b1110111;
}

class RVInstBOSCAMEBroadcastAcc<bits<5> rs2_val, bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs AccReg:$md), (ins AccReg:$ms1),
                        opcodestr, "$md, $ms1"> {
  bits<5> md;
  bits<5> ms1;

  let funct6 = 0b000111;
  let di     = 0b0;
  let rs2    = rs2_val;
  let rs1    = ms1;
  let eew    = eew_val;
  let rd     = md;
  let opcode = 0b1110111;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  // Row broadcast from TileReg/AccReg
  def BOSC_AME_MBCAR_M : RVInstBOSCAMEBroadcastTile<0b10001, 0b000, "mbcar.m">;
  def BOSC_AME_MBCBR_M : RVInstBOSCAMEBroadcastTile<0b10010, 0b000, "mbcbr.m">;
  def BOSC_AME_MBCCR_M : RVInstBOSCAMEBroadcastAcc<0b10000, 0b000, "mbccr.m">;
  // Column broadcast from TileReg
  def BOSC_AME_MBCACE8_M  : RVInstBOSCAMEBroadcastTile<0b10101, 0b000, "mbcace8.m">;
  def BOSC_AME_MBCACE16_M : RVInstBOSCAMEBroadcastTile<0b10101, 0b001, "mbcace16.m">;
  def BOSC_AME_MBCACE32_M : RVInstBOSCAMEBroadcastTile<0b10101, 0b010, "mbcace32.m">;
  def BOSC_AME_MBCACE64_M : RVInstBOSCAMEBroadcastTile<0b10101, 0b011, "mbcace64.m">;
  // Column broadcast from TileReg
  def BOSC_AME_MBCBCE8_M  : RVInstBOSCAMEBroadcastTile<0b10110, 0b000, "mbcbce8.m">;
  def BOSC_AME_MBCBCE16_M : RVInstBOSCAMEBroadcastTile<0b10110, 0b001, "mbcbce16.m">;
  def BOSC_AME_MBCBCE32_M : RVInstBOSCAMEBroadcastTile<0b10110, 0b010, "mbcbce32.m">;
  def BOSC_AME_MBCBCE64_M : RVInstBOSCAMEBroadcastTile<0b10110, 0b011, "mbcbce64.m">;
  // Column broadcast from AccReg
  def BOSC_AME_MBCCCE8_M  : RVInstBOSCAMEBroadcastAcc<0b10100, 0b000, "mbccce8.m">;
  def BOSC_AME_MBCCCE16_M : RVInstBOSCAMEBroadcastAcc<0b10100, 0b001, "mbccce16.m">;
  def BOSC_AME_MBCCCE32_M : RVInstBOSCAMEBroadcastAcc<0b10100, 0b010, "mbccce32.m">;
  def BOSC_AME_MBCCCE64_M : RVInstBOSCAMEBroadcastAcc<0b10100, 0b011, "mbccce64.m">;
  // Element broadcast from TileReg
  def BOSC_AME_MBCAEE8_M  : RVInstBOSCAMEBroadcastTile<0b11001, 0b000, "mbcaee8.m">;
  def BOSC_AME_MBCAEE16_M : RVInstBOSCAMEBroadcastTile<0b11001, 0b001, "mbcaee16.m">;
  def BOSC_AME_MBCAEE32_M : RVInstBOSCAMEBroadcastTile<0b11001, 0b010, "mbcaee32.m">;
  def BOSC_AME_MBCAEE64_M : RVInstBOSCAMEBroadcastTile<0b11001, 0b011, "mbcaee64.m">;
  // Element broadcast from TileReg
  def BOSC_AME_MBCBEE8_M  : RVInstBOSCAMEBroadcastTile<0b11010, 0b000, "mbcbee8.m">;
  def BOSC_AME_MBCBEE16_M : RVInstBOSCAMEBroadcastTile<0b11010, 0b001, "mbcbee16.m">;
  def BOSC_AME_MBCBEE32_M : RVInstBOSCAMEBroadcastTile<0b11010, 0b010, "mbcbee32.m">;
  def BOSC_AME_MBCBEE64_M : RVInstBOSCAMEBroadcastTile<0b11010, 0b011, "mbcbee64.m">;
  // Element broadcast from AccReg
  def BOSC_AME_MBCCEE8_M  : RVInstBOSCAMEBroadcastAcc<0b11000, 0b000, "mbccee8.m">;
  def BOSC_AME_MBCCEE16_M : RVInstBOSCAMEBroadcastAcc<0b11000, 0b001, "mbccee16.m">;
  def BOSC_AME_MBCCEE32_M : RVInstBOSCAMEBroadcastAcc<0b11000, 0b010, "mbccee32.m">;
  def BOSC_AME_MBCCEE64_M : RVInstBOSCAMEBroadcastAcc<0b11000, 0b011, "mbccee64.m">;
}

// Matrix Transpose Instructions
class RVInstBOSCAMETransposeTile<bits<5> rs2_val, bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs TileReg:$md), (ins TileReg:$ms1),
                        opcodestr, "$md, $ms1"> {
  bits<5> md;
  bits<5> ms1;

  let funct6 = 0b000111;
  let di     = 0b0;
  let rs2    = rs2_val;
  let rs1    = ms1;
  let eew    = eew_val;
  let rd     = md;
  let opcode = 0b1110111;
}

class RVInstBOSCAMETransposeAcc<bits<5> rs2_val, bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs AccReg:$md), (ins AccReg:$ms1),
                        opcodestr, "$md, $ms1"> {
  bits<5> md;
  bits<5> ms1;

  let funct6 = 0b000111;
  let di     = 0b0;
  let rs2    = rs2_val;
  let rs1    = ms1;
  let eew    = eew_val;
  let rd     = md;
  let opcode = 0b1110111;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  // Transpose square matrix of tile register
  def BOSC_AME_MTAE8_M : RVInstBOSCAMETransposeTile<0b11101, 0b000, "mtae8.m">;
  def BOSC_AME_MTAE16_M : RVInstBOSCAMETransposeTile<0b11101, 0b001, "mtae16.m">;
  def BOSC_AME_MTAE32_M : RVInstBOSCAMETransposeTile<0b11101, 0b010, "mtae32.m">;
  def BOSC_AME_MTAE64_M : RVInstBOSCAMETransposeTile<0b11101, 0b011, "mtae64.m">;
  // Transpose square matrix of tile register
  def BOSC_AME_MTBE8_M : RVInstBOSCAMETransposeTile<0b11110, 0b000, "mtbe8.m">;
  def BOSC_AME_MTBE16_M : RVInstBOSCAMETransposeTile<0b11110, 0b001, "mtbe16.m">;
  def BOSC_AME_MTBE32_M : RVInstBOSCAMETransposeTile<0b11110, 0b010, "mtbe32.m">;
  def BOSC_AME_MTBE64_M : RVInstBOSCAMETransposeTile<0b11110, 0b011, "mtbe64.m">;
  // Transpose square matrix of acc register
  def BOSC_AME_MTCE8_M : RVInstBOSCAMETransposeAcc<0b11100, 0b000, "mtce8.m">;
  def BOSC_AME_MTCE16_M : RVInstBOSCAMETransposeAcc<0b11100, 0b001, "mtce16.m">;
  def BOSC_AME_MTCE32_M : RVInstBOSCAMETransposeAcc<0b11100, 0b010, "mtce32.m">;
  def BOSC_AME_MTCE64_M : RVInstBOSCAMETransposeAcc<0b11100, 0b011, "mtce64.m">;
}

//===----------------------------------------------------------------------===//
// BOSC AME Data Move Instructions
//===----------------------------------------------------------------------===//
// Data Move Instruction format (32-bit):
// | 31:26  | 25 | 24:20 | 19:15 | 14:12 | 11:7 | 6:0    |
// | funct6 | di | rs2   | rs1   | eew   | rd   | OP-M32 |
//===----------------------------------------------------------------------===//
class RVInstBOSCAMEMove<dag outs, dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<6> funct6;
  bit     di;
  bits<5> rs2;
  bits<5> rs1;
  bits<3> eew;    // 0=int8, 1=int16, 2=int32, 3=int64, 7=int4
  bits<5> rd;
  bits<7> opcode = 0b1110111;

  let Inst{31-26} = funct6;
  let Inst{25}    = di;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = eew;
  let Inst{11-7}  = rd;
  let Inst{6-0} = 0b1110111;
}

// Data Move Instructions between Matrix Registers
class RVInstBOSCAMEMove_T_T<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs TileReg:$md), (ins TileReg:$ms1),
                        opcodestr, "$md, $ms1"> {
  bits<5> md;   // TileReg index for store
  bits<5> ms1;  // TileReg index for load

  let funct6 = 0b000111;
  let di     = 0b0;
  let rs2    = 0b00000;
  let rs1    = ms1;
  let eew    = eew_val;
  let rd     = md;
  let opcode = 0b1110111;
}

class RVInstBOSCAMEMove_A_A<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs AccReg:$md), (ins AccReg:$ms1),
                        opcodestr, "$md, $ms1"> {
  bits<5> md;   // AccReg index for store
  bits<5> ms1;  // AccReg index for load

  let funct6 = 0b000111;
  let di     = 0b0;
  let rs2    = 0b00001;
  let rs1    = ms1;
  let eew    = eew_val;
  let rd     = md;
  let opcode = 0b1110111;
}

class RVInstBOSCAMEMove_A_T<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs AccReg:$md), (ins TileReg:$ms1, GPR:$rs2),
                        opcodestr, "$md, $ms1, $rs2"> {
  bits<5> md;   // AccReg index for store
  bits<5> ms1;  // TileReg index for load

  let funct6 = 0b000100;
  let di     = 0b0;
  let rs1    = ms1;
  let eew    = eew_val;
  let rd     = md;
  let opcode = 0b1110111;
}

class RVInstBOSCAMEMove_T_A<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs TileReg:$md), (ins AccReg:$ms1, GPR:$rs2),
                        opcodestr, "$md, $ms1, $rs2"> {
  bits<5> md;   // TileReg index for load
  bits<5> ms1;  // AccReg index for store

  let funct6 = 0b000100;
  let di     = 0b1;
  let rs1    = ms1;
  let eew    = eew_val;
  let rd     = md;
  let opcode = 0b1110111;
}

class RVInstBOSCAMEMoveImm_A_T<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs AccReg:$md), (ins TileReg:$ms1, uimm5:$imm),
                         opcodestr, "$md, $ms1, $imm"> {
  bits<5> imm;
  bits<5> ms1;
  bits<5> md;

  let funct6 = 0b000100;
  let di     = 0b0;
  let rs2    = imm;
  let rs1    = ms1;
  let eew    = eew_val;
  let rd     = md;
  let opcode = 0b1110111;
}

class RVInstBOSCAMEMoveImm_T_A<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs TileReg:$md), (ins AccReg:$ms1, uimm5:$imm),
                         opcodestr, "$md, $ms1, $imm"> {
  bits<5> imm;
  bits<5> ms1;
  bits<5> md;

  let funct6 = 0b000100;
  let di     = 0b1;
  let rs2    = imm;
  let rs1    = ms1;
  let eew    = eew_val;
  let rd     = md;
  let opcode = 0b1110111;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {

  def BOSC_AME_MMVE8_T_T  : RVInstBOSCAMEMove_T_T<0b000, "mmve8.t.t">;
  def BOSC_AME_MMVE16_T_T : RVInstBOSCAMEMove_T_T<0b001, "mmve16.t.t">;
  def BOSC_AME_MMVE32_T_T : RVInstBOSCAMEMove_T_T<0b010, "mmve32.t.t">;
  def BOSC_AME_MMVE64_T_T : RVInstBOSCAMEMove_T_T<0b011, "mmve64.t.t">;

  def BOSC_AME_MMVE8_A_A  : RVInstBOSCAMEMove_A_A<0b000, "mmve8.a.a">;
  def BOSC_AME_MMVE16_A_A : RVInstBOSCAMEMove_A_A<0b001, "mmve16.a.a">;
  def BOSC_AME_MMVE32_A_A : RVInstBOSCAMEMove_A_A<0b010, "mmve32.a.a">;
  def BOSC_AME_MMVE64_A_A : RVInstBOSCAMEMove_A_A<0b011, "mmve64.a.a">;

  def BOSC_AME_MMVE8_A_T  : RVInstBOSCAMEMove_A_T<0b000, "mmve8.a.t">;
  def BOSC_AME_MMVE16_A_T : RVInstBOSCAMEMove_A_T<0b001, "mmve16.a.t">;
  def BOSC_AME_MMVE32_A_T : RVInstBOSCAMEMove_A_T<0b010, "mmve32.a.t">;
  def BOSC_AME_MMVE64_A_T : RVInstBOSCAMEMove_A_T<0b011, "mmve64.a.t">;

  def BOSC_AME_MMVE8_T_A  : RVInstBOSCAMEMove_T_A<0b000, "mmve8.t.a">;
  def BOSC_AME_MMVE16_T_A : RVInstBOSCAMEMove_T_A<0b001, "mmve16.t.a">;
  def BOSC_AME_MMVE32_T_A : RVInstBOSCAMEMove_T_A<0b010, "mmve32.t.a">;
  def BOSC_AME_MMVE64_T_A : RVInstBOSCAMEMove_T_A<0b011, "mmve64.t.a">;

  def BOSC_AME_MMVIE8_A_T  : RVInstBOSCAMEMoveImm_A_T<0b100, "mmvie8.a.t">;
  def BOSC_AME_MMVIE16_A_T : RVInstBOSCAMEMoveImm_A_T<0b101, "mmvie16.a.t">;
  def BOSC_AME_MMVIE32_A_T : RVInstBOSCAMEMoveImm_A_T<0b110, "mmvie32.a.t">;
  def BOSC_AME_MMVIE64_A_T : RVInstBOSCAMEMoveImm_A_T<0b111, "mmvie64.a.t">;

  def BOSC_AME_MMVIE8_T_A  : RVInstBOSCAMEMoveImm_T_A<0b100, "mmvie8.t.a">;
  def BOSC_AME_MMVIE16_T_A : RVInstBOSCAMEMoveImm_T_A<0b101, "mmvie16.t.a">;
  def BOSC_AME_MMVIE32_T_A : RVInstBOSCAMEMoveImm_T_A<0b110, "mmvie32.t.a">;
  def BOSC_AME_MMVIE64_T_A : RVInstBOSCAMEMoveImm_T_A<0b111, "mmvie64.t.a">;
}

// Data Move Instructions between Matrix and Integer
class RVInstBOSCAMEMove_X_T<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs GPR:$rd), (ins TileReg:$ms1, GPR:$rs2),
                        opcodestr, "$rd, $ms1, $rs2"> {
  bits<5> rs2;
  bits<5> ms1;
  bits<5> rd;

  let funct6 = 0b000101;
  let di     = 0b0;
  let rs1    = ms1;
  let eew    = eew_val;
}

class RVInstBOSCAMEMove_T_X<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs TileReg:$md), (ins GPR:$rs1, GPR:$rs2),
                        opcodestr, "$md, $rs1, $rs2"> {
  bits<5> rs2;
  bits<5> rs1;
  bits<5> md;

  let funct6 = 0b000101;
  let di     = 0b1;
  let eew    = eew_val;
  let rd     = md;
}

class RVInstBOSCAMEMove_X_A<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs GPR:$rd), (ins AccReg:$ms1, GPR:$rs2),
                        opcodestr, "$rd, $ms1, $rs2"> {
  bits<5> rs2;
  bits<5> ms1;
  bits<5> rd;

  let funct6 = 0b000101;
  let di     = 0b0;
  let rs1    = ms1;
  let eew    = eew_val;
}

class RVInstBOSCAMEMove_A_X<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs AccReg:$md), (ins GPR:$rs1, GPR:$rs2),
                        opcodestr, "$md, $rs1, $rs2"> {
  bits<5> rs2;
  bits<5> rs1;
  bits<5> md;

  let funct6 = 0b000101;
  let di     = 0b1;
  let eew    = eew_val;
  let rd     = md;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {

  def BOSC_AME_MMVE8_X_T  : RVInstBOSCAMEMove_X_T<0b000, "mmve8.x.t">;
  def BOSC_AME_MMVE16_X_T : RVInstBOSCAMEMove_X_T<0b001, "mmve16.x.t">;
  def BOSC_AME_MMVE32_X_T : RVInstBOSCAMEMove_X_T<0b010, "mmve32.x.t">;
  def BOSC_AME_MMVE64_X_T : RVInstBOSCAMEMove_X_T<0b011, "mmve64.x.t">;

  def BOSC_AME_MMVE8_T_X  : RVInstBOSCAMEMove_T_X<0b000, "mmve8.t.x">;
  def BOSC_AME_MMVE16_T_X : RVInstBOSCAMEMove_T_X<0b001, "mmve16.t.x">;
  def BOSC_AME_MMVE32_T_X : RVInstBOSCAMEMove_T_X<0b010, "mmve32.t.x">;
  def BOSC_AME_MMVE64_T_X : RVInstBOSCAMEMove_T_X<0b011, "mmve64.t.x">;

  def BOSC_AME_MMVE8_X_A  : RVInstBOSCAMEMove_X_A<0b100, "mmve8.x.a">;
  def BOSC_AME_MMVE16_X_A : RVInstBOSCAMEMove_X_A<0b101, "mmve16.x.a">;
  def BOSC_AME_MMVE32_X_A : RVInstBOSCAMEMove_X_A<0b110, "mmve32.x.a">;
  def BOSC_AME_MMVE64_X_A : RVInstBOSCAMEMove_X_A<0b111, "mmve64.x.a">;

  def BOSC_AME_MMVE8_A_X  : RVInstBOSCAMEMove_A_X<0b100, "mmve8.a.x">;
  def BOSC_AME_MMVE16_A_X : RVInstBOSCAMEMove_A_X<0b101, "mmve16.a.x">;
  def BOSC_AME_MMVE32_A_X : RVInstBOSCAMEMove_A_X<0b110, "mmve32.a.x">;
  def BOSC_AME_MMVE64_A_X : RVInstBOSCAMEMove_A_X<0b111, "mmve64.a.x">;
}

// Data Move Instructions between Matrix and Float-point
class RVInstBOSCAMEFMove_F_T<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs GPR:$rd), (ins TileReg:$ms1, GPR:$rs2),
                        opcodestr, "$rd, $ms1, $rs2"> {
  bits<5> rs2;
  bits<5> ms1;
  bits<5> rd;

  let funct6 = 0b000110;
  let di     = 0b0;
  let rs1    = ms1;
  let eew    = eew_val;
}

class RVInstBOSCAMEFMove_T_F<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs TileReg:$md), (ins GPR:$rs1, GPR:$rs2),
                        opcodestr, "$md, $rs1, $rs2"> {
  bits<5> rs2;
  bits<5> rs1;
  bits<5> md;

  let funct6 = 0b000110;
  let di     = 0b1;
  let eew    = eew_val;
  let rd     = md;
}

class RVInstBOSCAMEFMove_F_A<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs GPR:$rd), (ins AccReg:$ms1, GPR:$rs2),
                        opcodestr, "$rd, $ms1, $rs2"> {
  bits<5> rs2;
  bits<5> ms1;
  bits<5> rd;

  let funct6 = 0b000110;
  let di     = 0b0;
  let rs1    = ms1;
  let eew    = eew_val;
}

class RVInstBOSCAMEFMove_A_F<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs AccReg:$md), (ins GPR:$rs1, GPR:$rs2),
                        opcodestr, "$md, $rs1, $rs2"> {
  bits<5> rs2;
  bits<5> rs1;
  bits<5> md;

  let funct6 = 0b000110;
  let di     = 0b1;
  let eew    = eew_val;
  let rd     = md;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {

  def BOSC_AME_MFMVE8_F_T  : RVInstBOSCAMEFMove_F_T<0b000, "mfmve8.f.t">;
  def BOSC_AME_MFMVE16_F_T : RVInstBOSCAMEFMove_F_T<0b001, "mfmve16.f.t">;
  def BOSC_AME_MFMVE32_F_T : RVInstBOSCAMEFMove_F_T<0b010, "mfmve32.f.t">;
  def BOSC_AME_MFMVE64_F_T : RVInstBOSCAMEFMove_F_T<0b011, "mfmve64.f.t">;

  def BOSC_AME_MFMVE8_T_F  : RVInstBOSCAMEFMove_T_F<0b000, "mfmve8.t.f">;
  def BOSC_AME_MFMVE16_T_F : RVInstBOSCAMEFMove_T_F<0b001, "mfmve16.t.f">;
  def BOSC_AME_MFMVE32_T_F : RVInstBOSCAMEFMove_T_F<0b010, "mfmve32.t.f">;
  def BOSC_AME_MFMVE64_T_F : RVInstBOSCAMEFMove_T_F<0b011, "mfmve64.t.f">;

  def BOSC_AME_MFMVE8_F_A  : RVInstBOSCAMEFMove_F_A<0b100, "mfmve8.f.a">;
  def BOSC_AME_MFMVE16_F_A : RVInstBOSCAMEFMove_F_A<0b101, "mfmve16.f.a">;
  def BOSC_AME_MFMVE32_F_A : RVInstBOSCAMEFMove_F_A<0b110, "mfmve32.f.a">;
  def BOSC_AME_MFMVE64_F_A : RVInstBOSCAMEFMove_F_A<0b111, "mfmve64.f.a">;

  def BOSC_AME_MFMVE8_A_F  : RVInstBOSCAMEFMove_A_F<0b100, "mfmve8.a.f">;
  def BOSC_AME_MFMVE16_A_F : RVInstBOSCAMEFMove_A_F<0b101, "mfmve16.a.f">;
  def BOSC_AME_MFMVE32_A_F : RVInstBOSCAMEFMove_A_F<0b110, "mfmve32.a.f">;
  def BOSC_AME_MFMVE64_A_F : RVInstBOSCAMEFMove_A_F<0b111, "mfmve64.a.f">;
}

// Data Broadcast Instructions
class RVInstBOSCAMEBroadcastTile<bits<5> rs2_val, bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs TileReg:$md), (ins TileReg:$ms1),
                        opcodestr, "$md, $ms1"> {
  bits<5> md;
  bits<5> ms1;

  let funct6 = 0b000111;
  let di     = 0b0;
  let rs2    = rs2_val;
  let rs1    = ms1;
  let eew    = eew_val;
  let rd     = md;
  let opcode = 0b1110111;
}

class RVInstBOSCAMEBroadcastAcc<bits<5> rs2_val, bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs AccReg:$md), (ins AccReg:$ms1),
                        opcodestr, "$md, $ms1"> {
  bits<5> md;
  bits<5> ms1;

  let funct6 = 0b000111;
  let di     = 0b0;
  let rs2    = rs2_val;
  let rs1    = ms1;
  let eew    = eew_val;
  let rd     = md;
  let opcode = 0b1110111;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  // Row broadcast from TileReg/AccReg
  def BOSC_AME_MBCAR_M : RVInstBOSCAMEBroadcastTile<0b10001, 0b000, "mbcar.m">;
  def BOSC_AME_MBCBR_M : RVInstBOSCAMEBroadcastTile<0b10010, 0b000, "mbcbr.m">;
  def BOSC_AME_MBCCR_M : RVInstBOSCAMEBroadcastAcc<0b10000, 0b000, "mbccr.m">;
  // Column broadcast from TileReg
  def BOSC_AME_MBCACE8_M  : RVInstBOSCAMEBroadcastTile<0b10101, 0b000, "mbcace8.m">;
  def BOSC_AME_MBCACE16_M : RVInstBOSCAMEBroadcastTile<0b10101, 0b001, "mbcace16.m">;
  def BOSC_AME_MBCACE32_M : RVInstBOSCAMEBroadcastTile<0b10101, 0b010, "mbcace32.m">;
  def BOSC_AME_MBCACE64_M : RVInstBOSCAMEBroadcastTile<0b10101, 0b011, "mbcace64.m">;
  // Column broadcast from TileReg
  def BOSC_AME_MBCBCE8_M  : RVInstBOSCAMEBroadcastTile<0b10110, 0b000, "mbcbce8.m">;
  def BOSC_AME_MBCBCE16_M : RVInstBOSCAMEBroadcastTile<0b10110, 0b001, "mbcbce16.m">;
  def BOSC_AME_MBCBCE32_M : RVInstBOSCAMEBroadcastTile<0b10110, 0b010, "mbcbce32.m">;
  def BOSC_AME_MBCBCE64_M : RVInstBOSCAMEBroadcastTile<0b10110, 0b011, "mbcbce64.m">;
  // Column broadcast from AccReg
  def BOSC_AME_MBCCCE8_M  : RVInstBOSCAMEBroadcastAcc<0b10100, 0b000, "mbccce8.m">;
  def BOSC_AME_MBCCCE16_M : RVInstBOSCAMEBroadcastAcc<0b10100, 0b001, "mbccce16.m">;
  def BOSC_AME_MBCCCE32_M : RVInstBOSCAMEBroadcastAcc<0b10100, 0b010, "mbccce32.m">;
  def BOSC_AME_MBCCCE64_M : RVInstBOSCAMEBroadcastAcc<0b10100, 0b011, "mbccce64.m">;
  // Element broadcast from TileReg
  def BOSC_AME_MBCAEE8_M  : RVInstBOSCAMEBroadcastTile<0b11001, 0b000, "mbcaee8.m">;
  def BOSC_AME_MBCAEE16_M : RVInstBOSCAMEBroadcastTile<0b11001, 0b001, "mbcaee16.m">;
  def BOSC_AME_MBCAEE32_M : RVInstBOSCAMEBroadcastTile<0b11001, 0b010, "mbcaee32.m">;
  def BOSC_AME_MBCAEE64_M : RVInstBOSCAMEBroadcastTile<0b11001, 0b011, "mbcaee64.m">;
  // Element broadcast from TileReg
  def BOSC_AME_MBCBEE8_M  : RVInstBOSCAMEBroadcastTile<0b11010, 0b000, "mbcbee8.m">;
  def BOSC_AME_MBCBEE16_M : RVInstBOSCAMEBroadcastTile<0b11010, 0b001, "mbcbee16.m">;
  def BOSC_AME_MBCBEE32_M : RVInstBOSCAMEBroadcastTile<0b11010, 0b010, "mbcbee32.m">;
  def BOSC_AME_MBCBEE64_M : RVInstBOSCAMEBroadcastTile<0b11010, 0b011, "mbcbee64.m">;
  // Element broadcast from AccReg
  def BOSC_AME_MBCCEE8_M  : RVInstBOSCAMEBroadcastAcc<0b11000, 0b000, "mbccee8.m">;
  def BOSC_AME_MBCCEE16_M : RVInstBOSCAMEBroadcastAcc<0b11000, 0b001, "mbccee16.m">;
  def BOSC_AME_MBCCEE32_M : RVInstBOSCAMEBroadcastAcc<0b11000, 0b010, "mbccee32.m">;
  def BOSC_AME_MBCCEE64_M : RVInstBOSCAMEBroadcastAcc<0b11000, 0b011, "mbccee64.m">;
}

// Matrix Transpose Instructions
class RVInstBOSCAMETransposeTile<bits<5> rs2_val, bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs TileReg:$md), (ins TileReg:$ms1),
                        opcodestr, "$md, $ms1"> {
  bits<5> md;
  bits<5> ms1;

  let funct6 = 0b000111;
  let di     = 0b0;
  let rs2    = rs2_val;
  let rs1    = ms1;
  let eew    = eew_val;
  let rd     = md;
  let opcode = 0b1110111;
}

class RVInstBOSCAMETransposeAcc<bits<5> rs2_val, bits<3> eew_val, string opcodestr>
    : RVInstBOSCAMEMove<(outs AccReg:$md), (ins AccReg:$ms1),
                        opcodestr, "$md, $ms1"> {
  bits<5> md;
  bits<5> ms1;

  let funct6 = 0b000111;
  let di     = 0b0;
  let rs2    = rs2_val;
  let rs1    = ms1;
  let eew    = eew_val;
  let rd     = md;
  let opcode = 0b1110111;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  // Transpose square matrix of tile register
  def BOSC_AME_MTAE8_M : RVInstBOSCAMETransposeTile<0b11101, 0b000, "mtae8.m">;
  def BOSC_AME_MTAE16_M : RVInstBOSCAMETransposeTile<0b11101, 0b001, "mtae16.m">;
  def BOSC_AME_MTAE32_M : RVInstBOSCAMETransposeTile<0b11101, 0b010, "mtae32.m">;
  def BOSC_AME_MTAE64_M : RVInstBOSCAMETransposeTile<0b11101, 0b011, "mtae64.m">;
  // Transpose square matrix of tile register
  def BOSC_AME_MTBE8_M : RVInstBOSCAMETransposeTile<0b11110, 0b000, "mtbe8.m">;
  def BOSC_AME_MTBE16_M : RVInstBOSCAMETransposeTile<0b11110, 0b001, "mtbe16.m">;
  def BOSC_AME_MTBE32_M : RVInstBOSCAMETransposeTile<0b11110, 0b010, "mtbe32.m">;
  def BOSC_AME_MTBE64_M : RVInstBOSCAMETransposeTile<0b11110, 0b011, "mtbe64.m">;
  // Transpose square matrix of acc register
  def BOSC_AME_MTCE8_M : RVInstBOSCAMETransposeAcc<0b11100, 0b000, "mtce8.m">;
  def BOSC_AME_MTCE16_M : RVInstBOSCAMETransposeAcc<0b11100, 0b001, "mtce16.m">;
  def BOSC_AME_MTCE32_M : RVInstBOSCAMETransposeAcc<0b11100, 0b010, "mtce32.m">;
  def BOSC_AME_MTCE64_M : RVInstBOSCAMETransposeAcc<0b11100, 0b011, "mtce64.m">;
}

//===----------------------------------------------------------------------===//
// AME Zero/Initialize Instructions
//===----------------------------------------------------------------------===//
// mzero - Zero out an accumulation register
// Useful for initializing before accumulation loop

class BOSC_AME_MZERO<string opcodestr>
    : RVInst<(outs AccReg:$md), (ins),
               opcodestr, "$md", [], InstFormatOther> {
  bits<4> md;

  let Inst{6-0} = 0b1110111;
  let Inst{10-7} = md;
  let Inst{14-12} = 0b101;     // funct3 for arithmetic
  let Inst{19-15} = 0b00000;
  let Inst{24-20} = 0b00000;
  let Inst{25} = 0;
  let Inst{31-26} = 0b000000;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def BOSC_AME_MZERO_M : BOSC_AME_MZERO<"mzero">;

//===----------------------------------------------------------------------===//
// Pseudo Instructions for Index-Based Operations
//===----------------------------------------------------------------------===//
// These pseudo instructions accept i64 indices and have AsmString for direct
// assembly output. This allows the pseudo instructions to be printed directly
// without needing complex expansion logic.
//
// For load/store/mma instructions, the AsmString hardcodes "acc" and "tr"
// prefixes so that indices 0-7 are printed as acc0-acc7 and tr0-tr7.
//===----------------------------------------------------------------------===//

// Pseudo configuration instruction
let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 0, mayStore = 0,
    isCodeGenOnly = 1 in {
  // Register configuration instructions
  def BOSC_AME_MSETTYPE_PSEUDO : Pseudo<(outs GPR:$rd), (ins GPR:$rs1), []> {
    let AsmString = "msettype\t$rd, $rs1";
  }
  def BOSC_AME_MSETTILEM_PSEUDO : Pseudo<(outs GPR:$rd), (ins GPR:$rs1), []> {
    let AsmString = "msettilem\t$rd, $rs1";
  }
  def BOSC_AME_MSETTILEN_PSEUDO : Pseudo<(outs GPR:$rd), (ins GPR:$rs1), []> {
    let AsmString = "msettilen\t$rd, $rs1";
  }
  def BOSC_AME_MSETTILEK_PSEUDO : Pseudo<(outs GPR:$rd), (ins GPR:$rs1), []> {
    let AsmString = "msettilek\t$rd, $rs1";
  }
  // Immediate configuration instructions
  def BOSC_AME_MSETTYPEI_PSEUDO : Pseudo<(outs GPR:$rd), (ins i64imm:$imm), []> {
    let AsmString = "msettypei\t$rd, $imm";
  }
  def BOSC_AME_MSETTYPEHI_PSEUDO : Pseudo<(outs GPR:$rd), (ins i64imm:$imm), []> {
    let AsmString = "msettypehi\t$rd, $imm";
  }
  def BOSC_AME_MSETTILEMI_PSEUDO : Pseudo<(outs GPR:$rd), (ins i64imm:$imm), []> {
    let AsmString = "msettilemi\t$rd, $imm";
  }
  def BOSC_AME_MSETTILENI_PSEUDO : Pseudo<(outs GPR:$rd), (ins i64imm:$imm), []> {
    let AsmString = "msettileni\t$rd, $imm";
  }
  def BOSC_AME_MSETTILEKI_PSEUDO : Pseudo<(outs GPR:$rd), (ins i64imm:$imm), []> {
    let AsmString = "msettileki\t$rd, $imm";
  }
}

// Pseudo load instructions
let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 1, mayStore = 0,
    isCodeGenOnly = 1 in {
  def BOSC_AME_MLAE8_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlae8.m\ttr$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLAE16_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlae16.m\ttr$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLAE32_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlae32.m\ttr$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLAE64_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlae64.m\ttr$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLATE8_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlate8.m\ttr$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLATE16_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlate16.m\ttr$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLATE32_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlate32.m\ttr$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLATE64_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlate64.m\ttr$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLTRE8_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mltre8.m\ttr$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLTRE16_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mltre16.m\ttr$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLTRE32_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mltre32.m\ttr$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLTRE64_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mltre64.m\ttr$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLBE8_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlbe8.m\ttr$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLBE16_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlbe16.m\ttr$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLBE32_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlbe32.m\ttr$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLBE64_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlbe64.m\ttr$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLBTE8_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlbte8.m\ttr$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLBTE16_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlbte16.m\ttr$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLBTE32_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlbte32.m\ttr$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLBTE64_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlbte64.m\ttr$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLCE8_M_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlce8.m\tacc$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLCE16_M_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlce16.m\tacc$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLCE32_M_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlce32.m\tacc$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLCE64_M_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlce64.m\tacc$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLCTE8_M_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlcte8.m\tacc$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLCTE16_M_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlcte16.m\tacc$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLCTE32_M_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlcte32.m\tacc$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLCTE64_M_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlcte64.m\tacc$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLACCE8_M_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlacce8.m\tacc$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLACCE16_M_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlacce16.m\tacc$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLACCE32_M_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlacce32.m\tacc$md, ($rs1), $rs2";
  }
  def BOSC_AME_MLACCE64_M_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$md, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mlacce64.m\tacc$md, ($rs1), $rs2";
  }
}

// Pseudo store instructions
let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 0, mayStore = 1,
    isCodeGenOnly = 1 in {
  def BOSC_AME_MSAE8_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "msae8.m\ttr$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSAE16_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "msae16.m\ttr$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSAE32_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "msae32.m\ttr$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSAE64_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "msae64.m\ttr$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSATE8_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "msate8.m\ttr$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSATE16_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "msate16.m\ttr$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSATE32_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "msate32.m\ttr$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSATE64_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "msate64.m\ttr$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSTRE8_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mstre8.m\ttr$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSTRE16_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mstre16.m\ttr$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSTRE32_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mstre32.m\ttr$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSTRE64_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mstre64.m\ttr$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSBE8_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "msbe8.m\ttr$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSBE16_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "msbe16.m\ttr$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSBE32_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "msbe32.m\ttr$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSBE64_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "msbe64.m\ttr$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSBTE8_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "msbte8.m\ttr$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSBTE16_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "msbte16.m\ttr$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSBTE32_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "msbte32.m\ttr$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSBTE64_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "msbte64.m\ttr$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSCE8_M_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "msce8.m\tacc$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSCE16_M_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "msce16.m\tacc$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSCE32_M_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "msce32.m\tacc$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSCE64_M_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "msce64.m\tacc$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSCTE8_M_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mscte8.m\tacc$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSCTE16_M_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mscte16.m\tacc$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSCTE32_M_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mscte32.m\tacc$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSCTE64_M_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "mscte64.m\tacc$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSACCE8_M_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "msacce8.m\tacc$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSACCE16_M_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "msacce16.m\tacc$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSACCE32_M_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "msacce32.m\tacc$ms3, ($rs1), $rs2";
  }
  def BOSC_AME_MSACCE64_M_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
    let AsmString = "msacce64.m\tacc$ms3, ($rs1), $rs2";
  }
}

// Pseudo data move instructions
// Matrix to Matrix (Pure)
class RVInstBOSCAMEMove_T_T_Pseudo<string opcodestr>
    : Pseudo<(outs), (ins BOSCAMETileIndex:$md, BOSCAMETileIndex:$ms1), []> {
  let AsmString = opcodestr # "\ttr$md, tr$ms1";
}
class RVInstBOSCAMEMove_A_A_Pseudo<string opcodestr>
    : Pseudo<(outs), (ins BOSCAMEAccIndex:$md, BOSCAMEAccIndex:$ms1), []> {
  let AsmString = opcodestr # "\tacc$md, acc$ms1";
}

// Matrix to Matrix (with GPR index)
class RVInstBOSCAMEMove_A_T_Pseudo<string opcodestr>
    : Pseudo<(outs), (ins BOSCAMEAccIndex:$md, BOSCAMETileIndex:$ms1, GPR:$rs2), []> {
  let AsmString = opcodestr # "\tacc$md, tr$ms1, $rs2";
}
class RVInstBOSCAMEMove_T_A_Pseudo<string opcodestr>
    : Pseudo<(outs), (ins BOSCAMETileIndex:$md, BOSCAMEAccIndex:$ms1, GPR:$rs2), []> {
  let AsmString = opcodestr # "\ttr$md, acc$ms1, $rs2";
}

// Matrix to Matrix (with Immediate index)
class RVInstBOSCAMEMoveImm_A_T_Pseudo<string opcodestr>
    : Pseudo<(outs), (ins BOSCAMEAccIndex:$md, BOSCAMETileIndex:$ms1, uimm5:$imm), []> {
  let AsmString = opcodestr # "\tacc$md, tr$ms1, $imm";
}
class RVInstBOSCAMEMoveImm_T_A_Pseudo<string opcodestr>
    : Pseudo<(outs), (ins BOSCAMETileIndex:$md, BOSCAMEAccIndex:$ms1, uimm5:$imm), []> {
  let AsmString = opcodestr # "\ttr$md, acc$ms1, $imm";
}

// Matrix to Scalar GPR (Notice: rd is a real output GPR)
class RVInstBOSCAMEMove_X_T_Pseudo<string opcodestr>
    : Pseudo<(outs GPR:$rd), (ins BOSCAMETileIndex:$ms1, GPR:$rs2), []> {
  let AsmString = opcodestr # "\t$rd, tr$ms1, $rs2";
}
class RVInstBOSCAMEMove_X_A_Pseudo<string opcodestr>
    : Pseudo<(outs GPR:$rd), (ins BOSCAMEAccIndex:$ms1, GPR:$rs2), []> {
  let AsmString = opcodestr # "\t$rd, acc$ms1, $rs2";
}

// Scalar GPR to Matrix
class RVInstBOSCAMEMove_T_X_Pseudo<string opcodestr>
    : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
  let AsmString = opcodestr # "\ttr$md, $rs1, $rs2";
}
class RVInstBOSCAMEMove_A_X_Pseudo<string opcodestr>
    : Pseudo<(outs), (ins BOSCAMEAccIndex:$md, GPR:$rs1, GPR:$rs2), []> {
  let AsmString = opcodestr # "\tacc$md, $rs1, $rs2";
}

// Pseudo classes for Float-point Matrix moves
class RVInstBOSCAMEMove_F_T_Pseudo<string opcodestr>
    : Pseudo<(outs GPR:$rd), (ins BOSCAMETileIndex:$ms1, GPR:$rs2), []> {
  let AsmString = opcodestr # "\t$rd, tr$ms1, $rs2";
}

class RVInstBOSCAMEMove_T_F_Pseudo<string opcodestr>
    : Pseudo<(outs), (ins  BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
  let AsmString = opcodestr # "\ttr$md, $rs1, $rs2";
}

class RVInstBOSCAMEMove_F_A_Pseudo<string opcodestr>
    : Pseudo<(outs GPR:$rd), (ins BOSCAMEAccIndex:$ms1, GPR:$rs2), []> {
  let AsmString = opcodestr # "\t$rd, acc$ms1, $rs2";
}

class RVInstBOSCAMEMove_A_F_Pseudo<string opcodestr>
    : Pseudo<(outs), (ins BOSCAMEAccIndex:$md, GPR:$rs1, GPR:$rs2), []> {
  let AsmString = opcodestr # "\tacc$md, $rs1, $rs2";
}

class RVInstBOSCAMEBroadcastTile_Pseudo<string opcodestr>
      : RVInstBOSCAMEMove_T_T_Pseudo<opcodestr>;
class RVInstBOSCAMEBroadcastAcc_Pseudo<string opcodestr>
      : RVInstBOSCAMEMove_A_A_Pseudo<opcodestr>;
class RVInstBOSCAMETransposeTile_Pseudo<string opcodestr>
      : RVInstBOSCAMEMove_T_T_Pseudo<opcodestr>;
class RVInstBOSCAMETransposeAcc_Pseudo<string opcodestr>
      : RVInstBOSCAMEMove_A_A_Pseudo<opcodestr>;

let Predicates = [HasBuddyExt], hasSideEffects = 1, isCodeGenOnly = 1 in {
  //----------------------------------------------------------------------
  // Matrix-Matrix Move Pseudos
  //----------------------------------------------------------------------
  def BOSC_AME_MMVE8_T_T_PSEUDO  : RVInstBOSCAMEMove_T_T_Pseudo<"mmve8.t.t">;
  def BOSC_AME_MMVE16_T_T_PSEUDO : RVInstBOSCAMEMove_T_T_Pseudo<"mmve16.t.t">;
  def BOSC_AME_MMVE32_T_T_PSEUDO : RVInstBOSCAMEMove_T_T_Pseudo<"mmve32.t.t">;
  def BOSC_AME_MMVE64_T_T_PSEUDO : RVInstBOSCAMEMove_T_T_Pseudo<"mmve64.t.t">;

  def BOSC_AME_MMVE8_A_A_PSEUDO  : RVInstBOSCAMEMove_A_A_Pseudo<"mmve8.a.a">;
  def BOSC_AME_MMVE16_A_A_PSEUDO : RVInstBOSCAMEMove_A_A_Pseudo<"mmve16.a.a">;
  def BOSC_AME_MMVE32_A_A_PSEUDO : RVInstBOSCAMEMove_A_A_Pseudo<"mmve32.a.a">;
  def BOSC_AME_MMVE64_A_A_PSEUDO : RVInstBOSCAMEMove_A_A_Pseudo<"mmve64.a.a">;

  def BOSC_AME_MMVE8_A_T_PSEUDO  : RVInstBOSCAMEMove_A_T_Pseudo<"mmve8.a.t">;
  def BOSC_AME_MMVE16_A_T_PSEUDO : RVInstBOSCAMEMove_A_T_Pseudo<"mmve16.a.t">;
  def BOSC_AME_MMVE32_A_T_PSEUDO : RVInstBOSCAMEMove_A_T_Pseudo<"mmve32.a.t">;
  def BOSC_AME_MMVE64_A_T_PSEUDO : RVInstBOSCAMEMove_A_T_Pseudo<"mmve64.a.t">;

  def BOSC_AME_MMVE8_T_A_PSEUDO  : RVInstBOSCAMEMove_T_A_Pseudo<"mmve8.t.a">;
  def BOSC_AME_MMVE16_T_A_PSEUDO : RVInstBOSCAMEMove_T_A_Pseudo<"mmve16.t.a">;
  def BOSC_AME_MMVE32_T_A_PSEUDO : RVInstBOSCAMEMove_T_A_Pseudo<"mmve32.t.a">;
  def BOSC_AME_MMVE64_T_A_PSEUDO : RVInstBOSCAMEMove_T_A_Pseudo<"mmve64.t.a">;

  def BOSC_AME_MMVIE8_A_T_PSEUDO  : RVInstBOSCAMEMoveImm_A_T_Pseudo<"mmvie8.a.t">;
  def BOSC_AME_MMVIE16_A_T_PSEUDO : RVInstBOSCAMEMoveImm_A_T_Pseudo<"mmvie16.a.t">;
  def BOSC_AME_MMVIE32_A_T_PSEUDO : RVInstBOSCAMEMoveImm_A_T_Pseudo<"mmvie32.a.t">;
  def BOSC_AME_MMVIE64_A_T_PSEUDO : RVInstBOSCAMEMoveImm_A_T_Pseudo<"mmvie64.a.t">;

  def BOSC_AME_MMVIE8_T_A_PSEUDO  : RVInstBOSCAMEMoveImm_T_A_Pseudo<"mmvie8.t.a">;
  def BOSC_AME_MMVIE16_T_A_PSEUDO : RVInstBOSCAMEMoveImm_T_A_Pseudo<"mmvie16.t.a">;
  def BOSC_AME_MMVIE32_T_A_PSEUDO : RVInstBOSCAMEMoveImm_T_A_Pseudo<"mmvie32.t.a">;
  def BOSC_AME_MMVIE64_T_A_PSEUDO : RVInstBOSCAMEMoveImm_T_A_Pseudo<"mmvie64.t.a">;
  //----------------------------------------------------------------------
  // Int / Float Matrix-Scalar Move Pseudos
  //----------------------------------------------------------------------
  def BOSC_AME_MMVE8_X_T_PSEUDO  : RVInstBOSCAMEMove_X_T_Pseudo<"mmve8.x.t">;
  def BOSC_AME_MMVE16_X_T_PSEUDO : RVInstBOSCAMEMove_X_T_Pseudo<"mmve16.x.t">;
  def BOSC_AME_MMVE32_X_T_PSEUDO : RVInstBOSCAMEMove_X_T_Pseudo<"mmve32.x.t">;
  def BOSC_AME_MMVE64_X_T_PSEUDO : RVInstBOSCAMEMove_X_T_Pseudo<"mmve64.x.t">;

  def BOSC_AME_MMVE8_T_X_PSEUDO  : RVInstBOSCAMEMove_T_X_Pseudo<"mmve8.t.x">;
  def BOSC_AME_MMVE16_T_X_PSEUDO : RVInstBOSCAMEMove_T_X_Pseudo<"mmve16.t.x">;
  def BOSC_AME_MMVE32_T_X_PSEUDO : RVInstBOSCAMEMove_T_X_Pseudo<"mmve32.t.x">;
  def BOSC_AME_MMVE64_T_X_PSEUDO : RVInstBOSCAMEMove_T_X_Pseudo<"mmve64.t.x">;

  def BOSC_AME_MMVE8_X_A_PSEUDO  : RVInstBOSCAMEMove_X_A_Pseudo<"mmve8.x.a">;
  def BOSC_AME_MMVE16_X_A_PSEUDO : RVInstBOSCAMEMove_X_A_Pseudo<"mmve16.x.a">;
  def BOSC_AME_MMVE32_X_A_PSEUDO : RVInstBOSCAMEMove_X_A_Pseudo<"mmve32.x.a">;
  def BOSC_AME_MMVE64_X_A_PSEUDO : RVInstBOSCAMEMove_X_A_Pseudo<"mmve64.x.a">;

  def BOSC_AME_MMVE8_A_X_PSEUDO  : RVInstBOSCAMEMove_A_X_Pseudo<"mmve8.a.x">;
  def BOSC_AME_MMVE16_A_X_PSEUDO : RVInstBOSCAMEMove_A_X_Pseudo<"mmve16.a.x">;
  def BOSC_AME_MMVE32_A_X_PSEUDO : RVInstBOSCAMEMove_A_X_Pseudo<"mmve32.a.x">;
  def BOSC_AME_MMVE64_A_X_PSEUDO : RVInstBOSCAMEMove_A_X_Pseudo<"mmve64.a.x">;

  def BOSC_AME_MFMVE8_F_T_PSEUDO  : RVInstBOSCAMEMove_F_T_Pseudo<"mfmve8.f.t">;
  def BOSC_AME_MFMVE16_F_T_PSEUDO : RVInstBOSCAMEMove_F_T_Pseudo<"mfmve16.f.t">;
  def BOSC_AME_MFMVE32_F_T_PSEUDO : RVInstBOSCAMEMove_F_T_Pseudo<"mfmve32.f.t">;
  def BOSC_AME_MFMVE64_F_T_PSEUDO : RVInstBOSCAMEMove_F_T_Pseudo<"mfmve64.f.t">;

  def BOSC_AME_MFMVE8_T_F_PSEUDO  : RVInstBOSCAMEMove_T_F_Pseudo<"mfmve8.t.f">;
  def BOSC_AME_MFMVE16_T_F_PSEUDO : RVInstBOSCAMEMove_T_F_Pseudo<"mfmve16.t.f">;
  def BOSC_AME_MFMVE32_T_F_PSEUDO : RVInstBOSCAMEMove_T_F_Pseudo<"mfmve32.t.f">;
  def BOSC_AME_MFMVE64_T_F_PSEUDO : RVInstBOSCAMEMove_T_F_Pseudo<"mfmve64.t.f">;

  def BOSC_AME_MFMVE8_F_A_PSEUDO  : RVInstBOSCAMEMove_F_A_Pseudo<"mfmve8.f.a">;
  def BOSC_AME_MFMVE16_F_A_PSEUDO : RVInstBOSCAMEMove_F_A_Pseudo<"mfmve16.f.a">;
  def BOSC_AME_MFMVE32_F_A_PSEUDO : RVInstBOSCAMEMove_F_A_Pseudo<"mfmve32.f.a">;
  def BOSC_AME_MFMVE64_F_A_PSEUDO : RVInstBOSCAMEMove_F_A_Pseudo<"mfmve64.f.a">;

  def BOSC_AME_MFMVE8_A_F_PSEUDO  : RVInstBOSCAMEMove_A_F_Pseudo<"mfmve8.a.f">;
  def BOSC_AME_MFMVE16_A_F_PSEUDO : RVInstBOSCAMEMove_A_F_Pseudo<"mfmve16.a.f">;
  def BOSC_AME_MFMVE32_A_F_PSEUDO : RVInstBOSCAMEMove_A_F_Pseudo<"mfmve32.a.f">;
  def BOSC_AME_MFMVE64_A_F_PSEUDO : RVInstBOSCAMEMove_A_F_Pseudo<"mfmve64.a.f">;
  //----------------------------------------------------------------------
  // Broadcast Pseudos
  //----------------------------------------------------------------------
  def BOSC_AME_MBCAR_M_PSEUDO : RVInstBOSCAMEBroadcastTile_Pseudo<"mbcar.m">;
  def BOSC_AME_MBCBR_M_PSEUDO : RVInstBOSCAMEBroadcastTile_Pseudo<"mbcbr.m">;
  def BOSC_AME_MBCCR_M_PSEUDO : RVInstBOSCAMEBroadcastAcc_Pseudo<"mbccr.m">;

  def BOSC_AME_MBCACE8_M_PSEUDO  : RVInstBOSCAMEBroadcastTile_Pseudo<"mbcace8.m">;
  def BOSC_AME_MBCACE16_M_PSEUDO : RVInstBOSCAMEBroadcastTile_Pseudo<"mbcace16.m">;
  def BOSC_AME_MBCACE32_M_PSEUDO : RVInstBOSCAMEBroadcastTile_Pseudo<"mbcace32.m">;
  def BOSC_AME_MBCACE64_M_PSEUDO : RVInstBOSCAMEBroadcastTile_Pseudo<"mbcace64.m">;

  def BOSC_AME_MBCBCE8_M_PSEUDO  : RVInstBOSCAMEBroadcastTile_Pseudo<"mbcbce8.m">;
  def BOSC_AME_MBCBCE16_M_PSEUDO : RVInstBOSCAMEBroadcastTile_Pseudo<"mbcbce16.m">;
  def BOSC_AME_MBCBCE32_M_PSEUDO : RVInstBOSCAMEBroadcastTile_Pseudo<"mbcbce32.m">;
  def BOSC_AME_MBCBCE64_M_PSEUDO : RVInstBOSCAMEBroadcastTile_Pseudo<"mbcbce64.m">;

  def BOSC_AME_MBCCCE8_M_PSEUDO  : RVInstBOSCAMEBroadcastAcc_Pseudo<"mbccce8.m">;
  def BOSC_AME_MBCCCE16_M_PSEUDO : RVInstBOSCAMEBroadcastAcc_Pseudo<"mbccce16.m">;
  def BOSC_AME_MBCCCE32_M_PSEUDO : RVInstBOSCAMEBroadcastAcc_Pseudo<"mbccce32.m">;
  def BOSC_AME_MBCCCE64_M_PSEUDO : RVInstBOSCAMEBroadcastAcc_Pseudo<"mbccce64.m">;

  def BOSC_AME_MBCAEE8_M_PSEUDO  : RVInstBOSCAMEBroadcastTile_Pseudo<"mbcaee8.m">;
  def BOSC_AME_MBCAEE16_M_PSEUDO : RVInstBOSCAMEBroadcastTile_Pseudo<"mbcaee16.m">;
  def BOSC_AME_MBCAEE32_M_PSEUDO : RVInstBOSCAMEBroadcastTile_Pseudo<"mbcaee32.m">;
  def BOSC_AME_MBCAEE64_M_PSEUDO : RVInstBOSCAMEBroadcastTile_Pseudo<"mbcaee64.m">;

  def BOSC_AME_MBCBEE8_M_PSEUDO  : RVInstBOSCAMEBroadcastTile_Pseudo<"mbcbee8.m">;
  def BOSC_AME_MBCBEE16_M_PSEUDO : RVInstBOSCAMEBroadcastTile_Pseudo<"mbcbee16.m">;
  def BOSC_AME_MBCBEE32_M_PSEUDO : RVInstBOSCAMEBroadcastTile_Pseudo<"mbcbee32.m">;
  def BOSC_AME_MBCBEE64_M_PSEUDO : RVInstBOSCAMEBroadcastTile_Pseudo<"mbcbee64.m">;

  def BOSC_AME_MBCCEE8_M_PSEUDO  : RVInstBOSCAMEBroadcastAcc_Pseudo<"mbccee8.m">;
  def BOSC_AME_MBCCEE16_M_PSEUDO : RVInstBOSCAMEBroadcastAcc_Pseudo<"mbccee16.m">;
  def BOSC_AME_MBCCEE32_M_PSEUDO : RVInstBOSCAMEBroadcastAcc_Pseudo<"mbccee32.m">;
  def BOSC_AME_MBCCEE64_M_PSEUDO : RVInstBOSCAMEBroadcastAcc_Pseudo<"mbccee64.m">;
  //----------------------------------------------------------------------
  // Transpose Pseudos
  //----------------------------------------------------------------------
  def BOSC_AME_MTAE8_M_PSEUDO  : RVInstBOSCAMETransposeTile_Pseudo<"mtae8.m">;
  def BOSC_AME_MTAE16_M_PSEUDO : RVInstBOSCAMETransposeTile_Pseudo<"mtae16.m">;
  def BOSC_AME_MTAE32_M_PSEUDO : RVInstBOSCAMETransposeTile_Pseudo<"mtae32.m">;
  def BOSC_AME_MTAE64_M_PSEUDO : RVInstBOSCAMETransposeTile_Pseudo<"mtae64.m">;

  def BOSC_AME_MTBE8_M_PSEUDO  : RVInstBOSCAMETransposeTile_Pseudo<"mtbe8.m">;
  def BOSC_AME_MTBE16_M_PSEUDO : RVInstBOSCAMETransposeTile_Pseudo<"mtbe16.m">;
  def BOSC_AME_MTBE32_M_PSEUDO : RVInstBOSCAMETransposeTile_Pseudo<"mtbe32.m">;
  def BOSC_AME_MTBE64_M_PSEUDO : RVInstBOSCAMETransposeTile_Pseudo<"mtbe64.m">;

  def BOSC_AME_MTCE8_M_PSEUDO  : RVInstBOSCAMETransposeAcc_Pseudo<"mtce8.m">;
  def BOSC_AME_MTCE16_M_PSEUDO : RVInstBOSCAMETransposeAcc_Pseudo<"mtce16.m">;
  def BOSC_AME_MTCE32_M_PSEUDO : RVInstBOSCAMETransposeAcc_Pseudo<"mtce32.m">;
  def BOSC_AME_MTCE64_M_PSEUDO : RVInstBOSCAMETransposeAcc_Pseudo<"mtce64.m">;
}

// Pseudo element-wise instruction
// Base Pseudo class for 3-operand element-wise instructions (e.g., md, ms1, ms2)
class RVInstBOSCAMEEleWise3_Pseudo<string opcodestr>
    : Pseudo<(outs), (ins BOSCAMEAccIndex:$md, BOSCAMEAccIndex:$ms1, BOSCAMEAccIndex:$ms2), []> {
  let AsmString = opcodestr # "\tacc$md, acc$ms1, acc$ms2";
}

// Base Pseudo class for 2-operand element-wise instructions (e.g., md, ms1 for sqrt)
class RVInstBOSCAMEEleWise2_Pseudo<string opcodestr>
    : Pseudo<(outs), (ins BOSCAMEAccIndex:$md, BOSCAMEAccIndex:$ms1), []> {
  let AsmString = opcodestr # "\tacc$md, acc$ms1";
}

let Predicates = [HasBuddyExt], hasSideEffects = 1, isCodeGenOnly = 1 in {
  //===--------------------------------------------------------------------===//
  // 1. ADD / SUB - No Widen
  //===--------------------------------------------------------------------===//
  def BOSC_AME_MADDU_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"maddu.mm">;
  def BOSC_AME_MADDU_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"maddu.hb.mm">;
  def BOSC_AME_MADDU_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"maddu.b.mm">;
  def BOSC_AME_MADDU_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"maddu.h.mm">;
  def BOSC_AME_MADDU_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"maddu.w.mm">;
  def BOSC_AME_MADDU_DW_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"maddu.dw.mm">;

  def BOSC_AME_MSADDU_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"msaddu.mm">;
  def BOSC_AME_MSADDU_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"msaddu.hb.mm">;
  def BOSC_AME_MSADDU_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msaddu.b.mm">;
  def BOSC_AME_MSADDU_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msaddu.h.mm">;
  def BOSC_AME_MSADDU_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msaddu.w.mm">;
  def BOSC_AME_MSADDU_DW_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"msaddu.dw.mm">;

  def BOSC_AME_MADD_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"madd.mm">;
  def BOSC_AME_MADD_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"madd.hb.mm">;
  def BOSC_AME_MADD_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"madd.b.mm">;
  def BOSC_AME_MADD_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"madd.h.mm">;
  def BOSC_AME_MADD_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"madd.w.mm">;
  def BOSC_AME_MADD_DW_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"madd.dw.mm">;

  def BOSC_AME_MSADD_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"msadd.mm">;
  def BOSC_AME_MSADD_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"msadd.hb.mm">;
  def BOSC_AME_MSADD_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msadd.b.mm">;
  def BOSC_AME_MSADD_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msadd.h.mm">;
  def BOSC_AME_MSADD_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msadd.w.mm">;
  def BOSC_AME_MSADD_DW_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"msadd.dw.mm">;

  def BOSC_AME_MFADD_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mfadd.mm">;
  def BOSC_AME_MFADD_CF_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mfadd.cf.mm">;
  def BOSC_AME_MFADD_HF_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mfadd.hf.mm">;
  def BOSC_AME_MFADD_F_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mfadd.f.mm">;
  def BOSC_AME_MFADD_D_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mfadd.d.mm">;

  def BOSC_AME_MSUBU_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"msubu.mm">;
  def BOSC_AME_MSUBU_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"msubu.hb.mm">;
  def BOSC_AME_MSUBU_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msubu.b.mm">;
  def BOSC_AME_MSUBU_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msubu.h.mm">;
  def BOSC_AME_MSUBU_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msubu.w.mm">;
  def BOSC_AME_MSUBU_DW_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"msubu.dw.mm">;

  def BOSC_AME_MSSUBU_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mssubu.mm">;
  def BOSC_AME_MSSUBU_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mssubu.hb.mm">;
  def BOSC_AME_MSSUBU_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mssubu.b.mm">;
  def BOSC_AME_MSSUBU_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mssubu.h.mm">;
  def BOSC_AME_MSSUBU_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mssubu.w.mm">;
  def BOSC_AME_MSSUBU_DW_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mssubu.dw.mm">;

  def BOSC_AME_MSUB_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"msub.mm">;
  def BOSC_AME_MSUB_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"msub.hb.mm">;
  def BOSC_AME_MSUB_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msub.b.mm">;
  def BOSC_AME_MSUB_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msub.h.mm">;
  def BOSC_AME_MSUB_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msub.w.mm">;
  def BOSC_AME_MSUB_DW_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"msub.dw.mm">;

  def BOSC_AME_MSSUB_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mssub.mm">;
  def BOSC_AME_MSSUB_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mssub.hb.mm">;
  def BOSC_AME_MSSUB_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mssub.b.mm">;
  def BOSC_AME_MSSUB_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mssub.h.mm">;
  def BOSC_AME_MSSUB_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mssub.w.mm">;
  def BOSC_AME_MSSUB_DW_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mssub.dw.mm">;

  def BOSC_AME_MFSUB_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mfsub.mm">;
  def BOSC_AME_MFSUB_CF_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mfsub.cf.mm">;
  def BOSC_AME_MFSUB_HF_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mfsub.hf.mm">;
  def BOSC_AME_MFSUB_F_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mfsub.f.mm">;
  def BOSC_AME_MFSUB_D_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mfsub.d.mm">;
  //===--------------------------------------------------------------------===//
  // 2. ADD / SUB - Double Widen
  //===--------------------------------------------------------------------===//
  def BOSC_AME_MWADDU_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mwaddu.mm">;
  def BOSC_AME_MWADDU_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mwaddu.hb.mm">;
  def BOSC_AME_MWADDU_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mwaddu.b.mm">;
  def BOSC_AME_MWADDU_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mwaddu.h.mm">;
  def BOSC_AME_MWADDU_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mwaddu.w.mm">;

  def BOSC_AME_MWADD_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mwadd.mm">;
  def BOSC_AME_MWADD_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mwadd.hb.mm">;
  def BOSC_AME_MWADD_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mwadd.b.mm">;
  def BOSC_AME_MWADD_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mwadd.h.mm">;
  def BOSC_AME_MWADD_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mwadd.w.mm">;

  def BOSC_AME_MFWADD_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mfwadd.mm">;
  def BOSC_AME_MFWADD_CF_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mfwadd.cf.mm">;
  def BOSC_AME_MFWADD_HF_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mfwadd.hf.mm">;
  def BOSC_AME_MFWADD_F_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mfwadd.f.mm">;

  def BOSC_AME_MWSUBU_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mwsubu.mm">;
  def BOSC_AME_MWSUBU_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mwsubu.hb.mm">;
  def BOSC_AME_MWSUBU_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mwsubu.b.mm">;
  def BOSC_AME_MWSUBU_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mwsubu.h.mm">;
  def BOSC_AME_MWSUBU_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mwsubu.w.mm">;

  def BOSC_AME_MWSUB_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mwsub.mm">;
  def BOSC_AME_MWSUB_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mwsub.hb.mm">;
  def BOSC_AME_MWSUB_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mwsub.b.mm">;
  def BOSC_AME_MWSUB_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mwsub.h.mm">;
  def BOSC_AME_MWSUB_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mwsub.w.mm">;

  def BOSC_AME_MFWSUB_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mfwsub.mm">;
  def BOSC_AME_MFWSUB_CF_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mfwsub.cf.mm">;
  def BOSC_AME_MFWSUB_HF_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mfwsub.hf.mm">;
  def BOSC_AME_MFWSUB_F_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mfwsub.f.mm">;
  //===--------------------------------------------------------------------===//
  // 3. MIN / MAX - No Widen
  //===--------------------------------------------------------------------===//
  def BOSC_AME_MMINU_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mminu.mm">;
  def BOSC_AME_MMINU_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mminu.hb.mm">;
  def BOSC_AME_MMINU_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mminu.b.mm">;
  def BOSC_AME_MMINU_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mminu.h.mm">;
  def BOSC_AME_MMINU_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mminu.w.mm">;
  def BOSC_AME_MMINU_DW_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mminu.dw.mm">;

  def BOSC_AME_MMAXU_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mmaxu.mm">;
  def BOSC_AME_MMAXU_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mmaxu.hb.mm">;
  def BOSC_AME_MMAXU_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mmaxu.b.mm">;
  def BOSC_AME_MMAXU_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mmaxu.h.mm">;
  def BOSC_AME_MMAXU_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mmaxu.w.mm">;
  def BOSC_AME_MMAXU_DW_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mmaxu.dw.mm">;

  def BOSC_AME_MMIN_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mmin.mm">;
  def BOSC_AME_MMIN_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mmin.hb.mm">;
  def BOSC_AME_MMIN_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mmin.b.mm">;
  def BOSC_AME_MMIN_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mmin.h.mm">;
  def BOSC_AME_MMIN_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mmin.w.mm">;
  def BOSC_AME_MMIN_DW_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mmin.dw.mm">;

  def BOSC_AME_MMAX_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mmax.mm">;
  def BOSC_AME_MMAX_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mmax.hb.mm">;
  def BOSC_AME_MMAX_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mmax.b.mm">;
  def BOSC_AME_MMAX_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mmax.h.mm">;
  def BOSC_AME_MMAX_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mmax.w.mm">;
  def BOSC_AME_MMAX_DW_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mmax.dw.mm">;

  def BOSC_AME_MFMIN_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mfmin.mm">;
  def BOSC_AME_MFMIN_CF_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mfmin.cf.mm">;
  def BOSC_AME_MFMIN_HF_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mfmin.hf.mm">;
  def BOSC_AME_MFMIN_F_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mfmin.f.mm">;
  def BOSC_AME_MFMIN_D_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mfmin.d.mm">;

  def BOSC_AME_MFMAX_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mfmax.mm">;
  def BOSC_AME_MFMAX_CF_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mfmax.cf.mm">;
  def BOSC_AME_MFMAX_HF_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mfmax.hf.mm">;
  def BOSC_AME_MFMAX_F_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mfmax.f.mm">;
  def BOSC_AME_MFMAX_D_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mfmax.d.mm">;
  //===--------------------------------------------------------------------===//
  // 4. MUL / MULH / DIV - No Widen
  //===--------------------------------------------------------------------===//
  def BOSC_AME_MSMULU_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"msmulu.mm">;
  def BOSC_AME_MSMULU_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"msmulu.hb.mm">;
  def BOSC_AME_MSMULU_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msmulu.b.mm">;
  def BOSC_AME_MSMULU_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msmulu.h.mm">;
  def BOSC_AME_MSMULU_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msmulu.w.mm">;
  def BOSC_AME_MSMULU_DW_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"msmulu.dw.mm">;

  def BOSC_AME_MMUL_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mmul.mm">;
  def BOSC_AME_MMUL_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mmul.hb.mm">;
  def BOSC_AME_MMUL_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mmul.b.mm">;
  def BOSC_AME_MMUL_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mmul.h.mm">;
  def BOSC_AME_MMUL_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mmul.w.mm">;
  def BOSC_AME_MMUL_DW_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mmul.dw.mm">;

  def BOSC_AME_MSMUL_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"msmul.mm">;
  def BOSC_AME_MSMUL_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"msmul.hb.mm">;
  def BOSC_AME_MSMUL_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msmul.b.mm">;
  def BOSC_AME_MSMUL_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msmul.h.mm">;
  def BOSC_AME_MSMUL_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msmul.w.mm">;
  def BOSC_AME_MSMUL_DW_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"msmul.dw.mm">;

  def BOSC_AME_MFMUL_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mfmul.mm">;
  def BOSC_AME_MFMUL_CF_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mfmul.cf.mm">;
  def BOSC_AME_MFMUL_HF_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mfmul.hf.mm">;
  def BOSC_AME_MFMUL_F_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mfmul.f.mm">;
  def BOSC_AME_MFMUL_D_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mfmul.d.mm">;

  def BOSC_AME_MMULHU_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mmulhu.mm">;
  def BOSC_AME_MMULHU_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mmulhu.hb.mm">;
  def BOSC_AME_MMULHU_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mmulhu.b.mm">;
  def BOSC_AME_MMULHU_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mmulhu.h.mm">;
  def BOSC_AME_MMULHU_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mmulhu.w.mm">;
  def BOSC_AME_MMULHU_DW_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mmulhu.dw.mm">;

  def BOSC_AME_MMULH_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mmulh.mm">;
  def BOSC_AME_MMULH_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mmulh.hb.mm">;
  def BOSC_AME_MMULH_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mmulh.b.mm">;
  def BOSC_AME_MMULH_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mmulh.h.mm">;
  def BOSC_AME_MMULH_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mmulh.w.mm">;
  def BOSC_AME_MMULH_DW_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mmulh.dw.mm">;

  def BOSC_AME_MMULHSU_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mmulhsu.mm">;
  def BOSC_AME_MMULHSU_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mmulhsu.hb.mm">;
  def BOSC_AME_MMULHSU_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mmulhsu.b.mm">;
  def BOSC_AME_MMULHSU_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mmulhsu.h.mm">;
  def BOSC_AME_MMULHSU_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mmulhsu.w.mm">;
  def BOSC_AME_MMULHSU_DW_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mmulhsu.dw.mm">;

  def BOSC_AME_MSMULSU_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"msmulsu.mm">;
  def BOSC_AME_MSMULSU_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"msmulsu.hb.mm">;
  def BOSC_AME_MSMULSU_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msmulsu.b.mm">;
  def BOSC_AME_MSMULSU_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msmulsu.h.mm">;
  def BOSC_AME_MSMULSU_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msmulsu.w.mm">;
  def BOSC_AME_MSMULSU_DW_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"msmulsu.dw.mm">;

  def BOSC_AME_MFDIV_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mfdiv.mm">;
  def BOSC_AME_MFDIV_CF_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mfdiv.cf.mm">;
  def BOSC_AME_MFDIV_HF_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mfdiv.hf.mm">;
  def BOSC_AME_MFDIV_F_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mfdiv.f.mm">;
  def BOSC_AME_MFDIV_D_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mfdiv.d.mm">;
  //===--------------------------------------------------------------------===//
  // 5. MUL - Double Widen
  //===--------------------------------------------------------------------===//
  def BOSC_AME_MWMULU_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mwmulu.mm">;
  def BOSC_AME_MWMULU_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mwmulu.hb.mm">;
  def BOSC_AME_MWMULU_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mwmulu.b.mm">;
  def BOSC_AME_MWMULU_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mwmulu.h.mm">;
  def BOSC_AME_MWMULU_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mwmulu.w.mm">;

  def BOSC_AME_MWMUL_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mwmul.mm">;
  def BOSC_AME_MWMUL_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mwmul.hb.mm">;
  def BOSC_AME_MWMUL_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mwmul.b.mm">;
  def BOSC_AME_MWMUL_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mwmul.h.mm">;
  def BOSC_AME_MWMUL_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mwmul.w.mm">;

  def BOSC_AME_MWMULSU_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mwmulsu.mm">;
  def BOSC_AME_MWMULSU_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mwmulsu.hb.mm">;
  def BOSC_AME_MWMULSU_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mwmulsu.b.mm">;
  def BOSC_AME_MWMULSU_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mwmulsu.h.mm">;
  def BOSC_AME_MWMULSU_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mwmulsu.w.mm">;

  def BOSC_AME_MFWMUL_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"mfwmul.mm">;
  def BOSC_AME_MFWMUL_CF_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mfwmul.cf.mm">;
  def BOSC_AME_MFWMUL_HF_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mfwmul.hf.mm">;
  def BOSC_AME_MFWMUL_F_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mfwmul.f.mm">;
  //===--------------------------------------------------------------------===//
  // 6. LOGIC & SHIFT - No Widen
  //===--------------------------------------------------------------------===//
  def BOSC_AME_MAND_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mand.mm">;
  def BOSC_AME_MOR_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"mor.mm">;
  def BOSC_AME_MXOR_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"mxor.mm">;

  def BOSC_AME_MSLL_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"msll.mm">;
  def BOSC_AME_MSLL_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"msll.hb.mm">;
  def BOSC_AME_MSLL_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msll.b.mm">;
  def BOSC_AME_MSLL_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msll.h.mm">;
  def BOSC_AME_MSLL_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msll.w.mm">;
  def BOSC_AME_MSLL_DW_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"msll.dw.mm">;

  def BOSC_AME_MSRL_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"msrl.mm">;
  def BOSC_AME_MSRL_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"msrl.hb.mm">;
  def BOSC_AME_MSRL_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msrl.b.mm">;
  def BOSC_AME_MSRL_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msrl.h.mm">;
  def BOSC_AME_MSRL_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msrl.w.mm">;
  def BOSC_AME_MSRL_DW_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"msrl.dw.mm">;

  def BOSC_AME_MSRA_MM_PSEUDO    : RVInstBOSCAMEEleWise3_Pseudo<"msra.mm">;
  def BOSC_AME_MSRA_HB_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"msra.hb.mm">;
  def BOSC_AME_MSRA_B_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msra.b.mm">;
  def BOSC_AME_MSRA_H_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msra.h.mm">;
  def BOSC_AME_MSRA_W_MM_PSEUDO  : RVInstBOSCAMEEleWise3_Pseudo<"msra.w.mm">;
  def BOSC_AME_MSRA_DW_MM_PSEUDO : RVInstBOSCAMEEleWise3_Pseudo<"msra.dw.mm">;
  //===--------------------------------------------------------------------===//
  // 7. SQRT - No Widen (Unary, 2-Operand)
  //===--------------------------------------------------------------------===//
  def BOSC_AME_MFSQRT_MM_PSEUDO    : RVInstBOSCAMEEleWise2_Pseudo<"mfsqrt.mm">;
  def BOSC_AME_MFSQRT_CF_MM_PSEUDO : RVInstBOSCAMEEleWise2_Pseudo<"mfsqrt.cf.mm">;
  def BOSC_AME_MFSQRT_HF_MM_PSEUDO : RVInstBOSCAMEEleWise2_Pseudo<"mfsqrt.hf.mm">;
  def BOSC_AME_MFSQRT_F_MM_PSEUDO  : RVInstBOSCAMEEleWise2_Pseudo<"mfsqrt.f.mm">;
  def BOSC_AME_MFSQRT_D_MM_PSEUDO  : RVInstBOSCAMEEleWise2_Pseudo<"mfsqrt.d.mm">;
}

// Pseudo matrix multiply instruction
// md: AccReg index (0-7), ms1/ms2: TileReg indices (0-7)
// Base Pseudo class for all MMA instructions
class RVInstBOSCAMEMMA_Pseudo<string opcodestr>
    : Pseudo<(outs), (ins BOSCAMEAccIndex:$md, BOSCAMETileIndex:$ms1, BOSCAMETileIndex:$ms2), []> {
  let AsmString = opcodestr # "\tacc$md, tr$ms1, tr$ms2";
}

let Predicates = [HasBuddyExt], hasSideEffects = 1, isCodeGenOnly = 1 in {
  //===--------------------------------------------------------------------===//
  // 1. No-widen matrix multiply-accumulate
  //===--------------------------------------------------------------------===//
  // int
  def BOSC_AME_MMA_MM_PSEUDO      : RVInstBOSCAMEMMA_Pseudo<"mma.mm">;
  def BOSC_AME_MMA_H_MM_PSEUDO    : RVInstBOSCAMEMMA_Pseudo<"mma.h.mm">;
  def BOSC_AME_MMA_W_MM_PSEUDO    : RVInstBOSCAMEMMA_Pseudo<"mma.w.mm">;
  def BOSC_AME_MMA_DW_MM_PSEUDO   : RVInstBOSCAMEMMA_Pseudo<"mma.dw.mm">;
  // int, saturated
  def BOSC_AME_MSMA_MM_PSEUDO     : RVInstBOSCAMEMMA_Pseudo<"msma.mm">;
  def BOSC_AME_MSMA_H_MM_PSEUDO   : RVInstBOSCAMEMMA_Pseudo<"msma.h.mm">;
  def BOSC_AME_MSMA_W_MM_PSEUDO   : RVInstBOSCAMEMMA_Pseudo<"msma.w.mm">;
  def BOSC_AME_MSMA_DW_MM_PSEUDO  : RVInstBOSCAMEMMA_Pseudo<"msma.dw.mm">;
  // uint
  def BOSC_AME_MMAU_MM_PSEUDO     : RVInstBOSCAMEMMA_Pseudo<"mmau.mm">;
  def BOSC_AME_MMAU_H_MM_PSEUDO   : RVInstBOSCAMEMMA_Pseudo<"mmau.h.mm">;
  def BOSC_AME_MMAU_W_MM_PSEUDO   : RVInstBOSCAMEMMA_Pseudo<"mmau.w.mm">;
  def BOSC_AME_MMAU_DW_MM_PSEUDO  : RVInstBOSCAMEMMA_Pseudo<"mmau.dw.mm">;
  // uint, saturated
  def BOSC_AME_MSMAU_MM_PSEUDO    : RVInstBOSCAMEMMA_Pseudo<"msmau.mm">;
  def BOSC_AME_MSMAU_H_MM_PSEUDO  : RVInstBOSCAMEMMA_Pseudo<"msmau.h.mm">;
  def BOSC_AME_MSMAU_W_MM_PSEUDO  : RVInstBOSCAMEMMA_Pseudo<"msmau.w.mm">;
  def BOSC_AME_MSMAU_DW_MM_PSEUDO : RVInstBOSCAMEMMA_Pseudo<"msmau.dw.mm">;
  // float point
  def BOSC_AME_MFMA_MM_PSEUDO     : RVInstBOSCAMEMMA_Pseudo<"mfma.mm">;
  def BOSC_AME_MFMA_HF_MM_PSEUDO  : RVInstBOSCAMEMMA_Pseudo<"mfma.hf.mm">;
  def BOSC_AME_MFMA_F_MM_PSEUDO   : RVInstBOSCAMEMMA_Pseudo<"mfma.f.mm">;
  def BOSC_AME_MFMA_D_MM_PSEUDO   : RVInstBOSCAMEMMA_Pseudo<"mfma.d.mm">;
  //===--------------------------------------------------------------------===//
  // 2. Double-widen matrix multiply-accumulate
  //===--------------------------------------------------------------------===//
  // uint
  def BOSC_AME_MWMAU_MM_PSEUDO    : RVInstBOSCAMEMMA_Pseudo<"mwmau.mm">;
  def BOSC_AME_MWMAU_H_MM_PSEUDO  : RVInstBOSCAMEMMA_Pseudo<"mwmau.h.mm">;
  def BOSC_AME_MWMAU_W_MM_PSEUDO  : RVInstBOSCAMEMMA_Pseudo<"mwmau.w.mm">;
  // uint, saturated
  def BOSC_AME_MSWMAU_MM_PSEUDO   : RVInstBOSCAMEMMA_Pseudo<"mswmau.mm">;
  def BOSC_AME_MSWMAU_H_MM_PSEUDO : RVInstBOSCAMEMMA_Pseudo<"mswmau.h.mm">;
  def BOSC_AME_MSWMAU_W_MM_PSEUDO : RVInstBOSCAMEMMA_Pseudo<"mswmau.w.mm">;
  // int
  def BOSC_AME_MWMA_MM_PSEUDO     : RVInstBOSCAMEMMA_Pseudo<"mwma.mm">;
  def BOSC_AME_MWMA_H_MM_PSEUDO   : RVInstBOSCAMEMMA_Pseudo<"mwma.h.mm">;
  def BOSC_AME_MWMA_W_MM_PSEUDO   : RVInstBOSCAMEMMA_Pseudo<"mwma.w.mm">;
  // int, saturated
  def BOSC_AME_MSWMA_MM_PSEUDO    : RVInstBOSCAMEMMA_Pseudo<"mswma.mm">;
  def BOSC_AME_MSWMA_H_MM_PSEUDO  : RVInstBOSCAMEMMA_Pseudo<"mswma.h.mm">;
  def BOSC_AME_MSWMA_W_MM_PSEUDO  : RVInstBOSCAMEMMA_Pseudo<"mswma.w.mm">;
  // float point
  def BOSC_AME_MFWMA_MM_PSEUDO    : RVInstBOSCAMEMMA_Pseudo<"mfwma.mm">;
  def BOSC_AME_MFWMA_CF_MM_PSEUDO : RVInstBOSCAMEMMA_Pseudo<"mfwma.cf.mm">;
  def BOSC_AME_MFWMA_HF_MM_PSEUDO : RVInstBOSCAMEMMA_Pseudo<"mfwma.hf.mm">;
  def BOSC_AME_MFWMA_F_MM_PSEUDO  : RVInstBOSCAMEMMA_Pseudo<"mfwma.f.mm">;
  //===--------------------------------------------------------------------===//
  // 3. Quad-widen matrix multiply-accumulate
  //===--------------------------------------------------------------------===//
  // uint
  def BOSC_AME_MQMAU_MM_PSEUDO    : RVInstBOSCAMEMMA_Pseudo<"mqmau.mm">;
  def BOSC_AME_MQMAU_B_MM_PSEUDO  : RVInstBOSCAMEMMA_Pseudo<"mqmau.b.mm">;
  // uint, saturated
  def BOSC_AME_MSQMAU_MM_PSEUDO   : RVInstBOSCAMEMMA_Pseudo<"msqmau.mm">;
  def BOSC_AME_MSQMAU_B_MM_PSEUDO : RVInstBOSCAMEMMA_Pseudo<"msqmau.b.mm">;
  // int
  def BOSC_AME_MQMA_MM_PSEUDO     : RVInstBOSCAMEMMA_Pseudo<"mqma.mm">;
  def BOSC_AME_MQMA_B_MM_PSEUDO   : RVInstBOSCAMEMMA_Pseudo<"mqma.b.mm">;
  // int, saturated
  def BOSC_AME_MSQMA_MM_PSEUDO    : RVInstBOSCAMEMMA_Pseudo<"msqma.mm">;
  def BOSC_AME_MSQMA_B_MM_PSEUDO  : RVInstBOSCAMEMMA_Pseudo<"msqma.b.mm">;
  // float point
  def BOSC_AME_MFQMA_MM_PSEUDO    : RVInstBOSCAMEMMA_Pseudo<"mfqma.mm">;
  def BOSC_AME_MFQMA_CF_MM_PSEUDO : RVInstBOSCAMEMMA_Pseudo<"mfqma.cf.mm">;
  //===--------------------------------------------------------------------===//
  // 4. Oct-widen matrix multiply-accumulate
  //===--------------------------------------------------------------------===//
  // uint
  def BOSC_AME_MOMAU_MM_PSEUDO    : RVInstBOSCAMEMMA_Pseudo<"momau.mm">;
  def BOSC_AME_MOMAU_HB_MM_PSEUDO : RVInstBOSCAMEMMA_Pseudo<"momau.hb.mm">;
  // uint, saturated
  def BOSC_AME_MSOMAU_MM_PSEUDO   : RVInstBOSCAMEMMA_Pseudo<"msomau.mm">;
  def BOSC_AME_MSOMAU_HB_MM_PSEUDO: RVInstBOSCAMEMMA_Pseudo<"msomau.hb.mm">;
  // int
  def BOSC_AME_MOMA_MM_PSEUDO     : RVInstBOSCAMEMMA_Pseudo<"moma.mm">;
  def BOSC_AME_MOMA_HB_MM_PSEUDO  : RVInstBOSCAMEMMA_Pseudo<"moma.hb.mm">;
  // int, saturated
  def BOSC_AME_MSOMA_MM_PSEUDO    : RVInstBOSCAMEMMA_Pseudo<"msoma.mm">;
  def BOSC_AME_MSOMA_HB_MM_PSEUDO : RVInstBOSCAMEMMA_Pseudo<"msoma.hb.mm">;
}

//===----------------------------------------------------------------------===//
// Pattern Matching for 32-bit New Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasBuddyExt] in {
  // Configuration immediate instructions
  def : Pat<(i64 (int_riscv_buddy_bosc_msettypei timm:$imm)),
            (BOSC_AME_MSETTYPEI_PSEUDO timm:$imm)>;
  def : Pat<(i64 (int_riscv_buddy_bosc_msettypehi timm:$imm)),
            (BOSC_AME_MSETTYPEHI_PSEUDO timm:$imm)>;
  def : Pat<(i64 (int_riscv_buddy_bosc_msettilemi timm:$imm)),
            (BOSC_AME_MSETTILEMI_PSEUDO timm:$imm)>;
  def : Pat<(i64 (int_riscv_buddy_bosc_msettileni timm:$imm)),
            (BOSC_AME_MSETTILENI_PSEUDO timm:$imm)>;
  def : Pat<(i64 (int_riscv_buddy_bosc_msettileki timm:$imm)),
            (BOSC_AME_MSETTILEKI_PSEUDO timm:$imm)>;
  // Register versions
  def : Pat<(i64 (int_riscv_buddy_bosc_msettype  GPR:$rs1)),
            (BOSC_AME_MSETTYPE_PSEUDO  GPR:$rs1)>;
  def : Pat<(i64 (int_riscv_buddy_bosc_msettilem GPR:$rs1)),
            (BOSC_AME_MSETTILEM_PSEUDO GPR:$rs1)>;
  def : Pat<(i64 (int_riscv_buddy_bosc_msettilen GPR:$rs1)),
            (BOSC_AME_MSETTILEN_PSEUDO GPR:$rs1)>;
  def : Pat<(i64 (int_riscv_buddy_bosc_msettilek GPR:$rs1)),
            (BOSC_AME_MSETTILEK_PSEUDO GPR:$rs1)>;

  // Load instructions
  // Matrix A
  def : Pat<(int_riscv_buddy_bosc_mlae8_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLAE8_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlae16_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLAE16_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlae32_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLAE32_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlae64_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLAE64_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlate8_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLATE8_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlate16_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLATE16_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlate32_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLATE32_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlate64_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLATE64_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mltre8_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLTRE8_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mltre16_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLTRE16_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mltre32_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLTRE32_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mltre64_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLTRE64_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  // Matrix B
  def : Pat<(int_riscv_buddy_bosc_mlbe8_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLBE8_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlbe16_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLBE16_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlbe32_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLBE32_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlbe64_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLBE64_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlbte8_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLBTE8_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlbte16_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLBTE16_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlbte32_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLBTE32_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlbte64_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLBTE64_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  // Matrix C (Accumulator)
  def : Pat<(int_riscv_buddy_bosc_mlce8_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLCE8_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlce16_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLCE16_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlce32_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLCE32_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlce64_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLCE64_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlcte8_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLCTE8_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlcte16_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLCTE16_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlcte32_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLCTE32_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlcte64_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLCTE64_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlacce8_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLACCE8_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlacce16_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLACCE16_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlacce32_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLACCE32_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlacce64_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLACCE64_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;

  // Store instruction
  // Matrix A
  def : Pat<(int_riscv_buddy_bosc_msae8_m timm:$ms1, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSAE8_M_PSEUDO timm:$ms1, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_msae16_m timm:$ms1, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSAE16_M_PSEUDO timm:$ms1, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_msae32_m timm:$ms1, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSAE32_M_PSEUDO timm:$ms1, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_msae64_m timm:$ms1, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSAE64_M_PSEUDO timm:$ms1, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_msate8_m timm:$ms1, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSATE8_M_PSEUDO timm:$ms1, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_msate16_m timm:$ms1, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSATE16_M_PSEUDO timm:$ms1, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_msate32_m timm:$ms1, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSATE32_M_PSEUDO timm:$ms1, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_msate64_m timm:$ms1, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSATE64_M_PSEUDO timm:$ms1, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mstre8_m timm:$ms1, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSTRE8_M_PSEUDO timm:$ms1, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mstre16_m timm:$ms1, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSTRE16_M_PSEUDO timm:$ms1, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mstre32_m timm:$ms1, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSTRE32_M_PSEUDO timm:$ms1, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mstre64_m timm:$ms1, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSTRE64_M_PSEUDO timm:$ms1, GPR:$rs1, GPR:$rs2)>;
  // Matrix B
  def : Pat<(int_riscv_buddy_bosc_msbe8_m timm:$ms2, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSBE8_M_PSEUDO timm:$ms2, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_msbe16_m timm:$ms2, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSBE16_M_PSEUDO timm:$ms2, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_msbe32_m timm:$ms2, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSBE32_M_PSEUDO timm:$ms2, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_msbe64_m timm:$ms2, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSBE64_M_PSEUDO timm:$ms2, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_msbte8_m timm:$ms2, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSBTE8_M_PSEUDO timm:$ms2, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_msbte16_m timm:$ms2, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSBTE16_M_PSEUDO timm:$ms2, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_msbte32_m timm:$ms2, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSBTE32_M_PSEUDO timm:$ms2, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_msbte64_m timm:$ms2, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSBTE64_M_PSEUDO timm:$ms2, GPR:$rs1, GPR:$rs2)>;
  // Matrix C (Accumulator)
  def : Pat<(int_riscv_buddy_bosc_msce8_m timm:$ms3, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSCE8_M_PSEUDO timm:$ms3, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_msce16_m timm:$ms3, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSCE16_M_PSEUDO timm:$ms3, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_msce32_m timm:$ms3, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSCE32_M_PSEUDO timm:$ms3, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_msce64_m timm:$ms3, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSCE64_M_PSEUDO timm:$ms3, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mscte8_m timm:$ms3, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSCTE8_M_PSEUDO timm:$ms3, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mscte16_m timm:$ms3, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSCTE16_M_PSEUDO timm:$ms3, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mscte32_m timm:$ms3, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSCTE32_M_PSEUDO timm:$ms3, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mscte64_m timm:$ms3, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSCTE64_M_PSEUDO timm:$ms3, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_msacce8_m timm:$ms3, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSACCE8_M_PSEUDO timm:$ms3, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_msacce16_m timm:$ms3, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSACCE16_M_PSEUDO timm:$ms3, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_msacce32_m timm:$ms3, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSACCE32_M_PSEUDO timm:$ms3, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_msacce64_m timm:$ms3, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSACCE64_M_PSEUDO timm:$ms3, GPR:$rs1, GPR:$rs2)>;

  // Data move instructions
  //----------------------------------------------------------------------
  // 1. Matrix-Matrix Move Pseudos (Pure Matrix)
  //----------------------------------------------------------------------
  def : Pat<(int_riscv_buddy_bosc_mmve8_t_t  timm:$md, timm:$ms1),
            (BOSC_AME_MMVE8_T_T_PSEUDO  timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mmve16_t_t timm:$md, timm:$ms1),
            (BOSC_AME_MMVE16_T_T_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mmve32_t_t timm:$md, timm:$ms1),
            (BOSC_AME_MMVE32_T_T_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mmve64_t_t timm:$md, timm:$ms1),
            (BOSC_AME_MMVE64_T_T_PSEUDO timm:$md, timm:$ms1)>;

  def : Pat<(int_riscv_buddy_bosc_mmve8_a_a  timm:$md, timm:$ms1),
            (BOSC_AME_MMVE8_A_A_PSEUDO  timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mmve16_a_a timm:$md, timm:$ms1),
            (BOSC_AME_MMVE16_A_A_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mmve32_a_a timm:$md, timm:$ms1),
            (BOSC_AME_MMVE32_A_A_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mmve64_a_a timm:$md, timm:$ms1),
            (BOSC_AME_MMVE64_A_A_PSEUDO timm:$md, timm:$ms1)>;
  //----------------------------------------------------------------------
  // 2. Matrix-Matrix Move (with GPR Index)
  //----------------------------------------------------------------------
  def : Pat<(int_riscv_buddy_bosc_mmve8_a_t  timm:$md, timm:$ms1, i64:$rs2),
            (BOSC_AME_MMVE8_A_T_PSEUDO  timm:$md, timm:$ms1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mmve16_a_t timm:$md, timm:$ms1, i64:$rs2),
            (BOSC_AME_MMVE16_A_T_PSEUDO timm:$md, timm:$ms1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mmve32_a_t timm:$md, timm:$ms1, i64:$rs2),
            (BOSC_AME_MMVE32_A_T_PSEUDO timm:$md, timm:$ms1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mmve64_a_t timm:$md, timm:$ms1, i64:$rs2),
            (BOSC_AME_MMVE64_A_T_PSEUDO timm:$md, timm:$ms1, GPR:$rs2)>;

  def : Pat<(int_riscv_buddy_bosc_mmve8_t_a  timm:$md, timm:$ms1, i64:$rs2),
            (BOSC_AME_MMVE8_T_A_PSEUDO  timm:$md, timm:$ms1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mmve16_t_a timm:$md, timm:$ms1, i64:$rs2),
            (BOSC_AME_MMVE16_T_A_PSEUDO timm:$md, timm:$ms1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mmve32_t_a timm:$md, timm:$ms1, i64:$rs2),
            (BOSC_AME_MMVE32_T_A_PSEUDO timm:$md, timm:$ms1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mmve64_t_a timm:$md, timm:$ms1, i64:$rs2),
            (BOSC_AME_MMVE64_T_A_PSEUDO timm:$md, timm:$ms1, GPR:$rs2)>;
  //----------------------------------------------------------------------
  // 3. Matrix-Matrix Move (with Immediate Index)
  //----------------------------------------------------------------------
  def : Pat<(int_riscv_buddy_bosc_mmvie8_a_t  timm:$md, timm:$ms1, imm:$imm),
            (BOSC_AME_MMVIE8_A_T_PSEUDO  timm:$md, timm:$ms1, imm:$imm)>;
  def : Pat<(int_riscv_buddy_bosc_mmvie16_a_t timm:$md, timm:$ms1, imm:$imm),
            (BOSC_AME_MMVIE16_A_T_PSEUDO timm:$md, timm:$ms1, imm:$imm)>;
  def : Pat<(int_riscv_buddy_bosc_mmvie32_a_t timm:$md, timm:$ms1, imm:$imm),
            (BOSC_AME_MMVIE32_A_T_PSEUDO timm:$md, timm:$ms1, imm:$imm)>;
  def : Pat<(int_riscv_buddy_bosc_mmvie64_a_t timm:$md, timm:$ms1, imm:$imm),
            (BOSC_AME_MMVIE64_A_T_PSEUDO timm:$md, timm:$ms1, imm:$imm)>;

  def : Pat<(int_riscv_buddy_bosc_mmvie8_t_a  timm:$md, timm:$ms1, imm:$imm),
            (BOSC_AME_MMVIE8_T_A_PSEUDO  timm:$md, timm:$ms1, imm:$imm)>;
  def : Pat<(int_riscv_buddy_bosc_mmvie16_t_a timm:$md, timm:$ms1, imm:$imm),
            (BOSC_AME_MMVIE16_T_A_PSEUDO timm:$md, timm:$ms1, imm:$imm)>;
  def : Pat<(int_riscv_buddy_bosc_mmvie32_t_a timm:$md, timm:$ms1, imm:$imm),
            (BOSC_AME_MMVIE32_T_A_PSEUDO timm:$md, timm:$ms1, imm:$imm)>;
  def : Pat<(int_riscv_buddy_bosc_mmvie64_t_a timm:$md, timm:$ms1, imm:$imm),
            (BOSC_AME_MMVIE64_T_A_PSEUDO timm:$md, timm:$ms1, imm:$imm)>;
  //----------------------------------------------------------------------
  // 4. Matrix to Scalar Move (Returns GPR, matches i64 in IR)
  //----------------------------------------------------------------------
  def : Pat<(i64 (int_riscv_buddy_bosc_mmve8_x_t  timm:$ms1, i64:$rs2)),
            (BOSC_AME_MMVE8_X_T_PSEUDO  timm:$ms1, GPR:$rs2)>;
  def : Pat<(i64 (int_riscv_buddy_bosc_mmve16_x_t timm:$ms1, i64:$rs2)),
            (BOSC_AME_MMVE16_X_T_PSEUDO timm:$ms1, GPR:$rs2)>;
  def : Pat<(i64 (int_riscv_buddy_bosc_mmve32_x_t timm:$ms1, i64:$rs2)),
            (BOSC_AME_MMVE32_X_T_PSEUDO timm:$ms1, GPR:$rs2)>;
  def : Pat<(i64 (int_riscv_buddy_bosc_mmve64_x_t timm:$ms1, i64:$rs2)),
            (BOSC_AME_MMVE64_X_T_PSEUDO timm:$ms1, GPR:$rs2)>;

  def : Pat<(i64 (int_riscv_buddy_bosc_mmve8_x_a  timm:$ms1, i64:$rs2)),
            (BOSC_AME_MMVE8_X_A_PSEUDO  timm:$ms1, GPR:$rs2)>;
  def : Pat<(i64 (int_riscv_buddy_bosc_mmve16_x_a timm:$ms1, i64:$rs2)),
            (BOSC_AME_MMVE16_X_A_PSEUDO timm:$ms1, GPR:$rs2)>;
  def : Pat<(i64 (int_riscv_buddy_bosc_mmve32_x_a timm:$ms1, i64:$rs2)),
            (BOSC_AME_MMVE32_X_A_PSEUDO timm:$ms1, GPR:$rs2)>;
  def : Pat<(i64 (int_riscv_buddy_bosc_mmve64_x_a timm:$ms1, i64:$rs2)),
            (BOSC_AME_MMVE64_X_A_PSEUDO timm:$ms1, GPR:$rs2)>;

  def : Pat<(i64 (int_riscv_buddy_bosc_mfmve8_f_t  timm:$ms1, i64:$rs2)),
            (BOSC_AME_MFMVE8_F_T_PSEUDO  timm:$ms1, GPR:$rs2)>;
  def : Pat<(i64 (int_riscv_buddy_bosc_mfmve16_f_t timm:$ms1, i64:$rs2)),
            (BOSC_AME_MFMVE16_F_T_PSEUDO timm:$ms1, GPR:$rs2)>;
  def : Pat<(i64 (int_riscv_buddy_bosc_mfmve32_f_t timm:$ms1, i64:$rs2)),
            (BOSC_AME_MFMVE32_F_T_PSEUDO timm:$ms1, GPR:$rs2)>;
  def : Pat<(i64 (int_riscv_buddy_bosc_mfmve64_f_t timm:$ms1, i64:$rs2)),
            (BOSC_AME_MFMVE64_F_T_PSEUDO timm:$ms1, GPR:$rs2)>;

  def : Pat<(i64 (int_riscv_buddy_bosc_mfmve8_f_a  timm:$ms1, i64:$rs2)),
            (BOSC_AME_MFMVE8_F_A_PSEUDO  timm:$ms1, GPR:$rs2)>;
  def : Pat<(i64 (int_riscv_buddy_bosc_mfmve16_f_a timm:$ms1, i64:$rs2)),
            (BOSC_AME_MFMVE16_F_A_PSEUDO timm:$ms1, GPR:$rs2)>;
  def : Pat<(i64 (int_riscv_buddy_bosc_mfmve32_f_a timm:$ms1, i64:$rs2)),
            (BOSC_AME_MFMVE32_F_A_PSEUDO timm:$ms1, GPR:$rs2)>;
  def : Pat<(i64 (int_riscv_buddy_bosc_mfmve64_f_a timm:$ms1, i64:$rs2)),
            (BOSC_AME_MFMVE64_F_A_PSEUDO timm:$ms1, GPR:$rs2)>;
  //----------------------------------------------------------------------
  // 5. Scalar GPR to Matrix Move
  //----------------------------------------------------------------------
  def : Pat<(int_riscv_buddy_bosc_mmve8_t_x  timm:$md, i64:$rs1, i64:$rs2),
            (BOSC_AME_MMVE8_T_X_PSEUDO  timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mmve16_t_x timm:$md, i64:$rs1, i64:$rs2),
            (BOSC_AME_MMVE16_T_X_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mmve32_t_x timm:$md, i64:$rs1, i64:$rs2),
            (BOSC_AME_MMVE32_T_X_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mmve64_t_x timm:$md, i64:$rs1, i64:$rs2),
            (BOSC_AME_MMVE64_T_X_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;

  def : Pat<(int_riscv_buddy_bosc_mmve8_a_x  timm:$md, i64:$rs1, i64:$rs2),
            (BOSC_AME_MMVE8_A_X_PSEUDO  timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mmve16_a_x timm:$md, i64:$rs1, i64:$rs2),
            (BOSC_AME_MMVE16_A_X_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mmve32_a_x timm:$md, i64:$rs1, i64:$rs2),
            (BOSC_AME_MMVE32_A_X_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mmve64_a_x timm:$md, i64:$rs1, i64:$rs2),
            (BOSC_AME_MMVE64_A_X_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;

  def : Pat<(int_riscv_buddy_bosc_mfmve8_t_f  timm:$md, i64:$rs1, i64:$rs2),
            (BOSC_AME_MFMVE8_T_F_PSEUDO  timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mfmve16_t_f timm:$md, i64:$rs1, i64:$rs2),
            (BOSC_AME_MFMVE16_T_F_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mfmve32_t_f timm:$md, i64:$rs1, i64:$rs2),
            (BOSC_AME_MFMVE32_T_F_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mfmve64_t_f timm:$md, i64:$rs1, i64:$rs2),
            (BOSC_AME_MFMVE64_T_F_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;

  def : Pat<(int_riscv_buddy_bosc_mfmve8_a_f  timm:$md, i64:$rs1, i64:$rs2),
            (BOSC_AME_MFMVE8_A_F_PSEUDO  timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mfmve16_a_f timm:$md, i64:$rs1, i64:$rs2),
            (BOSC_AME_MFMVE16_A_F_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mfmve32_a_f timm:$md, i64:$rs1, i64:$rs2),
            (BOSC_AME_MFMVE32_A_F_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mfmve64_a_f timm:$md, i64:$rs1, i64:$rs2),
            (BOSC_AME_MFMVE64_A_F_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  //----------------------------------------------------------------------
  // 6. Broadcast Pseudos
  //----------------------------------------------------------------------
  def : Pat<(int_riscv_buddy_bosc_mbcar_m timm:$md, timm:$ms1),
            (BOSC_AME_MBCAR_M_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mbcbr_m timm:$md, timm:$ms1),
            (BOSC_AME_MBCBR_M_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mbccr_m timm:$md, timm:$ms1),
            (BOSC_AME_MBCCR_M_PSEUDO timm:$md, timm:$ms1)>;

  def : Pat<(int_riscv_buddy_bosc_mbcace8_m  timm:$md, timm:$ms1),
            (BOSC_AME_MBCACE8_M_PSEUDO  timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mbcace16_m timm:$md, timm:$ms1),
            (BOSC_AME_MBCACE16_M_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mbcace32_m timm:$md, timm:$ms1),
            (BOSC_AME_MBCACE32_M_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mbcace64_m timm:$md, timm:$ms1),
            (BOSC_AME_MBCACE64_M_PSEUDO timm:$md, timm:$ms1)>;

  def : Pat<(int_riscv_buddy_bosc_mbcbce8_m  timm:$md, timm:$ms1),
            (BOSC_AME_MBCBCE8_M_PSEUDO  timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mbcbce16_m timm:$md, timm:$ms1),
            (BOSC_AME_MBCBCE16_M_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mbcbce32_m timm:$md, timm:$ms1),
            (BOSC_AME_MBCBCE32_M_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mbcbce64_m timm:$md, timm:$ms1),
            (BOSC_AME_MBCBCE64_M_PSEUDO timm:$md, timm:$ms1)>;

  def : Pat<(int_riscv_buddy_bosc_mbccce8_m  timm:$md, timm:$ms1),
            (BOSC_AME_MBCCCE8_M_PSEUDO  timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mbccce16_m timm:$md, timm:$ms1),
            (BOSC_AME_MBCCCE16_M_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mbccce32_m timm:$md, timm:$ms1),
            (BOSC_AME_MBCCCE32_M_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mbccce64_m timm:$md, timm:$ms1),
            (BOSC_AME_MBCCCE64_M_PSEUDO timm:$md, timm:$ms1)>;

  def : Pat<(int_riscv_buddy_bosc_mbcaee8_m  timm:$md, timm:$ms1),
            (BOSC_AME_MBCAEE8_M_PSEUDO  timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mbcaee16_m timm:$md, timm:$ms1),
            (BOSC_AME_MBCAEE16_M_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mbcaee32_m timm:$md, timm:$ms1),
            (BOSC_AME_MBCAEE32_M_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mbcaee64_m timm:$md, timm:$ms1),
            (BOSC_AME_MBCAEE64_M_PSEUDO timm:$md, timm:$ms1)>;

  def : Pat<(int_riscv_buddy_bosc_mbcbee8_m  timm:$md, timm:$ms1),
            (BOSC_AME_MBCBEE8_M_PSEUDO  timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mbcbee16_m timm:$md, timm:$ms1),
            (BOSC_AME_MBCBEE16_M_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mbcbee32_m timm:$md, timm:$ms1),
            (BOSC_AME_MBCBEE32_M_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mbcbee64_m timm:$md, timm:$ms1),
            (BOSC_AME_MBCBEE64_M_PSEUDO timm:$md, timm:$ms1)>;

  def : Pat<(int_riscv_buddy_bosc_mbccee8_m  timm:$md, timm:$ms1),
            (BOSC_AME_MBCCEE8_M_PSEUDO  timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mbccee16_m timm:$md, timm:$ms1),
            (BOSC_AME_MBCCEE16_M_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mbccee32_m timm:$md, timm:$ms1),
            (BOSC_AME_MBCCEE32_M_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mbccee64_m timm:$md, timm:$ms1),
            (BOSC_AME_MBCCEE64_M_PSEUDO timm:$md, timm:$ms1)>;
  //----------------------------------------------------------------------
  // 7. Transpose Pseudos
  //----------------------------------------------------------------------
  def : Pat<(int_riscv_buddy_bosc_mtae8_m  timm:$md, timm:$ms1),
            (BOSC_AME_MTAE8_M_PSEUDO  timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mtae16_m timm:$md, timm:$ms1),
            (BOSC_AME_MTAE16_M_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mtae32_m timm:$md, timm:$ms1),
            (BOSC_AME_MTAE32_M_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mtae64_m timm:$md, timm:$ms1),
            (BOSC_AME_MTAE64_M_PSEUDO timm:$md, timm:$ms1)>;

  def : Pat<(int_riscv_buddy_bosc_mtbe8_m  timm:$md, timm:$ms1),
            (BOSC_AME_MTBE8_M_PSEUDO  timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mtbe16_m timm:$md, timm:$ms1),
            (BOSC_AME_MTBE16_M_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mtbe32_m timm:$md, timm:$ms1),
            (BOSC_AME_MTBE32_M_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mtbe64_m timm:$md, timm:$ms1),
            (BOSC_AME_MTBE64_M_PSEUDO timm:$md, timm:$ms1)>;

  def : Pat<(int_riscv_buddy_bosc_mtce8_m  timm:$md, timm:$ms1),
            (BOSC_AME_MTCE8_M_PSEUDO  timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mtce16_m timm:$md, timm:$ms1),
            (BOSC_AME_MTCE16_M_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mtce32_m timm:$md, timm:$ms1),
            (BOSC_AME_MTCE32_M_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mtce64_m timm:$md, timm:$ms1),
            (BOSC_AME_MTCE64_M_PSEUDO timm:$md, timm:$ms1)>;

  // Element-wise subtract instruction
  //===--------------------------------------------------------------------===//
  // 1. ADD / SUB - No Widen
  //===--------------------------------------------------------------------===//
  def : Pat<(int_riscv_buddy_bosc_maddu_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MADDU_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_maddu_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MADDU_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_maddu_b_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MADDU_B_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_maddu_h_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MADDU_H_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_maddu_w_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MADDU_W_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_maddu_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MADDU_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_msaddu_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSADDU_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msaddu_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSADDU_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msaddu_b_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSADDU_B_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msaddu_h_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSADDU_H_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msaddu_w_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSADDU_W_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msaddu_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSADDU_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_madd_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MADD_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_madd_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MADD_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_madd_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MADD_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_madd_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MADD_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_madd_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MADD_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_madd_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MADD_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_msadd_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSADD_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msadd_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSADD_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msadd_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSADD_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msadd_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSADD_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msadd_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSADD_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msadd_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSADD_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_mfadd_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFADD_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfadd_cf_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFADD_CF_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfadd_hf_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFADD_HF_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfadd_f_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFADD_F_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfadd_d_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFADD_D_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_msubu_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSUBU_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msubu_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSUBU_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msubu_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSUBU_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msubu_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSUBU_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msubu_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSUBU_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msubu_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSUBU_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_mssubu_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSSUBU_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mssubu_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSSUBU_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mssubu_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSSUBU_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mssubu_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSSUBU_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mssubu_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSSUBU_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mssubu_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSSUBU_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_msub_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSUB_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msub_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSUB_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msub_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSUB_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msub_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSUB_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msub_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSUB_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msub_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSUB_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_mssub_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSSUB_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mssub_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSSUB_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mssub_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSSUB_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mssub_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSSUB_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mssub_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSSUB_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mssub_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSSUB_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_mfsub_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFSUB_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfsub_cf_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFSUB_CF_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfsub_hf_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFSUB_HF_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfsub_f_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFSUB_F_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfsub_d_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFSUB_D_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  //===--------------------------------------------------------------------===//
  // 2. ADD / SUB - Double Widen
  //===--------------------------------------------------------------------===//
  def : Pat<(int_riscv_buddy_bosc_mwaddu_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWADDU_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwaddu_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWADDU_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwaddu_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWADDU_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwaddu_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWADDU_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwaddu_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWADDU_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_mwadd_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWADD_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwadd_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWADD_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwadd_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWADD_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwadd_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWADD_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwadd_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWADD_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_mfwadd_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFWADD_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfwadd_cf_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFWADD_CF_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfwadd_hf_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFWADD_HF_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfwadd_f_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFWADD_F_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_mwsubu_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWSUBU_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwsubu_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWSUBU_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwsubu_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWSUBU_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwsubu_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWSUBU_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwsubu_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWSUBU_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_mwsub_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWSUB_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwsub_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWSUB_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwsub_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWSUB_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwsub_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWSUB_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwsub_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWSUB_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_mfwsub_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFWSUB_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfwsub_cf_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFWSUB_CF_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfwsub_hf_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFWSUB_HF_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfwsub_f_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFWSUB_F_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  //===--------------------------------------------------------------------===//
  // 3. MIN / MAX - No Widen
  //===--------------------------------------------------------------------===//
  def : Pat<(int_riscv_buddy_bosc_mminu_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMINU_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mminu_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMINU_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mminu_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMINU_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mminu_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMINU_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mminu_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMINU_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mminu_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMINU_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_mmaxu_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMAXU_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmaxu_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMAXU_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmaxu_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMAXU_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmaxu_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMAXU_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmaxu_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMAXU_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmaxu_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMAXU_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_mmin_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMIN_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmin_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMIN_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmin_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMIN_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmin_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMIN_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmin_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMIN_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmin_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMIN_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_mmax_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMAX_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmax_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMAX_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmax_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMAX_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmax_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMAX_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmax_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMAX_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmax_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMAX_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_mfmin_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFMIN_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfmin_cf_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFMIN_CF_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfmin_hf_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFMIN_HF_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfmin_f_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFMIN_F_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfmin_d_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFMIN_D_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_mfmax_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFMAX_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfmax_cf_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFMAX_CF_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfmax_hf_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFMAX_HF_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfmax_f_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFMAX_F_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfmax_d_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFMAX_D_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  //===--------------------------------------------------------------------===//
  // 4. MUL / MULH / DIV - No Widen
  //===--------------------------------------------------------------------===//
  def : Pat<(int_riscv_buddy_bosc_msmulu_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMULU_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msmulu_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMULU_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msmulu_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMULU_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msmulu_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMULU_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msmulu_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMULU_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msmulu_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMULU_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_mmul_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMUL_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmul_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMUL_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmul_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMUL_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmul_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMUL_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmul_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMUL_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmul_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMUL_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_msmul_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMUL_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msmul_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMUL_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msmul_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMUL_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msmul_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMUL_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msmul_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMUL_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msmul_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMUL_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_mfmul_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFMUL_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfmul_cf_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFMUL_CF_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfmul_hf_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFMUL_HF_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfmul_f_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFMUL_F_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfmul_d_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFMUL_D_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_mmulhu_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMULHU_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmulhu_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMULHU_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmulhu_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMULHU_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmulhu_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMULHU_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmulhu_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMULHU_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmulhu_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMULHU_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_mmulh_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMULH_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmulh_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMULH_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmulh_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMULH_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmulh_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMULH_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmulh_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMULH_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmulh_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMULH_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_mmulhsu_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMULHSU_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmulhsu_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMULHSU_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmulhsu_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMULHSU_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmulhsu_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMULHSU_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmulhsu_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMULHSU_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmulhsu_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMULHSU_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_msmulsu_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMULSU_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msmulsu_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMULSU_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msmulsu_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMULSU_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msmulsu_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMULSU_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msmulsu_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMULSU_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msmulsu_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMULSU_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_mfdiv_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFDIV_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfdiv_cf_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFDIV_CF_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfdiv_hf_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFDIV_HF_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfdiv_f_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFDIV_F_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfdiv_d_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFDIV_D_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  //===--------------------------------------------------------------------===//
  // 5. MUL - Double Widen
  //===--------------------------------------------------------------------===//
  def : Pat<(int_riscv_buddy_bosc_mwmulu_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWMULU_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwmulu_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWMULU_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwmulu_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWMULU_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwmulu_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWMULU_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwmulu_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWMULU_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_mwmul_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWMUL_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwmul_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWMUL_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwmul_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWMUL_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwmul_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWMUL_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwmul_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWMUL_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_mwmulsu_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWMULSU_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwmulsu_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWMULSU_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwmulsu_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWMULSU_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwmulsu_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWMULSU_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwmulsu_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWMULSU_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_mfwmul_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFWMUL_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfwmul_cf_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFWMUL_CF_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfwmul_hf_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFWMUL_HF_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfwmul_f_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFWMUL_F_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  //===--------------------------------------------------------------------===//
  // 6. LOGIC & SHIFT - No Widen
  //===--------------------------------------------------------------------===//
  def : Pat<(int_riscv_buddy_bosc_mand_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MAND_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mor_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MOR_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mxor_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MXOR_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_msll_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSLL_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msll_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSLL_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msll_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSLL_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msll_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSLL_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msll_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSLL_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msll_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSLL_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_msrl_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSRL_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msrl_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSRL_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msrl_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSRL_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msrl_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSRL_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msrl_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSRL_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msrl_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSRL_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;

  def : Pat<(int_riscv_buddy_bosc_msra_mm    timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSRA_MM_PSEUDO    timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msra_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSRA_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msra_b_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSRA_B_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msra_h_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSRA_H_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msra_w_mm  timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSRA_W_MM_PSEUDO  timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msra_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSRA_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  //===--------------------------------------------------------------------===//
  // 7. SQRT - No Widen (Unary, 2-Operand)
  //===--------------------------------------------------------------------===//
  def : Pat<(int_riscv_buddy_bosc_mfsqrt_mm    timm:$md, timm:$ms1),
            (BOSC_AME_MFSQRT_MM_PSEUDO    timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mfsqrt_cf_mm timm:$md, timm:$ms1),
            (BOSC_AME_MFSQRT_CF_MM_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mfsqrt_hf_mm timm:$md, timm:$ms1),
            (BOSC_AME_MFSQRT_HF_MM_PSEUDO timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mfsqrt_f_mm  timm:$md, timm:$ms1),
            (BOSC_AME_MFSQRT_F_MM_PSEUDO  timm:$md, timm:$ms1)>;
  def : Pat<(int_riscv_buddy_bosc_mfsqrt_d_mm  timm:$md, timm:$ms1),
            (BOSC_AME_MFSQRT_D_MM_PSEUDO  timm:$md, timm:$ms1)>;

  // Matrix multiply instruction
  //===--------------------------------------------------------------------===//
  // 1. No-widen matrix multiply-accumulate
  //===--------------------------------------------------------------------===//
  // int
  def : Pat<(int_riscv_buddy_bosc_mma_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMA_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mma_h_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMA_H_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mma_w_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMA_W_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mma_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMA_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  // int, saturated
  def : Pat<(int_riscv_buddy_bosc_msma_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMA_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msma_h_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMA_H_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msma_w_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMA_W_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msma_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMA_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  // uint
  def : Pat<(int_riscv_buddy_bosc_mmau_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMAU_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmau_h_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMAU_H_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmau_w_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMAU_W_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mmau_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMAU_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  // uint, saturated
  def : Pat<(int_riscv_buddy_bosc_msmau_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMAU_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msmau_h_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMAU_H_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msmau_w_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMAU_W_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msmau_dw_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSMAU_DW_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  // float point
  def : Pat<(int_riscv_buddy_bosc_mfma_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFMA_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfma_hf_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFMA_HF_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfma_f_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFMA_F_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfma_d_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFMA_D_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  //===--------------------------------------------------------------------===//
  // 2. Double-widen matrix multiply-accumulate
  //===--------------------------------------------------------------------===//
  // uint
  def : Pat<(int_riscv_buddy_bosc_mwmau_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWMAU_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwmau_h_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWMAU_H_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwmau_w_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWMAU_W_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  // uint, saturated
  def : Pat<(int_riscv_buddy_bosc_mswmau_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSWMAU_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mswmau_h_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSWMAU_H_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mswmau_w_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSWMAU_W_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  // int
  def : Pat<(int_riscv_buddy_bosc_mwma_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWMA_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwma_h_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWMA_H_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mwma_w_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MWMA_W_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  // int, saturated
  def : Pat<(int_riscv_buddy_bosc_mswma_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSWMA_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mswma_h_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSWMA_H_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mswma_w_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSWMA_W_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  // float point
  def : Pat<(int_riscv_buddy_bosc_mfwma_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFWMA_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfwma_cf_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFWMA_CF_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfwma_hf_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFWMA_HF_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfwma_f_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFWMA_F_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  //===--------------------------------------------------------------------===//
  // 3. Quad-widen matrix multiply-accumulate
  //===--------------------------------------------------------------------===//
  // uint
  def : Pat<(int_riscv_buddy_bosc_mqmau_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MQMAU_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mqmau_b_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MQMAU_B_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  // uint, saturated
  def : Pat<(int_riscv_buddy_bosc_msqmau_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSQMAU_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msqmau_b_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSQMAU_B_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  // int
  def : Pat<(int_riscv_buddy_bosc_mqma_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MQMA_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mqma_b_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MQMA_B_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  // int, saturated
  def : Pat<(int_riscv_buddy_bosc_msqma_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSQMA_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msqma_b_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSQMA_B_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  // float point
  def : Pat<(int_riscv_buddy_bosc_mfqma_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFQMA_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_mfqma_cf_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MFQMA_CF_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  //===--------------------------------------------------------------------===//
  // 4. Oct-widen matrix multiply-accumulate
  //===--------------------------------------------------------------------===//
  // uint
  def : Pat<(int_riscv_buddy_bosc_momau_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MOMAU_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_momau_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MOMAU_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  // uint, saturated
  def : Pat<(int_riscv_buddy_bosc_msomau_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSOMAU_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msomau_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSOMAU_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  // int
  def : Pat<(int_riscv_buddy_bosc_moma_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MOMA_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_moma_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MOMA_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  // int, saturated
  def : Pat<(int_riscv_buddy_bosc_msoma_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSOMA_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
  def : Pat<(int_riscv_buddy_bosc_msoma_hb_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MSOMA_HB_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
}
