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
class BOSC_AME_MMA_MM<bits<3> eew_val, string opcodestr>
    : RVInstBOSCAME32<(outs AccReg:$md),
                  (ins AccReg:$md_in, TileReg:$ms1, TileReg:$ms2),
                  opcodestr, "$md, $ms1, $ms2"> {
  let Constraints = "$md = $md_in";  // Accumulator constraint
  let funct6 = 0b001000;     // No widening
  let fp = 0;
  let sa = 0b0;        // Matrix multiplication
  let sn = 0b1;      // (signed for integers, 0 for floating-point)
  let eew = eew_val;
  let ma = 0b1;       // Accumulate
  let opcode = 0b1110111;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  // mma.h.mm - int16 × int16 → int16 accumulate (no widen)
  def BOSC_AME_MMA_H_MM : BOSC_AME_MMA_MM<0b001, "mma.h.mm">;
  // mma.w.mm - int32 × int32 → int32 accumulate (no widen)
  def BOSC_AME_MMA_W_MM : BOSC_AME_MMA_MM<0b010, "mma.w.mm">;
  // mma.dw.mm - int64 × int64 → int64 accumulate (no widen)
  def BOSC_AME_MMA_DW_MM : BOSC_AME_MMA_MM<0b011, "mma.dw.mm">;
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
  let Inst{31-26} = 0b000001;
}

let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 0, mayStore = 0 in {
  // msettilem - set tile M dimension from register
  def BOSC_AME_MSETTILEM : RVInstBOSCAMEConfig32<0b101, "msettilem">;
  // msettilen - set tile N dimension from register
  def BOSC_AME_MSETTILEN : RVInstBOSCAMEConfig32<0b100, "msettilen">;
  // msettilek - set tile K dimension from register
  def BOSC_AME_MSETTILEK : RVInstBOSCAMEConfig32<0b110, "msettilek">;
}

let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 0, mayStore = 0 in {
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

// Pseudo instruction for msettilemi with i64 immediate
// Output: msettilemi rd, <imm>
let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 0, mayStore = 0,
    isCodeGenOnly = 1 in
def BOSC_AME_MSETTILEMI_PSEUDO : Pseudo<(outs GPR:$rd), (ins i64imm:$imm), []> {
  let AsmString = "msettilemi\t$rd, $imm";
}

// Pseudo instruction for msettileni with i64 immediate
let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 0, mayStore = 0,
    isCodeGenOnly = 1 in
def BOSC_AME_MSETTILENI_PSEUDO : Pseudo<(outs GPR:$rd), (ins i64imm:$imm), []> {
  let AsmString = "msettileni\t$rd, $imm";
}

// Pseudo instruction for msettileki with i64 immediate
let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 0, mayStore = 0,
    isCodeGenOnly = 1 in
def BOSC_AME_MSETTILEKI_PSEUDO : Pseudo<(outs GPR:$rd), (ins i64imm:$imm), []> {
  let AsmString = "msettileki\t$rd, $imm";
}

// Pseudo instruction for mzero with accumulator index
let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 0, mayStore = 0,
    isCodeGenOnly = 1 in
def BOSC_AME_MZERO_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$md), []> {
  let AsmString = "mzero\tacc$md";
}

// Pseudo instruction for mlae32.m with tile index
let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 1, mayStore = 0,
    isCodeGenOnly = 1 in
def BOSC_AME_MLAE32_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
  let AsmString = "mlae32.m\ttr$md, ($rs1), $rs2";
}

// Pseudo instruction for mlbe32.m with tile index
let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 1, mayStore = 0,
    isCodeGenOnly = 1 in
def BOSC_AME_MLBE32_M_PSEUDO : Pseudo<(outs), (ins BOSCAMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
  let AsmString = "mlbe32.m\ttr$md, ($rs1), $rs2";
}

// Pseudo instruction for msce32.m with accumulator index
let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 0, mayStore = 1,
    isCodeGenOnly = 1 in
def BOSC_AME_MSCE32_M_PSEUDO : Pseudo<(outs), (ins BOSCAMEAccIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
  let AsmString = "msce32.m\tacc$ms3, ($rs1), $rs2";
}

// Pseudo instruction for mma.w.mm.tile with indices
// md: AccReg index (0-7), ms1/ms2: TileReg indices (0-7)
let Predicates = [HasBuddyExt], hasSideEffects = 1,
    isCodeGenOnly = 1 in
def BOSC_AME_MMA_W_MM_PSEUDO : Pseudo<(outs),
    (ins BOSCAMEAccIndex:$md, BOSCAMETileIndex:$ms1, BOSCAMETileIndex:$ms2), []> {
  let AsmString = "mma.w.mm\tacc$md, tr$ms1, tr$ms2";
}

//===----------------------------------------------------------------------===//
// Pattern Matching for 32-bit New Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasBuddyExt] in {

  // Configuration immediate instructions
  def : Pat<(i64 (int_riscv_buddy_bosc_msettilemi timm:$imm)),
            (BOSC_AME_MSETTILEMI_PSEUDO timm:$imm)>;
  def : Pat<(i64 (int_riscv_buddy_bosc_msettileni timm:$imm)),
            (BOSC_AME_MSETTILENI_PSEUDO timm:$imm)>;
  def : Pat<(i64 (int_riscv_buddy_bosc_msettileki timm:$imm)),
            (BOSC_AME_MSETTILEKI_PSEUDO timm:$imm)>;

  // Zero instruction
  def : Pat<(int_riscv_buddy_bosc_mzero timm:$md),
            (BOSC_AME_MZERO_PSEUDO timm:$md)>;

  // Load instructions
  def : Pat<(int_riscv_buddy_bosc_mlae32_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLAE32_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_buddy_bosc_mlbe32_m timm:$md, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MLBE32_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;

  // Store instruction
  def : Pat<(int_riscv_buddy_bosc_msce32_m timm:$ms3, iPTR:$rs1, i64:$rs2),
            (BOSC_AME_MSCE32_M_PSEUDO timm:$ms3, GPR:$rs1, GPR:$rs2)>;

  // Matrix multiply instruction
  def : Pat<(int_riscv_buddy_bosc_mma_w_mm timm:$md, timm:$ms1, timm:$ms2),
            (BOSC_AME_MMA_W_MM_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
}
