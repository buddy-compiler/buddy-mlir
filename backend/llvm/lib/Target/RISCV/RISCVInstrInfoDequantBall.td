//===- RISCVInstrInfoDequantBall.td - DequantBall instructions -----------===//
//
// DequantBall instructions (dequant)
//
//===----------------------------------------------------------------------===//

// bb_dequant: funct7=52 (0b0110100)
let hasSideEffects = 1, mayLoad = 1, mayStore = 1, isCodeGenOnly = 1,
    Predicates = [HasBuddyExt] in
def BB_DEQUANT : RVInstR<0b0110100, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "bb_dequant", "$rs1, $rs2"> {
  let rd = 0;
}

// Intrinsic pattern
let Predicates = [HasBuddyExt] in
def : Pat<(int_riscv_bb_dequant GPR:$rs1, GPR:$rs2), (BB_DEQUANT GPR:$rs1, GPR:$rs2)>;
