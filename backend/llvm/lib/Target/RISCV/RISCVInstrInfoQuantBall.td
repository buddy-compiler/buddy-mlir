//===- RISCVInstrInfoQuantBall.td - QuantBall instructions ---------------===//
//
// QuantBall instructions (quant)
//
//===----------------------------------------------------------------------===//

// bb_quant: funct7=51 (0b0110011)
let hasSideEffects = 1, mayLoad = 1, mayStore = 1, isCodeGenOnly = 1,
    Predicates = [HasBuddyExt] in
def BB_QUANT : RVInstR<0b0110011, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "bb_quant", "$rs1, $rs2"> {
  let rd = 0;
}

// Intrinsic pattern
let Predicates = [HasBuddyExt] in
def : Pat<(int_riscv_bb_quant GPR:$rs1, GPR:$rs2), (BB_QUANT GPR:$rs1, GPR:$rs2)>;
