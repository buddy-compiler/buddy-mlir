//===- RISCVInstrInfoSystolicArrayBall.td - SystolicArrayBall instructions ===//
//
// SystolicArrayBall instructions (bbfp_mul)
//
//===----------------------------------------------------------------------===//

// bb_bbfp_mul: funct7=65 (0b1000001)
let isCodeGenOnly = 1, Predicates = [HasBuddyExt] in
def BB_BBFP_MUL : RVInstR<0b1000001, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "bb_bbfp_mul", "$rs1, $rs2"> {
  let rd = 0;
}

// Intrinsic pattern
let Predicates = [HasBuddyExt] in
def : Pat<(int_riscv_bb_bbfp_mul GPR:$rs1, GPR:$rs2), (BB_BBFP_MUL GPR:$rs1, GPR:$rs2)>;
