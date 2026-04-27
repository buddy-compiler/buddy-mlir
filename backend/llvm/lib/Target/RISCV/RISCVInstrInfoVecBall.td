//===- RISCVInstrInfoVecBall.td - VecBall instructions -------------------===//
//
// VecBall instructions (mul_warp16)
//
//===----------------------------------------------------------------------===//

// bb_mul_warp16: funct7=64 (0b1000000)
let isCodeGenOnly = 1, Predicates = [HasBuddyExt] in
def BB_MUL_WARP16 : RVInstR<0b1000000, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "bb_mul_warp16", "$rs1, $rs2"> {
  let rd = 0;
}

// Intrinsic pattern
let Predicates = [HasBuddyExt] in
def : Pat<(int_riscv_bb_mul_warp16 GPR:$rs1, GPR:$rs2), (BB_MUL_WARP16 GPR:$rs1, GPR:$rs2)>;
