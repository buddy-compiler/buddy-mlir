//===- RISCVInstrInfoIm2colBall.td - Im2colBall instructions -------------===//
//
// Im2colBall instructions (im2col)
//
//===----------------------------------------------------------------------===//

// bb_im2col: funct7=48 (0b0110000)
let isCodeGenOnly = 1, Predicates = [HasBuddyExt] in
def BB_IM2COL : RVInstR<0b0110000, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "bb_im2col", "$rs1, $rs2"> {
  let rd = 0;
}

// Intrinsic pattern
let Predicates = [HasBuddyExt] in
def : Pat<(int_riscv_bb_im2col GPR:$rs1, GPR:$rs2), (BB_IM2COL GPR:$rs1, GPR:$rs2)>;
