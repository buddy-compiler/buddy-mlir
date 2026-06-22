//===- RISCVInstrInfoReluBall.td - ReluBall instructions -----------------===//
//
// ReluBall instructions (relu)
//
//===----------------------------------------------------------------------===//

// bb_relu: funct7=50 (0b0110010)
let hasSideEffects = 1, mayLoad = 1, mayStore = 1, isCodeGenOnly = 1,
    Predicates = [HasBuddyExt] in
def BB_RELU : RVInstR<0b0110010, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "bb_relu", "$rs1, $rs2"> {
  let rd = 0;
}

// Intrinsic pattern
let Predicates = [HasBuddyExt] in
def : Pat<(int_riscv_bb_relu GPR:$rs1, GPR:$rs2), (BB_RELU GPR:$rs1, GPR:$rs2)>;
