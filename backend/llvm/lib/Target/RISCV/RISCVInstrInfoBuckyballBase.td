//===- RISCVInstrInfoBuckyballBase.td - Base instructions ----------------===//
//
// Common Buckyball instructions (mvin, mvout, mset, fence)
//
//===----------------------------------------------------------------------===//

// bb_mvin: funct7=33 (0b0100001)
let hasSideEffects = 1, mayLoad = 1, mayStore = 1, isCodeGenOnly = 1,
    Predicates = [HasBuddyExt] in
def BB_MVIN : RVInstR<0b0100001, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "bb_mvin", "$rs1, $rs2"> {
  let rd = 0;
}

// bb_mvout: funct7=16 (0b0010000)
let hasSideEffects = 1, mayLoad = 1, mayStore = 1, isCodeGenOnly = 1,
    Predicates = [HasBuddyExt] in
def BB_MVOUT : RVInstR<0b0010000, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "bb_mvout", "$rs1, $rs2"> {
  let rd = 0;
}

// bb_mset: funct7=32 (0b0100000)
let hasSideEffects = 1, mayLoad = 1, mayStore = 1, isCodeGenOnly = 1,
    Predicates = [HasBuddyExt] in
def BB_MSET : RVInstR<0b0100000, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "bb_mset", "$rs1, $rs2"> {
  let rd = 0;
}

// bb_fence: funct7=0 (0b0000000)
let hasSideEffects = 1, mayLoad = 1, mayStore = 1, isCodeGenOnly = 1,
    Predicates = [HasBuddyExt] in
def BB_FENCE : RVInstR<0b0000000, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "bb_fence", "$rs1, $rs2"> {
  let rd = 0;
}

// Intrinsic patterns
let Predicates = [HasBuddyExt] in {
  def : Pat<(int_riscv_bb_mvin GPR:$rs1, GPR:$rs2), (BB_MVIN GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_bb_mvout GPR:$rs1, GPR:$rs2), (BB_MVOUT GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_bb_mset GPR:$rs1, GPR:$rs2), (BB_MSET GPR:$rs1, GPR:$rs2)>;
  def : Pat<(int_riscv_bb_fence GPR:$rs1, GPR:$rs2), (BB_FENCE GPR:$rs1, GPR:$rs2)>;
}
