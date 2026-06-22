//===- RISCVInstrInfoTransposeBall.td - TransposeBall instructions -------===//
//
// TransposeBall instructions (transpose)
//
//===----------------------------------------------------------------------===//

// bb_transpose: funct7=49 (0b0110001)
let isCodeGenOnly = 1, Predicates = [HasBuddyExt] in
def BB_TRANSPOSE : RVInstR<0b0110001, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "bb_transpose", "$rs1, $rs2"> {
  let rd = 0;
}

// Intrinsic pattern
let Predicates = [HasBuddyExt] in
def : Pat<(int_riscv_bb_transpose GPR:$rs1, GPR:$rs2), (BB_TRANSPOSE GPR:$rs1, GPR:$rs2)>;
