//===- RISCVInstrInfoBuddyExtBB.td - Buckyball bb_* custom-3 instructions ---===//
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
// Buckyball custom-3 (bb_*) defs and intrinsic Pats. Included from
// RISCVInstrInfoBuddyExt.td. Gated by Subtarget feature +buddyext-bb (HasBuddyExt
// in RISCVBuddyExt.td). funct7 matches bb-tests/workloads/lib/bbhw/isa/*.c.
//
//===----------------------------------------------------------------------===//

// funct7 = decimal opcode in bb-tests/workloads/lib/bbhw/isa/*.c (bebop FUNCT_*).
let hasSideEffects = 1, mayLoad = 1, mayStore = 1, isCodeGenOnly = 1,
    Predicates = [HasBuddyExt] in
def BB_MVIN : RVInstR<0b0100001, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "bb_mvin", "$rs1, $rs2"> {
  let rd = 0;
}

let hasSideEffects = 1, mayLoad = 1, mayStore = 1, isCodeGenOnly = 1,
    Predicates = [HasBuddyExt] in
def BB_MVOUT : RVInstR<0b0010000, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "bb_mvout", "$rs1, $rs2"> {
  let rd = 0;
}

let isCodeGenOnly = 1, Predicates = [HasBuddyExt] in
def BB_BBFP_MUL : RVInstR<0b1000001, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "bb_bbfp_mul", "$rs1, $rs2"> {
  let rd = 0;
}

// bb_fence: isa funct7=0 (00_fence.c). Do not enable +buddyext with +buddyext-bb if
// Gemmini CONFIG uses the same funct7 (see RISCVInstrInfoBuddyExt.td).
let hasSideEffects = 1, mayLoad = 1, mayStore = 1, isCodeGenOnly = 1,
    Predicates = [HasBuddyExt] in
def BB_FENCE : RVInstR<0b0000000, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "bb_fence", "$rs1, $rs2"> {
  let rd = 0;
}

// funct7 = 64 (isa.h BB_MUL_WARP16_FUNC7). Was 0b0100000 (32), collided with mset.
let isCodeGenOnly = 1, Predicates = [HasBuddyExt] in
def BB_MUL_WARP16 : RVInstR<0b1000000, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "bb_mul_warp16", "$rs1, $rs2"> {
  let rd = 0;
}

let isCodeGenOnly = 1, Predicates = [HasBuddyExt] in
def BB_IM2COL : RVInstR<0b0110000, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "bb_im2col", "$rs1, $rs2"> {
  let rd = 0;
}

let isCodeGenOnly = 1, Predicates = [HasBuddyExt] in
def BB_TRANSPOSE : RVInstR<0b0110001, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "bb_transpose", "$rs1, $rs2"> {
  let rd = 0;
}

let Predicates = [HasBuddyExt] in
def : Pat<(int_riscv_bb_mvin GPR:$rs1, GPR:$rs2), (BB_MVIN GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in
def : Pat<(int_riscv_bb_mvout GPR:$rs1, GPR:$rs2), (BB_MVOUT GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in
def : Pat<(int_riscv_bb_bbfp_mul GPR:$rs1, GPR:$rs2), (BB_BBFP_MUL GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in
def : Pat<(int_riscv_bb_fence GPR:$rs1, GPR:$rs2), (BB_FENCE GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in
def : Pat<(int_riscv_bb_mul_warp16 GPR:$rs1, GPR:$rs2), (BB_MUL_WARP16 GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in
def : Pat<(int_riscv_bb_im2col GPR:$rs1, GPR:$rs2), (BB_IM2COL GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in
def : Pat<(int_riscv_bb_transpose GPR:$rs1, GPR:$rs2), (BB_TRANSPOSE GPR:$rs1, GPR:$rs2)>;

let hasSideEffects = 1, mayLoad = 1, mayStore = 1, isCodeGenOnly = 1,
    Predicates = [HasBuddyExt] in
def BB_RELU : RVInstR<0b0110010, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "bb_relu", "$rs1, $rs2"> {
  let rd = 0;
}

let hasSideEffects = 1, mayLoad = 1, mayStore = 1, isCodeGenOnly = 1,
    Predicates = [HasBuddyExt] in
def BB_QUANT : RVInstR<0b0110011, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "bb_quant", "$rs1, $rs2"> {
  let rd = 0;
}

let hasSideEffects = 1, mayLoad = 1, mayStore = 1, isCodeGenOnly = 1,
    Predicates = [HasBuddyExt] in
def BB_DEQUANT : RVInstR<0b0110100, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "bb_dequant", "$rs1, $rs2"> {
  let rd = 0;
}

let Predicates = [HasBuddyExt] in
def : Pat<(int_riscv_bb_relu GPR:$rs1, GPR:$rs2), (BB_RELU GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in
def : Pat<(int_riscv_bb_quant GPR:$rs1, GPR:$rs2), (BB_QUANT GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in
def : Pat<(int_riscv_bb_dequant GPR:$rs1, GPR:$rs2), (BB_DEQUANT GPR:$rs1, GPR:$rs2)>;

// funct7 = 32 (isa.h BB_MSET_FUNC7, bebop FUNCT_MSET). Was 0b0010111 (23).
let hasSideEffects = 1, mayLoad = 1, mayStore = 1, isCodeGenOnly = 1,
    Predicates = [HasBuddyExt] in
def BB_MSET : RVInstR<0b0100000, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "bb_mset", "$rs1, $rs2"> {
  let rd = 0;
}

let Predicates = [HasBuddyExt] in
def : Pat<(int_riscv_bb_mset GPR:$rs1, GPR:$rs2), (BB_MSET GPR:$rs1, GPR:$rs2)>;
