//===- RISCVInstrInfoBuddyExt.td ------------------------------------------===//
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
// This is the instruction information file of RISC-V buddy extension.
//
//===----------------------------------------------------------------------===//

include "llvm/IR/IntrinsicsRISCVBuddyExt.td"

let hasSideEffects = 1, mayLoad = 1, mayStore = 1, Predicates = [HasBuddyExt] in
def MVIN : RVInstR<0b0000010, 0b011, OPC_CUSTOM_3, (outs),
                   (ins GPR:$rs1, GPR:$rs2), "mvin","$rs1, $rs2"> {
  let rd = 0;              
}

let hasSideEffects = 1, mayLoad = 1, mayStore = 1, Predicates = [HasBuddyExt] in 
def MVOUT : RVInstR<0b0000011, 0b011, OPC_CUSTOM_3,(outs),
                    (ins GPR:$rs1, GPR:$rs2), "mvout","$rs1, $rs2">{
  let rd = 0;
}

let hasSideEffects = 1, mayLoad = 1, mayStore = 1, Predicates = [HasBuddyExt] in 
def FLUSH : RVInstR<0b0000111, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "flush", "$rs1"> {
  let rd = 0;
}

let Predicates = [HasBuddyExt] in 
def CONFIG_LD : RVInstR<0b0000000, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "config_ld", "$rs1, $rs2"> {
  let rd = 0;
}

let Predicates = [HasBuddyExt] in 
def CONFIG_ST : RVInstR<0b0000000, 0b011, OPC_CUSTOM_3,(outs),
                    (ins GPR:$rs1, GPR:$rs2), "config_st", "$rs1, $rs2"> {
  let rd = 0;
}

let Predicates = [HasBuddyExt] in 
def CONFIG_EX : RVInstR<0b0000000, 0b011, OPC_CUSTOM_3,(outs),
                    (ins GPR:$rs1, GPR:$rs2), "config_ex", "$rs1, $rs2"> {
  let rd = 0;
}

let hasSideEffects = 1, mayLoad = 1, mayStore =1, Predicates = [HasBuddyExt] in 
def PRELOAD : RVInstR<0b0000110, 0b011,OPC_CUSTOM_3,(outs),
                    (ins GPR:$rs1, GPR:$rs2), "preload", "$rs1, $rs2">{
  let rd = 0;
}

let Predicates = [HasBuddyExt] in 
def COMPUTE_PRELOADED : RVInstR<0b0000100, 0b011, OPC_CUSTOM_3, (outs),
                     (ins GPR:$rs1, GPR:$rs2), "compute_preloaded", "$rs1, $rs2">{
  let rd = 0;
}

let Predicates = [HasBuddyExt] in
def: PatGprGpr<int_riscv_mvin, MVIN>;

let Predicates = [HasBuddyExt] in
def : PatGprGpr<int_riscv_mvout, MVOUT>;

let Predicates = [HasBuddyExt] in 
def : PatGprGpr<int_riscv_flush, FLUSH>;

let Predicates = [HasBuddyExt] in 
def : PatGprGpr<int_riscv_configLd, CONFIG_LD>;

let Predicates = [HasBuddyExt] in 
def : PatGprGpr<int_riscv_configSt, CONFIG_ST>;

let Predicates = [HasBuddyExt] in 
def : PatGprGpr<int_riscv_configEx, CONFIG_EX>;

let Predicates = [HasBuddyExt] in 
def : PatGprGpr<int_riscv_preload, PRELOAD>;

let Predicates = [HasBuddyExt] in 
def : PatGprGpr<int_riscv_computePreloaded, COMPUTE_PRELOADED>;
