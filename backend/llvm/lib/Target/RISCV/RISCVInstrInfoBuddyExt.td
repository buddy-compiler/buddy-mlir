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
def COMPUTE_ACCUMULATED : RVInstR<0b0000101, 0b011, OPC_CUSTOM_3, (outs),
                      (ins GPR:$rs1, GPR:$rs2), "compute_accumulated", "$rs1, $rs2"> {
  let rd = 0;
}

let Predicates = [HasBuddyExt] in 
def LOOP_WS_CONFIG_BOUNDS : RVInstR<0b0001001, 0b011, OPC_CUSTOM_3,(outs),
                      (ins GPR:$rs1, GPR:$rs2), "loop_ws_config_bounds","$rs1, $rs2">{
  let rd = 0;
}

let Predicates = [HasBuddyExt] in 
def LOOP_WS_CONFIG_ADDRS_AB : RVInstR<0b0001010, 0b011, OPC_CUSTOM_3, (outs),
                      (ins GPR:$rs1, GPR:$rs2), "loop_ws_config_addrs_ab", "$rs1, $rs2"> {
  let rd = 0;
} 

let Predicates = [HasBuddyExt] in 
def LOOP_WS_CONFIG_ADDRS_DC : RVInstR<0b0001011, 0b011, OPC_CUSTOM_3, (outs), 
                      (ins GPR:$rs1, GPR:$rs2), "loop_ws_config_addrs_dc", "$rs1, $rs2"> {
  let rd = 0;
}

let Predicates = [HasBuddyExt] in 
def LOOP_WS_CONFIG_STRIDES_AB : RVInstR<0b0001100, 0b011, OPC_CUSTOM_3,(outs), 
                       (ins GPR:$rs1, GPR:$rs2), "loop_ws_config_strides_ab", "$rs1, $rs2"> {
  let rd = 0;
}

let Predicates = [HasBuddyExt] in 
def LOOP_WS_CONFIG_STRIDES_DC : RVInstR<0b0001101, 0b011, OPC_CUSTOM_3, (outs), 
                        (ins GPR:$rs1, GPR:$rs2), "loop_ws_config_strides_dc", "$rs1, $rs2"> {
  let rd = 0;
}

let Predicates = [HasBuddyExt] in 
def LOOP_WS : RVInstR<0b0001000, 0b011, OPC_CUSTOM_3, (outs), 
                        (ins GPR:$rs1, GPR:$rs2), "loop_ws", "$rs1, $rs2"> {
    let rd = 0;
}

let hasSideEffects = 1, mayLoad = 1, mayStore =1, Predicates = [HasBuddyExt] in 
def LOOP_CONV_WS : RVInstR<0b0001111, 0b011, OPC_CUSTOM_3, (outs), 
                        (ins GPR:$rs1, GPR:$rs2), "loop_conv_ws", "$rs1, $rs2"> {
    let rd = 0;
}

let Predicates = [HasBuddyExt] in 
def LOOP_CONV_WS_CONFIG1 : RVInstR<0b0010000, 0b011, OPC_CUSTOM_3, (outs),
                        (ins GPR:$rs1, GPR:$rs2), "loop_conv_ws_config1", "$rs1, $rs2"> {
    let rd = 0;
}

let Predicates = [HasBuddyExt] in 
def LOOP_CONV_WS_CONFIG2 : RVInstR<0b0010001, 0b011, OPC_CUSTOM_3, (outs),
                        (ins GPR:$rs1, GPR:$rs2), "loop_conv_ws_config2", "$rs1, $rs2"> {
    let rd = 0;
}

let Predicates = [HasBuddyExt] in 
def LOOP_CONV_WS_CONFIG3 : RVInstR<0b0010010, 0b011, OPC_CUSTOM_3, (outs), 
                        (ins GPR:$rs1, GPR:$rs2), "loop_conv_ws_config3", "$rs1, $rs2"> {
    let rd = 0;
}

let Predicates = [HasBuddyExt] in 
def LOOP_CONV_WS_CONFIG4 : RVInstR<0b0010011, 0b011, OPC_CUSTOM_3, (outs), 
                        (ins GPR:$rs1, GPR:$rs2), "loop_conv_ws_config4", "$rs1, $rs2"> {
    let rd = 0;
}

let Predicates = [HasBuddyExt] in 
def LOOP_CONV_WS_CONFIG5 : RVInstR<0b0010100, 0b011, OPC_CUSTOM_3, (outs), 
                        (ins GPR:$rs1, GPR:$rs2), "loop_conv_ws_config5", "$rs1, $rs2"> {
    let rd = 0;
}

let Predicates = [HasBuddyExt] in 
def LOOP_CONV_WS_CONFIG6 : RVInstR<0b0010101, 0b011, OPC_CUSTOM_3, (outs),
                        (ins GPR:$rs1, GPR:$rs2), "loop_conv_ws_config6", "$rs1, $rs2"> {
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

let Predicates = [HasBuddyExt] in 
def : PatGprGpr<int_riscv_computeAccumulated, COMPUTE_ACCUMULATED>;

let Predicates = [HasBuddyExt] in 
def : PatGprGpr<int_riscv_loopWsConfigBounds, LOOP_WS_CONFIG_BOUNDS>;

let Predicates = [HasBuddyExt] in 
def : PatGprGpr<int_riscv_loopWsConfigAddrsAB, LOOP_WS_CONFIG_ADDRS_AB>;

let Predicates = [HasBuddyExt] in 
def : PatGprGpr<int_riscv_loopWsConfigAddrsDC, LOOP_WS_CONFIG_ADDRS_DC>;

let Predicates = [HasBuddyExt] in 
def : PatGprGpr<int_riscv_loopWsConfigStridesAB, LOOP_WS_CONFIG_STRIDES_AB>;

let Predicates = [HasBuddyExt] in 
def : PatGprGpr<int_riscv_loopWsConfigStridesDC, LOOP_WS_CONFIG_STRIDES_DC>;

let Predicates = [HasBuddyExt] in 
def : PatGprGpr<int_riscv_loopWs, LOOP_WS>;

let Predicates = [HasBuddyExt] in 
def : PatGprGpr<int_riscv_loopConvWs, LOOP_CONV_WS>;

let Predicates = [HasBuddyExt] in 
def : PatGprGpr<int_riscv_loopConvWsConfig1, LOOP_CONV_WS_CONFIG1>;

let Predicates = [HasBuddyExt] in 
def : PatGprGpr<int_riscv_loopConvWsConfig2, LOOP_CONV_WS_CONFIG2>;

let Predicates = [HasBuddyExt] in 
def : PatGprGpr<int_riscv_loopConvWsConfig3, LOOP_CONV_WS_CONFIG3>;

let Predicates = [HasBuddyExt] in 
def : PatGprGpr<int_riscv_loopConvWsConfig4, LOOP_CONV_WS_CONFIG4>;

let Predicates = [HasBuddyExt] in 
def : PatGprGpr<int_riscv_loopConvWsConfig5, LOOP_CONV_WS_CONFIG5>;

let Predicates = [HasBuddyExt] in 
def : PatGprGpr<int_riscv_loopConvWsConfig6, LOOP_CONV_WS_CONFIG6>;
