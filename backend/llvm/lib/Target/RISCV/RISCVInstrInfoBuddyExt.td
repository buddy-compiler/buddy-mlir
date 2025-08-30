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

// Gemmini defines different values as func7
// - https://github.com/ucb-bar/gemmini-rocc-tests/blob/e326e7c43457ff08669fe88edcaa395d846474d8/include/gemmini.h#L25

// Gemmini uses 0x3 (0b011) as func3
// - https://github.com/IBM/rocc-software/blob/fddb795a0b52e82f8f4ce9ead9b1428440a62ab0/src/xcustom.h#L147

// Gemmini uses OPC_CUSTOM_3
// - https://github.com/IBM/rocc-software/blob/fddb795a0b52e82f8f4ce9ead9b1428440a62ab0/src/xcustom.h#L123

let hasSideEffects = 1, mayLoad = 1, mayStore = 1, Predicates = [HasBuddyExt] in
def MVIN : RVInstR<0b0000010, 0b011, OPC_CUSTOM_3, (outs),
                   (ins GPR:$rs1, GPR:$rs2), "mvin","$rs1, $rs2"> {
  let rd = 0;              
}

let hasSideEffects = 1, mayLoad = 1, mayStore = 1, Predicates = [HasBuddyExt] in
def MVIN2 : RVInstR<0b0000001, 0b011, OPC_CUSTOM_3, (outs),
                   (ins GPR:$rs1, GPR:$rs2), "mvin2","$rs1, $rs2"> {
  let rd = 0;
}

let hasSideEffects = 1, mayLoad = 1, mayStore = 1, Predicates = [HasBuddyExt] in
def MVIN3 : RVInstR<0b0001110, 0b011, OPC_CUSTOM_3, (outs),
                   (ins GPR:$rs1, GPR:$rs2), "mvin3","$rs1, $rs2"> {
  let rd = 0;
}

let hasSideEffects = 1, mayLoad = 1, mayStore = 1, Predicates = [HasBuddyExt] in 
def MVOUT : RVInstR<0b0000011, 0b011, OPC_CUSTOM_3, (outs),
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
def CONFIG_ST : RVInstR<0b0000000, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "config_st", "$rs1, $rs2"> {
  let rd = 0;
}

let Predicates = [HasBuddyExt] in 
def CONFIG_EX : RVInstR<0b0000000, 0b011, OPC_CUSTOM_3,(outs),
                    (ins GPR:$rs1, GPR:$rs2), "config_ex", "$rs1, $rs2"> {
  let rd = 0;
}

let Predicates = [HasBuddyExt] in 
def CONFIG_NORM : RVInstR<0b0000000, 0b011, OPC_CUSTOM_3,(outs),
                    (ins GPR:$rs1, GPR:$rs2), "config_norm", "$rs1, $rs2"> {
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
def : Pat<(int_riscv_mvin GPR:$rs1, GPR:$rs2), (MVIN GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in
def : Pat<(int_riscv_mvin2 GPR:$rs1, GPR:$rs2), (MVIN2 GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in
def : Pat<(int_riscv_mvin3 GPR:$rs1, GPR:$rs2), (MVIN3 GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in 
def : Pat<(int_riscv_mvout GPR:$rs1, GPR:$rs2), (MVOUT GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in  
def : Pat<(int_riscv_flush GPR:$rs1, GPR:$rs2), (FLUSH GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in 
def : Pat<(int_riscv_config_ld GPR:$rs1, GPR:$rs2), (CONFIG_LD GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in 
def : Pat<(int_riscv_config_st GPR:$rs1, GPR:$rs2), (CONFIG_ST GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in 
def : Pat<(int_riscv_config_ex GPR:$rs1, GPR:$rs2), (CONFIG_EX GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in
def : Pat<(int_riscv_config_norm GPR:$rs1, GPR:$rs2), (CONFIG_NORM GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in 
def : Pat<(int_riscv_preload GPR:$rs1, GPR:$rs2), (PRELOAD GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in 
def : Pat<(int_riscv_compute_preloaded GPR:$rs1, GPR:$rs2), (COMPUTE_PRELOADED GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in 
def : Pat<(int_riscv_compute_accumulated GPR:$rs1, GPR:$rs2), (COMPUTE_ACCUMULATED GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in 
def : Pat<(int_riscv_loop_ws_config_bounds GPR:$rs1, GPR:$rs2), (LOOP_WS_CONFIG_BOUNDS GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in 
def : Pat<(int_riscv_loop_ws_config_addrs_ab GPR:$rs1, GPR:$rs2), (LOOP_WS_CONFIG_ADDRS_AB GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in 
def : Pat<(int_riscv_loop_ws_config_addrs_dc GPR:$rs1, GPR:$rs2), (LOOP_WS_CONFIG_ADDRS_DC GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in 
def : Pat<(int_riscv_loop_ws_config_strides_ab GPR:$rs1, GPR:$rs2), (LOOP_WS_CONFIG_STRIDES_AB GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in 
def : Pat<(int_riscv_loop_ws_config_strides_dc GPR:$rs1, GPR:$rs2), (LOOP_WS_CONFIG_STRIDES_DC GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in 
def : Pat<(int_riscv_loop_ws GPR:$rs1, GPR:$rs2), (LOOP_WS GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in 
def : Pat<(int_riscv_loop_conv_ws GPR:$rs1, GPR:$rs2), (LOOP_CONV_WS GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in 
def : Pat<(int_riscv_loop_conv_ws_config1 GPR:$rs1, GPR:$rs2), (LOOP_CONV_WS_CONFIG1 GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in 
def : Pat<(int_riscv_loop_conv_ws_config2 GPR:$rs1, GPR:$rs2), (LOOP_CONV_WS_CONFIG2 GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in 
def : Pat<(int_riscv_loop_conv_ws_config3 GPR:$rs1, GPR:$rs2), (LOOP_CONV_WS_CONFIG3 GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in 
def : Pat<(int_riscv_loop_conv_ws_config4 GPR:$rs1, GPR:$rs2), (LOOP_CONV_WS_CONFIG4 GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in 
def : Pat<(int_riscv_loop_conv_ws_config5 GPR:$rs1, GPR:$rs2), (LOOP_CONV_WS_CONFIG5 GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasBuddyExt] in 
def : Pat<(int_riscv_loop_conv_ws_config6 GPR:$rs1, GPR:$rs2), (LOOP_CONV_WS_CONFIG6 GPR:$rs1, GPR:$rs2)>;
