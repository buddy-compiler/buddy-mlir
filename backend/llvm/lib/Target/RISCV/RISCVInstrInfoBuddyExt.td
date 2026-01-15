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
def CONFIG_LD : RVInstR<0b0100000, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "config_ld", "$rs1, $rs2"> {
  let rd = 0;
}

let Predicates = [HasBuddyExt] in
def CONFIG_ST : RVInstR<0b0100001, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "config_st", "$rs1, $rs2"> {
  let rd = 0;
}

let Predicates = [HasBuddyExt] in
def CONFIG_EX : RVInstR<0b0100010, 0b011, OPC_CUSTOM_3,(outs),
                    (ins GPR:$rs1, GPR:$rs2), "config_ex", "$rs1, $rs2"> {
  let rd = 0;
}

let Predicates = [HasBuddyExt] in
def CONFIG_NORM : RVInstR<0b0100011, 0b011, OPC_CUSTOM_3,(outs),
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

//===----------------------------------------------------------------------===//
// IME Extension Instructions
//===----------------------------------------------------------------------===//

class RVInstIME<bits<7> funct7, bits<3> funct3, dag outs, dag ins,
                string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<5> vs1;
  bits<5> vd;

  let Inst{31-25} = funct7;
  let Inst{24-20} = vs2;
  let Inst{19-15} = vs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = vd;
  let Inst{6-0} = OPC_CUSTOM_1.Value;

  let Uses = [VTYPE, VL];
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VMADOT : RVInstIME<0b1110001, 0b000,
                           (outs VRM4:$vd), 
                           (ins VRM4:$vd_in, VRM4:$vs1, VRM4:$vs2),
                           "vmadot", "$vd, $vs1, $vs2"> {
  let Constraints = "$vd = $vd_in";
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VMADOTU : RVInstIME<0b1110001, 0b011,
                            (outs VRM4:$vd),
                            (ins VRM4:$vd_in, VRM4:$vs1, VRM4:$vs2),
                            "vmadotu", "$vd, $vs1, $vs2"> {
  let Constraints = "$vd = $vd_in";
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VMADOTSU : RVInstIME<0b1110001, 0b001,
                             (outs VRM4:$vd),
                             (ins VRM4:$vd_in, VRM4:$vs1, VRM4:$vs2),
                             "vmadotsu", "$vd, $vs1, $vs2"> {
  let Constraints = "$vd = $vd_in";
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VMADOTUS : RVInstIME<0b1110001, 0b010,
                             (outs VRM4:$vd),
                             (ins VRM4:$vd_in, VRM4:$vs1, VRM4:$vs2),
                             "vmadotus", "$vd, $vs1, $vs2"> {
  let Constraints = "$vd = $vd_in";
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VFMADOT : RVInstIME<0b1110101, 0b000,
                            (outs VRM4:$vd),
                            (ins VRM4:$vd_in, VRM4:$vs1, VRM4:$vs2),
                            "vfmadot", "$vd, $vs1, $vs2"> {
  let Constraints = "$vd = $vd_in";
}

//===----------------------------------------------------------------------===//
// IME Sliding-Window Instructions
//===----------------------------------------------------------------------===//

// Integer slide-1 instructions
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VMADOT1 : RVInstIME<0b1110010, 0b000,
                            (outs VRM4:$vd),
                            (ins VRM4:$vd_in, VRM8:$vs1, VRM4:$vs2),
                            "vmadot1", "$vd, $vs1, $vs2"> {
  let Constraints = "$vd = $vd_in";
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VMADOT1U : RVInstIME<0b1110010, 0b011,
                             (outs VRM4:$vd),
                             (ins VRM4:$vd_in, VRM8:$vs1, VRM4:$vs2),
                             "vmadot1u", "$vd, $vs1, $vs2"> {
  let Constraints = "$vd = $vd_in";
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VMADOT1SU : RVInstIME<0b1110010, 0b001,
                              (outs VRM4:$vd),
                              (ins VRM4:$vd_in, VRM8:$vs1, VRM4:$vs2),
                              "vmadot1su", "$vd, $vs1, $vs2"> {
  let Constraints = "$vd = $vd_in";
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VMADOT1US : RVInstIME<0b1110010, 0b010,
                              (outs VRM4:$vd),
                              (ins VRM4:$vd_in, VRM8:$vs1, VRM4:$vs2),
                              "vmadot1us", "$vd, $vs1, $vs2"> {
  let Constraints = "$vd = $vd_in";
}

// Integer slide-2 instructions
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VMADOT2 : RVInstIME<0b1110011, 0b000,
                            (outs VRM4:$vd),
                            (ins VRM4:$vd_in, VRM8:$vs1, VRM4:$vs2),
                            "vmadot2", "$vd, $vs1, $vs2"> {
  let Constraints = "$vd = $vd_in";
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VMADOT2U : RVInstIME<0b1110011, 0b011,
                             (outs VRM4:$vd),
                             (ins VRM4:$vd_in, VRM8:$vs1, VRM4:$vs2),
                             "vmadot2u", "$vd, $vs1, $vs2"> {
  let Constraints = "$vd = $vd_in";
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VMADOT2SU : RVInstIME<0b1110011, 0b001,
                              (outs VRM4:$vd),
                              (ins VRM4:$vd_in, VRM8:$vs1, VRM4:$vs2),
                              "vmadot2su", "$vd, $vs1, $vs2"> {
  let Constraints = "$vd = $vd_in";
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VMADOT2US : RVInstIME<0b1110011, 0b010,
                              (outs VRM4:$vd),
                              (ins VRM4:$vd_in, VRM8:$vs1, VRM4:$vs2),
                              "vmadot2us", "$vd, $vs1, $vs2"> {
  let Constraints = "$vd = $vd_in";
}

// Integer slide-3 instructions
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VMADOT3 : RVInstIME<0b1110100, 0b000,
                            (outs VRM4:$vd),
                            (ins VRM4:$vd_in, VRM8:$vs1, VRM4:$vs2),
                            "vmadot3", "$vd, $vs1, $vs2"> {
  let Constraints = "$vd = $vd_in";
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VMADOT3U : RVInstIME<0b1110100, 0b011,
                             (outs VRM4:$vd),
                             (ins VRM4:$vd_in, VRM8:$vs1, VRM4:$vs2),
                             "vmadot3u", "$vd, $vs1, $vs2"> {
  let Constraints = "$vd = $vd_in";
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VMADOT3SU : RVInstIME<0b1110100, 0b001,
                              (outs VRM4:$vd),
                              (ins VRM4:$vd_in, VRM8:$vs1, VRM4:$vs2),
                              "vmadot3su", "$vd, $vs1, $vs2"> {
  let Constraints = "$vd = $vd_in";
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VMADOT3US : RVInstIME<0b1110100, 0b010,
                              (outs VRM4:$vd),
                              (ins VRM4:$vd_in, VRM8:$vs1, VRM4:$vs2),
                              "vmadot3us", "$vd, $vs1, $vs2"> {
  let Constraints = "$vd = $vd_in";
}

// Floating-point slide instructions
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VFMADOT1 : RVInstIME<0b1110110, 0b000,
                             (outs VRM4:$vd),
                             (ins VRM4:$vd_in, VRM8:$vs1, VRM4:$vs2),
                             "vfmadot1", "$vd, $vs1, $vs2"> {
  let Constraints = "$vd = $vd_in";
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VFMADOT2 : RVInstIME<0b1110111, 0b000,
                             (outs VRM4:$vd),
                             (ins VRM4:$vd_in, VRM8:$vs1, VRM4:$vs2),
                             "vfmadot2", "$vd, $vs1, $vs2"> {
  let Constraints = "$vd = $vd_in";
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VFMADOT3 : RVInstIME<0b1111000, 0b000,
                             (outs VRM4:$vd),
                             (ins VRM4:$vd_in, VRM8:$vs1, VRM4:$vs2),
                             "vfmadot3", "$vd, $vs1, $vs2"> {
  let Constraints = "$vd = $vd_in";
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

//===----------------------------------------------------------------------===//
// IME Extension Patterns
//===----------------------------------------------------------------------===//

let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv8i32 (int_riscv_ime_vmadot nxv8i32:$vd, nxv32i8:$vs1, nxv32i8:$vs2)),
            (IME_VMADOT VRM4:$vd, VRM4:$vs1, VRM4:$vs2)>;
}

let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv8i32 (int_riscv_ime_vmadotu nxv8i32:$vd, nxv32i8:$vs1, nxv32i8:$vs2)),
            (IME_VMADOTU VRM4:$vd, VRM4:$vs1, VRM4:$vs2)>;
}

let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv8i32 (int_riscv_ime_vmadotsu nxv8i32:$vd, nxv32i8:$vs1, nxv32i8:$vs2)),
            (IME_VMADOTSU VRM4:$vd, VRM4:$vs1, VRM4:$vs2)>;
}

let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv8i32 (int_riscv_ime_vmadotus nxv8i32:$vd, nxv32i8:$vs1, nxv32i8:$vs2)),
            (IME_VMADOTUS VRM4:$vd, VRM4:$vs1, VRM4:$vs2)>;
}

let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv16f16 (int_riscv_ime_vfmadot nxv16f16:$vd, nxv16f16:$vs1, nxv16f16:$vs2)),
            (IME_VFMADOT VRM4:$vd, VRM4:$vs1, VRM4:$vs2)>;
}

let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv8i32 (int_riscv_ime_vmadot1 nxv8i32:$vd, nxv64i8:$vs1, nxv32i8:$vs2)),
            (IME_VMADOT1 VRM4:$vd, VRM8:$vs1, VRM4:$vs2)>;
}

let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv8i32 (int_riscv_ime_vmadot1u nxv8i32:$vd, nxv64i8:$vs1, nxv32i8:$vs2)),
            (IME_VMADOT1U VRM4:$vd, VRM8:$vs1, VRM4:$vs2)>;
}

let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv8i32 (int_riscv_ime_vmadot1su nxv8i32:$vd, nxv64i8:$vs1, nxv32i8:$vs2)),
            (IME_VMADOT1SU VRM4:$vd, VRM8:$vs1, VRM4:$vs2)>;
}

let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv8i32 (int_riscv_ime_vmadot1us nxv8i32:$vd, nxv64i8:$vs1, nxv32i8:$vs2)),
            (IME_VMADOT1US VRM4:$vd, VRM8:$vs1, VRM4:$vs2)>;
}

let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv8i32 (int_riscv_ime_vmadot2 nxv8i32:$vd, nxv64i8:$vs1, nxv32i8:$vs2)),
            (IME_VMADOT2 VRM4:$vd, VRM8:$vs1, VRM4:$vs2)>;
}

let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv8i32 (int_riscv_ime_vmadot2u nxv8i32:$vd, nxv64i8:$vs1, nxv32i8:$vs2)),
            (IME_VMADOT2U VRM4:$vd, VRM8:$vs1, VRM4:$vs2)>;
}

let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv8i32 (int_riscv_ime_vmadot2su nxv8i32:$vd, nxv64i8:$vs1, nxv32i8:$vs2)),
            (IME_VMADOT2SU VRM4:$vd, VRM8:$vs1, VRM4:$vs2)>;
}

let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv8i32 (int_riscv_ime_vmadot2us nxv8i32:$vd, nxv64i8:$vs1, nxv32i8:$vs2)),
            (IME_VMADOT2US VRM4:$vd, VRM8:$vs1, VRM4:$vs2)>;
}

let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv8i32 (int_riscv_ime_vmadot3 nxv8i32:$vd, nxv64i8:$vs1, nxv32i8:$vs2)),
            (IME_VMADOT3 VRM4:$vd, VRM8:$vs1, VRM4:$vs2)>;
}

let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv8i32 (int_riscv_ime_vmadot3u nxv8i32:$vd, nxv64i8:$vs1, nxv32i8:$vs2)),
            (IME_VMADOT3U VRM4:$vd, VRM8:$vs1, VRM4:$vs2)>;
}

let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv8i32 (int_riscv_ime_vmadot3su nxv8i32:$vd, nxv64i8:$vs1, nxv32i8:$vs2)),
            (IME_VMADOT3SU VRM4:$vd, VRM8:$vs1, VRM4:$vs2)>;
}

let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv8i32 (int_riscv_ime_vmadot3us nxv8i32:$vd, nxv64i8:$vs1, nxv32i8:$vs2)),
            (IME_VMADOT3US VRM4:$vd, VRM8:$vs1, VRM4:$vs2)>;
}

let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv16f16 (int_riscv_ime_vfmadot1 nxv16f16:$vd, nxv32f16:$vs1, nxv16f16:$vs2)),
            (IME_VFMADOT1 VRM4:$vd, VRM8:$vs1, VRM4:$vs2)>;
}

let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv16f16 (int_riscv_ime_vfmadot2 nxv16f16:$vd, nxv32f16:$vs1, nxv16f16:$vs2)),
            (IME_VFMADOT2 VRM4:$vd, VRM8:$vs1, VRM4:$vs2)>;
}

let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv16f16 (int_riscv_ime_vfmadot3 nxv16f16:$vd, nxv32f16:$vs1, nxv16f16:$vs2)),
            (IME_VFMADOT3 VRM4:$vd, VRM8:$vs1, VRM4:$vs2)>;
}

class RVInstIMEN<bits<7> funct7, bits<3> funct3, dag outs, dag ins,
                 string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<5> rs1;  // GPR for dynamic slide value
  bits<5> vd;

  let Inst{31-25} = funct7;
  let Inst{24-20} = vs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = vd;
  let Inst{6-0} = OPC_CUSTOM_1.Value;

  let Uses = [VTYPE, VL];
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VMADOTN : RVInstIMEN<0b1111001, 0b000,
                             (outs VRM4:$vd),
                             (ins VRM4:$vd_in, VRM8:$vs1, VRM4:$vs2, GPR:$rs1),
                             "vmadotn", "$vd, $vs1, $vs2, $rs1"> {
  let Constraints = "$vd = $vd_in";
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VMADOTNU : RVInstIMEN<0b1111001, 0b011,
                              (outs VRM4:$vd),
                              (ins VRM4:$vd_in, VRM8:$vs1, VRM4:$vs2, GPR:$rs1),
                              "vmadotnu", "$vd, $vs1, $vs2, $rs1"> {
  let Constraints = "$vd = $vd_in";
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VMADOTNSU : RVInstIMEN<0b1111001, 0b001,
                               (outs VRM4:$vd),
                               (ins VRM4:$vd_in, VRM8:$vs1, VRM4:$vs2, GPR:$rs1),
                               "vmadotnsu", "$vd, $vs1, $vs2, $rs1"> {
  let Constraints = "$vd = $vd_in";
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VMADOTNUS : RVInstIMEN<0b1111001, 0b010,
                               (outs VRM4:$vd),
                               (ins VRM4:$vd_in, VRM8:$vs1, VRM4:$vs2, GPR:$rs1),
                               "vmadotnus", "$vd, $vs1, $vs2, $rs1"> {
  let Constraints = "$vd = $vd_in";
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VFMADOTN : RVInstIMEN<0b1111010, 0b000,
                              (outs VRM4:$vd),
                              (ins VRM4:$vd_in, VRM8:$vs1, VRM4:$vs2, GPR:$rs1),
                              "vfmadotn", "$vd, $vs1, $vs2, $rs1"> {
  let Constraints = "$vd = $vd_in";
}

let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv8i32 (int_riscv_ime_vmadotn nxv8i32:$vd, nxv64i8:$vs1, nxv32i8:$vs2, GPR:$rs1)),
            (IME_VMADOTN VRM4:$vd, VRM8:$vs1, VRM4:$vs2, GPR:$rs1)>;
}

let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv8i32 (int_riscv_ime_vmadotnu nxv8i32:$vd, nxv64i8:$vs1, nxv32i8:$vs2, GPR:$rs1)),
            (IME_VMADOTNU VRM4:$vd, VRM8:$vs1, VRM4:$vs2, GPR:$rs1)>;
}

let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv8i32 (int_riscv_ime_vmadotnsu nxv8i32:$vd, nxv64i8:$vs1, nxv32i8:$vs2, GPR:$rs1)),
            (IME_VMADOTNSU VRM4:$vd, VRM8:$vs1, VRM4:$vs2, GPR:$rs1)>;
}

let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv8i32 (int_riscv_ime_vmadotnus nxv8i32:$vd, nxv64i8:$vs1, nxv32i8:$vs2, GPR:$rs1)),
            (IME_VMADOTNUS VRM4:$vd, VRM8:$vs1, VRM4:$vs2, GPR:$rs1)>;
}

let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv16f16 (int_riscv_ime_vfmadotn nxv16f16:$vd, nxv32f16:$vs1, nxv16f16:$vs2, GPR:$rs1)),
            (IME_VFMADOTN VRM4:$vd, VRM8:$vs1, VRM4:$vs2, GPR:$rs1)>;
}
