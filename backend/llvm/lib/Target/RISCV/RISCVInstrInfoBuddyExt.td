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
def CONFIG : RVInstR<0b0000000, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "config", "$rs1, $rs2"> {
  let rd = 0;
}

let hasSideEffects = 1, mayLoad = 1, mayStore =1, Predicates = [HasBuddyExt] in
def PRELOAD : RVInstR<0b0000110, 0b011, OPC_CUSTOM_3, (outs),
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
def LOOP_WS_CONFIG_BOUNDS : RVInstR<0b0001001, 0b011, OPC_CUSTOM_3, (outs),
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
def : Pat<(int_riscv_config GPR:$rs1, GPR:$rs2), (CONFIG GPR:$rs1, GPR:$rs2)>;

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

// int16 vmadot patterns
let Predicates = [HasBuddyExt] in {
  def : Pat<(nxv8i32 (int_riscv_ime_vmadot nxv8i32:$vd, nxv16i16:$vs1, nxv16i16:$vs2)),
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

// Four-register IME "dot-n" ops: InstFormatR4 (vs1 @ 31-27, vs2 @ 24-20, rs1,
// vd). funct2 is the low 2 bits of the prior 7-bit funct7 constant
// (0b1111001 -> 0b01 for integer, 0b1111010 -> 0b10 for fp).
class RVInstIMEN4<bits<2> funct2, bits<3> funct3, dag outs, dag ins,
                  string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR4> {
  bits<5> vs1;
  bits<5> vs2;
  bits<5> rs1;
  bits<5> vd;

  let Inst{31-27} = vs1;
  let Inst{26-25} = funct2;
  let Inst{24-20} = vs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = vd;
  let Inst{6-0} = OPC_CUSTOM_1.Value;

  let Uses = [VTYPE, VL];
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VMADOTN : RVInstIMEN4<0b01, 0b000,
                              (outs VRM4:$vd),
                              (ins VRM4:$vd_in, VRM8:$vs1, VRM4:$vs2, GPR:$rs1),
                              "vmadotn", "$vd, $vs1, $vs2, $rs1"> {
  let Constraints = "$vd = $vd_in";
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VMADOTNU : RVInstIMEN4<0b01, 0b011,
                               (outs VRM4:$vd),
                               (ins VRM4:$vd_in, VRM8:$vs1, VRM4:$vs2, GPR:$rs1),
                               "vmadotnu", "$vd, $vs1, $vs2, $rs1"> {
  let Constraints = "$vd = $vd_in";
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VMADOTNSU : RVInstIMEN4<0b01, 0b001,
                                (outs VRM4:$vd),
                                (ins VRM4:$vd_in, VRM8:$vs1, VRM4:$vs2, GPR:$rs1),
                                "vmadotnsu", "$vd, $vs1, $vs2, $rs1"> {
  let Constraints = "$vd = $vd_in";
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VMADOTNUS : RVInstIMEN4<0b01, 0b010,
                                (outs VRM4:$vd),
                                (ins VRM4:$vd_in, VRM8:$vs1, VRM4:$vs2, GPR:$rs1),
                                "vmadotnus", "$vd, $vs1, $vs2, $rs1"> {
  let Constraints = "$vd = $vd_in";
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def IME_VFMADOTN : RVInstIMEN4<0b10, 0b000,
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

//===----------------------------------------------------------------------===//
// AME (RISC-V Matrix Extension) 64-bit Instructions
//===----------------------------------------------------------------------===//
// Reference: RISC-V Matrix Extension Specification
// 64-bit encoding format with prefix 0111111 at bits [6:0]
//
// Register Model:
//   - TileReg (tr0-tr7): Input matrices A and B, MLEN bits each
//   - AccReg (acc0-acc7): Accumulator matrix C, MLEN×AMUL bits each
//
// Data Flow:
//   Memory --[mlae/mlbe]--> TileReg --[mma/mwma/mqma]--> AccReg --[msce]--> Memory
//
// Matrix Multiplication Instruction Format (64-bit):
// | 63:59 | 58  | 57:55 | 54:52 | 51:49 | 48:47 | 46:44 | 43:39   | 38:32   |
// | sps   | sp  | typ2  | typ1  | typd  | bma   | frm   | funct5  | opcode  |
// | 31:26   | 25 | 24:20 | 19:15 | 14:12  | 11:7 | 6:0    |
// | funct6  | fp | ms2   | ms1   | funct3 | md   | suffix |
//
// suffix = 0111111 (AME prefix)
// funct3 = 100 (matrix multiplication)
//
// Widening Instructions (required for AMUL > 1):
//   - mqma.b.mm: int8 → int32 (quad-widen, AMUL ≥ 4, mmi8i32 MANDATORY)
//   - mwma.h.mm: int16 → int32 (double-widen, AMUL ≥ 2)
//   - mwma.w.mm: int32 → int64 (double-widen, AMUL ≥ 2)
//===----------------------------------------------------------------------===//

// AME Opcode suffix (bits [6:0])
def OPC_AME : RISCVOpcode<"OPC_AME", 0b0111111>;

//===----------------------------------------------------------------------===//
// AME Custom Operand Types for Tile Register Indices
//===----------------------------------------------------------------------===//
// These operand types allow intrinsics to pass immediate indices (0-7)
// for tile and accumulator registers. The AsmString of pseudo instructions
// hardcodes "acc" and "tr" prefixes so that indices 0-7 are printed as
// acc0-acc7 and tr0-tr7 respectively, without modifying LLVM submodule.

// AsmOperandClass for tile register index (0-7)
def AMETileIndexAsmOperand : AsmOperandClass {
  let Name = "AMETileIndex";
  let RenderMethod = "addImmOperands";
  let PredicateMethod = "isUImm3";
  let DiagnosticType = "InvalidAMETileIndex";
}

// AsmOperandClass for accumulator register index (0-7)
def AMEAccIndexAsmOperand : AsmOperandClass {
  let Name = "AMEAccIndex";
  let RenderMethod = "addImmOperands";
  let PredicateMethod = "isUImm3";
  let DiagnosticType = "InvalidAMEAccIndex";
}

// Operand type for TileReg index (0-7), printed with "tr" prefix in AsmString
def AMETileIndex : RISCVOp<i64> {
  let ParserMatchClass = AMETileIndexAsmOperand;
  let DecoderMethod = "decodeUImmOperand<3>";
  let OperandType = "OPERAND_UIMM3";
}

// Operand type for AccReg index (0-7), printed with "acc" prefix in AsmString
def AMEAccIndex : RISCVOp<i64> {
  let ParserMatchClass = AMEAccIndexAsmOperand;
  let DecoderMethod = "decodeUImmOperand<3>";
  let OperandType = "OPERAND_UIMM3";
}

//===----------------------------------------------------------------------===//
// AME 64-bit Instruction Format Base Class
//===----------------------------------------------------------------------===//

// Base class for AME 64-bit matrix multiplication instructions
// Uses TileReg for inputs (ms1, ms2) and AccReg for output (md)
class RVInstAME64<dag outs, dag ins, string opcodestr, string argstr>
    : RVInst64<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  // Low 32 bits
  bits<5> md;
  bits<5> ms1;
  bits<5> ms2;
  bits<6> funct6;
  bit fp;
  bits<3> funct3;

  // High 32 bits
  bits<5> funct5;
  bits<3> frm;
  bits<2> bma;
  bits<3> typd;
  bits<3> typ1;
  bits<3> typ2;
  bit sp;
  bits<5> sps;
  bits<7> opcode_hi;  // [38:32]

  // Encode low 32 bits (suffix word)
  let Inst{6-0} = 0b0111111;   // AME prefix
  let Inst{11-7} = md;
  let Inst{14-12} = funct3;
  let Inst{19-15} = ms1;
  let Inst{24-20} = ms2;
  let Inst{25} = fp;
  let Inst{31-26} = funct6;

  // Encode high 32 bits (opcode word)
  let Inst{38-32} = opcode_hi;
  let Inst{43-39} = funct5;
  let Inst{46-44} = frm;
  let Inst{48-47} = bma;
  let Inst{51-49} = typd;
  let Inst{54-52} = typ1;
  let Inst{57-55} = typ2;
  let Inst{58} = sp;
  let Inst{63-59} = sps;
}

//===----------------------------------------------------------------------===//
// AME Matrix Multiplication Instructions
//===----------------------------------------------------------------------===//
// Format: mma.{h|w|dw}.mm acc, tr, tr
// Semantics: acc = acc + tr1 * tr2
//
// Data Flow: TileReg × TileReg → AccReg (accumulate)
//   - ms1: TileReg for matrix A
//   - ms2: TileReg for matrix B
//   - md:  AccReg for accumulation result C
//
// typ1/typ2/typd encoding:
//   000 = int8 (b), 001 = int16 (h), 010 = int32 (w), 011 = int64 (dw)
//   100 = use mtype.msew, 111 = int4
//
// funct5 encoding:
//   00001 = mma (signed, no saturation)
//   00010 = mwma (double-widen)
//   00100 = mqma (quad-widen)
//   10001 = msma (signed, saturated)
//===----------------------------------------------------------------------===//

// No-widen matrix multiply-accumulate: acc = acc + tr1 * tr2
// Input and output have the same element width
class AME_MMA_MM<bits<3> typd_val, bits<3> typ1_val, bits<3> typ2_val,
                 string opcodestr>
    : RVInstAME64<(outs AccReg:$md),
                  (ins AccReg:$md_in, TileReg:$ms1, TileReg:$ms2),
                  opcodestr, "$md, $ms1, $ms2"> {
  let Constraints = "$md = $md_in";  // Accumulator constraint
  let funct6 = 0b000000;
  let fp = 0;
  let funct3 = 0b100;        // Matrix multiplication
  let opcode_hi = 0b0000011; // xxyyy11 where xx=00, yyy=001
  let funct5 = 0b00001;      // mma (signed, no saturation)
  let frm = 0b000;
  let bma = 0b00;            // Default: not agnostic
  let typd = typd_val;
  let typ1 = typ1_val;
  let typ2 = typ2_val;
  let sp = 0;                // No sparsity
  let sps = 0b00000;
}

// mma.h.mm - int16 × int16 → int16 accumulate (no widen)
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def AME_MMA_H_MM : AME_MMA_MM<0b001, 0b001, 0b001, "mma.h.mm">;

// mma.w.mm - int32 × int32 → int32 accumulate (no widen)
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def AME_MMA_W_MM : AME_MMA_MM<0b010, 0b010, 0b010, "mma.w.mm">;

// mma.dw.mm - int64 × int64 → int64 accumulate (no widen)
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def AME_MMA_DW_MM : AME_MMA_MM<0b011, 0b011, 0b011, "mma.dw.mm">;

//===----------------------------------------------------------------------===//
// AME Widening Matrix Multiplication Instructions
//===----------------------------------------------------------------------===//
// Double-widen: output element is 2× width of input elements
// Quad-widen: output element is 4× width of input elements
//
// These require AMUL ≥ 2 (double) or AMUL ≥ 4 (quad) to ensure
// accumulator has sufficient width.
//
// Mandatory: mqma.b.mm (int8→int32) for mmi8i32 feature
//===----------------------------------------------------------------------===//

// Double-widen: acc = acc + tr1 * tr2, output is 2× width
// Requires AMUL ≥ 2
class AME_MWMA_MM<bits<3> typd_val, bits<3> typ1_val, bits<3> typ2_val,
                  string opcodestr>
    : RVInstAME64<(outs AccReg:$md),
                  (ins AccReg:$md_in, TileReg:$ms1, TileReg:$ms2),
                  opcodestr, "$md, $ms1, $ms2"> {
  let Constraints = "$md = $md_in";
  let funct6 = 0b000000;
  let fp = 0;
  let funct3 = 0b100;
  let opcode_hi = 0b0000011;
  let funct5 = 0b00010;      // mwma (double-widen)
  let frm = 0b000;
  let bma = 0b00;
  let typd = typd_val;
  let typ1 = typ1_val;
  let typ2 = typ2_val;
  let sp = 0;
  let sps = 0b00000;
}

// mwma.h.mm - int16 × int16 → int32 accumulate (double-widen)
// Requires: AMUL ≥ 2, mmi16i32 feature
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def AME_MWMA_H_MM : AME_MWMA_MM<0b010, 0b001, 0b001, "mwma.h.mm">;

// mwma.w.mm - int32 × int32 → int64 accumulate (double-widen)
// Requires: AMUL ≥ 2, mmi32i64 feature
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def AME_MWMA_W_MM : AME_MWMA_MM<0b011, 0b010, 0b010, "mwma.w.mm">;

// Quad-widen: acc = acc + tr1 * tr2, output is 4× width
// Requires AMUL ≥ 4
class AME_MQMA_MM<bits<3> typd_val, bits<3> typ1_val, bits<3> typ2_val,
                  string opcodestr>
    : RVInstAME64<(outs AccReg:$md),
                  (ins AccReg:$md_in, TileReg:$ms1, TileReg:$ms2),
                  opcodestr, "$md, $ms1, $ms2"> {
  let Constraints = "$md = $md_in";
  let funct6 = 0b000000;
  let fp = 0;
  let funct3 = 0b100;
  let opcode_hi = 0b0000011;
  let funct5 = 0b00100;      // mqma (quad-widen)
  let frm = 0b000;
  let bma = 0b00;
  let typd = typd_val;
  let typ1 = typ1_val;
  let typ2 = typ2_val;
  let sp = 0;
  let sps = 0b00000;
}

// mqma.b.mm - int8 × int8 → int32 accumulate (quad-widen)
// MANDATORY for mmi8i32 feature (required by Spec)
// Requires: AMUL ≥ 4
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def AME_MQMA_B_MM : AME_MQMA_MM<0b010, 0b000, 0b000, "mqma.b.mm">;

//===----------------------------------------------------------------------===//
// AME Configuration Instructions
//===----------------------------------------------------------------------===//
// Configuration instruction format (64-bit):
// | 63:43           | 42:39  | 38:32   |
// | imm[31:11]      | funct4 | opcode  |
// | 31:26  | 25:20       | 19:15 | 14:12  | 11:7 | 6:0    |
// | funct6 | imm[10:5]   | rs1   | funct3 | rd   | suffix |
//===----------------------------------------------------------------------===//

class RVInstAMEConfig64<bits<6> funct6_val, string opcodestr>
    : RVInst64<(outs GPR:$rd), (ins GPR:$rs1),
               opcodestr, "$rd, $rs1", [], InstFormatOther> {
  bits<5> rd;
  bits<5> rs1;

  let Inst{6-0} = 0b0111111;   // AME prefix
  let Inst{11-7} = rd;
  let Inst{14-12} = 0b000;     // funct3 for config
  let Inst{19-15} = rs1;
  let Inst{25-20} = 0b000000;
  let Inst{31-26} = funct6_val;
  let Inst{38-32} = 0b0000011; // opcode
  let Inst{42-39} = 0b0000;    // funct4
  let Inst{63-43} = 0;         // imm[31:11] = 0
}

class RVInstAMEConfigImm64<bits<6> funct6_val, string opcodestr>
    : RVInst64<(outs GPR:$rd), (ins uimm32:$imm),
               opcodestr, "$rd, $imm", [], InstFormatOther> {
  bits<5> rd;
  bits<32> imm;

  let Inst{6-0} = 0b0111111;   // AME prefix
  let Inst{11-7} = rd;
  let Inst{14-12} = 0b000;     // funct3 for config
  let Inst{19-15} = imm{4-0};  // imm[4:0] in rs1 field
  let Inst{25-20} = imm{10-5}; // imm[10:5]
  let Inst{31-26} = funct6_val;
  let Inst{38-32} = 0b0000011; // opcode
  let Inst{42-39} = 0b0000;    // funct4
  let Inst{63-43} = imm{31-11}; // imm[31:11]
}

// msettilem - set tile M dimension from register
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def AME_MSETTILEM : RVInstAMEConfig64<0b000100, "msettilem">;

// msettilemi - set tile M dimension from immediate
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def AME_MSETTILEMI : RVInstAMEConfigImm64<0b000101, "msettilemi">;

// msettilen - set tile N dimension from register
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def AME_MSETTILEN : RVInstAMEConfig64<0b001000, "msettilen">;

// msettileni - set tile N dimension from immediate
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def AME_MSETTILENI : RVInstAMEConfigImm64<0b001001, "msettileni">;

// msettilek - set tile K dimension from register
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def AME_MSETTILEK : RVInstAMEConfig64<0b001100, "msettilek">;

// msettileki - set tile K dimension from immediate
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def AME_MSETTILEKI : RVInstAMEConfigImm64<0b001101, "msettileki">;

//===----------------------------------------------------------------------===//
// AME Load/Store Instructions
//===----------------------------------------------------------------------===//
// Load/Store instruction format (64-bit):
// | 63:51 | 50:49 | 48:47 | 46:44 | 43:39  | 38:32   |
// | resv  | mt    | bma   | eew   | funct5 | opcode  |
// | 31:26  | 25 | 24:20 | 19:15 | 14:12  | 11:7    | 6:0    |
// | funct6 | ls | rs2   | rs1   | funct3 | ms3/md  | suffix |
//
// mt (matrix type): 00=accumulator(C), 01=left(A), 10=right(B), 11=result
// eew (element width): 000=8b, 001=16b, 010=32b, 011=64b
//
// Register Usage:
//   - mt=01 (A) or mt=10 (B): Uses TileReg
//   - mt=00 (C): Uses AccReg
//
// Data Flow Examples:
//   mlae32.m tr0, (a0), a1    # Load matrix A into TileReg
//   mlbe32.m tr1, (a0), a1    # Load matrix B into TileReg
//   mqma.b.mm acc0, tr0, tr1  # Compute: AccReg = AccReg + TileReg × TileReg
//   msce32.m acc0, (a0), a1   # Store AccReg to memory
//===----------------------------------------------------------------------===//

// Load into TileReg (for matrix A and B)
class RVInstAMELoadTile64<bits<2> mt_val, bits<3> eew_val, string opcodestr>
    : RVInst64<(outs TileReg:$md), (ins GPR:$rs1, GPR:$rs2),
               opcodestr, "$md, $rs1, $rs2", [], InstFormatOther> {
  bits<5> md;
  bits<5> rs1;
  bits<5> rs2;

  let Inst{6-0} = 0b0111111;   // AME prefix
  let Inst{11-7} = md;
  let Inst{14-12} = 0b001;     // funct3 for load/store
  let Inst{19-15} = rs1;
  let Inst{24-20} = rs2;
  let Inst{25} = 0;            // ls = 0 for load
  let Inst{31-26} = 0b000000;  // funct6
  let Inst{38-32} = 0b0000011; // opcode
  let Inst{43-39} = 0b00000;   // funct5
  let Inst{46-44} = eew_val;   // eew
  let Inst{48-47} = 0b00;      // bma
  let Inst{50-49} = mt_val;    // mt
  let Inst{63-51} = 0;         // reserved
}

// Load into AccReg (for accumulator C)
class RVInstAMELoadAcc64<bits<2> mt_val, bits<3> eew_val, string opcodestr>
    : RVInst64<(outs AccReg:$md), (ins GPR:$rs1, GPR:$rs2),
               opcodestr, "$md, $rs1, $rs2", [], InstFormatOther> {
  bits<5> md;
  bits<5> rs1;
  bits<5> rs2;

  let Inst{6-0} = 0b0111111;
  let Inst{11-7} = md;
  let Inst{14-12} = 0b001;
  let Inst{19-15} = rs1;
  let Inst{24-20} = rs2;
  let Inst{25} = 0;
  let Inst{31-26} = 0b000000;
  let Inst{38-32} = 0b0000011;
  let Inst{43-39} = 0b00000;
  let Inst{46-44} = eew_val;
  let Inst{48-47} = 0b00;
  let Inst{50-49} = mt_val;
  let Inst{63-51} = 0;
}

// Store from TileReg (for matrix A and B)
class RVInstAMEStoreTile64<bits<2> mt_val, bits<3> eew_val, string opcodestr>
    : RVInst64<(outs), (ins TileReg:$ms3, GPR:$rs1, GPR:$rs2),
               opcodestr, "$ms3, $rs1, $rs2", [], InstFormatOther> {
  bits<5> ms3;
  bits<5> rs1;
  bits<5> rs2;

  let Inst{6-0} = 0b0111111;
  let Inst{11-7} = ms3;
  let Inst{14-12} = 0b001;
  let Inst{19-15} = rs1;
  let Inst{24-20} = rs2;
  let Inst{25} = 1;            // ls = 1 for store
  let Inst{31-26} = 0b000000;
  let Inst{38-32} = 0b0000011;
  let Inst{43-39} = 0b00000;
  let Inst{46-44} = eew_val;
  let Inst{48-47} = 0b00;
  let Inst{50-49} = mt_val;
  let Inst{63-51} = 0;
}

// Store from AccReg (for accumulator C)
class RVInstAMEStoreAcc64<bits<2> mt_val, bits<3> eew_val, string opcodestr>
    : RVInst64<(outs), (ins AccReg:$ms3, GPR:$rs1, GPR:$rs2),
               opcodestr, "$ms3, $rs1, $rs2", [], InstFormatOther> {
  bits<5> ms3;
  bits<5> rs1;
  bits<5> rs2;

  let Inst{6-0} = 0b0111111;
  let Inst{11-7} = ms3;
  let Inst{14-12} = 0b001;
  let Inst{19-15} = rs1;
  let Inst{24-20} = rs2;
  let Inst{25} = 1;
  let Inst{31-26} = 0b000000;
  let Inst{38-32} = 0b0000011;
  let Inst{43-39} = 0b00000;
  let Inst{46-44} = eew_val;
  let Inst{48-47} = 0b00;
  let Inst{50-49} = mt_val;
  let Inst{63-51} = 0;
}

//===----------------------------------------------------------------------===//
// Load matrix A (left operand) into TileReg - mlae*.m
// Syntax: mlae{8|16|32|64}.m tr, (rs1), rs2
//   tr:  Destination TileReg
//   rs1: Base address (GPR)
//   rs2: Row stride in bytes (GPR)
//===----------------------------------------------------------------------===//
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 1, mayStore = 0 in {
  def AME_MLAE8_M  : RVInstAMELoadTile64<0b01, 0b000, "mlae8.m">;
  def AME_MLAE16_M : RVInstAMELoadTile64<0b01, 0b001, "mlae16.m">;
  def AME_MLAE32_M : RVInstAMELoadTile64<0b01, 0b010, "mlae32.m">;
  def AME_MLAE64_M : RVInstAMELoadTile64<0b01, 0b011, "mlae64.m">;
}

//===----------------------------------------------------------------------===//
// Load matrix B (right operand) into TileReg - mlbe*.m
// Syntax: mlbe{8|16|32|64}.m tr, (rs1), rs2
//===----------------------------------------------------------------------===//
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 1, mayStore = 0 in {
  def AME_MLBE8_M  : RVInstAMELoadTile64<0b10, 0b000, "mlbe8.m">;
  def AME_MLBE16_M : RVInstAMELoadTile64<0b10, 0b001, "mlbe16.m">;
  def AME_MLBE32_M : RVInstAMELoadTile64<0b10, 0b010, "mlbe32.m">;
  def AME_MLBE64_M : RVInstAMELoadTile64<0b10, 0b011, "mlbe64.m">;
}

//===----------------------------------------------------------------------===//
// Load matrix C (accumulator) into AccReg - mlce*.m
// Syntax: mlce{8|16|32|64}.m acc, (rs1), rs2
//   acc: Destination AccReg (MLEN×AMUL bits)
// Note: eew here refers to the output element width after widening
//===----------------------------------------------------------------------===//
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 1, mayStore = 0 in {
  def AME_MLCE8_M  : RVInstAMELoadAcc64<0b00, 0b000, "mlce8.m">;
  def AME_MLCE16_M : RVInstAMELoadAcc64<0b00, 0b001, "mlce16.m">;
  def AME_MLCE32_M : RVInstAMELoadAcc64<0b00, 0b010, "mlce32.m">;
  def AME_MLCE64_M : RVInstAMELoadAcc64<0b00, 0b011, "mlce64.m">;
}

//===----------------------------------------------------------------------===//
// Store matrix A from TileReg - msae*.m
// Syntax: msae{8|16|32|64}.m tr, (rs1), rs2
//===----------------------------------------------------------------------===//
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 1 in {
  def AME_MSAE8_M  : RVInstAMEStoreTile64<0b01, 0b000, "msae8.m">;
  def AME_MSAE16_M : RVInstAMEStoreTile64<0b01, 0b001, "msae16.m">;
  def AME_MSAE32_M : RVInstAMEStoreTile64<0b01, 0b010, "msae32.m">;
  def AME_MSAE64_M : RVInstAMEStoreTile64<0b01, 0b011, "msae64.m">;
}

//===----------------------------------------------------------------------===//
// Store matrix B from TileReg - msbe*.m
// Syntax: msbe{8|16|32|64}.m tr, (rs1), rs2
//===----------------------------------------------------------------------===//
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 1 in {
  def AME_MSBE8_M  : RVInstAMEStoreTile64<0b10, 0b000, "msbe8.m">;
  def AME_MSBE16_M : RVInstAMEStoreTile64<0b10, 0b001, "msbe16.m">;
  def AME_MSBE32_M : RVInstAMEStoreTile64<0b10, 0b010, "msbe32.m">;
  def AME_MSBE64_M : RVInstAMEStoreTile64<0b10, 0b011, "msbe64.m">;
}

//===----------------------------------------------------------------------===//
// Store matrix C (accumulator) from AccReg - msce*.m
// Syntax: msce{8|16|32|64}.m acc, (rs1), rs2
//   acc: Source AccReg (MLEN×AMUL bits)
// This is the primary store for computation results
//===----------------------------------------------------------------------===//
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 1 in {
  def AME_MSCE8_M  : RVInstAMEStoreAcc64<0b00, 0b000, "msce8.m">;
  def AME_MSCE16_M : RVInstAMEStoreAcc64<0b00, 0b001, "msce16.m">;
  def AME_MSCE32_M : RVInstAMEStoreAcc64<0b00, 0b010, "msce32.m">;
  def AME_MSCE64_M : RVInstAMEStoreAcc64<0b00, 0b011, "msce64.m">;
}

//===----------------------------------------------------------------------===//
// AME Extension Pattern Matching
//===----------------------------------------------------------------------===//
// Connect LLVM intrinsics to AME machine instructions
//
// Register Model:
//   - TileReg (tr0-tr7): For input matrices A and B
//   - AccReg (acc0-acc7): For output/accumulator C
//
// Typical Data Flow for Matrix Multiplication (e.g., int8→int32):
//   1. msettilem/n/k: Configure tile dimensions
//   2. mlae8.m tr0, (a0), stride_a: Load matrix A into TileReg
//   3. mlbe8.m tr1, (a1), stride_b: Load matrix B into TileReg
//   4. mqma.b.mm acc0, tr0, tr1: Compute acc0 = acc0 + tr0 × tr1
//   5. msce32.m acc0, (a2), stride_c: Store AccReg to memory
//
// Note: For quad-widen (int8→int32), input uses 8-bit load (mlae8/mlbe8)
// but output uses 32-bit store (msce32) because AMUL=4 widens the output.
//===----------------------------------------------------------------------===//

// Configuration instruction patterns
// These use GPR operands and return the actual configured value
let Predicates = [HasBuddyExt] in {
  // msettilem - set tile M dimension from GPR, returns actual value
  def : Pat<(i64 (int_riscv_buddy_msettilem GPR:$rs1)),
            (AME_MSETTILEM GPR:$rs1)>;

  // msettilen - set tile N dimension from GPR, returns actual value
  def : Pat<(i64 (int_riscv_buddy_msettilen GPR:$rs1)),
            (AME_MSETTILEN GPR:$rs1)>;

  // msettilek - set tile K dimension from GPR, returns actual value
  def : Pat<(i64 (int_riscv_buddy_msettilek GPR:$rs1)),
            (AME_MSETTILEK GPR:$rs1)>;
}

//===----------------------------------------------------------------------===//
// AME Additional Integer Matrix Operations
//===----------------------------------------------------------------------===//
// Unsigned and mixed-sign matrix multiplication variants
//
// Naming convention:
//   - mma: signed × signed
//   - mmau: unsigned × unsigned
//   - mmasu: signed × unsigned
//   - mmaus: unsigned × signed
//===----------------------------------------------------------------------===//

// Unsigned no-widen multiply-accumulate
class AME_MMAU_MM<bits<3> typd_val, bits<3> typ1_val, bits<3> typ2_val,
                  string opcodestr>
    : RVInstAME64<(outs AccReg:$md),
                  (ins AccReg:$md_in, TileReg:$ms1, TileReg:$ms2),
                  opcodestr, "$md, $ms1, $ms2"> {
  let Constraints = "$md = $md_in";
  let funct6 = 0b000000;
  let fp = 0;
  let funct3 = 0b100;
  let opcode_hi = 0b0000011;
  let funct5 = 0b01001;      // mmau (unsigned)
  let frm = 0b000;
  let bma = 0b00;
  let typd = typd_val;
  let typ1 = typ1_val;
  let typ2 = typ2_val;
  let sp = 0;
  let sps = 0b00000;
}

// mmau.b.mm - uint8 × uint8 → uint32 (for unsigned int8)
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def AME_MMAU_B_MM : AME_MMAU_MM<0b010, 0b000, 0b000, "mmau.b.mm">;

// Unsigned quad-widen multiply-accumulate
class AME_MQMAU_MM<bits<3> typd_val, bits<3> typ1_val, bits<3> typ2_val,
                   string opcodestr>
    : RVInstAME64<(outs AccReg:$md),
                  (ins AccReg:$md_in, TileReg:$ms1, TileReg:$ms2),
                  opcodestr, "$md, $ms1, $ms2"> {
  let Constraints = "$md = $md_in";
  let funct6 = 0b000000;
  let fp = 0;
  let funct3 = 0b100;
  let opcode_hi = 0b0000011;
  let funct5 = 0b01100;      // mqmau (unsigned quad-widen)
  let frm = 0b000;
  let bma = 0b00;
  let typd = typd_val;
  let typ1 = typ1_val;
  let typ2 = typ2_val;
  let sp = 0;
  let sps = 0b00000;
}

// mqmau.b.mm - uint8 × uint8 → uint32 (quad-widen unsigned)
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def AME_MQMAU_B_MM : AME_MQMAU_MM<0b010, 0b000, 0b000, "mqmau.b.mm">;

// Mixed-sign: signed × unsigned
class AME_MQMASU_MM<bits<3> typd_val, bits<3> typ1_val, bits<3> typ2_val,
                    string opcodestr>
    : RVInstAME64<(outs AccReg:$md),
                  (ins AccReg:$md_in, TileReg:$ms1, TileReg:$ms2),
                  opcodestr, "$md, $ms1, $ms2"> {
  let Constraints = "$md = $md_in";
  let funct6 = 0b000000;
  let fp = 0;
  let funct3 = 0b100;
  let opcode_hi = 0b0000011;
  let funct5 = 0b00101;      // mqmasu (signed × unsigned quad-widen)
  let frm = 0b000;
  let bma = 0b00;
  let typd = typd_val;
  let typ1 = typ1_val;
  let typ2 = typ2_val;
  let sp = 0;
  let sps = 0b00000;
}

// mqmasu.b.mm - int8 × uint8 → int32
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def AME_MQMASU_B_MM : AME_MQMASU_MM<0b010, 0b000, 0b000, "mqmasu.b.mm">;

// Mixed-sign: unsigned × signed
class AME_MQMAUS_MM<bits<3> typd_val, bits<3> typ1_val, bits<3> typ2_val,
                    string opcodestr>
    : RVInstAME64<(outs AccReg:$md),
                  (ins AccReg:$md_in, TileReg:$ms1, TileReg:$ms2),
                  opcodestr, "$md, $ms1, $ms2"> {
  let Constraints = "$md = $md_in";
  let funct6 = 0b000000;
  let fp = 0;
  let funct3 = 0b100;
  let opcode_hi = 0b0000011;
  let funct5 = 0b00110;      // mqmaus (unsigned × signed quad-widen)
  let frm = 0b000;
  let bma = 0b00;
  let typd = typd_val;
  let typ1 = typ1_val;
  let typ2 = typ2_val;
  let sp = 0;
  let sps = 0b00000;
}

// mqmaus.b.mm - uint8 × int8 → int32
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def AME_MQMAUS_B_MM : AME_MQMAUS_MM<0b010, 0b000, 0b000, "mqmaus.b.mm">;

//===----------------------------------------------------------------------===//
// AME Zero/Initialize Instructions
//===----------------------------------------------------------------------===//
// mzero - Zero out an accumulation register
// Useful for initializing before accumulation loop

class AME_MZERO<string opcodestr>
    : RVInst64<(outs AccReg:$md), (ins),
               opcodestr, "$md", [], InstFormatOther> {
  bits<5> md;

  let Inst{6-0} = 0b0111111;
  let Inst{11-7} = md;
  let Inst{14-12} = 0b101;     // funct3 for arithmetic
  let Inst{19-15} = 0b00000;
  let Inst{24-20} = 0b00000;
  let Inst{25} = 0;
  let Inst{31-26} = 0b000000;
  let Inst{38-32} = 0b0000011;
  let Inst{43-39} = 0b10000;   // funct5 for zero
  let Inst{63-44} = 0;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def AME_MZERO_M : AME_MZERO<"mzero.m">;

//===----------------------------------------------------------------------===//
// AME Intrinsic Pattern Matching
//===----------------------------------------------------------------------===//
// These patterns map LLVM intrinsics to AME machine instructions.
//
// Note: AME intrinsics use i64 indices for tile registers instead of
// actual register operands. This is because the MLIR lowering generates
// calls with constant indices that get mapped to physical registers
// at the final code generation stage.
//
// For tile-based operations, the tile register index (0-7) is encoded
// directly into the instruction's register field.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Configuration Instruction Patterns
//===----------------------------------------------------------------------===//

// msettilem - set tile M dimension from GPR
let Predicates = [HasBuddyExt] in {
  def : Pat<(i64 (int_riscv_buddy_msettilem i64:$rs1)),
            (AME_MSETTILEM GPR:$rs1)>;
}

// msettilen - set tile N dimension from GPR
let Predicates = [HasBuddyExt] in {
  def : Pat<(i64 (int_riscv_buddy_msettilen i64:$rs1)),
            (AME_MSETTILEN GPR:$rs1)>;
}

// msettilek - set tile K dimension from GPR
let Predicates = [HasBuddyExt] in {
  def : Pat<(i64 (int_riscv_buddy_msettilek i64:$rs1)),
            (AME_MSETTILEK GPR:$rs1)>;
}

//===----------------------------------------------------------------------===//
// Pseudo Instructions for Index-Based Operations
//===----------------------------------------------------------------------===//
// These pseudo instructions accept i64 indices and have AsmString for direct
// assembly output. This allows the pseudo instructions to be printed directly
// without needing complex expansion logic.
//
// For load/store/mma instructions, the AsmString hardcodes "acc" and "tr"
// prefixes so that indices 0-7 are printed as acc0-acc7 and tr0-tr7.
//===----------------------------------------------------------------------===//

// Pseudo instruction for msettilemi with i64 immediate
// Output: msettilemi x0, <imm>
let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 0, mayStore = 0,
    isCodeGenOnly = 1 in
def AME_MSETTILEMI_PSEUDO : Pseudo<(outs), (ins i64imm:$imm), []> {
  let AsmString = "msettilemi\tx0, $imm";
}

// Pseudo instruction for msettileni with i64 immediate
let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 0, mayStore = 0,
    isCodeGenOnly = 1 in
def AME_MSETTILENI_PSEUDO : Pseudo<(outs), (ins i64imm:$imm), []> {
  let AsmString = "msettileni\tx0, $imm";
}

// Pseudo instruction for msettileki with i64 immediate
let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 0, mayStore = 0,
    isCodeGenOnly = 1 in
def AME_MSETTILEKI_PSEUDO : Pseudo<(outs), (ins i64imm:$imm), []> {
  let AsmString = "msettileki\tx0, $imm";
}

// Pseudo instruction for mzero with accumulator index
let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 0, mayStore = 0,
    isCodeGenOnly = 1 in
def AME_MZERO_PSEUDO : Pseudo<(outs), (ins AMEAccIndex:$md), []> {
  let AsmString = "mzero.m\tacc$md";
}

// Pseudo instruction for mlae32.m with tile index
let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 1, mayStore = 0,
    isCodeGenOnly = 1 in
def AME_MLAE32_M_PSEUDO : Pseudo<(outs), (ins AMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
  let AsmString = "mlae32.m\ttr$md, ($rs1), $rs2";
}

// Pseudo instruction for mlbe32.m with tile index
let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 1, mayStore = 0,
    isCodeGenOnly = 1 in
def AME_MLBE32_M_PSEUDO : Pseudo<(outs), (ins AMETileIndex:$md, GPR:$rs1, GPR:$rs2), []> {
  let AsmString = "mlbe32.m\ttr$md, ($rs1), $rs2";
}

// Pseudo instruction for msce32.m with accumulator index
let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 0, mayStore = 1,
    isCodeGenOnly = 1 in
def AME_MSCE32_M_PSEUDO : Pseudo<(outs), (ins AMEAccIndex:$ms3, GPR:$rs1, GPR:$rs2), []> {
  let AsmString = "msce32.m\tacc$ms3, ($rs1), $rs2";
}

// Pseudo instruction for mma.w.mm.tile with indices
// md: AccReg index (0-7), ms1/ms2: TileReg indices (0-7)
let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 1, mayStore = 1,
    isCodeGenOnly = 1 in
def AME_MMA_W_MM_TILE_PSEUDO : Pseudo<(outs),
    (ins AMEAccIndex:$md, AMETileIndex:$ms1, AMETileIndex:$ms2), []> {
  let AsmString = "mma.w.mm\tacc$md, tr$ms1, tr$ms2";
}

//===----------------------------------------------------------------------===//
// Pattern Matching for Immediate Configuration Instructions
//===----------------------------------------------------------------------===//

// msettilemi - set tile M dimension with immediate
let Predicates = [HasBuddyExt] in {
  def : Pat<(int_riscv_buddy_msettilemi timm:$imm),
            (AME_MSETTILEMI_PSEUDO timm:$imm)>;
}

// msettileni - set tile N dimension with immediate
let Predicates = [HasBuddyExt] in {
  def : Pat<(int_riscv_buddy_msettileni timm:$imm),
            (AME_MSETTILENI_PSEUDO timm:$imm)>;
}

// msettileki - set tile K dimension with immediate
let Predicates = [HasBuddyExt] in {
  def : Pat<(int_riscv_buddy_msettileki timm:$imm),
            (AME_MSETTILEKI_PSEUDO timm:$imm)>;
}

//===----------------------------------------------------------------------===//
// Pattern Matching for Zero Instruction
//===----------------------------------------------------------------------===//

let Predicates = [HasBuddyExt] in {
  def : Pat<(int_riscv_buddy_mzero timm:$md),
            (AME_MZERO_PSEUDO timm:$md)>;
}

//===----------------------------------------------------------------------===//
// Pattern Matching for Load Instructions
//===----------------------------------------------------------------------===//

// mlae32.m - Load 32-bit left matrix A to tile register
let Predicates = [HasBuddyExt] in {
  def : Pat<(int_riscv_buddy_mlae32_m timm:$md, iPTR:$rs1, i64:$rs2),
            (AME_MLAE32_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
}

// mlbe32.m - Load 32-bit right matrix B to tile register
let Predicates = [HasBuddyExt] in {
  def : Pat<(int_riscv_buddy_mlbe32_m timm:$md, iPTR:$rs1, i64:$rs2),
            (AME_MLBE32_M_PSEUDO timm:$md, GPR:$rs1, GPR:$rs2)>;
}

//===----------------------------------------------------------------------===//
// Pattern Matching for Store Instructions
//===----------------------------------------------------------------------===//

// msce32.m - Store 32-bit output matrix C from accumulator
let Predicates = [HasBuddyExt] in {
  def : Pat<(int_riscv_buddy_msce32_m timm:$ms3, iPTR:$rs1, i64:$rs2),
            (AME_MSCE32_M_PSEUDO timm:$ms3, GPR:$rs1, GPR:$rs2)>;
}

//===----------------------------------------------------------------------===//
// Pattern Matching for Tile-Based Matrix Multiply
//===----------------------------------------------------------------------===//

// mma.w.mm.tile - int32 tile matrix multiply: md = md + ms1 x ms2
let Predicates = [HasBuddyExt] in {
  def : Pat<(int_riscv_buddy_mma_w_mm_tile timm:$md, timm:$ms1, timm:$ms2),
            (AME_MMA_W_MM_TILE_PSEUDO timm:$md, timm:$ms1, timm:$ms2)>;
}

//===----------------------------------------------------------------------===//
// AME Summary
//===----------------------------------------------------------------------===//
// Complete instruction set for basic matrix operations:
//
// Configuration:
//   - msettilem, msettilemi: Set M dimension
//   - msettilen, msettileni: Set N dimension
//   - msettilek, msettileki: Set K dimension
//
// Load (Memory → Register):
//   - mlae{8|16|32|64}.m: Load A into TileReg
//   - mlbe{8|16|32|64}.m: Load B into TileReg
//   - mlce{8|16|32|64}.m: Load C into AccReg
//
// Compute (TileReg × TileReg → AccReg):
//   No-widen:
//     - mma.{h|w|dw}.mm: Signed int16/32/64
//   Double-widen (AMUL ≥ 2):
//     - mwma.{h|w}.mm: int16→int32, int32→int64
//   Quad-widen (AMUL ≥ 4, mmi8i32 MANDATORY):
//     - mqma.b.mm: int8 → int32 (signed)
//     - mqmau.b.mm: uint8 → uint32 (unsigned)
//     - mqmasu.b.mm: int8 × uint8 → int32
//     - mqmaus.b.mm: uint8 × int8 → int32
//
// Store (Register → Memory):
//   - msae{8|16|32|64}.m: Store TileReg A
//   - msbe{8|16|32|64}.m: Store TileReg B
//   - msce{8|16|32|64}.m: Store AccReg C
//
// Utility:
//   - mzero.m: Zero out AccReg
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// XTAME (RISC-V Matrix Extension) 32-bit Instructions
//===----------------------------------------------------------------------===//
// Reference: RISC-V Matrix Extension Specification

def MatrixIndexAsmOperand : AsmOperandClass {
  let Name = "MatrixIndex";
  let RenderMethod = "addImmOperands";
  let PredicateMethod = "isUImm3";
  let DiagnosticType = "InvalidMatrixIndex";
}

// Operand type for MatrixReg index (0-7), printed with "m" prefix in AsmString
def MatrixIndex : RISCVOp<i64> {
  let ParserMatchClass = MatrixIndexAsmOperand;
  let DecoderMethod = "decodeUImmOperand<3>";
  let OperandType = "OPERAND_UIMM3";
}

//===----------------------------------------------------------------------===//
// XTAME 32-bit Instruction Format Base Class
//===----------------------------------------------------------------------===//

// Base class for XTAME 32-bit matrix multiplication instructions
// Uses MatrixReg for inputs (ms1, ms2) and MatrixReg for output (md)
class RVInstXTAME32<dag outs, dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<4> func4;
  bits<3> uop = 0b000;        // usually 0
  bit     size;
  bits<3> ms2;
  bits<3> ms1;
  bits<3> md;
  bits<3> func3 = 0b000;      // usually 0
  bits<2> size_w;
  bits<3> type;
  bits<7> opcode = 0b0101011;

  let Inst{31-28} = func4;
  let Inst{27-25} = uop;
  let Inst{24} = size;
  let Inst{23-21} = ms2;
  let Inst{20-18} = ms1;
  let Inst{17-15} = md;
  let Inst{14-12} = func3;
  let Inst{11-10} = size_w;
  let Inst{9-7} = type;
  let Inst{6-0} = opcode;
}

//===----------------------------------------------------------------------===//
// AME Matrix Multiplication Instructions
//===----------------------------------------------------------------------===//
class TH_MMACC<bits<3> type_val, string opcodestr>
    : RVInstXTAME32<(outs MatrixReg:$md),
                  (ins MatrixReg:$md_in, MatrixReg:$ms2, MatrixReg:$ms1),
                  opcodestr, "$md, $ms2, $ms1"> {
  let Constraints = "$md = $md_in";
  let func4   = 0b0010;
  let uop     = 0b000;
  let size    = 0;
  let func3   = 0b000;
  let size_w    = 0b00;
  let type    = type_val;
  let opcode  = 0b0101011;
}

// th.mmacc.w.b - int8 × int8 → int32 accumulate
let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def TH_MMACC_W_B : TH_MMACC<0b000, "th.mmacc.w.b">;

//===----------------------------------------------------------------------===//
// AME 32-bit Configuration Instructions (Register Type)
// Reference: RISC-V Matrix Specification v0.5.0, Section 4.1, Table 7
// Format:
// | 31 | 30:28 | 27:25 | 24:20 | 19:15 | 14:12 | 11:9 | 8 | 7 | 6:0      |
// | 1  | func  | 111   | 00000 | rs1   | 000   | 000  | 0 | 0 | 0101011  |
//===----------------------------------------------------------------------===//

// Base class for 32-bit register-type configuration instructions
class RVInstXTAMEConfig32Reg<dag outs, dag ins, string opcodestr, string argstr,
                           bits<3> func_val>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<5> rs1;

  let Inst{31}   = 1;               // register type
  let Inst{30-28}= func_val;
  let Inst{27-25}= 0b111;           // uop fixed to 111
  let Inst{24-20}= 0b00000;         // reserved
  let Inst{19-15}= rs1;
  let Inst{14-12}= 0b000;           // func3 fixed to 000
  let Inst{11-9} = 0b000;           // nop field
  let Inst{8}    = 0;               // mxb = 0
  let Inst{7}    = 0;               // mxa = 0
  let Inst{6-0}  = 0b0101011;       // custom-1 opcode
}

//===----------------------------------------------------------------------===//
// AME 32-bit Configuration Instructions (Immediate Type)
// Reference: RISC-V Matrix Specification v0.5.0, Section 4.1, Table 7
// Format:
// | 31 | 30:28 | 27:25 | 24:18 | 17:15 | 14:12 | 11:9 | 8 | 7 | 6:0      |
// | 0  | func  | 111   | uimm7 | 000   | 000   | 000  | 0 | 0 | 0101011  |
//===----------------------------------------------------------------------===//

// Base class for 32-bit immediate-type configuration instructions
class RVInstXTAMEConfig32Imm<dag outs, dag ins, string opcodestr, string argstr,
                           bits<3> func_val>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<7> imm;

  let Inst{31}   = 0;               // immediate type
  let Inst{30-28}= func_val;
  let Inst{27-25}= 0b111;           // uop fixed to 111
  let Inst{24-18}= imm{6-0};        // imm[6:0]
  let Inst{17-15}= 0b000;           // reserved
  let Inst{14-12}= 0b000;           // func3 fixed to 000
  let Inst{11-9} = 0b000;           // nop field
  let Inst{8}    = 0;               // mxb = 0
  let Inst{7}    = 0;               // mxa = 0
  let Inst{6-0}  = 0b0101011;       // custom-1 opcode
}

//===----------------------------------------------------------------------===//
// Concrete Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  // th.mcfgk - configure tile K dimension (func=000)
  def TH_MCFGK : RVInstXTAMEConfig32Reg<(outs), (ins GPR:$rs1),
                                       "th.mcfgk", "$rs1", 0b000>;

  // th.mcfgm - configure tile M dimension (func=001)
  def TH_MCFGM : RVInstXTAMEConfig32Reg<(outs), (ins GPR:$rs1),
                                       "th.mcfgm", "$rs1", 0b001>;

  // th.mcfgn - configure tile N dimension (func=010)
  def TH_MCFGN : RVInstXTAMEConfig32Reg<(outs), (ins GPR:$rs1),
                                       "th.mcfgn", "$rs1", 0b010>;

  // th.mcfg - configure entire tile size (func=111)
  def TH_MCFG  : RVInstXTAMEConfig32Reg<(outs), (ins GPR:$rs1),
                                       "th.mcfg", "$rs1", 0b111>;
}

let Predicates = [HasBuddyExt], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  // th.mcfgki - configure tile K dimension from immediate (func=000)
  def TH_MCFGKI : RVInstXTAMEConfig32Imm<(outs), (ins uimm32:$imm),
                                        "th.mcfgki", "$imm", 0b000>;

  // th.mcfgmi - configure tile M dimension from immediate (func=001)
  def TH_MCFGMI : RVInstXTAMEConfig32Imm<(outs), (ins uimm32:$imm),
                                        "th.mcfgmi", "$imm", 0b001>;

  // th.mcfgni - configure tile N dimension from immediate (func=010)
  def TH_MCFGNI : RVInstXTAMEConfig32Imm<(outs), (ins uimm32:$imm),
                                        "th.mcfgni", "$imm", 0b010>;
}

//===----------------------------------------------------------------------===//
// XTAME 32-bit Load/Store Instructions
// Reference: RISC-V Matrix Specification v0.5.0, Section 4.4, Table 10
// Format:
// | 31:28 | 27:25 | 24:20 | 19:15 | 14:12 | 11:10 | 9:7   | 6:0          |
// | func4 | uop   | rs2   | rs1   | func3 | size  | md/ms3| opcode       |
// func3 = 000, opcode = 0b0101011 (custom-1)
//===----------------------------------------------------------------------===//

// Base class for all 32-bit load instructions (uop = 100)
class RVInstXTAMELoad32<bits<4> func4_val, string opcodestr>
    : RVInst<(outs MatrixReg:$md), (ins GPR:$rs2, GPR:$rs1),
               opcodestr, "$md, $rs2, $rs1", [], InstFormatOther> {
  bits<5> rs2;
  bits<5> rs1;
  bits<3> md;
  bits<2> size;

  let Inst{31-28} = func4_val;
  let Inst{27-25} = 0b100;          // uop = 100 for load
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b000;          // func3 = 000
  let Inst{11-10} = size;           // size to be set in derived classes
  let Inst{9-7}   = md;
  let Inst{6-0}   = 0b0101011;      // custom-1 opcode
}

// Base class for all 32-bit store instructions (uop = 101)
class RVInstXTAMEStore32<bits<4> func4_val, string opcodestr>
    : RVInst<(outs), (ins MatrixReg:$ms3, GPR:$rs1, GPR:$rs2),
               opcodestr, "$ms3, $rs2, $rs1", [], InstFormatOther> {
  bits<5> rs2;
  bits<5> rs1;
  bits<3> ms3;
  bits<2> size;

  let Inst{31-28} = func4_val;
  let Inst{27-25} = 0b101;          // uop = 101 for store
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b000;          // func3 = 000
  let Inst{11-10} = size;           // size to be set in derived classes
  let Inst{9-7}   = ms3;
  let Inst{6-0}   = 0b0101011;      // custom-1 opcode
}

//===----------------------------------------------------------------------===//
// Normal Load Instructions (func4 = 0000)
// th.mld.e8, th.mld.e16, th.mld.e32, th.mld.e64
//===----------------------------------------------------------------------===//
let Predicates = [HasBuddyExt], mayLoad = 1, mayStore = 0, hasSideEffects = 0 in {
  def TH_MLDE8  : RVInstXTAMELoad32<0b0000, "th.mlde8">  { let size = 0b00; }
  def TH_MLDE16 : RVInstXTAMELoad32<0b0000, "th.mlde16"> { let size = 0b01; }
  def TH_MLDE32 : RVInstXTAMELoad32<0b0000, "th.mlde32"> { let size = 0b10; }
  def TH_MLDE64 : RVInstXTAMELoad32<0b0000, "th.mlde64"> { let size = 0b11; }
}

//===----------------------------------------------------------------------===//
// Normal Store Instructions (func4 = 0000)
// th.mst.e8, th.mst.e16, th.mst.e32, th.mst.e64
//===----------------------------------------------------------------------===//
let Predicates = [HasBuddyExt], mayLoad = 0, mayStore = 1, hasSideEffects = 0 in {
  def TH_MSTE8  : RVInstXTAMEStore32<0b0000, "th.mste8">  { let size = 0b00; }
  def TH_MSTE16 : RVInstXTAMEStore32<0b0000, "th.mste16"> { let size = 0b01; }
  def TH_MSTE32 : RVInstXTAMEStore32<0b0000, "th.mste32"> { let size = 0b10; }
  def TH_MSTE64 : RVInstXTAMEStore32<0b0000, "th.mste64"> { let size = 0b11; }
}

//===----------------------------------------------------------------------===//
// Transposed Load Instructions (func4 = 0011)
// th.mldt.e8, th.mldt.e16, th.mldt.e32, th.mldt.e64
//===----------------------------------------------------------------------===//
let Predicates = [HasBuddyExt], mayLoad = 1, mayStore = 0, hasSideEffects = 0 in {
  def TH_MLDTE8  : RVInstXTAMELoad32<0b0011, "th.mldte8">  { let size = 0b00; }
  def TH_MLDTE16 : RVInstXTAMELoad32<0b0011, "th.mldte16"> { let size = 0b01; }
  def TH_MLDTE32 : RVInstXTAMELoad32<0b0011, "th.mldte32"> { let size = 0b10; }
  def TH_MLDTE64 : RVInstXTAMELoad32<0b0011, "th.mldte64"> { let size = 0b11; }
}

//===----------------------------------------------------------------------===//
// MISC Instructions (format "others") - Base class
//===----------------------------------------------------------------------===//

class RVInstXTAMEMISC<dag outs, dag ins, string opcodestr, string argstr,
                    bits<4> func4_val, bits<3> uop_val>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<4> func4 = func4_val;
  bits<3> uop   = uop_val;
  bit     ctrl  = 0;              // bit 24 must be 0 for mzero
  bits<3> ms2   = 0b000;          // fixed to 000
  bits<3> ms1   = 0b000;          // fixed to 000
  bits<3> md;                     // destination base matrix register
  bits<3> func3 = 0b000;          // fixed to 000
  bits<2> size  = 0b00;           // size field (00 for mzero)
  bits<3> uimm3;                  // immediate for register count encoding
  bits<7> opcode = 0b0101011;     // custom-1

  let Inst{31-28} = func4;
  let Inst{27-25} = uop;
  let Inst{24}    = ctrl;
  let Inst{23-21} = ms2;
  let Inst{20-18} = ms1;
  let Inst{17-15} = md;
  let Inst{14-12} = func3;
  let Inst{11-10} = size;
  let Inst{9-7}   = uimm3;
  let Inst{6-0}   = opcode;
}

//===----------------------------------------------------------------------===//
// th.mzero variants (different uimm3 encodings)
//===----------------------------------------------------------------------===//

let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 0, mayStore = 0 in {
  // Zero 1 matrix register (uimm3 = 000)
  def TH_MZERO   : RVInstXTAMEMISC<(outs), (ins MatrixReg:$md),
                                 "th.mzero", "$md",
                                 0b1010, 0b000> {
    let uimm3 = 0b000;
  }

  // Zero 2 matrix registers (uimm3 = 001)
  def TH_MZERO2R : RVInstXTAMEMISC<(outs), (ins MatrixReg:$md),
                                 "th.mzero2r", "$md",
                                 0b1010, 0b000> {
    let uimm3 = 0b001;
  }

  // Zero 4 matrix registers (uimm3 = 011)
  def TH_MZERO4R : RVInstXTAMEMISC<(outs), (ins MatrixReg:$md),
                                 "th.mzero4r", "$md",
                                 0b1010, 0b000> {
    let uimm3 = 0b011;
  }

  // Zero 8 matrix registers (uimm3 = 111)
  def TH_MZERO8R : RVInstXTAMEMISC<(outs), (ins MatrixReg:$md),
                                 "th.mzero8r", "$md",
                                 0b1010, 0b000> {
    let uimm3 = 0b111;
  }
}

//===----------------------------------------------------------------------===//
// Pseudo Instructions for 32-bit New Instructions
//===----------------------------------------------------------------------===//

// Configuration immediate instructions (no matrix register)
let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 0, mayStore = 0,
    isCodeGenOnly = 1 in {
  def TH_MCFGMI_PSEUDO : Pseudo<(outs), (ins i64imm:$imm), []> {
    let AsmString = "th.mcfgmi\t$imm";
  }
  def TH_MCFGNI_PSEUDO : Pseudo<(outs), (ins i64imm:$imm), []> {
    let AsmString = "th.mcfgni\t$imm";
  }
  def TH_MCFGKI_PSEUDO : Pseudo<(outs), (ins i64imm:$imm), []> {
    let AsmString = "th.mcfgki\t$imm";
  }
}

// Zero instruction (single matrix register)
let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 0, mayStore = 0,
    isCodeGenOnly = 1 in {
  def TH_MZERO_PSEUDO : Pseudo<(outs), (ins MatrixIndex:$md), []> {
    let AsmString = "th.mzero\tm$md";
  }
}

// Load instructions (normal and transposed)
let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 1, mayStore = 0,
    isCodeGenOnly = 1 in {
  def TH_MLDE8_PSEUDO : Pseudo<(outs), (ins MatrixIndex:$md, GPR:$rs2, GPR:$rs1), []> {
    let AsmString = "th.mlde8\tm$md, $rs2, ($rs1)";
  }
  def TH_MLDTE8_PSEUDO : Pseudo<(outs), (ins MatrixIndex:$md, GPR:$rs2, GPR:$rs1), []> {
    let AsmString = "th.mldte8\tm$md, $rs2, ($rs1)";
  }
}

// Store instruction
let Predicates = [HasBuddyExt], hasSideEffects = 1, mayLoad = 0, mayStore = 1,
    isCodeGenOnly = 1 in {
  def TH_MSTE32_PSEUDO : Pseudo<(outs), (ins MatrixIndex:$ms3, GPR:$rs2, GPR:$rs1), []> {
    let AsmString = "th.mste32\tm$ms3, $rs2, ($rs1)";
  }
}

// Matrix multiply instruction
let Predicates = [HasBuddyExt], hasSideEffects = 1,
    isCodeGenOnly = 1 in {
  def TH_MMACC_W_B_PSEUDO : Pseudo<(outs),
      (ins MatrixIndex:$md, MatrixIndex:$ms2, MatrixIndex:$ms1), []> {
    let AsmString = "th.mmacc.w.b\tm$md, m$ms2, m$ms1";
  }
}

//===----------------------------------------------------------------------===//
// Pattern Matching for 32-bit New Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasBuddyExt] in {

  // Configuration immediate instructions
  def : Pat<(int_riscv_buddy_th_mcfgmi timm:$imm),
            (TH_MCFGMI_PSEUDO timm:$imm)>;
  def : Pat<(int_riscv_buddy_th_mcfgni timm:$imm),
            (TH_MCFGNI_PSEUDO timm:$imm)>;
  def : Pat<(int_riscv_buddy_th_mcfgki timm:$imm),
            (TH_MCFGKI_PSEUDO timm:$imm)>;

  // Zero instruction
  def : Pat<(int_riscv_buddy_th_mzero timm:$md),
            (TH_MZERO_PSEUDO timm:$md)>;

  // Load instructions
  def : Pat<(int_riscv_buddy_th_mlde8 timm:$md, i64:$rs2, iPTR:$rs1),
            (TH_MLDE8_PSEUDO timm:$md, GPR:$rs2, GPR:$rs1)>;
  def : Pat<(int_riscv_buddy_th_mldte8 timm:$md, i64:$rs2, iPTR:$rs1),
            (TH_MLDTE8_PSEUDO timm:$md, GPR:$rs2, GPR:$rs1)>;

  // Store instruction
  def : Pat<(int_riscv_buddy_th_mste32 timm:$ms3, i64:$rs2, iPTR:$rs1),
            (TH_MSTE32_PSEUDO timm:$ms3, GPR:$rs2, GPR:$rs1)>;

  // Matrix multiply instruction
  def : Pat<(int_riscv_buddy_th_mmacc_w_b timm:$md, timm:$ms2, timm:$ms1),
            (TH_MMACC_W_B_PSEUDO timm:$md, timm:$ms2, timm:$ms1)>;
}
