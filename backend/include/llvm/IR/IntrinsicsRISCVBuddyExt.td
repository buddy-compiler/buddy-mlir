//===- IntrinsicsRISCVBuddyExt.td -----------------------------------------===//
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
// This is the costum intrinsic definition file of RISC-V buddy extension.
//
//===----------------------------------------------------------------------===//
let TargetPrefix = "riscv" in
def int_riscv_mvin : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty],[]>;

let TargetPrefix = "riscv" in
def int_riscv_mvin2 : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty],[]>;

let TargetPrefix = "riscv" in
def int_riscv_mvin3 : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty],[]>;

let TargetPrefix = "riscv" in
def int_riscv_mvout : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in
def int_riscv_flush : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in
def int_riscv_config_ld : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in
def int_riscv_config_st : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in
def int_riscv_config_ex : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in
def int_riscv_config_norm : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in
def int_riscv_preload : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in
def int_riscv_compute_preloaded : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in
def int_riscv_compute_accumulated : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in
def int_riscv_loop_ws_config_bounds : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in
def int_riscv_loop_ws_config_addrs_ab : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in
def int_riscv_loop_ws_config_addrs_dc : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in
def int_riscv_loop_ws_config_strides_ab : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in
def int_riscv_loop_ws_config_strides_dc : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in
def int_riscv_loop_ws : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in
def int_riscv_loop_conv_ws : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in
def int_riscv_loop_conv_ws_config1 : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in
def int_riscv_loop_conv_ws_config2 : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in
def int_riscv_loop_conv_ws_config3 : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in
def int_riscv_loop_conv_ws_config4 : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in
def int_riscv_loop_conv_ws_config5 : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in
def int_riscv_loop_conv_ws_config6 : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

//===----------------------------------------------------------------------===//
// IME Extension Intrinsics
//===----------------------------------------------------------------------===//

let TargetPrefix = "riscv" in
def int_riscv_ime_vmadot : Intrinsic<[llvm_anyvector_ty],
                                     [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty],
                                     [IntrNoMem]>;

let TargetPrefix = "riscv" in
def int_riscv_ime_vmadotu : Intrinsic<[llvm_anyvector_ty],
                                      [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty],
                                      [IntrNoMem]>;

let TargetPrefix = "riscv" in
def int_riscv_ime_vmadotsu : Intrinsic<[llvm_anyvector_ty],
                                       [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty],
                                       [IntrNoMem]>;

let TargetPrefix = "riscv" in
def int_riscv_ime_vmadotus : Intrinsic<[llvm_anyvector_ty],
                                       [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty],
                                       [IntrNoMem]>;

let TargetPrefix = "riscv" in
def int_riscv_ime_vfmadot : Intrinsic<[llvm_anyvector_ty],
                                      [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty],
                                      [IntrNoMem]>;

let TargetPrefix = "riscv" in
def int_riscv_ime_vmadot1 : Intrinsic<[llvm_anyvector_ty],
                                      [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty],
                                      [IntrNoMem]>;

let TargetPrefix = "riscv" in
def int_riscv_ime_vmadot1u : Intrinsic<[llvm_anyvector_ty],
                                       [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty],
                                       [IntrNoMem]>;

let TargetPrefix = "riscv" in
def int_riscv_ime_vmadot1su : Intrinsic<[llvm_anyvector_ty],
                                        [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty],
                                        [IntrNoMem]>;

let TargetPrefix = "riscv" in
def int_riscv_ime_vmadot1us : Intrinsic<[llvm_anyvector_ty],
                                        [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty],
                                        [IntrNoMem]>;

let TargetPrefix = "riscv" in
def int_riscv_ime_vmadot2 : Intrinsic<[llvm_anyvector_ty],
                                      [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty],
                                      [IntrNoMem]>;

let TargetPrefix = "riscv" in
def int_riscv_ime_vmadot2u : Intrinsic<[llvm_anyvector_ty],
                                       [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty],
                                       [IntrNoMem]>;

let TargetPrefix = "riscv" in
def int_riscv_ime_vmadot2su : Intrinsic<[llvm_anyvector_ty],
                                        [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty],
                                        [IntrNoMem]>;

let TargetPrefix = "riscv" in
def int_riscv_ime_vmadot2us : Intrinsic<[llvm_anyvector_ty],
                                        [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty],
                                        [IntrNoMem]>;

let TargetPrefix = "riscv" in
def int_riscv_ime_vmadot3 : Intrinsic<[llvm_anyvector_ty],
                                      [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty],
                                      [IntrNoMem]>;

let TargetPrefix = "riscv" in
def int_riscv_ime_vmadot3u : Intrinsic<[llvm_anyvector_ty],
                                       [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty],
                                       [IntrNoMem]>;

let TargetPrefix = "riscv" in
def int_riscv_ime_vmadot3su : Intrinsic<[llvm_anyvector_ty],
                                        [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty],
                                        [IntrNoMem]>;

let TargetPrefix = "riscv" in
def int_riscv_ime_vmadot3us : Intrinsic<[llvm_anyvector_ty],
                                        [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty],
                                        [IntrNoMem]>;

let TargetPrefix = "riscv" in
def int_riscv_ime_vmadotn : Intrinsic<[llvm_anyvector_ty],
                                      [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty, llvm_i64_ty],
                                      [IntrNoMem]>;

let TargetPrefix = "riscv" in
def int_riscv_ime_vmadotnu : Intrinsic<[llvm_anyvector_ty],
                                       [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty, llvm_i64_ty],
                                       [IntrNoMem]>;

let TargetPrefix = "riscv" in
def int_riscv_ime_vmadotnsu : Intrinsic<[llvm_anyvector_ty],
                                        [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty, llvm_i64_ty],
                                        [IntrNoMem]>;

let TargetPrefix = "riscv" in
def int_riscv_ime_vmadotnus : Intrinsic<[llvm_anyvector_ty],
                                        [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty, llvm_i64_ty],
                                        [IntrNoMem]>;

let TargetPrefix = "riscv" in
def int_riscv_ime_vfmadot1 : Intrinsic<[llvm_anyvector_ty],
                                       [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty],
                                       [IntrNoMem]>;

let TargetPrefix = "riscv" in
def int_riscv_ime_vfmadot2 : Intrinsic<[llvm_anyvector_ty],
                                       [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty],
                                       [IntrNoMem]>;

let TargetPrefix = "riscv" in
def int_riscv_ime_vfmadot3 : Intrinsic<[llvm_anyvector_ty],
                                       [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty],
                                       [IntrNoMem]>;

let TargetPrefix = "riscv" in
def int_riscv_ime_vfmadotn : Intrinsic<[llvm_anyvector_ty],
                                       [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyvector_ty, llvm_i64_ty],
                                       [IntrNoMem]>;
