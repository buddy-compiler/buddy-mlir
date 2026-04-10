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
def int_riscv_config : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

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

//===----------------------------------------------------------------------===//
// AME (RISC-V Matrix Extension) Intrinsics
//===----------------------------------------------------------------------===//

// Matrix configuration intrinsics (with register)
let TargetPrefix = "riscv" in {
  // msettype - Set matrix type configuration
  def int_riscv_buddy_msettype : Intrinsic<[llvm_i64_ty], [llvm_i64_ty], [IntrNoMem]>;

  // msettilem - Set tile M dimension from register
  def int_riscv_buddy_msettilem : Intrinsic<[llvm_i64_ty], [llvm_i64_ty], [IntrNoMem]>;

  // msettilen - Set tile N dimension from register
  def int_riscv_buddy_msettilen : Intrinsic<[llvm_i64_ty], [llvm_i64_ty], [IntrNoMem]>;

  // msettilek - Set tile K dimension from register
  def int_riscv_buddy_msettilek : Intrinsic<[llvm_i64_ty], [llvm_i64_ty], [IntrNoMem]>;
}

// Matrix configuration intrinsics (with immediate)
let TargetPrefix = "riscv" in {
  // msettilemi - Set tile M dimension with immediate
  def int_riscv_buddy_msettilemi : Intrinsic<[], [llvm_i64_ty],
                                             [IntrNoMem, IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;

  // msettileni - Set tile N dimension with immediate
  def int_riscv_buddy_msettileni : Intrinsic<[], [llvm_i64_ty],
                                             [IntrNoMem, IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;

  // msettileki - Set tile K dimension with immediate
  def int_riscv_buddy_msettileki : Intrinsic<[], [llvm_i64_ty],
                                             [IntrNoMem, IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;
}

// Matrix load intrinsics (load to tile register)
// Format: md = tile register index, base = address, stride = row byte stride
let TargetPrefix = "riscv" in {
  // mlae32.m - Load 32-bit left matrix A to tile register
  def int_riscv_buddy_mlae32_m : Intrinsic<[],
                                           [llvm_i64_ty, llvm_ptr_ty, llvm_i64_ty],
                                           [IntrReadMem, IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;

  // mlae64.m - Load 64-bit left matrix A to tile register
  def int_riscv_buddy_mlae64_m : Intrinsic<[],
                                           [llvm_i64_ty, llvm_ptr_ty, llvm_i64_ty],
                                           [IntrReadMem, IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;

  // mlbe32.m - Load 32-bit right matrix B to tile register
  def int_riscv_buddy_mlbe32_m : Intrinsic<[],
                                           [llvm_i64_ty, llvm_ptr_ty, llvm_i64_ty],
                                           [IntrReadMem, IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;

  // mlbe64.m - Load 64-bit right matrix B to tile register
  def int_riscv_buddy_mlbe64_m : Intrinsic<[],
                                           [llvm_i64_ty, llvm_ptr_ty, llvm_i64_ty],
                                           [IntrReadMem, IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;

  // mlce32.m - Load 32-bit output matrix C to accumulator
  def int_riscv_buddy_mlce32_m : Intrinsic<[],
                                           [llvm_i64_ty, llvm_ptr_ty, llvm_i64_ty],
                                           [IntrReadMem, IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;

  // mlce64.m - Load 64-bit output matrix C to accumulator
  def int_riscv_buddy_mlce64_m : Intrinsic<[],
                                           [llvm_i64_ty, llvm_ptr_ty, llvm_i64_ty],
                                           [IntrReadMem, IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;
}

// Matrix store intrinsics (store from tile register)
let TargetPrefix = "riscv" in {
  // msce32.m - Store 32-bit output matrix C from accumulator
  def int_riscv_buddy_msce32_m : Intrinsic<[],
                                           [llvm_i64_ty, llvm_ptr_ty, llvm_i64_ty],
                                           [IntrWriteMem, IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;

  // msce64.m - Store 64-bit output matrix C from accumulator
  def int_riscv_buddy_msce64_m : Intrinsic<[],
                                           [llvm_i64_ty, llvm_ptr_ty, llvm_i64_ty],
                                           [IntrWriteMem, IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;
}

// Matrix zero intrinsic
let TargetPrefix = "riscv" in {
  def int_riscv_buddy_mzero : Intrinsic<[], [llvm_i64_ty],
                                        [IntrNoMem, IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;
}

// Tile register matrix multiplication intrinsics (operate on tile registers)
let TargetPrefix = "riscv" in {
  // mma.w.mm - int32 tile matrix multiply: md = md + ms1 x ms2
  def int_riscv_buddy_mma_w_mm_tile : Intrinsic<[],
                                                [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
                                                [IntrHasSideEffects, ImmArg<ArgIndex<0>>,
                                                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

  // mma.dw.mm - int64 tile matrix multiply: md = md + ms1 x ms2
  def int_riscv_buddy_mma_dw_mm_tile : Intrinsic<[],
                                                 [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
                                                 [IntrHasSideEffects, ImmArg<ArgIndex<0>>,
                                                  ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;
}

// Legacy matrix load/store intrinsics (for backward compatibility)
let TargetPrefix = "riscv" in {
  // mlae - Load matrix A with element width
  def int_riscv_buddy_mlae : Intrinsic<[llvm_anyvector_ty],
                                       [llvm_ptr_ty, llvm_i64_ty, llvm_i64_ty],
                                       [IntrReadMem]>;

  // mlbe - Load matrix B with element width
  def int_riscv_buddy_mlbe : Intrinsic<[llvm_anyvector_ty],
                                       [llvm_ptr_ty, llvm_i64_ty, llvm_i64_ty],
                                       [IntrReadMem]>;

  // mlce - Load matrix C (accumulator)
  def int_riscv_buddy_mlce : Intrinsic<[llvm_anyvector_ty],
                                       [llvm_ptr_ty, llvm_i64_ty, llvm_i64_ty],
                                       [IntrReadMem]>;

  // msce - Store matrix C (accumulator)
  def int_riscv_buddy_msce : Intrinsic<[],
                                       [llvm_anyvector_ty, llvm_ptr_ty, llvm_i64_ty, llvm_i64_ty],
                                       [IntrWriteMem]>;
}

// Signed integer matrix multiplication intrinsics
let TargetPrefix = "riscv" in {
  // mqma.b.mm - int8 quad-widen matrix multiply (int8 x int8 -> int32)
  def int_riscv_buddy_mqma_b_mm : Intrinsic<[],
                                            [llvm_ptr_ty, llvm_ptr_ty, llvm_ptr_ty,
                                             llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
                                            []>;

  // mma.h.mm - int16 matrix multiply
  def int_riscv_buddy_mma_h_mm : Intrinsic<[],
                                           [llvm_ptr_ty, llvm_ptr_ty, llvm_ptr_ty,
                                            llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
                                           []>;

  // mma.w.mm - int32 matrix multiply
  def int_riscv_buddy_mma_w_mm : Intrinsic<[],
                                           [llvm_ptr_ty, llvm_ptr_ty, llvm_ptr_ty,
                                            llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
                                           []>;

  // mma.dw.mm - int64 matrix multiply
  def int_riscv_buddy_mma_dw_mm : Intrinsic<[],
                                            [llvm_ptr_ty, llvm_ptr_ty, llvm_ptr_ty,
                                             llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
                                            []>;

  // mwma.h.mm - int16 double-widen matrix multiply (int16 x int16 -> int32)
  def int_riscv_buddy_mwma_h_mm : Intrinsic<[],
                                            [llvm_ptr_ty, llvm_ptr_ty, llvm_ptr_ty,
                                             llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
                                            []>;
}

// Unsigned integer matrix multiplication intrinsics
let TargetPrefix = "riscv" in {
  // mqmau.b.mm - uint8 quad-widen matrix multiply
  def int_riscv_buddy_mqmau_b_mm : Intrinsic<[],
                                             [llvm_ptr_ty, llvm_ptr_ty, llvm_ptr_ty,
                                              llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
                                             []>;

  // mmau.h.mm - uint16 matrix multiply
  def int_riscv_buddy_mmau_h_mm : Intrinsic<[],
                                            [llvm_ptr_ty, llvm_ptr_ty, llvm_ptr_ty,
                                             llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
                                            []>;
}

// Floating-point matrix multiplication intrinsics
let TargetPrefix = "riscv" in {
  // mfma.f.mm - fp32 matrix multiply
  def int_riscv_buddy_mfma_f_mm : Intrinsic<[],
                                            [llvm_ptr_ty, llvm_ptr_ty, llvm_ptr_ty,
                                             llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
                                            []>;

  // mfma.hf.mm - fp16 matrix multiply
  def int_riscv_buddy_mfma_hf_mm : Intrinsic<[],
                                             [llvm_ptr_ty, llvm_ptr_ty, llvm_ptr_ty,
                                              llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
                                             []>;

  // mfwma.hf.mm - fp16 double-widen matrix multiply (fp16 x fp16 -> fp32)
  def int_riscv_buddy_mfwma_hf_mm : Intrinsic<[],
                                              [llvm_ptr_ty, llvm_ptr_ty, llvm_ptr_ty,
                                               llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
                                              []>;
}
