//===- IntrinsicsRISCVBuddyBOSCExt.td -------------------------------------===//
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
// This file defines the intrinsics for the BOSC AME extension.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AME (RISC-V Matrix Extension) Intrinsics
//===----------------------------------------------------------------------===//

// Matrix configuration intrinsics (with immediate)
let TargetPrefix = "riscv" in {
  // msettilemi - Set tile M dimension with immediate
  def int_riscv_buddy_bosc_msettilemi : Intrinsic<[llvm_i64_ty], [llvm_i64_ty],
                                             [IntrNoMem, IntrHasSideEffects,
                                              ImmArg<ArgIndex<0>>]>;

  // msettileni - Set tile N dimension with immediate
  def int_riscv_buddy_bosc_msettileni : Intrinsic<[llvm_i64_ty], [llvm_i64_ty],
                                             [IntrNoMem, IntrHasSideEffects,
                                              ImmArg<ArgIndex<0>>]>;

  // msettileki - Set tile K dimension with immediate
  def int_riscv_buddy_bosc_msettileki : Intrinsic<[llvm_i64_ty], [llvm_i64_ty],
                                             [IntrNoMem, IntrHasSideEffects,
                                             ImmArg<ArgIndex<0>>]>;
}

// Matrix load intrinsics (load to tile register)
// Format: md = tile register index, base = address, stride = row byte stride
let TargetPrefix = "riscv" in {
  // mlae32.m - Load 32-bit left matrix A to tile register
  def int_riscv_buddy_bosc_mlae32_m : Intrinsic<[],
                                           [llvm_i64_ty, llvm_ptr_ty, llvm_i64_ty],
                                           [IntrReadMem, IntrHasSideEffects,
                                            ImmArg<ArgIndex<0>>]>;

  // mlbe32.m - Load 32-bit right matrix B to tile register
  def int_riscv_buddy_bosc_mlbe32_m : Intrinsic<[],
                                           [llvm_i64_ty, llvm_ptr_ty, llvm_i64_ty],
                                           [IntrReadMem, IntrHasSideEffects,
                                            ImmArg<ArgIndex<0>>]>;
}

// Matrix store intrinsics (store from tile register)
let TargetPrefix = "riscv" in {
  // msce32.m - Store 32-bit output matrix C from accumulator
  def int_riscv_buddy_bosc_msce32_m : Intrinsic<[],
                                           [llvm_i64_ty, llvm_ptr_ty, llvm_i64_ty],
                                           [IntrWriteMem, IntrHasSideEffects,
                                            ImmArg<ArgIndex<0>>]>;
}

// Matrix zero intrinsic
let TargetPrefix = "riscv" in {
  def int_riscv_buddy_bosc_mzero : Intrinsic<[], [llvm_i64_ty],
                                        [IntrNoMem, IntrHasSideEffects,
                                         ImmArg<ArgIndex<0>>]>;
}

// Tile register matrix multiplication intrinsics (operate on tile registers)
let TargetPrefix = "riscv" in {
  // mma.w.mm - int32 tile matrix multiply: md = md + ms1 x ms2
  def int_riscv_buddy_bosc_mma_w_mm : Intrinsic<[],[llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
                                           [IntrHasSideEffects, ImmArg<ArgIndex<0>>,
                                            ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;
}
