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
  // Configuration Register Instructions
  // msettype - Set the mtype CSR based on register value
  def int_riscv_buddy_bosc_msettype : Intrinsic<[llvm_i64_ty], [llvm_i64_ty],
                                                [IntrNoMem, IntrHasSideEffects]>;

  // msettilem - Set tile M dimension from register value
  def int_riscv_buddy_bosc_msettilem : Intrinsic<[llvm_i64_ty], [llvm_i64_ty],
                                                 [IntrNoMem, IntrHasSideEffects]>;

  // msettilen - Set tile N dimension from register value
  def int_riscv_buddy_bosc_msettilen : Intrinsic<[llvm_i64_ty], [llvm_i64_ty],
                                                 [IntrNoMem, IntrHasSideEffects]>;

  // msettilek - Set tile K dimension from register value
  def int_riscv_buddy_bosc_msettilek : Intrinsic<[llvm_i64_ty], [llvm_i64_ty],
                                                 [IntrNoMem, IntrHasSideEffects]>;

  // msettypei - Set the mtype CSR based on immediate
  def int_riscv_buddy_bosc_msettypei : Intrinsic<[llvm_i64_ty], [llvm_i64_ty],
                                             [IntrNoMem, IntrHasSideEffects,
                                              ImmArg<ArgIndex<0>>]>;

  def int_riscv_buddy_bosc_msettypehi : Intrinsic<[llvm_i64_ty], [llvm_i64_ty],
                                             [IntrNoMem, IntrHasSideEffects,
                                              ImmArg<ArgIndex<0>>]>;

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

//===----------------------------------------------------------------------===//
// BOSC AME Load/Store Intrinsic Base Classes
//===----------------------------------------------------------------------===//
// Base class for all Matrix Load Intrinsics
// Signature: void (index: i64, base_ptr: ptr, stride: i64)
class BOSC_AME_Load_Intr
    : Intrinsic<[], [llvm_i64_ty, llvm_ptr_ty, llvm_i64_ty],
                [IntrReadMem, IntrHasSideEffects,
                ImmArg<ArgIndex<0>>]>;

// Base class for all Matrix Store Intrinsics
// Signature: void (index: i64, base_ptr: ptr, stride: i64)
class BOSC_AME_Store_Intr
    : Intrinsic<[], [llvm_i64_ty, llvm_ptr_ty, llvm_i64_ty],
                [IntrWriteMem, IntrHasSideEffects,
                ImmArg<ArgIndex<0>>]>;

//===----------------------------------------------------------------------===//
// Load/Store Intrinsic Instantiations
//===----------------------------------------------------------------------===//
let TargetPrefix = "riscv" in {
  //----------------------------------------------------------------------
  // 1. Matrix Load Intrinsics
  //----------------------------------------------------------------------
  // Matrix A (TileReg)
  def int_riscv_buddy_bosc_mlae8_m   : BOSC_AME_Load_Intr;
  def int_riscv_buddy_bosc_mlae16_m  : BOSC_AME_Load_Intr;
  def int_riscv_buddy_bosc_mlae32_m  : BOSC_AME_Load_Intr;
  def int_riscv_buddy_bosc_mlae64_m  : BOSC_AME_Load_Intr;

  def int_riscv_buddy_bosc_mlate8_m  : BOSC_AME_Load_Intr;
  def int_riscv_buddy_bosc_mlate16_m : BOSC_AME_Load_Intr;
  def int_riscv_buddy_bosc_mlate32_m : BOSC_AME_Load_Intr;
  def int_riscv_buddy_bosc_mlate64_m : BOSC_AME_Load_Intr;

  def int_riscv_buddy_bosc_mltre8_m  : BOSC_AME_Load_Intr;
  def int_riscv_buddy_bosc_mltre16_m : BOSC_AME_Load_Intr;
  def int_riscv_buddy_bosc_mltre32_m : BOSC_AME_Load_Intr;
  def int_riscv_buddy_bosc_mltre64_m : BOSC_AME_Load_Intr;

  // Matrix B (TileReg)
  def int_riscv_buddy_bosc_mlbe8_m   : BOSC_AME_Load_Intr;
  def int_riscv_buddy_bosc_mlbe16_m  : BOSC_AME_Load_Intr;
  def int_riscv_buddy_bosc_mlbe32_m  : BOSC_AME_Load_Intr;
  def int_riscv_buddy_bosc_mlbe64_m  : BOSC_AME_Load_Intr;

  def int_riscv_buddy_bosc_mlbte8_m  : BOSC_AME_Load_Intr;
  def int_riscv_buddy_bosc_mlbte16_m : BOSC_AME_Load_Intr;
  def int_riscv_buddy_bosc_mlbte32_m : BOSC_AME_Load_Intr;
  def int_riscv_buddy_bosc_mlbte64_m : BOSC_AME_Load_Intr;

  // Matrix C (AccReg)
  def int_riscv_buddy_bosc_mlce8_m   : BOSC_AME_Load_Intr;
  def int_riscv_buddy_bosc_mlce16_m  : BOSC_AME_Load_Intr;
  def int_riscv_buddy_bosc_mlce32_m  : BOSC_AME_Load_Intr;
  def int_riscv_buddy_bosc_mlce64_m  : BOSC_AME_Load_Intr;

  def int_riscv_buddy_bosc_mlcte8_m  : BOSC_AME_Load_Intr;
  def int_riscv_buddy_bosc_mlcte16_m : BOSC_AME_Load_Intr;
  def int_riscv_buddy_bosc_mlcte32_m : BOSC_AME_Load_Intr;
  def int_riscv_buddy_bosc_mlcte64_m : BOSC_AME_Load_Intr;

  def int_riscv_buddy_bosc_mlacce8_m : BOSC_AME_Load_Intr;
  def int_riscv_buddy_bosc_mlacce16_m: BOSC_AME_Load_Intr;
  def int_riscv_buddy_bosc_mlacce32_m: BOSC_AME_Load_Intr;
  def int_riscv_buddy_bosc_mlacce64_m: BOSC_AME_Load_Intr;
  //----------------------------------------------------------------------
  // 2. Matrix Store Intrinsics
  //----------------------------------------------------------------------
  // Matrix A (TileReg)
  def int_riscv_buddy_bosc_msae8_m   : BOSC_AME_Store_Intr;
  def int_riscv_buddy_bosc_msae16_m  : BOSC_AME_Store_Intr;
  def int_riscv_buddy_bosc_msae32_m  : BOSC_AME_Store_Intr;
  def int_riscv_buddy_bosc_msae64_m  : BOSC_AME_Store_Intr;

  def int_riscv_buddy_bosc_msate8_m  : BOSC_AME_Store_Intr;
  def int_riscv_buddy_bosc_msate16_m : BOSC_AME_Store_Intr;
  def int_riscv_buddy_bosc_msate32_m : BOSC_AME_Store_Intr;
  def int_riscv_buddy_bosc_msate64_m : BOSC_AME_Store_Intr;

  def int_riscv_buddy_bosc_mstre8_m  : BOSC_AME_Store_Intr;
  def int_riscv_buddy_bosc_mstre16_m : BOSC_AME_Store_Intr;
  def int_riscv_buddy_bosc_mstre32_m : BOSC_AME_Store_Intr;
  def int_riscv_buddy_bosc_mstre64_m : BOSC_AME_Store_Intr;

  // Matrix B (TileReg)
  def int_riscv_buddy_bosc_msbe8_m   : BOSC_AME_Store_Intr;
  def int_riscv_buddy_bosc_msbe16_m  : BOSC_AME_Store_Intr;
  def int_riscv_buddy_bosc_msbe32_m  : BOSC_AME_Store_Intr;
  def int_riscv_buddy_bosc_msbe64_m  : BOSC_AME_Store_Intr;

  def int_riscv_buddy_bosc_msbte8_m  : BOSC_AME_Store_Intr;
  def int_riscv_buddy_bosc_msbte16_m : BOSC_AME_Store_Intr;
  def int_riscv_buddy_bosc_msbte32_m : BOSC_AME_Store_Intr;
  def int_riscv_buddy_bosc_msbte64_m : BOSC_AME_Store_Intr;

  // Matrix C (AccReg)
  def int_riscv_buddy_bosc_msce8_m   : BOSC_AME_Store_Intr;
  def int_riscv_buddy_bosc_msce16_m  : BOSC_AME_Store_Intr;
  def int_riscv_buddy_bosc_msce32_m  : BOSC_AME_Store_Intr;
  def int_riscv_buddy_bosc_msce64_m  : BOSC_AME_Store_Intr;

  def int_riscv_buddy_bosc_mscte8_m  : BOSC_AME_Store_Intr;
  def int_riscv_buddy_bosc_mscte16_m : BOSC_AME_Store_Intr;
  def int_riscv_buddy_bosc_mscte32_m : BOSC_AME_Store_Intr;
  def int_riscv_buddy_bosc_mscte64_m : BOSC_AME_Store_Intr;

  def int_riscv_buddy_bosc_msacce8_m : BOSC_AME_Store_Intr;
  def int_riscv_buddy_bosc_msacce16_m: BOSC_AME_Store_Intr;
  def int_riscv_buddy_bosc_msacce32_m: BOSC_AME_Store_Intr;
  def int_riscv_buddy_bosc_msacce64_m: BOSC_AME_Store_Intr;
}

//===----------------------------------------------------------------------===//
// Data Move / Broadcast / Transpose Intrinsic Base Classes
//===----------------------------------------------------------------------===//

// 1. Matrix to Matrix (2 Immediates: md, ms1) -> No return
class BOSC_AME_Move_T_T_Intr
    : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty],
                [IntrNoMem, IntrHasSideEffects,
                ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

// 2. Matrix to Matrix with GPR Index (2 Immediates: md, ms1; 1 GPR: rs2) -> No return
class BOSC_AME_Move_A_T_Intr
    : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
                [IntrNoMem, IntrHasSideEffects,
                ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

// 3. Matrix to Matrix with Immediate Index (3 Immediates: md, ms1, imm) -> No return
class BOSC_AME_MoveImm_A_T_Intr
    : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
                [IntrNoMem, IntrHasSideEffects,
                ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>,
                ImmArg<ArgIndex<2>>]>;

// 4. Matrix to Scalar Move (1 Immediate: ms1; 1 GPR: rs2) -> Returns GPR (i64)
class BOSC_AME_Move_X_T_Intr
    : Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty],
                [IntrNoMem, IntrHasSideEffects,
                ImmArg<ArgIndex<0>>]>;

// 5. Scalar to Matrix Move (1 Immediate: md; 2 GPRs: rs1, rs2) -> No return
class BOSC_AME_Move_T_X_Intr
    : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
                [IntrNoMem, IntrHasSideEffects,
                ImmArg<ArgIndex<0>>]>;


//===----------------------------------------------------------------------===//
// Intrinsic Instantiations
//===----------------------------------------------------------------------===//
let TargetPrefix = "riscv" in {
  //----------------------------------------------------------------------
  // 1. Matrix-Matrix Move Pseudos (Pure Matrix)
  //----------------------------------------------------------------------
  def int_riscv_buddy_bosc_mmve8_t_t  : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mmve16_t_t : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mmve32_t_t : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mmve64_t_t : BOSC_AME_Move_T_T_Intr;

  def int_riscv_buddy_bosc_mmve8_a_a  : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mmve16_a_a : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mmve32_a_a : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mmve64_a_a : BOSC_AME_Move_T_T_Intr;
  //----------------------------------------------------------------------
  // 2. Matrix-Matrix Move (with GPR Index)
  //----------------------------------------------------------------------
  def int_riscv_buddy_bosc_mmve8_a_t  : BOSC_AME_Move_A_T_Intr;
  def int_riscv_buddy_bosc_mmve16_a_t : BOSC_AME_Move_A_T_Intr;
  def int_riscv_buddy_bosc_mmve32_a_t : BOSC_AME_Move_A_T_Intr;
  def int_riscv_buddy_bosc_mmve64_a_t : BOSC_AME_Move_A_T_Intr;

  def int_riscv_buddy_bosc_mmve8_t_a  : BOSC_AME_Move_A_T_Intr;
  def int_riscv_buddy_bosc_mmve16_t_a : BOSC_AME_Move_A_T_Intr;
  def int_riscv_buddy_bosc_mmve32_t_a : BOSC_AME_Move_A_T_Intr;
  def int_riscv_buddy_bosc_mmve64_t_a : BOSC_AME_Move_A_T_Intr;
  //----------------------------------------------------------------------
  // 3. Matrix-Matrix Move (with Immediate Index)
  //----------------------------------------------------------------------
  def int_riscv_buddy_bosc_mmvie8_a_t  : BOSC_AME_MoveImm_A_T_Intr;
  def int_riscv_buddy_bosc_mmvie16_a_t : BOSC_AME_MoveImm_A_T_Intr;
  def int_riscv_buddy_bosc_mmvie32_a_t : BOSC_AME_MoveImm_A_T_Intr;
  def int_riscv_buddy_bosc_mmvie64_a_t : BOSC_AME_MoveImm_A_T_Intr;

  def int_riscv_buddy_bosc_mmvie8_t_a  : BOSC_AME_MoveImm_A_T_Intr;
  def int_riscv_buddy_bosc_mmvie16_t_a : BOSC_AME_MoveImm_A_T_Intr;
  def int_riscv_buddy_bosc_mmvie32_t_a : BOSC_AME_MoveImm_A_T_Intr;
  def int_riscv_buddy_bosc_mmvie64_t_a : BOSC_AME_MoveImm_A_T_Intr;
  //----------------------------------------------------------------------
  // 4. Matrix to Scalar Move (Returns GPR)
  //----------------------------------------------------------------------
  def int_riscv_buddy_bosc_mmve8_x_t  : BOSC_AME_Move_X_T_Intr;
  def int_riscv_buddy_bosc_mmve16_x_t : BOSC_AME_Move_X_T_Intr;
  def int_riscv_buddy_bosc_mmve32_x_t : BOSC_AME_Move_X_T_Intr;
  def int_riscv_buddy_bosc_mmve64_x_t : BOSC_AME_Move_X_T_Intr;

  def int_riscv_buddy_bosc_mmve8_x_a  : BOSC_AME_Move_X_T_Intr;
  def int_riscv_buddy_bosc_mmve16_x_a : BOSC_AME_Move_X_T_Intr;
  def int_riscv_buddy_bosc_mmve32_x_a : BOSC_AME_Move_X_T_Intr;
  def int_riscv_buddy_bosc_mmve64_x_a : BOSC_AME_Move_X_T_Intr;

  def int_riscv_buddy_bosc_mfmve8_f_t  : BOSC_AME_Move_X_T_Intr;
  def int_riscv_buddy_bosc_mfmve16_f_t : BOSC_AME_Move_X_T_Intr;
  def int_riscv_buddy_bosc_mfmve32_f_t : BOSC_AME_Move_X_T_Intr;
  def int_riscv_buddy_bosc_mfmve64_f_t : BOSC_AME_Move_X_T_Intr;

  def int_riscv_buddy_bosc_mfmve8_f_a  : BOSC_AME_Move_X_T_Intr;
  def int_riscv_buddy_bosc_mfmve16_f_a : BOSC_AME_Move_X_T_Intr;
  def int_riscv_buddy_bosc_mfmve32_f_a : BOSC_AME_Move_X_T_Intr;
  def int_riscv_buddy_bosc_mfmve64_f_a : BOSC_AME_Move_X_T_Intr;
  //----------------------------------------------------------------------
  // 5. Scalar GPR to Matrix Move
  //----------------------------------------------------------------------
  def int_riscv_buddy_bosc_mmve8_t_x  : BOSC_AME_Move_T_X_Intr;
  def int_riscv_buddy_bosc_mmve16_t_x : BOSC_AME_Move_T_X_Intr;
  def int_riscv_buddy_bosc_mmve32_t_x : BOSC_AME_Move_T_X_Intr;
  def int_riscv_buddy_bosc_mmve64_t_x : BOSC_AME_Move_T_X_Intr;

  def int_riscv_buddy_bosc_mmve8_a_x  : BOSC_AME_Move_T_X_Intr;
  def int_riscv_buddy_bosc_mmve16_a_x : BOSC_AME_Move_T_X_Intr;
  def int_riscv_buddy_bosc_mmve32_a_x : BOSC_AME_Move_T_X_Intr;
  def int_riscv_buddy_bosc_mmve64_a_x : BOSC_AME_Move_T_X_Intr;

  def int_riscv_buddy_bosc_mfmve8_t_f  : BOSC_AME_Move_T_X_Intr;
  def int_riscv_buddy_bosc_mfmve16_t_f : BOSC_AME_Move_T_X_Intr;
  def int_riscv_buddy_bosc_mfmve32_t_f : BOSC_AME_Move_T_X_Intr;
  def int_riscv_buddy_bosc_mfmve64_t_f : BOSC_AME_Move_T_X_Intr;

  def int_riscv_buddy_bosc_mfmve8_a_f  : BOSC_AME_Move_T_X_Intr;
  def int_riscv_buddy_bosc_mfmve16_a_f : BOSC_AME_Move_T_X_Intr;
  def int_riscv_buddy_bosc_mfmve32_a_f : BOSC_AME_Move_T_X_Intr;
  def int_riscv_buddy_bosc_mfmve64_a_f : BOSC_AME_Move_T_X_Intr;
  //----------------------------------------------------------------------
  // 6. Broadcast Intrinsics
  //----------------------------------------------------------------------
  def int_riscv_buddy_bosc_mbcar_m : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mbcbr_m : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mbccr_m : BOSC_AME_Move_T_T_Intr;

  def int_riscv_buddy_bosc_mbcace8_m  : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mbcace16_m : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mbcace32_m : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mbcace64_m : BOSC_AME_Move_T_T_Intr;

  def int_riscv_buddy_bosc_mbcbce8_m  : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mbcbce16_m : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mbcbce32_m : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mbcbce64_m : BOSC_AME_Move_T_T_Intr;

  def int_riscv_buddy_bosc_mbccce8_m  : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mbccce16_m : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mbccce32_m : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mbccce64_m : BOSC_AME_Move_T_T_Intr;

  def int_riscv_buddy_bosc_mbcaee8_m  : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mbcaee16_m : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mbcaee32_m : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mbcaee64_m : BOSC_AME_Move_T_T_Intr;

  def int_riscv_buddy_bosc_mbcbee8_m  : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mbcbee16_m : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mbcbee32_m : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mbcbee64_m : BOSC_AME_Move_T_T_Intr;

  def int_riscv_buddy_bosc_mbccee8_m  : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mbccee16_m : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mbccee32_m : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mbccee64_m : BOSC_AME_Move_T_T_Intr;
  //----------------------------------------------------------------------
  // 7. Transpose Intrinsics
  //----------------------------------------------------------------------
  def int_riscv_buddy_bosc_mtae8_m  : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mtae16_m : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mtae32_m : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mtae64_m : BOSC_AME_Move_T_T_Intr;

  def int_riscv_buddy_bosc_mtbe8_m  : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mtbe16_m : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mtbe32_m : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mtbe64_m : BOSC_AME_Move_T_T_Intr;

  def int_riscv_buddy_bosc_mtce8_m  : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mtce16_m : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mtce32_m : BOSC_AME_Move_T_T_Intr;
  def int_riscv_buddy_bosc_mtce64_m : BOSC_AME_Move_T_T_Intr;
}

//===----------------------------------------------------------------------===//
// Element-Wise Intrinsic Base Classes
//===----------------------------------------------------------------------===//
// Base class for 3-operand element-wise intrinsics (md, ms1, ms2)
// Signature: void (md_idx: i64, ms1_idx: i64, ms2_idx: i64)
class BOSC_AME_EleWise3_Intr
    : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>,
                 ImmArg<ArgIndex<2>>]>;

// Base class for 2-operand element-wise intrinsics (md, ms1) - Used for SQRT
// Signature: void (md_idx: i64, ms1_idx: i64)
class BOSC_AME_EleWise2_Intr
    : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

//===----------------------------------------------------------------------===//
// Element-Wise Intrinsic Instantiations
//===----------------------------------------------------------------------===//
let TargetPrefix = "riscv" in {
  //===--------------------------------------------------------------------===//
  // 1. ADD / SUB - No Widen
  //===--------------------------------------------------------------------===//
  def int_riscv_buddy_bosc_maddu_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_maddu_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_maddu_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_maddu_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_maddu_w_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_maddu_dw_mm : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_msaddu_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msaddu_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msaddu_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msaddu_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msaddu_w_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msaddu_dw_mm : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_madd_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_madd_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_madd_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_madd_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_madd_w_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_madd_dw_mm : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_msadd_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msadd_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msadd_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msadd_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msadd_w_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msadd_dw_mm : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_mfadd_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfadd_cf_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfadd_hf_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfadd_f_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfadd_d_mm  : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_msubu_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msubu_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msubu_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msubu_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msubu_w_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msubu_dw_mm : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_mssubu_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mssubu_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mssubu_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mssubu_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mssubu_w_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mssubu_dw_mm : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_msub_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msub_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msub_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msub_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msub_w_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msub_dw_mm : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_mssub_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mssub_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mssub_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mssub_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mssub_w_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mssub_dw_mm : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_mfsub_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfsub_cf_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfsub_hf_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfsub_f_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfsub_d_mm  : BOSC_AME_EleWise3_Intr;
  //===--------------------------------------------------------------------===//
  // 2. ADD / SUB - Double Widen
  //===--------------------------------------------------------------------===//
  def int_riscv_buddy_bosc_mwaddu_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwaddu_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwaddu_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwaddu_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwaddu_w_mm  : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_mwadd_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwadd_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwadd_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwadd_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwadd_w_mm  : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_mfwadd_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfwadd_cf_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfwadd_hf_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfwadd_f_mm  : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_mwsubu_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwsubu_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwsubu_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwsubu_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwsubu_w_mm  : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_mwsub_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwsub_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwsub_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwsub_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwsub_w_mm  : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_mfwsub_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfwsub_cf_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfwsub_hf_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfwsub_f_mm  : BOSC_AME_EleWise3_Intr;
  //===--------------------------------------------------------------------===//
  // 3. MIN / MAX - No Widen
  //===--------------------------------------------------------------------===//
  def int_riscv_buddy_bosc_mminu_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mminu_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mminu_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mminu_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mminu_w_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mminu_dw_mm : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_mmaxu_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmaxu_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmaxu_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmaxu_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmaxu_w_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmaxu_dw_mm : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_mmin_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmin_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmin_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmin_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmin_w_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmin_dw_mm : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_mmax_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmax_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmax_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmax_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmax_w_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmax_dw_mm : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_mfmin_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfmin_cf_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfmin_hf_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfmin_f_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfmin_d_mm  : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_mfmax_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfmax_cf_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfmax_hf_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfmax_f_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfmax_d_mm  : BOSC_AME_EleWise3_Intr;
  //===--------------------------------------------------------------------===//
  // 4. MUL / MULH / DIV - No Widen
  //===--------------------------------------------------------------------===//
  def int_riscv_buddy_bosc_msmulu_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msmulu_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msmulu_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msmulu_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msmulu_w_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msmulu_dw_mm : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_mmul_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmul_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmul_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmul_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmul_w_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmul_dw_mm : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_msmul_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msmul_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msmul_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msmul_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msmul_w_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msmul_dw_mm : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_mfmul_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfmul_cf_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfmul_hf_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfmul_f_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfmul_d_mm  : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_mmulhu_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmulhu_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmulhu_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmulhu_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmulhu_w_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmulhu_dw_mm : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_mmulh_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmulh_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmulh_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmulh_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmulh_w_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmulh_dw_mm : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_mmulhsu_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmulhsu_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmulhsu_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmulhsu_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmulhsu_w_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mmulhsu_dw_mm : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_msmulsu_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msmulsu_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msmulsu_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msmulsu_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msmulsu_w_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msmulsu_dw_mm : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_mfdiv_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfdiv_cf_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfdiv_hf_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfdiv_f_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfdiv_d_mm  : BOSC_AME_EleWise3_Intr;
  //===--------------------------------------------------------------------===//
  // 5. MUL - Double Widen
  //===--------------------------------------------------------------------===//
  def int_riscv_buddy_bosc_mwmulu_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwmulu_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwmulu_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwmulu_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwmulu_w_mm  : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_mwmul_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwmul_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwmul_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwmul_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwmul_w_mm  : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_mwmulsu_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwmulsu_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwmulsu_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwmulsu_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mwmulsu_w_mm  : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_mfwmul_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfwmul_cf_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfwmul_hf_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mfwmul_f_mm  : BOSC_AME_EleWise3_Intr;
  //===--------------------------------------------------------------------===//
  // 6. LOGIC & SHIFT - No Widen
  //===--------------------------------------------------------------------===//
  def int_riscv_buddy_bosc_mand_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mor_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_mxor_mm : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_msll_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msll_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msll_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msll_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msll_w_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msll_dw_mm : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_msrl_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msrl_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msrl_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msrl_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msrl_w_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msrl_dw_mm : BOSC_AME_EleWise3_Intr;

  def int_riscv_buddy_bosc_msra_mm    : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msra_hb_mm : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msra_b_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msra_h_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msra_w_mm  : BOSC_AME_EleWise3_Intr;
  def int_riscv_buddy_bosc_msra_dw_mm : BOSC_AME_EleWise3_Intr;
  //===--------------------------------------------------------------------===//
  // 7. SQRT - No Widen (Unary, 2-Operand)
  // NOTE: Uses EleWise2_Intr
  //===--------------------------------------------------------------------===//
  def int_riscv_buddy_bosc_mfsqrt_mm    : BOSC_AME_EleWise2_Intr;
  def int_riscv_buddy_bosc_mfsqrt_cf_mm : BOSC_AME_EleWise2_Intr;
  def int_riscv_buddy_bosc_mfsqrt_hf_mm : BOSC_AME_EleWise2_Intr;
  def int_riscv_buddy_bosc_mfsqrt_f_mm  : BOSC_AME_EleWise2_Intr;
  def int_riscv_buddy_bosc_mfsqrt_d_mm  : BOSC_AME_EleWise2_Intr;
}

//===----------------------------------------------------------------------===//
// Matrix Multiply-Accumulate (MMA) Intrinsic Base Class
//===----------------------------------------------------------------------===//
// Base class for all 3-operand Matrix Multiply-Accumulate intrinsics
// Signature: void (md_idx: i64, ms1_idx: i64, ms2_idx: i64)
class BOSC_AME_MMA_Intr
    : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>,
                 ImmArg<ArgIndex<2>>]>;

//===----------------------------------------------------------------------===//
// MMA Intrinsic Instantiations
//===----------------------------------------------------------------------===//
let TargetPrefix = "riscv" in {
  //===--------------------------------------------------------------------===//
  // 1. No-widen matrix multiply-accumulate
  //===--------------------------------------------------------------------===//
  // int
  def int_riscv_buddy_bosc_mma_mm    : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_mma_h_mm  : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_mma_w_mm  : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_mma_dw_mm : BOSC_AME_MMA_Intr;

  // int, saturated
  def int_riscv_buddy_bosc_msma_mm    : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_msma_h_mm  : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_msma_w_mm  : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_msma_dw_mm : BOSC_AME_MMA_Intr;

  // uint
  def int_riscv_buddy_bosc_mmau_mm    : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_mmau_h_mm  : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_mmau_w_mm  : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_mmau_dw_mm : BOSC_AME_MMA_Intr;

  // uint, saturated
  def int_riscv_buddy_bosc_msmau_mm    : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_msmau_h_mm  : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_msmau_w_mm  : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_msmau_dw_mm : BOSC_AME_MMA_Intr;

  // float point
  def int_riscv_buddy_bosc_mfma_mm    : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_mfma_hf_mm : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_mfma_f_mm  : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_mfma_d_mm  : BOSC_AME_MMA_Intr;
  //===--------------------------------------------------------------------===//
  // 2. Double-widen matrix multiply-accumulate
  //===--------------------------------------------------------------------===//
  // uint
  def int_riscv_buddy_bosc_mwmau_mm   : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_mwmau_h_mm : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_mwmau_w_mm : BOSC_AME_MMA_Intr;

  // uint, saturated
  def int_riscv_buddy_bosc_mswmau_mm   : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_mswmau_h_mm : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_mswmau_w_mm : BOSC_AME_MMA_Intr;

  // int
  def int_riscv_buddy_bosc_mwma_mm   : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_mwma_h_mm : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_mwma_w_mm : BOSC_AME_MMA_Intr;

  // int, saturated
  def int_riscv_buddy_bosc_mswma_mm   : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_mswma_h_mm : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_mswma_w_mm : BOSC_AME_MMA_Intr;

  // float point
  def int_riscv_buddy_bosc_mfwma_mm    : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_mfwma_cf_mm : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_mfwma_hf_mm : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_mfwma_f_mm  : BOSC_AME_MMA_Intr;
  //===--------------------------------------------------------------------===//
  // 3. Quad-widen matrix multiply-accumulate
  //===--------------------------------------------------------------------===//
  // uint
  def int_riscv_buddy_bosc_mqmau_mm   : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_mqmau_b_mm : BOSC_AME_MMA_Intr;

  // uint, saturated
  def int_riscv_buddy_bosc_msqmau_mm   : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_msqmau_b_mm : BOSC_AME_MMA_Intr;

  // int
  def int_riscv_buddy_bosc_mqma_mm   : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_mqma_b_mm : BOSC_AME_MMA_Intr;

  // int, saturated
  def int_riscv_buddy_bosc_msqma_mm   : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_msqma_b_mm : BOSC_AME_MMA_Intr;

  // float point
  def int_riscv_buddy_bosc_mfqma_mm    : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_mfqma_cf_mm : BOSC_AME_MMA_Intr;
  //===--------------------------------------------------------------------===//
  // 4. Oct-widen matrix multiply-accumulate
  //===--------------------------------------------------------------------===//
  // uint
  def int_riscv_buddy_bosc_momau_mm    : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_momau_hb_mm : BOSC_AME_MMA_Intr;

  // uint, saturated
  def int_riscv_buddy_bosc_msomau_mm    : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_msomau_hb_mm : BOSC_AME_MMA_Intr;

  // int
  def int_riscv_buddy_bosc_moma_mm    : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_moma_hb_mm : BOSC_AME_MMA_Intr;

  // int, saturated
  def int_riscv_buddy_bosc_msoma_mm    : BOSC_AME_MMA_Intr;
  def int_riscv_buddy_bosc_msoma_hb_mm : BOSC_AME_MMA_Intr;
}
