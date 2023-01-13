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
def int_riscv_mvin : Intrinsic<[],[llvm_i64_ty, llvm_i64_ty],[]>;

let TargetPrefix = "riscv" in
def int_riscv_mvout : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in 
def int_riscv_flush : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in 
def int_riscv_configld : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in 
def int_riscv_configst : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;
