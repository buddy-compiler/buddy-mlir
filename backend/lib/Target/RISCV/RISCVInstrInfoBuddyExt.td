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

let hasSideEffects = 1, mayLoad = 1, mayStore = 1 , Predicates = [HasBuddyExt] in
def MVIN : RVInstR<0b0000000, 0b000, OPC_CUSTOM_0, (outs),
                   (ins GPR:$rd, GPR:$rs1), "mvin","$rd, $rs1"> {
  let rs2 = 0;
}

let Predicates = [HasBuddyExt] in
def: PatGprGpr<int_riscv_mvin, MVIN>;
