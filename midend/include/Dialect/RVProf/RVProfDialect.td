//===- RVProfDialect.td - RVProf Dialect Definition --------*- tablegen -*-===//
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
// This is the top level file for the RVProf dialect.
// RVProf (RISC-V Profiling) is a dialect for runtime performance analysis
// on RISC-V platforms using rdcycle instruction.
//
//===----------------------------------------------------------------------===//

#ifndef RVPROF_RVPROFDIALECT_TD
#define RVPROF_RVPROFDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// RVProf Dialect Definition.
//===----------------------------------------------------------------------===//

def RVProf_Dialect : Dialect {
  let name = "rvprof";
  let summary = "RISC-V Profiling Dialect for runtime performance analysis.";
  let description = [{
    The `rvprof` dialect provides operations for instrumenting MLIR code
    with profiling markers that are lowered to RISC-V rdcycle-based timing
    instrumentation. This dialect is specifically designed for RISC-V platforms
    and outputs Perfetto trace format for visualization.
  }];
  let cppNamespace = "::buddy::rvprof";
}

//===----------------------------------------------------------------------===//
// Base RVProf Operation Definition.
//===----------------------------------------------------------------------===//

class RVProf_Op<string mnemonic, list<Trait> traits = []> :
    Op<RVProf_Dialect, mnemonic, traits>;

#endif // RVPROF_RVPROFDIALECT_TD
