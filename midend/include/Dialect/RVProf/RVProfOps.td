//===- RVProfOps.td - RVProf Dialect Ops -------------------*- tablegen -*-===//
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
// This is the top level file for operations in the RVProf dialect.
//
//===----------------------------------------------------------------------===//

#ifndef RVPROF_RVPROFOPS_TD
#define RVPROF_RVPROFOPS_TD

include "RVProf/RVProfDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// RVProf Operations
//===----------------------------------------------------------------------===//

def RVProf_RegionBeginOp : RVProf_Op<"region_begin", []> {
  let summary = "Mark the beginning of a profiled region";
  let description = [{
    The `rvprof.region_begin` operation marks the start of a code region
    to be profiled. It takes a string attribute representing the region name.
    This operation will be lowered to a call to __rvprof_region_begin() runtime
    function that records the start timestamp using RISC-V rdcycle instruction.

    Example:
    ```mlir
    rvprof.region_begin "linalg.matmul_0"
    ```
  }];

  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "$name attr-dict";
}

def RVProf_RegionEndOp : RVProf_Op<"region_end", []> {
  let summary = "Mark the end of a profiled region";
  let description = [{
    The `rvprof.region_end` operation marks the end of a code region
    to be profiled. It takes a string attribute representing the region name,
    which should match the corresponding region_begin operation.
    This operation will be lowered to a call to __rvprof_region_end() runtime
    function that records the end timestamp using RISC-V rdcycle instruction.

    Example:
    ```mlir
    rvprof.region_end "linalg.matmul_0"
    ```
  }];

  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "$name attr-dict";
}

#endif // RVPROF_RVPROFOPS_TD
