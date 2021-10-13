//===- DIPOps.td - DIP Dialect Ops -------------------------*- tablegen -*-===//
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
// This is the top level file for operations in the DIP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DIP_DIPOPS_TD
#define DIP_DIPOPS_TD

include "DIPDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def DIP_Corr2DOp : DIP_Op<"Corr2D">
{
  let summary = "Operation for performing 2D correlation";

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "inputMemref",
                           [MemRead]>:$memrefI,
                       Arg<AnyRankedOrUnrankedMemRef, "kernelMemref",
                           [MemRead]>:$memrefK,
                       Arg<AnyRankedOrUnrankedMemRef, "outputMemref",
                           [MemRead]>:$memrefCO,
                       Index : $centerX, Index : $centerY, 
                       Index : $boundaryOption);

  let assemblyFormat = [{
    $memrefI `,` $memrefK `,` $memrefCO `,` $centerX `,` $centerY `,` $boundaryOption attr-dict `:` type($memrefI) `,` type($memrefK) `,` type($memrefCO) `,` type($centerX) `,` type($centerY) `,` type($boundaryOption)
  }];
}

#endif // DIP_DIPOPS_TD
