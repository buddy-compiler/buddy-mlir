//===- VIROps.td - Dynamic Vector IR Dialect Ops --------------------------===//
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
// This is the top level file for operations in the dynamic vector IR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef VIR_VIROPS_TD
#define VIR_VIROPS_TD

include "VIR/VIRDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Dynamic Vector Region Operation - set_vl
//===----------------------------------------------------------------------===//

def VIR_SetVLOp : VIR_Op<"set_vl"> {
  let summary = "Dynamic Vector Region SetVL Operation.";
  let arguments = (ins Index:$vl);
  let results = (outs Variadic<AnyTypeOf<[AnyInteger, AnyFloat]>>:$results);
  // TODO: Determine the region limitation.
  let regions = (region AnyRegion:$region);

  let assemblyFormat = "$vl attr-dict `:` type($vl)"
      "$region (`->` type($results)^)?";
}

#endif // VIR_VIROPS_TD
