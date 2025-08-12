//===-- VIROps.td - Dynamic Vector IR Operation Definitions ---------------===//
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
// This file defines the operation set for the VIR dialect,
// including vector length management and dynamic vector operations.
//
//===----------------------------------------------------------------------===//

#ifndef VIR_VIROPS_TD
#define VIR_VIROPS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "VIR/VIRDialect.td"

//===----------------------------------------------------------------------===//
// Base VIR Operation Class
//===----------------------------------------------------------------------===//

class VIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<VIR_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Dynamic Vector Region Operation - set_vl
//===----------------------------------------------------------------------===//

def VIR_SetVLOp : VIR_Op<"set_vl"> {
  let summary = "Dynamic Vector Region SetVL Operation.";
  let description = [{
    The `set_vl` operation sets the vector length for subsequent dynamic vector
    operations within its region. This enables runtime vector length management
    for hardware-agnostic vector computations.
  }];
  let arguments = (ins Index:$vl);
  let results = (outs Variadic<AnyTypeOf<[AnyInteger, AnyFloat]>>:$results);
  // TODO: Determine the region limitation.
  let regions = (region AnyRegion:$region);

  let assemblyFormat = "$vl attr-dict `:` type($vl)"
      "$region (`->` type($results)^)?";
}

#endif // VIR_VIROPS_TD
