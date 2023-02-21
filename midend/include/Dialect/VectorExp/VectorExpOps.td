//===- VectorExpOps.td - Vector Experiment Dialect Ops --------------------===//
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
// This is the top level file for operations in the vector experiment dialect.
//
//===----------------------------------------------------------------------===//

#ifndef VECTOREXP_VECTOREXPOPS_TD
#define VECTOREXP_VECTOREXPOPS_TD

include "VectorExpDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Vector Predication Operation
//===----------------------------------------------------------------------===//

def VectorExp_PredicationOp : VectorExp_Op<"predication"> {
  let summary = "Vector Experiment Predication Operation.";
  let arguments = (ins AnyVector:$mask,
                       AnyInteger:$vl
                  );
  let results = (outs AnyType:$result);
  let regions = (region AnyRegion:$region);

  let assemblyFormat = "$mask `,` $vl attr-dict `:` type($mask) `,` type($vl) "
      "$region `:` type($result)";
}

#endif // VECTOREXP_VECTOREXPOPS_TD
