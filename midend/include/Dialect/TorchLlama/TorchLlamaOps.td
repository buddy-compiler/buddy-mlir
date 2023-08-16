//===- TorchLlamaOps.td - TorchLlama Dialect Ops -------------------------*- tablegen -*-===//
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
// This is the top level file for operations in the torchllama dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHLLAMA_TORCHLLAMAOPS_TD
#define TORCHLLAMA_TORCHLLAMAOPS_TD

include "TorchLlamaDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// TorchLlama constant operation.
//===----------------------------------------------------------------------===//

def TorchLlama_ConstantOp : TorchLlama_Op<"constant", []> {
  let summary = "Constant Operation vs prim::Constant";

  let arguments = (ins Tensor:$value)
  let results = (outs Tensor);
  
  let builders = [
    OpBuilder<"Builder *b, OperationState &state, Value input, Type resultType">
  ];
}

#endif // TORCHLLAMA_TORCHLLAMAOPS_TD
