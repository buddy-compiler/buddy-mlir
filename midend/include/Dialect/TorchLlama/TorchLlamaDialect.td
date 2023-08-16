//===- TorchLlamaDialect.td - TorchLlama Dialect Definition --------------*- tablegen -*-===//
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
// This is the top level file for the torchllama dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHLLAMA_TORCHLLAMADIALECT_TD
#define TORCHLLAMA_TORCHLLAMADIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TorchLlama Dialect Definition.
//===----------------------------------------------------------------------===//

def TorchLlama_Dialect : Dialect {
  let name = "torchllama";
  let summary = "The TorchLlama Dialect.";
  let description = [{
    The `torchllama` dialect is for testing and demonstrating.
  }];
  let cppNamespace = "::buddy::torchllama";
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base TorchLlama Operation Definition.
//===----------------------------------------------------------------------===//

class TorchLlama_Op<string mnemonic, list<Trait> traits = []> :
    Op<TorchLlama_Dialect, mnemonic, traits>;

#endif // TORCHLLAMA_TORCHLLAMADIALECT_TD
