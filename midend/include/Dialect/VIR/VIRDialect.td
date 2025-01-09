//===- VIRDialect.td - Dynamic Vector IR Dialect Definition ---------------===//
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
// This is the top level file for the dynamic vector IR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef VIR_VIRDIALECT_TD
#define VIR_VIRDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Vector Experiment Dialect Definition.
//===----------------------------------------------------------------------===//

def VIR_Dialect : Dialect {
  let name = "vir";
  let summary = "The Dynamic Vector IR Dialect.";
  let description = [{
    The `VIR` dialect is a dynamic vector IR, serving as a unified hardware
    abstraction layer for multiple hardware backends.
  }];
  let cppNamespace = "::buddy::vir";
}

//===----------------------------------------------------------------------===//
// Base Vector Experiment Operation Definition.
//===----------------------------------------------------------------------===//

class VIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<VIR_Dialect, mnemonic, traits>;

#endif // VIR_VIRDIALECT_TD
