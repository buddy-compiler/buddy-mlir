//===- DAPDialect.td - dap Dialect Definition --------------*- tablegen -*-===//
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
// This is the top level file for the DAP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DAP_DAPDIALECT_TD
#define DAP_DAPDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// DAP Dialect Definition.
//===----------------------------------------------------------------------===//

def DAP_Dialect : Dialect {
  let name = "dap";
  let summary = "The DAP Dialect.";
  let description = [{
    Digital audio processing dialect(`DAP`) dialect is created for the purpose
    of developing a MLIR backend for performing audio processing operations.
  }];
  let cppNamespace = "::buddy::dap";
}

//===----------------------------------------------------------------------===//
// Base DAP Operation Definition.
//===----------------------------------------------------------------------===//

class DAP_Op<string mnemonic, list<Trait> traits = []> :
    Op<DAP_Dialect, mnemonic, traits>;

#endif // DAP_DAPDIALECT_TD
