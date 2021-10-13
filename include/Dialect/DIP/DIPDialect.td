//===- DIPDialect.td - DIP Dialect Definition --------------*- tablegen -*-===//
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
// This is the top level file for the DIP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DIP_DIPDIALECT_TD
#define DIP_DIPDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// DIP Dialect Definition.
//===----------------------------------------------------------------------===//

def DIP_Dialect : Dialect {
  let name = "DIP";
  let summary = "The DIP Dialect.";
  let description = [{
    The `DIP` dialect is for digital image processing.
  }];
  let cppNamespace = "::Buddy::DIP";
}

//===----------------------------------------------------------------------===//
// Base DIP Operation Definition.
//===----------------------------------------------------------------------===//

class DIP_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<DIP_Dialect, mnemonic, traits>;

#endif // DIP_DIPDIALECT_TD
