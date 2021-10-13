//===- BudDialect.td - Bud Dialect Definition --------------*- tablegen -*-===//
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
// This is the top level file for the bud dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BUD_BUDDIALECT_TD
#define BUD_BUDDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Bud Dialect Definition.
//===----------------------------------------------------------------------===//

def Bud_Dialect : Dialect {
  let name = "bud";
  let summary = "The Bud Dialect.";
  let description = [{
    The `bud` dialect is for testing and demonstrating.
  }];
  let cppNamespace = "::buddy::bud";
}

//===----------------------------------------------------------------------===//
// Base Bud Operation Definition.
//===----------------------------------------------------------------------===//

class Bud_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Bud_Dialect, mnemonic, traits>;

#endif // BUD_BUDDIALECT_TD
