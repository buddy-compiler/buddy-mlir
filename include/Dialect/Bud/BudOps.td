//===- BudOps.td - Bud Dialect Ops -------------------------*- tablegen -*-===//
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
// This is the top level file for operations in the bud dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BUD_BUDOPS_TD
#define BUD_BUDOPS_TD

include "BudDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Bud_TestConstantOp : Bud_Op<"test_constant", []> {
  let summary = "Test Constant Operation.";

  let results = (outs AnyInteger:$result);

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

#endif // BUD_BUDOPS_TD