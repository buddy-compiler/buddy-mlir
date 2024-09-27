//===- TraceDialect.td - Trace Dialect Definition ----------*- tablegen -*-===//
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
// This is the top level file for the Trace dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TRACE_TRACEDIALECT_TD
#define TRACE_TRACEDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Trace Dialect Definition.
//===----------------------------------------------------------------------===//

def Trace_Dialect : Dialect {
  let name = "trace";
  let summary = "The Trace Dialect.";
  let description = [{
    The `trace` dialect is for trace.
  }];
  let cppNamespace = "::buddy::trace";
}

//===----------------------------------------------------------------------===//
// Base Trace Operation Definition.
//===----------------------------------------------------------------------===//

class Trace_Op<string mnemonic, list<Trait> traits = []> :
    Op<Trace_Dialect, mnemonic, traits>;

#endif // TRACE_TRACEDIALECT_TD
