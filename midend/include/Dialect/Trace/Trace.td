//===-- Trace.td - Buddy trace dialect definitions ---------*- tablegen -*-===//
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

#ifndef BUDDY_TRACE_OPS
#define BUDDY_TRACE_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def BuddyTrace_Dialect : Dialect {
  let name = "buddy_trace";
  let cppNamespace = "::buddy::trace";
  let summary = "Buddy trace instrumentation dialect";
  let description = [{
    The `buddy_trace` dialect represents explicit profiling and debugging
    observation points. Backend-specific passes lower these ops to runtime
    calls, target intrinsics, or device trace instructions.
  }];
}

class BuddyTrace_Op<string mnemonic, list<Trait> traits = []> :
    Op<BuddyTrace_Dialect, mnemonic, traits>;

def BuddyTrace_StartOp : BuddyTrace_Op<"start"> {
  let summary = "Start a traced region";
  let arguments = (ins I64Attr:$id, OptionalAttr<StrAttr>:$tag);
  let assemblyFormat = "attr-dict";
}

def BuddyTrace_EndOp : BuddyTrace_Op<"end", [SameOperandsAndResultType]> {
  let summary = "End a traced region and forward the traced value";
  let arguments = (ins AnyType:$input, I64Attr:$id,
                       OptionalAttr<StrAttr>:$tag);
  let results = (outs AnyType:$output);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

#endif // BUDDY_TRACE_OPS
