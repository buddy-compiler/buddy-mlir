//===- TraceOps.td - Trace Dialect Ops ---------------------*- tablegen -*-===//
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
// This is the top level file for operations in the trace dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TRACE_TRACEOPS_TD
#define TRACE_TRACEOPS_TD

include "TraceDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"

//===----------------------------------------------------------------------===//
// Start timing operation.
//===----------------------------------------------------------------------===//
def Trace_TimeStartOp : Trace_Op<"time_start", []> {
  let summary = "Start trace an operation";
  let description = [{
    This operation marks the beginning of a trace analysis.
  }];
  // let results = (outs F64:$result); // Returns a f64 type representing the start time.
  // let assemblyFormat = "attr-dict `:` `->` type($result) "; // Simplified format, showing only result type.
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// End timing operation.
//===----------------------------------------------------------------------===//
def Trace_TimeEndOp : Trace_Op<"time_end", []> {
  let summary = "End trace an operation";
  let description = [{
    This operation marks the end of a trace analysis.
  }];
  // let results = (outs F64:$result); // Returns a f64 type representing the start time.
  // let assemblyFormat = "attr-dict `:` `->` type($result) "; // Simplified format, showing only result type.
  let assemblyFormat = "attr-dict";
}


//===----------------------------------------------------------------------===//
// Timing_scope OP
//===----------------------------------------------------------------------===//
def Trace_ScopeOp : Trace_Op<"scope", []> {
  let summary = "An operation that introduces a timing scope";
  let description = [{
    `scope` is an operation that introduces a new timing scope. 
    It contains a single region which holds the operations within this scope.
  }];

  let regions = (region AnyRegion:$region);

  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    attr-dict `->` `(` type($results) `)` $region
  }];

  // let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
  
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def Trace_YieldOp : Trace_Op<"yield", [Pure, Terminator, HasParent<"ScopeOp">]> {
  let summary = "timing scope yield and termination operation";
  let description = [{
    "scf.yield" yields an SSA value from the SCF dialect op region and
    terminates the regions. The semantics of how the values are yielded is
    defined by the parent operation.
    If "scf.yield" has any operands, the operands must match the parent
    operation's results.
    If the parent operation defines no values, then the "scf.yield" may be
    left out in the custom syntax and the builders will insert one implicitly.
    Otherwise, it has to be present in the syntax to indicate which values are
    yielded.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{  attr-dict ($results^ `:` type($results))? }];
}

#endif // TRACE_TRACEOPS_TD
