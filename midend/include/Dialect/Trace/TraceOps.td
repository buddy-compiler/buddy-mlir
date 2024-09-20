//===- TraceOps.td - Trace Dialect Ops -------------------------*- tablegen -*-===//
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
  let results = (outs F64:$result); // Returns a f64 type representing the start time.
  let assemblyFormat = "attr-dict `:` `->` type($result) "; // Simplified format, showing only result type.
}

//===----------------------------------------------------------------------===//
// End timing operation.
//===----------------------------------------------------------------------===//
def Trace_TimeEndOp : Trace_Op<"time_end", []> {
  let summary = "End trace an operation";
  let description = [{
    This operation marks the end of a trace analysis.
  }];
  let results = (outs F64:$result); // Returns a f64 type representing the start time.
  let assemblyFormat = "attr-dict `:` `->` type($result) "; // Simplified format, showing only result type.
}



#endif // TRACE_TRACEOPS_TD
