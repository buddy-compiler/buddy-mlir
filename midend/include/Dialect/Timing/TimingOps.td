//===- TimingOps.td - Timing Dialect Ops -------------------------*- tablegen -*-===//
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
// This is the top level file for operations in the timing dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TIMING_TIMINGOPS_TD
#define TIMING_TIMINGOPS_TD

include "TimingDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"

//===----------------------------------------------------------------------===//
// Test constant operation.
//===----------------------------------------------------------------------===//

def Timing_TestConstantOp : Timing_Op<"test_constant", []> {
  let summary = "Test Constant Operation.";

  let results = (outs AnyInteger:$result);

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Test print operation.
//===----------------------------------------------------------------------===//

def Timing_TestPrintOp : Timing_Op<"test_print", []> {
  let summary = "Test Print Operation.";

  let results = (outs AnyInteger:$result);

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Test enum attribute in the operation.
//===----------------------------------------------------------------------===//

def Timing_TestEnumAttrOpAdd : I32EnumAttrCase<"ADD", 0, "add">;
def Timing_TestEnumAttrOpSub : I32EnumAttrCase<"SUB", 1, "sub">;

def Timing_TestEnumAttrOperation : I32EnumAttr<"TestEnumAttrOperation",
    "The arithmetic enum attribute.",
    [
      Timing_TestEnumAttrOpAdd,
      Timing_TestEnumAttrOpSub
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::buddy::timing";
}

def Timing_TestEnumAttrOperationAttr : EnumAttr<Timing_Dialect, Timing_TestEnumAttrOperation, "test_enum_attr_op">;

def Timing_TestEnumAttrOp : Timing_Op<"test_enum_attr",
    [SameOperandsAndResultType]>,
    Arguments<(ins AnyInteger:$lhs, AnyInteger:$rhs,
               Timing_TestEnumAttrOperationAttr:$arith)>,
    Results<(outs AnyInteger:$result)> {
  let summary = "Test Enum Attribute.";

  let assemblyFormat = [{
    $arith $lhs `,` $rhs attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Test array attribute in the operation.
//===----------------------------------------------------------------------===//

def CoordinateArrayAttr : ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>;

def Timing_TestArrayAttrOp : Timing_Op<"test_array_attr"> {
  let summary = "Test Array Attribute.";

  let arguments = (ins 
        Arg<AnyMemRef, "the reference to load from", [MemRead]>:$base,
        CoordinateArrayAttr:$coordinate);
  let results = (outs AnyType:$result);

  let assemblyFormat =
      "$base attr-dict `:` type($base) `,` type($result)";
}

//===----------------------------------------------------------------------===//
// Start counting operation.
//===----------------------------------------------------------------------===//
def Timing_StartOp : Timing_Op<"start", []> {
  let summary = "Start timing an operation";
  let description = [{
    This operation marks the beginning of a timing analysis.
  }];
  let results = (outs F64:$result); // Returns a f64 type representing the start time.
  let assemblyFormat = "attr-dict `:` `->` type($result) "; // Simplified format, showing only result type.
}

//===----------------------------------------------------------------------===//
// End counting operation.
//===----------------------------------------------------------------------===//
def Timing_EndOp : Timing_Op<"end", []> {
  let summary = "End timing an operation";
  let description = [{
    This operation marks the end of a timing analysis.
  }];
  let results = (outs F64:$result); // Returns a f64 type representing the start time.
  let assemblyFormat = "attr-dict `:` `->` type($result) "; // Simplified format, showing only result type.
}


//===----------------------------------------------------------------------===//
// Create timing vector operation.
//===----------------------------------------------------------------------===//
def TimingVectorCreateOp : Timing_Op<"vector_create", []> {
  let summary = "Creates a vector for timing storage.";
  let description = [{
    This operation creates a vector with specified size to store timing data.
  }];
  let arguments = (ins I64:$size);
  let results = (outs Builtin_Vector:$vector);
}

//===----------------------------------------------------------------------===//
// Store time operation.
//===----------------------------------------------------------------------===//
def TimingVectorStoreOp : Timing_Op<"vector_store", []> {
  let summary = "Stores a timing value in a vector.";
  let description = [{
    This operation stores a timing value at a specified index in the vector.
  }];
  let arguments = (ins Builtin_Vector:$vector, F64:$value, I64:$index);
}


#endif // TIMING_TIMINGOPS_TD
