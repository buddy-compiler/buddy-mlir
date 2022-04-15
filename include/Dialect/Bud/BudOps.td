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
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Test constant operation.
//===----------------------------------------------------------------------===//

def Bud_TestConstantOp : Bud_Op<"test_constant", []> {
  let summary = "Test Constant Operation.";

  let results = (outs AnyInteger:$result);

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Test print operation.
//===----------------------------------------------------------------------===//

def Bud_TestPrintOp : Bud_Op<"test_print", []> {
  let summary = "Test Print Operation.";

  let results = (outs AnyInteger:$result);

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Test enum attribute in the operation.
//===----------------------------------------------------------------------===//

def Bud_TestEnumAttrOpAdd : I32EnumAttrCase<"ADD", 0, "add">;
def Bud_TestEnumAttrOpSub : I32EnumAttrCase<"SUB", 1, "sub">;

def Bud_TestEnumAttrOperation : I32EnumAttr<"TestEnumAttrOperation",
    "The arithmetic enum attribute.",
    [
      Bud_TestEnumAttrOpAdd,
      Bud_TestEnumAttrOpSub
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::buddy::bud";
}

def Bud_TestEnumAttrOperationAttr : EnumAttr<Bud_Dialect, Bud_TestEnumAttrOperation, "test_enum_attr_op">;

def Bud_TestEnumAttrOp : Bud_Op<"test_enum_attr",
    [SameOperandsAndResultType]>,
    Arguments<(ins AnyInteger:$lhs, AnyInteger:$rhs,
               Bud_TestEnumAttrOperationAttr:$arith)>,
    Results<(outs AnyInteger:$result)> {
  let summary = "Test Enum Attribute.";

  let assemblyFormat = [{
    $arith $lhs `,` $rhs attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Test array attribute in the operation.
//===----------------------------------------------------------------------===//

def CoordinateArrayAttr : Confined<I64ArrayAttr, [ArrayCount<2>]>;

def Bud_TestArrayAttrOp : Bud_Op<"test_array_attr"> {
  let summary = "Test Array Attribute.";

  let arguments = (ins 
        Arg<AnyMemRef, "the reference to load from", [MemRead]>:$base,
        CoordinateArrayAttr:$coordinate);
  let results = (outs AnyType:$result);

  let assemblyFormat =
      "$base attr-dict `:` type($base) `,` type($result)";
}

#endif // BUD_BUDOPS_TD
