//===- ScheOps.td - Sche Dialect Ops -------------------------*- tablegen -*-===//
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
// This is the top level file for operations in the sche dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SCHE_SCHEOPS_TD
#define SCHE_SCHEOPS_TD

include "ScheDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
// include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// On device operation.
//===----------------------------------------------------------------------===//

def Sche_OnDeviceOp : Sche_Op<"on_device", []> {
  let summary = "Mark the region of operations running on target device.";

  let arguments = (ins StrAttr: $targetId, StrAttr: $targetConfig, Variadic<AnyType>);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  // let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "llvm::StringRef":$targetId, "llvm::StringRef":$targetConfig, "mlir:: TypeRange": $resultTypes,
                 "mlir::ValueRange":$args, CArg<"mlir::function_ref<void(mlir::OpBuilder &, mlir::Location, mlir::ValueRange)>",
                  "nullptr">:$bodyBuilder)>
  ];

  let assemblyFormat = [{
    `(`operands`)` attr-dict functional-type(operands, results) $region
  }];
}

//===----------------------------------------------------------------------===//
// Launch function operation.
//===----------------------------------------------------------------------===//

def Sche_LaunchFuncOp : Sche_Op<"launch_func"> {
  let summary = "Launch function on device.";

  let arguments = (ins SymbolRefAttr:$func, Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $func `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

//===----------------------------------------------------------------------===//
// Test enum attribute in the operation.
//===----------------------------------------------------------------------===//

// def Bud_TestEnumAttrOpAdd : I32EnumAttrCase<"ADD", 0, "add">;
// def Bud_TestEnumAttrOpSub : I32EnumAttrCase<"SUB", 1, "sub">;

// def Bud_TestEnumAttrOperation : I32EnumAttr<"TestEnumAttrOperation",
//     "The arithmetic enum attribute.",
//     [
//       Bud_TestEnumAttrOpAdd,
//       Bud_TestEnumAttrOpSub
//     ]>{
//   let genSpecializedAttr = 0;
//   let cppNamespace = "::buddy::bud";
// }

// def Bud_TestEnumAttrOperationAttr : EnumAttr<Bud_Dialect, Bud_TestEnumAttrOperation, "test_enum_attr_op">;

// def Bud_TestEnumAttrOp : Bud_Op<"test_enum_attr",
//     [SameOperandsAndResultType]>,
//     Arguments<(ins AnyInteger:$lhs, AnyInteger:$rhs,
//                Bud_TestEnumAttrOperationAttr:$arith)>,
//     Results<(outs AnyInteger:$result)> {
//   let summary = "Test Enum Attribute.";

//   let assemblyFormat = [{
//     $arith $lhs `,` $rhs attr-dict `:` type($result)
//   }];
// }

//===----------------------------------------------------------------------===//
// Test array attribute in the operation.
//===----------------------------------------------------------------------===//

// def CoordinateArrayAttr : ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>;

// def Bud_TestArrayAttrOp : Bud_Op<"test_array_attr"> {
//   let summary = "Test Array Attribute.";

//   let arguments = (ins 
//         Arg<AnyMemRef, "the reference to load from", [MemRead]>:$base,
//         CoordinateArrayAttr:$coordinate);
//   let results = (outs AnyType:$result);

//   let assemblyFormat =
//       "$base attr-dict `:` type($base) `,` type($result)";
// }

#endif // SCHE_SCHEOPS_TD
