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

include "Sche/ScheDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

// include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// On device operation.
//===----------------------------------------------------------------------===//

def Sche_OnDeviceOp : Sche_Op<"on_device", [Sche_AsyncOpInterface, AttrSizedOperandSegments, AttrSizedResultSegments]> {
  let summary = "Mark the region of operations running on target device.";

  let arguments = (ins Variadic<Sche_AsyncToken>:$asyncDependencies, StrAttr: $targetId, StrAttr: $targetConfig, Variadic<AnyType>: $innerOperands);
  let results = (outs Optional<Sche_AsyncToken>:$asyncToken, Variadic<AnyType>:$innerResults);
  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<(ins "llvm::StringRef":$targetId, "llvm::StringRef":$targetConfig, "mlir:: TypeRange": $innerResultTypes,
                 "mlir::ValueRange":$innerOperands, CArg<"mlir::function_ref<void(mlir::OpBuilder &, mlir::Location, mlir::ValueRange)>",
                  "nullptr">:$bodyBuilder, CArg<"mlir::Type", "nullptr">:$asyncTokenType, CArg<"mlir::ValueRange", "mlir::ValueRange{}">:$asyncDependencies)>
  ];

  let hasCustomAssemblyFormat = 1;
   let skipDefaultBuilders = 1;
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

def Sche_ReturnOp : Sche_Op<"return", [ReturnLike, Terminator]>{
  let summary = "Terminator in body of sche.on_device op";

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def Sche_WaitOp : Sche_Op<"wait", [Sche_AsyncOpInterface]> {
  let summary = "Wait for async gpu ops to complete.";
  
  let arguments = (ins Variadic<Sche_AsyncToken>:$asyncDependencies);
  let results = (outs Optional<Sche_AsyncToken>:$asyncToken);

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies) attr-dict
  }];

}

#endif // SCHE_SCHEOPS_TD
