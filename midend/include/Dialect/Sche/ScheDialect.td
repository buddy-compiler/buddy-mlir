//===- ScheDialect.td - Sche Dialect Definition --------------*- tablegen -*-===//
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
// This is the top level file for the sche dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SCHE_SCHEDIALECT_TD
#define SCHE_SCHEDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Sche Dialect Definition.
//===----------------------------------------------------------------------===//

def Sche_Dialect : Dialect {
  let name = "sche";
  let summary = "The Sche Dialect.";
  let description = [{
    The `sche` dialect is for scheduling operators on heterogeneous devices.
  }];
  let cppNamespace = "::buddy::sche";
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Sche Interfaces.
//===----------------------------------------------------------------------===//

def Sche_AsyncOpInterface : OpInterface<"AsyncOpInterface"> {
  let cppNamespace = "::buddy::sche";

  let methods = [
    InterfaceMethod<[{
        Query the operands that represent async dependency tokens.
      }],
      "::mlir::OperandRange", "getAsyncDependencies", (ins), [{}], [{
        ConcreteOp op = ::llvm::cast<ConcreteOp>(this->getOperation());
        return op.getAsyncDependencies();
      }]
    >,
    InterfaceMethod<[{
        Adds a new token to the list of async dependencies if it is not already there.
      }],
      "void", "addAsyncDependency", (ins "::mlir::Value":$token),
      [{}], [{
        if (!::llvm::is_contained(this->getAsyncDependencies(), token))
          ::buddy::sche::addAsyncDependency(this->getOperation(), token);
      }]
    >,
    InterfaceMethod<[{
        Query the result that represents the async token to depend on.
      }],
      "::mlir::Value", "getAsyncToken"
    >
  ];
}

//===----------------------------------------------------------------------===//
// Sche Types.
//===----------------------------------------------------------------------===//

def Sche_AsyncToken : DialectType<
  Sche_Dialect, CPred<"isa<::buddy::sche::AsyncTokenType>($_self)">, "async token type">,
             BuildableType<"buddy::sche::AsyncTokenType::get($_builder.getContext())">;

//===----------------------------------------------------------------------===//
// Base Sche Operation Definition.
//===----------------------------------------------------------------------===//

class Sche_Op<string mnemonic, list<Trait> traits = []> :
    Op<Sche_Dialect, mnemonic, traits>;

#endif // SCHE_SCHEDIALECT_TD
