//===- VectorExpDialect.td - Vector Experiment Dialect Definition ---------===//
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
// This is the top level file for the vector experiment dialect.
//
//===----------------------------------------------------------------------===//

#ifndef VECTOREXP_VECTOREXPDIALECT_TD
#define VECTOREXP_VECTOREXPDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Vector Experiment Dialect Definition.
//===----------------------------------------------------------------------===//

def VectorExp_Dialect : Dialect {
  let name = "vector_exp";
  let summary = "The Vector Experiment Dialect.";
  let description = [{
    The `vector_exp` dialect is for testing and demonstrating ideas of vector
    dialect.
  }];
  let cppNamespace = "::buddy::vector_exp";
}

//===----------------------------------------------------------------------===//
// Base Vector Experiment Operation Definition.
//===----------------------------------------------------------------------===//

class VectorExp_Op<string mnemonic, list<Trait> traits = []> :
    Op<VectorExp_Dialect, mnemonic, traits>;

#endif // VECTOREXP_VECTOREXPDIALECT_TD
