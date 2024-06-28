//===- MemrefExpDialect.td - Memref Experiment Dialect Definition ---------===//
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
// This is the top level file for the memref experiment dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MEMREFEXP_MEMREFEXPDIALECT_TD
#define MEMREFEXP_MEMREFEXPDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Memref Experiment Dialect Definition.
//===----------------------------------------------------------------------===//

def MemrefExp_Dialect : Dialect {
  let name = "memref_exp";
  let summary = "The Memref Experiment Dialect.";
  let cppNamespace = "::buddy::memref_exp";
  let description = [{
    The `memref_exp` dialect is for testing and demonstrating ideas of memref
    dialect.
  }];
}

//===----------------------------------------------------------------------===//
// Base Memref Experiment Operation Definition.
//===----------------------------------------------------------------------===//

class MemrefExp_Op<string mnemonic, list<Trait> traits = []> :
    Op<MemrefExp_Dialect, mnemonic, traits>;

#endif // MEMREFEXP_MEMREFEXPDIALECT_TD
