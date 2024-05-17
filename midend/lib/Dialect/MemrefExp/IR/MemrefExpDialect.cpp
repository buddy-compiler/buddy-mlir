//===- MemrefExpDialect.cpp - Memref Experiment Dialect Definition --------===//
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
// This file defines memref experiment dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#include "MemrefExp/MemrefExpDialect.h"
#include "MemrefExp/MemrefExpOps.h"

using namespace mlir;
using namespace buddy::memref_exp;

#include "MemrefExp/MemrefExpOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "MemrefExp/MemrefExpOps.cpp.inc"

void MemrefExpDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MemrefExp/MemrefExpOps.cpp.inc"
      >();
}
