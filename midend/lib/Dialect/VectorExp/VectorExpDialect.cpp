//===- VectorExpDialect.cpp - Vector Experiment Dialect Definition --------===//
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
// This file defines vector experiment dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/SourceMgr.h"

#include "VectorExp/VectorExpDialect.h"
#include "VectorExp/VectorExpOps.h"

using namespace mlir;
using namespace buddy::vector_exp;

#include "VectorExp/VectorExpOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Vector Experiment Dialect.
//===----------------------------------------------------------------------===//

void VectorExpDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "VectorExp/VectorExpOps.cpp.inc"
      >();
}
