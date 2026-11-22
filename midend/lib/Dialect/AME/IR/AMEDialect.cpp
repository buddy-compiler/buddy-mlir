//====- AMEDialect.cpp - MLIR AME dialect implementation ------------------===//
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

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "Dialect/AME/AMEDialect.h"
#include "Dialect/AME/AMEOps.h"

using namespace mlir;
using namespace buddy::ame;

#include "AME/AMEDialect.cpp.inc"

#define GET_OP_CLASSES
#include "AME/AME.cpp.inc"

void AMEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "AME/AME.cpp.inc"
      >();
}

/// LLVM returns zip/unzip pairs as a literal struct. The intrinsic overload
/// requires both fields and both sources to have exactly the same matrix type.
static LogicalResult verifyMatrixPair(Operation *op) {
  auto result = llvm::cast<LLVM::LLVMStructType>(op->getResult(0).getType());
  if (result.isIdentified() || result.isOpaque() || result.isPacked() ||
      result.getBody().size() != 2 ||
      result.getBody()[0] != op->getOperand(0).getType() ||
      result.getBody()[1] != op->getOperand(0).getType() ||
      op->getOperand(1).getType() != op->getOperand(0).getType())
    return op->emitOpError("requires a literal, unpacked two-matrix result "
                           "matching both operands");
  return success();
}

LogicalResult McolunzipEwOp::verify() { return verifyMatrixPair(*this); }
LogicalResult McolzipEwOp::verify() { return verifyMatrixPair(*this); }
LogicalResult MrowunzipEwOp::verify() { return verifyMatrixPair(*this); }
LogicalResult MrowzipEwOp::verify() { return verifyMatrixPair(*this); }
