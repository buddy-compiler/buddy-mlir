//===-- VIROps.cpp - Dynamic Vector IR Operation Implementation -----------===//
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
// This file implements the operation set for the Dynamic Vector IR (VIR)
// dialect, including vector length management and dynamic vector operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#include "VIR/VIRDialect.h"
#include "VIR/VIROps.h"

#define GET_OP_CLASSES
#include "VIR/VIR.cpp.inc"

using namespace mlir;
using namespace buddy::vir;

LogicalResult SetVLOp::verify() {
  if (!getRegion().empty() && getRegion().front().getNumArguments() != 0)
    return emitOpError("region must not have block arguments");
  return success();
}

// One index per memref dim or none, and a dim of the memref per dim of the
// dynamic vector.
static LogicalResult verifyAccess(Operation *op, MemRefType memref,
                                  ValueRange indices, DynamicVectorType vec) {
  if (!indices.empty() && int64_t(indices.size()) != memref.getRank())
    return op->emitOpError("needs no index or one per memref dim, ")
           << memref.getRank();
  if (int64_t(vec.getShape().size()) > memref.getRank())
    return op->emitOpError("needs a memref of at least the rank of the "
                           "dynamic vector, ")
           << vec.getShape().size();
  return success();
}

LogicalResult LoadOp::verify() {
  return verifyAccess(*this, getBase().getType(), getIndices(), getType());
}

LogicalResult StoreOp::verify() {
  return verifyAccess(*this, getBase().getType(), getIndices(),
                      getValue().getType());
}

LogicalResult ShuffleOp::verify() {
  ArrayRef<int64_t> shape = getV1().getType().getShape();
  if (shape.size() < 2)
    return emitOpError("needs a dynamic vector with a leading static dim");
  int64_t rows = shape.front();
  if (int64_t(getMask().size()) != rows)
    return emitOpError("needs one mask index per row, ") << rows;
  for (int64_t index : getMask())
    if (index < 0 || index >= 2 * rows)
      return emitOpError("mask index ")
             << index << " is out of the " << 2 * rows << " rows";
  return success();
}
