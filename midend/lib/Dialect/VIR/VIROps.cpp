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

#include "VIR/VIRDialect.h"
#include "VIR/VIROps.h"

#define GET_OP_CLASSES
#include "VIR/VIR.cpp.inc"

static ::mlir::LogicalResult
verifyIntegerExtOp(::mlir::Operation *op,
                   buddy::vir::DynamicVectorType inputType,
                   buddy::vir::DynamicVectorType resultType) {
  auto inputInt =
      ::mlir::dyn_cast<::mlir::IntegerType>(inputType.getElementType());
  auto resultInt =
      ::mlir::dyn_cast<::mlir::IntegerType>(resultType.getElementType());
  if (!inputInt)
    return op->emitOpError("input element type must be integer");
  if (!resultInt)
    return op->emitOpError("result element type must be integer");

  if (inputType.getShape() != resultType.getShape())
    return op->emitOpError("input and result shapes must match");
  if (inputType.getScalingFactor() != resultType.getScalingFactor())
    return op->emitOpError("input and result scaling factors must match");
  if (resultInt.getWidth() <= inputInt.getWidth())
    return op->emitOpError(
        "result integer width must be greater than input integer width");

  return ::mlir::success();
}

::mlir::LogicalResult buddy::vir::ScatterOp::verify() {
  auto valueType = getValue().getType();
  auto indexVecType = getIndexVec().getType();
  auto maskType = getMask().getType();
  ::mlir::Type indexElementType = indexVecType.getElementType();

  if (!::llvm::isa<::mlir::IntegerType, ::mlir::IndexType>(indexElementType))
    return emitOpError(
        "scatter index vector element type must be integer or index");

  if (valueType.getShape() != indexVecType.getShape())
    return emitOpError("scatter value and index vector shapes must match");
  if (valueType.getShape() != maskType.getShape())
    return emitOpError("scatter value and mask vector shapes must match");

  if (valueType.getScalingFactor() != indexVecType.getScalingFactor())
    return emitOpError(
        "scatter value and index vector scaling factors must match");
  if (valueType.getScalingFactor() != maskType.getScalingFactor())
    return emitOpError(
        "scatter value and mask vector scaling factors must match");
  if (!maskType.getElementType().isInteger(1))
    return emitOpError("scatter mask vector element type must be i1");

  auto baseType = ::mlir::cast<::mlir::ShapedType>(getBase().getType());
  if (valueType.getElementType() != baseType.getElementType())
    return emitOpError(
        "scatter value element type must match memref element type");

  return ::mlir::success();
}

::mlir::LogicalResult buddy::vir::ExtSIOp::verify() {
  return verifyIntegerExtOp(getOperation(), getIn().getType(),
                            getResult().getType());
}

::mlir::LogicalResult buddy::vir::ExtUIOp::verify() {
  return verifyIntegerExtOp(getOperation(), getIn().getType(),
                            getResult().getType());
}
