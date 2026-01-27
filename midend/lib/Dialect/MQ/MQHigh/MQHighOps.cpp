//===- MQHighOps.cpp - MQHigh dialect operations implementation ------------*- C++ -*-===//
//
// This file implements the operations for the MQHigh dialect.
//
//===----------------------------------------------------------------------===//

#include "MQ/MQHigh/MQHighOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"

#define GET_OP_IMPLEMENTATION
#include "MQ/MQHigh/MQHighOps.cpp.inc"

using namespace buddy::mqhigh;

// //===----------------------------------------------------------------------===//
// // MQHigh_MatMulOp implementation
// //===----------------------------------------------------------------------===//

// mlir::LogicalResult MQHighMatMulOp::verify() {
//   // Verify that the input tensors have valid shapes for matrix multiplication.
// //   auto AType = getAType().dyn_cast<mlir::RankedTensorType>();
// //   auto BType = getBType().dyn_cast<mlir::RankedTensorType>();
// //   if (!AType || !BType) {
// //     return emitError("MatMul operands must be ranked tensors");
// //   }

// //   // Check that the last dimension of A matches the second-to-last dimension of B.
// //   if (AType.getShape().back() != BType.getShape()[BType.getRank() - 2]) {
// //     return emitError("MatMul operand dimensions mismatch");
// //   }

//   return mlir::success();
// }

// //===----------------------------------------------------------------------===//
// // MQHigh_PackOp implementation
// //===----------------------------------------------------------------------===//

// mlir::LogicalResult MQHigh_PackOp::verify() {
//   // Verify that the input tensor has a valid shape for packing.
//   auto inputType = getInputType().dyn_cast<mlir::RankedTensorType>();
//   if (!inputType) {
//     return emitError("Pack operand must be a ranked tensor");
//   }

//   return mlir::success();
// }

// //===----------------------------------------------------------------------===//
// // MQHigh_UnpackOp implementation
// //===----------------------------------------------------------------------===//

// mlir::LogicalResult MQHigh_UnpackOp::verify() {
//   // Verify that the input tensor has a valid shape for unpacking.
//   auto inputType = getInputType().dyn_cast<mlir::RankedTensorType>();
//   if (!inputType) {
//     return emitError("Unpack operand must be a ranked tensor");
//   }

//   return mlir::success();
// }
