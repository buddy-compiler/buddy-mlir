//====- Utils.cpp ---------------------------------------------------------===//
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
// This file implements generic utility functions for the buddy compiler
// ecosystem.
//
//===----------------------------------------------------------------------===//

#ifndef UTILS_UTILS_DEF
#define UTILS_UTILS_DEF

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/IR/AffineValueMap.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>

using namespace mlir;

namespace buddy {

// Function to test whether a value is equivalent to zero or not.
Value zeroCond(OpBuilder &builder, Location loc, Type elemType, Value value,
               Value zeroElem) {
  Value cond;
  auto bitWidth = elemType.getIntOrFloatBitWidth();
  if (elemType.isF32() || elemType.isF64()) {
    cond = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ONE, value,
                                         zeroElem);
  } else if (elemType.isInteger(bitWidth)) {
    cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, value,
                                         zeroElem);
  }
  return cond;
}

// Create an inverted mask having all 1's shifted to right side.
Value createInvertedMask(OpBuilder &builder, Location loc, Value strideVal,
                         VectorType vectorMaskTy, Value leftIndex) {
  Value leftMask =
      builder.create<vector::CreateMaskOp>(loc, vectorMaskTy, leftIndex);
  Value maskInverter =
      builder.create<vector::CreateMaskOp>(loc, vectorMaskTy, strideVal);
  Value rightMask = builder.create<arith::SubIOp>(loc, maskInverter, leftMask);
  return rightMask;
}

// Cast a value from index type to f32 type.
Value indexToF32(OpBuilder &builder, Location loc, Value val) {
  Value interm1 =
      builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), val);
  return builder.create<arith::SIToFPOp>(loc, builder.getF32Type(), interm1);
}

// Cast a value from f32 type to index type.
Value F32ToIndex(OpBuilder &builder, Location loc, Value val) {
  Value interm1 =
      builder.create<arith::FPToUIOp>(loc, builder.getI32Type(), val);
  return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                            interm1);
}

// Round off floating point value to nearest integer type value.
Value roundOff(OpBuilder &builder, Location loc, Value val) {
  Value ceilVal = builder.create<math::CeilOp>(loc, val);
  Value floorVal = builder.create<math::FloorOp>(loc, val);

  Value diffCeil = builder.create<arith::SubFOp>(loc, ceilVal, val);
  Value diffFloor = builder.create<arith::SubFOp>(loc, val, floorVal);

  Value diffCond = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT,
                                                 diffCeil, diffFloor);

  return builder.create<arith::SelectOp>(loc, diffCond, floorVal, ceilVal);
}

// Bound values to permissible range of allocatable values w.r.t output image.
Value valBound(OpBuilder &builder, Location loc, Value val, Value lastElemF32,
               Value c0F32) {
  Value interm1 = builder.create<arith::MaxFOp>(loc, val, c0F32);
  return builder.create<arith::MinFOp>(loc, interm1, lastElemF32);
}

// Equivalent of std::iota.
Value iotaVec(OpBuilder &builder, Location loc, MLIRContext *ctx,
              Value indexStart, Value strideVal, VectorType vecType, Value c0,
              int64_t stride) {
  // ToDo : Try to get rid of memref load/store, find less expensive ways of
  // implementing this function.
  MemRefType memTy = MemRefType::get({stride}, builder.getF32Type());
  Value tempMem = builder.create<memref::AllocOp>(loc, memTy);

  builder.create<AffineForOp>(
      loc, ValueRange{c0}, builder.getDimIdentityMap(), ValueRange{strideVal},
      builder.getDimIdentityMap(), 1, llvm::None,
      [&](OpBuilder &builder, Location loc, Value iv, ValueRange iterArg) {
        Value iotaValIndex = builder.create<arith::AddIOp>(loc, iv, indexStart);
        Value iotaVal = indexToF32(builder, loc, iotaValIndex);

        builder.create<memref::StoreOp>(loc, iotaVal, tempMem, ValueRange{iv});
        builder.create<AffineYieldOp>(loc);
      });

  return builder.create<vector::LoadOp>(loc, vecType, tempMem, ValueRange{c0});
}

// Cast index type value to f32 type and then expand it in a vector.
Value castAndExpand(OpBuilder &builder, Location loc, Value val,
                    VectorType vecType) {
  Value interm1 = indexToF32(builder, loc, val);
  return builder.create<vector::SplatOp>(loc, vecType, interm1);
}

} // namespace buddy

#endif // UTILS_UTILS_DEF
