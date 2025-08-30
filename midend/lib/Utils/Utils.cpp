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

#include "mlir/IR/BuiltinTypes.h"
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/IR/AffineValueMap.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>

#include <initializer_list>
#include <numeric>

using namespace mlir;

namespace buddy {

// Inserts a constant op with value 0 into a location `loc` based on type
// `type`. Supported types are : f32, f64, integer types.
Value insertZeroConstantOp(MLIRContext *ctx, OpBuilder &builder, Location loc,
                           Type elemTy) {
  Value op = {};
  auto bitWidth = elemTy.getIntOrFloatBitWidth();
  if (elemTy.isF32() || elemTy.isF64()) {
    FloatType floatType = elemTy.isF32()
                              ? static_cast<FloatType>(Float32Type::get(ctx))
                              : static_cast<FloatType>(Float64Type::get(ctx));
    auto zero = APFloat::getZero(floatType.getFloatSemantics());
    op = builder.create<arith::ConstantFloatOp>(loc, zero, floatType);
  } else if (elemTy.isInteger(bitWidth)) {
    IntegerType type = IntegerType::get(ctx, bitWidth);
    op = builder.create<arith::ConstantIntOp>(loc, 0, type);
  }

  return op;
}

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
  Value interm1 = builder.create<arith::MaximumFOp>(loc, val, c0F32);
  return builder.create<arith::MinimumFOp>(loc, interm1, lastElemF32);
}

// check if lb <= val < ub and returns Value 0 or 1
Value inBound(OpBuilder &builder, Location loc, Value val, Value lb, Value ub) {
  Value greaterThanLb =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, lb, val);
  Value lowerThanUb =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, val, ub);
  return builder.create<arith::AndIOp>(loc, greaterThanLb, lowerThanUb);
}

// Equivalent of std::iota.
Value iotaVec(OpBuilder &builder, Location loc, MLIRContext *ctx,
              Value indexStart, Value strideVal, VectorType vecType, Value c0,
              int64_t stride) {
  // ToDo : Try to get rid of memref load/store, find less expensive ways of
  // implementing this function.
  MemRefType memTy = MemRefType::get({stride}, builder.getF32Type());
  Value tempMem = builder.create<memref::AllocOp>(loc, memTy);

  builder.create<affine::AffineForOp>(
      loc, ValueRange{c0}, builder.getDimIdentityMap(), ValueRange{strideVal},
      builder.getDimIdentityMap(), 1, std::nullopt,
      [&](OpBuilder &builder, Location loc, Value iv, ValueRange iterArg) {
        Value iotaValIndex = builder.create<arith::AddIOp>(loc, iv, indexStart);
        Value iotaVal = indexToF32(builder, loc, iotaValIndex);

        builder.create<memref::StoreOp>(loc, iotaVal, tempMem, ValueRange{iv});
        builder.create<affine::AffineYieldOp>(loc);
      });

  return builder.create<vector::LoadOp>(loc, vecType, tempMem, ValueRange{c0});
}

// Generate vector[0, 1, ..., length - 1] with f32 type
Value iotaVec0F32(OpBuilder &builder, Location loc, int64_t length) {
  MLIRContext *ctx = builder.getContext();
  std::vector<float> vec(length);
  std::iota(vec.begin(), vec.end(), .0);
  Value res = builder.create<arith::ConstantOp>(
      loc,
      DenseFPElementsAttr::get(VectorType::get(length, Float32Type::get(ctx)),
                               ArrayRef<float>(vec)));
  return res;
}

// Cast index type value to f32 type and then expand it in a vector.
Value castAndExpand(OpBuilder &builder, Location loc, Value val,
                    VectorType vecType) {
  Value interm1 = indexToF32(builder, loc, val);
  return builder.create<vector::SplatOp>(loc, vecType, interm1);
}

// print values(for debug use)
void printValues(OpBuilder &builder, Location loc,
                 std::initializer_list<Value> values) {
  if (empty(values))
    return;
  Type valueTy = values.begin()->getType();
  VectorType vecTy = VectorType::get({(long)values.size()}, valueTy);
  Value vec = builder.create<vector::SplatOp>(loc, vecTy, *values.begin());
  int idx = 0;
  for (auto value : values) {
    if (idx != 0) {
      // all values should have same type
      assert(value.getType() == valueTy);
      Value idxVal = builder.create<arith::ConstantIndexOp>(loc, idx);
      vec = builder.create<vector::InsertElementOp>(loc, value, vec, idxVal);
    }
    idx++;
  }
  builder.create<vector::PrintOp>(loc, vec);
}

// Function for calculating complex addition of 2 input 1D complex vectors.
// Separate vectors for real and imaginary parts are expected.
inline std::vector<Value> complexVecAddI(OpBuilder &builder, Location loc,
                                         Value vec1Real, Value vec1Imag,
                                         Value vec2Real, Value vec2Imag) {
  return {builder.create<arith::AddFOp>(loc, vec1Real, vec2Real),
          builder.create<arith::AddFOp>(loc, vec1Imag, vec2Imag)};
}

// Function for calculating complex subtraction of 2 input 1D complex vectors.
// Separate vectors for real and imaginary parts are expected.
inline std::vector<Value> complexVecSubI(OpBuilder &builder, Location loc,
                                         Value vec1Real, Value vec1Imag,
                                         Value vec2Real, Value vec2Imag) {
  return {builder.create<arith::SubFOp>(loc, vec1Real, vec2Real),
          builder.create<arith::SubFOp>(loc, vec1Imag, vec2Imag)};
}

// Function for calculating complex product of 2 input 1D complex vectors.
// Separate vectors for real and imaginary parts are expected.
std::vector<Value> complexVecMulI(OpBuilder &builder, Location loc,
                                  Value vec1Real, Value vec1Imag,
                                  Value vec2Real, Value vec2Imag) {
  Value int1 = builder.create<arith::MulFOp>(loc, vec1Real, vec2Real);
  Value int2 = builder.create<arith::MulFOp>(loc, vec1Imag, vec2Imag);
  Value int3 = builder.create<arith::MulFOp>(loc, vec1Real, vec2Imag);
  Value int4 = builder.create<arith::MulFOp>(loc, vec1Imag, vec2Real);

  return {builder.create<arith::SubFOp>(loc, int1, int2),
          builder.create<arith::AddFOp>(loc, int3, int4)};
}

// Function for calculating Transpose of 2D input MemRef.
void scalar2DMemRefTranspose(OpBuilder &builder, Location loc, Value memref1,
                             Value memref2, Value memref1NumRows,
                             Value memref1NumCols, Value memref2NumRows,
                             Value memref2NumCols, Value c0) {
  SmallVector<Value, 8> lowerBounds(2, c0);
  SmallVector<Value, 8> upperBounds{memref1NumRows, memref1NumCols};
  SmallVector<int64_t, 8> steps(2, 1);

  affine::buildAffineLoopNest(
      builder, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        Value pixelVal = builder.create<memref::LoadOp>(
            loc, builder.getF32Type(), memref1, ValueRange{ivs[0], ivs[1]});

        builder.create<memref::StoreOp>(loc, pixelVal, memref2,
                                        ValueRange{ivs[1], ivs[0]});
      });
}

// Function for calculating Hadamard product of complex type 2D MemRefs.
// Separate MemRefs for real and imaginary parts are expected.
void vector2DMemRefMultiply(OpBuilder &builder, Location loc, Value memRef1Real,
                            Value memRef1Imag, Value memRef2Real,
                            Value memRef2Imag, Value memRef3Real,
                            Value memRef3Imag, Value memRefNumRows,
                            Value memRefNumCols, Value c0, VectorType vecType) {
  SmallVector<Value, 8> lowerBounds(2, c0);
  SmallVector<Value, 8> upperBounds{memRefNumRows, memRefNumCols};
  SmallVector<int64_t, 8> steps(2, 1);

  affine::buildAffineLoopNest(
      builder, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        Value pixelVal1Real = builder.create<vector::LoadOp>(
            loc, vecType, memRef1Real, ValueRange{ivs[0], ivs[1]});
        Value pixelVal1Imag = builder.create<vector::LoadOp>(
            loc, vecType, memRef1Imag, ValueRange{ivs[0], ivs[1]});

        Value pixelVal2Real = builder.create<vector::LoadOp>(
            loc, vecType, memRef2Real, ValueRange{ivs[0], ivs[1]});
        Value pixelVal2Imag = builder.create<vector::LoadOp>(
            loc, vecType, memRef2Imag, ValueRange{ivs[0], ivs[1]});

        std::vector<Value> resVecs =
            complexVecMulI(builder, loc, pixelVal1Real, pixelVal1Imag,
                           pixelVal2Real, pixelVal2Imag);

        builder.create<vector::StoreOp>(loc, resVecs[0], memRef3Real,
                                        ValueRange{ivs[0], ivs[1]});
        builder.create<vector::StoreOp>(loc, resVecs[1], memRef3Imag,
                                        ValueRange{ivs[0], ivs[1]});
      });
}

// Function for implementing Cooley Tukey Butterfly algortihm for calculating
// inverse of discrete Fourier transform of invidiual 1D components of 2D input
// MemRef. Separate MemRefs for real and imaginary parts are expected.
void idft1DCooleyTukeyButterfly(OpBuilder &builder, Location loc,
                                Value memRefReal2D, Value memRefImag2D,
                                Value memRefLength, Value strideVal,
                                VectorType vecType, Value rowIndex, Value c0,
                                Value c1, int64_t step) {
  // Cooley Tukey Butterfly algorithm implementation.
  Value subProbs = builder.create<arith::ShRSIOp>(loc, memRefLength, c1);
  Value subProbSize, half = c1, i, jBegin, jEnd, j, angle;
  Value wStepReal, wStepImag, wReal, wImag, tmp1Real, tmp1Imag, tmp2Real,
      tmp2Imag;
  Value wRealVec, wImagVec, wStepRealVec, wStepImagVec;
  Value tmp2RealTemp, tmp2ImagTemp;

  Value upperBound =
      F32ToIndex(builder, loc,
                 builder.create<math::Log2Op>(
                     loc, indexToF32(builder, loc, memRefLength)));
  Value pos2MPI = builder.create<arith::ConstantFloatOp>(
      loc, (llvm::APFloat)(float)(2.0 * M_PI), builder.getF32Type());

  builder.create<scf::ForOp>(
      loc, c0, upperBound, c1, ValueRange{subProbs, half},
      [&](OpBuilder &builder, Location loc, ValueRange iv,
          ValueRange outerIterVR) {
        subProbSize = builder.create<arith::ShLIOp>(loc, outerIterVR[1], c1);
        angle = builder.create<arith::DivFOp>(
            loc, pos2MPI, indexToF32(builder, loc, subProbSize));

        wStepReal = builder.create<math::CosOp>(loc, angle);
        wStepRealVec =
            builder.create<vector::BroadcastOp>(loc, vecType, wStepReal);

        wStepImag = builder.create<math::SinOp>(loc, angle);
        wStepImagVec =
            builder.create<vector::BroadcastOp>(loc, vecType, wStepImag);

        builder.create<scf::ForOp>(
            loc, c0, outerIterVR[0], c1, ValueRange{},
            [&](OpBuilder &builder, Location loc, ValueRange iv1, ValueRange) {
              jBegin = builder.create<arith::MulIOp>(loc, iv1[0], subProbSize);
              jEnd = builder.create<arith::AddIOp>(loc, jBegin, outerIterVR[1]);
              wReal = builder.create<arith::ConstantFloatOp>(
                  loc, (llvm::APFloat)1.0f, builder.getF32Type());
              wImag = builder.create<arith::ConstantFloatOp>(
                  loc, (llvm::APFloat)0.0f, builder.getF32Type());

              wRealVec =
                  builder.create<vector::BroadcastOp>(loc, vecType, wReal);
              wImagVec =
                  builder.create<vector::BroadcastOp>(loc, vecType, wImag);

              // Vectorize stuff inside this loop (take care of tail processing
              // as well)
              builder.create<scf::ForOp>(
                  loc, jBegin, jEnd, strideVal, ValueRange{wRealVec, wImagVec},
                  [&](OpBuilder &builder, Location loc, ValueRange iv2,
                      ValueRange wVR) {
                    tmp1Real = builder.create<vector::LoadOp>(
                        loc, vecType, memRefReal2D,
                        ValueRange{rowIndex, iv2[0]});
                    tmp1Imag = builder.create<vector::LoadOp>(
                        loc, vecType, memRefImag2D,
                        ValueRange{rowIndex, iv2[0]});

                    Value secondIndex = builder.create<arith::AddIOp>(
                        loc, iv2[0], outerIterVR[1]);
                    tmp2RealTemp = builder.create<vector::LoadOp>(
                        loc, vecType, memRefReal2D,
                        ValueRange{rowIndex, secondIndex});
                    tmp2ImagTemp = builder.create<vector::LoadOp>(
                        loc, vecType, memRefImag2D,
                        ValueRange{rowIndex, secondIndex});

                    std::vector<Value> tmp2Vec =
                        complexVecMulI(builder, loc, tmp2RealTemp, tmp2ImagTemp,
                                       wVR[0], wVR[1]);

                    std::vector<Value> int1Vec =
                        complexVecAddI(builder, loc, tmp1Real, tmp1Imag,
                                       tmp2Vec[0], tmp2Vec[1]);
                    builder.create<vector::StoreOp>(
                        loc, int1Vec[0], memRefReal2D,
                        ValueRange{rowIndex, iv2[0]});
                    builder.create<vector::StoreOp>(
                        loc, int1Vec[1], memRefImag2D,
                        ValueRange{rowIndex, iv2[0]});

                    std::vector<Value> int2Vec =
                        complexVecSubI(builder, loc, tmp1Real, tmp1Imag,
                                       tmp2Vec[0], tmp2Vec[1]);
                    builder.create<vector::StoreOp>(
                        loc, int2Vec[0], memRefReal2D,
                        ValueRange{rowIndex, secondIndex});
                    builder.create<vector::StoreOp>(
                        loc, int2Vec[1], memRefImag2D,
                        ValueRange{rowIndex, secondIndex});

                    std::vector<Value> wUpdate =
                        complexVecMulI(builder, loc, wVR[0], wVR[1],
                                       wStepRealVec, wStepImagVec);

                    builder.create<scf::YieldOp>(
                        loc, ValueRange{wUpdate[0], wUpdate[1]});
                  });

              builder.create<scf::YieldOp>(loc);
            });
        Value updatedSubProbs =
            builder.create<arith::ShRSIOp>(loc, outerIterVR[0], c1);

        builder.create<scf::YieldOp>(loc,
                                     ValueRange{updatedSubProbs, subProbSize});
      });

  Value memRefLengthVec = builder.create<vector::BroadcastOp>(
      loc, vecType, indexToF32(builder, loc, memRefLength));

  builder.create<scf::ForOp>(
      loc, c0, memRefLength, strideVal, ValueRange{},
      [&](OpBuilder &builder, Location loc, ValueRange iv, ValueRange) {
        Value tempVecReal = builder.create<vector::LoadOp>(
            loc, vecType, memRefReal2D, ValueRange{rowIndex, iv[0]});
        Value tempResVecReal =
            builder.create<arith::DivFOp>(loc, tempVecReal, memRefLengthVec);
        builder.create<vector::StoreOp>(loc, tempResVecReal, memRefReal2D,
                                        ValueRange{rowIndex, iv[0]});

        Value tempVecImag = builder.create<vector::LoadOp>(
            loc, vecType, memRefImag2D, ValueRange{rowIndex, iv[0]});
        Value tempResVecImag =
            builder.create<arith::DivFOp>(loc, tempVecImag, memRefLengthVec);
        builder.create<vector::StoreOp>(loc, tempResVecImag, memRefImag2D,
                                        ValueRange{rowIndex, iv[0]});

        builder.create<scf::YieldOp>(loc);
      });
}

// Function for implementing Gentleman Sande Butterfly algortihm for calculating
// discrete Fourier transform of invidiual 1D components of 2D input MemRef.
// Separate MemRefs for real and imaginary parts are expected.
void dft1DGentlemanSandeButterfly(OpBuilder &builder, Location loc,
                                  Value memRefReal2D, Value memRefImag2D,
                                  Value memRefLength, Value strideVal,
                                  VectorType vecType, Value rowIndex, Value c0,
                                  Value c1, int64_t step) {
  // Gentleman Sande Butterfly algorithm implementation.
  Value subProbs = c1, subProbSize = memRefLength, i, jBegin, jEnd, j, half,
        angle;
  Value wStepReal, wStepImag, wReal, wImag, tmp1Real, tmp1Imag, tmp2Real,
      tmp2Imag;
  Value wRealVec, wImagVec, wStepRealVec, wStepImagVec;

  Value upperBound =
      F32ToIndex(builder, loc,
                 builder.create<math::Log2Op>(
                     loc, indexToF32(builder, loc, memRefLength)));
  Value neg2MPI = builder.create<arith::ConstantFloatOp>(
      loc, (llvm::APFloat)(float)(-2.0 * M_PI), builder.getF32Type());

  builder.create<scf::ForOp>(
      loc, c0, upperBound, c1, ValueRange{subProbs, subProbSize},
      [&](OpBuilder &builder, Location loc, ValueRange iv,
          ValueRange outerIterVR) {
        half = builder.create<arith::ShRSIOp>(loc, outerIterVR[1], c1);
        angle = builder.create<arith::DivFOp>(
            loc, neg2MPI, indexToF32(builder, loc, outerIterVR[1]));

        wStepReal = builder.create<math::CosOp>(loc, angle);
        wStepRealVec =
            builder.create<vector::BroadcastOp>(loc, vecType, wStepReal);

        wStepImag = builder.create<math::SinOp>(loc, angle);
        wStepImagVec =
            builder.create<vector::BroadcastOp>(loc, vecType, wStepImag);

        builder.create<scf::ForOp>(
            loc, c0, outerIterVR[0], c1, ValueRange{},
            [&](OpBuilder &builder, Location loc, ValueRange iv1, ValueRange) {
              jBegin =
                  builder.create<arith::MulIOp>(loc, iv1[0], outerIterVR[1]);
              jEnd = builder.create<arith::AddIOp>(loc, jBegin, half);
              wReal = builder.create<arith::ConstantFloatOp>(
                  loc, (llvm::APFloat)1.0f, builder.getF32Type());
              wImag = builder.create<arith::ConstantFloatOp>(
                  loc, (llvm::APFloat)0.0f, builder.getF32Type());

              wRealVec =
                  builder.create<vector::BroadcastOp>(loc, vecType, wReal);
              wImagVec =
                  builder.create<vector::BroadcastOp>(loc, vecType, wImag);

              // Vectorize stuff inside this loop (take care of tail processing
              // as well)
              builder.create<scf::ForOp>(
                  loc, jBegin, jEnd, strideVal, ValueRange{wRealVec, wImagVec},
                  [&](OpBuilder &builder, Location loc, ValueRange iv2,
                      ValueRange wVR) {
                    tmp1Real = builder.create<vector::LoadOp>(
                        loc, vecType, memRefReal2D,
                        ValueRange{rowIndex, iv2[0]});
                    tmp1Imag = builder.create<vector::LoadOp>(
                        loc, vecType, memRefImag2D,
                        ValueRange{rowIndex, iv2[0]});

                    Value secondIndex =
                        builder.create<arith::AddIOp>(loc, iv2[0], half);
                    tmp2Real = builder.create<vector::LoadOp>(
                        loc, vecType, memRefReal2D,
                        ValueRange{rowIndex, secondIndex});
                    tmp2Imag = builder.create<vector::LoadOp>(
                        loc, vecType, memRefImag2D,
                        ValueRange{rowIndex, secondIndex});

                    std::vector<Value> int1Vec = complexVecAddI(
                        builder, loc, tmp1Real, tmp1Imag, tmp2Real, tmp2Imag);
                    builder.create<vector::StoreOp>(
                        loc, int1Vec[0], memRefReal2D,
                        ValueRange{rowIndex, iv2[0]});
                    builder.create<vector::StoreOp>(
                        loc, int1Vec[1], memRefImag2D,
                        ValueRange{rowIndex, iv2[0]});

                    std::vector<Value> int2Vec = complexVecSubI(
                        builder, loc, tmp1Real, tmp1Imag, tmp2Real, tmp2Imag);
                    std::vector<Value> int3Vec = complexVecMulI(
                        builder, loc, int2Vec[0], int2Vec[1], wVR[0], wVR[1]);

                    builder.create<vector::StoreOp>(
                        loc, int3Vec[0], memRefReal2D,
                        ValueRange{rowIndex, secondIndex});
                    builder.create<vector::StoreOp>(
                        loc, int3Vec[1], memRefImag2D,
                        ValueRange{rowIndex, secondIndex});

                    std::vector<Value> wUpdate =
                        complexVecMulI(builder, loc, wVR[0], wVR[1],
                                       wStepRealVec, wStepImagVec);

                    builder.create<scf::YieldOp>(
                        loc, ValueRange{wUpdate[0], wUpdate[1]});
                  });

              builder.create<scf::YieldOp>(loc);
            });
        Value updatedSubProbs =
            builder.create<arith::ShLIOp>(loc, outerIterVR[0], c1);

        builder.create<scf::YieldOp>(loc, ValueRange{updatedSubProbs, half});
      });
}

// Function for applying inverse of discrete fourier transform on a 2D MemRef.
// Separate MemRefs for real and imaginary parts are expected.
void idft2D(OpBuilder &builder, Location loc, Value container2DReal,
            Value container2DImag, Value container2DRows, Value container2DCols,
            Value intermediateReal, Value intermediateImag, Value c0, Value c1,
            Value strideVal, VectorType vecType) {
  builder.create<affine::AffineForOp>(
      loc, ValueRange{c0}, builder.getDimIdentityMap(),
      ValueRange{container2DRows}, builder.getDimIdentityMap(), 1, std::nullopt,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
          ValueRange itrArg) {
        idft1DCooleyTukeyButterfly(builder, loc, container2DReal,
                                   container2DImag, container2DCols, strideVal,
                                   vecType, iv, c0, c1, 1);

        nestedBuilder.create<affine::AffineYieldOp>(nestedLoc);
      });

  scalar2DMemRefTranspose(builder, loc, container2DReal, intermediateReal,
                          container2DRows, container2DCols, container2DCols,
                          container2DRows, c0);
  scalar2DMemRefTranspose(builder, loc, container2DImag, intermediateImag,
                          container2DRows, container2DCols, container2DCols,
                          container2DRows, c0);

  builder.create<affine::AffineForOp>(
      loc, ValueRange{c0}, builder.getDimIdentityMap(),
      ValueRange{container2DCols}, builder.getDimIdentityMap(), 1, std::nullopt,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
          ValueRange itrArg) {
        idft1DCooleyTukeyButterfly(builder, loc, intermediateReal,
                                   intermediateImag, container2DRows, strideVal,
                                   vecType, iv, c0, c1, 1);

        nestedBuilder.create<affine::AffineYieldOp>(nestedLoc);
      });

  Value transposeCond = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ne, container2DRows, container2DCols);
  builder.create<scf::IfOp>(
      loc, transposeCond,
      [&](OpBuilder &builder, Location loc) {
        scalar2DMemRefTranspose(builder, loc, intermediateReal, container2DReal,
                                container2DCols, container2DRows,
                                container2DRows, container2DCols, c0);
        scalar2DMemRefTranspose(builder, loc, intermediateImag, container2DImag,
                                container2DCols, container2DRows,
                                container2DRows, container2DCols, c0);

        builder.create<scf::YieldOp>(loc);
      },
      [&](OpBuilder &builder, Location loc) {
        builder.create<memref::CopyOp>(loc, intermediateReal, container2DReal);
        builder.create<memref::CopyOp>(loc, intermediateImag, container2DImag);

        builder.create<scf::YieldOp>(loc);
      });
}

// Function for applying discrete fourier transform on a 2D MemRef. Separate
// MemRefs for real and imaginary parts are expected.
void dft2D(OpBuilder &builder, Location loc, Value container2DReal,
           Value container2DImag, Value container2DRows, Value container2DCols,
           Value intermediateReal, Value intermediateImag, Value c0, Value c1,
           Value strideVal, VectorType vecType) {
  builder.create<affine::AffineForOp>(
      loc, ValueRange{c0}, builder.getDimIdentityMap(),
      ValueRange{container2DRows}, builder.getDimIdentityMap(), 1, std::nullopt,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
          ValueRange itrArg) {
        dft1DGentlemanSandeButterfly(builder, loc, container2DReal,
                                     container2DImag, container2DCols,
                                     strideVal, vecType, iv, c0, c1, 1);

        nestedBuilder.create<affine::AffineYieldOp>(nestedLoc);
      });

  scalar2DMemRefTranspose(builder, loc, container2DReal, intermediateReal,
                          container2DRows, container2DCols, container2DCols,
                          container2DRows, c0);
  scalar2DMemRefTranspose(builder, loc, container2DImag, intermediateImag,
                          container2DRows, container2DCols, container2DCols,
                          container2DRows, c0);

  builder.create<affine::AffineForOp>(
      loc, ValueRange{c0}, builder.getDimIdentityMap(),
      ValueRange{container2DCols}, builder.getDimIdentityMap(), 1, std::nullopt,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
          ValueRange itrArg) {
        dft1DGentlemanSandeButterfly(builder, loc, intermediateReal,
                                     intermediateImag, container2DRows,
                                     strideVal, vecType, iv, c0, c1, 1);

        nestedBuilder.create<affine::AffineYieldOp>(nestedLoc);
      });

  Value transposeCond = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ne, container2DRows, container2DCols);
  builder.create<scf::IfOp>(
      loc, transposeCond,
      [&](OpBuilder &builder, Location loc) {
        scalar2DMemRefTranspose(builder, loc, intermediateReal, container2DReal,
                                container2DCols, container2DRows,
                                container2DRows, container2DCols, c0);
        scalar2DMemRefTranspose(builder, loc, intermediateImag, container2DImag,
                                container2DCols, container2DRows,
                                container2DRows, container2DCols, c0);

        builder.create<scf::YieldOp>(loc);
      },
      [&](OpBuilder &builder, Location loc) {
        builder.create<memref::CopyOp>(loc, intermediateReal, container2DReal);
        builder.create<memref::CopyOp>(loc, intermediateImag, container2DImag);

        builder.create<scf::YieldOp>(loc);
      });
}

} // namespace buddy

#endif // UTILS_UTILS_DEF
