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
    op = arith::ConstantFloatOp::create(builder, loc, floatType, zero);
  } else if (elemTy.isInteger(bitWidth)) {
    IntegerType type = IntegerType::get(ctx, bitWidth);
    op = arith::ConstantIntOp::create(builder, loc, type, 0);
  }

  return op;
}

// Function to test whether a value is equivalent to zero or not.
Value zeroCond(OpBuilder &builder, Location loc, Type elemType, Value value,
               Value zeroElem) {
  Value cond;
  auto bitWidth = elemType.getIntOrFloatBitWidth();
  if (elemType.isF32() || elemType.isF64()) {
    cond = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::ONE, value,
                                         zeroElem);
  } else if (elemType.isInteger(bitWidth)) {
    cond = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne, value,
                                         zeroElem);
  }
  return cond;
}

// Create an inverted mask having all 1's shifted to right side.
Value createInvertedMask(OpBuilder &builder, Location loc, Value strideVal,
                         VectorType vectorMaskTy, Value leftIndex) {
  Value leftMask =
      vector::CreateMaskOp::create(builder, loc, vectorMaskTy, leftIndex);
  Value maskInverter =
      vector::CreateMaskOp::create(builder, loc, vectorMaskTy, strideVal);
  Value rightMask = arith::SubIOp::create(builder, loc, maskInverter, leftMask);
  return rightMask;
}

// Cast a value from index type to f32 type.
Value indexToF32(OpBuilder &builder, Location loc, Value val) {
  Value interm1 =
      arith::IndexCastOp::create(builder, loc, builder.getI32Type(), val);
  return arith::SIToFPOp::create(builder, loc, builder.getF32Type(), interm1);
}

// Cast a value from f32 type to index type.
Value F32ToIndex(OpBuilder &builder, Location loc, Value val) {
  Value interm1 =
      arith::FPToUIOp::create(builder, loc, builder.getI32Type(), val);
  return arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                            interm1);
}

// Round off floating point value to nearest integer type value.
Value roundOff(OpBuilder &builder, Location loc, Value val) {
  Value ceilVal = math::CeilOp::create(builder, loc, val);
  Value floorVal = math::FloorOp::create(builder, loc, val);

  Value diffCeil = arith::SubFOp::create(builder, loc, ceilVal, val);
  Value diffFloor = arith::SubFOp::create(builder, loc, val, floorVal);

  Value diffCond = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OGT,
                                                 diffCeil, diffFloor);

  return arith::SelectOp::create(builder, loc, diffCond, floorVal, ceilVal);
}

// Bound values to permissible range of allocatable values w.r.t output image.
Value valBound(OpBuilder &builder, Location loc, Value val, Value lastElemF32,
               Value c0F32) {
  Value interm1 = arith::MaximumFOp::create(builder, loc, val, c0F32);
  return arith::MinimumFOp::create(builder, loc, interm1, lastElemF32);
}

// check if lb <= val < ub and returns Value 0 or 1
Value inBound(OpBuilder &builder, Location loc, Value val, Value lb, Value ub) {
  Value greaterThanLb =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sle, lb, val);
  Value lowerThanUb =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt, val, ub);
  return arith::AndIOp::create(builder, loc, greaterThanLb, lowerThanUb);
}

// Equivalent of std::iota.
Value iotaVec(OpBuilder &builder, Location loc, MLIRContext *ctx,
              Value indexStart, Value strideVal, VectorType vecType, Value c0,
              int64_t stride) {
  // ToDo : Try to get rid of memref load/store, find less expensive ways of
  // implementing this function.
  MemRefType memTy = MemRefType::get({stride}, builder.getF32Type());
  Value tempMem = memref::AllocOp::create(builder, loc, memTy);

  affine::AffineForOp::create(builder, 
      loc, ValueRange{c0}, builder.getDimIdentityMap(), ValueRange{strideVal},
      builder.getDimIdentityMap(), 1, ValueRange{},
      [&](OpBuilder &builder, Location loc, Value iv, ValueRange iterArg) {
        Value iotaValIndex = arith::AddIOp::create(builder, loc, iv, indexStart);
        Value iotaVal = indexToF32(builder, loc, iotaValIndex);

        memref::StoreOp::create(builder, loc, iotaVal, tempMem, ValueRange{iv});
        affine::AffineYieldOp::create(builder, loc);
      });

  return vector::LoadOp::create(builder, loc, vecType, tempMem, ValueRange{c0});
}

// Generate vector[0, 1, ..., length - 1] with f32 type
Value iotaVec0F32(OpBuilder &builder, Location loc, int64_t length) {
  MLIRContext *ctx = builder.getContext();
  std::vector<float> vec(length);
  std::iota(vec.begin(), vec.end(), .0);
  Value res = arith::ConstantOp::create(builder, 
      loc,
      DenseFPElementsAttr::get(VectorType::get(length, Float32Type::get(ctx)),
                               ArrayRef<float>(vec)));
  return res;
}

// Cast index type value to f32 type and then expand it in a vector.
Value castAndExpand(OpBuilder &builder, Location loc, Value val,
                    VectorType vecType) {
  Value interm1 = indexToF32(builder, loc, val);
  return vector::BroadcastOp::create(builder, loc, vecType, interm1);
}

// print values(for debug use)
void printValues(OpBuilder &builder, Location loc,
                 std::initializer_list<Value> values) {
  if (empty(values))
    return;
  Type valueTy = values.begin()->getType();
  VectorType vecTy = VectorType::get({(long)values.size()}, valueTy);
  Value vec = vector::BroadcastOp::create(builder, loc, vecTy, *values.begin());
  int idx = 0;
  for (auto value : values) {
    if (idx != 0) {
      // all values should have same type
      assert(value.getType() == valueTy);
      Value idxVal = arith::ConstantIndexOp::create(builder, loc, idx);
      vec = vector::InsertOp::create(builder, loc, value, vec, idxVal);
    }
    idx++;
  }
  vector::PrintOp::create(builder, loc, vec);
}

// Function for calculating complex addition of 2 input 1D complex vectors.
// Separate vectors for real and imaginary parts are expected.
inline std::vector<Value> complexVecAddI(OpBuilder &builder, Location loc,
                                         Value vec1Real, Value vec1Imag,
                                         Value vec2Real, Value vec2Imag) {
  return {arith::AddFOp::create(builder, loc, vec1Real, vec2Real),
          arith::AddFOp::create(builder, loc, vec1Imag, vec2Imag)};
}

// Function for calculating complex subtraction of 2 input 1D complex vectors.
// Separate vectors for real and imaginary parts are expected.
inline std::vector<Value> complexVecSubI(OpBuilder &builder, Location loc,
                                         Value vec1Real, Value vec1Imag,
                                         Value vec2Real, Value vec2Imag) {
  return {arith::SubFOp::create(builder, loc, vec1Real, vec2Real),
          arith::SubFOp::create(builder, loc, vec1Imag, vec2Imag)};
}

// Function for calculating complex product of 2 input 1D complex vectors.
// Separate vectors for real and imaginary parts are expected.
std::vector<Value> complexVecMulI(OpBuilder &builder, Location loc,
                                  Value vec1Real, Value vec1Imag,
                                  Value vec2Real, Value vec2Imag) {
  Value int1 = arith::MulFOp::create(builder, loc, vec1Real, vec2Real);
  Value int2 = arith::MulFOp::create(builder, loc, vec1Imag, vec2Imag);
  Value int3 = arith::MulFOp::create(builder, loc, vec1Real, vec2Imag);
  Value int4 = arith::MulFOp::create(builder, loc, vec1Imag, vec2Real);

  return {arith::SubFOp::create(builder, loc, int1, int2),
          arith::AddFOp::create(builder, loc, int3, int4)};
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
        Value pixelVal = memref::LoadOp::create(builder, 
            loc, builder.getF32Type(), memref1, ValueRange{ivs[0], ivs[1]});

        memref::StoreOp::create(builder, loc, pixelVal, memref2,
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
        Value pixelVal1Real = vector::LoadOp::create(builder, 
            loc, vecType, memRef1Real, ValueRange{ivs[0], ivs[1]});
        Value pixelVal1Imag = vector::LoadOp::create(builder, 
            loc, vecType, memRef1Imag, ValueRange{ivs[0], ivs[1]});

        Value pixelVal2Real = vector::LoadOp::create(builder, 
            loc, vecType, memRef2Real, ValueRange{ivs[0], ivs[1]});
        Value pixelVal2Imag = vector::LoadOp::create(builder, 
            loc, vecType, memRef2Imag, ValueRange{ivs[0], ivs[1]});

        std::vector<Value> resVecs =
            complexVecMulI(builder, loc, pixelVal1Real, pixelVal1Imag,
                           pixelVal2Real, pixelVal2Imag);

        vector::StoreOp::create(builder, loc, resVecs[0], memRef3Real,
                                        ValueRange{ivs[0], ivs[1]});
        vector::StoreOp::create(builder, loc, resVecs[1], memRef3Imag,
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
  Value subProbs = arith::ShRSIOp::create(builder, loc, memRefLength, c1);
  Value subProbSize, half = c1, i, jBegin, jEnd, j, angle;
  Value wStepReal, wStepImag, wReal, wImag, tmp1Real, tmp1Imag, tmp2Real,
      tmp2Imag;
  Value wRealVec, wImagVec, wStepRealVec, wStepImagVec;
  Value tmp2RealTemp, tmp2ImagTemp;

  Value upperBound =
      F32ToIndex(builder, loc,
                 math::Log2Op::create(builder, 
                     loc, indexToF32(builder, loc, memRefLength)));
  Value pos2MPI = arith::ConstantFloatOp::create(builder, 
      loc, builder.getF32Type(), (llvm::APFloat)(float)(2.0 * M_PI));

  scf::ForOp::create(builder, 
      loc, c0, upperBound, c1, ValueRange{subProbs, half},
      [&](OpBuilder &builder, Location loc, ValueRange iv,
          ValueRange outerIterVR) {
        subProbSize = arith::ShLIOp::create(builder, loc, outerIterVR[1], c1);
        angle = arith::DivFOp::create(builder, 
            loc, pos2MPI, indexToF32(builder, loc, subProbSize));

        wStepReal = math::CosOp::create(builder, loc, angle);
        wStepRealVec =
            vector::BroadcastOp::create(builder, loc, vecType, wStepReal);

        wStepImag = math::SinOp::create(builder, loc, angle);
        wStepImagVec =
            vector::BroadcastOp::create(builder, loc, vecType, wStepImag);

        scf::ForOp::create(builder, 
            loc, c0, outerIterVR[0], c1, ValueRange{},
            [&](OpBuilder &builder, Location loc, ValueRange iv1, ValueRange) {
              jBegin = arith::MulIOp::create(builder, loc, iv1[0], subProbSize);
              jEnd = arith::AddIOp::create(builder, loc, jBegin, outerIterVR[1]);
              wReal = arith::ConstantFloatOp::create(builder, 
                  loc, builder.getF32Type(), (llvm::APFloat)1.0f);
              wImag = arith::ConstantFloatOp::create(builder, 
                  loc, builder.getF32Type(), (llvm::APFloat)0.0f);

              wRealVec =
                  vector::BroadcastOp::create(builder, loc, vecType, wReal);
              wImagVec =
                  vector::BroadcastOp::create(builder, loc, vecType, wImag);

              // Vectorize stuff inside this loop (take care of tail processing
              // as well)
              scf::ForOp::create(builder, 
                  loc, jBegin, jEnd, strideVal, ValueRange{wRealVec, wImagVec},
                  [&](OpBuilder &builder, Location loc, ValueRange iv2,
                      ValueRange wVR) {
                    tmp1Real = vector::LoadOp::create(builder, 
                        loc, vecType, memRefReal2D,
                        ValueRange{rowIndex, iv2[0]});
                    tmp1Imag = vector::LoadOp::create(builder, 
                        loc, vecType, memRefImag2D,
                        ValueRange{rowIndex, iv2[0]});

                    Value secondIndex = arith::AddIOp::create(builder, 
                        loc, iv2[0], outerIterVR[1]);
                    tmp2RealTemp = vector::LoadOp::create(builder, 
                        loc, vecType, memRefReal2D,
                        ValueRange{rowIndex, secondIndex});
                    tmp2ImagTemp = vector::LoadOp::create(builder, 
                        loc, vecType, memRefImag2D,
                        ValueRange{rowIndex, secondIndex});

                    std::vector<Value> tmp2Vec =
                        complexVecMulI(builder, loc, tmp2RealTemp, tmp2ImagTemp,
                                       wVR[0], wVR[1]);

                    std::vector<Value> int1Vec =
                        complexVecAddI(builder, loc, tmp1Real, tmp1Imag,
                                       tmp2Vec[0], tmp2Vec[1]);
                    vector::StoreOp::create(builder, 
                        loc, int1Vec[0], memRefReal2D,
                        ValueRange{rowIndex, iv2[0]});
                    vector::StoreOp::create(builder, 
                        loc, int1Vec[1], memRefImag2D,
                        ValueRange{rowIndex, iv2[0]});

                    std::vector<Value> int2Vec =
                        complexVecSubI(builder, loc, tmp1Real, tmp1Imag,
                                       tmp2Vec[0], tmp2Vec[1]);
                    vector::StoreOp::create(builder, 
                        loc, int2Vec[0], memRefReal2D,
                        ValueRange{rowIndex, secondIndex});
                    vector::StoreOp::create(builder, 
                        loc, int2Vec[1], memRefImag2D,
                        ValueRange{rowIndex, secondIndex});

                    std::vector<Value> wUpdate =
                        complexVecMulI(builder, loc, wVR[0], wVR[1],
                                       wStepRealVec, wStepImagVec);

                    scf::YieldOp::create(builder, 
                        loc, ValueRange{wUpdate[0], wUpdate[1]});
                  });

              scf::YieldOp::create(builder, loc);
            });
        Value updatedSubProbs =
            arith::ShRSIOp::create(builder, loc, outerIterVR[0], c1);

        scf::YieldOp::create(builder, loc,
                                     ValueRange{updatedSubProbs, subProbSize});
      });

  Value memRefLengthVec = vector::BroadcastOp::create(builder, 
      loc, vecType, indexToF32(builder, loc, memRefLength));

  scf::ForOp::create(builder, 
      loc, c0, memRefLength, strideVal, ValueRange{},
      [&](OpBuilder &builder, Location loc, ValueRange iv, ValueRange) {
        Value tempVecReal = vector::LoadOp::create(builder, 
            loc, vecType, memRefReal2D, ValueRange{rowIndex, iv[0]});
        Value tempResVecReal =
            arith::DivFOp::create(builder, loc, tempVecReal, memRefLengthVec);
        vector::StoreOp::create(builder, loc, tempResVecReal, memRefReal2D,
                                        ValueRange{rowIndex, iv[0]});

        Value tempVecImag = vector::LoadOp::create(builder, 
            loc, vecType, memRefImag2D, ValueRange{rowIndex, iv[0]});
        Value tempResVecImag =
            arith::DivFOp::create(builder, loc, tempVecImag, memRefLengthVec);
        vector::StoreOp::create(builder, loc, tempResVecImag, memRefImag2D,
                                        ValueRange{rowIndex, iv[0]});

        scf::YieldOp::create(builder, loc);
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
                 math::Log2Op::create(builder, 
                     loc, indexToF32(builder, loc, memRefLength)));
  Value neg2MPI = arith::ConstantFloatOp::create(builder, 
      loc, builder.getF32Type(), (llvm::APFloat)(float)(-2.0 * M_PI));

  scf::ForOp::create(builder, 
      loc, c0, upperBound, c1, ValueRange{subProbs, subProbSize},
      [&](OpBuilder &builder, Location loc, ValueRange iv,
          ValueRange outerIterVR) {
        half = arith::ShRSIOp::create(builder, loc, outerIterVR[1], c1);
        angle = arith::DivFOp::create(builder, 
            loc, neg2MPI, indexToF32(builder, loc, outerIterVR[1]));

        wStepReal = math::CosOp::create(builder, loc, angle);
        wStepRealVec =
            vector::BroadcastOp::create(builder, loc, vecType, wStepReal);

        wStepImag = math::SinOp::create(builder, loc, angle);
        wStepImagVec =
            vector::BroadcastOp::create(builder, loc, vecType, wStepImag);

        scf::ForOp::create(builder, 
            loc, c0, outerIterVR[0], c1, ValueRange{},
            [&](OpBuilder &builder, Location loc, ValueRange iv1, ValueRange) {
              jBegin =
                  arith::MulIOp::create(builder, loc, iv1[0], outerIterVR[1]);
              jEnd = arith::AddIOp::create(builder, loc, jBegin, half);
              wReal = arith::ConstantFloatOp::create(builder, 
                  loc, builder.getF32Type(), (llvm::APFloat)1.0f);
              wImag = arith::ConstantFloatOp::create(builder, 
                  loc, builder.getF32Type(), (llvm::APFloat)0.0f);

              wRealVec =
                  vector::BroadcastOp::create(builder, loc, vecType, wReal);
              wImagVec =
                  vector::BroadcastOp::create(builder, loc, vecType, wImag);

              // Vectorize stuff inside this loop (take care of tail processing
              // as well)
              scf::ForOp::create(builder, 
                  loc, jBegin, jEnd, strideVal, ValueRange{wRealVec, wImagVec},
                  [&](OpBuilder &builder, Location loc, ValueRange iv2,
                      ValueRange wVR) {
                    tmp1Real = vector::LoadOp::create(builder, 
                        loc, vecType, memRefReal2D,
                        ValueRange{rowIndex, iv2[0]});
                    tmp1Imag = vector::LoadOp::create(builder, 
                        loc, vecType, memRefImag2D,
                        ValueRange{rowIndex, iv2[0]});

                    Value secondIndex =
                        arith::AddIOp::create(builder, loc, iv2[0], half);
                    tmp2Real = vector::LoadOp::create(builder, 
                        loc, vecType, memRefReal2D,
                        ValueRange{rowIndex, secondIndex});
                    tmp2Imag = vector::LoadOp::create(builder, 
                        loc, vecType, memRefImag2D,
                        ValueRange{rowIndex, secondIndex});

                    std::vector<Value> int1Vec = complexVecAddI(
                        builder, loc, tmp1Real, tmp1Imag, tmp2Real, tmp2Imag);
                    vector::StoreOp::create(builder, 
                        loc, int1Vec[0], memRefReal2D,
                        ValueRange{rowIndex, iv2[0]});
                    vector::StoreOp::create(builder, 
                        loc, int1Vec[1], memRefImag2D,
                        ValueRange{rowIndex, iv2[0]});

                    std::vector<Value> int2Vec = complexVecSubI(
                        builder, loc, tmp1Real, tmp1Imag, tmp2Real, tmp2Imag);
                    std::vector<Value> int3Vec = complexVecMulI(
                        builder, loc, int2Vec[0], int2Vec[1], wVR[0], wVR[1]);

                    vector::StoreOp::create(builder, 
                        loc, int3Vec[0], memRefReal2D,
                        ValueRange{rowIndex, secondIndex});
                    vector::StoreOp::create(builder, 
                        loc, int3Vec[1], memRefImag2D,
                        ValueRange{rowIndex, secondIndex});

                    std::vector<Value> wUpdate =
                        complexVecMulI(builder, loc, wVR[0], wVR[1],
                                       wStepRealVec, wStepImagVec);

                    scf::YieldOp::create(builder, 
                        loc, ValueRange{wUpdate[0], wUpdate[1]});
                  });

              scf::YieldOp::create(builder, loc);
            });
        Value updatedSubProbs =
            arith::ShLIOp::create(builder, loc, outerIterVR[0], c1);

        scf::YieldOp::create(builder, loc, ValueRange{updatedSubProbs, half});
      });
}

// Function for applying inverse of discrete fourier transform on a 2D MemRef.
// Separate MemRefs for real and imaginary parts are expected.
void idft2D(OpBuilder &builder, Location loc, Value container2DReal,
            Value container2DImag, Value container2DRows, Value container2DCols,
            Value intermediateReal, Value intermediateImag, Value c0, Value c1,
            Value strideVal, VectorType vecType) {
  affine::AffineForOp::create(builder, 
      loc, ValueRange{c0}, builder.getDimIdentityMap(),
      ValueRange{container2DRows}, builder.getDimIdentityMap(), 1, ValueRange{},
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
          ValueRange itrArg) {
        idft1DCooleyTukeyButterfly(builder, loc, container2DReal,
                                   container2DImag, container2DCols, strideVal,
                                   vecType, iv, c0, c1, 1);

        affine::AffineYieldOp::create(nestedBuilder, nestedLoc);
      });

  scalar2DMemRefTranspose(builder, loc, container2DReal, intermediateReal,
                          container2DRows, container2DCols, container2DCols,
                          container2DRows, c0);
  scalar2DMemRefTranspose(builder, loc, container2DImag, intermediateImag,
                          container2DRows, container2DCols, container2DCols,
                          container2DRows, c0);

  affine::AffineForOp::create(builder, 
      loc, ValueRange{c0}, builder.getDimIdentityMap(),
      ValueRange{container2DCols}, builder.getDimIdentityMap(), 1, ValueRange{},
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
          ValueRange itrArg) {
        idft1DCooleyTukeyButterfly(builder, loc, intermediateReal,
                                   intermediateImag, container2DRows, strideVal,
                                   vecType, iv, c0, c1, 1);

        affine::AffineYieldOp::create(nestedBuilder, nestedLoc);
      });

  Value transposeCond = arith::CmpIOp::create(builder, 
      loc, arith::CmpIPredicate::ne, container2DRows, container2DCols);
  scf::IfOp::create(builder, 
      loc, transposeCond,
      [&](OpBuilder &builder, Location loc) {
        scalar2DMemRefTranspose(builder, loc, intermediateReal, container2DReal,
                                container2DCols, container2DRows,
                                container2DRows, container2DCols, c0);
        scalar2DMemRefTranspose(builder, loc, intermediateImag, container2DImag,
                                container2DCols, container2DRows,
                                container2DRows, container2DCols, c0);

        scf::YieldOp::create(builder, loc);
      },
      [&](OpBuilder &builder, Location loc) {
        memref::CopyOp::create(builder, loc, intermediateReal, container2DReal);
        memref::CopyOp::create(builder, loc, intermediateImag, container2DImag);

        scf::YieldOp::create(builder, loc);
      });
}

// Function for applying discrete fourier transform on a 2D MemRef. Separate
// MemRefs for real and imaginary parts are expected.
void dft2D(OpBuilder &builder, Location loc, Value container2DReal,
           Value container2DImag, Value container2DRows, Value container2DCols,
           Value intermediateReal, Value intermediateImag, Value c0, Value c1,
           Value strideVal, VectorType vecType) {
  affine::AffineForOp::create(builder, 
      loc, ValueRange{c0}, builder.getDimIdentityMap(),
      ValueRange{container2DRows}, builder.getDimIdentityMap(), 1, ValueRange{},
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
          ValueRange itrArg) {
        dft1DGentlemanSandeButterfly(builder, loc, container2DReal,
                                     container2DImag, container2DCols,
                                     strideVal, vecType, iv, c0, c1, 1);

        affine::AffineYieldOp::create(nestedBuilder, nestedLoc);
      });

  scalar2DMemRefTranspose(builder, loc, container2DReal, intermediateReal,
                          container2DRows, container2DCols, container2DCols,
                          container2DRows, c0);
  scalar2DMemRefTranspose(builder, loc, container2DImag, intermediateImag,
                          container2DRows, container2DCols, container2DCols,
                          container2DRows, c0);

  affine::AffineForOp::create(builder, 
      loc, ValueRange{c0}, builder.getDimIdentityMap(),
      ValueRange{container2DCols}, builder.getDimIdentityMap(), 1, ValueRange{},
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
          ValueRange itrArg) {
        dft1DGentlemanSandeButterfly(builder, loc, intermediateReal,
                                     intermediateImag, container2DRows,
                                     strideVal, vecType, iv, c0, c1, 1);

        affine::AffineYieldOp::create(nestedBuilder, nestedLoc);
      });

  Value transposeCond = arith::CmpIOp::create(builder, 
      loc, arith::CmpIPredicate::ne, container2DRows, container2DCols);
  scf::IfOp::create(builder, 
      loc, transposeCond,
      [&](OpBuilder &builder, Location loc) {
        scalar2DMemRefTranspose(builder, loc, intermediateReal, container2DReal,
                                container2DCols, container2DRows,
                                container2DRows, container2DCols, c0);
        scalar2DMemRefTranspose(builder, loc, intermediateImag, container2DImag,
                                container2DCols, container2DRows,
                                container2DRows, container2DCols, c0);

        scf::YieldOp::create(builder, loc);
      },
      [&](OpBuilder &builder, Location loc) {
        memref::CopyOp::create(builder, loc, intermediateReal, container2DReal);
        memref::CopyOp::create(builder, loc, intermediateImag, container2DImag);

        scf::YieldOp::create(builder, loc);
      });
}

} // namespace buddy

#endif // UTILS_UTILS_DEF
