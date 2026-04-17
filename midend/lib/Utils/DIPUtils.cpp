//====- DIPUtils.cpp ------------------------------------------------------===//
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
// This file implements DIP dialect specific utility functions for the buddy
// compiler ecosystem.
//
//===----------------------------------------------------------------------===//

#ifndef UTILS_DIPUTILS_DEF
#define UTILS_DIPUTILS_DEF

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/IR/AffineValueMap.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <numeric>
#include <vector>

#include "DIP/DIPDialect.h"
#include "DIP/DIPOps.h"
#include "Utils/AffineTransformUtils.h"
#include "Utils/DIPUtils.h"
#include "Utils/Utils.h"

using namespace mlir;

namespace buddy {
namespace dip {
template DIP_ERROR
checkDIPCommonTypes<dip::Corr2DOp>(dip::Corr2DOp,
                                   const std::vector<Value> &args);
template DIP_ERROR
checkDIPCommonTypes<dip::Rotate2DOp>(dip::Rotate2DOp,
                                     const std::vector<Value> &args);

template DIP_ERROR
checkDIPCommonTypes<dip::Rotate4DOp>(dip::Rotate4DOp,
                                     const std::vector<Value> &args);

template DIP_ERROR
checkDIPCommonTypes<dip::Resize2DOp>(dip::Resize2DOp,
                                     const std::vector<Value> &args);

template DIP_ERROR
checkDIPCommonTypes<dip::Resize4D_NHWCOp>(dip::Resize4D_NHWCOp,
                                          const std::vector<Value> &args);

template DIP_ERROR
checkDIPCommonTypes<dip::Resize4D_NCHWOp>(dip::Resize4D_NCHWOp,
                                          const std::vector<Value> &args);

template DIP_ERROR
checkDIPCommonTypes<dip::Erosion2DOp>(dip::Erosion2DOp,
                                      const std::vector<Value> &args);
template DIP_ERROR
checkDIPCommonTypes<dip::Dilation2DOp>(dip::Dilation2DOp,
                                       const std::vector<Value> &args);
template DIP_ERROR
checkDIPCommonTypes<dip::Opening2DOp>(dip::Opening2DOp,
                                      const std::vector<Value> &args);
template DIP_ERROR
checkDIPCommonTypes<dip::Closing2DOp>(dip::Closing2DOp,
                                      const std::vector<Value> &args);
template DIP_ERROR
checkDIPCommonTypes<dip::TopHat2DOp>(dip::TopHat2DOp,
                                     const std::vector<Value> &args);
template DIP_ERROR
checkDIPCommonTypes<dip::BottomHat2DOp>(dip::BottomHat2DOp,
                                        const std::vector<Value> &args);
template DIP_ERROR
checkDIPCommonTypes<dip::MorphGrad2DOp>(dip::MorphGrad2DOp,
                                        const std::vector<Value> &args);

// Function for applying type check mechanisms for all DIP dialect operations.
template <typename DIPOP>
DIP_ERROR checkDIPCommonTypes(DIPOP op, const std::vector<Value> &args) {

  const auto getType = [&](int argIndex) { return args[argIndex].getType(); };

  const auto getElementType = [&](int argIndex) {
    return mlir::cast<mlir::MemRefType>(getType(argIndex)).getElementType();
  };

  // NB: we can infer element type for all related memrefs to be the same as
  // input since we verified that the operand types are the same.
  const auto notSameElementTypeForMemrefs = [](const Type &type) {
    const auto &bitWidth = type.getIntOrFloatBitWidth();
    return !type.isF64() && !type.isF32() && !type.isInteger(bitWidth);
  };

  if (op->getName().stripDialect() == "corr_2d") {
    auto inElemTy = getElementType(0);
    auto kElemTy = getElementType(1);
    auto outElemTy = getElementType(2);
    auto constElemTy = getType(3);

    if (inElemTy != kElemTy || kElemTy != outElemTy ||
        outElemTy != constElemTy) {
      return DIP_ERROR::INCONSISTENT_TYPES;
    }

    if (notSameElementTypeForMemrefs(inElemTy)) {
      return DIP_ERROR::UNSUPPORTED_TYPE;
    }
  } else if (op->getName().stripDialect() == "rotate_2d" ||
             op->getName().stripDialect() == "rotate_4d" ||
             op->getName().stripDialect() == "resize_2d" ||
             op->getName().stripDialect() == "resize_4d_nhwc" ||
             op->getName().stripDialect() == "resize_4d_nchw") {
    auto inElemTy = getElementType(0);
    auto outElemTy = getElementType(1);

    if (inElemTy != outElemTy) {
      return DIP_ERROR::INCONSISTENT_TYPES;
    }

    // NB: we can infer element type for all related memrefs to be the same as
    // input since we verified that the operand types are the same.
    if (notSameElementTypeForMemrefs(inElemTy)) {
      return DIP_ERROR::UNSUPPORTED_TYPE;
    }
  } else if (op->getName().stripDialect() == "erosion_2d" ||
             op->getName().stripDialect() == "dilation_2d") {

    auto inElemTy = getElementType(0);
    auto kElemTy = getElementType(1);
    auto outElemTy = getElementType(2);
    auto copyElemTy = getElementType(3);
    auto constElemTy = getType(4);

    if (inElemTy != kElemTy || kElemTy != outElemTy ||
        outElemTy != copyElemTy || copyElemTy != constElemTy) {
      return DIP_ERROR::INCONSISTENT_TYPES;
    }

    if (notSameElementTypeForMemrefs(inElemTy)) {
      return DIP_ERROR::UNSUPPORTED_TYPE;
    }
  } else if (op->getName().stripDialect() == "opening_2d" ||
             op->getName().stripDialect() == "closing_2d") {
    auto inElemTy = getElementType(0);
    auto kElemTy = getElementType(1);
    auto outElemTy = getElementType(2);
    auto outElemTy1 = getElementType(3);
    auto copyElemTy = getElementType(4);
    auto copyElemTy1 = getElementType(5);
    auto constElemTy = getType(6);

    if (inElemTy != kElemTy || kElemTy != outElemTy ||
        outElemTy != outElemTy1 || outElemTy1 != copyElemTy ||
        copyElemTy != copyElemTy1 || copyElemTy1 != constElemTy) {
      return DIP_ERROR::INCONSISTENT_TYPES;
    }

    if (notSameElementTypeForMemrefs(inElemTy)) {
      return DIP_ERROR::UNSUPPORTED_TYPE;
    }
  } else if (op->getName().stripDialect() == "tophat_2d" ||
             op->getName().stripDialect() == "bottomhat_2d" ||
             op->getName().stripDialect() == "morphgrad_2d") {
    auto inElemTy = getElementType(0);
    auto kElemTy = getElementType(1);
    auto outElemTy = getElementType(2);
    auto outElemTy1 = getElementType(3);
    auto outElemTy2 = getElementType(4);
    auto inElemTy1 = getElementType(5);
    auto copyElemTy = getElementType(6);
    auto copyElemTy1 = getElementType(7);
    auto constElemTy = getType(8);

    if (inElemTy != kElemTy || kElemTy != outElemTy ||
        outElemTy != outElemTy1 || outElemTy1 != outElemTy2 ||
        outElemTy2 != inElemTy1 || inElemTy1 != copyElemTy ||
        copyElemTy != copyElemTy1 || copyElemTy1 != constElemTy) {
      return DIP_ERROR::INCONSISTENT_TYPES;
    }

    if (notSameElementTypeForMemrefs(inElemTy)) {
      return DIP_ERROR::UNSUPPORTED_TYPE;
    }
  }

  return DIP_ERROR::NO_ERROR;
}

// Inserts FMA operation into a given location `loc` based on type `type`.
// Note: FMA is done by Multiply and Add for integer types, because there is no
// dedicated FMA operation for them.
// Supported types: f32, f64, integer types
Value insertFMAOp(OpBuilder &builder, Location loc, VectorType type,
                  Value inputVec, Value kernelVec, Value outputVec) {
  Value res = {};
  auto elemTy = type.getElementType();
  auto bitWidth = elemTy.getIntOrFloatBitWidth();
  if (elemTy.isF32() || elemTy.isF64()) {
    res = vector::FMAOp::create(builder, loc, inputVec, kernelVec, outputVec);
  } else if (elemTy.isInteger(bitWidth)) {
    Value mul = arith::MulIOp::create(builder, loc, inputVec, kernelVec);
    res = arith::AddIOp::create(builder, loc, mul, outputVec);
  }

  return res;
}

// Calculate result of FMA and store it in output memref. This function cannot
// handle tail processing.
void calcAndStoreFMAwoTailProcessing(OpBuilder &builder, Location loc,
                                     VectorType vecType, Value inputVec,
                                     Value kernelVec, Value output,
                                     Value beginIdx, Value endIdx) {
  Value outputVec = vector::LoadOp::create(builder, 
      loc, vecType, output, ValueRange{beginIdx, endIdx});
  Value resVec =
      insertFMAOp(builder, loc, vecType, inputVec, kernelVec, outputVec);
  vector::StoreOp::create(builder, loc, resVec, output,
                                  ValueRange{beginIdx, endIdx});
}

// Checks if we encountered a tail (columns remaining after processing in
// batches of stride size).
Value tailChecker(OpBuilder &builder, Location loc, AffineMap calcHelper,
                  Value strideVal, Value kernelSize, Value c1, Value pseudoCol,
                  Value colPivot) {
  Value tailChecker = affine::AffineApplyOp::create(builder, 
      loc, calcHelper, ValueRange{strideVal, kernelSize, c1});
  Value colEndDistance =
      arith::SubIOp::create(builder, loc, pseudoCol, colPivot);
  Value tailCond = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                                                 colEndDistance, tailChecker);
  return tailCond;
}

// Creates the required mask which is to be used for tail processing.
Value tailMaskCreator(OpBuilder &builder, Location loc, Value inputCol,
                      Value colPivot, VectorType vectorMaskTy) {
  Value extraElemCount = arith::SubIOp::create(builder, loc, inputCol, colPivot);
  Value tailMask =
      vector::CreateMaskOp::create(builder, loc, vectorMaskTy, extraElemCount);
  return tailMask;
}

// Calculate result of FMA and store it in output memref. This function can
// handle tail processing.
void calcAndStoreFMAwTailProcessing(OpBuilder &builder, Location loc,
                                    VectorType vecType, Value inputVec,
                                    Value kernelVec, Value output,
                                    Value beginIdx, Value endIdx,
                                    Value tailCond, Value zeroPadding,
                                    Value inputCol, VectorType vectorMaskTy) {
  scf::IfOp::create(builder, 
      loc, tailCond,
      [&](OpBuilder &builder, Location loc) {
        Value outputVec = vector::LoadOp::create(builder, 
            loc, vecType, output, ValueRange{beginIdx, endIdx});
        Value resVec =
            insertFMAOp(builder, loc, vecType, inputVec, kernelVec, outputVec);
        vector::StoreOp::create(builder, loc, resVec, output,
                                        ValueRange{beginIdx, endIdx});

        scf::YieldOp::create(builder, loc);
      },
      [&](OpBuilder &builder, Location loc) {
        Value extraElemMask =
            tailMaskCreator(builder, loc, inputCol, endIdx, vectorMaskTy);
        Value outputVec = vector::MaskedLoadOp::create(builder, 
            loc, vecType, output, ValueRange{beginIdx, endIdx}, extraElemMask,
            zeroPadding);
        Value resVec =
            insertFMAOp(builder, loc, vecType, inputVec, kernelVec, outputVec);
        vector::MaskedStoreOp::create(builder, 
            loc, output, ValueRange{beginIdx, endIdx}, extraElemMask, resVec);

        scf::YieldOp::create(builder, loc);
      });
}

// Apply 3 shear method and return mapped values.
std::vector<Value> shearTransform(OpBuilder &builder, Location loc,
                                  Value originalX, Value originalY,
                                  Value sinVec, Value tanVec) {
  Value yTan1 = arith::MulFOp::create(builder, loc, tanVec, originalY);
  Value xIntermediate1 = arith::SubFOp::create(builder, loc, originalX, yTan1);
  Value xIntermediate = roundOff(builder, loc, xIntermediate1);

  Value xSin = arith::MulFOp::create(builder, loc, xIntermediate, sinVec);
  Value newY1 = arith::AddFOp::create(builder, loc, xSin, originalY);
  Value newY = roundOff(builder, loc, newY1);

  Value yTan2 = arith::MulFOp::create(builder, loc, newY, tanVec);
  Value newX1 = arith::SubFOp::create(builder, loc, xIntermediate, yTan2);
  Value newX = roundOff(builder, loc, newX1);

  return {newY, newX};
}

// Apply standard rotation matrix transformation and return mapped values.
std::vector<Value> standardRotate(OpBuilder &builder, Location loc,
                                  Value originalX, Value originalY,
                                  Value sinVec, Value cosVec) {
  Value ySin = arith::MulFOp::create(builder, loc, originalY, sinVec);
  Value yCos = arith::MulFOp::create(builder, loc, originalY, cosVec);

  Value xSin = arith::MulFOp::create(builder, loc, originalX, sinVec);
  Value xCos = arith::MulFOp::create(builder, loc, originalX, cosVec);

  Value newY1 = arith::SubFOp::create(builder, loc, yCos, xSin);
  Value newX1 = arith::AddFOp::create(builder, loc, ySin, xCos);

  return {roundOff(builder, loc, newY1), roundOff(builder, loc, newX1)};
}

// Get center co-ordinates w.r.t given dimension.
Value getCenter(OpBuilder &builder, Location loc, MLIRContext *ctx, Value dim) {
  Value dimF32 = indexToF32(builder, loc, dim);
  Value c1f = arith::ConstantFloatOp::create(builder, loc, builder.getF32Type(),
                                                     (llvm::APFloat)1.0f);
  Value c2f = arith::ConstantFloatOp::create(builder, loc, builder.getF32Type(),
                                                     (llvm::APFloat)2.0f);

  Value temp1 = arith::AddFOp::create(builder, loc, dimF32, c1f);
  Value temp2 = arith::DivFOp::create(builder, loc, temp1, c2f);
  Value center = arith::SubFOp::create(builder, loc, temp2, c1f);
  Value centerRound = roundOff(builder, loc, center);

  return F32ToIndex(builder, loc, centerRound);
}

// Scale pixel co-ordinates appropriately before calculating their rotated
// position(s).
Value pixelScaling(OpBuilder &builder, Location loc, Value imageDImF32Vec,
                   Value coordVec, Value imageCenterF32Vec, Value c1F32Vec) {
  Value interm1 = arith::SubFOp::create(builder, loc, imageDImF32Vec, coordVec);
  Value interm2 =
      arith::SubFOp::create(builder, loc, interm1, imageCenterF32Vec);

  return arith::SubFOp::create(builder, loc, interm2, c1F32Vec);
}

// Extract values present at a particular index in two vectors for using
// those values to load an element from a memref.
std::vector<Value> extractIndices(OpBuilder &builder, Location loc, Value xVec,
                                  Value yVec, Value vecIndex, Value xUpperBound,
                                  Value yUpperBound, Value c0F32) {
  Value xPos = vector::ExtractOp::create(builder, loc, xVec, vecIndex);
  Value yPos = vector::ExtractOp::create(builder, loc, yVec, vecIndex);

  Value xPosBound = valBound(builder, loc, xPos, xUpperBound, c0F32);
  Value yPosBound = valBound(builder, loc, yPos, yUpperBound, c0F32);

  return {F32ToIndex(builder, loc, xPosBound),
          F32ToIndex(builder, loc, yPosBound)};
}

// Fill appropriate pixel data in its corresponding co-ordinate of the output
// image.
void fillPixels(OpBuilder &builder, Location loc, Value resXVec, Value resYVec,
                Value xVec, Value yVec, Value input, Value output, Value c0,
                Value strideVal, Value outputRowLastElemF32,
                Value outputColLastElemF32, Value inputRowLastElemF32,
                Value inputColLastElemF32, Value c0F32) {
  affine::AffineForOp::create(builder, 
      loc, ValueRange{c0}, builder.getDimIdentityMap(), ValueRange{strideVal},
      builder.getDimIdentityMap(), /*step*/ 1, ValueRange{},
      [&](OpBuilder &builder, Location loc, ValueRange ivs,
          ValueRange iterArg) {
        std::vector<Value> origIndices =
            extractIndices(builder, loc, xVec, yVec, ivs[0],
                           inputColLastElemF32, inputRowLastElemF32, c0F32);
        std::vector<Value> resIndices =
            extractIndices(builder, loc, resXVec, resYVec, ivs[0],
                           outputColLastElemF32, outputRowLastElemF32, c0F32);

        Value pixelVal = memref::LoadOp::create(builder, 
            loc, builder.getF32Type(), input,
            ValueRange{origIndices[1], origIndices[0]});
        memref::StoreOp::create(builder, 
            loc, pixelVal, output, ValueRange{resIndices[1], resIndices[0]});

        affine::AffineYieldOp::create(builder, loc);
      });
}

// Fill appropriate pixel data in its corresponding co-ordinate of the output
// image.
void fillPixelsNearestNeighbour4D(
    OpBuilder &builder, Location loc, Value ivs0, Value ivs1, Value resXVec,
    Value resYVec, Value xVec, Value yVec, Value input, Value output, Value c0,
    Value strideVal, Value outputRowLastElemF32, Value outputColLastElemF32,
    Value inputRowLastElemF32, Value inputColLastElemF32, Value c0F32,
    Value dataCondition) {
  affine::AffineForOp::create(builder, 
      loc, ValueRange{c0}, builder.getDimIdentityMap(), ValueRange{strideVal},
      builder.getDimIdentityMap(), /*step*/ 1, ValueRange{},
      [&](OpBuilder &builder, Location loc, ValueRange ivs,
          ValueRange iterArg) {
        std::vector<Value> origIndices =
            extractIndices(builder, loc, xVec, yVec, ivs[0],
                           inputColLastElemF32, inputRowLastElemF32, c0F32);
        std::vector<Value> resIndices =
            extractIndices(builder, loc, resXVec, resYVec, ivs[0],
                           outputColLastElemF32, outputRowLastElemF32, c0F32);

        auto ifop = scf::IfOp::create(builder, 
            loc, dataCondition,
            [&](OpBuilder &builder, Location loc) {
              Value pixelVal = memref::LoadOp::create(builder, 
                  loc, builder.getF32Type(), input,
                  ValueRange{ivs0, origIndices[1], origIndices[0], ivs1});
              scf::YieldOp::create(builder, loc, pixelVal);
            },
            [&](OpBuilder &builder, Location loc) {
              Value pixelVal = memref::LoadOp::create(builder, 
                  loc, builder.getF32Type(), input,
                  ValueRange{ivs0, ivs1, origIndices[1], origIndices[0]});
              scf::YieldOp::create(builder, loc, pixelVal);
            });
        Value pixelVal = ifop.getResult(0);

        scf::IfOp::create(builder, 
            loc, dataCondition,
            [&](OpBuilder &builder, Location loc) {
              memref::StoreOp::create(builder, 
                  loc, pixelVal, output,
                  ValueRange{ivs0, resIndices[1], resIndices[0], ivs1});
              scf::YieldOp::create(builder, loc);
            },
            [&](OpBuilder &builder, Location loc) {
              memref::StoreOp::create(builder, 
                  loc, pixelVal, output,
                  ValueRange{ivs0, ivs1, resIndices[1], resIndices[0]});
              scf::YieldOp::create(builder, loc);
            });

        affine::AffineYieldOp::create(builder, loc);
      });
}

// Calculate tan(angle / 2) where angle is a function parameter.
Value customTanVal(OpBuilder &builder, Location loc, Value angleVal) {
  Value c2F32 = arith::ConstantFloatOp::create(builder, 
      loc, builder.getF32Type(), (llvm::APFloat)2.0f);
  Value angleVal_2 = arith::DivFOp::create(builder, loc, angleVal, c2F32);

  Value sinVal = math::SinOp::create(builder, loc, angleVal_2);
  Value cosVal = math::CosOp::create(builder, loc, angleVal_2);

  return arith::DivFOp::create(builder, loc, sinVal, cosVal);
}

// Calculate the real affine matrix for rotation by
// getting the rotation matrix and modfiying it to
// preserve the full original image .
SmallVector<Value, 6> calculateRotationMatrix(OpBuilder &builder, Location loc,
                                              Value inputCol, Value inputRow,
                                              Value outputCol, Value outputRow,
                                              Value angleVal) {
  Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);

  // let alpha = scale * cos(angle), beta = scale * sin(angle)
  // the affine matrix would be as follow:
  // [[alpha, beta, (1 - alpha) * centerx - beta * centery],
  //  [-beta, alpha, beta * centerx + (1 - alpha) * centery]]
  Value centerX = arith::ShRSIOp::create(builder, loc, inputCol, c1);
  Value centerY = arith::ShRSIOp::create(builder, loc, inputRow, c1);
  Value centerXF32 = indexToF32(builder, loc, centerX);
  Value centerYF32 = indexToF32(builder, loc, centerY);

  //  scaling ratio = 1.
  Value scale = indexToF32(builder, loc, c1);

  auto affineMatrix = dip::getRotationMatrix(builder, loc, centerXF32,
                                             centerYF32, angleVal, scale);

  //  modify the affine matrix to preserve the full original
  //  image after rotation
  Value deltaXI = arith::SubIOp::create(builder, loc, outputCol, inputCol);
  Value deltaYI = arith::SubIOp::create(builder, loc, outputRow, inputRow);
  Value deltaXIDiv2 = arith::ShRSIOp::create(builder, loc, deltaXI, c1);
  Value deltaYIDiv2 = arith::ShRSIOp::create(builder, loc, deltaYI, c1);
  Value deltaXFDiv2 = indexToF32(builder, loc, deltaXIDiv2);
  Value deltaYFDiv2 = indexToF32(builder, loc, deltaYIDiv2);

  affineMatrix[2] =
      arith::AddFOp::create(builder, loc, affineMatrix[2], deltaXFDiv2);
  affineMatrix[5] =
      arith::AddFOp::create(builder, loc, affineMatrix[5], deltaYFDiv2);

  return affineMatrix;
}

// Get affine matrix used in rotation.
SmallVector<Value, 6> getRotationMatrix(OpBuilder &builder, Location loc,
                                        Value centerX, Value centerY,
                                        Value angle, Value scale) {
  // the format of this matrix is
  // [[m0, m1, m2],
  //  [m3, m4, m5]]
  // and the rotation will be calculated as
  // x = m0 * x_new + m1 * y_new + m2
  // y_new = m3 * x_new + m4 * y_new + m
  Value alpha0 = math::CosOp::create(builder, loc, angle);
  Value alpha = arith::MulFOp::create(builder, loc, alpha0, scale);
  Value beta0 = math::SinOp::create(builder, loc, angle);
  Value beta = arith::MulFOp::create(builder, loc, beta0, scale);
  Value oneMinusAlpha = arith::SubFOp::create(builder, 
      loc,
      arith::ConstantOp::create(builder, loc,
                                        builder.getF32FloatAttr((float)1.)),
      alpha);
  Value m20 = arith::MulFOp::create(builder, loc, oneMinusAlpha, centerX);
  Value m21 = arith::MulFOp::create(builder, loc, beta, centerY);
  Value m50 = arith::MulFOp::create(builder, loc, beta, centerX);
  Value m51 = arith::MulFOp::create(builder, loc, oneMinusAlpha, centerY);
  Value m0 = alpha;
  Value m1 = beta;
  Value m2 = arith::SubFOp::create(builder, loc, m20, m21);
  Value m3 = arith::NegFOp::create(builder, loc, beta);
  Value m4 = alpha;
  Value m5 = arith::AddFOp::create(builder, loc, m50, m51);

  return SmallVector<Value, 6>{m0, m1, m2, m3, m4, m5};
}

// Compute the inverse of the affine matrix
// After inverting the matrix, the calculation of the affine transformation can
// become x_src = m0 * x_dst + m1 * y_dst + m2 y_src = m3 * x_dst + m4 * y_dst +
// m5
inline void inverseAffineMatrix(OpBuilder &builder, Location loc,
                                SmallVector<Value, 6> &affineMatrix) {
  Value c0F32 = arith::ConstantOp::create(builder, 
      loc, builder.getF32FloatAttr((float).0));
  Value m0pm4 =
      arith::MulFOp::create(builder, loc, affineMatrix[0], affineMatrix[4]);
  Value m1pm3 =
      arith::MulFOp::create(builder, loc, affineMatrix[1], affineMatrix[3]);
  Value D = arith::SubFOp::create(builder, loc, m0pm4, m1pm3);
  Value dEq0 =
      arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OEQ, D, c0F32);
  auto scfRes = scf::IfOp::create(builder, 
      loc, dEq0,
      [&](OpBuilder &thenBuilder, Location thenLoc) {
        scf::YieldOp::create(thenBuilder, thenLoc, ValueRange{c0F32});
      },
      [&](OpBuilder &elseBuilder, Location elseLoc) {
        Value c1F32 = arith::ConstantOp::create(elseBuilder, 
            elseLoc, builder.getF32FloatAttr((float)1.));
        Value res = arith::DivFOp::create(elseBuilder, elseLoc, c1F32, D);
        scf::YieldOp::create(elseBuilder, elseLoc, ValueRange{res});
      });
  D = scfRes.getResult(0);
  Value negD = arith::NegFOp::create(builder, loc, D);
  Value a0 = arith::MulFOp::create(builder, loc, affineMatrix[4], D);
  Value a4 = arith::MulFOp::create(builder, loc, affineMatrix[0], D);
  affineMatrix[0] = a0;
  affineMatrix[1] = arith::MulFOp::create(builder, loc, affineMatrix[1], negD);
  affineMatrix[3] = arith::MulFOp::create(builder, loc, affineMatrix[3], negD);
  affineMatrix[4] = a4;
  Value m0pm2 =
      arith::MulFOp::create(builder, loc, affineMatrix[0], affineMatrix[2]);
  Value m1pm5 =
      arith::MulFOp::create(builder, loc, affineMatrix[1], affineMatrix[5]);
  Value negB1 = arith::AddFOp::create(builder, loc, m0pm2, m1pm5);
  Value m2pm3 =
      arith::MulFOp::create(builder, loc, affineMatrix[2], affineMatrix[3]);
  Value m4pm5 =
      arith::MulFOp::create(builder, loc, affineMatrix[4], affineMatrix[5]);
  Value negB2 = arith::AddFOp::create(builder, loc, m2pm3, m4pm5);
  affineMatrix[2] = arith::NegFOp::create(builder, loc, negB1);
  affineMatrix[5] = arith::NegFOp::create(builder, loc, negB2);
}

// Controls affine transform application.
void affineTransformController(OpBuilder &builder, Location loc,
                               MLIRContext *ctx, Value input, Value output,
                               SmallVector<Value, 6> affineMatrix,
                               int64_t stride, dip::ImageFormat format) {
  VectorType vectorTyF32 = VectorType::get({stride}, Float32Type::get(ctx));
  VectorType vectorTyI32 = VectorType::get({stride}, IntegerType::get(ctx, 32));

  Value c0Index = arith::ConstantIndexOp::create(builder, loc, 0);
  Value c1Index = arith::ConstantIndexOp::create(builder, loc, 1);

  inverseAffineMatrix(builder, loc, affineMatrix);

  Value m0Vec =
      vector::BroadcastOp::create(builder, loc, vectorTyF32, affineMatrix[0]);
  Value m2Vec =
      vector::BroadcastOp::create(builder, loc, vectorTyF32, affineMatrix[2]);
  Value m3Vec =
      vector::BroadcastOp::create(builder, loc, vectorTyF32, affineMatrix[3]);
  Value m5Vec =
      vector::BroadcastOp::create(builder, loc, vectorTyF32, affineMatrix[5]);

  //  get the output image dimensions
  int dimIndex = -1;
  if (format == dip::ImageFormat::HW) {
    dimIndex = 0;
  } else if (format == dip::ImageFormat::NHWC) {
    dimIndex = 1;
  } else if (format == dip::ImageFormat::NCHW) {
    dimIndex = 2;
  }
  Value rowIndex = arith::ConstantIndexOp::create(builder, loc, dimIndex);
  Value colIndex = arith::ConstantIndexOp::create(builder, loc, dimIndex + 1);
  Value outputRow = memref::DimOp::create(builder, loc, output, rowIndex);
  Value outputCol = memref::DimOp::create(builder, loc, output, colIndex);

  Value strideVal = arith::ConstantIndexOp::create(builder, loc, stride);
  Value outputColStrideRatio =
      arith::DivUIOp::create(builder, loc, outputCol, strideVal);
  Value outputColMultiple = arith::MulIOp::create(builder, 
      loc, arith::AddIOp::create(builder, loc, outputColStrideRatio, c1Index),
      strideVal);

  Value xVecInitial = iotaVec0F32(builder, loc, stride);

  MemRefType dynamicTypeI32 =
      MemRefType::get(ShapedType::kDynamic, IntegerType::get(ctx, 32));

  // compute x*m0+m2 and x*m3+m5 and store the results into xMm0 and xMm3
  Value xMm0 =
      memref::AllocOp::create(builder, loc, dynamicTypeI32, outputColMultiple);
  Value xMm3 =
      memref::AllocOp::create(builder, loc, dynamicTypeI32, outputColMultiple);

  // RSV_BITS = reserved bits, how many bits should be reserved for fraction
  // part
  // TODO: make reserved bits configurable
  const int RSV_BITS = 5;
  Value c_rsv = arith::ConstantOp::create(builder, 
      loc, builder.getF32FloatAttr((float)(1 << RSV_BITS)));
  Value rsv_delta = arith::ConstantOp::create(builder, 
      loc, builder.getI32IntegerAttr(1 << (RSV_BITS - 1)));
  Value c_rsvVec = vector::BroadcastOp::create(builder, loc, vectorTyF32, c_rsv);
  Value rsv_deltaVec =
      vector::BroadcastOp::create(builder, loc, vectorTyI32, rsv_delta);

  affine::AffineForOp::create(builder, 
      loc, ValueRange{c0Index}, builder.getDimIdentityMap(),
      ValueRange{outputColMultiple}, builder.getDimIdentityMap(), stride,
      ValueRange{},
      [&](OpBuilder &builderFor, Location locFor, ValueRange ivsFor,
          ValueRange iterArg) {
        Value delta = vector::BroadcastOp::create(builderFor, 
            locFor, vectorTyF32, indexToF32(builderFor, locFor, ivsFor[0]));
        Value xVec =
            arith::AddFOp::create(builderFor, locFor, xVecInitial, delta);
        Value x0xM0 = arith::MulFOp::create(builderFor, locFor, xVec, m0Vec);
        Value x1xM3 = arith::MulFOp::create(builderFor, locFor, xVec, m3Vec);
        Value x0xM0addM2 =
            arith::AddFOp::create(builderFor, locFor, x0xM0, m2Vec);
        Value x1xM3addM5 =
            arith::AddFOp::create(builderFor, locFor, x1xM3, m5Vec);
        Value x0xM0addM2xrsv =
            arith::MulFOp::create(builderFor, locFor, x0xM0addM2, c_rsvVec);
        Value x1xM3addM5xrsv =
            arith::MulFOp::create(builderFor, locFor, x1xM3addM5, c_rsvVec);
        Value x0 = arith::FPToSIOp::create(builderFor, locFor, vectorTyI32,
                                                      x0xM0addM2xrsv);
        Value x1 = arith::FPToSIOp::create(builderFor, locFor, vectorTyI32,
                                                      x1xM3addM5xrsv);
        Value x0addrsv_delta =
            arith::AddIOp::create(builderFor, locFor, x0, rsv_deltaVec);
        Value x1addrsv_delta =
            arith::AddIOp::create(builderFor, locFor, x1, rsv_deltaVec);
        vector::StoreOp::create(builderFor, locFor, x0addrsv_delta, xMm0,
                                           ValueRange{ivsFor[0]});
        vector::StoreOp::create(builderFor, locFor, x1addrsv_delta, xMm3,
                                           ValueRange{ivsFor[0]});
        affine::AffineYieldOp::create(builderFor, locFor);
      });

  affineTransformCore(builder, loc, ctx, input, output, c0Index, outputRow,
                      c0Index, outputCol, affineMatrix[1], affineMatrix[4],
                      xMm0, xMm3, stride, RSV_BITS, 0, format);

  memref::DeallocOp::create(builder, loc, xMm0);
  memref::DeallocOp::create(builder, loc, xMm3);
}

// Controls shear transform application.
void shearTransformController(
    OpBuilder &builder, Location loc, MLIRContext *ctx,
    SmallVector<Value, 8> lowerBounds, SmallVector<Value, 8> upperBounds,
    SmallVector<int64_t, 8> steps, Value strideVal, Value input, Value output,
    Value sinVec, Value tanVec, Value inputRowF32Vec, Value inputColF32Vec,
    Value inputCenterYF32Vec, Value inputCenterXF32Vec,
    Value outputCenterYF32Vec, Value outputCenterXF32Vec,
    Value outputRowLastElemF32, Value outputColLastElemF32,
    Value inputRowLastElemF32, Value inputColLastElemF32, Value c0, Value c0F32,
    Value c1F32Vec, VectorType vectorTy32, int64_t stride, FloatType f32) {
  affine::buildAffineLoopNest(
      builder, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        Value ivs0F32 = indexToF32(builder, loc, ivs[0]);
        Value yVec = vector::BroadcastOp::create(builder, loc, vectorTy32, ivs0F32);
        Value xVec = iotaVec(builder, loc, ctx, ivs[1], strideVal, vectorTy32,
                             c0, stride);

        Value yVecModified = pixelScaling(builder, loc, inputRowF32Vec, yVec,
                                          inputCenterYF32Vec, c1F32Vec);
        Value xVecModified = pixelScaling(builder, loc, inputColF32Vec, xVec,
                                          inputCenterXF32Vec, c1F32Vec);

        std::vector<Value> resIndices = shearTransform(
            builder, loc, xVecModified, yVecModified, sinVec, tanVec);
        Value resYVec = arith::SubFOp::create(builder, loc, outputCenterYF32Vec,
                                                      resIndices[0]);
        Value resXVec = arith::SubFOp::create(builder, loc, outputCenterXF32Vec,
                                                      resIndices[1]);

        fillPixels(builder, loc, resXVec, resYVec, xVec, yVec, input, output,
                   c0, strideVal, outputRowLastElemF32, outputColLastElemF32,
                   inputRowLastElemF32, inputColLastElemF32, c0F32);
      });
}

// Controls standard rotation matrix application.
void standardRotateController(
    OpBuilder &builder, Location loc, MLIRContext *ctx,
    SmallVector<Value, 8> lowerBounds, SmallVector<Value, 8> upperBounds,
    SmallVector<int64_t, 8> steps, Value strideVal, Value input, Value output,
    Value sinVec, Value angleVal, Value inputRowF32Vec, Value inputColF32Vec,
    Value inputCenterYF32Vec, Value inputCenterXF32Vec,
    Value outputCenterYF32Vec, Value outputCenterXF32Vec,
    Value outputRowLastElemF32, Value outputColLastElemF32,
    Value inputRowLastElemF32, Value inputColLastElemF32, Value c0, Value c0F32,
    Value c1F32Vec, VectorType vectorTy32, int64_t stride, FloatType f32) {
  Value cosVal = math::CosOp::create(builder, loc, angleVal);
  Value cosVec = vector::BroadcastOp::create(builder, loc, vectorTy32, cosVal);

  affine::buildAffineLoopNest(
      builder, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        Value ivs0F32 = indexToF32(builder, loc, ivs[0]);
        Value yVec = vector::BroadcastOp::create(builder, loc, vectorTy32, ivs0F32);
        Value xVec = iotaVec(builder, loc, ctx, ivs[1], strideVal, vectorTy32,
                             c0, stride);

        Value yVecModified = pixelScaling(builder, loc, inputRowF32Vec, yVec,
                                          inputCenterYF32Vec, c1F32Vec);
        Value xVecModified = pixelScaling(builder, loc, inputColF32Vec, xVec,
                                          inputCenterXF32Vec, c1F32Vec);

        std::vector<Value> resIndices = standardRotate(
            builder, loc, xVecModified, yVecModified, sinVec, cosVec);
        Value resYVec = arith::SubFOp::create(builder, loc, outputCenterYF32Vec,
                                                      resIndices[0]);
        Value resXVec = arith::SubFOp::create(builder, loc, outputCenterXF32Vec,
                                                      resIndices[1]);

        fillPixels(builder, loc, resXVec, resYVec, xVec, yVec, input, output,
                   c0, strideVal, inputRowLastElemF32, inputColLastElemF32,
                   outputRowLastElemF32, outputColLastElemF32, c0F32);
      });
}

// Fills pixels in bilinear interpolation fashion.
void fillPixelsBilinearInterpolate(
    OpBuilder &builder, Location loc, Value resXVec, Value resYVec,
    Value xVec_L, Value yVec_L, Value xVec_H, Value yVec_H, Value input,
    Value output, Value c0, Value strideVal, Value xVecWeight, Value yVecWeight,
    Value outputRowLastElemF32, Value outputColLastElemF32,
    Value inputRowLastElemF32, Value inputColLastElemF32, Value c0F32,
    Value c1F32) {
  affine::AffineForOp::create(builder, 
      loc, ValueRange{c0}, builder.getDimIdentityMap(), ValueRange{strideVal},
      builder.getDimIdentityMap(), /*step*/ 1, ValueRange{},
      [&](OpBuilder &builder, Location loc, ValueRange ivs,
          ValueRange iterArg) {
        std::vector<Value> resIndices =
            extractIndices(builder, loc, resXVec, resYVec, ivs[0],
                           outputColLastElemF32, outputRowLastElemF32, c0F32);

        std::vector<Value> inputIndices_L =
            extractIndices(builder, loc, xVec_L, yVec_L, ivs[0],
                           inputColLastElemF32, inputRowLastElemF32, c0F32);
        std::vector<Value> inputIndices_H =
            extractIndices(builder, loc, xVec_H, yVec_H, ivs[0],
                           inputColLastElemF32, inputRowLastElemF32, c0F32);

        std::vector<Value> indexWeights;
        Value xPos_temp =
            vector::ExtractOp::create(builder, loc, xVecWeight, ivs[0]);
        Value yPos_temp =
            vector::ExtractOp::create(builder, loc, yVecWeight, ivs[0]);

        indexWeights.push_back(
            valBound(builder, loc, xPos_temp, inputColLastElemF32, c0F32));
        indexWeights.push_back(
            valBound(builder, loc, yPos_temp, inputRowLastElemF32, c0F32));

        std::vector<Value> indexWeights_UnitComplements = {
            arith::SubFOp::create(builder, loc, c1F32, indexWeights[0]),
            arith::SubFOp::create(builder, loc, c1F32, indexWeights[1])};

        Value pixelVal_a = memref::LoadOp::create(builder, 
            loc, builder.getF32Type(), input,
            ValueRange{inputIndices_L[1], inputIndices_L[0]});
        Value pixelVal_b = memref::LoadOp::create(builder, 
            loc, builder.getF32Type(), input,
            ValueRange{inputIndices_H[1], inputIndices_L[0]});
        Value pixelVal_c = memref::LoadOp::create(builder, 
            loc, builder.getF32Type(), input,
            ValueRange{inputIndices_L[1], inputIndices_H[0]});
        Value pixelVal_d = memref::LoadOp::create(builder, 
            loc, builder.getF32Type(), input,
            ValueRange{inputIndices_H[1], inputIndices_H[0]});

        Value weightVal1 =
            arith::MulFOp::create(builder, loc, indexWeights_UnitComplements[0],
                                          indexWeights_UnitComplements[1]);
        Value weightVal2 = arith::MulFOp::create(builder, 
            loc, indexWeights[0], indexWeights_UnitComplements[1]);
        Value weightVal3 = arith::MulFOp::create(builder, 
            loc, indexWeights[1], indexWeights_UnitComplements[0]);
        Value weightVal4 = arith::MulFOp::create(builder, loc, indexWeights[0],
                                                         indexWeights[1]);

        Value interm1 =
            arith::MulFOp::create(builder, loc, pixelVal_a, weightVal1);
        Value interm2 =
            arith::MulFOp::create(builder, loc, pixelVal_b, weightVal2);
        Value interm3 =
            arith::MulFOp::create(builder, loc, pixelVal_c, weightVal3);
        Value interm4 =
            arith::MulFOp::create(builder, loc, pixelVal_d, weightVal4);

        Value pixel_interm1 =
            arith::AddFOp::create(builder, loc, interm1, interm2);
        Value pixel_interm2 =
            arith::AddFOp::create(builder, loc, interm3, interm4);
        Value pixel_interm3 =
            arith::AddFOp::create(builder, loc, pixel_interm1, pixel_interm2);

        Value pixelVal = roundOff(builder, loc, pixel_interm3);

        memref::StoreOp::create(builder, 
            loc, pixelVal, output, ValueRange{resIndices[1], resIndices[0]});

        affine::AffineYieldOp::create(builder, loc);
      });
}

// Fills pixels in bilinear interpolation fashion.
void fillPixelsBilinearInterpolate4D(
    OpBuilder &builder, Location loc, Value ivs0, Value ivs1, Value resXVec,
    Value resYVec, Value xVec_L, Value yVec_L, Value xVec_H, Value yVec_H,
    Value input, Value output, Value c0, Value strideVal, Value xVecWeight,
    Value yVecWeight, Value outputRowLastElemF32, Value outputColLastElemF32,
    Value inputRowLastElemF32, Value inputColLastElemF32, Value c0F32,
    Value c1F32, Value dataCondition) {
  affine::AffineForOp::create(builder, 
      loc, ValueRange{c0}, builder.getDimIdentityMap(), ValueRange{strideVal},
      builder.getDimIdentityMap(), /*step*/ 1, ValueRange{},
      [&](OpBuilder &builder, Location loc, ValueRange ivs,
          ValueRange iterArg) {
        std::vector<Value> resIndices =
            extractIndices(builder, loc, resXVec, resYVec, ivs[0],
                           outputColLastElemF32, outputRowLastElemF32, c0F32);

        std::vector<Value> inputIndices_L =
            extractIndices(builder, loc, xVec_L, yVec_L, ivs[0],
                           inputColLastElemF32, inputRowLastElemF32, c0F32);
        std::vector<Value> inputIndices_H =
            extractIndices(builder, loc, xVec_H, yVec_H, ivs[0],
                           inputColLastElemF32, inputRowLastElemF32, c0F32);

        std::vector<Value> indexWeights;
        Value xPos_temp =
            vector::ExtractOp::create(builder, loc, xVecWeight, ivs[0]);
        Value yPos_temp =
            vector::ExtractOp::create(builder, loc, yVecWeight, ivs[0]);

        indexWeights.push_back(
            valBound(builder, loc, xPos_temp, inputColLastElemF32, c0F32));
        indexWeights.push_back(
            valBound(builder, loc, yPos_temp, inputRowLastElemF32, c0F32));

        std::vector<Value> indexWeights_UnitComplements = {
            arith::SubFOp::create(builder, loc, c1F32, indexWeights[0]),
            arith::SubFOp::create(builder, loc, c1F32, indexWeights[1])};

        auto ifop = scf::IfOp::create(builder, 
            loc, dataCondition,
            [&](OpBuilder &builder, Location loc) {
              Value pixelVal_a = memref::LoadOp::create(builder, 
                  loc, builder.getF32Type(), input,
                  ValueRange{ivs0, inputIndices_L[1], inputIndices_L[0], ivs1});
              Value pixelVal_b = memref::LoadOp::create(builder, 
                  loc, builder.getF32Type(), input,
                  ValueRange{ivs0, inputIndices_H[1], inputIndices_L[0], ivs1});
              Value pixelVal_c = memref::LoadOp::create(builder, 
                  loc, builder.getF32Type(), input,
                  ValueRange{ivs0, inputIndices_L[1], inputIndices_H[0], ivs1});
              Value pixelVal_d = memref::LoadOp::create(builder, 
                  loc, builder.getF32Type(), input,
                  ValueRange{ivs0, inputIndices_H[1], inputIndices_H[0], ivs1});
              scf::YieldOp::create(builder, 
                  loc,
                  ValueRange{pixelVal_a, pixelVal_b, pixelVal_c, pixelVal_d});
            },
            [&](OpBuilder &builder, Location loc) {
              Value pixelVal_a = memref::LoadOp::create(builder, 
                  loc, builder.getF32Type(), input,
                  ValueRange{ivs0, ivs1, inputIndices_L[1], inputIndices_L[0]});
              Value pixelVal_b = memref::LoadOp::create(builder, 
                  loc, builder.getF32Type(), input,
                  ValueRange{ivs0, ivs1, inputIndices_H[1], inputIndices_L[0]});
              Value pixelVal_c = memref::LoadOp::create(builder, 
                  loc, builder.getF32Type(), input,
                  ValueRange{ivs0, ivs1, inputIndices_L[1], inputIndices_H[0]});
              Value pixelVal_d = memref::LoadOp::create(builder, 
                  loc, builder.getF32Type(), input,
                  ValueRange{ivs0, ivs1, inputIndices_H[1], inputIndices_H[0]});
              scf::YieldOp::create(builder, 
                  loc,
                  ValueRange{pixelVal_a, pixelVal_b, pixelVal_c, pixelVal_d});
            });
        Value pixelVal_a = ifop.getResult(0);
        Value pixelVal_b = ifop.getResult(1);
        Value pixelVal_c = ifop.getResult(2);
        Value pixelVal_d = ifop.getResult(3);

        Value weightVal1 =
            arith::MulFOp::create(builder, loc, indexWeights_UnitComplements[0],
                                          indexWeights_UnitComplements[1]);
        Value weightVal2 = arith::MulFOp::create(builder, 
            loc, indexWeights[0], indexWeights_UnitComplements[1]);
        Value weightVal3 = arith::MulFOp::create(builder, 
            loc, indexWeights[1], indexWeights_UnitComplements[0]);
        Value weightVal4 = arith::MulFOp::create(builder, loc, indexWeights[0],
                                                         indexWeights[1]);

        Value interm1 =
            arith::MulFOp::create(builder, loc, pixelVal_a, weightVal1);
        Value interm2 =
            arith::MulFOp::create(builder, loc, pixelVal_b, weightVal2);
        Value interm3 =
            arith::MulFOp::create(builder, loc, pixelVal_c, weightVal3);
        Value interm4 =
            arith::MulFOp::create(builder, loc, pixelVal_d, weightVal4);

        Value pixel_interm1 =
            arith::AddFOp::create(builder, loc, interm1, interm2);
        Value pixel_interm2 =
            arith::AddFOp::create(builder, loc, interm3, interm4);
        Value pixelVal =
            arith::AddFOp::create(builder, loc, pixel_interm1, pixel_interm2);

        // Value pixelVal = roundOff(builder, loc, pixel_interm3);

        scf::IfOp::create(builder, 
            loc, dataCondition,
            [&](OpBuilder &builder, Location loc) {
              memref::StoreOp::create(builder, 
                  loc, pixelVal, output,
                  ValueRange{ivs0, resIndices[1], resIndices[0], ivs1});
              scf::YieldOp::create(builder, loc);
            },
            [&](OpBuilder &builder, Location loc) {
              memref::StoreOp::create(builder, 
                  loc, pixelVal, output,
                  ValueRange{ivs0, ivs1, resIndices[1], resIndices[0]});
              scf::YieldOp::create(builder, loc);
            });

        affine::AffineYieldOp::create(builder, loc);
      });
}

// Helper function for resizing an image using nearest neighbour interpolation
// mechanism.
void NearestNeighbourInterpolationResizing(
    OpBuilder &builder, Location loc, MLIRContext *ctx,
    SmallVector<Value, 8> lowerBounds, SmallVector<Value, 8> upperBounds,
    SmallVector<int64_t, 8> steps, Value strideVal, Value input, Value output,
    Value horizontalScalingFactorVec, Value verticalScalingFactorVec,
    Value outputRowLastElemF32, Value outputColLastElemF32,
    Value inputRowLastElemF32, Value inputColLastElemF32, VectorType vectorTy32,
    int64_t stride, Value c0, Value c0F32) {
  affine::buildAffineLoopNest(
      builder, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        Value ivs0F32 = indexToF32(builder, loc, ivs[0]);
        Value yVec = vector::BroadcastOp::create(builder, loc, vectorTy32, ivs0F32);
        Value xVec = iotaVec(builder, loc, ctx, ivs[1], strideVal, vectorTy32,
                             c0, stride);

        Value resXVecInterm = arith::MulFOp::create(builder, 
            loc, xVec, horizontalScalingFactorVec);
        Value resYVecInterm =
            arith::MulFOp::create(builder, loc, yVec, verticalScalingFactorVec);

        Value resXVec = roundOff(builder, loc, resXVecInterm);
        Value resYVec = roundOff(builder, loc, resYVecInterm);

        fillPixels(builder, loc, xVec, yVec, resXVec, resYVec, input, output,
                   c0, strideVal, outputRowLastElemF32, outputColLastElemF32,
                   inputRowLastElemF32, inputColLastElemF32, c0F32);
      });
}

// Helper function for resizing 4D an image using nearest neighbour
// interpolation mechanism.
void NearestNeighbourInterpolationResizing4D(
    OpBuilder &builder, Location loc, MLIRContext *ctx,
    SmallVector<Value, 8> lowerBounds, SmallVector<Value, 8> upperBounds,
    SmallVector<int64_t, 8> steps, Value strideVal, Value input, Value output,
    Value horizontalScalingFactorVec, Value verticalScalingFactorVec,
    Value outputRowLastElemF32, Value outputColLastElemF32,
    Value inputRowLastElemF32, Value inputColLastElemF32, VectorType vectorTy32,
    int64_t stride, Value c0, Value c0F32, Value dataCondition) {
  affine::buildAffineLoopNest(
      builder, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        Value ivs2F32 = indexToF32(builder, loc, ivs[2]);
        Value yVec = vector::BroadcastOp::create(builder, loc, vectorTy32, ivs2F32);
        Value xVec = iotaVec(builder, loc, ctx, ivs[3], strideVal, vectorTy32,
                             c0, stride);

        Value resXVecInterm =
            arith::MulFOp::create(builder, loc, xVec, verticalScalingFactorVec);
        Value resYVecInterm = arith::MulFOp::create(builder, 
            loc, yVec, horizontalScalingFactorVec);

        Value resXVec = roundOff(builder, loc, resXVecInterm);
        Value resYVec = roundOff(builder, loc, resYVecInterm);

        fillPixelsNearestNeighbour4D(
            builder, loc, ivs[0], ivs[1], xVec, yVec, resXVec, resYVec, input,
            output, c0, strideVal, outputRowLastElemF32, outputColLastElemF32,
            inputRowLastElemF32, inputColLastElemF32, c0F32, dataCondition);
      });
}

// Helper function for resizing an image using bilinear interpolation mechanism.
void BilinearInterpolationResizing(
    OpBuilder &builder, Location loc, MLIRContext *ctx,
    SmallVector<Value, 8> lowerBounds, SmallVector<Value, 8> upperBounds,
    SmallVector<int64_t, 8> steps, Value strideVal, Value input, Value output,
    Value horizontalScalingFactorVec, Value verticalScalingFactorVec,
    Value outputRowLastElemF32, Value outputColLastElemF32,
    Value inputRowLastElemF32, Value inputColLastElemF32, VectorType vectorTy32,
    int64_t stride, Value c0, Value c0F32, Value c1F32) {
  affine::buildAffineLoopNest(
      builder, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        Value ivs0F32 = indexToF32(builder, loc, ivs[0]);
        Value yVec = vector::BroadcastOp::create(builder, loc, vectorTy32, ivs0F32);
        Value xVec = iotaVec(builder, loc, ctx, ivs[1], strideVal, vectorTy32,
                             c0, stride);

        Value xVecInterm = arith::MulFOp::create(builder, 
            loc, xVec, horizontalScalingFactorVec);
        Value yVecInterm =
            arith::MulFOp::create(builder, loc, yVec, verticalScalingFactorVec);

        Value xVecInterm_L = math::FloorOp::create(builder, loc, xVecInterm);
        Value xVecInterm_H = math::CeilOp::create(builder, loc, xVecInterm);

        Value yVecInterm_L = math::FloorOp::create(builder, loc, yVecInterm);
        Value yVecInterm_H = math::CeilOp::create(builder, loc, yVecInterm);

        Value xVecWeight =
            arith::SubFOp::create(builder, loc, xVecInterm, xVecInterm_L);
        Value yVecWeight =
            arith::SubFOp::create(builder, loc, yVecInterm, yVecInterm_L);

        fillPixelsBilinearInterpolate(
            builder, loc, xVec, yVec, xVecInterm_L, yVecInterm_L, xVecInterm_H,
            yVecInterm_H, input, output, c0, strideVal, xVecWeight, yVecWeight,
            outputRowLastElemF32, outputColLastElemF32, inputRowLastElemF32,
            inputColLastElemF32, c0F32, c1F32);
      });
}

// Helper function for resizing 4D an image using bilinear interpolation
// mechanism.
void BilinearInterpolationResizing4D(
    OpBuilder &builder, Location loc, MLIRContext *ctx,
    SmallVector<Value, 8> lowerBounds, SmallVector<Value, 8> upperBounds,
    SmallVector<int64_t, 8> steps, Value strideVal, Value input, Value output,
    Value horizontalScalingFactorVec, Value verticalScalingFactorVec,
    Value outputRowLastElemF32, Value outputColLastElemF32,
    Value inputRowLastElemF32, Value inputColLastElemF32, VectorType vectorTy32,
    int64_t stride, Value c0, Value c0F32, Value c1F32, Value dataCondition) {
  affine::buildAffineLoopNest(
      builder, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        Value ivs0F32 = indexToF32(builder, loc, ivs[2]);
        Value yVec = vector::BroadcastOp::create(builder, loc, vectorTy32, ivs0F32);
        Value xVec = iotaVec(builder, loc, ctx, ivs[3], strideVal, vectorTy32,
                             c0, stride);

        Value xVecInterm =
            arith::MulFOp::create(builder, loc, xVec, verticalScalingFactorVec);
        Value yVecInterm = arith::MulFOp::create(builder, 
            loc, yVec, horizontalScalingFactorVec);

        Value xVecInterm_L = math::FloorOp::create(builder, loc, xVecInterm);
        Value xVecInterm_H = math::CeilOp::create(builder, loc, xVecInterm);

        Value yVecInterm_L = math::FloorOp::create(builder, loc, yVecInterm);
        Value yVecInterm_H = math::CeilOp::create(builder, loc, yVecInterm);

        Value xVecWeight =
            arith::SubFOp::create(builder, loc, xVecInterm, xVecInterm_L);
        Value yVecWeight =
            arith::SubFOp::create(builder, loc, yVecInterm, yVecInterm_L);

        fillPixelsBilinearInterpolate4D(
            builder, loc, ivs[0], ivs[1], xVec, yVec, xVecInterm_L,
            yVecInterm_L, xVecInterm_H, yVecInterm_H, input, output, c0,
            strideVal, xVecWeight, yVecWeight, outputRowLastElemF32,
            outputColLastElemF32, inputRowLastElemF32, inputColLastElemF32,
            c0F32, c1F32, dataCondition);
      });
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

// Util function for morphological transformations ; compares two vectors and
// returns a mask
Value createCompVecMorph(OpBuilder &builder, Location loc, VectorType type,
                         Value inputVec, Value outputVec, DIP_OP op) {
  Value compVec = {};
  auto elemTy = type.getElementType();
  auto bitWidth = elemTy.getIntOrFloatBitWidth();
  if (elemTy.isF32() || elemTy.isF64()) {
    if (op == DIP_OP::EROSION_2D) {
      compVec = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OGE,
                                              inputVec, outputVec);
    } else if (op == DIP_OP::DILATION_2D) {
      compVec = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OLE,
                                              inputVec, outputVec);
    }
  } else if (elemTy.isInteger(bitWidth)) {
    if (op == DIP_OP::EROSION_2D) {
      compVec = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                                              inputVec, outputVec);
    } else if (op == DIP_OP::DILATION_2D) {
      compVec = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sle,
                                              inputVec, outputVec);
    }
  }
  return compVec;
}

// Utility function for morphological operations ; cannot handle tail processing
void calcAndStorewoTailProcessingMorph(
    OpBuilder &builder, Location loc, VectorType vecType, Value inputVec,
    Value kernelVec, Value output, Value beginIdx, Value endIdx,
    Value zeroPadding, Value inputCol, VectorType vectorMaskTy, Type elemTy,
    Value kernelValue, Value zeroPaddingElem, DIP_OP op) {
  Value outputVec = vector::LoadOp::create(builder, 
      loc, vecType, output, ValueRange{beginIdx, endIdx});
  Value compVec = {};
  if (op == DIP_OP::EROSION_2D) {
    compVec = createCompVecMorph(builder, loc, vecType, inputVec, outputVec,
                                 DIP_OP::EROSION_2D);
  } else if (op == DIP_OP::DILATION_2D) {
    compVec = createCompVecMorph(builder, loc, vecType, inputVec, outputVec,
                                 DIP_OP::DILATION_2D);
  }

  Value resVec = vector::MaskedLoadOp::create(builder, 
      loc, vecType, output, ValueRange{beginIdx, endIdx}, compVec, inputVec);

  vector::StoreOp::create(builder, loc, resVec, output,
                                  ValueRange{beginIdx, endIdx});
}

// Utility function for morphological transformations, can handle tail
// processing
void calcAndStorewTailProcessingMorph(
    OpBuilder &builder, Location loc, VectorType vecType, Value inputVec,
    Value kernelVec, Value output, Value beginIdx, Value endIdx, Value tailCond,
    Value zeroPadding, Value inputCol, VectorType vectorMaskTy, Type elemTy,
    Value kernelValue, Value zeroPaddingElem, DIP_OP op) {
  scf::IfOp::create(builder, 
      loc, tailCond,
      [&](OpBuilder &builder, Location loc) {
        Value outputVec = vector::LoadOp::create(builder, 
            loc, vecType, output, ValueRange{beginIdx, endIdx});

        Value compVec = {};
        if (op == DIP_OP::EROSION_2D) {
          compVec = createCompVecMorph(builder, loc, vecType, inputVec,
                                       outputVec, DIP_OP::EROSION_2D);
        } else if (op == DIP_OP::DILATION_2D) {
          compVec = createCompVecMorph(builder, loc, vecType, inputVec,
                                       outputVec, DIP_OP::DILATION_2D);
        }

        Value resVec = vector::MaskedLoadOp::create(builder, 
            loc, vecType, output, ValueRange{beginIdx, endIdx}, compVec,
            inputVec);

        vector::StoreOp::create(builder, loc, resVec, output,
                                        ValueRange{beginIdx, endIdx});

        scf::YieldOp::create(builder, loc);
      },
      [&](OpBuilder &builder, Location loc) {
        Value extraElemMask =
            tailMaskCreator(builder, loc, inputCol, endIdx, vectorMaskTy);

        Value outputVec = vector::MaskedLoadOp::create(builder, 
            loc, vecType, output, ValueRange{beginIdx, endIdx}, extraElemMask,
            zeroPadding);

        Value compVec;
        if (op == DIP_OP::EROSION_2D) {
          compVec = createCompVecMorph(builder, loc, vecType, inputVec,
                                       outputVec, DIP_OP::EROSION_2D);
        } else if (op == DIP_OP::DILATION_2D) {
          compVec = createCompVecMorph(builder, loc, vecType, inputVec,
                                       outputVec, DIP_OP::DILATION_2D);
        }

        Value resVec = vector::MaskedLoadOp::create(builder, 
            loc, vecType, output, ValueRange{beginIdx, endIdx}, compVec,
            inputVec);

        vector::MaskedStoreOp::create(builder, 
            loc, output, ValueRange{beginIdx, endIdx}, extraElemMask, resVec);

        scf::YieldOp::create(builder, loc);
      });
}

void traverseImagewBoundaryExtrapolation(
    OpBuilder &rewriter, Location loc, MLIRContext *ctx, Value input,
    Value kernel, Value output, Value centerX, Value centerY,
    Value constantValue, Value strideVal, Type elemTy,
    buddy::dip::BoundaryOption boundaryOptionAttr, int64_t stride, DIP_OP op) {
  // Create constant indices.
  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);

  IntegerType i1 = IntegerType::get(ctx, 1);

  // Create DimOp.
  Value inputRow = memref::DimOp::create(rewriter, loc, input, c0);
  Value inputCol = memref::DimOp::create(rewriter, loc, input, c1);
  Value kernelRow = memref::DimOp::create(rewriter, loc, kernel, c0);
  Value kernelCol = memref::DimOp::create(rewriter, loc, kernel, c1);

  // Variables used for detecting rowMid, rowDown, colMid and colRight
  // regions.
  Value rowMidHelper = arith::AddIOp::create(rewriter, loc, inputRow, centerY);
  Value colMidHelper = arith::AddIOp::create(rewriter, loc, inputCol, centerX);

  SmallVector<Value, 8> lowerBounds(4, c0);
  SmallVector<Value, 8> uperBounds{inputRow, kernelRow, inputCol, kernelCol};
  SmallVector<int64_t, 8> steps{1, 1, stride, 1};

  VectorType vectorTy32 = VectorType::get({stride}, elemTy);
  VectorType vectorMaskTy = VectorType::get({stride}, i1);

  Value zeroPaddingElem = insertZeroConstantOp(ctx, rewriter, loc, elemTy);
  Value zeroPadding =
      vector::BroadcastOp::create(rewriter, loc, vectorTy32, zeroPaddingElem);

  AffineExpr a, b, c;
  bindDims(ctx, a, b, c);
  AffineMap calcHelper = AffineMap::get(3, 0, {a + b - c}, ctx);

  Value pseudoCol = affine::AffineApplyOp::create(rewriter, 
      loc, calcHelper, ValueRange{inputCol, kernelCol, c1});

  affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, uperBounds, steps,
      [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        // Indices of current pixel with respect to pseudo image containing
        // extrapolated boundaries.
        Value currRow = arith::AddIOp::create(builder, loc, ivs[0], ivs[1]);
        Value currCol = arith::AddIOp::create(builder, loc, ivs[2], ivs[3]);

        Value kernelValue = memref::LoadOp::create(builder, 
            loc, kernel, ValueRange{ivs[1], ivs[3]});
        Value kernelVec =
            vector::BroadcastOp::create(builder, loc, vectorTy32, kernelValue);

        // Pixel indices with respect to the actual image.
        Value imRow = arith::SubIOp::create(builder, loc, currRow, centerY);
        Value imCol = arith::SubIOp::create(builder, loc, currCol, centerX);

        // Index of pixel used for determining right region.
        Value colLastElem =
            arith::AddIOp::create(builder, loc, currCol, strideVal);

        Value rowUpCond = arith::CmpIOp::create(builder, 
            loc, arith::CmpIPredicate::slt, currRow, centerY);

        // Condition to check if the kernel value is a non-zero number.
        Value kernelNonZeroCond =
            zeroCond(builder, loc, elemTy, kernelValue, zeroPaddingElem);
        scf::IfOp::create(builder, 
            loc, kernelNonZeroCond, [&](OpBuilder &builder, Location loc) {
              scf::IfOp::create(builder, 
                  loc, rowUpCond,
                  [&](OpBuilder &builder, Location loc) {
                    // rowUp
                    if (boundaryOptionAttr ==
                        buddy::dip::BoundaryOption::ConstantPadding) {
                      Value inputVec = vector::BroadcastOp::create(builder, 
                          loc, vectorTy32, constantValue);
                      if (op == DIP_OP::CORRELATION_2D) {
                        calcAndStoreFMAwoTailProcessing(
                            builder, loc, vectorTy32, inputVec, kernelVec,
                            output, ivs[0], ivs[2]);
                      } else if (op == DIP_OP::DILATION_2D) {
                        Value tailCond =
                            tailChecker(builder, loc, calcHelper, strideVal,
                                        kernelCol, c1, pseudoCol, ivs[2]);

                        calcAndStorewTailProcessingMorph(
                            builder, loc, vectorTy32, inputVec, kernelVec,
                            output, ivs[0], ivs[2], tailCond, zeroPadding,
                            inputCol, vectorMaskTy, elemTy, kernelValue,
                            zeroPaddingElem, DIP_OP::DILATION_2D);
                      } else if (op == DIP_OP::EROSION_2D) {
                        Value tailCond =
                            tailChecker(builder, loc, calcHelper, strideVal,
                                        kernelCol, c1, pseudoCol, ivs[2]);

                        calcAndStorewTailProcessingMorph(
                            builder, loc, vectorTy32, inputVec, kernelVec,
                            output, ivs[0], ivs[2], tailCond, zeroPadding,
                            inputCol, vectorMaskTy, elemTy, kernelValue,
                            zeroPaddingElem, DIP_OP::EROSION_2D);
                      }
                    } else {
                      Value colLeftCond = arith::CmpIOp::create(builder, 
                          loc, arith::CmpIPredicate::slt, currCol, centerX);

                      scf::IfOp::create(builder, 
                          loc, colLeftCond,
                          [&](OpBuilder &builder, Location loc) {
                            // colLeft & rowUp
                            Value inputVec;
                            Value leftMaskElem = arith::SubIOp::create(builder, 
                                loc, centerX, currCol);
                            Value leftMask =
                                createInvertedMask(builder, loc, strideVal,
                                                   vectorMaskTy, leftMaskElem);

                            if (boundaryOptionAttr ==
                                buddy::dip::BoundaryOption::ReplicatePadding) {
                              Value paddingVal = memref::LoadOp::create(builder, 
                                  loc, input, ValueRange{c0, c0});
                              Value padding =
                                  vector::BroadcastOp::create(builder, 
                                      loc, vectorTy32, paddingVal);

                              Value leftPaddingOffset =
                                  arith::SubIOp::create(builder, loc, c0,
                                                                leftMaskElem);
                              inputVec = vector::MaskedLoadOp::create(builder, 
                                  loc, vectorTy32, input,
                                  ValueRange{c0, leftPaddingOffset}, leftMask,
                                  padding);
                            }

                            if (op == DIP_OP::CORRELATION_2D) {
                              calcAndStoreFMAwoTailProcessing(
                                  builder, loc, vectorTy32, inputVec, kernelVec,
                                  output, ivs[0], ivs[2]);
                            } else if (op == DIP_OP::EROSION_2D) {
                              calcAndStorewoTailProcessingMorph(
                                  builder, loc, vectorTy32, inputVec, kernelVec,
                                  output, ivs[0], ivs[2], zeroPadding, inputCol,
                                  vectorMaskTy, elemTy, kernelValue,
                                  zeroPaddingElem, DIP_OP::EROSION_2D);
                            } else if (op == DIP_OP::DILATION_2D) {
                              calcAndStorewoTailProcessingMorph(
                                  builder, loc, vectorTy32, inputVec, kernelVec,
                                  output, ivs[0], ivs[2], zeroPadding, inputCol,
                                  vectorMaskTy, elemTy, kernelValue,
                                  zeroPaddingElem, DIP_OP::DILATION_2D);
                            }

                            scf::YieldOp::create(builder, loc);
                          },
                          [&](OpBuilder &builder, Location loc) {
                            // (colMid or colRight) & rowUp
                            Value colMidCond = arith::CmpIOp::create(builder, 
                                loc, arith::CmpIPredicate::slt, colLastElem,
                                colMidHelper);

                            scf::IfOp::create(builder, 
                                loc, colMidCond,
                                [&](OpBuilder &builder, Location loc) {
                                  // colMid & rowUp
                                  Value inputVec;
                                  if (boundaryOptionAttr ==
                                      buddy::dip::BoundaryOption::
                                          ReplicatePadding) {
                                    inputVec = vector::LoadOp::create(builder, 
                                        loc, vectorTy32, input,
                                        ValueRange{c0, imCol});
                                  }

                                  if (op == DIP_OP::CORRELATION_2D) {
                                    calcAndStoreFMAwoTailProcessing(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2]);
                                  } else if (op == DIP_OP::EROSION_2D) {
                                    calcAndStorewoTailProcessingMorph(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        zeroPadding, inputCol, vectorMaskTy,
                                        elemTy, kernelValue, zeroPaddingElem,
                                        DIP_OP::EROSION_2D);
                                  } else if (op == DIP_OP::DILATION_2D) {
                                    calcAndStorewoTailProcessingMorph(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        zeroPadding, inputCol, vectorMaskTy,
                                        elemTy, kernelValue, zeroPaddingElem,
                                        DIP_OP::DILATION_2D);
                                  }
                                  scf::YieldOp::create(builder, loc);
                                },
                                [&](OpBuilder &builder, Location loc) {
                                  // colRight & rowUp
                                  Value inputVec;
                                  Value rightMaskHelper =
                                      arith::SubIOp::create(builder, 
                                          loc, colLastElem, colMidHelper);
                                  Value rightMaskElem =
                                      arith::SubIOp::create(builder, 
                                          loc, strideVal, rightMaskHelper);
                                  Value rightMask =
                                      vector::CreateMaskOp::create(builder, 
                                          loc, vectorMaskTy, rightMaskElem);

                                  if (boundaryOptionAttr ==
                                      buddy::dip::BoundaryOption::
                                          ReplicatePadding) {
                                    Value rightRange =
                                        arith::SubIOp::create(builder, 
                                            loc, inputCol, c1);
                                    Value paddingVal =
                                        memref::LoadOp::create(builder, 
                                            loc, input,
                                            ValueRange{c0, rightRange});
                                    Value padding =
                                        vector::BroadcastOp::create(builder, 
                                            loc, vectorTy32, paddingVal);

                                    inputVec =
                                        vector::MaskedLoadOp::create(builder, 
                                            loc, vectorTy32, input,
                                            ValueRange{c0, imCol}, rightMask,
                                            padding);
                                  }
                                  Value tailCond = tailChecker(
                                      builder, loc, calcHelper, strideVal,
                                      kernelCol, c1, pseudoCol, ivs[2]);

                                  if (op == DIP_OP::CORRELATION_2D) {
                                    calcAndStoreFMAwTailProcessing(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, zeroPadding, inputCol,
                                        vectorMaskTy);
                                  } else if (op == DIP_OP::DILATION_2D) {
                                    calcAndStorewTailProcessingMorph(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, zeroPadding, inputCol,
                                        vectorMaskTy, elemTy, kernelValue,
                                        zeroPaddingElem, DIP_OP::DILATION_2D);
                                  } else if (op == DIP_OP::EROSION_2D) {
                                    calcAndStorewTailProcessingMorph(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, zeroPadding, inputCol,
                                        vectorMaskTy, elemTy, kernelValue,
                                        zeroPaddingElem, DIP_OP::EROSION_2D);
                                  }

                                  scf::YieldOp::create(builder, loc);
                                });
                            scf::YieldOp::create(builder, loc);
                          });
                    }
                    scf::YieldOp::create(builder, loc);
                  },
                  [&](OpBuilder &builder, Location loc) {
                    // rowMid or rowDown
                    Value rowMidCond = arith::CmpIOp::create(builder, 
                        loc, arith::CmpIPredicate::slt, currRow, rowMidHelper);

                    scf::IfOp::create(builder, 
                        loc, rowMidCond,
                        [&](OpBuilder &builder, Location loc) {
                          // rowMid
                          Value colLeftCond = arith::CmpIOp::create(builder, 
                              loc, arith::CmpIPredicate::slt, currCol, centerX);

                          scf::IfOp::create(builder, 
                              loc, colLeftCond,
                              [&](OpBuilder &builder, Location loc) {
                                // colLeft & rowMid
                                Value inputVec;
                                Value leftMaskElem =
                                    arith::SubIOp::create(builder, loc, centerX,
                                                                  currCol);
                                Value leftMask = createInvertedMask(
                                    builder, loc, strideVal, vectorMaskTy,
                                    leftMaskElem);

                                if (boundaryOptionAttr ==
                                    buddy::dip::BoundaryOption::
                                        ConstantPadding) {
                                  Value padding =
                                      vector::BroadcastOp::create(builder, 
                                          loc, vectorTy32, constantValue);

                                  Value leftPaddingOffset =
                                      arith::SubIOp::create(builder, 
                                          loc, c0, leftMaskElem);
                                  inputVec =
                                      vector::MaskedLoadOp::create(builder, 
                                          loc, vectorTy32, input,
                                          ValueRange{imRow, leftPaddingOffset},
                                          leftMask, padding);
                                } else if (boundaryOptionAttr ==
                                           buddy::dip::BoundaryOption::
                                               ReplicatePadding) {
                                  Value paddingVal =
                                      memref::LoadOp::create(builder, 
                                          loc, input, ValueRange{imRow, c0});
                                  Value padding =
                                      vector::BroadcastOp::create(builder, 
                                          loc, vectorTy32, paddingVal);

                                  Value leftPaddingOffset =
                                      arith::SubIOp::create(builder, 
                                          loc, c0, leftMaskElem);
                                  inputVec =
                                      vector::MaskedLoadOp::create(builder, 
                                          loc, vectorTy32, input,
                                          ValueRange{imRow, leftPaddingOffset},
                                          leftMask, padding);
                                }

                                if (op == DIP_OP::CORRELATION_2D) {
                                  calcAndStoreFMAwoTailProcessing(
                                      builder, loc, vectorTy32, inputVec,
                                      kernelVec, output, ivs[0], ivs[2]);
                                } else if (op == DIP_OP::EROSION_2D) {
                                  calcAndStorewoTailProcessingMorph(
                                      builder, loc, vectorTy32, inputVec,
                                      kernelVec, output, ivs[0], ivs[2],
                                      zeroPadding, inputCol, vectorMaskTy,
                                      elemTy, kernelValue, zeroPaddingElem,
                                      DIP_OP::EROSION_2D);
                                } else if (op == DIP_OP::DILATION_2D) {
                                  calcAndStorewoTailProcessingMorph(
                                      builder, loc, vectorTy32, inputVec,
                                      kernelVec, output, ivs[0], ivs[2],
                                      zeroPadding, inputCol, vectorMaskTy,
                                      elemTy, kernelValue, zeroPaddingElem,
                                      DIP_OP::DILATION_2D);
                                }

                                scf::YieldOp::create(builder, loc);
                              },
                              [&](OpBuilder &builder, Location loc) {
                                // (colMid or colRight) & rowMid
                                Value colMidCond =
                                    arith::CmpIOp::create(builder, 
                                        loc, arith::CmpIPredicate::slt,
                                        colLastElem, colMidHelper);

                                scf::IfOp::create(builder, 
                                    loc, colMidCond,
                                    [&](OpBuilder &builder, Location loc) {
                                      // colMid & rowMid
                                      Value inputVec =
                                          vector::LoadOp::create(builder, 
                                              loc, vectorTy32, input,
                                              ValueRange{imRow, imCol});

                                      if (op == DIP_OP::CORRELATION_2D) {
                                        calcAndStoreFMAwoTailProcessing(
                                            builder, loc, vectorTy32, inputVec,
                                            kernelVec, output, ivs[0], ivs[2]);
                                      } else if (op == DIP_OP::EROSION_2D) {
                                        calcAndStorewoTailProcessingMorph(
                                            builder, loc, vectorTy32, inputVec,
                                            kernelVec, output, ivs[0], ivs[2],
                                            zeroPadding, inputCol, vectorMaskTy,
                                            elemTy, kernelValue,
                                            zeroPaddingElem,
                                            DIP_OP::EROSION_2D);
                                      } else if (op == DIP_OP::DILATION_2D) {
                                        calcAndStorewoTailProcessingMorph(
                                            builder, loc, vectorTy32, inputVec,
                                            kernelVec, output, ivs[0], ivs[2],
                                            zeroPadding, inputCol, vectorMaskTy,
                                            elemTy, kernelValue,
                                            zeroPaddingElem,
                                            DIP_OP::DILATION_2D);
                                      }

                                      scf::YieldOp::create(builder, loc);
                                    },
                                    [&](OpBuilder &builder, Location loc) {
                                      // colRight & rowMid
                                      Value inputVec;
                                      Value rightMaskHelper =
                                          arith::SubIOp::create(builder, 
                                              loc, colLastElem, colMidHelper);
                                      Value rightMaskElem =
                                          arith::SubIOp::create(builder, 
                                              loc, strideVal, rightMaskHelper);
                                      Value rightMask =
                                          vector::CreateMaskOp::create(builder, 
                                              loc, vectorMaskTy, rightMaskElem);

                                      if (boundaryOptionAttr ==
                                          buddy::dip::BoundaryOption::
                                              ConstantPadding) {
                                        Value padding =
                                            vector::BroadcastOp::create(builder, 
                                                loc, vectorTy32, constantValue);

                                        inputVec =
                                            vector::MaskedLoadOp::create(
                                                builder, 
                                                    loc, vectorTy32, input,
                                                    ValueRange{imRow, imCol},
                                                    rightMask, padding);
                                      } else if (boundaryOptionAttr ==
                                                 buddy::dip::BoundaryOption::
                                                     ReplicatePadding) {
                                        Value rightRange =
                                            arith::SubIOp::create(builder, 
                                                loc, inputCol, c1);
                                        Value paddingVal =
                                            memref::LoadOp::create(builder, 
                                                loc, input,
                                                ValueRange{imRow, rightRange});
                                        Value padding =
                                            vector::BroadcastOp::create(builder, 
                                                loc, vectorTy32, paddingVal);

                                        inputVec =
                                            vector::MaskedLoadOp::create(
                                                builder, 
                                                    loc, vectorTy32, input,
                                                    ValueRange{imRow, imCol},
                                                    rightMask, padding);
                                      }
                                      Value tailCond = tailChecker(
                                          builder, loc, calcHelper, strideVal,
                                          kernelCol, c1, pseudoCol, ivs[2]);

                                      if (op == DIP_OP::CORRELATION_2D) {
                                        calcAndStoreFMAwTailProcessing(
                                            builder, loc, vectorTy32, inputVec,
                                            kernelVec, output, ivs[0], ivs[2],
                                            tailCond, zeroPadding, inputCol,
                                            vectorMaskTy);
                                      } else if (op == DIP_OP::DILATION_2D) {
                                        calcAndStorewTailProcessingMorph(
                                            builder, loc, vectorTy32, inputVec,
                                            kernelVec, output, ivs[0], ivs[2],
                                            tailCond, zeroPadding, inputCol,
                                            vectorMaskTy, elemTy, kernelValue,
                                            zeroPaddingElem,
                                            DIP_OP::DILATION_2D);
                                      } else if (op == DIP_OP::EROSION_2D) {
                                        calcAndStorewTailProcessingMorph(
                                            builder, loc, vectorTy32, inputVec,
                                            kernelVec, output, ivs[0], ivs[2],
                                            tailCond, zeroPadding, inputCol,
                                            vectorMaskTy, elemTy, kernelValue,
                                            zeroPaddingElem,
                                            DIP_OP::EROSION_2D);
                                      }
                                      scf::YieldOp::create(builder, loc);
                                    });
                                scf::YieldOp::create(builder, loc);
                              });
                          scf::YieldOp::create(builder, loc);
                        },
                        [&](OpBuilder &builder, Location loc) {
                          // rowDown
                          if (boundaryOptionAttr ==
                              buddy::dip::BoundaryOption::ConstantPadding) {
                            Value inputVec =
                                vector::BroadcastOp::create(builder, 
                                    loc, vectorTy32, constantValue);

                            if (op == DIP_OP::CORRELATION_2D) {
                              calcAndStoreFMAwoTailProcessing(
                                  builder, loc, vectorTy32, inputVec, kernelVec,
                                  output, ivs[0], ivs[2]);
                            } else if (op == DIP_OP::EROSION_2D) {
                              calcAndStorewoTailProcessingMorph(
                                  builder, loc, vectorTy32, inputVec, kernelVec,
                                  output, ivs[0], ivs[2], zeroPadding, inputCol,
                                  vectorMaskTy, elemTy, kernelValue,
                                  zeroPaddingElem, DIP_OP::EROSION_2D);
                            } else if (op == DIP_OP::DILATION_2D) {
                              calcAndStorewoTailProcessingMorph(
                                  builder, loc, vectorTy32, inputVec, kernelVec,
                                  output, ivs[0], ivs[2], zeroPadding, inputCol,
                                  vectorMaskTy, elemTy, kernelValue,
                                  zeroPaddingElem, DIP_OP::DILATION_2D);
                            }
                          } else {
                            Value colLeftCond = arith::CmpIOp::create(builder, 
                                loc, arith::CmpIPredicate::slt, currCol,
                                centerX);

                            scf::IfOp::create(builder, 
                                loc, colLeftCond,
                                [&](OpBuilder &builder, Location loc) {
                                  // colLeft & rowDown
                                  Value inputVec;
                                  Value downRange =
                                      arith::SubIOp::create(builder, 
                                          loc, inputRow, c1);
                                  Value leftMaskElem =
                                      arith::SubIOp::create(builder, 
                                          loc, centerX, currCol);
                                  Value leftMask = createInvertedMask(
                                      builder, loc, strideVal, vectorMaskTy,
                                      leftMaskElem);

                                  if (boundaryOptionAttr ==
                                      buddy::dip::BoundaryOption::
                                          ReplicatePadding) {
                                    Value paddingVal =
                                        memref::LoadOp::create(builder, 
                                            loc, input,
                                            ValueRange{downRange, c0});
                                    Value padding =
                                        vector::BroadcastOp::create(builder, 
                                            loc, vectorTy32, paddingVal);

                                    Value leftPaddingOffset =
                                        arith::SubIOp::create(builder, 
                                            loc, c0, leftMaskElem);
                                    inputVec =
                                        vector::MaskedLoadOp::create(builder, 
                                            loc, vectorTy32, input,
                                            ValueRange{downRange,
                                                       leftPaddingOffset},
                                            leftMask, padding);
                                  }

                                  if (op == DIP_OP::CORRELATION_2D) {
                                    calcAndStoreFMAwoTailProcessing(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2]);
                                  } else if (op == DIP_OP::EROSION_2D) {
                                    calcAndStorewoTailProcessingMorph(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        zeroPadding, inputCol, vectorMaskTy,
                                        elemTy, kernelValue, zeroPaddingElem,
                                        DIP_OP::EROSION_2D);
                                  } else if (op == DIP_OP::DILATION_2D) {
                                    calcAndStorewoTailProcessingMorph(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        zeroPadding, inputCol, vectorMaskTy,
                                        elemTy, kernelValue, zeroPaddingElem,
                                        DIP_OP::DILATION_2D);
                                  }

                                  scf::YieldOp::create(builder, loc);
                                },
                                [&](OpBuilder &builder, Location loc) {
                                  // (colMid or colRight) & rowDown
                                  Value colMidCond =
                                      arith::CmpIOp::create(builder, 
                                          loc, arith::CmpIPredicate::slt,
                                          colLastElem, colMidHelper);

                                  scf::IfOp::create(builder, 
                                      loc, colMidCond,
                                      [&](OpBuilder &builder, Location loc) {
                                        // colMid & rowDown
                                        Value inputVec;
                                        Value downRange =
                                            arith::SubIOp::create(builder, 
                                                loc, inputRow, c1);
                                        if (boundaryOptionAttr ==
                                            buddy::dip::BoundaryOption::
                                                ReplicatePadding) {
                                          inputVec =
                                              vector::LoadOp::create(builder, 
                                                  loc, vectorTy32, input,
                                                  ValueRange{downRange, imCol});
                                        }

                                        if (op == DIP_OP::CORRELATION_2D) {
                                          calcAndStoreFMAwoTailProcessing(
                                              builder, loc, vectorTy32,
                                              inputVec, kernelVec, output,
                                              ivs[0], ivs[2]);
                                        } else if (op == DIP_OP::EROSION_2D) {
                                          calcAndStorewoTailProcessingMorph(
                                              builder, loc, vectorTy32,
                                              inputVec, kernelVec, output,
                                              ivs[0], ivs[2], zeroPadding,
                                              inputCol, vectorMaskTy, elemTy,
                                              kernelValue, zeroPaddingElem,
                                              DIP_OP::EROSION_2D);
                                        } else if (op == DIP_OP::DILATION_2D) {
                                          calcAndStorewoTailProcessingMorph(
                                              builder, loc, vectorTy32,
                                              inputVec, kernelVec, output,
                                              ivs[0], ivs[2], zeroPadding,
                                              inputCol, vectorMaskTy, elemTy,
                                              kernelValue, zeroPaddingElem,
                                              DIP_OP::DILATION_2D);
                                        }

                                        scf::YieldOp::create(builder, loc);
                                      },
                                      [&](OpBuilder &builder, Location loc) {
                                        // colRight & rowDown
                                        Value inputVec;
                                        Value rightMaskHelper =
                                            arith::SubIOp::create(builder, 
                                                loc, colLastElem, colMidHelper);
                                        Value rightMaskElem =
                                            arith::SubIOp::create(builder, 
                                                loc, strideVal,
                                                rightMaskHelper);
                                        Value rightMask =
                                            vector::CreateMaskOp::create(
                                                builder, 
                                                    loc, vectorMaskTy,
                                                    rightMaskElem);

                                        Value downRange =
                                            arith::SubIOp::create(builder, 
                                                loc, inputRow, c1);
                                        Value rightRange =
                                            arith::SubIOp::create(builder, 
                                                loc, inputCol, c1);

                                        if (boundaryOptionAttr ==
                                            buddy::dip::BoundaryOption::
                                                ReplicatePadding) {

                                          Value paddingVal =
                                              memref::LoadOp::create(builder, 
                                                  loc, input,
                                                  ValueRange{downRange,
                                                             rightRange});
                                          Value padding =
                                              vector::BroadcastOp::create(
                                                  builder, 
                                                      loc, vectorTy32,
                                                      paddingVal);

                                          inputVec =
                                              vector::MaskedLoadOp::create(
                                                  builder, 
                                                      loc, vectorTy32, input,
                                                      ValueRange{downRange,
                                                                 imCol},
                                                      rightMask, padding);
                                        }
                                        Value tailCond = tailChecker(
                                            builder, loc, calcHelper, strideVal,
                                            kernelCol, c1, pseudoCol, ivs[2]);

                                        if (op == DIP_OP::CORRELATION_2D) {
                                          calcAndStoreFMAwTailProcessing(
                                              builder, loc, vectorTy32,
                                              inputVec, kernelVec, output,
                                              ivs[0], ivs[2], tailCond,
                                              zeroPadding, inputCol,
                                              vectorMaskTy);
                                        } else if (op == DIP_OP::DILATION_2D) {
                                          calcAndStorewTailProcessingMorph(
                                              builder, loc, vectorTy32,
                                              inputVec, kernelVec, output,
                                              ivs[0], ivs[2], tailCond,
                                              zeroPadding, inputCol,
                                              vectorMaskTy, elemTy, kernelValue,
                                              zeroPaddingElem,
                                              DIP_OP::DILATION_2D);
                                        } else if (op == DIP_OP::EROSION_2D) {
                                          calcAndStorewTailProcessingMorph(
                                              builder, loc, vectorTy32,
                                              inputVec, kernelVec, output,
                                              ivs[0], ivs[2], tailCond,
                                              zeroPadding, inputCol,
                                              vectorMaskTy, elemTy, kernelValue,
                                              zeroPaddingElem,
                                              DIP_OP::EROSION_2D);
                                        }
                                        scf::YieldOp::create(builder, loc);
                                      });
                                  scf::YieldOp::create(builder, loc);
                                });
                          }
                          scf::YieldOp::create(builder, loc);
                        });
                    scf::YieldOp::create(builder, loc);
                  });

              scf::YieldOp::create(builder, loc);
            });
      });
}

} // namespace dip
} // namespace buddy

#endif // UTILS_DIPUTILS_DEF
