//====- DIPUtils.h --------------------------------------------------------===//
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
// This file defines DIP dialect specific utility functions for the buddy
// compiler ecosystem.
//
//===----------------------------------------------------------------------===//

#ifndef INCLUDE_UTILS_DIPUTILS_H
#define INCLUDE_UTILS_DIPUTILS_H

#include "Utils/Utils.h"

// Specify operation names which will be used for performing operation specific
// tasks inside generic utility functions.
enum class DIP_OP { CORRELATION_2D };

// Inserts a constant op with value 0 into a location `loc` based on type
// `type`. Supported types are : f32, f64, integer types
Value insertZeroConstantOp(MLIRContext *ctx, OpBuilder &builder, Location loc,
                           Type elemTy) {
  Value op = {};
  auto bitWidth = elemTy.getIntOrFloatBitWidth();
  if (elemTy.isF32() || elemTy.isF64()) {
    FloatType type =
        elemTy.isF32() ? FloatType::getF32(ctx) : FloatType::getF64(ctx);
    auto zero = APFloat::getZero(type.getFloatSemantics());
    op = builder.create<ConstantFloatOp>(loc, zero, type);
  } else if (elemTy.isInteger(bitWidth)) {
    IntegerType type = IntegerType::get(ctx, bitWidth);
    op = builder.create<ConstantIntOp>(loc, 0, type);
  }

  return op;
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
    res = builder.create<vector::FMAOp>(loc, inputVec, kernelVec, outputVec);
  } else if (elemTy.isInteger(bitWidth)) {
    Value mul = builder.create<arith::MulIOp>(loc, inputVec, kernelVec);
    res = builder.create<arith::AddIOp>(loc, mul, outputVec);
  }

  return res;
}

// Calculate result of FMA and store it in output memref. This function cannot
// handle tail processing.
void calcAndStoreFMAwoTailProcessing(OpBuilder &builder, Location loc,
                                     VectorType vecType, Value inputVec,
                                     Value kernelVec, Value output,
                                     Value beginIdx, Value endIdx) {
  Value outputVec = builder.create<LoadOp>(loc, vecType, output,
                                           ValueRange{beginIdx, endIdx});
  Value resVec =
      insertFMAOp(builder, loc, vecType, inputVec, kernelVec, outputVec);
  builder.create<StoreOp>(loc, resVec, output, ValueRange{beginIdx, endIdx});
}

// Checks if we encountered a tail (columns remaining after processing in
// batches of stride size).
Value tailChecker(OpBuilder &builder, Location loc, AffineMap calcHelper,
                  Value strideVal, Value kernelSize, Value c1, Value pseudoCol,
                  Value colPivot) {
  Value tailChecker = builder.create<AffineApplyOp>(
      loc, calcHelper, ValueRange{strideVal, kernelSize, c1});
  Value colEndDistance = builder.create<SubIOp>(loc, pseudoCol, colPivot);
  Value tailCond = builder.create<CmpIOp>(loc, CmpIPredicate::sge,
                                          colEndDistance, tailChecker);
  return tailCond;
}

// Creates the required mask which is to be used for tail processing.
Value tailMaskCreator(OpBuilder &builder, Location loc, Value inputCol,
                      Value colPivot, VectorType vectorMaskTy) {
  Value extraElemCount = builder.create<SubIOp>(loc, inputCol, colPivot);
  Value tailMask =
      builder.create<CreateMaskOp>(loc, vectorMaskTy, extraElemCount);
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
  builder.create<scf::IfOp>(
      loc, tailCond,
      [&](OpBuilder &builder, Location loc) {
        Value outputVec = builder.create<LoadOp>(loc, vecType, output,
                                                 ValueRange{beginIdx, endIdx});
        Value resVec =
            insertFMAOp(builder, loc, vecType, inputVec, kernelVec, outputVec);
        builder.create<StoreOp>(loc, resVec, output,
                                ValueRange{beginIdx, endIdx});

        builder.create<scf::YieldOp>(loc);
      },
      [&](OpBuilder &builder, Location loc) {
        Value extraElemMask =
            tailMaskCreator(builder, loc, inputCol, endIdx, vectorMaskTy);
        Value outputVec = builder.create<MaskedLoadOp>(
            loc, vecType, output, ValueRange{beginIdx, endIdx}, extraElemMask,
            zeroPadding);
        Value resVec =
            insertFMAOp(builder, loc, vecType, inputVec, kernelVec, outputVec);
        builder.create<MaskedStoreOp>(loc, output, ValueRange{beginIdx, endIdx},
                                      extraElemMask, resVec);

        builder.create<scf::YieldOp>(loc);
      });
}

// Apply 3 shear method and return mapped values.
std::vector<Value> shearTransform(OpBuilder &builder, Location loc,
                                  Value originalX, Value originalY,
                                  Value sinVec, Value tanVec) {
  Value yTan1 = builder.create<arith::MulFOp>(loc, tanVec, originalY);
  Value xIntermediate1 = builder.create<arith::SubFOp>(loc, originalX, yTan1);
  Value xIntermediate = roundOff(builder, loc, xIntermediate1);

  Value xSin = builder.create<arith::MulFOp>(loc, xIntermediate, sinVec);
  Value newY1 = builder.create<arith::AddFOp>(loc, xSin, originalY);
  Value newY = roundOff(builder, loc, newY1);

  Value yTan2 = builder.create<arith::MulFOp>(loc, newY, tanVec);
  Value newX1 = builder.create<arith::SubFOp>(loc, xIntermediate, yTan2);
  Value newX = roundOff(builder, loc, newX1);

  return {newY, newX};
}

// Apply standard rotation matrix transformation and return mapped values.
std::vector<Value> standardRotate(OpBuilder &builder, Location loc,
                                  Value originalX, Value originalY,
                                  Value sinVec, Value cosVec) {
  Value ySin = builder.create<arith::MulFOp>(loc, originalY, sinVec);
  Value yCos = builder.create<arith::MulFOp>(loc, originalY, cosVec);

  Value xSin = builder.create<arith::MulFOp>(loc, originalX, sinVec);
  Value xCos = builder.create<arith::MulFOp>(loc, originalX, cosVec);

  Value newY1 = builder.create<arith::SubFOp>(loc, yCos, xSin);
  Value newX1 = builder.create<arith::AddFOp>(loc, ySin, xCos);

  return {roundOff(builder, loc, newY1), roundOff(builder, loc, newX1)};
}

// Get center co-ordinates w.r.t given dimension.
Value getCenter(OpBuilder &builder, Location loc, MLIRContext *ctx, Value dim) {
  Value dimF32 = indexToF32(builder, loc, dim);
  Value c1f = builder.create<ConstantFloatOp>(loc, (llvm::APFloat)1.0f,
                                              builder.getF32Type());
  Value c2f = builder.create<ConstantFloatOp>(loc, (llvm::APFloat)2.0f,
                                              builder.getF32Type());

  Value temp1 = builder.create<arith::AddFOp>(loc, dimF32, c1f);
  Value temp2 = builder.create<arith::DivFOp>(loc, temp1, c2f);
  Value center = builder.create<arith::SubFOp>(loc, temp2, c1f);
  Value centerRound = roundOff(builder, loc, center);

  return F32ToIndex(builder, loc, centerRound);
}

// Scale pixel co-ordinates appropriately before calculating their rotated
// position(s).
Value pixelScaling(OpBuilder &builder, Location loc, Value imageDImF32Vec,
                   Value coordVec, Value imageCenterF32Vec, Value c1F32Vec) {
  Value interm1 = builder.create<arith::SubFOp>(loc, imageDImF32Vec, coordVec);
  Value interm2 =
      builder.create<arith::SubFOp>(loc, interm1, imageCenterF32Vec);

  return builder.create<arith::SubFOp>(loc, interm2, c1F32Vec);
}

// Extract values present at a particular index in two vectors for using
// those values to load an element from a memref.
std::vector<Value> extractIndices(OpBuilder &builder, Location loc, Value xVec,
                                  Value yVec, Value vecIndex, Value xUpperBound,
                                  Value yUpperBound, Value c0F32) {
  Value xPos = builder.create<vector::ExtractElementOp>(loc, xVec, vecIndex);
  Value yPos = builder.create<vector::ExtractElementOp>(loc, yVec, vecIndex);

  Value xPosBound = valBound(builder, loc, xPos, xUpperBound, c0F32);
  Value yPosBound = valBound(builder, loc, yPos, yUpperBound, c0F32);

  return {F32ToIndex(builder, loc, xPosBound),
          F32ToIndex(builder, loc, yPosBound)};
}

// Fill appropriate pixel data in its corresponding rotated co-ordinate of
// output image.
void fillPixels(OpBuilder &builder, Location loc, Value resXVec, Value resYVec,
                Value xVec, Value yVec, Value input, Value output, Value c0,
                Value strideVal, Value outputRowLastElemF32,
                Value outputColLastElemF32, Value inputRowLastElemF32,
                Value inputColLastElemF32, Value c0F32) {
  builder.create<AffineForOp>(
      loc, ValueRange{c0}, builder.getDimIdentityMap(), ValueRange{strideVal},
      builder.getDimIdentityMap(), /*step*/ 1, llvm::None,
      [&](OpBuilder &builder, Location loc, ValueRange ivs,
          ValueRange iterArg) {
        std::vector<Value> origIndices =
            extractIndices(builder, loc, xVec, yVec, ivs[0],
                           inputRowLastElemF32, inputColLastElemF32, c0F32);
        std::vector<Value> resIndices =
            extractIndices(builder, loc, resXVec, resYVec, ivs[0],
                           outputRowLastElemF32, outputColLastElemF32, c0F32);

        Value pixelVal = builder.create<memref::LoadOp>(
            loc, builder.getF32Type(), input,
            ValueRange{origIndices[0], origIndices[1]});
        builder.create<memref::StoreOp>(
            loc, pixelVal, output, ValueRange{resIndices[0], resIndices[1]});

        builder.create<AffineYieldOp>(loc);
      });
}

// Calculate tan(angle / 2) where angle is a function parameter.
Value customTanVal(OpBuilder &builder, Location loc, Value angleVal) {
  Value c2F32 = builder.create<ConstantFloatOp>(loc, (llvm::APFloat)2.0f,
                                                builder.getF32Type());
  Value angleVal_2 = builder.create<arith::DivFOp>(loc, angleVal, c2F32);

  Value sinVal = builder.create<math::SinOp>(loc, angleVal_2);
  Value cosVal = builder.create<math::CosOp>(loc, angleVal_2);

  return builder.create<arith::DivFOp>(loc, sinVal, cosVal);
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
  buildAffineLoopNest(
      builder, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        Value ivs0F32 = indexToF32(builder, loc, ivs[0]);
        Value yVec = builder.create<vector::SplatOp>(loc, vectorTy32, ivs0F32);
        Value xVec = iotaVec(builder, loc, ctx, ivs[1], strideVal, vectorTy32,
                             c0, stride);

        Value yVecModified = pixelScaling(builder, loc, inputRowF32Vec, yVec,
                                          inputCenterYF32Vec, c1F32Vec);
        Value xVecModified = pixelScaling(builder, loc, inputColF32Vec, xVec,
                                          inputCenterXF32Vec, c1F32Vec);

        std::vector<Value> resIndices = shearTransform(
            builder, loc, xVecModified, yVecModified, sinVec, tanVec);
        Value resYVec = builder.create<arith::SubFOp>(loc, outputCenterYF32Vec,
                                                      resIndices[0]);
        Value resXVec = builder.create<arith::SubFOp>(loc, outputCenterXF32Vec,
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
  Value cosVal = builder.create<math::CosOp>(loc, angleVal);
  Value cosVec = builder.create<vector::BroadcastOp>(loc, vectorTy32, cosVal);

  buildAffineLoopNest(
      builder, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        Value ivs0F32 = indexToF32(builder, loc, ivs[0]);
        Value yVec = builder.create<vector::SplatOp>(loc, vectorTy32, ivs0F32);
        Value xVec = iotaVec(builder, loc, ctx, ivs[1], strideVal, vectorTy32,
                             c0, stride);

        Value yVecModified = pixelScaling(builder, loc, inputRowF32Vec, yVec,
                                          inputCenterYF32Vec, c1F32Vec);
        Value xVecModified = pixelScaling(builder, loc, inputColF32Vec, xVec,
                                          inputCenterXF32Vec, c1F32Vec);

        std::vector<Value> resIndices = standardRotate(
            builder, loc, xVecModified, yVecModified, sinVec, cosVec);
        Value resYVec = builder.create<arith::SubFOp>(loc, outputCenterYF32Vec,
                                                      resIndices[0]);
        Value resXVec = builder.create<arith::SubFOp>(loc, outputCenterXF32Vec,
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
  builder.create<AffineForOp>(
      loc, ValueRange{c0}, builder.getDimIdentityMap(), ValueRange{strideVal},
      builder.getDimIdentityMap(), /*step*/ 1, llvm::None,
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
            builder.create<vector::ExtractElementOp>(loc, xVecWeight, ivs[0]);
        Value yPos_temp =
            builder.create<vector::ExtractElementOp>(loc, yVecWeight, ivs[0]);

        indexWeights.push_back(
            valBound(builder, loc, xPos_temp, inputColLastElemF32, c0F32));
        indexWeights.push_back(
            valBound(builder, loc, yPos_temp, inputRowLastElemF32, c0F32));

        std::vector<Value> indexWeights_UnitComplements = {
            builder.create<arith::SubFOp>(loc, c1F32, indexWeights[0]),
            builder.create<arith::SubFOp>(loc, c1F32, indexWeights[1])};

        Value pixelVal_a = builder.create<memref::LoadOp>(
            loc, builder.getF32Type(), input,
            ValueRange{inputIndices_L[0], inputIndices_L[1]});
        Value pixelVal_b = builder.create<memref::LoadOp>(
            loc, builder.getF32Type(), input,
            ValueRange{inputIndices_H[0], inputIndices_L[1]});
        Value pixelVal_c = builder.create<memref::LoadOp>(
            loc, builder.getF32Type(), input,
            ValueRange{inputIndices_L[0], inputIndices_H[1]});
        Value pixelVal_d = builder.create<memref::LoadOp>(
            loc, builder.getF32Type(), input,
            ValueRange{inputIndices_H[0], inputIndices_H[1]});

        Value weightVal1 =
            builder.create<arith::MulFOp>(loc, indexWeights_UnitComplements[0],
                                          indexWeights_UnitComplements[1]);
        Value weightVal2 = builder.create<arith::MulFOp>(
            loc, indexWeights[0], indexWeights_UnitComplements[1]);
        Value weightVal3 = builder.create<arith::MulFOp>(
            loc, indexWeights[1], indexWeights_UnitComplements[0]);
        Value weightVal4 = builder.create<arith::MulFOp>(loc, indexWeights[0],
                                                         indexWeights[1]);

        Value interm1 =
            builder.create<arith::MulFOp>(loc, pixelVal_a, weightVal1);
        Value interm2 =
            builder.create<arith::MulFOp>(loc, pixelVal_b, weightVal2);
        Value interm3 =
            builder.create<arith::MulFOp>(loc, pixelVal_c, weightVal3);
        Value interm4 =
            builder.create<arith::MulFOp>(loc, pixelVal_d, weightVal4);

        Value pixel_interm1 =
            builder.create<arith::AddFOp>(loc, interm1, interm2);
        Value pixel_interm2 =
            builder.create<arith::AddFOp>(loc, interm3, interm4);
        Value pixel_interm3 =
            builder.create<arith::AddFOp>(loc, pixel_interm1, pixel_interm2);

        Value pixelVal = roundOff(builder, loc, pixel_interm3);

        builder.create<memref::StoreOp>(
            loc, pixelVal, output, ValueRange{resIndices[0], resIndices[1]});

        builder.create<AffineYieldOp>(loc);
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
  buildAffineLoopNest(
      builder, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        Value ivs0F32 = indexToF32(builder, loc, ivs[0]);
        Value yVec = builder.create<vector::SplatOp>(loc, vectorTy32, ivs0F32);
        Value xVec = iotaVec(builder, loc, ctx, ivs[1], strideVal, vectorTy32,
                             c0, stride);

        Value resXVecInterm = builder.create<arith::MulFOp>(
            loc, xVec, horizontalScalingFactorVec);
        Value resYVecInterm =
            builder.create<arith::MulFOp>(loc, yVec, verticalScalingFactorVec);

        Value resXVec = roundOff(builder, loc, resXVecInterm);
        Value resYVec = roundOff(builder, loc, resYVecInterm);

        fillPixels(builder, loc, xVec, yVec, resXVec, resYVec, input, output,
                   c0, strideVal, outputRowLastElemF32, outputColLastElemF32,
                   inputRowLastElemF32, inputColLastElemF32, c0F32);
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
  buildAffineLoopNest(
      builder, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        Value ivs0F32 = indexToF32(builder, loc, ivs[0]);
        Value yVec = builder.create<vector::SplatOp>(loc, vectorTy32, ivs0F32);
        Value xVec = iotaVec(builder, loc, ctx, ivs[1], strideVal, vectorTy32,
                             c0, stride);

        Value xVecInterm = builder.create<arith::MulFOp>(
            loc, xVec, horizontalScalingFactorVec);
        Value yVecInterm =
            builder.create<arith::MulFOp>(loc, yVec, verticalScalingFactorVec);

        Value xVecInterm_L = builder.create<math::FloorOp>(loc, xVecInterm);
        Value xVecInterm_H = builder.create<math::CeilOp>(loc, xVecInterm);

        Value yVecInterm_L = builder.create<math::FloorOp>(loc, yVecInterm);
        Value yVecInterm_H = builder.create<math::CeilOp>(loc, yVecInterm);

        Value xVecWeight =
            builder.create<arith::SubFOp>(loc, xVecInterm, xVecInterm_L);
        Value yVecWeight =
            builder.create<arith::SubFOp>(loc, yVecInterm, yVecInterm_L);

        fillPixelsBilinearInterpolate(
            builder, loc, xVec, yVec, xVecInterm_L, yVecInterm_L, xVecInterm_H,
            yVecInterm_H, input, output, c0, strideVal, xVecWeight, yVecWeight,
            outputRowLastElemF32, outputColLastElemF32, inputRowLastElemF32,
            inputColLastElemF32, c0F32, c1F32);
      });
}

void traverseImagewBoundaryExtrapolation(
    OpBuilder &rewriter, Location loc, MLIRContext *ctx, Value input,
    Value kernel, Value output, Value centerX, Value centerY,
    Value constantValue, Value strideVal, Type elemTy,
    buddy::dip::BoundaryOption boundaryOptionAttr, int64_t stride, DIP_OP op) {
  // Create constant indices.
  Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);

  IntegerType i1 = IntegerType::get(ctx, 1);

  // Create DimOp.
  Value inputRow = rewriter.create<memref::DimOp>(loc, input, c0);
  Value inputCol = rewriter.create<memref::DimOp>(loc, input, c1);
  Value kernelSize = rewriter.create<memref::DimOp>(loc, kernel, c0);

  // Variables used for detecting rowMid, rowDown, colMid and colRight
  // regions.
  Value rowMidHelper = rewriter.create<AddIOp>(loc, inputRow, centerY);
  Value colMidHelper = rewriter.create<AddIOp>(loc, inputCol, centerX);

  SmallVector<Value, 8> lowerBounds(4, c0);
  SmallVector<Value, 8> uperBounds{inputRow, kernelSize, inputCol, kernelSize};
  SmallVector<int64_t, 8> steps{1, 1, stride, 1};

  VectorType vectorTy32 = VectorType::get({stride}, elemTy);
  VectorType vectorMaskTy = VectorType::get({stride}, i1);

  Value zeroPaddingElem = insertZeroConstantOp(ctx, rewriter, loc, elemTy);
  Value zeroPadding =
      rewriter.create<BroadcastOp>(loc, vectorTy32, zeroPaddingElem);

  AffineExpr a, b, c;
  bindDims(ctx, a, b, c);
  AffineMap calcHelper = AffineMap::get(3, 0, {a + b - c}, ctx);

  Value pseudoCol = rewriter.create<AffineApplyOp>(
      loc, calcHelper, ValueRange{inputCol, kernelSize, c1});

  buildAffineLoopNest(
      rewriter, loc, lowerBounds, uperBounds, steps,
      [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        // Indices of current pixel with respect to pseudo image containing
        // extrapolated boundaries.
        Value currRow = builder.create<AddIOp>(loc, ivs[0], ivs[1]);
        Value currCol = builder.create<AddIOp>(loc, ivs[2], ivs[3]);

        Value kernelValue = builder.create<memref::LoadOp>(
            loc, kernel, ValueRange{ivs[1], ivs[3]});
        Value kernelVec =
            builder.create<BroadcastOp>(loc, vectorTy32, kernelValue);

        // Pixel indices with respect to the actual image.
        Value imRow = builder.create<SubIOp>(loc, currRow, centerY);
        Value imCol = builder.create<SubIOp>(loc, currCol, centerX);

        // Index of pixel used for determining right region.
        Value colLastElem = builder.create<AddIOp>(loc, currCol, strideVal);

        Value rowUpCond =
            builder.create<CmpIOp>(loc, CmpIPredicate::slt, currRow, centerY);

        builder.create<scf::IfOp>(
            loc, rowUpCond,
            [&](OpBuilder &builder, Location loc) {
              // rowUp
              if (boundaryOptionAttr ==
                  buddy::dip::BoundaryOption::ConstantPadding) {
                Value inputVec =
                    builder.create<BroadcastOp>(loc, vectorTy32, constantValue);

                if (op == DIP_OP::CORRELATION_2D) {
                  calcAndStoreFMAwoTailProcessing(builder, loc, vectorTy32,
                                                  inputVec, kernelVec, output,
                                                  ivs[0], ivs[2]);
                }
              } else {
                Value colLeftCond = builder.create<CmpIOp>(
                    loc, CmpIPredicate::slt, currCol, centerX);

                builder.create<scf::IfOp>(
                    loc, colLeftCond,
                    [&](OpBuilder &builder, Location loc) {
                      // colLeft & rowUp
                      Value inputVec;
                      Value leftMaskElem =
                          builder.create<SubIOp>(loc, centerX, currCol);
                      Value leftMask = createInvertedMask(
                          builder, loc, strideVal, vectorMaskTy, leftMaskElem);

                      if (boundaryOptionAttr ==
                          buddy::dip::BoundaryOption::ReplicatePadding) {
                        Value paddingVal = builder.create<memref::LoadOp>(
                            loc, input, ValueRange{c0, c0});
                        Value padding = builder.create<BroadcastOp>(
                            loc, vectorTy32, paddingVal);

                        Value leftPaddingOffset =
                            builder.create<SubIOp>(loc, c0, leftMaskElem);
                        inputVec = builder.create<vector::MaskedLoadOp>(
                            loc, vectorTy32, input,
                            ValueRange{c0, leftPaddingOffset}, leftMask,
                            padding);
                      }

                      if (op == DIP_OP::CORRELATION_2D) {
                        calcAndStoreFMAwoTailProcessing(
                            builder, loc, vectorTy32, inputVec, kernelVec,
                            output, ivs[0], ivs[2]);
                      }

                      builder.create<scf::YieldOp>(loc);
                    },
                    [&](OpBuilder &builder, Location loc) {
                      // (colMid or colRight) & rowUp
                      Value colMidCond = builder.create<CmpIOp>(
                          loc, CmpIPredicate::sle, colLastElem, colMidHelper);

                      builder.create<scf::IfOp>(
                          loc, colMidCond,
                          [&](OpBuilder &builder, Location loc) {
                            // colMid & rowUp
                            Value inputVec;
                            if (boundaryOptionAttr ==
                                buddy::dip::BoundaryOption::ReplicatePadding) {
                              inputVec =
                                  builder.create<LoadOp>(loc, vectorTy32, input,
                                                         ValueRange{c0, imCol});
                            }

                            if (op == DIP_OP::CORRELATION_2D) {
                              calcAndStoreFMAwoTailProcessing(
                                  builder, loc, vectorTy32, inputVec, kernelVec,
                                  output, ivs[0], ivs[2]);
                            }

                            builder.create<scf::YieldOp>(loc);
                          },
                          [&](OpBuilder &builder, Location loc) {
                            // colRight & rowUp
                            Value inputVec;
                            Value rightMaskHelper = builder.create<SubIOp>(
                                loc, colLastElem, colMidHelper);
                            Value rightMaskElem = builder.create<SubIOp>(
                                loc, strideVal, rightMaskHelper);
                            Value rightMask = builder.create<CreateMaskOp>(
                                loc, vectorMaskTy, rightMaskElem);

                            if (boundaryOptionAttr ==
                                buddy::dip::BoundaryOption::ReplicatePadding) {
                              Value rightRange =
                                  builder.create<SubIOp>(loc, inputCol, c1);
                              Value paddingVal = builder.create<memref::LoadOp>(
                                  loc, input, ValueRange{c0, rightRange});
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, paddingVal);

                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, input, ValueRange{c0, imCol},
                                  rightMask, padding);
                            }
                            Value tailCond =
                                tailChecker(builder, loc, calcHelper, strideVal,
                                            kernelSize, c1, pseudoCol, ivs[2]);

                            if (op == DIP_OP::CORRELATION_2D) {
                              calcAndStoreFMAwTailProcessing(
                                  builder, loc, vectorTy32, inputVec, kernelVec,
                                  output, ivs[0], ivs[2], tailCond, zeroPadding,
                                  inputCol, vectorMaskTy);
                            }

                            builder.create<scf::YieldOp>(loc);
                          });
                      builder.create<scf::YieldOp>(loc);
                    });
              }
              builder.create<scf::YieldOp>(loc);
            },
            [&](OpBuilder &builder, Location loc) {
              // rowMid or rowDown
              Value rowMidCond = builder.create<CmpIOp>(loc, CmpIPredicate::slt,
                                                        currRow, rowMidHelper);

              builder.create<scf::IfOp>(
                  loc, rowMidCond,
                  [&](OpBuilder &builder, Location loc) {
                    // rowMid
                    Value colLeftCond = builder.create<CmpIOp>(
                        loc, CmpIPredicate::slt, currCol, centerX);

                    builder.create<scf::IfOp>(
                        loc, colLeftCond,
                        [&](OpBuilder &builder, Location loc) {
                          // colLeft & rowMid
                          Value inputVec;
                          Value leftMaskElem =
                              builder.create<SubIOp>(loc, centerX, currCol);
                          Value leftMask =
                              createInvertedMask(builder, loc, strideVal,
                                                 vectorMaskTy, leftMaskElem);

                          if (boundaryOptionAttr ==
                              buddy::dip::BoundaryOption::ConstantPadding) {
                            Value padding = builder.create<BroadcastOp>(
                                loc, vectorTy32, constantValue);

                            Value leftPaddingOffset =
                                builder.create<SubIOp>(loc, c0, leftMaskElem);
                            inputVec = builder.create<MaskedLoadOp>(
                                loc, vectorTy32, input,
                                ValueRange{imRow, leftPaddingOffset}, leftMask,
                                padding);
                          } else if (boundaryOptionAttr ==
                                     buddy::dip::BoundaryOption::
                                         ReplicatePadding) {
                            Value paddingVal = builder.create<memref::LoadOp>(
                                loc, input, ValueRange{imRow, c0});
                            Value padding = builder.create<BroadcastOp>(
                                loc, vectorTy32, paddingVal);

                            Value leftPaddingOffset =
                                builder.create<SubIOp>(loc, c0, leftMaskElem);
                            inputVec = builder.create<MaskedLoadOp>(
                                loc, vectorTy32, input,
                                ValueRange{imRow, leftPaddingOffset}, leftMask,
                                padding);
                          }

                          if (op == DIP_OP::CORRELATION_2D) {
                            calcAndStoreFMAwoTailProcessing(
                                builder, loc, vectorTy32, inputVec, kernelVec,
                                output, ivs[0], ivs[2]);
                          }

                          builder.create<scf::YieldOp>(loc);
                        },
                        [&](OpBuilder &builder, Location loc) {
                          // (colMid or colRight) & rowMid
                          Value colMidCond =
                              builder.create<CmpIOp>(loc, CmpIPredicate::sle,
                                                     colLastElem, colMidHelper);

                          builder.create<scf::IfOp>(
                              loc, colMidCond,
                              [&](OpBuilder &builder, Location loc) {
                                // colMid & rowMid
                                Value inputVec = builder.create<LoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{imRow, imCol});

                                if (op == DIP_OP::CORRELATION_2D) {
                                  calcAndStoreFMAwoTailProcessing(
                                      builder, loc, vectorTy32, inputVec,
                                      kernelVec, output, ivs[0], ivs[2]);
                                }

                                builder.create<scf::YieldOp>(loc);
                              },
                              [&](OpBuilder &builder, Location loc) {
                                // colRight & rowMid
                                Value inputVec;
                                Value rightMaskHelper = builder.create<SubIOp>(
                                    loc, colLastElem, colMidHelper);
                                Value rightMaskElem = builder.create<SubIOp>(
                                    loc, strideVal, rightMaskHelper);
                                Value rightMask = builder.create<CreateMaskOp>(
                                    loc, vectorMaskTy, rightMaskElem);

                                if (boundaryOptionAttr ==
                                    buddy::dip::BoundaryOption::
                                        ConstantPadding) {
                                  Value padding = builder.create<BroadcastOp>(
                                      loc, vectorTy32, constantValue);

                                  inputVec = builder.create<MaskedLoadOp>(
                                      loc, vectorTy32, input,
                                      ValueRange{imRow, imCol}, rightMask,
                                      padding);
                                } else if (boundaryOptionAttr ==
                                           buddy::dip::BoundaryOption::
                                               ReplicatePadding) {
                                  Value rightRange =
                                      builder.create<SubIOp>(loc, inputCol, c1);
                                  Value paddingVal =
                                      builder.create<memref::LoadOp>(
                                          loc, input,
                                          ValueRange{imRow, rightRange});
                                  Value padding = builder.create<BroadcastOp>(
                                      loc, vectorTy32, paddingVal);

                                  inputVec = builder.create<MaskedLoadOp>(
                                      loc, vectorTy32, input,
                                      ValueRange{imRow, imCol}, rightMask,
                                      padding);
                                }
                                Value tailCond = tailChecker(
                                    builder, loc, calcHelper, strideVal,
                                    kernelSize, c1, pseudoCol, ivs[2]);

                                if (op == DIP_OP::CORRELATION_2D) {
                                  calcAndStoreFMAwTailProcessing(
                                      builder, loc, vectorTy32, inputVec,
                                      kernelVec, output, ivs[0], ivs[2],
                                      tailCond, zeroPadding, inputCol,
                                      vectorMaskTy);
                                }

                                builder.create<scf::YieldOp>(loc);
                              });
                          builder.create<scf::YieldOp>(loc);
                        });
                    builder.create<scf::YieldOp>(loc);
                  },
                  [&](OpBuilder &builder, Location loc) {
                    // rowDown
                    if (boundaryOptionAttr ==
                        buddy::dip::BoundaryOption::ConstantPadding) {
                      Value inputVec = builder.create<BroadcastOp>(
                          loc, vectorTy32, constantValue);

                      if (op == DIP_OP::CORRELATION_2D) {
                        calcAndStoreFMAwoTailProcessing(
                            builder, loc, vectorTy32, inputVec, kernelVec,
                            output, ivs[0], ivs[2]);
                      }
                    } else {
                      Value colLeftCond = builder.create<CmpIOp>(
                          loc, CmpIPredicate::slt, currCol, centerX);

                      builder.create<scf::IfOp>(
                          loc, colLeftCond,
                          [&](OpBuilder &builder, Location loc) {
                            // colLeft & rowDown
                            Value inputVec;
                            Value downRange =
                                builder.create<SubIOp>(loc, inputRow, c1);
                            Value leftMaskElem =
                                builder.create<SubIOp>(loc, centerX, currCol);
                            Value leftMask =
                                createInvertedMask(builder, loc, strideVal,
                                                   vectorMaskTy, leftMaskElem);

                            if (boundaryOptionAttr ==
                                buddy::dip::BoundaryOption::ReplicatePadding) {
                              Value paddingVal = builder.create<memref::LoadOp>(
                                  loc, input, ValueRange{downRange, c0});
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, paddingVal);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, input,
                                  ValueRange{downRange, leftPaddingOffset},
                                  leftMask, padding);
                            }

                            if (op == DIP_OP::CORRELATION_2D) {
                              calcAndStoreFMAwoTailProcessing(
                                  builder, loc, vectorTy32, inputVec, kernelVec,
                                  output, ivs[0], ivs[2]);
                            }

                            builder.create<scf::YieldOp>(loc);
                          },
                          [&](OpBuilder &builder, Location loc) {
                            // (colMid or colRight) & rowDown
                            Value colMidCond = builder.create<CmpIOp>(
                                loc, CmpIPredicate::sle, colLastElem,
                                colMidHelper);

                            builder.create<scf::IfOp>(
                                loc, colMidCond,
                                [&](OpBuilder &builder, Location loc) {
                                  // colMid & rowDown
                                  Value inputVec;
                                  Value downRange =
                                      builder.create<SubIOp>(loc, inputRow, c1);
                                  if (boundaryOptionAttr ==
                                      buddy::dip::BoundaryOption::
                                          ReplicatePadding) {
                                    inputVec = builder.create<LoadOp>(
                                        loc, vectorTy32, input,
                                        ValueRange{downRange, imCol});
                                  }

                                  if (op == DIP_OP::CORRELATION_2D) {
                                    calcAndStoreFMAwoTailProcessing(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2]);
                                  }

                                  builder.create<scf::YieldOp>(loc);
                                },
                                [&](OpBuilder &builder, Location loc) {
                                  // colRight & rowDown
                                  Value inputVec;
                                  Value rightMaskHelper =
                                      builder.create<SubIOp>(loc, colLastElem,
                                                             colMidHelper);
                                  Value rightMaskElem = builder.create<SubIOp>(
                                      loc, strideVal, rightMaskHelper);
                                  Value rightMask =
                                      builder.create<CreateMaskOp>(
                                          loc, vectorMaskTy, rightMaskElem);

                                  Value downRange =
                                      builder.create<SubIOp>(loc, inputRow, c1);
                                  Value rightRange =
                                      builder.create<SubIOp>(loc, inputCol, c1);

                                  if (boundaryOptionAttr ==
                                      buddy::dip::BoundaryOption::
                                          ReplicatePadding) {

                                    Value paddingVal =
                                        builder.create<memref::LoadOp>(
                                            loc, input,
                                            ValueRange{downRange, rightRange});
                                    Value padding =
                                        builder.create<vector::BroadcastOp>(
                                            loc, vectorTy32, paddingVal);

                                    inputVec = builder.create<MaskedLoadOp>(
                                        loc, vectorTy32, input,
                                        ValueRange{downRange, imCol}, rightMask,
                                        padding);
                                  }
                                  Value tailCond = tailChecker(
                                      builder, loc, calcHelper, strideVal,
                                      kernelSize, c1, pseudoCol, ivs[2]);

                                  if (op == DIP_OP::CORRELATION_2D) {
                                    calcAndStoreFMAwTailProcessing(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, zeroPadding, inputCol,
                                        vectorMaskTy);
                                  }

                                  builder.create<scf::YieldOp>(loc);
                                });
                            builder.create<scf::YieldOp>(loc);
                          });
                    }
                    builder.create<scf::YieldOp>(loc);
                  });
              builder.create<scf::YieldOp>(loc);
            });
      });
}

#endif
