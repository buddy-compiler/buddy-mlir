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

// Calculate result of FMA and store it in output memref. This function cannot
// handle tail processing.
void calcAndStoreFMAwoTailProcessing(OpBuilder &builder, Location loc,
                                     VectorType vecType, Value inputVec,
                                     Value kernelVec, Value output,
                                     Value beginIdx, Value endIdx) {
  Value outputVec = builder.create<LoadOp>(loc, vecType, output,
                                           ValueRange{beginIdx, endIdx});
  Value resVec = builder.create<FMAOp>(loc, inputVec, kernelVec, outputVec);
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
            builder.create<FMAOp>(loc, inputVec, kernelVec, outputVec);
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
            builder.create<FMAOp>(loc, inputVec, kernelVec, outputVec);
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
#endif
