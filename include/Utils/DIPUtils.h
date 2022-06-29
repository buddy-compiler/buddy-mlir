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

// Fill appropriate pixel data in its corresponding rotated co-ordinate of
// output image.
void fillPixels(OpBuilder &builder, Location loc, Value resXVec, Value resYVec,
                Value xVec, Value yVec, Value input, Value output, Value c0,
                Value strideVal, Value outputRowLastElemF32,
                Value outputColLastElemF32, Value c0F32) {
  builder.create<AffineForOp>(
      loc, ValueRange{c0}, builder.getDimIdentityMap(), ValueRange{strideVal},
      builder.getDimIdentityMap(), /*step*/ 1, llvm::None,
      [&](OpBuilder &builder, Location loc, ValueRange ivs,
          ValueRange iterArg) {
        Value resXPos =
            builder.create<vector::ExtractElementOp>(loc, resXVec, ivs[0]);
        Value resYPos =
            builder.create<vector::ExtractElementOp>(loc, resYVec, ivs[0]);

        Value resXPosBound =
            valBound(builder, loc, resXPos, outputColLastElemF32, c0F32);
        Value resYPosBound =
            valBound(builder, loc, resYPos, outputRowLastElemF32, c0F32);

        Value resXPosIndex = F32ToIndex(builder, loc, resXPosBound);
        Value resYPosIndex = F32ToIndex(builder, loc, resYPosBound);

        Value xPos =
            builder.create<vector::ExtractElementOp>(loc, xVec, ivs[0]);
        Value yPos =
            builder.create<vector::ExtractElementOp>(loc, yVec, ivs[0]);

        Value xPosIndex = F32ToIndex(builder, loc, xPos);
        Value yPosIndex = F32ToIndex(builder, loc, yPos);

        Value pixelVal = builder.create<memref::LoadOp>(
            loc, builder.getF32Type(), input, ValueRange{xPosIndex, yPosIndex});
        builder.create<memref::StoreOp>(loc, pixelVal, output,
                                        ValueRange{resXPosIndex, resYPosIndex});

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
    Value outputRowLastElemF32, Value outputColLastElemF32, Value c0,
    Value c0F32, Value c1F32Vec, VectorType vectorTy32, int64_t stride,
    FloatType f32) {
  buildAffineLoopNest(
      builder, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        Value ivs0F32 = indexToF32(builder, loc, ivs[0]);
        Value yVec = builder.create<vector::SplatOp>(loc, vectorTy32, ivs0F32);
        Value xVec = iotaVec(builder, loc, ctx, ivs[1], strideVal,
                             vectorTy32, c0, stride);

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
                   c0F32);
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
    Value outputRowLastElemF32, Value outputColLastElemF32, Value c0,
    Value c0F32, Value c1F32Vec, VectorType vectorTy32, int64_t stride,
    FloatType f32) {
  Value cosVal = builder.create<math::CosOp>(loc, angleVal);
  Value cosVec = builder.create<vector::BroadcastOp>(loc, vectorTy32, cosVal);

  buildAffineLoopNest(
      builder, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        Value ivs0F32 = indexToF32(builder, loc, ivs[0]);
        Value yVec = builder.create<vector::SplatOp>(loc, vectorTy32, ivs0F32);
        Value xVec = iotaVec(builder, loc, ctx, ivs[1], strideVal,
                             vectorTy32, c0, stride);

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
                   c0, strideVal, outputRowLastElemF32, outputColLastElemF32,
                   c0F32);
      });
}

#endif
