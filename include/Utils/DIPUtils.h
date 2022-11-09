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

#include "DIP/DIPOps.h"
#include "Utils/Utils.h"
#include <stdarg.h>

using namespace mlir;

namespace buddy {
namespace dip {

// Specify operation names which will be used for performing operation specific
// tasks inside generic utility functions.
enum class DIP_OP { CORRELATION_2D, EROSION_2D, DILATION_2D };

// Specify error codes specific to DIP dialect which might be used for exiting
// from lowering passes with appropriate messages.
enum class DIP_ERROR { INCONSISTENT_TYPES, UNSUPPORTED_TYPE, NO_ERROR };

// Inserts a constant op with value 0 into a location `loc` based on type
// `type`. Supported types are : f32, f64, integer types.
Value insertZeroConstantOp(MLIRContext *ctx, OpBuilder &builder, Location loc,
                           Type elemTy);

// Inserts FMA operation into a given location `loc` based on type `type`.
// Note: FMA is done by Multiply and Add for integer types, because there is no
// dedicated FMA operation for them.
// Supported types: f32, f64, integer types
Value insertFMAOp(OpBuilder &builder, Location loc, VectorType type,
                  Value inputVec, Value kernelVec, Value outputVec);

// Calculate result of FMA and store it in output memref. This function cannot
// handle tail processing.
void calcAndStoreFMAwoTailProcessing(OpBuilder &builder, Location loc,
                                     VectorType vecType, Value inputVec,
                                     Value kernelVec, Value output,
                                     Value beginIdx, Value endIdx);

// Checks if we encountered a tail (columns remaining after processing in
// batches of stride size).
Value tailChecker(OpBuilder &builder, Location loc, AffineMap calcHelper,
                  Value strideVal, Value kernelSize, Value c1, Value pseudoCol,
                  Value colPivot);

// Creates the required mask which is to be used for tail processing.
Value tailMaskCreator(OpBuilder &builder, Location loc, Value inputCol,
                      Value colPivot, VectorType vectorMaskTy);

// Calculate result of FMA and store it in output memref. This function can
// handle tail processing.
void calcAndStoreFMAwTailProcessing(OpBuilder &builder, Location loc,
                                    VectorType vecType, Value inputVec,
                                    Value kernelVec, Value output,
                                    Value beginIdx, Value endIdx,
                                    Value tailCond, Value zeroPadding,
                                    Value inputCol, VectorType vectorMaskTy);

// Apply 3 shear method and return mapped values.
std::vector<Value> shearTransform(OpBuilder &builder, Location loc,
                                  Value originalX, Value originalY,
                                  Value sinVec, Value tanVec);

// Apply standard rotation matrix transformation and return mapped values.
std::vector<Value> standardRotate(OpBuilder &builder, Location loc,
                                  Value originalX, Value originalY,
                                  Value sinVec, Value cosVec);

// Get center co-ordinates w.r.t given dimension.
Value getCenter(OpBuilder &builder, Location loc, MLIRContext *ctx, Value dim);

// Scale pixel co-ordinates appropriately before calculating their rotated
// position(s).
Value pixelScaling(OpBuilder &builder, Location loc, Value imageDImF32Vec,
                   Value coordVec, Value imageCenterF32Vec, Value c1F32Vec);

// Extract values present at a particular index in two vectors for using
// those values to load an element from a memref.
std::vector<Value> extractIndices(OpBuilder &builder, Location loc, Value xVec,
                                  Value yVec, Value vecIndex, Value xUpperBound,
                                  Value yUpperBound, Value c0F32);

// Fill appropriate pixel data in its corresponding rotated co-ordinate of
// output image.
void fillPixels(OpBuilder &builder, Location loc, Value resXVec, Value resYVec,
                Value xVec, Value yVec, Value input, Value output, Value c0,
                Value strideVal, Value outputRowLastElemF32,
                Value outputColLastElemF32, Value inputRowLastElemF32,
                Value inputColLastElemF32, Value c0F32);

// Calculate tan(angle / 2) where angle is a function parameter.
Value customTanVal(OpBuilder &builder, Location loc, Value angleVal);

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
    Value c1F32Vec, VectorType vectorTy32, int64_t stride, FloatType f32);

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
    Value c1F32Vec, VectorType vectorTy32, int64_t stride, FloatType f32);

// Fills pixels in bilinear interpolation fashion.
void fillPixelsBilinearInterpolate(
    OpBuilder &builder, Location loc, Value resXVec, Value resYVec,
    Value xVec_L, Value yVec_L, Value xVec_H, Value yVec_H, Value input,
    Value output, Value c0, Value strideVal, Value xVecWeight, Value yVecWeight,
    Value outputRowLastElemF32, Value outputColLastElemF32,
    Value inputRowLastElemF32, Value inputColLastElemF32, Value c0F32,
    Value c1F32);

// Helper function for resizing an image using nearest neighbour interpolation
// mechanism.
void NearestNeighbourInterpolationResizing(
    OpBuilder &builder, Location loc, MLIRContext *ctx,
    SmallVector<Value, 8> lowerBounds, SmallVector<Value, 8> upperBounds,
    SmallVector<int64_t, 8> steps, Value strideVal, Value input, Value output,
    Value horizontalScalingFactorVec, Value verticalScalingFactorVec,
    Value outputRowLastElemF32, Value outputColLastElemF32,
    Value inputRowLastElemF32, Value inputColLastElemF32, VectorType vectorTy32,
    int64_t stride, Value c0, Value c0F32);

// Helper function for resizing an image using bilinear interpolation mechanism.
void BilinearInterpolationResizing(
    OpBuilder &builder, Location loc, MLIRContext *ctx,
    SmallVector<Value, 8> lowerBounds, SmallVector<Value, 8> upperBounds,
    SmallVector<int64_t, 8> steps, Value strideVal, Value input, Value output,
    Value horizontalScalingFactorVec, Value verticalScalingFactorVec,
    Value outputRowLastElemF32, Value outputColLastElemF32,
    Value inputRowLastElemF32, Value inputColLastElemF32, VectorType vectorTy32,
    int64_t stride, Value c0, Value c0F32, Value c1F32);

// Function to test whether a value is equivalent to zero or not.
Value zeroCond(OpBuilder &builder, Location loc, Type elemType, Value value,
               Value zeroElem);

// Util function for morphological transformations ; compares two vectors and
// returns a mask
Value createCompVecMorph(OpBuilder &builder, Location loc, VectorType type,
                         Value inputVec, Value outputVec, DIP_OP op);

// Utility function for morphological operations ; cannot handle tail processing
void calcAndStorewoTailProcessingMorph(
    OpBuilder &builder, Location loc, VectorType vecType, Value inputVec,
    Value kernelVec, Value output, Value beginIdx, Value endIdx,
    Value zeroPadding, Value inputCol, VectorType vectorMaskTy, Type elemTy,
    Value kernelValue, Value zeroPaddingElem, DIP_OP op);

// Utility function for morphological transformations, can handle tail
// processing
void calcAndStorewTailProcessingMorph(
    OpBuilder &builder, Location loc, VectorType vecType, Value inputVec,
    Value kernelVec, Value output, Value beginIdx, Value endIdx, Value tailCond,
    Value zeroPadding, Value inputCol, VectorType vectorMaskTy, Type elemTy,
    Value kernelValue, Value zeroPaddingElem, DIP_OP op);

void traverseImagewBoundaryExtrapolation(
    OpBuilder &rewriter, Location loc, MLIRContext *ctx, Value input,
    Value kernel, Value output, Value centerX, Value centerY,
    Value constantValue, Value strideVal, Type elemTy,
    buddy::dip::BoundaryOption boundaryOptionAttr, int64_t stride, DIP_OP op);

// Function for applying type check mechanisms for all DIP dialect operations.
template <typename DIPOP>
DIP_ERROR checkDIPCommonTypes(DIPOP op, const std::vector<Value> &args);
} // namespace dip
} // namespace buddy

#endif // INCLUDE_UTILS_DIPUTILS_H
