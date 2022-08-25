//====- LowerDIPPass.cpp - dip Dialect Lowering Pass  ---------------------===//
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
// This file defines DIP dialect lowering pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

#include "DIP/DIPDialect.h"
#include "DIP/DIPOps.h"
#include "Utils/DIPUtils.h"
#include "Utils/Utils.h"
#include <vector>

using namespace mlir;
using namespace buddy;
using namespace vector;
using namespace mlir::arith;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class DIPCorr2DOpLowering : public OpRewritePattern<dip::Corr2DOp> {
public:
  using OpRewritePattern<dip::Corr2DOp>::OpRewritePattern;

  explicit DIPCorr2DOpLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
  }

  LogicalResult matchAndRewrite(dip::Corr2DOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Create constant indices.
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);

    // Register operand values.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    Value centerX = op->getOperand(3);
    Value centerY = op->getOperand(4);
    Value constantValue = op->getOperand(5);
    auto boundaryOptionAttr = op.boundary_option();
    Value strideVal = rewriter.create<ConstantIndexOp>(loc, stride);

    FloatType f32 = FloatType::getF32(ctx);
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
    SmallVector<Value, 8> uperBounds{inputRow, kernelSize, inputCol,
                                     kernelSize};
    SmallVector<int64_t, 8> steps{1, 1, stride, 1};

    VectorType vectorTy32 = VectorType::get({stride}, f32);
    VectorType vectorMaskTy = VectorType::get({stride}, i1);

    Value zeroPaddingElem =
        rewriter.create<ConstantFloatOp>(loc, (APFloat)(float)0, f32);
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
                    dip::BoundaryOption::ConstantPadding) {
                  Value inputVec = builder.create<BroadcastOp>(loc, vectorTy32,
                                                               constantValue);

                  calcAndStoreFMAwoTailProcessing(builder, loc, vectorTy32,
                                                  inputVec, kernelVec, output,
                                                  ivs[0], ivs[2]);
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
                        Value leftMask =
                            createInvertedMask(builder, loc, strideVal,
                                               vectorMaskTy, leftMaskElem);

                        if (boundaryOptionAttr ==
                            dip::BoundaryOption::ReplicatePadding) {
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
                        calcAndStoreFMAwoTailProcessing(
                            builder, loc, vectorTy32, inputVec, kernelVec,
                            output, ivs[0], ivs[2]);

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
                                  dip::BoundaryOption::ReplicatePadding) {
                                inputVec = builder.create<LoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{c0, imCol});
                              }
                              calcAndStoreFMAwoTailProcessing(
                                  builder, loc, vectorTy32, inputVec, kernelVec,
                                  output, ivs[0], ivs[2]);

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
                                  dip::BoundaryOption::ReplicatePadding) {
                                Value rightRange =
                                    builder.create<SubIOp>(loc, inputCol, c1);
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, input, ValueRange{c0, rightRange});
                                Value padding = builder.create<BroadcastOp>(
                                    loc, vectorTy32, paddingVal);

                                inputVec = builder.create<MaskedLoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{c0, imCol}, rightMask, padding);
                              }
                              Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                              calcAndStoreFMAwTailProcessing(
                                  builder, loc, vectorTy32, inputVec, kernelVec,
                                  output, ivs[0], ivs[2], tailCond, zeroPadding,
                                  inputCol, vectorMaskTy);

                              builder.create<scf::YieldOp>(loc);
                            });
                        builder.create<scf::YieldOp>(loc);
                      });
                }
                builder.create<scf::YieldOp>(loc);
              },
              [&](OpBuilder &builder, Location loc) {
                // rowMid or rowDown
                Value rowMidCond = builder.create<CmpIOp>(
                    loc, CmpIPredicate::slt, currRow, rowMidHelper);

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
                                dip::BoundaryOption::ConstantPadding) {
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, constantValue);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, input,
                                  ValueRange{imRow, leftPaddingOffset},
                                  leftMask, padding);
                            } else if (boundaryOptionAttr ==
                                       dip::BoundaryOption::ReplicatePadding) {
                              Value paddingVal = builder.create<memref::LoadOp>(
                                  loc, input, ValueRange{imRow, c0});
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, paddingVal);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, input,
                                  ValueRange{imRow, leftPaddingOffset},
                                  leftMask, padding);
                            }
                            calcAndStoreFMAwoTailProcessing(
                                builder, loc, vectorTy32, inputVec, kernelVec,
                                output, ivs[0], ivs[2]);

                            builder.create<scf::YieldOp>(loc);
                          },
                          [&](OpBuilder &builder, Location loc) {
                            // (colMid or colRight) & rowMid
                            Value colMidCond = builder.create<CmpIOp>(
                                loc, CmpIPredicate::sle, colLastElem,
                                colMidHelper);

                            builder.create<scf::IfOp>(
                                loc, colMidCond,
                                [&](OpBuilder &builder, Location loc) {
                                  // colMid & rowMid
                                  Value inputVec = builder.create<LoadOp>(
                                      loc, vectorTy32, input,
                                      ValueRange{imRow, imCol});
                                  calcAndStoreFMAwoTailProcessing(
                                      builder, loc, vectorTy32, inputVec,
                                      kernelVec, output, ivs[0], ivs[2]);

                                  builder.create<scf::YieldOp>(loc);
                                },
                                [&](OpBuilder &builder, Location loc) {
                                  // colRight & rowMid
                                  Value inputVec;
                                  Value rightMaskHelper =
                                      builder.create<SubIOp>(loc, colLastElem,
                                                             colMidHelper);
                                  Value rightMaskElem = builder.create<SubIOp>(
                                      loc, strideVal, rightMaskHelper);
                                  Value rightMask =
                                      builder.create<CreateMaskOp>(
                                          loc, vectorMaskTy, rightMaskElem);

                                  if (boundaryOptionAttr ==
                                      dip::BoundaryOption::ConstantPadding) {
                                    Value padding = builder.create<BroadcastOp>(
                                        loc, vectorTy32, constantValue);

                                    inputVec = builder.create<MaskedLoadOp>(
                                        loc, vectorTy32, input,
                                        ValueRange{imRow, imCol}, rightMask,
                                        padding);
                                  } else if (boundaryOptionAttr ==
                                             dip::BoundaryOption::
                                                 ReplicatePadding) {
                                    Value rightRange = builder.create<SubIOp>(
                                        loc, inputCol, c1);
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
                                  calcAndStoreFMAwTailProcessing(
                                      builder, loc, vectorTy32, inputVec,
                                      kernelVec, output, ivs[0], ivs[2],
                                      tailCond, zeroPadding, inputCol,
                                      vectorMaskTy);

                                  builder.create<scf::YieldOp>(loc);
                                });
                            builder.create<scf::YieldOp>(loc);
                          });
                      builder.create<scf::YieldOp>(loc);
                    },
                    [&](OpBuilder &builder, Location loc) {
                      // rowDown
                      if (boundaryOptionAttr ==
                          dip::BoundaryOption::ConstantPadding) {
                        Value inputVec = builder.create<BroadcastOp>(
                            loc, vectorTy32, constantValue);

                        calcAndStoreFMAwoTailProcessing(
                            builder, loc, vectorTy32, inputVec, kernelVec,
                            output, ivs[0], ivs[2]);
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
                              Value leftMask = createInvertedMask(
                                  builder, loc, strideVal, vectorMaskTy,
                                  leftMaskElem);

                              if (boundaryOptionAttr ==
                                  dip::BoundaryOption::ReplicatePadding) {
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, input, ValueRange{downRange, c0});
                                Value padding = builder.create<BroadcastOp>(
                                    loc, vectorTy32, paddingVal);

                                Value leftPaddingOffset =
                                    builder.create<SubIOp>(loc, c0,
                                                           leftMaskElem);
                                inputVec = builder.create<MaskedLoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{downRange, leftPaddingOffset},
                                    leftMask, padding);
                              }
                              calcAndStoreFMAwoTailProcessing(
                                  builder, loc, vectorTy32, inputVec, kernelVec,
                                  output, ivs[0], ivs[2]);

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
                                    Value downRange = builder.create<SubIOp>(
                                        loc, inputRow, c1);
                                    if (boundaryOptionAttr ==
                                        dip::BoundaryOption::ReplicatePadding) {
                                      inputVec = builder.create<LoadOp>(
                                          loc, vectorTy32, input,
                                          ValueRange{downRange, imCol});
                                    }
                                    calcAndStoreFMAwoTailProcessing(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2]);

                                    builder.create<scf::YieldOp>(loc);
                                  },
                                  [&](OpBuilder &builder, Location loc) {
                                    // colRight & rowDown
                                    Value inputVec;
                                    Value rightMaskHelper =
                                        builder.create<SubIOp>(loc, colLastElem,
                                                               colMidHelper);
                                    Value rightMaskElem =
                                        builder.create<SubIOp>(loc, strideVal,
                                                               rightMaskHelper);
                                    Value rightMask =
                                        builder.create<CreateMaskOp>(
                                            loc, vectorMaskTy, rightMaskElem);

                                    Value downRange = builder.create<SubIOp>(
                                        loc, inputRow, c1);
                                    Value rightRange = builder.create<SubIOp>(
                                        loc, inputCol, c1);

                                    if (boundaryOptionAttr ==
                                        dip::BoundaryOption::ReplicatePadding) {

                                      Value paddingVal =
                                          builder.create<memref::LoadOp>(
                                              loc, input,
                                              ValueRange{downRange,
                                                         rightRange});
                                      Value padding =
                                          builder.create<vector::BroadcastOp>(
                                              loc, vectorTy32, paddingVal);

                                      inputVec = builder.create<MaskedLoadOp>(
                                          loc, vectorTy32, input,
                                          ValueRange{downRange, imCol},
                                          rightMask, padding);
                                    }
                                    Value tailCond = tailChecker(
                                        builder, loc, calcHelper, strideVal,
                                        kernelSize, c1, pseudoCol, ivs[2]);
                                    calcAndStoreFMAwTailProcessing(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, zeroPadding, inputCol,
                                        vectorMaskTy);

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
    // Remove the origin convolution operation.
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;
};

class DIPRotate2DOpLowering : public OpRewritePattern<dip::Rotate2DOp> {
public:
  using OpRewritePattern<dip::Rotate2DOp>::OpRewritePattern;

  explicit DIPRotate2DOpLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
  }

  LogicalResult matchAndRewrite(dip::Rotate2DOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Register operand values.
    Value input = op->getOperand(0);
    Value angleVal = op->getOperand(1);
    Value output = op->getOperand(2);

    Value strideVal = rewriter.create<ConstantIndexOp>(loc, stride);
    FloatType f32 = FloatType::getF32(ctx);
    VectorType vectorTy32 = VectorType::get({stride}, f32);

    // Create constant indices.
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);

    Value c0F32 = indexToF32(rewriter, loc, c0);
    Value c1F32 = indexToF32(rewriter, loc, c1);
    Value c1F32Vec = rewriter.create<vector::SplatOp>(loc, vectorTy32, c1F32);

    // Get input image dimensions.
    Value inputRow = rewriter.create<memref::DimOp>(loc, input, c0);
    Value inputCol = rewriter.create<memref::DimOp>(loc, input, c1);

    // Create f32 type vectors from input dimensions.
    Value inputRowF32Vec = castAndExpand(rewriter, loc, inputRow, vectorTy32);
    Value inputColF32Vec = castAndExpand(rewriter, loc, inputCol, vectorTy32);

    // Get output image dimensions.
    Value outputRow = rewriter.create<memref::DimOp>(loc, output, c0);
    Value outputCol = rewriter.create<memref::DimOp>(loc, output, c1);

    // Obtain extreme allocatable value(s) in input and output for bounding
    // purpose.
    Value inputRowLastElem = rewriter.create<arith::SubIOp>(loc, inputRow, c1);
    Value inputRowLastElemF32 = indexToF32(rewriter, loc, inputRowLastElem);

    Value inputColLastElem = rewriter.create<arith::SubIOp>(loc, inputCol, c1);
    Value inputColLastElemF32 = indexToF32(rewriter, loc, inputColLastElem);

    Value outputRowLastElem =
        rewriter.create<arith::SubIOp>(loc, outputRow, c1);
    Value outputRowLastElemF32 = indexToF32(rewriter, loc, outputRowLastElem);

    Value outputColLastElem =
        rewriter.create<arith::SubIOp>(loc, outputCol, c1);
    Value outputColLastElemF32 = indexToF32(rewriter, loc, outputColLastElem);

    // Determine lower bound for second call of rotation function (this is done
    // for efficient tail processing).
    Value inputColStrideRatio =
        rewriter.create<arith::DivUIOp>(loc, inputCol, strideVal);
    Value inputColMultiple =
        rewriter.create<arith::MulIOp>(loc, strideVal, inputColStrideRatio);

    // Bounds for first call to rotation function (doesn't involve tail
    // processing).
    SmallVector<Value, 8> lowerBounds1(2, c0);
    SmallVector<Value, 8> upperBounds1{inputRow, inputColMultiple};

    // Bounds for second call to rotation function (involves tail processing).
    SmallVector<Value, 8> lowerBounds2{c0, inputColMultiple};
    SmallVector<Value, 8> upperBounds2{inputRow, inputCol};

    SmallVector<int64_t, 8> steps{1, stride};
    Value strideTailVal =
        rewriter.create<arith::SubIOp>(loc, inputCol, inputColMultiple);

    // Get input image center.
    Value inputCenterY = getCenter(rewriter, loc, ctx, inputRow);
    Value inputCenterX = getCenter(rewriter, loc, ctx, inputCol);

    Value inputCenterYF32Vec =
        castAndExpand(rewriter, loc, inputCenterY, vectorTy32);
    Value inputCenterXF32Vec =
        castAndExpand(rewriter, loc, inputCenterX, vectorTy32);

    // Get output image center.
    Value outputCenterY = getCenter(rewriter, loc, ctx, outputRow);
    Value outputCenterX = getCenter(rewriter, loc, ctx, outputCol);

    Value outputCenterYF32Vec =
        castAndExpand(rewriter, loc, outputCenterY, vectorTy32);
    Value outputCenterXF32Vec =
        castAndExpand(rewriter, loc, outputCenterX, vectorTy32);

    // Get sin(angle) which will be used in further calculations.
    Value sinVal = rewriter.create<math::SinOp>(loc, angleVal);
    Value sinVec =
        rewriter.create<vector::BroadcastOp>(loc, vectorTy32, sinVal);

    // Get tan(angle / 2) which will be used in further calculations.
    Value tanVal = customTanVal(rewriter, loc, angleVal);
    Value tanVec =
        rewriter.create<vector::BroadcastOp>(loc, vectorTy32, tanVal);

    // Determine the condition for chosing ideal rotation strategy.
    Value tanBound =
        rewriter.create<ConstantFloatOp>(loc, (llvm::APFloat)8.10f, f32);
    Value tanValAbs = rewriter.create<math::AbsOp>(loc, tanVal);
    Value transformCond = rewriter.create<arith::CmpFOp>(
        loc, CmpFPredicate::OGT, tanBound, tanValAbs);

    // For both rotation strategies, tail processing is handled in second call.
    rewriter.create<scf::IfOp>(
        loc, transformCond,
        [&](OpBuilder &builder, Location loc) {
          shearTransformController(
              builder, loc, ctx, lowerBounds1, upperBounds1, steps, strideVal,
              input, output, sinVec, tanVec, inputRowF32Vec, inputColF32Vec,
              inputCenterYF32Vec, inputCenterXF32Vec, outputCenterYF32Vec,
              outputCenterXF32Vec, outputRowLastElemF32, outputColLastElemF32,
              inputRowLastElemF32, inputColLastElemF32, c0, c0F32, c1F32Vec,
              vectorTy32, stride, f32);

          shearTransformController(
              builder, loc, ctx, lowerBounds2, upperBounds2, steps,
              strideTailVal, input, output, sinVec, tanVec, inputRowF32Vec,
              inputColF32Vec, inputCenterYF32Vec, inputCenterXF32Vec,
              outputCenterYF32Vec, outputCenterXF32Vec, outputRowLastElemF32,
              outputColLastElemF32, inputRowLastElemF32, inputColLastElemF32,
              c0, c0F32, c1F32Vec, vectorTy32, stride, f32);

          builder.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &builder, Location loc) {
          standardRotateController(
              builder, loc, ctx, lowerBounds1, upperBounds1, steps, strideVal,
              input, output, sinVec, angleVal, inputRowF32Vec, inputColF32Vec,
              inputCenterYF32Vec, inputCenterXF32Vec, outputCenterYF32Vec,
              outputCenterXF32Vec, outputRowLastElemF32, outputColLastElemF32,
              inputRowLastElemF32, inputColLastElemF32, c0, c0F32, c1F32Vec,
              vectorTy32, stride, f32);

          standardRotateController(
              builder, loc, ctx, lowerBounds2, upperBounds2, steps,
              strideTailVal, input, output, sinVec, angleVal, inputRowF32Vec,
              inputColF32Vec, inputCenterYF32Vec, inputCenterXF32Vec,
              outputCenterYF32Vec, outputCenterXF32Vec, outputRowLastElemF32,
              outputColLastElemF32, inputRowLastElemF32, inputColLastElemF32,
              c0, c0F32, c1F32Vec, vectorTy32, stride, f32);

          builder.create<scf::YieldOp>(loc);
        });

    // Remove the origin rotation operation.
    rewriter.eraseOp(op);
    return success();
  }

  int64_t stride;
};

class DIPResize2DOpLowering : public OpRewritePattern<dip::Resize2DOp> {
public:
  using OpRewritePattern<dip::Resize2DOp>::OpRewritePattern;

  explicit DIPResize2DOpLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
  }

  LogicalResult matchAndRewrite(dip::Resize2DOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Register operand values.
    Value input = op->getOperand(0);
    Value horizontalScalingFactor = op->getOperand(1);
    Value verticalScalingFactor = op->getOperand(2);
    Value output = op->getOperand(3);
    auto interpolationAttr = op.interpolation_type();
    Value strideVal = rewriter.create<ConstantIndexOp>(loc, stride);

    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    Value c0F32 = indexToF32(rewriter, loc, c0);

    Value inputRow = rewriter.create<memref::DimOp>(loc, input, c0);
    Value inputCol = rewriter.create<memref::DimOp>(loc, input, c1);

    Value outputRow = rewriter.create<memref::DimOp>(loc, output, c0);
    Value outputCol = rewriter.create<memref::DimOp>(loc, output, c1);

    // Determine lower bound for second call of rotation function (this is done
    // for efficient tail processing).
    Value outputColStrideRatio =
        rewriter.create<arith::DivUIOp>(loc, outputCol, strideVal);
    Value outputColMultiple =
        rewriter.create<arith::MulIOp>(loc, strideVal, outputColStrideRatio);

    SmallVector<Value, 8> lowerBounds1{c0, c0};
    SmallVector<Value, 8> upperBounds1{outputRow, outputColMultiple};

    SmallVector<int64_t, 8> steps{1, stride};
    Value strideTailVal =
        rewriter.create<arith::SubIOp>(loc, outputCol, outputColMultiple);

    SmallVector<Value, 8> lowerBounds2{c0, outputColMultiple};
    SmallVector<Value, 8> upperBounds2{outputRow, outputCol};

    FloatType f32 = FloatType::getF32(ctx);
    VectorType vectorTy32 = VectorType::get({stride}, f32);

    Value horizontalScalingFactorVec = rewriter.create<vector::SplatOp>(
        loc, vectorTy32, horizontalScalingFactor);
    Value verticalScalingFactorVec = rewriter.create<vector::SplatOp>(
        loc, vectorTy32, verticalScalingFactor);

    // Obtain extreme allocatable value(s) in input and output for bounding
    // purpose.
    Value inputRowLastElem = rewriter.create<arith::SubIOp>(loc, inputRow, c1);
    Value inputRowLastElemF32 = indexToF32(rewriter, loc, inputRowLastElem);

    Value inputColLastElem = rewriter.create<arith::SubIOp>(loc, inputCol, c1);
    Value inputColLastElemF32 = indexToF32(rewriter, loc, inputColLastElem);

    Value outputRowLastElem =
        rewriter.create<arith::SubIOp>(loc, outputRow, c1);
    Value outputRowLastElemF32 = indexToF32(rewriter, loc, outputRowLastElem);

    Value outputColLastElem =
        rewriter.create<arith::SubIOp>(loc, outputCol, c1);
    Value outputColLastElemF32 = indexToF32(rewriter, loc, outputColLastElem);

    if (interpolationAttr ==
        dip::InterpolationType::NearestNeighbourInterpolation) {
      NearestNeighbourInterpolationResizing(
          rewriter, loc, ctx, lowerBounds1, upperBounds1, steps, strideVal,
          input, output, horizontalScalingFactorVec, verticalScalingFactorVec,
          outputRowLastElemF32, outputColLastElemF32, inputRowLastElemF32,
          inputColLastElemF32, vectorTy32, stride, c0, c0F32);

      NearestNeighbourInterpolationResizing(
          rewriter, loc, ctx, lowerBounds2, upperBounds2, steps, strideTailVal,
          input, output, horizontalScalingFactorVec, verticalScalingFactorVec,
          outputRowLastElemF32, outputColLastElemF32, inputRowLastElemF32,
          inputColLastElemF32, vectorTy32, stride, c0, c0F32);
    } else if (interpolationAttr ==
               dip::InterpolationType::BilinearInterpolation) {
      Value c1F32 = indexToF32(rewriter, loc, c1);

      BilinearInterpolationResizing(
          rewriter, loc, ctx, lowerBounds1, upperBounds1, steps, strideVal,
          input, output, horizontalScalingFactorVec, verticalScalingFactorVec,
          outputRowLastElemF32, outputColLastElemF32, inputRowLastElemF32,
          inputColLastElemF32, vectorTy32, stride, c0, c0F32, c1F32);

      BilinearInterpolationResizing(
          rewriter, loc, ctx, lowerBounds2, upperBounds2, steps, strideTailVal,
          input, output, horizontalScalingFactorVec, verticalScalingFactorVec,
          outputRowLastElemF32, outputColLastElemF32, inputRowLastElemF32,
          inputColLastElemF32, vectorTy32, stride, c0, c0F32, c1F32);
    }

    // Remove the original resize operation.
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;
};
class DIPErosion2DOpLowering : public OpRewritePattern<dip::Erosion2DOp>
{

public:
  using OpRewritePattern<dip::Erosion2DOp>::OpRewritePattern;

  explicit DIPErosion2DOpLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
  }

  LogicalResult matchAndRewrite(dip::Erosion2DOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Create constant indices.
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);

    // Register operand values.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    Value centerX = op->getOperand(3);
    Value centerY = op->getOperand(4);
    Value constantValue = op->getOperand(5);
    auto boundaryOptionAttr = op.boundary_option();
    Value strideVal = rewriter.create<ConstantIndexOp>(loc, stride);

    FloatType f32 = FloatType::getF32(ctx);
    IntegerType i1 = IntegerType::get(ctx, 1);

    // Create DimOp.
    Value inputRow = rewriter.create<memref::DimOp>(loc, input, c0);
    Value inputCol = rewriter.create<memref::DimOp>(loc, input, c1);
    Value outputrow = rewriter.create<memref::DimOp>(loc, output, c0);
    Value outputcol = rewriter.create<memref::DimOp>(loc,output,c1);
    Value kernelSize = rewriter.create<memref::DimOp>(loc, kernel, c0);

    // Variables used for detecting rowMid, rowDown, colMid and colRight
    // regions.
    Value rowMidHelper = rewriter.create<AddIOp>(loc, inputRow, centerY);
    Value colMidHelper = rewriter.create<AddIOp>(loc, inputCol, centerX);

    SmallVector<Value, 8> lowerBounds(4, c0);
    SmallVector<Value, 8> uperBounds{inputRow, kernelSize, inputCol,
                                     kernelSize};
    SmallVector<int64_t, 8> steps{1, 1, stride, 1};

    VectorType vectorTy32 = VectorType::get({stride}, f32);
    VectorType VectorOne = VectorType::get({1}, f32);
    VectorType vectorMaskTy = VectorType::get({stride}, i1);

    Value zeroPaddingElem =
        rewriter.create<ConstantFloatOp>(loc, (APFloat)(float)0, f32);
    Value zeroPadding =
        rewriter.create<BroadcastOp>(loc, vectorTy32, zeroPaddingElem);
        Value Paddingel = rewriter.create<ConstantFloatOp>(loc, (APFloat)(float)-256, f32);
        Value PaddingElement = rewriter.create<BroadcastOp>(loc, VectorOne, Paddingel);
        Value Paddingvec = rewriter.create<BroadcastOp>(loc, vectorTy32, Paddingel);

         SmallVector<Value, 8> lowerBounds1(2,c0);
   SmallVector<Value, 8> upperBounds1{outputrow,outputcol};
   SmallVector<int64_t, 8> steps1{1, 1};
   // Initializes the Output memref with -256 
    buildAffineLoopNest(
        rewriter, loc, lowerBounds1, upperBounds1, steps1,
        [&](OpBuilder &builder, Location loc, ValueRange ivs1)
        {
             builder.create<StoreOp>(loc, PaddingElement, output,
                                ValueRange{ivs1[0], ivs1[1]});
        });
        
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
                    dip::BoundaryOption::ConstantPadding) {
                  Value inputVec = builder.create<BroadcastOp>(loc, vectorTy32,
                                                               constantValue);
                                                                Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                
                
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       
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
                        Value leftMask =
                            createInvertedMask(builder, loc, strideVal,
                                               vectorMaskTy, leftMaskElem);

                        if (boundaryOptionAttr ==
                            dip::BoundaryOption::ReplicatePadding) {
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
                         Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                    
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
                         

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
                                  dip::BoundaryOption::ReplicatePadding) {
                                inputVec = builder.create<LoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{c0, imCol});
                              }
                               Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                              
                        
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       
                                 

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
                                  dip::BoundaryOption::ReplicatePadding) {
                                Value rightRange =
                                    builder.create<SubIOp>(loc, inputCol, c1);
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, input, ValueRange{c0, rightRange});
                                Value padding = builder.create<BroadcastOp>(
                                    loc, vectorTy32, paddingVal);

                                inputVec = builder.create<MaskedLoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{c0, imCol}, rightMask, padding);
                              }
                              Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
            
                               
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       
                              

                              builder.create<scf::YieldOp>(loc);
                            });
                        builder.create<scf::YieldOp>(loc);
                      });
                }
                builder.create<scf::YieldOp>(loc);
              },
              [&](OpBuilder &builder, Location loc) {
                // rowMid or rowDown
                Value rowMidCond = builder.create<CmpIOp>(
                    loc, CmpIPredicate::slt, currRow, rowMidHelper);

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
                                dip::BoundaryOption::ConstantPadding) {
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, constantValue);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, input,
                                  ValueRange{imRow, leftPaddingOffset},
                                  leftMask, padding);
                            }  if (boundaryOptionAttr ==
                                       dip::BoundaryOption::ReplicatePadding) {
                              Value paddingVal = builder.create<memref::LoadOp>(
                                  loc, input, ValueRange{imRow, c0});
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, paddingVal);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, input,
                                  ValueRange{imRow, leftPaddingOffset},
                                  leftMask, padding);
                            }
                             Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                        
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       

                            builder.create<scf::YieldOp>(loc);
                          },
                          [&](OpBuilder &builder, Location loc) {
                            // (colMid or colRight) & rowMid
                            Value colMidCond = builder.create<CmpIOp>(
                                loc, CmpIPredicate::sle, colLastElem,
                                colMidHelper);

                            builder.create<scf::IfOp>(
                                loc, colMidCond,
                                [&](OpBuilder &builder, Location loc) {
                                  // colMid & rowMid
                                  Value inputVec = builder.create<LoadOp>(
                                      loc, vectorTy32, input,
                                      ValueRange{imRow, imCol});
                                       Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                            
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       

                                  builder.create<scf::YieldOp>(loc);
                                },
                                [&](OpBuilder &builder, Location loc) {
                                  // colRight & rowMid
                                  Value inputVec;
                                  Value rightMaskHelper =
                                      builder.create<SubIOp>(loc, colLastElem,
                                                             colMidHelper);
                                  Value rightMaskElem = builder.create<SubIOp>(
                                      loc, strideVal, rightMaskHelper);
                                  Value rightMask =
                                      builder.create<CreateMaskOp>(
                                          loc, vectorMaskTy, rightMaskElem);

                                  if (boundaryOptionAttr ==
                                      dip::BoundaryOption::ConstantPadding) {
                                    Value padding = builder.create<BroadcastOp>(
                                        loc, vectorTy32, constantValue);

                                    inputVec = builder.create<MaskedLoadOp>(
                                        loc, vectorTy32, input,
                                        ValueRange{imRow, imCol}, rightMask,
                                        padding);
                                  } else if (boundaryOptionAttr ==
                                             dip::BoundaryOption::
                                                 ReplicatePadding) {
                                    Value rightRange = builder.create<SubIOp>(
                                        loc, inputCol, c1);
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
                                
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        

                                  builder.create<scf::YieldOp>(loc);
                                });
                            builder.create<scf::YieldOp>(loc);
                          });
                      builder.create<scf::YieldOp>(loc);
                    },
                    [&](OpBuilder &builder, Location loc) {
                      // rowDown
                      if (boundaryOptionAttr ==
                          dip::BoundaryOption::ConstantPadding) {
                        Value inputVec = builder.create<BroadcastOp>(
                            loc, vectorTy32, constantValue);
                             Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);


                         
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
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
                              Value leftMask = createInvertedMask(
                                  builder, loc, strideVal, vectorMaskTy,
                                  leftMaskElem);

                              if (boundaryOptionAttr ==
                                  dip::BoundaryOption::ReplicatePadding) {
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, input, ValueRange{downRange, c0});
                                Value padding = builder.create<BroadcastOp>(
                                    loc, vectorTy32, paddingVal);

                                Value leftPaddingOffset =
                                    builder.create<SubIOp>(loc, c0,
                                                           leftMaskElem);
                                inputVec = builder.create<MaskedLoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{downRange, leftPaddingOffset},
                                    leftMask, padding);
                              }
                               Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                    
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                      

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
                                    Value downRange = builder.create<SubIOp>(
                                        loc, inputRow, c1);
                                    if (boundaryOptionAttr ==
                                        dip::BoundaryOption::ReplicatePadding) {
                                      inputVec = builder.create<LoadOp>(
                                          loc, vectorTy32, input,
                                          ValueRange{downRange, imCol});
                                    }
                                     Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);

                                
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
                                        

                                    builder.create<scf::YieldOp>(loc);
                                  },
                                  [&](OpBuilder &builder, Location loc) {
                                    // colRight & rowDown
                                    Value inputVec;
                                    Value rightMaskHelper =
                                        builder.create<SubIOp>(loc, colLastElem,
                                                               colMidHelper);
                                    Value rightMaskElem =
                                        builder.create<SubIOp>(loc, strideVal,
                                                               rightMaskHelper);
                                    Value rightMask =
                                        builder.create<CreateMaskOp>(
                                            loc, vectorMaskTy, rightMaskElem);

                                    Value downRange = builder.create<SubIOp>(
                                        loc, inputRow, c1);
                                    Value rightRange = builder.create<SubIOp>(
                                        loc, inputCol, c1);

                                    if (boundaryOptionAttr ==
                                        dip::BoundaryOption::ReplicatePadding) {

                                      Value paddingVal =
                                          builder.create<memref::LoadOp>(
                                              loc, input,
                                              ValueRange{downRange,
                                                         rightRange});
                                      Value padding =
                                          builder.create<vector::BroadcastOp>(
                                              loc, vectorTy32, paddingVal);

                                      inputVec = builder.create<MaskedLoadOp>(
                                          loc, vectorTy32, input,
                                          ValueRange{downRange, imCol},
                                          rightMask, padding);
                                    }
                                    Value tailCond = tailChecker(
                                        builder, loc, calcHelper, strideVal,
                                        kernelSize, c1, pseudoCol, ivs[2]);

                            
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       
                                     

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
    // Remove the origin convolution operation.
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;

};

class DIPDilation2DOpLowering : public OpRewritePattern<dip::Dilation2DOp>
{

public:
  using OpRewritePattern<dip::Dilation2DOp>::OpRewritePattern;

  explicit DIPDilation2DOpLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
  }

  LogicalResult matchAndRewrite(dip::Dilation2DOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Create constant indices.
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);

    // Register operand values.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    Value centerX = op->getOperand(3);
    Value centerY = op->getOperand(4);
    Value constantValue = op->getOperand(5);
    auto boundaryOptionAttr = op.boundary_option();
    Value strideVal = rewriter.create<ConstantIndexOp>(loc, stride);

    FloatType f32 = FloatType::getF32(ctx);
    IntegerType i1 = IntegerType::get(ctx, 1);

    // Create DimOp.
    Value inputRow = rewriter.create<memref::DimOp>(loc, input, c0);
    Value inputCol = rewriter.create<memref::DimOp>(loc, input, c1);
    Value outputrow = rewriter.create<memref::DimOp>(loc, output, c0);
    Value outputcol = rewriter.create<memref::DimOp>(loc,output,c1);
    Value kernelSize = rewriter.create<memref::DimOp>(loc, kernel, c0);

    // Variables used for detecting rowMid, rowDown, colMid and colRight
    // regions.
    Value rowMidHelper = rewriter.create<AddIOp>(loc, inputRow, centerY);
    Value colMidHelper = rewriter.create<AddIOp>(loc, inputCol, centerX);

    SmallVector<Value, 8> lowerBounds(4, c0);
    SmallVector<Value, 8> uperBounds{inputRow, kernelSize, inputCol,
                                     kernelSize};
    SmallVector<int64_t, 8> steps{1, 1, stride, 1};

    VectorType vectorTy32 = VectorType::get({stride}, f32);
    VectorType VectorOne = VectorType::get({1}, f32);
    VectorType vectorMaskTy = VectorType::get({stride}, i1);

    Value zeroPaddingElem =
        rewriter.create<ConstantFloatOp>(loc, (APFloat)(float)0, f32);
    Value zeroPadding =
        rewriter.create<BroadcastOp>(loc, vectorTy32, zeroPaddingElem);
        Value Paddingel = rewriter.create<ConstantFloatOp>(loc, (APFloat)(float)-256, f32);
        Value PaddingElement = rewriter.create<BroadcastOp>(loc, VectorOne, Paddingel);
        Value Paddingvec = rewriter.create<BroadcastOp>(loc, vectorTy32, Paddingel);

         SmallVector<Value, 8> lowerBounds1(2,c0);
   SmallVector<Value, 8> upperBounds1{outputrow,outputcol};
   SmallVector<int64_t, 8> steps1{1, 1};
   // Initializes the Output memref with -256 
    buildAffineLoopNest(
        rewriter, loc, lowerBounds1, upperBounds1, steps1,
        [&](OpBuilder &builder, Location loc, ValueRange ivs1)
        {
             builder.create<StoreOp>(loc, PaddingElement, output,
                                ValueRange{ivs1[0], ivs1[1]});
        });

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
                    dip::BoundaryOption::ConstantPadding) {
                  Value inputVec = builder.create<BroadcastOp>(loc, vectorTy32,
                                                               constantValue);
                                                                Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                    
            
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                      
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
                        Value leftMask =
                            createInvertedMask(builder, loc, strideVal,
                                               vectorMaskTy, leftMaskElem);

                        if (boundaryOptionAttr ==
                            dip::BoundaryOption::ReplicatePadding) {
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
                         Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                        
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
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
                                  dip::BoundaryOption::ReplicatePadding) {
                                inputVec = builder.create<LoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{c0, imCol});
                              }
                               Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                        
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        

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
                                  dip::BoundaryOption::ReplicatePadding) {
                                Value rightRange =
                                    builder.create<SubIOp>(loc, inputCol, c1);
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, input, ValueRange{c0, rightRange});
                                Value padding = builder.create<BroadcastOp>(
                                    loc, vectorTy32, paddingVal);

                                inputVec = builder.create<MaskedLoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{c0, imCol}, rightMask, padding);
                              }
                              Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);

                               
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       


                              builder.create<scf::YieldOp>(loc);
                            });
                        builder.create<scf::YieldOp>(loc);
                      });
                }
                builder.create<scf::YieldOp>(loc);
              },
              [&](OpBuilder &builder, Location loc) {
                // rowMid or rowDown
                Value rowMidCond = builder.create<CmpIOp>(
                    loc, CmpIPredicate::slt, currRow, rowMidHelper);

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
                                dip::BoundaryOption::ConstantPadding) {
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, constantValue);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, input,
                                  ValueRange{imRow, leftPaddingOffset},
                                  leftMask, padding);
                            }  if (boundaryOptionAttr ==
                                       dip::BoundaryOption::ReplicatePadding) {
                              Value paddingVal = builder.create<memref::LoadOp>(
                                  loc, input, ValueRange{imRow, c0});
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, paddingVal);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, input,
                                  ValueRange{imRow, leftPaddingOffset},
                                  leftMask, padding);
                            }
                             Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                            
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       
                            builder.create<scf::YieldOp>(loc);
                          },
                          [&](OpBuilder &builder, Location loc) {
                            // (colMid or colRight) & rowMid
                            Value colMidCond = builder.create<CmpIOp>(
                                loc, CmpIPredicate::sle, colLastElem,
                                colMidHelper);

                            builder.create<scf::IfOp>(
                                loc, colMidCond,
                                [&](OpBuilder &builder, Location loc) {
                                  // colMid & rowMid
                                  Value inputVec = builder.create<LoadOp>(
                                      loc, vectorTy32, input,
                                      ValueRange{imRow, imCol});
                                       Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                                    
                                 
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        


                                  builder.create<scf::YieldOp>(loc);
                                },
                                [&](OpBuilder &builder, Location loc) {
                                  // colRight & rowMid
                                  Value inputVec;
                                  Value rightMaskHelper =
                                      builder.create<SubIOp>(loc, colLastElem,
                                                             colMidHelper);
                                  Value rightMaskElem = builder.create<SubIOp>(
                                      loc, strideVal, rightMaskHelper);
                                  Value rightMask =
                                      builder.create<CreateMaskOp>(
                                          loc, vectorMaskTy, rightMaskElem);

                                  if (boundaryOptionAttr ==
                                      dip::BoundaryOption::ConstantPadding) {
                                    Value padding = builder.create<BroadcastOp>(
                                        loc, vectorTy32, constantValue);

                                    inputVec = builder.create<MaskedLoadOp>(
                                        loc, vectorTy32, input,
                                        ValueRange{imRow, imCol}, rightMask,
                                        padding);
                                  } else if (boundaryOptionAttr ==
                                             dip::BoundaryOption::
                                                 ReplicatePadding) {
                                    Value rightRange = builder.create<SubIOp>(
                                        loc, inputCol, c1);
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
                                    
                                
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        

                                  builder.create<scf::YieldOp>(loc);
                                });
                            builder.create<scf::YieldOp>(loc);
                          });
                      builder.create<scf::YieldOp>(loc);
                    },
                    [&](OpBuilder &builder, Location loc) {
                      // rowDown
                      if (boundaryOptionAttr ==
                          dip::BoundaryOption::ConstantPadding) {
                        Value inputVec = builder.create<BroadcastOp>(
                            loc, vectorTy32, constantValue);
                             Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);

                          
                
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       
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
                              Value leftMask = createInvertedMask(
                                  builder, loc, strideVal, vectorMaskTy,
                                  leftMaskElem);

                              if (boundaryOptionAttr ==
                                  dip::BoundaryOption::ReplicatePadding) {
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, input, ValueRange{downRange, c0});
                                Value padding = builder.create<BroadcastOp>(
                                    loc, vectorTy32, paddingVal);

                                Value leftPaddingOffset =
                                    builder.create<SubIOp>(loc, c0,
                                                           leftMaskElem);
                                inputVec = builder.create<MaskedLoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{downRange, leftPaddingOffset},
                                    leftMask, padding);
                              }
                               Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
 
                                
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        

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
                                    Value downRange = builder.create<SubIOp>(
                                        loc, inputRow, c1);
                                    if (boundaryOptionAttr ==
                                        dip::BoundaryOption::ReplicatePadding) {
                                      inputVec = builder.create<LoadOp>(
                                          loc, vectorTy32, input,
                                          ValueRange{downRange, imCol});
                                    }
                                     Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                                   
                                
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        

                                    builder.create<scf::YieldOp>(loc);
                                  },
                                  [&](OpBuilder &builder, Location loc) {
                                    // colRight & rowDown
                                    Value inputVec;
                                    Value rightMaskHelper =
                                        builder.create<SubIOp>(loc, colLastElem,
                                                               colMidHelper);
                                    Value rightMaskElem =
                                        builder.create<SubIOp>(loc, strideVal,
                                                               rightMaskHelper);
                                    Value rightMask =
                                        builder.create<CreateMaskOp>(
                                            loc, vectorMaskTy, rightMaskElem);

                                    Value downRange = builder.create<SubIOp>(
                                        loc, inputRow, c1);
                                    Value rightRange = builder.create<SubIOp>(
                                        loc, inputCol, c1);

                                    if (boundaryOptionAttr ==
                                        dip::BoundaryOption::ReplicatePadding) {

                                      Value paddingVal =
                                          builder.create<memref::LoadOp>(
                                              loc, input,
                                              ValueRange{downRange,
                                                         rightRange});
                                      Value padding =
                                          builder.create<vector::BroadcastOp>(
                                              loc, vectorTy32, paddingVal);

                                      inputVec = builder.create<MaskedLoadOp>(
                                          loc, vectorTy32, input,
                                          ValueRange{downRange, imCol},
                                          rightMask, padding);
                                    }
                                    Value tailCond = tailChecker(
                                        builder, loc, calcHelper, strideVal,
                                        kernelSize, c1, pseudoCol, ivs[2]);
                                    
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
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
    // Remove the origin convolution operation.
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;

};

class DIPOpening2DOpLowering : public OpRewritePattern<dip::Opening2DOp>
{
public:
  using OpRewritePattern<dip::Opening2DOp>::OpRewritePattern;

  explicit DIPOpening2DOpLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
  }

  LogicalResult matchAndRewrite(dip::Opening2DOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Create constant indices.
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);

    // Register operand values.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    Value output1 = op->getOperand(3);
    Value centerX = op->getOperand(4);
    Value centerY = op->getOperand(5);
    Value constantValue = op->getOperand(6);
    auto boundaryOptionAttr = op.boundary_option();
    Value strideVal = rewriter.create<ConstantIndexOp>(loc, stride);

    FloatType f32 = FloatType::getF32(ctx);
    IntegerType i1 = IntegerType::get(ctx, 1);

    // Create DimOp.
    Value inputRow = rewriter.create<memref::DimOp>(loc, input, c0);
    Value inputCol = rewriter.create<memref::DimOp>(loc, input, c1);
    Value outputrow = rewriter.create<memref::DimOp>(loc, output1, c0);
    Value outputcol = rewriter.create<memref::DimOp>(loc,output1,c1);
    Value kernelSize = rewriter.create<memref::DimOp>(loc, kernel, c0);

    // Variables used for detecting rowMid, rowDown, colMid and colRight
    // regions.
    Value rowMidHelper = rewriter.create<AddIOp>(loc, inputRow, centerY);
    Value colMidHelper = rewriter.create<AddIOp>(loc, inputCol, centerX);

    SmallVector<Value, 8> lowerBounds(4, c0);
    SmallVector<Value, 8> uperBounds{inputRow, kernelSize, inputCol,
                                     kernelSize};
    SmallVector<int64_t, 8> steps{1, 1, stride, 1};

    VectorType vectorTy32 = VectorType::get({stride}, f32);
    VectorType VectorOne = VectorType::get({1}, f32);
    VectorType vectorMaskTy = VectorType::get({stride}, i1);

    Value zeroPaddingElem =
        rewriter.create<ConstantFloatOp>(loc, (APFloat)(float)0, f32);
    Value zeroPadding =
        rewriter.create<BroadcastOp>(loc, vectorTy32, zeroPaddingElem);
        Value Paddingel = rewriter.create<ConstantFloatOp>(loc, (APFloat)(float)-256, f32);
        Value PaddingElement = rewriter.create<BroadcastOp>(loc, VectorOne, Paddingel);
        Value Paddingvec = rewriter.create<BroadcastOp>(loc, vectorTy32, Paddingel);

         SmallVector<Value, 8> lowerBounds1(2,c0);
   SmallVector<Value, 8> upperBounds1{outputrow,outputcol};
   SmallVector<int64_t, 8> steps1{1, 1};
   // Initializes the Output memref with -256 
    buildAffineLoopNest(
        rewriter, loc, lowerBounds1, upperBounds1, steps1,
        [&](OpBuilder &builder, Location loc, ValueRange ivs1)
        {
             builder.create<StoreOp>(loc, PaddingElement, output1,
                                ValueRange{ivs1[0], ivs1[1]});
        });

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
                    dip::BoundaryOption::ConstantPadding) {
                  Value inputVec = builder.create<BroadcastOp>(loc, vectorTy32,
                                                               constantValue);
                                                                Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                
                
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
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
                        Value leftMask =
                            createInvertedMask(builder, loc, strideVal,
                                               vectorMaskTy, leftMaskElem);

                        if (boundaryOptionAttr ==
                            dip::BoundaryOption::ReplicatePadding) {
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
                         Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                        
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
                         

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
                                  dip::BoundaryOption::ReplicatePadding) {
                                inputVec = builder.create<LoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{c0, imCol});
                              }
                               Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                              
                        
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
                                 

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
                                  dip::BoundaryOption::ReplicatePadding) {
                                Value rightRange =
                                    builder.create<SubIOp>(loc, inputCol, c1);
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, input, ValueRange{c0, rightRange});
                                Value padding = builder.create<BroadcastOp>(
                                    loc, vectorTy32, paddingVal);

                                inputVec = builder.create<MaskedLoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{c0, imCol}, rightMask, padding);
                              }
                              Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
            
                            
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
                              

                              builder.create<scf::YieldOp>(loc);
                            });
                        builder.create<scf::YieldOp>(loc);
                      });
                }
                builder.create<scf::YieldOp>(loc);
              },
              [&](OpBuilder &builder, Location loc) {
                // rowMid or rowDown
                Value rowMidCond = builder.create<CmpIOp>(
                    loc, CmpIPredicate::slt, currRow, rowMidHelper);

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
                                dip::BoundaryOption::ConstantPadding) {
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, constantValue);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, input,
                                  ValueRange{imRow, leftPaddingOffset},
                                  leftMask, padding);
                            }  if (boundaryOptionAttr ==
                                       dip::BoundaryOption::ReplicatePadding) {
                              Value paddingVal = builder.create<memref::LoadOp>(
                                  loc, input, ValueRange{imRow, c0});
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, paddingVal);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, input,
                                  ValueRange{imRow, leftPaddingOffset},
                                  leftMask, padding);
                            }
                             Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                              
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       

                            builder.create<scf::YieldOp>(loc);
                          },
                          [&](OpBuilder &builder, Location loc) {
                            // (colMid or colRight) & rowMid
                            Value colMidCond = builder.create<CmpIOp>(
                                loc, CmpIPredicate::sle, colLastElem,
                                colMidHelper);

                            builder.create<scf::IfOp>(
                                loc, colMidCond,
                                [&](OpBuilder &builder, Location loc) {
                                  // colMid & rowMid
                                  Value inputVec = builder.create<LoadOp>(
                                      loc, vectorTy32, input,
                                      ValueRange{imRow, imCol});
                                       Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                            
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       

                                  builder.create<scf::YieldOp>(loc);
                                },
                                [&](OpBuilder &builder, Location loc) {
                                  // colRight & rowMid
                                  Value inputVec;
                                  Value rightMaskHelper =
                                      builder.create<SubIOp>(loc, colLastElem,
                                                             colMidHelper);
                                  Value rightMaskElem = builder.create<SubIOp>(
                                      loc, strideVal, rightMaskHelper);
                                  Value rightMask =
                                      builder.create<CreateMaskOp>(
                                          loc, vectorMaskTy, rightMaskElem);

                                  if (boundaryOptionAttr ==
                                      dip::BoundaryOption::ConstantPadding) {
                                    Value padding = builder.create<BroadcastOp>(
                                        loc, vectorTy32, constantValue);

                                    inputVec = builder.create<MaskedLoadOp>(
                                        loc, vectorTy32, input,
                                        ValueRange{imRow, imCol}, rightMask,
                                        padding);
                                  } else if (boundaryOptionAttr ==
                                             dip::BoundaryOption::
                                                 ReplicatePadding) {
                                    Value rightRange = builder.create<SubIOp>(
                                        loc, inputCol, c1);
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
                                   
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
                                    

                                  builder.create<scf::YieldOp>(loc);
                                });
                            builder.create<scf::YieldOp>(loc);
                          });
                      builder.create<scf::YieldOp>(loc);
                    },
                    [&](OpBuilder &builder, Location loc) {
                      // rowDown
                      if (boundaryOptionAttr ==
                          dip::BoundaryOption::ConstantPadding) {
                        Value inputVec = builder.create<BroadcastOp>(
                            loc, vectorTy32, constantValue);
                             Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);


                    
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       
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
                              Value leftMask = createInvertedMask(
                                  builder, loc, strideVal, vectorMaskTy,
                                  leftMaskElem);

                              if (boundaryOptionAttr ==
                                  dip::BoundaryOption::ReplicatePadding) {
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, input, ValueRange{downRange, c0});
                                Value padding = builder.create<BroadcastOp>(
                                    loc, vectorTy32, paddingVal);

                                Value leftPaddingOffset =
                                    builder.create<SubIOp>(loc, c0,
                                                           leftMaskElem);
                                inputVec = builder.create<MaskedLoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{downRange, leftPaddingOffset},
                                    leftMask, padding);
                              }
                               Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                        
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        

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
                                    Value downRange = builder.create<SubIOp>(
                                        loc, inputRow, c1);
                                    if (boundaryOptionAttr ==
                                        dip::BoundaryOption::ReplicatePadding) {
                                      inputVec = builder.create<LoadOp>(
                                          loc, vectorTy32, input,
                                          ValueRange{downRange, imCol});
                                    }
                                     Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);

                                
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                    

                                    builder.create<scf::YieldOp>(loc);
                                  },
                                  [&](OpBuilder &builder, Location loc) {
                                    // colRight & rowDown
                                    Value inputVec;
                                    Value rightMaskHelper =
                                        builder.create<SubIOp>(loc, colLastElem,
                                                               colMidHelper);
                                    Value rightMaskElem =
                                        builder.create<SubIOp>(loc, strideVal,
                                                               rightMaskHelper);
                                    Value rightMask =
                                        builder.create<CreateMaskOp>(
                                            loc, vectorMaskTy, rightMaskElem);

                                    Value downRange = builder.create<SubIOp>(
                                        loc, inputRow, c1);
                                    Value rightRange = builder.create<SubIOp>(
                                        loc, inputCol, c1);

                                    if (boundaryOptionAttr ==
                                        dip::BoundaryOption::ReplicatePadding) {

                                      Value paddingVal =
                                          builder.create<memref::LoadOp>(
                                              loc, input,
                                              ValueRange{downRange,
                                                         rightRange});
                                      Value padding =
                                          builder.create<vector::BroadcastOp>(
                                              loc, vectorTy32, paddingVal);

                                      inputVec = builder.create<MaskedLoadOp>(
                                          loc, vectorTy32, input,
                                          ValueRange{downRange, imCol},
                                          rightMask, padding);
                                    }
                                    Value tailCond = tailChecker(
                                        builder, loc, calcHelper, strideVal,
                                        kernelSize, c1, pseudoCol, ivs[2]);

                                    
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       
                                     

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

        Value inputRow1 = rewriter.create<memref::DimOp>(loc,output1, c0);
    Value inputCol1 = rewriter.create<memref::DimOp>(loc, output1, c1);
    Value outputrow1 = rewriter.create<memref::DimOp>(loc, output, c0);
    Value outputcol1 = rewriter.create<memref::DimOp>(loc,output,c1);


    // Variables used for detecting rowMid, rowDown, colMid and colRight
    // regions.
    Value rowMidHelper1 = rewriter.create<AddIOp>(loc, inputRow1, centerY);
    Value colMidHelper1 = rewriter.create<AddIOp>(loc, inputCol1, centerX);

    SmallVector<Value, 8> lowerBounds2(4, c0);
    SmallVector<Value, 8> uperBounds2{inputRow1, kernelSize, inputCol1,
                                     kernelSize};
    SmallVector<int64_t, 8> steps2{1, 1, stride, 1};



         SmallVector<Value, 8> lowerBounds3(2,c0);
   SmallVector<Value, 8> upperBounds3{outputrow1,outputcol1};
   SmallVector<int64_t, 8> steps3{1, 1};
   // Initializes the Output memref with -256 
    buildAffineLoopNest(
        rewriter, loc, lowerBounds3, upperBounds3, steps3,
        [&](OpBuilder &builder, Location loc, ValueRange ivs3)
        {
             builder.create<StoreOp>(loc, PaddingElement, output,
                                ValueRange{ivs3[0], ivs3[1]});
        });

  

    Value pseudoCol1 = rewriter.create<AffineApplyOp>(
        loc, calcHelper, ValueRange{inputCol1, kernelSize, c1});

    buildAffineLoopNest(
        rewriter, loc, lowerBounds2, uperBounds2, steps2,
        [&](OpBuilder &builder, Location loc, ValueRange ivs2) {
          // Indices of current pixel with respect to pseudo image containing
          // extrapolated boundaries.
          Value currRow = builder.create<AddIOp>(loc, ivs2[0], ivs2[1]);
          Value currCol = builder.create<AddIOp>(loc, ivs2[2], ivs2[3]);

          Value kernelValue = builder.create<memref::LoadOp>(
              loc, kernel, ValueRange{ivs2[1], ivs2[3]});
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
                    dip::BoundaryOption::ConstantPadding) {
                  Value inputVec = builder.create<BroadcastOp>(loc, vectorTy32,
                                                               constantValue);
                                                                Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol1, ivs2[2]);
                    
                    
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol1,
                                        vectorMaskTy);
                                        
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
                        Value leftMask =
                            createInvertedMask(builder, loc, strideVal,
                                               vectorMaskTy, leftMaskElem);

                        if (boundaryOptionAttr ==
                            dip::BoundaryOption::ReplicatePadding) {
                          Value paddingVal = builder.create<memref::LoadOp>(
                              loc, output1, ValueRange{c0, c0});
                          Value padding = builder.create<BroadcastOp>(
                              loc, vectorTy32, paddingVal);

                          Value leftPaddingOffset =
                              builder.create<SubIOp>(loc, c0, leftMaskElem);
                          inputVec = builder.create<vector::MaskedLoadOp>(
                              loc, vectorTy32, output1,
                              ValueRange{c0, leftPaddingOffset}, leftMask,
                              padding);
                        }
                         Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol1, ivs2[2]);
                        
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol1,
                                        vectorMaskTy);
                                       
                        builder.create<scf::YieldOp>(loc);
                      },
                      [&](OpBuilder &builder, Location loc) {
                        // (colMid or colRight) & rowUp
                        Value colMidCond = builder.create<CmpIOp>(
                            loc, CmpIPredicate::sle, colLastElem, colMidHelper1);

                        builder.create<scf::IfOp>(
                            loc, colMidCond,
                            [&](OpBuilder &builder, Location loc) {
                              // colMid & rowUp
                              Value inputVec;
                              if (boundaryOptionAttr ==
                                  dip::BoundaryOption::ReplicatePadding) {
                                inputVec = builder.create<LoadOp>(
                                    loc, vectorTy32, output1,
                                    ValueRange{c0, imCol});
                              }
                               Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol1, ivs2[2]);
                            
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol1,
                                        vectorMaskTy);
                                        

                              builder.create<scf::YieldOp>(loc);
                            },
                            [&](OpBuilder &builder, Location loc) {
                              // colRight & rowUp
                              Value inputVec;
                              Value rightMaskHelper = builder.create<SubIOp>(
                                  loc, colLastElem, colMidHelper1);
                              Value rightMaskElem = builder.create<SubIOp>(
                                  loc, strideVal, rightMaskHelper);
                              Value rightMask = builder.create<CreateMaskOp>(
                                  loc, vectorMaskTy, rightMaskElem);

                              if (boundaryOptionAttr ==
                                  dip::BoundaryOption::ReplicatePadding) {
                                Value rightRange =
                                    builder.create<SubIOp>(loc, inputCol1, c1);
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, output1, ValueRange{c0, rightRange});
                                Value padding = builder.create<BroadcastOp>(
                                    loc, vectorTy32, paddingVal);

                                inputVec = builder.create<MaskedLoadOp>(
                                    loc, vectorTy32, output1,
                                    ValueRange{c0, imCol}, rightMask, padding);
                              }
                              Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol1, ivs2[2]);

                        
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol1,
                                        vectorMaskTy);
                                        


                              builder.create<scf::YieldOp>(loc);
                            });
                        builder.create<scf::YieldOp>(loc);
                      });
                }
                builder.create<scf::YieldOp>(loc);
              },
              [&](OpBuilder &builder, Location loc) {
                // rowMid or rowDown
                Value rowMidCond = builder.create<CmpIOp>(
                    loc, CmpIPredicate::slt, currRow, rowMidHelper1);

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
                                dip::BoundaryOption::ConstantPadding) {
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, constantValue);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, output1,
                                  ValueRange{imRow, leftPaddingOffset},
                                  leftMask, padding);
                            }  if (boundaryOptionAttr ==
                                       dip::BoundaryOption::ReplicatePadding) {
                              Value paddingVal = builder.create<memref::LoadOp>(
                                  loc, output1, ValueRange{imRow, c0});
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, paddingVal);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, output1,
                                  ValueRange{imRow, leftPaddingOffset},
                                  leftMask, padding);
                            }
                             Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs2[2]);
                            
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       
                            builder.create<scf::YieldOp>(loc);
                          },
                          [&](OpBuilder &builder, Location loc) {
                            // (colMid or colRight) & rowMid
                            Value colMidCond = builder.create<CmpIOp>(
                                loc, CmpIPredicate::sle, colLastElem,
                                colMidHelper1);

                            builder.create<scf::IfOp>(
                                loc, colMidCond,
                                [&](OpBuilder &builder, Location loc) {
                                  // colMid & rowMid
                                  Value inputVec = builder.create<LoadOp>(
                                      loc, vectorTy32, output1,
                                      ValueRange{imRow, imCol});
                                       Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol1, ivs2[2]);
                                    
                                 
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        


                                  builder.create<scf::YieldOp>(loc);
                                },
                                [&](OpBuilder &builder, Location loc) {
                                  // colRight & rowMid
                                  Value inputVec;
                                  Value rightMaskHelper =
                                      builder.create<SubIOp>(loc, colLastElem,
                                                             colMidHelper1);
                                  Value rightMaskElem = builder.create<SubIOp>(
                                      loc, strideVal, rightMaskHelper);
                                  Value rightMask =
                                      builder.create<CreateMaskOp>(
                                          loc, vectorMaskTy, rightMaskElem);

                                  if (boundaryOptionAttr ==
                                      dip::BoundaryOption::ConstantPadding) {
                                    Value padding = builder.create<BroadcastOp>(
                                        loc, vectorTy32, constantValue);

                                    inputVec = builder.create<MaskedLoadOp>(
                                        loc, vectorTy32, output1,
                                        ValueRange{imRow, imCol}, rightMask,
                                        padding);
                                  } else if (boundaryOptionAttr ==
                                             dip::BoundaryOption::
                                                 ReplicatePadding) {
                                    Value rightRange = builder.create<SubIOp>(
                                        loc, inputCol, c1);
                                    Value paddingVal =
                                        builder.create<memref::LoadOp>(
                                            loc, output1,
                                            ValueRange{imRow, rightRange});
                                    Value padding = builder.create<BroadcastOp>(
                                        loc, vectorTy32, paddingVal);

                                    inputVec = builder.create<MaskedLoadOp>(
                                        loc, vectorTy32, output1,
                                        ValueRange{imRow, imCol}, rightMask,
                                        padding);
                                  }
                                  Value tailCond = tailChecker(
                                      builder, loc, calcHelper, strideVal,
                                      kernelSize, c1, pseudoCol, ivs2[2]);
                                    
                                
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        

                                  builder.create<scf::YieldOp>(loc);
                                });
                            builder.create<scf::YieldOp>(loc);
                          });
                      builder.create<scf::YieldOp>(loc);
                    },
                    [&](OpBuilder &builder, Location loc) {
                      // rowDown
                      if (boundaryOptionAttr ==
                          dip::BoundaryOption::ConstantPadding) {
                        Value inputVec = builder.create<BroadcastOp>(
                            loc, vectorTy32, constantValue);
                             Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs2[2]);

                          
                        
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
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
                              Value leftMask = createInvertedMask(
                                  builder, loc, strideVal, vectorMaskTy,
                                  leftMaskElem);

                              if (boundaryOptionAttr ==
                                  dip::BoundaryOption::ReplicatePadding) {
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, input, ValueRange{downRange, c0});
                                Value padding = builder.create<BroadcastOp>(
                                    loc, vectorTy32, paddingVal);

                                Value leftPaddingOffset =
                                    builder.create<SubIOp>(loc, c0,
                                                           leftMaskElem);
                                inputVec = builder.create<MaskedLoadOp>(
                                    loc, vectorTy32, output1,
                                    ValueRange{downRange, leftPaddingOffset},
                                    leftMask, padding);
                              }
                               Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs2[2]);
 
                        
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       

                              builder.create<scf::YieldOp>(loc);
                            },
                            [&](OpBuilder &builder, Location loc) {
                              // (colMid or colRight) & rowDown
                              Value colMidCond = builder.create<CmpIOp>(
                                  loc, CmpIPredicate::sle, colLastElem,
                                  colMidHelper1);

                              builder.create<scf::IfOp>(
                                  loc, colMidCond,
                                  [&](OpBuilder &builder, Location loc) {
                                    // colMid & rowDown
                                    Value inputVec;
                                    Value downRange = builder.create<SubIOp>(
                                        loc, inputRow, c1);
                                    if (boundaryOptionAttr ==
                                        dip::BoundaryOption::ReplicatePadding) {
                                      inputVec = builder.create<LoadOp>(
                                          loc, vectorTy32, output1,
                                          ValueRange{downRange, imCol});
                                    }
                                     Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs2[2]);
                                   
                                
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        

                                    builder.create<scf::YieldOp>(loc);
                                  },
                                  [&](OpBuilder &builder, Location loc) {
                                    // colRight & rowDown
                                    Value inputVec;
                                    Value rightMaskHelper =
                                        builder.create<SubIOp>(loc, colLastElem,
                                                               colMidHelper1);
                                    Value rightMaskElem =
                                        builder.create<SubIOp>(loc, strideVal,
                                                               rightMaskHelper);
                                    Value rightMask =
                                        builder.create<CreateMaskOp>(
                                            loc, vectorMaskTy, rightMaskElem);

                                    Value downRange = builder.create<SubIOp>(
                                        loc, inputRow, c1);
                                    Value rightRange = builder.create<SubIOp>(
                                        loc, inputCol, c1);

                                    if (boundaryOptionAttr ==
                                        dip::BoundaryOption::ReplicatePadding) {

                                      Value paddingVal =
                                          builder.create<memref::LoadOp>(
                                              loc, output1,
                                              ValueRange{downRange,
                                                         rightRange});
                                      Value padding =
                                          builder.create<vector::BroadcastOp>(
                                              loc, vectorTy32, paddingVal);

                                      inputVec = builder.create<MaskedLoadOp>(
                                          loc, vectorTy32, output1,
                                          ValueRange{downRange, imCol},
                                          rightMask, padding);
                                    }
                                    Value tailCond = tailChecker(
                                        builder, loc, calcHelper, strideVal,
                                        kernelSize, c1, pseudoCol, ivs2[2]);
                                
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
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

    // Remove the origin convolution operation.
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;


};

class DIPClosing2DOpLowering : public OpRewritePattern<dip::Closing2DOp>
{
public:
  using OpRewritePattern<dip::Closing2DOp>::OpRewritePattern;

  explicit DIPClosing2DOpLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
  }

  LogicalResult matchAndRewrite(dip::Closing2DOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Create constant indices.
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);

    // Register operand values.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    Value output1 = op->getOperand(3);
    Value centerX = op->getOperand(4);
    Value centerY = op->getOperand(5);
    Value constantValue = op->getOperand(6);
    auto boundaryOptionAttr = op.boundary_option();
    Value strideVal = rewriter.create<ConstantIndexOp>(loc, stride);

    FloatType f32 = FloatType::getF32(ctx);
    IntegerType i1 = IntegerType::get(ctx, 1);

    // Create DimOp.
    Value inputRow = rewriter.create<memref::DimOp>(loc, input, c0);
    Value inputCol = rewriter.create<memref::DimOp>(loc, input, c1);
    Value outputrow = rewriter.create<memref::DimOp>(loc, output1, c0);
    Value outputcol = rewriter.create<memref::DimOp>(loc,output1,c1);
    Value kernelSize = rewriter.create<memref::DimOp>(loc, kernel, c0);

    // Variables used for detecting rowMid, rowDown, colMid and colRight
    // regions.
    Value rowMidHelper = rewriter.create<AddIOp>(loc, inputRow, centerY);
    Value colMidHelper = rewriter.create<AddIOp>(loc, inputCol, centerX);

    SmallVector<Value, 8> lowerBounds(4, c0);
    SmallVector<Value, 8> uperBounds{inputRow, kernelSize, inputCol,
                                     kernelSize};
    SmallVector<int64_t, 8> steps{1, 1, stride, 1};

    VectorType vectorTy32 = VectorType::get({stride}, f32);
    VectorType VectorOne = VectorType::get({1}, f32);
    VectorType vectorMaskTy = VectorType::get({stride}, i1);

    Value zeroPaddingElem =
        rewriter.create<ConstantFloatOp>(loc, (APFloat)(float)0, f32);
    Value zeroPadding =
        rewriter.create<BroadcastOp>(loc, vectorTy32, zeroPaddingElem);
        Value Paddingel = rewriter.create<ConstantFloatOp>(loc, (APFloat)(float)-256, f32);
        Value PaddingElement = rewriter.create<BroadcastOp>(loc, VectorOne, Paddingel);
        Value Paddingvec = rewriter.create<BroadcastOp>(loc, vectorTy32, Paddingel);

         SmallVector<Value, 8> lowerBounds1(2,c0);
   SmallVector<Value, 8> upperBounds1{outputrow,outputcol};
   SmallVector<int64_t, 8> steps1{1, 1};
   // Initializes the Output memref with -256 
    buildAffineLoopNest(
        rewriter, loc, lowerBounds1, upperBounds1, steps1,
        [&](OpBuilder &builder, Location loc, ValueRange ivs1)
        {
             builder.create<StoreOp>(loc, PaddingElement, output1,
                                ValueRange{ivs1[0], ivs1[1]});
        });

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
                    dip::BoundaryOption::ConstantPadding) {
                  Value inputVec = builder.create<BroadcastOp>(loc, vectorTy32,
                                                               constantValue);
                                                                Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                
                
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
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
                        Value leftMask =
                            createInvertedMask(builder, loc, strideVal,
                                               vectorMaskTy, leftMaskElem);

                        if (boundaryOptionAttr ==
                            dip::BoundaryOption::ReplicatePadding) {
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
                         Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                    
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
                         

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
                                  dip::BoundaryOption::ReplicatePadding) {
                                inputVec = builder.create<LoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{c0, imCol});
                              }
                               Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                              
                
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
                                 

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
                                  dip::BoundaryOption::ReplicatePadding) {
                                Value rightRange =
                                    builder.create<SubIOp>(loc, inputCol, c1);
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, input, ValueRange{c0, rightRange});
                                Value padding = builder.create<BroadcastOp>(
                                    loc, vectorTy32, paddingVal);

                                inputVec = builder.create<MaskedLoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{c0, imCol}, rightMask, padding);
                              }
                              Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
            
                            
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
                              

                              builder.create<scf::YieldOp>(loc);
                            });
                        builder.create<scf::YieldOp>(loc);
                      });
                }
                builder.create<scf::YieldOp>(loc);
              },
              [&](OpBuilder &builder, Location loc) {
                // rowMid or rowDown
                Value rowMidCond = builder.create<CmpIOp>(
                    loc, CmpIPredicate::slt, currRow, rowMidHelper);

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
                                dip::BoundaryOption::ConstantPadding) {
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, constantValue);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, input,
                                  ValueRange{imRow, leftPaddingOffset},
                                  leftMask, padding);
                            }  if (boundaryOptionAttr ==
                                       dip::BoundaryOption::ReplicatePadding) {
                              Value paddingVal = builder.create<memref::LoadOp>(
                                  loc, input, ValueRange{imRow, c0});
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, paddingVal);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, input,
                                  ValueRange{imRow, leftPaddingOffset},
                                  leftMask, padding);
                            }
                             Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                              
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        

                            builder.create<scf::YieldOp>(loc);
                          },
                          [&](OpBuilder &builder, Location loc) {
                            // (colMid or colRight) & rowMid
                            Value colMidCond = builder.create<CmpIOp>(
                                loc, CmpIPredicate::sle, colLastElem,
                                colMidHelper);

                            builder.create<scf::IfOp>(
                                loc, colMidCond,
                                [&](OpBuilder &builder, Location loc) {
                                  // colMid & rowMid
                                  Value inputVec = builder.create<LoadOp>(
                                      loc, vectorTy32, input,
                                      ValueRange{imRow, imCol});
                                       Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                            
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        

                                  builder.create<scf::YieldOp>(loc);
                                },
                                [&](OpBuilder &builder, Location loc) {
                                  // colRight & rowMid
                                  Value inputVec;
                                  Value rightMaskHelper =
                                      builder.create<SubIOp>(loc, colLastElem,
                                                             colMidHelper);
                                  Value rightMaskElem = builder.create<SubIOp>(
                                      loc, strideVal, rightMaskHelper);
                                  Value rightMask =
                                      builder.create<CreateMaskOp>(
                                          loc, vectorMaskTy, rightMaskElem);

                                  if (boundaryOptionAttr ==
                                      dip::BoundaryOption::ConstantPadding) {
                                    Value padding = builder.create<BroadcastOp>(
                                        loc, vectorTy32, constantValue);

                                    inputVec = builder.create<MaskedLoadOp>(
                                        loc, vectorTy32, input,
                                        ValueRange{imRow, imCol}, rightMask,
                                        padding);
                                  } else if (boundaryOptionAttr ==
                                             dip::BoundaryOption::
                                                 ReplicatePadding) {
                                    Value rightRange = builder.create<SubIOp>(
                                        loc, inputCol, c1);
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
                                
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        

                                  builder.create<scf::YieldOp>(loc);
                                });
                            builder.create<scf::YieldOp>(loc);
                          });
                      builder.create<scf::YieldOp>(loc);
                    },
                    [&](OpBuilder &builder, Location loc) {
                      // rowDown
                      if (boundaryOptionAttr ==
                          dip::BoundaryOption::ConstantPadding) {
                        Value inputVec = builder.create<BroadcastOp>(
                            loc, vectorTy32, constantValue);
                             Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);


                    
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
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
                              Value leftMask = createInvertedMask(
                                  builder, loc, strideVal, vectorMaskTy,
                                  leftMaskElem);

                              if (boundaryOptionAttr ==
                                  dip::BoundaryOption::ReplicatePadding) {
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, input, ValueRange{downRange, c0});
                                Value padding = builder.create<BroadcastOp>(
                                    loc, vectorTy32, paddingVal);

                                Value leftPaddingOffset =
                                    builder.create<SubIOp>(loc, c0,
                                                           leftMaskElem);
                                inputVec = builder.create<MaskedLoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{downRange, leftPaddingOffset},
                                    leftMask, padding);
                              }
                               Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                        
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        

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
                                    Value downRange = builder.create<SubIOp>(
                                        loc, inputRow, c1);
                                    if (boundaryOptionAttr ==
                                        dip::BoundaryOption::ReplicatePadding) {
                                      inputVec = builder.create<LoadOp>(
                                          loc, vectorTy32, input,
                                          ValueRange{downRange, imCol});
                                    }
                                     Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);

                                
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        

                                    builder.create<scf::YieldOp>(loc);
                                  },
                                  [&](OpBuilder &builder, Location loc) {
                                    // colRight & rowDown
                                    Value inputVec;
                                    Value rightMaskHelper =
                                        builder.create<SubIOp>(loc, colLastElem,
                                                               colMidHelper);
                                    Value rightMaskElem =
                                        builder.create<SubIOp>(loc, strideVal,
                                                               rightMaskHelper);
                                    Value rightMask =
                                        builder.create<CreateMaskOp>(
                                            loc, vectorMaskTy, rightMaskElem);

                                    Value downRange = builder.create<SubIOp>(
                                        loc, inputRow, c1);
                                    Value rightRange = builder.create<SubIOp>(
                                        loc, inputCol, c1);

                                    if (boundaryOptionAttr ==
                                        dip::BoundaryOption::ReplicatePadding) {

                                      Value paddingVal =
                                          builder.create<memref::LoadOp>(
                                              loc, input,
                                              ValueRange{downRange,
                                                         rightRange});
                                      Value padding =
                                          builder.create<vector::BroadcastOp>(
                                              loc, vectorTy32, paddingVal);

                                      inputVec = builder.create<MaskedLoadOp>(
                                          loc, vectorTy32, input,
                                          ValueRange{downRange, imCol},
                                          rightMask, padding);
                                    }
                                    Value tailCond = tailChecker(
                                        builder, loc, calcHelper, strideVal,
                                        kernelSize, c1, pseudoCol, ivs[2]);

                                    
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
                                     

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

        Value inputRow1 = rewriter.create<memref::DimOp>(loc,output1, c0);
    Value inputCol1 = rewriter.create<memref::DimOp>(loc, output1, c1);
    Value outputrow1 = rewriter.create<memref::DimOp>(loc, output, c0);
    Value outputcol1 = rewriter.create<memref::DimOp>(loc,output,c1);


    // Variables used for detecting rowMid, rowDown, colMid and colRight
    // regions.
    Value rowMidHelper1 = rewriter.create<AddIOp>(loc, inputRow1, centerY);
    Value colMidHelper1 = rewriter.create<AddIOp>(loc, inputCol1, centerX);

    SmallVector<Value, 8> lowerBounds2(4, c0);
    SmallVector<Value, 8> uperBounds2{inputRow1, kernelSize, inputCol1,
                                     kernelSize};
    SmallVector<int64_t, 8> steps2{1, 1, stride, 1};



         SmallVector<Value, 8> lowerBounds3(2,c0);
   SmallVector<Value, 8> upperBounds3{outputrow1,outputcol1};
   SmallVector<int64_t, 8> steps3{1, 1};
   // Initializes the Output memref with -256 
    buildAffineLoopNest(
        rewriter, loc, lowerBounds3, upperBounds3, steps3,
        [&](OpBuilder &builder, Location loc, ValueRange ivs3)
        {
             builder.create<StoreOp>(loc, PaddingElement, output,
                                ValueRange{ivs3[0], ivs3[1]});
        });

  

    Value pseudoCol1 = rewriter.create<AffineApplyOp>(
        loc, calcHelper, ValueRange{inputCol1, kernelSize, c1});

    buildAffineLoopNest(
        rewriter, loc, lowerBounds2, uperBounds2, steps2,
        [&](OpBuilder &builder, Location loc, ValueRange ivs2) {
          // Indices of current pixel with respect to pseudo image containing
          // extrapolated boundaries.
          Value currRow = builder.create<AddIOp>(loc, ivs2[0], ivs2[1]);
          Value currCol = builder.create<AddIOp>(loc, ivs2[2], ivs2[3]);

          Value kernelValue = builder.create<memref::LoadOp>(
              loc, kernel, ValueRange{ivs2[1], ivs2[3]});
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
                    dip::BoundaryOption::ConstantPadding) {
                  Value inputVec = builder.create<BroadcastOp>(loc, vectorTy32,
                                                               constantValue);
                                                                Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol1, ivs2[2]);
                    
                    
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol1,
                                        vectorMaskTy);
                                        
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
                        Value leftMask =
                            createInvertedMask(builder, loc, strideVal,
                                               vectorMaskTy, leftMaskElem);

                        if (boundaryOptionAttr ==
                            dip::BoundaryOption::ReplicatePadding) {
                          Value paddingVal = builder.create<memref::LoadOp>(
                              loc, output1, ValueRange{c0, c0});
                          Value padding = builder.create<BroadcastOp>(
                              loc, vectorTy32, paddingVal);

                          Value leftPaddingOffset =
                              builder.create<SubIOp>(loc, c0, leftMaskElem);
                          inputVec = builder.create<vector::MaskedLoadOp>(
                              loc, vectorTy32, output1,
                              ValueRange{c0, leftPaddingOffset}, leftMask,
                              padding);
                        }
                         Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol1, ivs2[2]);
                        
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol1,
                                        vectorMaskTy);
                                        
                        builder.create<scf::YieldOp>(loc);
                      },
                      [&](OpBuilder &builder, Location loc) {
                        // (colMid or colRight) & rowUp
                        Value colMidCond = builder.create<CmpIOp>(
                            loc, CmpIPredicate::sle, colLastElem, colMidHelper1);

                        builder.create<scf::IfOp>(
                            loc, colMidCond,
                            [&](OpBuilder &builder, Location loc) {
                              // colMid & rowUp
                              Value inputVec;
                              if (boundaryOptionAttr ==
                                  dip::BoundaryOption::ReplicatePadding) {
                                inputVec = builder.create<LoadOp>(
                                    loc, vectorTy32, output1,
                                    ValueRange{c0, imCol});
                              }
                               Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol1, ivs2[2]);
                    
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol1,
                                        vectorMaskTy);
                                        
                                    

                              builder.create<scf::YieldOp>(loc);
                            },
                            [&](OpBuilder &builder, Location loc) {
                              // colRight & rowUp
                              Value inputVec;
                              Value rightMaskHelper = builder.create<SubIOp>(
                                  loc, colLastElem, colMidHelper1);
                              Value rightMaskElem = builder.create<SubIOp>(
                                  loc, strideVal, rightMaskHelper);
                              Value rightMask = builder.create<CreateMaskOp>(
                                  loc, vectorMaskTy, rightMaskElem);

                              if (boundaryOptionAttr ==
                                  dip::BoundaryOption::ReplicatePadding) {
                                Value rightRange =
                                    builder.create<SubIOp>(loc, inputCol1, c1);
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, output1, ValueRange{c0, rightRange});
                                Value padding = builder.create<BroadcastOp>(
                                    loc, vectorTy32, paddingVal);

                                inputVec = builder.create<MaskedLoadOp>(
                                    loc, vectorTy32, output1,
                                    ValueRange{c0, imCol}, rightMask, padding);
                              }
                              Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol1, ivs2[2]);

                            
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol1,
                                        vectorMaskTy);
                                       


                              builder.create<scf::YieldOp>(loc);
                            });
                        builder.create<scf::YieldOp>(loc);
                      });
                }
                builder.create<scf::YieldOp>(loc);
              },
              [&](OpBuilder &builder, Location loc) {
                // rowMid or rowDown
                Value rowMidCond = builder.create<CmpIOp>(
                    loc, CmpIPredicate::slt, currRow, rowMidHelper1);

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
                                dip::BoundaryOption::ConstantPadding) {
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, constantValue);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, output1,
                                  ValueRange{imRow, leftPaddingOffset},
                                  leftMask, padding);
                            }  if (boundaryOptionAttr ==
                                       dip::BoundaryOption::ReplicatePadding) {
                              Value paddingVal = builder.create<memref::LoadOp>(
                                  loc, output1, ValueRange{imRow, c0});
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, paddingVal);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, output1,
                                  ValueRange{imRow, leftPaddingOffset},
                                  leftMask, padding);
                            }
                             Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs2[2]);
                
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       
                            builder.create<scf::YieldOp>(loc);
                          },
                          [&](OpBuilder &builder, Location loc) {
                            // (colMid or colRight) & rowMid
                            Value colMidCond = builder.create<CmpIOp>(
                                loc, CmpIPredicate::sle, colLastElem,
                                colMidHelper1);

                            builder.create<scf::IfOp>(
                                loc, colMidCond,
                                [&](OpBuilder &builder, Location loc) {
                                  // colMid & rowMid
                                  Value inputVec = builder.create<LoadOp>(
                                      loc, vectorTy32, output1,
                                      ValueRange{imRow, imCol});
                                       Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol1, ivs2[2]);
                                    
                                 
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                      


                                  builder.create<scf::YieldOp>(loc);
                                },
                                [&](OpBuilder &builder, Location loc) {
                                  // colRight & rowMid
                                  Value inputVec;
                                  Value rightMaskHelper =
                                      builder.create<SubIOp>(loc, colLastElem,
                                                             colMidHelper1);
                                  Value rightMaskElem = builder.create<SubIOp>(
                                      loc, strideVal, rightMaskHelper);
                                  Value rightMask =
                                      builder.create<CreateMaskOp>(
                                          loc, vectorMaskTy, rightMaskElem);

                                  if (boundaryOptionAttr ==
                                      dip::BoundaryOption::ConstantPadding) {
                                    Value padding = builder.create<BroadcastOp>(
                                        loc, vectorTy32, constantValue);

                                    inputVec = builder.create<MaskedLoadOp>(
                                        loc, vectorTy32, output1,
                                        ValueRange{imRow, imCol}, rightMask,
                                        padding);
                                  } else if (boundaryOptionAttr ==
                                             dip::BoundaryOption::
                                                 ReplicatePadding) {
                                    Value rightRange = builder.create<SubIOp>(
                                        loc, inputCol, c1);
                                    Value paddingVal =
                                        builder.create<memref::LoadOp>(
                                            loc, output1,
                                            ValueRange{imRow, rightRange});
                                    Value padding = builder.create<BroadcastOp>(
                                        loc, vectorTy32, paddingVal);

                                    inputVec = builder.create<MaskedLoadOp>(
                                        loc, vectorTy32, output1,
                                        ValueRange{imRow, imCol}, rightMask,
                                        padding);
                                  }
                                  Value tailCond = tailChecker(
                                      builder, loc, calcHelper, strideVal,
                                      kernelSize, c1, pseudoCol, ivs2[2]);
                                    
                                   
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        

                                  builder.create<scf::YieldOp>(loc);
                                });
                            builder.create<scf::YieldOp>(loc);
                          });
                      builder.create<scf::YieldOp>(loc);
                    },
                    [&](OpBuilder &builder, Location loc) {
                      // rowDown
                      if (boundaryOptionAttr ==
                          dip::BoundaryOption::ConstantPadding) {
                        Value inputVec = builder.create<BroadcastOp>(
                            loc, vectorTy32, constantValue);
                             Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs2[2]);

                          
                
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       
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
                              Value leftMask = createInvertedMask(
                                  builder, loc, strideVal, vectorMaskTy,
                                  leftMaskElem);

                              if (boundaryOptionAttr ==
                                  dip::BoundaryOption::ReplicatePadding) {
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, input, ValueRange{downRange, c0});
                                Value padding = builder.create<BroadcastOp>(
                                    loc, vectorTy32, paddingVal);

                                Value leftPaddingOffset =
                                    builder.create<SubIOp>(loc, c0,
                                                           leftMaskElem);
                                inputVec = builder.create<MaskedLoadOp>(
                                    loc, vectorTy32, output1,
                                    ValueRange{downRange, leftPaddingOffset},
                                    leftMask, padding);
                              }
                               Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs2[2]);
 
                        
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                      

                              builder.create<scf::YieldOp>(loc);
                            },
                            [&](OpBuilder &builder, Location loc) {
                              // (colMid or colRight) & rowDown
                              Value colMidCond = builder.create<CmpIOp>(
                                  loc, CmpIPredicate::sle, colLastElem,
                                  colMidHelper1);

                              builder.create<scf::IfOp>(
                                  loc, colMidCond,
                                  [&](OpBuilder &builder, Location loc) {
                                    // colMid & rowDown
                                    Value inputVec;
                                    Value downRange = builder.create<SubIOp>(
                                        loc, inputRow, c1);
                                    if (boundaryOptionAttr ==
                                        dip::BoundaryOption::ReplicatePadding) {
                                      inputVec = builder.create<LoadOp>(
                                          loc, vectorTy32, output1,
                                          ValueRange{downRange, imCol});
                                    }
                                     Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs2[2]);
                                   
                                
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       

                                    builder.create<scf::YieldOp>(loc);
                                  },
                                  [&](OpBuilder &builder, Location loc) {
                                    // colRight & rowDown
                                    Value inputVec;
                                    Value rightMaskHelper =
                                        builder.create<SubIOp>(loc, colLastElem,
                                                               colMidHelper1);
                                    Value rightMaskElem =
                                        builder.create<SubIOp>(loc, strideVal,
                                                               rightMaskHelper);
                                    Value rightMask =
                                        builder.create<CreateMaskOp>(
                                            loc, vectorMaskTy, rightMaskElem);

                                    Value downRange = builder.create<SubIOp>(
                                        loc, inputRow, c1);
                                    Value rightRange = builder.create<SubIOp>(
                                        loc, inputCol, c1);

                                    if (boundaryOptionAttr ==
                                        dip::BoundaryOption::ReplicatePadding) {

                                      Value paddingVal =
                                          builder.create<memref::LoadOp>(
                                              loc, output1,
                                              ValueRange{downRange,
                                                         rightRange});
                                      Value padding =
                                          builder.create<vector::BroadcastOp>(
                                              loc, vectorTy32, paddingVal);

                                      inputVec = builder.create<MaskedLoadOp>(
                                          loc, vectorTy32, output1,
                                          ValueRange{downRange, imCol},
                                          rightMask, padding);
                                    }
                                    Value tailCond = tailChecker(
                                        builder, loc, calcHelper, strideVal,
                                        kernelSize, c1, pseudoCol, ivs2[2]);
                            
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                      
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

    // Remove the origin convolution operation.
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;


};


class DIPTopHat2DOpLowering : public OpRewritePattern<dip::TopHat2DOp>
{
public:
  using OpRewritePattern<dip::TopHat2DOp>::OpRewritePattern;

  explicit DIPTopHat2DOpLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
  }

  LogicalResult matchAndRewrite(dip::TopHat2DOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Create constant indices.
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);

    // Register operand values.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    Value output1 = op->getOperand(3);
    Value output2 = op->getOperand(4);
    Value centerX = op->getOperand(5);
    Value centerY = op->getOperand(6);
    Value constantValue = op->getOperand(7);
    auto boundaryOptionAttr = op.boundary_option();
    Value strideVal = rewriter.create<ConstantIndexOp>(loc, stride);

    FloatType f32 = FloatType::getF32(ctx);
    IntegerType i1 = IntegerType::get(ctx, 1);

    // Create DimOp.
    Value inputRow = rewriter.create<memref::DimOp>(loc, input, c0);
    Value inputCol = rewriter.create<memref::DimOp>(loc, input, c1);
    Value outputrow = rewriter.create<memref::DimOp>(loc, output1, c0);
    Value outputcol = rewriter.create<memref::DimOp>(loc,output1,c1);
    Value kernelSize = rewriter.create<memref::DimOp>(loc, kernel, c0);

    // Variables used for detecting rowMid, rowDown, colMid and colRight
    // regions.
    Value rowMidHelper = rewriter.create<AddIOp>(loc, inputRow, centerY);
    Value colMidHelper = rewriter.create<AddIOp>(loc, inputCol, centerX);

    SmallVector<Value, 8> lowerBounds(4, c0);
    SmallVector<Value, 8> uperBounds{inputRow, kernelSize, inputCol,
                                     kernelSize};
    SmallVector<int64_t, 8> steps{1, 1, stride, 1};

    VectorType vectorTy32 = VectorType::get({stride}, f32);
    VectorType VectorOne = VectorType::get({1}, f32);
    VectorType vectorMaskTy = VectorType::get({stride}, i1);

    Value zeroPaddingElem =
        rewriter.create<ConstantFloatOp>(loc, (APFloat)(float)0, f32);
    Value zeroPadding =
        rewriter.create<BroadcastOp>(loc, vectorTy32, zeroPaddingElem);
        Value Paddingel = rewriter.create<ConstantFloatOp>(loc, (APFloat)(float)-256, f32);
        Value PaddingElement = rewriter.create<BroadcastOp>(loc, VectorOne, Paddingel);
        Value Paddingvec = rewriter.create<BroadcastOp>(loc, vectorTy32, Paddingel);

         SmallVector<Value, 8> lowerBounds1(2,c0);
   SmallVector<Value, 8> upperBounds1{outputrow,outputcol};
   SmallVector<int64_t, 8> steps1{1, 1};
   // Initializes the Output memref with -256 
    buildAffineLoopNest(
        rewriter, loc, lowerBounds1, upperBounds1, steps1,
        [&](OpBuilder &builder, Location loc, ValueRange ivs1)
        {
             builder.create<StoreOp>(loc, PaddingElement, output1,
                                ValueRange{ivs1[0], ivs1[1]});
        });

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
                    dip::BoundaryOption::ConstantPadding) {
                  Value inputVec = builder.create<BroadcastOp>(loc, vectorTy32,
                                                               constantValue);
                                                                Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                
                
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       
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
                        Value leftMask =
                            createInvertedMask(builder, loc, strideVal,
                                               vectorMaskTy, leftMaskElem);

                        if (boundaryOptionAttr ==
                            dip::BoundaryOption::ReplicatePadding) {
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
                         Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                    
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       
                         

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
                                  dip::BoundaryOption::ReplicatePadding) {
                                inputVec = builder.create<LoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{c0, imCol});
                              }
                               Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                              
                
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       

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
                                  dip::BoundaryOption::ReplicatePadding) {
                                Value rightRange =
                                    builder.create<SubIOp>(loc, inputCol, c1);
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, input, ValueRange{c0, rightRange});
                                Value padding = builder.create<BroadcastOp>(
                                    loc, vectorTy32, paddingVal);

                                inputVec = builder.create<MaskedLoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{c0, imCol}, rightMask, padding);
                              }
                              Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
            
                               
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       
                              

                              builder.create<scf::YieldOp>(loc);
                            });
                        builder.create<scf::YieldOp>(loc);
                      });
                }
                builder.create<scf::YieldOp>(loc);
              },
              [&](OpBuilder &builder, Location loc) {
                // rowMid or rowDown
                Value rowMidCond = builder.create<CmpIOp>(
                    loc, CmpIPredicate::slt, currRow, rowMidHelper);

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
                                dip::BoundaryOption::ConstantPadding) {
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, constantValue);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, input,
                                  ValueRange{imRow, leftPaddingOffset},
                                  leftMask, padding);
                            }  if (boundaryOptionAttr ==
                                       dip::BoundaryOption::ReplicatePadding) {
                              Value paddingVal = builder.create<memref::LoadOp>(
                                  loc, input, ValueRange{imRow, c0});
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, paddingVal);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, input,
                                  ValueRange{imRow, leftPaddingOffset},
                                  leftMask, padding);
                            }
                             Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                            
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       

                            builder.create<scf::YieldOp>(loc);
                          },
                          [&](OpBuilder &builder, Location loc) {
                            // (colMid or colRight) & rowMid
                            Value colMidCond = builder.create<CmpIOp>(
                                loc, CmpIPredicate::sle, colLastElem,
                                colMidHelper);

                            builder.create<scf::IfOp>(
                                loc, colMidCond,
                                [&](OpBuilder &builder, Location loc) {
                                  // colMid & rowMid
                                  Value inputVec = builder.create<LoadOp>(
                                      loc, vectorTy32, input,
                                      ValueRange{imRow, imCol});
                                       Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                        
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        

                                  builder.create<scf::YieldOp>(loc);
                                },
                                [&](OpBuilder &builder, Location loc) {
                                  // colRight & rowMid
                                  Value inputVec;
                                  Value rightMaskHelper =
                                      builder.create<SubIOp>(loc, colLastElem,
                                                             colMidHelper);
                                  Value rightMaskElem = builder.create<SubIOp>(
                                      loc, strideVal, rightMaskHelper);
                                  Value rightMask =
                                      builder.create<CreateMaskOp>(
                                          loc, vectorMaskTy, rightMaskElem);

                                  if (boundaryOptionAttr ==
                                      dip::BoundaryOption::ConstantPadding) {
                                    Value padding = builder.create<BroadcastOp>(
                                        loc, vectorTy32, constantValue);

                                    inputVec = builder.create<MaskedLoadOp>(
                                        loc, vectorTy32, input,
                                        ValueRange{imRow, imCol}, rightMask,
                                        padding);
                                  } else if (boundaryOptionAttr ==
                                             dip::BoundaryOption::
                                                 ReplicatePadding) {
                                    Value rightRange = builder.create<SubIOp>(
                                        loc, inputCol, c1);
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
                                
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        

                                  builder.create<scf::YieldOp>(loc);
                                });
                            builder.create<scf::YieldOp>(loc);
                          });
                      builder.create<scf::YieldOp>(loc);
                    },
                    [&](OpBuilder &builder, Location loc) {
                      // rowDown
                      if (boundaryOptionAttr ==
                          dip::BoundaryOption::ConstantPadding) {
                        Value inputVec = builder.create<BroadcastOp>(
                            loc, vectorTy32, constantValue);
                             Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);


            
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
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
                              Value leftMask = createInvertedMask(
                                  builder, loc, strideVal, vectorMaskTy,
                                  leftMaskElem);

                              if (boundaryOptionAttr ==
                                  dip::BoundaryOption::ReplicatePadding) {
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, input, ValueRange{downRange, c0});
                                Value padding = builder.create<BroadcastOp>(
                                    loc, vectorTy32, paddingVal);

                                Value leftPaddingOffset =
                                    builder.create<SubIOp>(loc, c0,
                                                           leftMaskElem);
                                inputVec = builder.create<MaskedLoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{downRange, leftPaddingOffset},
                                    leftMask, padding);
                              }
                               Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                    
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       

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
                                    Value downRange = builder.create<SubIOp>(
                                        loc, inputRow, c1);
                                    if (boundaryOptionAttr ==
                                        dip::BoundaryOption::ReplicatePadding) {
                                      inputVec = builder.create<LoadOp>(
                                          loc, vectorTy32, input,
                                          ValueRange{downRange, imCol});
                                    }
                                     Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);

                                    
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       

                                    builder.create<scf::YieldOp>(loc);
                                  },
                                  [&](OpBuilder &builder, Location loc) {
                                    // colRight & rowDown
                                    Value inputVec;
                                    Value rightMaskHelper =
                                        builder.create<SubIOp>(loc, colLastElem,
                                                               colMidHelper);
                                    Value rightMaskElem =
                                        builder.create<SubIOp>(loc, strideVal,
                                                               rightMaskHelper);
                                    Value rightMask =
                                        builder.create<CreateMaskOp>(
                                            loc, vectorMaskTy, rightMaskElem);

                                    Value downRange = builder.create<SubIOp>(
                                        loc, inputRow, c1);
                                    Value rightRange = builder.create<SubIOp>(
                                        loc, inputCol, c1);

                                    if (boundaryOptionAttr ==
                                        dip::BoundaryOption::ReplicatePadding) {

                                      Value paddingVal =
                                          builder.create<memref::LoadOp>(
                                              loc, input,
                                              ValueRange{downRange,
                                                         rightRange});
                                      Value padding =
                                          builder.create<vector::BroadcastOp>(
                                              loc, vectorTy32, paddingVal);

                                      inputVec = builder.create<MaskedLoadOp>(
                                          loc, vectorTy32, input,
                                          ValueRange{downRange, imCol},
                                          rightMask, padding);
                                    }
                                    Value tailCond = tailChecker(
                                        builder, loc, calcHelper, strideVal,
                                        kernelSize, c1, pseudoCol, ivs[2]);

                            
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       
                                     

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

        Value inputRow1 = rewriter.create<memref::DimOp>(loc,output1, c0);
    Value inputCol1 = rewriter.create<memref::DimOp>(loc, output1, c1);
    Value outputrow1 = rewriter.create<memref::DimOp>(loc, output, c0);
    Value outputcol1 = rewriter.create<memref::DimOp>(loc,output,c1);


    // Variables used for detecting rowMid, rowDown, colMid and colRight
    // regions.
    Value rowMidHelper1 = rewriter.create<AddIOp>(loc, inputRow1, centerY);
    Value colMidHelper1 = rewriter.create<AddIOp>(loc, inputCol1, centerX);

    SmallVector<Value, 8> lowerBounds2(4, c0);
    SmallVector<Value, 8> uperBounds2{inputRow1, kernelSize, inputCol1,
                                     kernelSize};
    SmallVector<int64_t, 8> steps2{1, 1, stride, 1};



         SmallVector<Value, 8> lowerBounds3(2,c0);
   SmallVector<Value, 8> upperBounds3{outputrow1,outputcol1};
   SmallVector<int64_t, 8> steps3{1, 1};
   // Initializes the Output memref with -256 
    buildAffineLoopNest(
        rewriter, loc, lowerBounds3, upperBounds3, steps3,
        [&](OpBuilder &builder, Location loc, ValueRange ivs3)
        {
             builder.create<StoreOp>(loc, PaddingElement, output2,
                                ValueRange{ivs3[0], ivs3[1]});
        });

       


    Value pseudoCol1 = rewriter.create<AffineApplyOp>(
        loc, calcHelper, ValueRange{inputCol1, kernelSize, c1});

    buildAffineLoopNest(
        rewriter, loc, lowerBounds2, uperBounds2, steps2,
        [&](OpBuilder &builder, Location loc, ValueRange ivs2) {
          // Indices of current pixel with respect to pseudo image containing
          // extrapolated boundaries.
          Value currRow = builder.create<AddIOp>(loc, ivs2[0], ivs2[1]);
          Value currCol = builder.create<AddIOp>(loc, ivs2[2], ivs2[3]);

          Value kernelValue = builder.create<memref::LoadOp>(
              loc, kernel, ValueRange{ivs2[1], ivs2[3]});
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
                    dip::BoundaryOption::ConstantPadding) {
                  Value inputVec = builder.create<BroadcastOp>(loc, vectorTy32,
                                                               constantValue);
                                                                Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol1, ivs2[2]);
                    
                
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output2, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol1,
                                        vectorMaskTy);
                                       
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
                        Value leftMask =
                            createInvertedMask(builder, loc, strideVal,
                                               vectorMaskTy, leftMaskElem);

                        if (boundaryOptionAttr ==
                            dip::BoundaryOption::ReplicatePadding) {
                          Value paddingVal = builder.create<memref::LoadOp>(
                              loc, output1, ValueRange{c0, c0});
                          Value padding = builder.create<BroadcastOp>(
                              loc, vectorTy32, paddingVal);

                          Value leftPaddingOffset =
                              builder.create<SubIOp>(loc, c0, leftMaskElem);
                          inputVec = builder.create<vector::MaskedLoadOp>(
                              loc, vectorTy32, output1,
                              ValueRange{c0, leftPaddingOffset}, leftMask,
                              padding);
                        }
                         Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol1, ivs2[2]);
                         
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output2, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol1,
                                        vectorMaskTy);
                                        
                        builder.create<scf::YieldOp>(loc);
                      },
                      [&](OpBuilder &builder, Location loc) {
                        // (colMid or colRight) & rowUp
                        Value colMidCond = builder.create<CmpIOp>(
                            loc, CmpIPredicate::sle, colLastElem, colMidHelper1);

                        builder.create<scf::IfOp>(
                            loc, colMidCond,
                            [&](OpBuilder &builder, Location loc) {
                              // colMid & rowUp
                              Value inputVec;
                              if (boundaryOptionAttr ==
                                  dip::BoundaryOption::ReplicatePadding) {
                                inputVec = builder.create<LoadOp>(
                                    loc, vectorTy32, output1,
                                    ValueRange{c0, imCol});
                              }
                               Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol1, ivs2[2]);
                        
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output2, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol1,
                                        vectorMaskTy);
                                      

                              builder.create<scf::YieldOp>(loc);
                            },
                            [&](OpBuilder &builder, Location loc) {
                              // colRight & rowUp
                              Value inputVec;
                              Value rightMaskHelper = builder.create<SubIOp>(
                                  loc, colLastElem, colMidHelper1);
                              Value rightMaskElem = builder.create<SubIOp>(
                                  loc, strideVal, rightMaskHelper);
                              Value rightMask = builder.create<CreateMaskOp>(
                                  loc, vectorMaskTy, rightMaskElem);

                              if (boundaryOptionAttr ==
                                  dip::BoundaryOption::ReplicatePadding) {
                                Value rightRange =
                                    builder.create<SubIOp>(loc, inputCol1, c1);
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, output1, ValueRange{c0, rightRange});
                                Value padding = builder.create<BroadcastOp>(
                                    loc, vectorTy32, paddingVal);

                                inputVec = builder.create<MaskedLoadOp>(
                                    loc, vectorTy32, output1,
                                    ValueRange{c0, imCol}, rightMask, padding);
                              }
                              Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol1, ivs2[2]);

                               
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output2, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol1,
                                        vectorMaskTy);
                                        


                              builder.create<scf::YieldOp>(loc);
                            });
                        builder.create<scf::YieldOp>(loc);
                      });
                }
                builder.create<scf::YieldOp>(loc);
              },
              [&](OpBuilder &builder, Location loc) {
                // rowMid or rowDown
                Value rowMidCond = builder.create<CmpIOp>(
                    loc, CmpIPredicate::slt, currRow, rowMidHelper1);

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
                                dip::BoundaryOption::ConstantPadding) {
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, constantValue);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, output1,
                                  ValueRange{imRow, leftPaddingOffset},
                                  leftMask, padding);
                            }  if (boundaryOptionAttr ==
                                       dip::BoundaryOption::ReplicatePadding) {
                              Value paddingVal = builder.create<memref::LoadOp>(
                                  loc, output1, ValueRange{imRow, c0});
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, paddingVal);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, output1,
                                  ValueRange{imRow, leftPaddingOffset},
                                  leftMask, padding);
                            }
                             Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs2[2]);
                        
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output2, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
                            builder.create<scf::YieldOp>(loc);
                          },
                          [&](OpBuilder &builder, Location loc) {
                            // (colMid or colRight) & rowMid
                            Value colMidCond = builder.create<CmpIOp>(
                                loc, CmpIPredicate::sle, colLastElem,
                                colMidHelper1);

                            builder.create<scf::IfOp>(
                                loc, colMidCond,
                                [&](OpBuilder &builder, Location loc) {
                                  // colMid & rowMid
                                  Value inputVec = builder.create<LoadOp>(
                                      loc, vectorTy32, output1,
                                      ValueRange{imRow, imCol});
                                       Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol1, ivs2[2]);
                                    
                            
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output2, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        


                                  builder.create<scf::YieldOp>(loc);
                                },
                                [&](OpBuilder &builder, Location loc) {
                                  // colRight & rowMid
                                  Value inputVec;
                                  Value rightMaskHelper =
                                      builder.create<SubIOp>(loc, colLastElem,
                                                             colMidHelper1);
                                  Value rightMaskElem = builder.create<SubIOp>(
                                      loc, strideVal, rightMaskHelper);
                                  Value rightMask =
                                      builder.create<CreateMaskOp>(
                                          loc, vectorMaskTy, rightMaskElem);

                                  if (boundaryOptionAttr ==
                                      dip::BoundaryOption::ConstantPadding) {
                                    Value padding = builder.create<BroadcastOp>(
                                        loc, vectorTy32, constantValue);

                                    inputVec = builder.create<MaskedLoadOp>(
                                        loc, vectorTy32, output1,
                                        ValueRange{imRow, imCol}, rightMask,
                                        padding);
                                  } else if (boundaryOptionAttr ==
                                             dip::BoundaryOption::
                                                 ReplicatePadding) {
                                    Value rightRange = builder.create<SubIOp>(
                                        loc, inputCol, c1);
                                    Value paddingVal =
                                        builder.create<memref::LoadOp>(
                                            loc, output1,
                                            ValueRange{imRow, rightRange});
                                    Value padding = builder.create<BroadcastOp>(
                                        loc, vectorTy32, paddingVal);

                                    inputVec = builder.create<MaskedLoadOp>(
                                        loc, vectorTy32, output1,
                                        ValueRange{imRow, imCol}, rightMask,
                                        padding);
                                  }
                                  Value tailCond = tailChecker(
                                      builder, loc, calcHelper, strideVal,
                                      kernelSize, c1, pseudoCol, ivs2[2]);
                                    
                                   
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output2, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        

                                  builder.create<scf::YieldOp>(loc);
                                });
                            builder.create<scf::YieldOp>(loc);
                          });
                      builder.create<scf::YieldOp>(loc);
                    },
                    [&](OpBuilder &builder, Location loc) {
                      // rowDown
                      if (boundaryOptionAttr ==
                          dip::BoundaryOption::ConstantPadding) {
                        Value inputVec = builder.create<BroadcastOp>(
                            loc, vectorTy32, constantValue);
                             Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs2[2]);

                          
                        
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output2, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
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
                              Value leftMask = createInvertedMask(
                                  builder, loc, strideVal, vectorMaskTy,
                                  leftMaskElem);

                              if (boundaryOptionAttr ==
                                  dip::BoundaryOption::ReplicatePadding) {
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, input, ValueRange{downRange, c0});
                                Value padding = builder.create<BroadcastOp>(
                                    loc, vectorTy32, paddingVal);

                                Value leftPaddingOffset =
                                    builder.create<SubIOp>(loc, c0,
                                                           leftMaskElem);
                                inputVec = builder.create<MaskedLoadOp>(
                                    loc, vectorTy32, output1,
                                    ValueRange{downRange, leftPaddingOffset},
                                    leftMask, padding);
                              }
                               Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs2[2]);
 
                        
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output2, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        

                              builder.create<scf::YieldOp>(loc);
                            },
                            [&](OpBuilder &builder, Location loc) {
                              // (colMid or colRight) & rowDown
                              Value colMidCond = builder.create<CmpIOp>(
                                  loc, CmpIPredicate::sle, colLastElem,
                                  colMidHelper1);

                              builder.create<scf::IfOp>(
                                  loc, colMidCond,
                                  [&](OpBuilder &builder, Location loc) {
                                    // colMid & rowDown
                                    Value inputVec;
                                    Value downRange = builder.create<SubIOp>(
                                        loc, inputRow, c1);
                                    if (boundaryOptionAttr ==
                                        dip::BoundaryOption::ReplicatePadding) {
                                      inputVec = builder.create<LoadOp>(
                                          loc, vectorTy32, output1,
                                          ValueRange{downRange, imCol});
                                    }
                                     Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs2[2]);
                                   
                                     
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output2, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        

                                    builder.create<scf::YieldOp>(loc);
                                  },
                                  [&](OpBuilder &builder, Location loc) {
                                    // colRight & rowDown
                                    Value inputVec;
                                    Value rightMaskHelper =
                                        builder.create<SubIOp>(loc, colLastElem,
                                                               colMidHelper1);
                                    Value rightMaskElem =
                                        builder.create<SubIOp>(loc, strideVal,
                                                               rightMaskHelper);
                                    Value rightMask =
                                        builder.create<CreateMaskOp>(
                                            loc, vectorMaskTy, rightMaskElem);

                                    Value downRange = builder.create<SubIOp>(
                                        loc, inputRow, c1);
                                    Value rightRange = builder.create<SubIOp>(
                                        loc, inputCol, c1);

                                    if (boundaryOptionAttr ==
                                        dip::BoundaryOption::ReplicatePadding) {

                                      Value paddingVal =
                                          builder.create<memref::LoadOp>(
                                              loc, output1,
                                              ValueRange{downRange,
                                                         rightRange});
                                      Value padding =
                                          builder.create<vector::BroadcastOp>(
                                              loc, vectorTy32, paddingVal);

                                      inputVec = builder.create<MaskedLoadOp>(
                                          loc, vectorTy32, output1,
                                          ValueRange{downRange, imCol},
                                          rightMask, padding);
                                    }
                                    Value tailCond = tailChecker(
                                        builder, loc, calcHelper, strideVal,
                                        kernelSize, c1, pseudoCol, ivs2[2]);
                        
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output2, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
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
                SmallVector<Value, 8> lowerbounds4(2,c0);
        SmallVector<Value, 8> upperbounds4{outputrow1,outputcol1};
        SmallVector<int64_t, 8> steps4{1,1};

        buildAffineLoopNest(
            rewriter, loc, lowerbounds4, upperbounds4,steps4,
            [&](OpBuilder &builder, Location loc, ValueRange ivs4)
            {
                       Value ou = builder.create<LoadOp>(loc,VectorOne,output2,ValueRange{ivs4[0],ivs4[1]});
                       Value in = builder.create<LoadOp>(loc,VectorOne,input,ValueRange{ivs4[0],ivs4[1]});
                       Value res = builder.create<SubFOp>(loc,in,ou);
                       builder.create<StoreOp>(loc, res, output,ValueRange{ivs4[0],ivs4[1]});
            }
        );

    
    // Remove the origin convolution operation.
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;


};

class DIPBottomHat2DOpLowering : public OpRewritePattern<dip::BottomHat2DOp>
{
public:
  using OpRewritePattern<dip::BottomHat2DOp>::OpRewritePattern;

  explicit DIPBottomHat2DOpLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
  }

  LogicalResult matchAndRewrite(dip::BottomHat2DOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Create constant indices.
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);

    // Register operand values.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    Value output1 = op->getOperand(3);
    Value output2 = op->getOperand(4);
    Value centerX = op->getOperand(5);
    Value centerY = op->getOperand(6);
    Value constantValue = op->getOperand(7);
    auto boundaryOptionAttr = op.boundary_option();
    Value strideVal = rewriter.create<ConstantIndexOp>(loc, stride);

    FloatType f32 = FloatType::getF32(ctx);
    IntegerType i1 = IntegerType::get(ctx, 1);

    // Create DimOp.
    Value inputRow = rewriter.create<memref::DimOp>(loc, input, c0);
    Value inputCol = rewriter.create<memref::DimOp>(loc, input, c1);
    Value outputrow = rewriter.create<memref::DimOp>(loc, output1, c0);
    Value outputcol = rewriter.create<memref::DimOp>(loc,output1,c1);
    Value kernelSize = rewriter.create<memref::DimOp>(loc, kernel, c0);

    // Variables used for detecting rowMid, rowDown, colMid and colRight
    // regions.
    Value rowMidHelper = rewriter.create<AddIOp>(loc, inputRow, centerY);
    Value colMidHelper = rewriter.create<AddIOp>(loc, inputCol, centerX);

    SmallVector<Value, 8> lowerBounds(4, c0);
    SmallVector<Value, 8> uperBounds{inputRow, kernelSize, inputCol,
                                     kernelSize};
    SmallVector<int64_t, 8> steps{1, 1, stride, 1};

    VectorType vectorTy32 = VectorType::get({stride}, f32);
    VectorType VectorOne = VectorType::get({1}, f32);
    VectorType vectorMaskTy = VectorType::get({stride}, i1);

    Value zeroPaddingElem =
        rewriter.create<ConstantFloatOp>(loc, (APFloat)(float)0, f32);
    Value zeroPadding =
        rewriter.create<BroadcastOp>(loc, vectorTy32, zeroPaddingElem);
        Value Paddingel = rewriter.create<ConstantFloatOp>(loc, (APFloat)(float)-256, f32);
        Value PaddingElement = rewriter.create<BroadcastOp>(loc, VectorOne, Paddingel);
        Value Paddingvec = rewriter.create<BroadcastOp>(loc, vectorTy32, Paddingel);

         SmallVector<Value, 8> lowerBounds1(2,c0);
   SmallVector<Value, 8> upperBounds1{outputrow,outputcol};
   SmallVector<int64_t, 8> steps1{1, 1};
   // Initializes the Output memref with -256 
    buildAffineLoopNest(
        rewriter, loc, lowerBounds1, upperBounds1, steps1,
        [&](OpBuilder &builder, Location loc, ValueRange ivs1)
        {
             builder.create<StoreOp>(loc, PaddingElement, output1,
                                ValueRange{ivs1[0], ivs1[1]});
        });

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
                    dip::BoundaryOption::ConstantPadding) {
                  Value inputVec = builder.create<BroadcastOp>(loc, vectorTy32,
                                                               constantValue);
                                                                Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                
                
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
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
                        Value leftMask =
                            createInvertedMask(builder, loc, strideVal,
                                               vectorMaskTy, leftMaskElem);

                        if (boundaryOptionAttr ==
                            dip::BoundaryOption::ReplicatePadding) {
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
                         Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                    
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
                         

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
                                  dip::BoundaryOption::ReplicatePadding) {
                                inputVec = builder.create<LoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{c0, imCol});
                              }
                               Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                              
                    
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
                                 

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
                                  dip::BoundaryOption::ReplicatePadding) {
                                Value rightRange =
                                    builder.create<SubIOp>(loc, inputCol, c1);
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, input, ValueRange{c0, rightRange});
                                Value padding = builder.create<BroadcastOp>(
                                    loc, vectorTy32, paddingVal);

                                inputVec = builder.create<MaskedLoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{c0, imCol}, rightMask, padding);
                              }
                              Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
            
                
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
                              

                              builder.create<scf::YieldOp>(loc);
                            });
                        builder.create<scf::YieldOp>(loc);
                      });
                }
                builder.create<scf::YieldOp>(loc);
              },
              [&](OpBuilder &builder, Location loc) {
                // rowMid or rowDown
                Value rowMidCond = builder.create<CmpIOp>(
                    loc, CmpIPredicate::slt, currRow, rowMidHelper);

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
                                dip::BoundaryOption::ConstantPadding) {
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, constantValue);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, input,
                                  ValueRange{imRow, leftPaddingOffset},
                                  leftMask, padding);
                            }  if (boundaryOptionAttr ==
                                       dip::BoundaryOption::ReplicatePadding) {
                              Value paddingVal = builder.create<memref::LoadOp>(
                                  loc, input, ValueRange{imRow, c0});
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, paddingVal);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, input,
                                  ValueRange{imRow, leftPaddingOffset},
                                  leftMask, padding);
                            }
                             Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                            
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        

                            builder.create<scf::YieldOp>(loc);
                          },
                          [&](OpBuilder &builder, Location loc) {
                            // (colMid or colRight) & rowMid
                            Value colMidCond = builder.create<CmpIOp>(
                                loc, CmpIPredicate::sle, colLastElem,
                                colMidHelper);

                            builder.create<scf::IfOp>(
                                loc, colMidCond,
                                [&](OpBuilder &builder, Location loc) {
                                  // colMid & rowMid
                                  Value inputVec = builder.create<LoadOp>(
                                      loc, vectorTy32, input,
                                      ValueRange{imRow, imCol});
                                       Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                        
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        

                                  builder.create<scf::YieldOp>(loc);
                                },
                                [&](OpBuilder &builder, Location loc) {
                                  // colRight & rowMid
                                  Value inputVec;
                                  Value rightMaskHelper =
                                      builder.create<SubIOp>(loc, colLastElem,
                                                             colMidHelper);
                                  Value rightMaskElem = builder.create<SubIOp>(
                                      loc, strideVal, rightMaskHelper);
                                  Value rightMask =
                                      builder.create<CreateMaskOp>(
                                          loc, vectorMaskTy, rightMaskElem);

                                  if (boundaryOptionAttr ==
                                      dip::BoundaryOption::ConstantPadding) {
                                    Value padding = builder.create<BroadcastOp>(
                                        loc, vectorTy32, constantValue);

                                    inputVec = builder.create<MaskedLoadOp>(
                                        loc, vectorTy32, input,
                                        ValueRange{imRow, imCol}, rightMask,
                                        padding);
                                  } else if (boundaryOptionAttr ==
                                             dip::BoundaryOption::
                                                 ReplicatePadding) {
                                    Value rightRange = builder.create<SubIOp>(
                                        loc, inputCol, c1);
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
                                   
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        

                                  builder.create<scf::YieldOp>(loc);
                                });
                            builder.create<scf::YieldOp>(loc);
                          });
                      builder.create<scf::YieldOp>(loc);
                    },
                    [&](OpBuilder &builder, Location loc) {
                      // rowDown
                      if (boundaryOptionAttr ==
                          dip::BoundaryOption::ConstantPadding) {
                        Value inputVec = builder.create<BroadcastOp>(
                            loc, vectorTy32, constantValue);
                             Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);


                         
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
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
                              Value leftMask = createInvertedMask(
                                  builder, loc, strideVal, vectorMaskTy,
                                  leftMaskElem);

                              if (boundaryOptionAttr ==
                                  dip::BoundaryOption::ReplicatePadding) {
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, input, ValueRange{downRange, c0});
                                Value padding = builder.create<BroadcastOp>(
                                    loc, vectorTy32, paddingVal);

                                Value leftPaddingOffset =
                                    builder.create<SubIOp>(loc, c0,
                                                           leftMaskElem);
                                inputVec = builder.create<MaskedLoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{downRange, leftPaddingOffset},
                                    leftMask, padding);
                              }
                               Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                            
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        

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
                                    Value downRange = builder.create<SubIOp>(
                                        loc, inputRow, c1);
                                    if (boundaryOptionAttr ==
                                        dip::BoundaryOption::ReplicatePadding) {
                                      inputVec = builder.create<LoadOp>(
                                          loc, vectorTy32, input,
                                          ValueRange{downRange, imCol});
                                    }
                                     Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);

                            
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        

                                    builder.create<scf::YieldOp>(loc);
                                  },
                                  [&](OpBuilder &builder, Location loc) {
                                    // colRight & rowDown
                                    Value inputVec;
                                    Value rightMaskHelper =
                                        builder.create<SubIOp>(loc, colLastElem,
                                                               colMidHelper);
                                    Value rightMaskElem =
                                        builder.create<SubIOp>(loc, strideVal,
                                                               rightMaskHelper);
                                    Value rightMask =
                                        builder.create<CreateMaskOp>(
                                            loc, vectorMaskTy, rightMaskElem);

                                    Value downRange = builder.create<SubIOp>(
                                        loc, inputRow, c1);
                                    Value rightRange = builder.create<SubIOp>(
                                        loc, inputCol, c1);

                                    if (boundaryOptionAttr ==
                                        dip::BoundaryOption::ReplicatePadding) {

                                      Value paddingVal =
                                          builder.create<memref::LoadOp>(
                                              loc, input,
                                              ValueRange{downRange,
                                                         rightRange});
                                      Value padding =
                                          builder.create<vector::BroadcastOp>(
                                              loc, vectorTy32, paddingVal);

                                      inputVec = builder.create<MaskedLoadOp>(
                                          loc, vectorTy32, input,
                                          ValueRange{downRange, imCol},
                                          rightMask, padding);
                                    }
                                    Value tailCond = tailChecker(
                                        builder, loc, calcHelper, strideVal,
                                        kernelSize, c1, pseudoCol, ivs[2]);

                                
                   compAndStorewTailProcessingFlatdilation(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output1, ivs[0], ivs[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                        
                                     

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

        Value inputRow1 = rewriter.create<memref::DimOp>(loc,output1, c0);
    Value inputCol1 = rewriter.create<memref::DimOp>(loc, output1, c1);
    Value outputrow1 = rewriter.create<memref::DimOp>(loc, output, c0);
    Value outputcol1 = rewriter.create<memref::DimOp>(loc,output,c1);


    // Variables used for detecting rowMid, rowDown, colMid and colRight
    // regions.
    Value rowMidHelper1 = rewriter.create<AddIOp>(loc, inputRow1, centerY);
    Value colMidHelper1 = rewriter.create<AddIOp>(loc, inputCol1, centerX);

    SmallVector<Value, 8> lowerBounds2(4, c0);
    SmallVector<Value, 8> uperBounds2{inputRow1, kernelSize, inputCol1,
                                     kernelSize};
    SmallVector<int64_t, 8> steps2{1, 1, stride, 1};



         SmallVector<Value, 8> lowerBounds3(2,c0);
   SmallVector<Value, 8> upperBounds3{outputrow1,outputcol1};
   SmallVector<int64_t, 8> steps3{1, 1};
   // Initializes the Output memref with -256 
    buildAffineLoopNest(
        rewriter, loc, lowerBounds3, upperBounds3, steps3,
        [&](OpBuilder &builder, Location loc, ValueRange ivs3)
        {
             builder.create<StoreOp>(loc, PaddingElement, output2,
                                ValueRange{ivs3[0], ivs3[1]});
        });

       


    Value pseudoCol1 = rewriter.create<AffineApplyOp>(
        loc, calcHelper, ValueRange{inputCol1, kernelSize, c1});

    buildAffineLoopNest(
        rewriter, loc, lowerBounds2, uperBounds2, steps2,
        [&](OpBuilder &builder, Location loc, ValueRange ivs2) {
          // Indices of current pixel with respect to pseudo image containing
          // extrapolated boundaries.
          Value currRow = builder.create<AddIOp>(loc, ivs2[0], ivs2[1]);
          Value currCol = builder.create<AddIOp>(loc, ivs2[2], ivs2[3]);

          Value kernelValue = builder.create<memref::LoadOp>(
              loc, kernel, ValueRange{ivs2[1], ivs2[3]});
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
                    dip::BoundaryOption::ConstantPadding) {
                  Value inputVec = builder.create<BroadcastOp>(loc, vectorTy32,
                                                               constantValue);
                                                                Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol1, ivs2[2]);
                    
            
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output2, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol1,
                                        vectorMaskTy);
                                        
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
                        Value leftMask =
                            createInvertedMask(builder, loc, strideVal,
                                               vectorMaskTy, leftMaskElem);

                        if (boundaryOptionAttr ==
                            dip::BoundaryOption::ReplicatePadding) {
                          Value paddingVal = builder.create<memref::LoadOp>(
                              loc, output1, ValueRange{c0, c0});
                          Value padding = builder.create<BroadcastOp>(
                              loc, vectorTy32, paddingVal);

                          Value leftPaddingOffset =
                              builder.create<SubIOp>(loc, c0, leftMaskElem);
                          inputVec = builder.create<vector::MaskedLoadOp>(
                              loc, vectorTy32, output1,
                              ValueRange{c0, leftPaddingOffset}, leftMask,
                              padding);
                        }
                         Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol1, ivs2[2]);
                
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output2, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol1,
                                        vectorMaskTy);
                                        
                        builder.create<scf::YieldOp>(loc);
                      },
                      [&](OpBuilder &builder, Location loc) {
                        // (colMid or colRight) & rowUp
                        Value colMidCond = builder.create<CmpIOp>(
                            loc, CmpIPredicate::sle, colLastElem, colMidHelper1);

                        builder.create<scf::IfOp>(
                            loc, colMidCond,
                            [&](OpBuilder &builder, Location loc) {
                              // colMid & rowUp
                              Value inputVec;
                              if (boundaryOptionAttr ==
                                  dip::BoundaryOption::ReplicatePadding) {
                                inputVec = builder.create<LoadOp>(
                                    loc, vectorTy32, output1,
                                    ValueRange{c0, imCol});
                              }
                               Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol1, ivs2[2]);
                                
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output2, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol1,
                                        vectorMaskTy);
                                        

                              builder.create<scf::YieldOp>(loc);
                            },
                            [&](OpBuilder &builder, Location loc) {
                              // colRight & rowUp
                              Value inputVec;
                              Value rightMaskHelper = builder.create<SubIOp>(
                                  loc, colLastElem, colMidHelper1);
                              Value rightMaskElem = builder.create<SubIOp>(
                                  loc, strideVal, rightMaskHelper);
                              Value rightMask = builder.create<CreateMaskOp>(
                                  loc, vectorMaskTy, rightMaskElem);

                              if (boundaryOptionAttr ==
                                  dip::BoundaryOption::ReplicatePadding) {
                                Value rightRange =
                                    builder.create<SubIOp>(loc, inputCol1, c1);
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, output1, ValueRange{c0, rightRange});
                                Value padding = builder.create<BroadcastOp>(
                                    loc, vectorTy32, paddingVal);

                                inputVec = builder.create<MaskedLoadOp>(
                                    loc, vectorTy32, output1,
                                    ValueRange{c0, imCol}, rightMask, padding);
                              }
                              Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol1, ivs2[2]);

                               
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output2, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol1,
                                        vectorMaskTy);
                                        


                              builder.create<scf::YieldOp>(loc);
                            });
                        builder.create<scf::YieldOp>(loc);
                      });
                }
                builder.create<scf::YieldOp>(loc);
              },
              [&](OpBuilder &builder, Location loc) {
                // rowMid or rowDown
                Value rowMidCond = builder.create<CmpIOp>(
                    loc, CmpIPredicate::slt, currRow, rowMidHelper1);

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
                                dip::BoundaryOption::ConstantPadding) {
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, constantValue);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, output1,
                                  ValueRange{imRow, leftPaddingOffset},
                                  leftMask, padding);
                            }  if (boundaryOptionAttr ==
                                       dip::BoundaryOption::ReplicatePadding) {
                              Value paddingVal = builder.create<memref::LoadOp>(
                                  loc, output1, ValueRange{imRow, c0});
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, paddingVal);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, output1,
                                  ValueRange{imRow, leftPaddingOffset},
                                  leftMask, padding);
                            }
                             Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs2[2]);
                            
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output2, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       
                            builder.create<scf::YieldOp>(loc);
                          },
                          [&](OpBuilder &builder, Location loc) {
                            // (colMid or colRight) & rowMid
                            Value colMidCond = builder.create<CmpIOp>(
                                loc, CmpIPredicate::sle, colLastElem,
                                colMidHelper1);

                            builder.create<scf::IfOp>(
                                loc, colMidCond,
                                [&](OpBuilder &builder, Location loc) {
                                  // colMid & rowMid
                                  Value inputVec = builder.create<LoadOp>(
                                      loc, vectorTy32, output1,
                                      ValueRange{imRow, imCol});
                                       Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol1, ivs2[2]);
                                    
                                 
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output2, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       


                                  builder.create<scf::YieldOp>(loc);
                                },
                                [&](OpBuilder &builder, Location loc) {
                                  // colRight & rowMid
                                  Value inputVec;
                                  Value rightMaskHelper =
                                      builder.create<SubIOp>(loc, colLastElem,
                                                             colMidHelper1);
                                  Value rightMaskElem = builder.create<SubIOp>(
                                      loc, strideVal, rightMaskHelper);
                                  Value rightMask =
                                      builder.create<CreateMaskOp>(
                                          loc, vectorMaskTy, rightMaskElem);

                                  if (boundaryOptionAttr ==
                                      dip::BoundaryOption::ConstantPadding) {
                                    Value padding = builder.create<BroadcastOp>(
                                        loc, vectorTy32, constantValue);

                                    inputVec = builder.create<MaskedLoadOp>(
                                        loc, vectorTy32, output1,
                                        ValueRange{imRow, imCol}, rightMask,
                                        padding);
                                  } else if (boundaryOptionAttr ==
                                             dip::BoundaryOption::
                                                 ReplicatePadding) {
                                    Value rightRange = builder.create<SubIOp>(
                                        loc, inputCol, c1);
                                    Value paddingVal =
                                        builder.create<memref::LoadOp>(
                                            loc, output1,
                                            ValueRange{imRow, rightRange});
                                    Value padding = builder.create<BroadcastOp>(
                                        loc, vectorTy32, paddingVal);

                                    inputVec = builder.create<MaskedLoadOp>(
                                        loc, vectorTy32, output1,
                                        ValueRange{imRow, imCol}, rightMask,
                                        padding);
                                  }
                                  Value tailCond = tailChecker(
                                      builder, loc, calcHelper, strideVal,
                                      kernelSize, c1, pseudoCol, ivs2[2]);
                                    
                                
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output2, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       

                                  builder.create<scf::YieldOp>(loc);
                                });
                            builder.create<scf::YieldOp>(loc);
                          });
                      builder.create<scf::YieldOp>(loc);
                    },
                    [&](OpBuilder &builder, Location loc) {
                      // rowDown
                      if (boundaryOptionAttr ==
                          dip::BoundaryOption::ConstantPadding) {
                        Value inputVec = builder.create<BroadcastOp>(
                            loc, vectorTy32, constantValue);
                             Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs2[2]);

                          
                        
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output2, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       
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
                              Value leftMask = createInvertedMask(
                                  builder, loc, strideVal, vectorMaskTy,
                                  leftMaskElem);

                              if (boundaryOptionAttr ==
                                  dip::BoundaryOption::ReplicatePadding) {
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, input, ValueRange{downRange, c0});
                                Value padding = builder.create<BroadcastOp>(
                                    loc, vectorTy32, paddingVal);

                                Value leftPaddingOffset =
                                    builder.create<SubIOp>(loc, c0,
                                                           leftMaskElem);
                                inputVec = builder.create<MaskedLoadOp>(
                                    loc, vectorTy32, output1,
                                    ValueRange{downRange, leftPaddingOffset},
                                    leftMask, padding);
                              }
                               Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs2[2]);
 
                                
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output2, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       

                              builder.create<scf::YieldOp>(loc);
                            },
                            [&](OpBuilder &builder, Location loc) {
                              // (colMid or colRight) & rowDown
                              Value colMidCond = builder.create<CmpIOp>(
                                  loc, CmpIPredicate::sle, colLastElem,
                                  colMidHelper1);

                              builder.create<scf::IfOp>(
                                  loc, colMidCond,
                                  [&](OpBuilder &builder, Location loc) {
                                    // colMid & rowDown
                                    Value inputVec;
                                    Value downRange = builder.create<SubIOp>(
                                        loc, inputRow, c1);
                                    if (boundaryOptionAttr ==
                                        dip::BoundaryOption::ReplicatePadding) {
                                      inputVec = builder.create<LoadOp>(
                                          loc, vectorTy32, output1,
                                          ValueRange{downRange, imCol});
                                    }
                                     Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs2[2]);
                                   
                                     
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output2, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       

                                    builder.create<scf::YieldOp>(loc);
                                  },
                                  [&](OpBuilder &builder, Location loc) {
                                    // colRight & rowDown
                                    Value inputVec;
                                    Value rightMaskHelper =
                                        builder.create<SubIOp>(loc, colLastElem,
                                                               colMidHelper1);
                                    Value rightMaskElem =
                                        builder.create<SubIOp>(loc, strideVal,
                                                               rightMaskHelper);
                                    Value rightMask =
                                        builder.create<CreateMaskOp>(
                                            loc, vectorMaskTy, rightMaskElem);

                                    Value downRange = builder.create<SubIOp>(
                                        loc, inputRow, c1);
                                    Value rightRange = builder.create<SubIOp>(
                                        loc, inputCol, c1);

                                    if (boundaryOptionAttr ==
                                        dip::BoundaryOption::ReplicatePadding) {

                                      Value paddingVal =
                                          builder.create<memref::LoadOp>(
                                              loc, output1,
                                              ValueRange{downRange,
                                                         rightRange});
                                      Value padding =
                                          builder.create<vector::BroadcastOp>(
                                              loc, vectorTy32, paddingVal);

                                      inputVec = builder.create<MaskedLoadOp>(
                                          loc, vectorTy32, output1,
                                          ValueRange{downRange, imCol},
                                          rightMask, padding);
                                    }
                                    Value tailCond = tailChecker(
                                        builder, loc, calcHelper, strideVal,
                                        kernelSize, c1, pseudoCol, ivs2[2]);
                                
                   compAndStorewTailProcessingFlaterosion(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output2, ivs2[0], ivs2[2],
                                        tailCond, Paddingvec,zeroPadding, inputCol,
                                        vectorMaskTy);
                                       
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
                SmallVector<Value, 8> lowerbounds4(2,c0);
        SmallVector<Value, 8> upperbounds4{outputrow1,outputcol1};
        SmallVector<int64_t, 8> steps4{1,1};

        buildAffineLoopNest(
            rewriter, loc, lowerbounds4, upperbounds4,steps4,
            [&](OpBuilder &builder, Location loc, ValueRange ivs4)
            {
                       Value ou = builder.create<LoadOp>(loc,VectorOne,output2,ValueRange{ivs4[0],ivs4[1]});
                       Value in = builder.create<LoadOp>(loc,VectorOne,input,ValueRange{ivs4[0],ivs4[1]});
                       Value res = builder.create<SubFOp>(loc,ou,in);
                       builder.create<StoreOp>(loc, res, output,ValueRange{ivs4[0],ivs4[1]});
            }
        );

    
    // Remove the origin convolution operation.
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;


};



} // end anonymous namespace

void populateLowerDIPConversionPatterns(RewritePatternSet &patterns,
                                        int64_t stride) {
  patterns.add<DIPCorr2DOpLowering>(patterns.getContext(), stride);
  patterns.add<DIPRotate2DOpLowering>(patterns.getContext(), stride);
  patterns.add<DIPResize2DOpLowering>(patterns.getContext(), stride);
  patterns.add<DIPErosion2DOpLowering>(patterns.getContext(),stride);
  patterns.add<DIPDilation2DOpLowering>(patterns.getContext(), stride);
  patterns.add<DIPOpening2DOpLowering>(patterns.getContext(),stride);
  patterns.add<DIPClosing2DOpLowering>(patterns.getContext(),stride);
  patterns.add<DIPTopHat2DOpLowering>(patterns.getContext(),stride);
  patterns.add<DIPBottomHat2DOpLowering>(patterns.getContext(),stride); 
}

//===----------------------------------------------------------------------===//
// LowerDIPPass
//===----------------------------------------------------------------------===//

namespace {
class LowerDIPPass : public PassWrapper<LowerDIPPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerDIPPass)
  LowerDIPPass() = default;
  LowerDIPPass(const LowerDIPPass &) {}
  explicit LowerDIPPass(int64_t strideParam) { stride = strideParam; }

  StringRef getArgument() const final { return "lower-dip"; }
  StringRef getDescription() const final { return "Lower DIP Dialect."; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<buddy::dip::DIPDialect, func::FuncDialect,
                memref::MemRefDialect, scf::SCFDialect, VectorDialect,
                AffineDialect, arith::ArithmeticDialect, math::MathDialect>();
  }

  Option<int64_t> stride{*this, "DIP-strip-mining",
                         llvm::cl::desc("Strip mining size."),
                         llvm::cl::init(32)};
};
} // end anonymous namespace.

void LowerDIPPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<AffineDialect, scf::SCFDialect, func::FuncDialect,
                         memref::MemRefDialect, VectorDialect,
                         arith::ArithmeticDialect, math::MathDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();

  RewritePatternSet patterns(context);
  populateLowerDIPConversionPatterns(patterns, stride);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerDIPPass() { PassRegistration<LowerDIPPass>(); }
} // namespace buddy
} // namespace mlir
