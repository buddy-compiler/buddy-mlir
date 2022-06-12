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
#include <vector>

using namespace mlir;
using namespace buddy;
using namespace vector;
using namespace mlir::arith;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
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

// Create an inverted mask having all 1's shifted to right side.
Value createInvertedMask(OpBuilder &builder, Location loc, Value strideVal,
                         VectorType vectorMaskTy, Value leftIndex) {
  Value leftMask = builder.create<CreateMaskOp>(loc, vectorMaskTy, leftIndex);
  Value maskInverter =
      builder.create<CreateMaskOp>(loc, vectorMaskTy, strideVal);
  Value rightMask = builder.create<SubIOp>(loc, maskInverter, leftMask);
  return rightMask;
}

class DIPCorr2DLowering : public OpRewritePattern<dip::Corr2DOp> {
public:
  using OpRewritePattern<dip::Corr2DOp>::OpRewritePattern;

  explicit DIPCorr2DLowering(MLIRContext *context, int64_t strideParam)
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

  Value diffCond = builder.create<arith::CmpFOp>(loc, CmpFPredicate::OGT,
                                                 diffCeil, diffFloor);

  return builder.create<arith::SelectOp>(loc, diffCond, floorVal, ceilVal);
}

// Bound values to permissible range of allocatable values w.r.t output image.
Value valBound(OpBuilder &builder, Location loc, Value val, Value lastElemF32,
               Value c0F32) {
  Value interm1 = builder.create<arith::MaxFOp>(loc, val, c0F32);
  return builder.create<arith::MinFOp>(loc, interm1, lastElemF32);
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

// Cast index type value to f32 type and then expand it in a vector.
Value castAndExpand(OpBuilder &builder, Location loc, Value val,
                    VectorType vecType) {
  Value interm1 = indexToF32(builder, loc, val);
  return builder.create<vector::SplatOp>(loc, vecType, interm1);
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
        Value xLowerBoundMult =
            builder.create<arith::DivUIOp>(loc, ivs[1], strideVal);
        Value xLowerBound =
            builder.create<arith::MulIOp>(loc, strideVal, xLowerBoundMult);

        Value ivs0F32 = indexToF32(builder, loc, ivs[0]);
        Value yVec = builder.create<vector::SplatOp>(loc, vectorTy32, ivs0F32);
        Value xVec = iotaVec(builder, loc, ctx, xLowerBound, strideVal,
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
        Value xLowerBoundMult =
            builder.create<arith::DivUIOp>(loc, ivs[1], strideVal);
        Value xLowerBound =
            builder.create<arith::MulIOp>(loc, strideVal, xLowerBoundMult);

        Value ivs0F32 = indexToF32(builder, loc, ivs[0]);
        Value yVec = builder.create<vector::SplatOp>(loc, vectorTy32, ivs0F32);
        Value xVec = iotaVec(builder, loc, ctx, xLowerBound, strideVal,
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

    // Obtain extreme allocatable value(s) in output for bounding purpose.
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

    SmallVector<intptr_t, 8> steps{1, stride};
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
              c0, c0F32, c1F32Vec, vectorTy32, stride, f32);

          shearTransformController(
              builder, loc, ctx, lowerBounds2, upperBounds2, steps,
              strideTailVal, input, output, sinVec, tanVec, inputRowF32Vec,
              inputColF32Vec, inputCenterYF32Vec, inputCenterXF32Vec,
              outputCenterYF32Vec, outputCenterXF32Vec, outputRowLastElemF32,
              outputColLastElemF32, c0, c0F32, c1F32Vec, vectorTy32, stride,
              f32);

          builder.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &builder, Location loc) {
          standardRotateController(
              builder, loc, ctx, lowerBounds1, upperBounds1, steps, strideVal,
              input, output, sinVec, angleVal, inputRowF32Vec, inputColF32Vec,
              inputCenterYF32Vec, inputCenterXF32Vec, outputCenterYF32Vec,
              outputCenterXF32Vec, outputRowLastElemF32, outputColLastElemF32,
              c0, c0F32, c1F32Vec, vectorTy32, stride, f32);

          standardRotateController(
              builder, loc, ctx, lowerBounds2, upperBounds2, steps,
              strideTailVal, input, output, sinVec, angleVal, inputRowF32Vec,
              inputColF32Vec, inputCenterYF32Vec, inputCenterXF32Vec,
              outputCenterYF32Vec, outputCenterXF32Vec, outputRowLastElemF32,
              outputColLastElemF32, c0, c0F32, c1F32Vec, vectorTy32, stride,
              f32);

          builder.create<scf::YieldOp>(loc);
        });

    // Remove the origin rotation operation.
    rewriter.eraseOp(op);
    return success();
  }

  int64_t stride;
};
} // end anonymous namespace

void populateLowerDIPConversionPatterns(RewritePatternSet &patterns,
                                        int64_t stride) {
  patterns.add<DIPCorr2DLowering>(patterns.getContext(), stride);
  patterns.add<DIPRotate2DOpLowering>(patterns.getContext(), stride);
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
    registry.insert<buddy::dip::DIPDialect, func::FuncDialect,
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
