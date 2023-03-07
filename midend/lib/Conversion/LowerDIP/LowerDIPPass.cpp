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

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Bufferization/Transforms/Bufferize.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <vector>

#include "DIP/DIPDialect.h"
#include "DIP/DIPOps.h"
#include "Utils/DIPUtils.h"
#include "Utils/Utils.h"

using namespace mlir;
using namespace buddy;

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
    auto *ctx = op->getContext();

    // Register operand values.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    Value centerX = op->getOperand(3);
    Value centerY = op->getOperand(4);
    Value constantValue = op->getOperand(5);
    dip::BoundaryOption boundaryOptionAttr = op.getBoundaryOption();
    Value strideVal = rewriter.create<arith::ConstantIndexOp>(loc, stride);

    auto inElemTy = input.getType().cast<MemRefType>().getElementType();
    dip::DIP_ERROR error = dip::checkDIPCommonTypes<dip::Corr2DOp>(
        op, {input, kernel, output, constantValue});

    if (error == dip::DIP_ERROR::INCONSISTENT_TYPES) {
      return op->emitOpError() << "input, kernel, output and constant must "
                                  "have the same element type";
    } else if (error == dip::DIP_ERROR::UNSUPPORTED_TYPE) {
      return op->emitOpError() << "supports only f32, f64 and integer types. "
                               << inElemTy << "is passed";
    }

    traverseImagewBoundaryExtrapolation(rewriter, loc, ctx, input, kernel,
                                        output, centerX, centerY, constantValue,
                                        strideVal, inElemTy, boundaryOptionAttr,
                                        stride, dip::DIP_OP::CORRELATION_2D);
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

    auto inElemTy = input.getType().cast<MemRefType>().getElementType();
    dip::DIP_ERROR error =
        dip::checkDIPCommonTypes<dip::Rotate2DOp>(op, {input, output});

    if (error == dip::DIP_ERROR::INCONSISTENT_TYPES) {
      return op->emitOpError()
             << "input, and output must have the same element type";
    } else if (error == dip::DIP_ERROR::UNSUPPORTED_TYPE) {
      return op->emitOpError() << "supports only f32, f64 and integer types. "
                               << inElemTy << "is passed";
    }

    Value strideVal = rewriter.create<arith::ConstantIndexOp>(loc, stride);
    FloatType f32 = FloatType::getF32(ctx);
    VectorType vectorTy32 = VectorType::get({stride}, f32);

    // Create constant indices.
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

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
    Value inputCenterY = dip::getCenter(rewriter, loc, ctx, inputRow);
    Value inputCenterX = dip::getCenter(rewriter, loc, ctx, inputCol);

    Value inputCenterYF32Vec =
        castAndExpand(rewriter, loc, inputCenterY, vectorTy32);
    Value inputCenterXF32Vec =
        castAndExpand(rewriter, loc, inputCenterX, vectorTy32);

    // Get output image center.
    Value outputCenterY = dip::getCenter(rewriter, loc, ctx, outputRow);
    Value outputCenterX = dip::getCenter(rewriter, loc, ctx, outputCol);

    Value outputCenterYF32Vec =
        castAndExpand(rewriter, loc, outputCenterY, vectorTy32);
    Value outputCenterXF32Vec =
        castAndExpand(rewriter, loc, outputCenterX, vectorTy32);

    // Get sin(angle) which will be used in further calculations.
    Value sinVal = rewriter.create<math::SinOp>(loc, angleVal);
    Value sinVec =
        rewriter.create<vector::BroadcastOp>(loc, vectorTy32, sinVal);

    // Get tan(angle / 2) which will be used in further calculations.
    Value tanVal = dip::customTanVal(rewriter, loc, angleVal);
    Value tanVec =
        rewriter.create<vector::BroadcastOp>(loc, vectorTy32, tanVal);

    // Determine the condition for chosing ideal rotation strategy.
    Value tanBound =
        rewriter.create<arith::ConstantFloatOp>(loc, (llvm::APFloat)8.10f, f32);
    Value tanValAbs = rewriter.create<math::AbsFOp>(loc, tanVal);
    Value transformCond = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OGT, tanBound, tanValAbs);

    // For both rotation strategies, tail processing is handled in second call.
    rewriter.create<scf::IfOp>(
        loc, transformCond,
        [&](OpBuilder &builder, Location loc) {
          dip::shearTransformController(
              builder, loc, ctx, lowerBounds1, upperBounds1, steps, strideVal,
              input, output, sinVec, tanVec, inputRowF32Vec, inputColF32Vec,
              inputCenterYF32Vec, inputCenterXF32Vec, outputCenterYF32Vec,
              outputCenterXF32Vec, outputRowLastElemF32, outputColLastElemF32,
              inputRowLastElemF32, inputColLastElemF32, c0, c0F32, c1F32Vec,
              vectorTy32, stride, f32);

          dip::shearTransformController(
              builder, loc, ctx, lowerBounds2, upperBounds2, steps,
              strideTailVal, input, output, sinVec, tanVec, inputRowF32Vec,
              inputColF32Vec, inputCenterYF32Vec, inputCenterXF32Vec,
              outputCenterYF32Vec, outputCenterXF32Vec, outputRowLastElemF32,
              outputColLastElemF32, inputRowLastElemF32, inputColLastElemF32,
              c0, c0F32, c1F32Vec, vectorTy32, stride, f32);

          builder.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &builder, Location loc) {
          dip::standardRotateController(
              builder, loc, ctx, lowerBounds1, upperBounds1, steps, strideVal,
              input, output, sinVec, angleVal, inputRowF32Vec, inputColF32Vec,
              inputCenterYF32Vec, inputCenterXF32Vec, outputCenterYF32Vec,
              outputCenterXF32Vec, outputRowLastElemF32, outputColLastElemF32,
              inputRowLastElemF32, inputColLastElemF32, c0, c0F32, c1F32Vec,
              vectorTy32, stride, f32);

          dip::standardRotateController(
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
    auto interpolationAttr = op.getInterpolationType();
    Value strideVal = rewriter.create<arith::ConstantIndexOp>(loc, stride);

    auto inElemTy = input.getType().cast<MemRefType>().getElementType();
    dip::DIP_ERROR error =
        dip::checkDIPCommonTypes<dip::Resize2DOp>(op, {input, output});

    if (error == dip::DIP_ERROR::INCONSISTENT_TYPES) {
      return op->emitOpError()
             << "input, and output must have the same element type";
    } else if (error == dip::DIP_ERROR::UNSUPPORTED_TYPE) {
      return op->emitOpError() << "supports only f32, f64 and integer types. "
                               << inElemTy << "is passed";
    }

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c0F32 = indexToF32(rewriter, loc, c0);

    Value inputRow = rewriter.create<memref::DimOp>(loc, input, c0);
    Value inputCol = rewriter.create<memref::DimOp>(loc, input, c1);

    Value outputRow = rewriter.create<memref::DimOp>(loc, output, c0);
    Value outputCol = rewriter.create<memref::DimOp>(loc, output, c1);

    // Determine lower bound for second call of resize function (this is done
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
      dip::NearestNeighbourInterpolationResizing(
          rewriter, loc, ctx, lowerBounds1, upperBounds1, steps, strideVal,
          input, output, horizontalScalingFactorVec, verticalScalingFactorVec,
          outputRowLastElemF32, outputColLastElemF32, inputRowLastElemF32,
          inputColLastElemF32, vectorTy32, stride, c0, c0F32);

      dip::NearestNeighbourInterpolationResizing(
          rewriter, loc, ctx, lowerBounds2, upperBounds2, steps, strideTailVal,
          input, output, horizontalScalingFactorVec, verticalScalingFactorVec,
          outputRowLastElemF32, outputColLastElemF32, inputRowLastElemF32,
          inputColLastElemF32, vectorTy32, stride, c0, c0F32);
    } else if (interpolationAttr ==
               dip::InterpolationType::BilinearInterpolation) {
      Value c1F32 = indexToF32(rewriter, loc, c1);

      dip::BilinearInterpolationResizing(
          rewriter, loc, ctx, lowerBounds1, upperBounds1, steps, strideVal,
          input, output, horizontalScalingFactorVec, verticalScalingFactorVec,
          outputRowLastElemF32, outputColLastElemF32, inputRowLastElemF32,
          inputColLastElemF32, vectorTy32, stride, c0, c0F32, c1F32);

      dip::BilinearInterpolationResizing(
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

class DIPErosion2DOpLowering : public OpRewritePattern<dip::Erosion2DOp> {
public:
  using OpRewritePattern<dip::Erosion2DOp>::OpRewritePattern;

  explicit DIPErosion2DOpLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
  }

  LogicalResult matchAndRewrite(dip::Erosion2DOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto *ctx = op->getContext();

    // Register operand values.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    Value copymemref = op->getOperand(3);
    Value centerX = op->getOperand(4);
    Value centerY = op->getOperand(5);
    Value iterations = op->getOperand(6);
    Value constantValue = op->getOperand(7);
    dip::BoundaryOption boundaryOptionAttr = op.getBoundaryOption();
    Value strideVal = rewriter.create<arith::ConstantIndexOp>(loc, stride);

    auto inElemTy = input.getType().cast<MemRefType>().getElementType();
    dip::DIP_ERROR error = dip::checkDIPCommonTypes<dip::Erosion2DOp>(
        op, {input, kernel, output, copymemref, constantValue});

    if (error == dip::DIP_ERROR::INCONSISTENT_TYPES) {
      return op->emitOpError()
             << "input, kernel, output, copymemref and constant must "
                "have the same element type";
    } else if (error == dip::DIP_ERROR::UNSUPPORTED_TYPE) {
      return op->emitOpError() << "supports only f32, f64 and integer types. "
                               << inElemTy << "is passed";
    }

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    rewriter.create<AffineForOp>(
        loc, ValueRange{c0}, rewriter.getDimIdentityMap(),
        ValueRange{iterations}, rewriter.getDimIdentityMap(), /*Step=*/1,
        std::nullopt,
        [&](OpBuilder &builder, Location nestedLoc, Value iv,
            ValueRange itrArgs) {
          Value cond = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sge, iv, c1);
          builder.create<scf::IfOp>(
              loc, cond, [&](OpBuilder &builder, Location loc) {
                builder.create<memref::CopyOp>(loc, output, input);
                builder.create<scf::YieldOp>(loc);
              });
          builder.create<memref::CopyOp>(loc, copymemref, output);
          traverseImagewBoundaryExtrapolation(
              rewriter, loc, ctx, input, kernel, output, centerX, centerY,
              constantValue, strideVal, inElemTy, boundaryOptionAttr, stride,
              dip::DIP_OP::EROSION_2D);
          builder.create<AffineYieldOp>(loc);
        });

    // Remove the origin erosion operation.
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;
};

class DIPDilation2DOpLowering : public OpRewritePattern<dip::Dilation2DOp> {

public:
  using OpRewritePattern<dip::Dilation2DOp>::OpRewritePattern;

  explicit DIPDilation2DOpLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
  }

  LogicalResult matchAndRewrite(dip::Dilation2DOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto *ctx = op->getContext();

    // Register operand values.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    Value copymemref = op->getOperand(3);
    Value centerX = op->getOperand(4);
    Value centerY = op->getOperand(5);
    Value iterations = op->getOperand(6);
    Value constantValue = op->getOperand(7);
    dip::BoundaryOption boundaryOptionAttr = op.getBoundaryOption();
    Value strideVal = rewriter.create<arith::ConstantIndexOp>(loc, stride);

    auto inElemTy = input.getType().cast<MemRefType>().getElementType();
    dip::DIP_ERROR error = dip::checkDIPCommonTypes<dip::Dilation2DOp>(
        op, {input, kernel, output, copymemref, constantValue});

    if (error == dip::DIP_ERROR::INCONSISTENT_TYPES) {
      return op->emitOpError()
             << "input, kernel, output, copymemref and constant must "
                "have the same element type";
    } else if (error == dip::DIP_ERROR::UNSUPPORTED_TYPE) {
      return op->emitOpError() << "supports only f32, f64 and integer types. "
                               << inElemTy << "is passed";
    }
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    rewriter.create<AffineForOp>(
        loc, ValueRange{c0}, rewriter.getDimIdentityMap(),
        ValueRange{iterations}, rewriter.getDimIdentityMap(), /*Step=*/1,
        std::nullopt,
        [&](OpBuilder &builder, Location nestedLoc, Value iv,
            ValueRange itrArgs) {
          Value cond = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sge, iv, c1);
          builder.create<scf::IfOp>(
              loc, cond, [&](OpBuilder &builder, Location loc) {
                builder.create<memref::CopyOp>(loc, output, input);
                builder.create<scf::YieldOp>(loc);
              });
          builder.create<memref::CopyOp>(loc, copymemref, output);
          traverseImagewBoundaryExtrapolation(
              rewriter, loc, ctx, input, kernel, output, centerX, centerY,
              constantValue, strideVal, inElemTy, boundaryOptionAttr, stride,
              dip::DIP_OP::DILATION_2D);
          builder.create<AffineYieldOp>(loc);
        });

    // Remove the origin dilation operation.
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;
};

class DIPOpening2DOpLowering : public OpRewritePattern<dip::Opening2DOp> {
public:
  using OpRewritePattern<dip::Opening2DOp>::OpRewritePattern;

  explicit DIPOpening2DOpLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
  }

  LogicalResult matchAndRewrite(dip::Opening2DOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto *ctx = op->getContext();

    // Register operand values.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    Value output1 = op->getOperand(3);
    Value copymemref = op->getOperand(4);
    Value copymemref1 = op->getOperand(5);
    Value centerX = op->getOperand(6);
    Value centerY = op->getOperand(7);
    Value iterations = op->getOperand(8);
    Value constantValue = op->getOperand(9);
    dip::BoundaryOption boundaryOptionAttr = op.getBoundaryOption();
    Value strideVal = rewriter.create<arith::ConstantIndexOp>(loc, stride);

    auto inElemTy = input.getType().cast<MemRefType>().getElementType();
    dip::DIP_ERROR error = dip::checkDIPCommonTypes<dip::Opening2DOp>(
        op, {input, kernel, output, output1, copymemref, copymemref1,
             constantValue});

    if (error == dip::DIP_ERROR::INCONSISTENT_TYPES) {
      return op->emitOpError() << "input, kernel, output, output1, copymemref, "
                                  "copymemref1 and constant must "
                                  "have the same element type";
    } else if (error == dip::DIP_ERROR::UNSUPPORTED_TYPE) {
      return op->emitOpError() << "supports only f32, f64 and integer types. "
                               << inElemTy << "is passed";
    }
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    rewriter.create<AffineForOp>(
        loc, ValueRange{c0}, rewriter.getDimIdentityMap(),
        ValueRange{iterations}, rewriter.getDimIdentityMap(), /*Step=*/1,
        std::nullopt,
        [&](OpBuilder &builder, Location nestedLoc, Value iv,
            ValueRange itrArgs) {
          Value cond = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sge, iv, c1);
          builder.create<scf::IfOp>(
              loc, cond, [&](OpBuilder &builder, Location loc) {
                builder.create<memref::CopyOp>(loc, output1, input);
                builder.create<scf::YieldOp>(loc);
              });
          builder.create<memref::CopyOp>(loc, copymemref, output1);
          traverseImagewBoundaryExtrapolation(
              rewriter, loc, ctx, input, kernel, output1, centerX, centerY,
              constantValue, strideVal, inElemTy, boundaryOptionAttr, stride,
              dip::DIP_OP::EROSION_2D);
          builder.create<AffineYieldOp>(loc);
        });

    rewriter.create<AffineForOp>(
        loc, ValueRange{c0}, rewriter.getDimIdentityMap(),
        ValueRange{iterations}, rewriter.getDimIdentityMap(), /*Step=*/1,
        std::nullopt,
        [&](OpBuilder &builder, Location nestedLoc, Value iv,
            ValueRange itrArgs) {
          Value cond = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sge, iv, c1);
          builder.create<scf::IfOp>(
              loc, cond, [&](OpBuilder &builder, Location loc) {
                builder.create<memref::CopyOp>(loc, output, output1);
                builder.create<scf::YieldOp>(loc);
              });
          builder.create<memref::CopyOp>(loc, copymemref1, output);
          traverseImagewBoundaryExtrapolation(
              rewriter, loc, ctx, output1, kernel, output, centerX, centerY,
              constantValue, strideVal, inElemTy, boundaryOptionAttr, stride,
              dip::DIP_OP::DILATION_2D);
          builder.create<AffineYieldOp>(loc);
        });

    // Remove the origin opening operation.
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;
};

class DIPClosing2DOpLowering : public OpRewritePattern<dip::Closing2DOp> {
public:
  using OpRewritePattern<dip::Closing2DOp>::OpRewritePattern;

  explicit DIPClosing2DOpLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
  }

  LogicalResult matchAndRewrite(dip::Closing2DOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto *ctx = op->getContext();

    // Register operand values.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    Value output1 = op->getOperand(3);
    Value copymemref = op->getOperand(4);
    Value copymemref1 = op->getOperand(5);
    Value centerX = op->getOperand(6);
    Value centerY = op->getOperand(7);
    Value iterations = op->getOperand(8);
    Value constantValue = op->getOperand(9);
    dip::BoundaryOption boundaryOptionAttr = op.getBoundaryOption();
    Value strideVal = rewriter.create<arith::ConstantIndexOp>(loc, stride);

    auto inElemTy = input.getType().cast<MemRefType>().getElementType();
    dip::DIP_ERROR error = dip::checkDIPCommonTypes<dip::Closing2DOp>(
        op, {input, kernel, output, output1, copymemref, copymemref1,
             constantValue});

    if (error == dip::DIP_ERROR::INCONSISTENT_TYPES) {
      return op->emitOpError() << "input, kernel, output, output1, copymemref, "
                                  "copymemref1 and constant must "
                                  "have the same element type";
    } else if (error == dip::DIP_ERROR::UNSUPPORTED_TYPE) {
      return op->emitOpError() << "supports only f32, f64 and integer types. "
                               << inElemTy << "is passed";
    }
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    rewriter.create<AffineForOp>(
        loc, ValueRange{c0}, rewriter.getDimIdentityMap(),
        ValueRange{iterations}, rewriter.getDimIdentityMap(), /*Step=*/1,
        std::nullopt,
        [&](OpBuilder &builder, Location nestedLoc, Value iv,
            ValueRange itrArgs) {
          Value cond = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sge, iv, c1);
          builder.create<scf::IfOp>(
              loc, cond, [&](OpBuilder &builder, Location loc) {
                builder.create<memref::CopyOp>(loc, output1, input);
                builder.create<scf::YieldOp>(loc);
              });
          builder.create<memref::CopyOp>(loc, copymemref, output1);
          traverseImagewBoundaryExtrapolation(
              rewriter, loc, ctx, input, kernel, output1, centerX, centerY,
              constantValue, strideVal, inElemTy, boundaryOptionAttr, stride,
              dip::DIP_OP::DILATION_2D);
          builder.create<AffineYieldOp>(loc);
        });

    rewriter.create<AffineForOp>(
        loc, ValueRange{c0}, rewriter.getDimIdentityMap(),
        ValueRange{iterations}, rewriter.getDimIdentityMap(), /*Step=*/1,
        std::nullopt,
        [&](OpBuilder &builder, Location nestedLoc, Value iv,
            ValueRange itrArgs) {
          Value cond = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sge, iv, c1);
          builder.create<scf::IfOp>(
              loc, cond, [&](OpBuilder &builder, Location loc) {
                builder.create<memref::CopyOp>(loc, output, output1);
                builder.create<scf::YieldOp>(loc);
              });
          builder.create<memref::CopyOp>(loc, copymemref1, output);
          traverseImagewBoundaryExtrapolation(
              rewriter, loc, ctx, output1, kernel, output, centerX, centerY,
              constantValue, strideVal, inElemTy, boundaryOptionAttr, stride,
              dip::DIP_OP::EROSION_2D);
          builder.create<AffineYieldOp>(loc);
        });

    // Remove the origin closing operation.
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;
};

class DIPTopHat2DOpLowering : public OpRewritePattern<dip::TopHat2DOp> {
public:
  using OpRewritePattern<dip::TopHat2DOp>::OpRewritePattern;

  explicit DIPTopHat2DOpLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
  }

  LogicalResult matchAndRewrite(dip::TopHat2DOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto *ctx = op->getContext();

    // Create constant indices.
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Register operand values.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    Value output1 = op->getOperand(3);
    Value output2 = op->getOperand(4);
    Value input1 = op->getOperand(5);
    Value copymemref = op->getOperand(6);
    Value copymemref1 = op->getOperand(7);
    Value centerX = op->getOperand(8);
    Value centerY = op->getOperand(9);
    Value iterations = op->getOperand(10);
    Value constantValue = op->getOperand(11);
    dip::BoundaryOption boundaryOptionAttr = op.getBoundaryOption();
    Value strideVal = rewriter.create<arith::ConstantIndexOp>(loc, stride);

    auto inElemTy = input.getType().cast<MemRefType>().getElementType();
    auto bitWidth = inElemTy.getIntOrFloatBitWidth();
    dip::DIP_ERROR error = dip::checkDIPCommonTypes<dip::TopHat2DOp>(
        op, {input, kernel, output, output1, output2, input1, copymemref,
             copymemref1, constantValue});

    if (error == dip::DIP_ERROR::INCONSISTENT_TYPES) {
      return op->emitOpError()
             << "input, kernel, output, output1, output2, input1, copymemref, "
                "copymemref1 and constant must "
                "have the same element type";
    } else if (error == dip::DIP_ERROR::UNSUPPORTED_TYPE) {
      return op->emitOpError() << "supports only f32, f64 and integer types. "
                               << inElemTy << "is passed";
    }
    rewriter.create<memref::CopyOp>(loc, input, input1);
    rewriter.create<AffineForOp>(
        loc, ValueRange{c0}, rewriter.getDimIdentityMap(),
        ValueRange{iterations}, rewriter.getDimIdentityMap(), /*Step=*/1,
        std::nullopt,
        [&](OpBuilder &builder, Location nestedLoc, Value iv,
            ValueRange itrArgs) {
          Value cond = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sge, iv, c1);
          builder.create<scf::IfOp>(
              loc, cond, [&](OpBuilder &builder, Location loc) {
                builder.create<memref::CopyOp>(loc, output1, input1);
                builder.create<scf::YieldOp>(loc);
              });
          builder.create<memref::CopyOp>(loc, copymemref, output1);
          traverseImagewBoundaryExtrapolation(
              rewriter, loc, ctx, input1, kernel, output1, centerX, centerY,
              constantValue, strideVal, inElemTy, boundaryOptionAttr, stride,
              dip::DIP_OP::EROSION_2D);
          builder.create<AffineYieldOp>(loc);
        });

    rewriter.create<AffineForOp>(
        loc, ValueRange{c0}, rewriter.getDimIdentityMap(),
        ValueRange{iterations}, rewriter.getDimIdentityMap(), /*Step=*/1,
        std::nullopt,
        [&](OpBuilder &builder, Location nestedLoc, Value iv,
            ValueRange itrArgs) {
          Value cond = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sge, iv, c1);
          builder.create<scf::IfOp>(
              loc, cond, [&](OpBuilder &builder, Location loc) {
                builder.create<memref::CopyOp>(loc, output2, output1);
                builder.create<scf::YieldOp>(loc);
              });
          builder.create<memref::CopyOp>(loc, copymemref1, output2);
          traverseImagewBoundaryExtrapolation(
              rewriter, loc, ctx, output1, kernel, output2, centerX, centerY,
              constantValue, strideVal, inElemTy, boundaryOptionAttr, stride,
              dip::DIP_OP::DILATION_2D);
          builder.create<AffineYieldOp>(loc);
        });

    Value outputRow = rewriter.create<memref::DimOp>(loc, output, c0);
    Value outputCol = rewriter.create<memref::DimOp>(loc, output, c1);

    SmallVector<Value, 8> lowerbounds4(2, c0);
    SmallVector<Value, 8> upperbounds4{outputRow, outputCol};
    SmallVector<int64_t, 8> steps4{1, stride};
    VectorType vectorTy32 = VectorType::get({stride}, inElemTy);
    IntegerType i1 = IntegerType::get(ctx, 1);
    VectorType vectorMaskTy = VectorType::get({stride}, i1);
    Value zeroPaddingElem =
        dip::insertZeroConstantOp(ctx, rewriter, loc, inElemTy);
    Value zeroPaddingVec =
        rewriter.create<vector::BroadcastOp>(loc, vectorTy32, zeroPaddingElem);

    if (inElemTy.isF32() || inElemTy.isF64()) {
      buildAffineLoopNest(
          rewriter, loc, lowerbounds4, upperbounds4, steps4,
          [&](OpBuilder &builder, Location loc, ValueRange ivs4) {
            Value pseudoCol =
                builder.create<arith::AddIOp>(loc, ivs4[1], strideVal);
            Value pseudoCol1 =
                builder.create<arith::SubIOp>(loc, pseudoCol, c1);
            Value cond = builder.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::sgt, pseudoCol1, outputCol);
            builder.create<scf::IfOp>(
                loc, cond,
                [&](OpBuilder &builder, Location loc) {
                  Value res =
                      builder.create<arith::SubIOp>(loc, pseudoCol1, outputCol);
                  Value maskVal =
                      builder.create<arith::SubIOp>(loc, strideVal, res);
                  Value maskVec = builder.create<vector::CreateMaskOp>(
                      loc, vectorMaskTy, maskVal);
                  Value ou1 = builder.create<vector::MaskedLoadOp>(
                      loc, vectorTy32, output2, ValueRange{ivs4[0], ivs4[1]},
                      maskVec, zeroPaddingVec);
                  Value ou2 = builder.create<vector::MaskedLoadOp>(
                      loc, vectorTy32, input, ValueRange{ivs4[0], ivs4[1]},
                      maskVec, zeroPaddingVec);
                  Value resVec = builder.create<arith::SubFOp>(loc, ou2, ou1);
                  builder.create<vector::MaskedStoreOp>(
                      loc, output, ValueRange{ivs4[0], ivs4[1]}, maskVec,
                      resVec);
                  builder.create<scf::YieldOp>(loc);
                },
                [&](OpBuilder &builder, Location loc) {
                  Value ou1 = builder.create<vector::LoadOp>(
                      loc, vectorTy32, output2, ValueRange{ivs4[0], ivs4[1]});
                  Value ou2 = builder.create<vector::LoadOp>(
                      loc, vectorTy32, input, ValueRange{ivs4[0], ivs4[1]});
                  Value resVec = builder.create<arith::SubFOp>(loc, ou2, ou1);
                  builder.create<vector::StoreOp>(loc, resVec, output,
                                                  ValueRange{ivs4[0], ivs4[1]});
                  builder.create<scf::YieldOp>(loc);
                });
          }

      );
    } else if (inElemTy.isInteger(bitWidth)) {
      buildAffineLoopNest(
          rewriter, loc, lowerbounds4, upperbounds4, steps4,
          [&](OpBuilder &builder, Location loc, ValueRange ivs4) {
            Value pseudoCol =
                builder.create<arith::AddIOp>(loc, ivs4[1], strideVal);
            Value pseudoCol1 =
                builder.create<arith::SubIOp>(loc, pseudoCol, c1);
            Value cond = builder.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::sgt, pseudoCol1, outputCol);
            builder.create<scf::IfOp>(
                loc, cond,
                [&](OpBuilder &builder, Location loc) {
                  Value res =
                      builder.create<arith::SubIOp>(loc, pseudoCol1, outputCol);
                  Value maskVal =
                      builder.create<arith::SubIOp>(loc, strideVal, res);
                  Value maskVec = builder.create<vector::CreateMaskOp>(
                      loc, vectorMaskTy, maskVal);
                  Value ou1 = builder.create<vector::MaskedLoadOp>(
                      loc, vectorTy32, output1, ValueRange{ivs4[0], ivs4[1]},
                      maskVec, zeroPaddingVec);
                  Value ou2 = builder.create<vector::MaskedLoadOp>(
                      loc, vectorTy32, output2, ValueRange{ivs4[0], ivs4[1]},
                      maskVec, zeroPaddingVec);
                  Value resVec = builder.create<arith::SubIOp>(loc, ou1, ou2);
                  builder.create<vector::MaskedStoreOp>(
                      loc, output, ValueRange{ivs4[0], ivs4[1]}, maskVec,
                      resVec);
                  builder.create<scf::YieldOp>(loc);
                },
                [&](OpBuilder &builder, Location loc) {
                  Value ou1 = builder.create<vector::LoadOp>(
                      loc, vectorTy32, output1, ValueRange{ivs4[0], ivs4[1]});
                  Value ou2 = builder.create<vector::LoadOp>(
                      loc, vectorTy32, output2, ValueRange{ivs4[0], ivs4[1]});
                  Value resVec = builder.create<arith::SubIOp>(loc, ou1, ou2);
                  builder.create<vector::StoreOp>(loc, resVec, output,
                                                  ValueRange{ivs4[0], ivs4[1]});
                  builder.create<scf::YieldOp>(loc);
                });
          }

      );
    }

    // Remove the origin tophat operation.
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;
};

class DIPBottomHat2DOpLowering : public OpRewritePattern<dip::BottomHat2DOp> {
public:
  using OpRewritePattern<dip::BottomHat2DOp>::OpRewritePattern;

  explicit DIPBottomHat2DOpLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
  }

  LogicalResult matchAndRewrite(dip::BottomHat2DOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto *ctx = op->getContext();

    // Create constant indices.
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Register operand values.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    Value output1 = op->getOperand(3);
    Value output2 = op->getOperand(4);
    Value input1 = op->getOperand(5);
    Value copymemref = op->getOperand(6);
    Value copymemref1 = op->getOperand(7);
    Value centerX = op->getOperand(8);
    Value centerY = op->getOperand(9);
    Value iterations = op->getOperand(10);
    Value constantValue = op->getOperand(11);
    dip::BoundaryOption boundaryOptionAttr = op.getBoundaryOption();
    Value strideVal = rewriter.create<arith::ConstantIndexOp>(loc, stride);

    auto inElemTy = input.getType().cast<MemRefType>().getElementType();
    auto bitWidth = inElemTy.getIntOrFloatBitWidth();
    dip::DIP_ERROR error = dip::checkDIPCommonTypes<dip::BottomHat2DOp>(
        op, {input, kernel, output, output1, output2, input1, copymemref,
             copymemref1, constantValue});

    if (error == dip::DIP_ERROR::INCONSISTENT_TYPES) {
      return op->emitOpError()
             << "input, kernel, output, output1, output2, input1, copymemref, "
                "copymemref1 and constant must "
                "have the same element type";
    } else if (error == dip::DIP_ERROR::UNSUPPORTED_TYPE) {
      return op->emitOpError() << "supports only f32, f64 and integer types. "
                               << inElemTy << "is passed";
    }
    rewriter.create<memref::CopyOp>(loc, input, input1);
    rewriter.create<AffineForOp>(
        loc, ValueRange{c0}, rewriter.getDimIdentityMap(),
        ValueRange{iterations}, rewriter.getDimIdentityMap(), /*Step=*/1,
        std::nullopt,
        [&](OpBuilder &builder, Location nestedLoc, Value iv,
            ValueRange itrArgs) {
          Value cond = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sge, iv, c1);
          builder.create<scf::IfOp>(
              loc, cond, [&](OpBuilder &builder, Location loc) {
                builder.create<memref::CopyOp>(loc, output1, input1);
                builder.create<scf::YieldOp>(loc);
              });
          builder.create<memref::CopyOp>(loc, copymemref, output1);
          traverseImagewBoundaryExtrapolation(
              rewriter, loc, ctx, input1, kernel, output1, centerX, centerY,
              constantValue, strideVal, inElemTy, boundaryOptionAttr, stride,
              dip::DIP_OP::DILATION_2D);
          builder.create<AffineYieldOp>(loc);
        });

    rewriter.create<AffineForOp>(
        loc, ValueRange{c0}, rewriter.getDimIdentityMap(),
        ValueRange{iterations}, rewriter.getDimIdentityMap(), /*Step=*/1,
        std::nullopt,
        [&](OpBuilder &builder, Location nestedLoc, Value iv,
            ValueRange itrArgs) {
          Value cond = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sge, iv, c1);
          builder.create<scf::IfOp>(
              loc, cond, [&](OpBuilder &builder, Location loc) {
                builder.create<memref::CopyOp>(loc, output2, output1);
                builder.create<scf::YieldOp>(loc);
              });
          builder.create<memref::CopyOp>(loc, copymemref1, output2);
          traverseImagewBoundaryExtrapolation(
              rewriter, loc, ctx, output1, kernel, output2, centerX, centerY,
              constantValue, strideVal, inElemTy, boundaryOptionAttr, stride,
              dip::DIP_OP::EROSION_2D);
          builder.create<AffineYieldOp>(loc);
        });

    Value outputRow = rewriter.create<memref::DimOp>(loc, output, c0);
    Value outputCol = rewriter.create<memref::DimOp>(loc, output, c1);

    SmallVector<Value, 8> lowerbounds4(2, c0);
    SmallVector<Value, 8> upperbounds4{outputRow, outputCol};
    SmallVector<int64_t, 8> steps4{1, stride};

    VectorType vectorTy32 = VectorType::get({stride}, inElemTy);
    IntegerType i1 = IntegerType::get(ctx, 1);
    VectorType vectorMaskTy = VectorType::get({stride}, i1);
    Value zeroPaddingElem =
        dip::insertZeroConstantOp(ctx, rewriter, loc, inElemTy);
    Value zeroPaddingVec =
        rewriter.create<vector::BroadcastOp>(loc, vectorTy32, zeroPaddingElem);

    if (inElemTy.isF32() || inElemTy.isF64()) {
      buildAffineLoopNest(
          rewriter, loc, lowerbounds4, upperbounds4, steps4,
          [&](OpBuilder &builder, Location loc, ValueRange ivs4) {
            Value pseudoCol =
                builder.create<arith::AddIOp>(loc, ivs4[1], strideVal);
            Value pseudoCol1 =
                builder.create<arith::SubIOp>(loc, pseudoCol, c1);
            Value cond = builder.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::sgt, pseudoCol1, outputCol);
            builder.create<scf::IfOp>(
                loc, cond,
                [&](OpBuilder &builder, Location loc) {
                  Value res =
                      builder.create<arith::SubIOp>(loc, pseudoCol1, outputCol);
                  Value maskVal =
                      builder.create<arith::SubIOp>(loc, strideVal, res);
                  Value maskVec = builder.create<vector::CreateMaskOp>(
                      loc, vectorMaskTy, maskVal);
                  Value ou = builder.create<vector::MaskedLoadOp>(
                      loc, vectorTy32, output2, ValueRange{ivs4[0], ivs4[1]},
                      maskVec, zeroPaddingVec);
                  Value in = builder.create<vector::MaskedLoadOp>(
                      loc, vectorTy32, input, ValueRange{ivs4[0], ivs4[1]},
                      maskVec, zeroPaddingVec);
                  Value resVec = builder.create<arith::SubFOp>(loc, ou, in);
                  builder.create<vector::MaskedStoreOp>(
                      loc, output, ValueRange{ivs4[0], ivs4[1]}, maskVec,
                      resVec);
                  builder.create<scf::YieldOp>(loc);
                },
                [&](OpBuilder &builder, Location loc) {
                  Value ou = builder.create<vector::LoadOp>(
                      loc, vectorTy32, output2, ValueRange{ivs4[0], ivs4[1]});
                  Value in = builder.create<vector::LoadOp>(
                      loc, vectorTy32, input, ValueRange{ivs4[0], ivs4[1]});
                  Value resVec = builder.create<arith::SubFOp>(loc, ou, in);
                  builder.create<vector::StoreOp>(loc, resVec, output,
                                                  ValueRange{ivs4[0], ivs4[1]});
                  builder.create<scf::YieldOp>(loc);
                });
          }

      );
    } else if (inElemTy.isInteger(bitWidth)) {
      buildAffineLoopNest(
          rewriter, loc, lowerbounds4, upperbounds4, steps4,
          [&](OpBuilder &builder, Location loc, ValueRange ivs4) {
            Value pseudoCol =
                builder.create<arith::AddIOp>(loc, ivs4[1], strideVal);
            Value pseudoCol1 =
                builder.create<arith::SubIOp>(loc, pseudoCol, c1);
            Value cond = builder.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::sgt, pseudoCol1, outputCol);
            builder.create<scf::IfOp>(
                loc, cond,
                [&](OpBuilder &builder, Location loc) {
                  Value res =
                      builder.create<arith::SubIOp>(loc, pseudoCol1, outputCol);
                  Value maskVal =
                      builder.create<arith::SubIOp>(loc, strideVal, res);
                  Value maskVec = builder.create<vector::CreateMaskOp>(
                      loc, vectorMaskTy, maskVal);
                  Value ou = builder.create<vector::MaskedLoadOp>(
                      loc, vectorTy32, output2, ValueRange{ivs4[0], ivs4[1]},
                      maskVec, zeroPaddingVec);
                  Value in = builder.create<vector::MaskedLoadOp>(
                      loc, vectorTy32, input, ValueRange{ivs4[0], ivs4[1]},
                      maskVec, zeroPaddingVec);
                  Value resVec = builder.create<arith::SubIOp>(loc, ou, in);
                  builder.create<vector::MaskedStoreOp>(
                      loc, output, ValueRange{ivs4[0], ivs4[1]}, maskVec,
                      resVec);
                  builder.create<scf::YieldOp>(loc);
                },
                [&](OpBuilder &builder, Location loc) {
                  Value ou = builder.create<vector::LoadOp>(
                      loc, vectorTy32, output2, ValueRange{ivs4[0], ivs4[1]});
                  Value in = builder.create<vector::LoadOp>(
                      loc, vectorTy32, input, ValueRange{ivs4[0], ivs4[1]});
                  Value resVec = builder.create<arith::SubIOp>(loc, ou, in);
                  builder.create<vector::StoreOp>(loc, resVec, output,
                                                  ValueRange{ivs4[0], ivs4[1]});
                  builder.create<scf::YieldOp>(loc);
                });
          });
    }
    // Remove the origin bottomhat operation.
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;
};

class DIPMorphGrad2DOpLowering : public OpRewritePattern<dip::MorphGrad2DOp> {
public:
  using OpRewritePattern<dip::MorphGrad2DOp>::OpRewritePattern;

  explicit DIPMorphGrad2DOpLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
  }

  LogicalResult matchAndRewrite(dip::MorphGrad2DOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto *ctx = op->getContext();

    // Register operand values.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    Value output1 = op->getOperand(3);
    Value output2 = op->getOperand(4);
    Value input1 = op->getOperand(5);
    Value copymemref = op->getOperand(6);
    Value copymemref1 = op->getOperand(7);
    Value centerX = op->getOperand(8);
    Value centerY = op->getOperand(9);
    Value iterations = op->getOperand(10);
    Value constantValue = op->getOperand(11);
    dip::BoundaryOption boundaryOptionAttr = op.getBoundaryOption();
    Value strideVal = rewriter.create<arith::ConstantIndexOp>(loc, stride);

    auto inElemTy = input.getType().cast<MemRefType>().getElementType();
    auto bitWidth = inElemTy.getIntOrFloatBitWidth();
    dip::DIP_ERROR error = dip::checkDIPCommonTypes<dip::MorphGrad2DOp>(
        op, {input, kernel, output, output1, output2, input1, copymemref,
             copymemref1, constantValue});

    if (error == dip::DIP_ERROR::INCONSISTENT_TYPES) {
      return op->emitOpError()
             << "input, kernel, output, output1, output2, input1, copymemref, "
                "copymemref1 and constant must "
                "have the same element type";
    } else if (error == dip::DIP_ERROR::UNSUPPORTED_TYPE) {
      return op->emitOpError() << "supports only f32, f64 and integer types. "
                               << inElemTy << "is passed";
    }
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    rewriter.create<memref::CopyOp>(loc, input, input1);

    rewriter.create<AffineForOp>(
        loc, ValueRange{c0}, rewriter.getDimIdentityMap(),
        ValueRange{iterations}, rewriter.getDimIdentityMap(), /*Step=*/1,
        std::nullopt,
        [&](OpBuilder &builder, Location nestedLoc, Value iv,
            ValueRange itrArgs) {
          Value cond = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sge, iv, c1);
          builder.create<scf::IfOp>(
              loc, cond, [&](OpBuilder &builder, Location loc) {
                builder.create<memref::CopyOp>(loc, output1, input);
                builder.create<scf::YieldOp>(loc);
              });
          builder.create<memref::CopyOp>(loc, copymemref, output1);
          traverseImagewBoundaryExtrapolation(
              rewriter, loc, ctx, input, kernel, output1, centerX, centerY,
              constantValue, strideVal, inElemTy, boundaryOptionAttr, stride,
              dip::DIP_OP::DILATION_2D);
          builder.create<AffineYieldOp>(loc);
        });

    rewriter.create<AffineForOp>(
        loc, ValueRange{c0}, rewriter.getDimIdentityMap(),
        ValueRange{iterations}, rewriter.getDimIdentityMap(), /*Step=*/1,
        std::nullopt,
        [&](OpBuilder &builder, Location nestedLoc, Value iv,
            ValueRange itrArgs) {
          Value cond = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sge, iv, c1);
          builder.create<scf::IfOp>(
              loc, cond, [&](OpBuilder &builder, Location loc) {
                builder.create<memref::CopyOp>(loc, output2, input1);
                builder.create<scf::YieldOp>(loc);
              });
          builder.create<memref::CopyOp>(loc, copymemref1, output2);
          traverseImagewBoundaryExtrapolation(
              rewriter, loc, ctx, input1, kernel, output2, centerX, centerY,
              constantValue, strideVal, inElemTy, boundaryOptionAttr, stride,
              dip::DIP_OP::EROSION_2D);
          builder.create<AffineYieldOp>(loc);
        });

    Value outputRow = rewriter.create<memref::DimOp>(loc, output, c0);
    Value outputCol = rewriter.create<memref::DimOp>(loc, output, c1);

    SmallVector<Value, 8> lowerbounds4(2, c0);
    SmallVector<Value, 8> upperbounds4{outputRow, outputCol};
    SmallVector<int64_t, 8> steps4{1, stride};

    VectorType vectorTy32 = VectorType::get({stride}, inElemTy);
    IntegerType i1 = IntegerType::get(ctx, 1);
    VectorType vectorMaskTy = VectorType::get({stride}, i1);
    Value zeroPaddingElem =
        dip::insertZeroConstantOp(ctx, rewriter, loc, inElemTy);
    Value zeroPaddingVec =
        rewriter.create<vector::BroadcastOp>(loc, vectorTy32, zeroPaddingElem);

    if (inElemTy.isF32() || inElemTy.isF64()) {
      buildAffineLoopNest(
          rewriter, loc, lowerbounds4, upperbounds4, steps4,
          [&](OpBuilder &builder, Location loc, ValueRange ivs4) {
            Value pseudoCol =
                builder.create<arith::AddIOp>(loc, ivs4[1], strideVal);
            Value pseudoCol1 =
                builder.create<arith::SubIOp>(loc, pseudoCol, c1);
            Value cond = builder.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::sgt, pseudoCol1, outputCol);
            builder.create<scf::IfOp>(
                loc, cond,
                [&](OpBuilder &builder, Location loc) {
                  Value res =
                      builder.create<arith::SubIOp>(loc, pseudoCol1, outputCol);
                  Value maskVal =
                      builder.create<arith::SubIOp>(loc, strideVal, res);
                  Value maskVec = builder.create<vector::CreateMaskOp>(
                      loc, vectorMaskTy, maskVal);
                  Value ou1 = builder.create<vector::MaskedLoadOp>(
                      loc, vectorTy32, output1, ValueRange{ivs4[0], ivs4[1]},
                      maskVec, zeroPaddingVec);
                  Value ou2 = builder.create<vector::MaskedLoadOp>(
                      loc, vectorTy32, output2, ValueRange{ivs4[0], ivs4[1]},
                      maskVec, zeroPaddingVec);
                  Value resVec = builder.create<arith::SubFOp>(loc, ou1, ou2);
                  builder.create<vector::MaskedStoreOp>(
                      loc, output, ValueRange{ivs4[0], ivs4[1]}, maskVec,
                      resVec);
                  builder.create<scf::YieldOp>(loc);
                },
                [&](OpBuilder &builder, Location loc) {
                  Value ou1 = builder.create<vector::LoadOp>(
                      loc, vectorTy32, output1, ValueRange{ivs4[0], ivs4[1]});
                  Value ou2 = builder.create<vector::LoadOp>(
                      loc, vectorTy32, output2, ValueRange{ivs4[0], ivs4[1]});
                  Value resVec = builder.create<arith::SubFOp>(loc, ou1, ou2);
                  builder.create<vector::StoreOp>(loc, resVec, output,
                                                  ValueRange{ivs4[0], ivs4[1]});
                  builder.create<scf::YieldOp>(loc);
                });
          }

      );
    } else if (inElemTy.isInteger(bitWidth)) {
      buildAffineLoopNest(
          rewriter, loc, lowerbounds4, upperbounds4, steps4,
          [&](OpBuilder &builder, Location loc, ValueRange ivs4) {
            Value pseudoCol =
                builder.create<arith::AddIOp>(loc, ivs4[1], strideVal);
            Value pseudoCol1 =
                builder.create<arith::SubIOp>(loc, pseudoCol, c1);
            Value cond = builder.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::sgt, pseudoCol1, outputCol);
            builder.create<scf::IfOp>(
                loc, cond,
                [&](OpBuilder &builder, Location loc) {
                  Value res =
                      builder.create<arith::SubIOp>(loc, pseudoCol1, outputCol);
                  Value maskVal =
                      builder.create<arith::SubIOp>(loc, strideVal, res);
                  Value maskVec = builder.create<vector::CreateMaskOp>(
                      loc, vectorMaskTy, maskVal);
                  Value ou1 = builder.create<vector::MaskedLoadOp>(
                      loc, vectorTy32, output1, ValueRange{ivs4[0], ivs4[1]},
                      maskVec, zeroPaddingVec);
                  Value ou2 = builder.create<vector::MaskedLoadOp>(
                      loc, vectorTy32, output2, ValueRange{ivs4[0], ivs4[1]},
                      maskVec, zeroPaddingVec);
                  Value resVec = builder.create<arith::SubFOp>(loc, ou1, ou2);
                  builder.create<vector::MaskedStoreOp>(
                      loc, output, ValueRange{ivs4[0], ivs4[1]}, maskVec,
                      resVec);
                },
                [&](OpBuilder &builder, Location loc) {
                  Value ou1 = builder.create<vector::LoadOp>(
                      loc, vectorTy32, output1, ValueRange{ivs4[0], ivs4[1]});
                  Value ou2 = builder.create<vector::LoadOp>(
                      loc, vectorTy32, output2, ValueRange{ivs4[0], ivs4[1]});
                  Value resVec = builder.create<arith::SubFOp>(loc, ou1, ou2);
                  builder.create<vector::StoreOp>(loc, resVec, output,
                                                  ValueRange{ivs4[0], ivs4[1]});
                });
          });
    }

    // Remove the origin morphological gradient operation.
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
  patterns.add<DIPErosion2DOpLowering>(patterns.getContext(), stride);
  patterns.add<DIPDilation2DOpLowering>(patterns.getContext(), stride);
  patterns.add<DIPOpening2DOpLowering>(patterns.getContext(), stride);
  patterns.add<DIPClosing2DOpLowering>(patterns.getContext(), stride);
  patterns.add<DIPTopHat2DOpLowering>(patterns.getContext(), stride);
  patterns.add<DIPBottomHat2DOpLowering>(patterns.getContext(), stride);
  patterns.add<DIPMorphGrad2DOpLowering>(patterns.getContext(), stride);
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
                memref::MemRefDialect, scf::SCFDialect, vector::VectorDialect,
                AffineDialect, arith::ArithDialect, math::MathDialect>();
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
                         memref::MemRefDialect, vector::VectorDialect,
                         arith::ArithDialect, math::MathDialect>();
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
