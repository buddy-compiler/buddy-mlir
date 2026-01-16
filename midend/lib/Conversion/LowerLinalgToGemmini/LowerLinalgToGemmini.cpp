//====- LowerLinalgToGemmini.cpp - Linalg Dialect Lowering Pass -----------===//
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
// This file defines Linalg dialect lowering pass.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Gemmini/GemminiDialect.h"
#include "Gemmini/GemminiOps.h"
using namespace mlir;
using namespace buddy;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
class MatmulLowering : public OpRewritePattern<linalg::MatmulOp> {
public:
  explicit MatmulLowering(MLIRContext *context, std::string accType)
      : OpRewritePattern(context), accType(accType) {}
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::MatmulOp matMulOp,
                                PatternRewriter &rewriter) const override {
    auto inputs = matMulOp.getInputs();
    auto ouputs = matMulOp.getOutputs();
    Location loc = matMulOp.getLoc();
    Value input0 = inputs[0];
    Value input1 = inputs[1];
    Value output0 = ouputs[0];
    MemRefType input0Type = dyn_cast<MemRefType>(input0.getType());
    MemRefType input1Type = dyn_cast<MemRefType>(input1.getType());
    MemRefType outputType = dyn_cast<MemRefType>(output0.getType());
    if (!input0Type || !input1Type || !outputType)
      return failure();
    bool needCollapse = false;
    SmallVector<int64_t, 3> aShape;
    SmallVector<int64_t, 3> bShape;
    SmallVector<int64_t, 3> oShape;
    aShape.append(input0Type.getShape().begin(), input0Type.getShape().end());
    bShape.append(input1Type.getShape().begin(), input1Type.getShape().end());
    oShape.append(outputType.getShape().begin(), outputType.getShape().end());
    if (input0Type.getRank() == 3 && input1Type.getRank() == 3 &&
        outputType.getRank() == 3 && aShape[0] == 1 && bShape[0] == 1 &&
        oShape[0] == 1)
      needCollapse = true;
    MemRefType biasType =
        MemRefType::get(input0Type.getShape(), rewriter.getI32Type());
    TypedAttr fillOpInputAttr = rewriter.getI32IntegerAttr(0);
    Type fillOpInsType = rewriter.getI32Type();
    if (accType == "f32") {
      biasType = MemRefType::get(input0Type.getShape(), rewriter.getF32Type());
      fillOpInputAttr = rewriter.getF32FloatAttr(0);
      fillOpInsType = rewriter.getF32Type();
    }
    llvm::APFloat scale1((float)1.0);
    llvm::APFloat scale0((float)0.0);
    Value bias = rewriter.create<memref::AllocOp>(loc, biasType);
    Value fillOpInputValue =
        rewriter.create<arith::ConstantOp>(loc, fillOpInsType, fillOpInputAttr);
    rewriter.create<linalg::FillOp>(loc, fillOpInputValue, bias);
    // Collapse 3D (1xMxK, 1xKxN, 1xMxN) -> 2D (MxK, KxN, MxN)
    Value aVal = input0;
    Value bVal = input1;
    Value oVal = output0;
    Value biasVal = bias;
    if (needCollapse) {
      SmallVector<SmallVector<int64_t, 2>, 2> reassoc = {{0, 1}, {2}};
      aVal = rewriter.create<memref::CollapseShapeOp>(loc, input0, reassoc);
      bVal = rewriter.create<memref::CollapseShapeOp>(loc, input1, reassoc);
      oVal = rewriter.create<memref::CollapseShapeOp>(loc, output0, reassoc);
      biasVal = rewriter.create<memref::CollapseShapeOp>(loc, bias, reassoc);
    }

    rewriter.replaceOpWithNewOp<gemmini::TileMatMulOp>(
        matMulOp, aVal, bVal, oVal, biasVal, /*aScaleFactor = */ scale1,
        /*bScaleFactor = */ scale1, /*dScaleFactor = */ scale1, /*act = */ 0,
        /*accScale = */ scale1, /*bertScale = */ scale0);
    rewriter.create<memref::DeallocOp>(loc, bias);
    return success();
  }

private:
  std::string accType;
};

class Conv2DNhwcFhwcLowering
    : public OpRewritePattern<linalg::Conv2DNhwcFhwcOp> {
public:
  explicit Conv2DNhwcFhwcLowering(MLIRContext *context, std::string accType)
      : OpRewritePattern(context), accType(accType) {}
  using OpRewritePattern<linalg::Conv2DNhwcFhwcOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::Conv2DNhwcFhwcOp convOp,
                                PatternRewriter &rewriter) const override {
    Value input = convOp.getInputs()[0];
    Value kernel = convOp.getInputs()[1];
    Value output = convOp.getOutputs()[0];
    Location loc = convOp.getLoc();
    MemRefType inputType = dyn_cast<MemRefType>(input.getType());
    MemRefType kernelType = dyn_cast<MemRefType>(kernel.getType());
    MemRefType outputType = dyn_cast<MemRefType>(output.getType());
    Type kernelElemType = kernelType.getElementType();
    Type outputElemType = outputType.getElementType();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    DenseIntElementsAttr dilationsAttr = convOp.getDilationsAttr();
    DenseIntElementsAttr stridesAttr = convOp.getStridesAttr();
    size_t dilations = 1;
    size_t strides = 1;
    if (dilationsAttr)
      dilations = (*dilationsAttr.begin()).getLimitedValue();
    if (stridesAttr)
      strides = (*stridesAttr.begin()).getLimitedValue();
    if (inputShape[1] != inputShape[2])
      return failure();
    ArrayRef<int64_t> kernelShape = kernelType.getShape();
    if (kernelShape[1] != kernelShape[2])
      return failure();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    // Create kernelMat(hwc, f) and outputMat(nhw, c).
    SmallVector<int64_t> kernelMatShape = {
        kernelShape[1] * kernelShape[2] * kernelShape[3], kernelShape[0]};
    MemRefType kernelMatType = MemRefType::get(kernelMatShape, kernelElemType);
    Value kernelMat = rewriter.create<memref::AllocOp>(loc, kernelMatType);
    SmallVector<int64_t> outputMatShape = {
        outputShape[0] * outputShape[1] * outputShape[2], outputShape[3]};
    MemRefType outputMatType = MemRefType::get(outputMatShape, outputElemType);
    Value outputMat = rewriter.create<memref::AllocOp>(loc, outputMatType);
    MemRefType biasType =
        MemRefType::get(outputShape[3], rewriter.getI32Type());
    if (accType == "f32")
      biasType = MemRefType::get(outputShape[3], rewriter.getF32Type());
    Value bias = rewriter.create<memref::AllocOp>(loc, biasType);
    TypedAttr attr = rewriter.getI32IntegerAttr(0);
    if (accType == "f32")
      attr = rewriter.getF32FloatAttr(0);
    Value constant0 = rewriter.create<arith::ConstantOp>(loc, attr);
    SmallVector<Value, 1> inputs = {constant0};
    SmallVector<Value, 1> outputs = {bias};
    rewriter.create<linalg::FillOp>(loc, inputs, outputs);
    Operation *loopOp = nullptr;
    SmallVector<Value, 4> loopIvs;
    for (size_t i = 0; i != kernelShape.size(); i++) {
      Value lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value upperBound =
          rewriter.create<arith::ConstantIndexOp>(loc, kernelShape[i]);
      Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto loop =
          rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
      loopIvs.push_back(loop.getInductionVar());
      if (i == 0)
        loopOp = loop.getOperation();
      rewriter.setInsertionPointToStart(loop.getBody());
    }
    Value kernelDim =
        rewriter.create<arith::ConstantIndexOp>(loc, kernelShape[1]);
    Value inChannels =
        rewriter.create<arith::ConstantIndexOp>(loc, kernelShape[3]);
    // Conv kernel mapping (f,h,w,c) -> (h*w*c, f)
    Value tmp0 = rewriter.create<arith::MulIOp>(loc, loopIvs[1], kernelDim);
    tmp0 = rewriter.create<arith::MulIOp>(loc, tmp0, inChannels);
    Value tmp1 = rewriter.create<arith::MulIOp>(loc, loopIvs[2], inChannels);
    tmp0 = rewriter.create<arith::AddIOp>(loc, tmp0, tmp1);
    tmp0 = rewriter.create<arith::AddIOp>(loc, tmp0, loopIvs[3]);
    Value element = rewriter.create<memref::LoadOp>(loc, kernel, loopIvs);
    SmallVector<Value, 2> indices = {tmp0, loopIvs[0]};
    rewriter.create<memref::StoreOp>(loc, element, kernelMat, indices);
    rewriter.setInsertionPointAfter(loopOp);
    attr = rewriter.getI64IntegerAttr(outputShape[1]);
    Value outRowDim = rewriter.create<arith::ConstantOp>(loc, attr);
    attr = rewriter.getI64IntegerAttr(outputShape[2]);
    Value outColDim = rewriter.create<arith::ConstantOp>(loc, attr);
    kernelDim = rewriter.create<arith::ConstantOp>(loc, attr);
    kernelDim = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(kernelShape[1]));
    rewriter.create<gemmini::TileConvOp>(
        loc, input, kernelMat, bias, outputMat, outRowDim, outColDim, kernelDim,
        llvm::APFloat(float(1.0)), strides, dilations);
    // After the conv operation is completed, the data in outputMat needs to be
    // transferred into output (2-D to 4-D).
    loopIvs.clear();
    indices.clear();
    for (size_t i = 0; i < outputShape.size(); i++) {
      Value lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value upperBound =
          rewriter.create<arith::ConstantIndexOp>(loc, outputShape[i]);
      Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto loop =
          rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
      loopIvs.push_back(loop.getInductionVar());
      if (i == 0)
        loopOp = loop.getOperation();
      rewriter.setInsertionPointToStart(loop.getBody());
    }
    // Map output from 2D (N*H*W, C) back to NHWC (n,h,w,c)
    Value outH = rewriter.create<arith::ConstantIndexOp>(loc, outputShape[1]);
    Value outW = rewriter.create<arith::ConstantIndexOp>(loc, outputShape[2]);
    // Calculate the row index in the 2D matrix: n * (H*W) + h * W + w
    tmp0 = rewriter.create<arith::MulIOp>(loc, loopIvs[0], outH);
    tmp0 = rewriter.create<arith::MulIOp>(loc, tmp0, outW);
    tmp1 = rewriter.create<arith::MulIOp>(loc, loopIvs[1], outW);
    tmp0 = rewriter.create<arith::AddIOp>(loc, tmp0, tmp1);
    tmp0 = rewriter.create<arith::AddIOp>(loc, tmp0, loopIvs[2]);
    // The index in the 2D matrix is [n*H*W + h*W + w, c]
    indices.assign({tmp0, loopIvs[3]});
    tmp0 = rewriter.create<memref::LoadOp>(loc, outputMat, indices);
    rewriter.create<memref::StoreOp>(loc, tmp0, output, loopIvs);
    rewriter.setInsertionPointAfter(loopOp);
    rewriter.create<memref::DeallocOp>(loc, kernelMat);
    rewriter.create<memref::DeallocOp>(loc, outputMat);
    rewriter.create<memref::DeallocOp>(loc, bias);
    rewriter.eraseOp(convOp);
    return success();
  }

private:
  std::string accType;
};

class Conv2DNchwFchwLowering
    : public OpRewritePattern<linalg::Conv2DNchwFchwOp> {
public:
  explicit Conv2DNchwFchwLowering(MLIRContext *context, std::string accType)
      : OpRewritePattern(context), accType(accType) {}
  using OpRewritePattern<linalg::Conv2DNchwFchwOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp convOp,
                                PatternRewriter &rewriter) const override {
    auto inputs = convOp.getInputs();
    Value input0 = inputs[0];
    Value input1 = inputs[1];
    Value output = convOp.getOutputs()[0];
    Location loc = convOp.getLoc();
    MemRefType inputType = dyn_cast<MemRefType>(input0.getType());
    MemRefType weightsType = dyn_cast<MemRefType>(input1.getType());
    MemRefType outputType = dyn_cast<MemRefType>(output.getType());
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    ArrayRef<int64_t> weightsShape = weightsType.getShape();
    Type inputElemType = inputType.getElementType();
    Type weightsElemType = weightsType.getElementType();
    Type outputElemType = outputType.getElementType();
    DenseIntElementsAttr dilationsAttr = convOp.getDilationsAttr();
    DenseIntElementsAttr stridesAttr = convOp.getStrides();
    size_t dilations = 1;
    size_t strides = 1;
    // Gemmini only support 1-D dilations.
    if (dilationsAttr)
      dilations = (*dilationsAttr.begin()).getLimitedValue();
    if (stridesAttr)
      strides = (*stridesAttr.begin()).getLimitedValue();
    SmallVector<int64_t> inputMatShape = {inputShape[0], inputShape[2],
                                          inputShape[3], inputShape[1]};
    SmallVector<int64_t> weightsMatShape = {
        weightsShape[1] * weightsShape[2] * weightsShape[3], weightsShape[0]};
    MemRefType inputMatType = MemRefType::get(inputMatShape, inputElemType);
    MemRefType weightsMatType =
        MemRefType::get(weightsMatShape, weightsElemType);
    Value inputMat = rewriter.create<memref::AllocOp>(loc, inputMatType);
    Value weightsMat = rewriter.create<memref::AllocOp>(loc, weightsMatType);
    MemRefType biasType =
        MemRefType::get(weightsShape[0], rewriter.getI32Type());
    if (accType == "f32")
      biasType = MemRefType::get(weightsShape[0], rewriter.getF32Type());
    SmallVector<int64_t, 2> outputMatShape = {
        inputShape[0] * outputShape[2] * outputShape[3], outputShape[1]};
    MemRefType outputMatType = MemRefType::get(outputMatShape, outputElemType);
    Value bias = rewriter.create<memref::AllocOp>(loc, biasType);
    Value outputMat = rewriter.create<memref::AllocOp>(loc, outputMatType);
    TypedAttr outDimAttr = rewriter.getI64IntegerAttr(outputShape[2]);
    Value outDim = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(), outDimAttr);
    Value kernelDim =
        rewriter.create<arith::ConstantIndexOp>(loc, weightsShape[2]);
    Value inChannels =
        rewriter.create<arith::ConstantIndexOp>(loc, inputShape[1]);
    SmallVector<Value, 4> loopIvs0;
    SmallVector<Value, 4> loopIvs1;
    Operation *loopOp = nullptr;
    for (unsigned i = 0, e = inputShape.size(); i != e; i++) {
      Value lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value upperBound =
          rewriter.create<arith::ConstantIndexOp>(loc, inputShape[i]);
      Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto loop =
          rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
      loopIvs0.push_back(loop.getInductionVar());
      rewriter.setInsertionPointToStart(loop.getBody());
      if (i == 0)
        loopOp = loop.getOperation();
    }
    loopIvs1.push_back(loopIvs0[0]);
    loopIvs1.push_back(loopIvs0[2]);
    loopIvs1.push_back(loopIvs0[3]);
    loopIvs1.push_back(loopIvs0[1]);
    Value element = rewriter.create<memref::LoadOp>(loc, input0, loopIvs0);
    rewriter.create<memref::StoreOp>(loc, element, inputMat, loopIvs1);
    rewriter.setInsertionPointAfter(loopOp);
    loopIvs0.clear();
    loopIvs1.clear();
    for (unsigned i = 0, e = weightsShape.size(); i != e; i++) {
      Value lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value upperBound =
          rewriter.create<arith::ConstantIndexOp>(loc, weightsShape[i]);
      Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto loop =
          rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
      loopIvs0.push_back(loop.getInductionVar());
      rewriter.setInsertionPointToStart(loop.getBody());
      if (i == 0)
        loopOp = loop.getOperation();
    }
    Value tmp0 =
        rewriter.create<arith::MulIOp>(loc, /*krow*/ loopIvs0[2], kernelDim);
    tmp0 = rewriter.create<arith::MulIOp>(loc, tmp0, inChannels);
    Value tmp1 =
        rewriter.create<arith::MulIOp>(loc, /*kcol*/ loopIvs0[3], inChannels);
    tmp0 = rewriter.create<arith::AddIOp>(loc, tmp0, tmp1);
    tmp0 = rewriter.create<arith::AddIOp>(loc, tmp0, /*inchannel*/ loopIvs0[1]);
    tmp1 = rewriter.create<memref::LoadOp>(loc, input1, loopIvs0);
    SmallVector<Value, 2> valueRange = {tmp0, loopIvs0[0]};
    rewriter.create<memref::StoreOp>(loc, tmp1, weightsMat, valueRange);
    rewriter.setInsertionPointAfter(loopOp);
    kernelDim = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(weightsShape[2]));
    rewriter.create<gemmini::TileConvOp>(
        loc, inputMat, weightsMat, bias, outputMat, outDim, outDim, kernelDim,
        llvm::APFloat(float(1.0)), strides, dilations);
    rewriter.eraseOp(convOp);
    loopIvs0.clear();
    loopIvs1.clear();
    for (unsigned i = 0, e = outputShape.size(); i != e; i++) {
      Value lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value upperBound =
          rewriter.create<arith::ConstantIndexOp>(loc, outputShape[i]);
      Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto loop =
          rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
      loopIvs0.push_back(loop.getInductionVar());
      rewriter.setInsertionPointToStart(loop.getBody());
      if (i == 0)
        loopOp = loop.getOperation();
    }
    outDim = rewriter.create<arith::ConstantIndexOp>(loc, outputShape[2]);
    tmp0 = rewriter.create<arith::MulIOp>(loc, loopIvs0[0], outDim);
    tmp0 = rewriter.create<arith::MulIOp>(loc, tmp0, outDim);
    tmp1 = rewriter.create<arith::MulIOp>(loc, loopIvs0[2], outDim);
    tmp0 = rewriter.create<arith::AddIOp>(loc, tmp0, tmp1);
    tmp0 = rewriter.create<arith::AddIOp>(loc, tmp0, loopIvs0[3]);
    loopIvs1.push_back(tmp0);
    loopIvs1.push_back(loopIvs0[1]);
    tmp1 = rewriter.create<memref::LoadOp>(loc, outputMat, loopIvs1);
    rewriter.create<memref::StoreOp>(loc, tmp1, output, loopIvs0);
    rewriter.setInsertionPointAfter(loopOp);
    rewriter.create<memref::DeallocOp>(loc, inputMat);
    rewriter.create<memref::DeallocOp>(loc, weightsMat);
    rewriter.create<memref::DeallocOp>(loc, outputMat);
    rewriter.create<memref::DeallocOp>(loc, bias);
    return success();
  }

private:
  std::string accType;
};

class Conv2DNhwcHwcfLowering
    : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
public:
  explicit Conv2DNhwcHwcfLowering(MLIRContext *context, std::string accType)
      : OpRewritePattern(context), accType(accType) {}
  using OpRewritePattern<linalg::Conv2DNhwcHwcfOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {
    Value input = convOp.getInputs()[0];
    Value kernel = convOp.getInputs()[1];
    Value output = convOp.getOutputs()[0];
    Location loc = convOp.getLoc();
    MemRefType inputType = dyn_cast<MemRefType>(input.getType());
    MemRefType kernelType = dyn_cast<MemRefType>(kernel.getType());
    MemRefType outputType = dyn_cast<MemRefType>(output.getType());
    Type kernelElemType = kernelType.getElementType();
    Type outputElemType = outputType.getElementType();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    DenseIntElementsAttr dilationsAttr = convOp.getDilationsAttr();
    DenseIntElementsAttr stridesAttr = convOp.getStrides();
    size_t dilations = 1;
    size_t strides = 1;
    // Gemmini only support 1-D dilations.
    if (dilationsAttr)
      dilations = (*dilationsAttr.begin()).getLimitedValue();
    if (stridesAttr)
      strides = (*stridesAttr.begin()).getLimitedValue();

    if (inputShape[1] != inputShape[2])
      return failure();
    ArrayRef<int64_t> kernelShape = kernelType.getShape();
    if (kernelShape[0] != kernelShape[1])
      return failure();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    // Create kernelMat and outputMat.
    SmallVector<int64_t> memRefShape = {
        kernelShape[0] * kernelShape[1] * kernelShape[2], kernelShape[3]};
    MemRefType kernelMatType = MemRefType::get(memRefShape, kernelElemType);
    Value kernelMat = rewriter.create<memref::AllocOp>(loc, kernelMatType);
    memRefShape.assign(
        {outputShape[0] * outputShape[1] * outputShape[2], outputShape[3]});
    MemRefType outputMatType = MemRefType::get(memRefShape, outputElemType);
    Value outputMat = rewriter.create<memref::AllocOp>(loc, outputMatType);
    memRefShape.assign({outputShape[3]});
    MemRefType biasType = MemRefType::get(memRefShape, rewriter.getI32Type());
    if (accType == "f32")
      biasType = MemRefType::get(memRefShape, rewriter.getF32Type());
    Value bias = rewriter.create<memref::AllocOp>(loc, biasType);
    TypedAttr attr = rewriter.getI32IntegerAttr(0);
    if (accType == "f32")
      attr = rewriter.getF32FloatAttr(0);
    Value constant0 = rewriter.create<arith::ConstantOp>(loc, attr);
    SmallVector<Value, 1> inputs = {constant0};
    SmallVector<Value, 1> outputs = {bias};
    rewriter.create<linalg::FillOp>(loc, inputs, outputs);
    // Transferring kernel data to kernelMat.
    Value lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Operation *loopOp = nullptr;
    SmallVector<Value, 4> loopIvs;
    for (size_t i = 0; i != kernelShape.size(); i++) {
      Value upperBound =
          rewriter.create<arith::ConstantIndexOp>(loc, kernelShape[i]);
      auto loop =
          rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
      loopIvs.push_back(loop.getInductionVar());
      if (i == 0)
        loopOp = loop.getOperation();
      rewriter.setInsertionPointToStart(loop.getBody());
    }
    Value kernelDim =
        rewriter.create<arith::ConstantIndexOp>(loc, kernelShape[1]);
    Value inChannels =
        rewriter.create<arith::ConstantIndexOp>(loc, kernelShape[2]);
    Value tmp0 = rewriter.create<arith::MulIOp>(loc, loopIvs[0], kernelDim);
    tmp0 = rewriter.create<arith::MulIOp>(loc, tmp0, inChannels);
    Value tmp1 = rewriter.create<arith::MulIOp>(loc, loopIvs[1], inChannels);
    tmp0 = rewriter.create<arith::AddIOp>(loc, tmp0, tmp1);
    tmp0 = rewriter.create<arith::AddIOp>(loc, tmp0, loopIvs[2]);
    tmp1 = rewriter.create<memref::LoadOp>(loc, kernel, loopIvs);
    SmallVector<Value, 2> indices = {tmp0, loopIvs[3]};
    rewriter.create<memref::StoreOp>(loc, tmp1, kernelMat, indices);
    rewriter.setInsertionPointAfter(loopOp);
    attr = rewriter.getI64IntegerAttr(outputShape[1]);
    Value outDim = rewriter.create<arith::ConstantOp>(loc, attr);
    attr = rewriter.getI64IntegerAttr(kernelShape[1]);
    kernelDim = rewriter.create<arith::ConstantOp>(loc, attr);
    rewriter.create<gemmini::TileConvOp>(
        loc, input, kernelMat, bias, outputMat, outDim, outDim, kernelDim,
        llvm::APFloat(float(1.0)), strides, dilations);
    // after the conv operation is completed, the data in outputmat needs to be
    // transferred into output.
    loopIvs.clear();
    indices.clear();
    for (size_t i = 0; i < outputShape.size(); i++) {
      Value upperBound =
          rewriter.create<arith::ConstantIndexOp>(loc, outputShape[i]);
      auto loop =
          rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
      loopIvs.push_back(loop.getInductionVar());
      if (i == 0)
        loopOp = loop.getOperation();
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // Because outputRow is equal to outputCol,here you only need to use
    // outputRow.
    Value row = rewriter.create<arith::ConstantIndexOp>(loc, outputShape[1]);
    tmp0 = rewriter.create<arith::MulIOp>(loc, loopIvs[0], row);
    tmp0 = rewriter.create<arith::MulIOp>(loc, tmp0, row);
    tmp1 = rewriter.create<arith::MulIOp>(loc, row, loopIvs[1]);
    tmp0 = rewriter.create<arith::AddIOp>(loc, tmp0, tmp1);
    tmp0 = rewriter.create<arith::AddIOp>(loc, tmp0, loopIvs[2]);
    indices.assign({tmp0, loopIvs[3]});
    tmp0 = rewriter.create<memref::LoadOp>(loc, outputMat, indices);
    rewriter.create<memref::StoreOp>(loc, tmp0, output, loopIvs);
    rewriter.setInsertionPointAfter(loopOp);
    rewriter.create<memref::DeallocOp>(loc, kernelMat);
    rewriter.create<memref::DeallocOp>(loc, outputMat);
    rewriter.create<memref::DeallocOp>(loc, bias);
    rewriter.eraseOp(convOp);
    return success();
  }

private:
  std::string accType;
};

class BatchMatMulOpLowering : public OpRewritePattern<linalg::BatchMatmulOp> {
public:
  using OpRewritePattern<linalg::BatchMatmulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::BatchMatmulOp batchMatMulOp,
                                PatternRewriter &rewriter) const override {
    Location loc = batchMatMulOp.getLoc();
    auto inputs = batchMatMulOp.getInputs();
    Value input0 = inputs[0];
    Value input1 = inputs[1];
    Value output = batchMatMulOp.getOutputs()[0];
    MemRefType input0Type = dyn_cast<MemRefType>(input0.getType());
    MemRefType input1Type = dyn_cast<MemRefType>(input1.getType());
    MemRefType outputType = dyn_cast<MemRefType>(output.getType());
    if (!input0Type || !input1Type || !outputType)
      return failure();
    ArrayRef<int64_t> input0Shape = input0Type.getShape();
    ArrayRef<int64_t> input1Shape = input1Type.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    Type elemType = input0Type.getElementType();
    for (unsigned i = 0; i != input0Shape[0]; i++) {
      SmallVector<int64_t> staticOffsets = {i, 0, 0};
      SmallVector<int64_t> staticSizes = {1, input0Shape[1], input0Shape[2]};
      SmallVector<int64_t> staticStrides = {1, 1, 1};
      Value subInput0 = rewriter.create<memref::SubViewOp>(
          loc, input0, staticOffsets, staticSizes, staticStrides);
      // If rank is 3 with leading 1, collapse to 2D [M,K]
      if (dyn_cast<MemRefType>(subInput0.getType()).getRank() == 3 &&
          dyn_cast<MemRefType>(subInput0.getType()).getShape()[0] == 1) {
        SmallVector<SmallVector<int64_t, 2>, 2> reassoc = {{0, 1}, {2}};
        subInput0 =
            rewriter.create<memref::CollapseShapeOp>(loc, subInput0, reassoc);
      }
      staticSizes.assign({1, input1Shape[1], input1Shape[2]});
      Value subInput1 = rewriter.create<memref::SubViewOp>(
          loc, input1, staticOffsets, staticSizes, staticStrides);
      if (dyn_cast<MemRefType>(subInput1.getType()).getRank() == 3 &&
          dyn_cast<MemRefType>(subInput1.getType()).getShape()[0] == 1) {
        SmallVector<SmallVector<int64_t, 2>, 2> reassoc = {{0, 1}, {2}};
        subInput1 =
            rewriter.create<memref::CollapseShapeOp>(loc, subInput1, reassoc);
      }
      staticSizes.assign({1, outputShape[1], outputShape[2]});
      Value subOutput = rewriter.create<memref::SubViewOp>(
          loc, output, staticOffsets, staticSizes, staticStrides);
      if (dyn_cast<MemRefType>(subOutput.getType()).getRank() == 3 &&
          dyn_cast<MemRefType>(subOutput.getType()).getShape()[0] == 1) {
        SmallVector<SmallVector<int64_t, 2>, 2> reassoc = {{0, 1}, {2}};
        subOutput =
            rewriter.create<memref::CollapseShapeOp>(loc, subOutput, reassoc);
      }
      SmallVector<Value> inputs = {subInput0, subInput1};
      SmallVector<Value> output = {subOutput};
      rewriter.create<linalg::MatmulOp>(batchMatMulOp.getLoc(), inputs, output);
    }
    rewriter.eraseOp(batchMatMulOp.getOperation());
    return success();
  }
};

} // namespace

void populateLowerLinalgToGemminiConversionPatterns(RewritePatternSet &patterns,
                                                    std::string accType) {
  patterns.add<MatmulLowering>(patterns.getContext(), accType);
  patterns.add<Conv2DNhwcFhwcLowering>(patterns.getContext(), accType);
  patterns.add<Conv2DNchwFchwLowering>(patterns.getContext(), accType);
  patterns.add<Conv2DNhwcHwcfLowering>(patterns.getContext(), accType);
  patterns.add<BatchMatMulOpLowering>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// LowerLinalgToGemmini
//===----------------------------------------------------------------------===//

namespace {
class LowerLinalgToGemminiPass
    : public PassWrapper<LowerLinalgToGemminiPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLinalgToGemminiPass);
  LowerLinalgToGemminiPass() = default;
  LowerLinalgToGemminiPass(const LowerLinalgToGemminiPass &) {}
  StringRef getArgument() const final { return "convert-linalg-to-gemmini"; }
  StringRef getDescription() const final {
    return "convert linalg dialect to gemmini dialect";
  }
  void runOnOperation() override;
  Option<std::string> accType{*this, "acc_t",
                              llvm::cl::desc("The type of acc_t."),
                              llvm::cl::init("i32")};
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gemmini::GemminiDialect, func::FuncDialect,
                    memref::MemRefDialect, linalg::LinalgDialect,
                    arith::ArithDialect, scf::SCFDialect>();
  }
};
} // namespace

void LowerLinalgToGemminiPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  ConversionTarget target(*context);
  target.addLegalDialect<memref::MemRefDialect, gemmini::GemminiDialect,
                         arith::ArithDialect, scf::SCFDialect>();
  target.addLegalOp<linalg::FillOp, linalg::YieldOp>();
  RewritePatternSet patterns(context);
  populateLowerLinalgToGemminiConversionPatterns(patterns, accType);
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerLinalgToGemminiPass() {
  PassRegistration<LowerLinalgToGemminiPass>();
}
} // namespace buddy
} // namespace mlir
