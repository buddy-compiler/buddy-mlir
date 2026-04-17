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
    Value bias = memref::AllocOp::create(rewriter, loc, biasType);
    Value fillOpInputValue =
        arith::ConstantOp::create(rewriter, loc, fillOpInsType, fillOpInputAttr);
    linalg::FillOp::create(rewriter, loc, fillOpInputValue, bias);
    // Collapse 3D (1xMxK, 1xKxN, 1xMxN) -> 2D (MxK, KxN, MxN)
    Value aVal = input0;
    Value bVal = input1;
    Value oVal = output0;
    Value biasVal = bias;
    if (needCollapse) {
      SmallVector<SmallVector<int64_t, 2>, 2> reassoc = {{0, 1}, {2}};
      aVal = memref::CollapseShapeOp::create(rewriter, loc, input0, reassoc);
      bVal = memref::CollapseShapeOp::create(rewriter, loc, input1, reassoc);
      oVal = memref::CollapseShapeOp::create(rewriter, loc, output0, reassoc);
      biasVal = memref::CollapseShapeOp::create(rewriter, loc, bias, reassoc);
    }

    rewriter.replaceOpWithNewOp<gemmini::TileMatMulOp>(
        matMulOp, aVal, bVal, oVal, biasVal, /*aScaleFactor = */ scale1,
        /*bScaleFactor = */ scale1, /*dScaleFactor = */ scale1, /*act = */ 0,
        /*accScale = */ scale1, /*bertScale = */ scale0);
    memref::DeallocOp::create(rewriter, loc, bias);
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
    Value kernelMat = memref::AllocOp::create(rewriter, loc, kernelMatType);
    SmallVector<int64_t> outputMatShape = {
        outputShape[0] * outputShape[1] * outputShape[2], outputShape[3]};
    MemRefType outputMatType = MemRefType::get(outputMatShape, outputElemType);
    Value outputMat = memref::AllocOp::create(rewriter, loc, outputMatType);
    MemRefType biasType =
        MemRefType::get(outputShape[3], rewriter.getI32Type());
    if (accType == "f32")
      biasType = MemRefType::get(outputShape[3], rewriter.getF32Type());
    Value bias = memref::AllocOp::create(rewriter, loc, biasType);
    TypedAttr attr = rewriter.getI32IntegerAttr(0);
    if (accType == "f32")
      attr = rewriter.getF32FloatAttr(0);
    Value constant0 = arith::ConstantOp::create(rewriter, loc, attr);
    SmallVector<Value, 1> inputs = {constant0};
    SmallVector<Value, 1> outputs = {bias};
    linalg::FillOp::create(rewriter, loc, inputs, outputs);
    Operation *loopOp = nullptr;
    SmallVector<Value, 4> loopIvs;
    for (size_t i = 0; i != kernelShape.size(); i++) {
      Value lowerBound = arith::ConstantIndexOp::create(rewriter, loc, 0);
      Value upperBound =
          arith::ConstantIndexOp::create(rewriter, loc, kernelShape[i]);
      Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);
      auto loop =
          scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step);
      loopIvs.push_back(loop.getInductionVar());
      if (i == 0)
        loopOp = loop.getOperation();
      rewriter.setInsertionPointToStart(loop.getBody());
    }
    Value kernelWidth =
        arith::ConstantIndexOp::create(rewriter, loc, kernelShape[2]);
    Value inChannels =
        arith::ConstantIndexOp::create(rewriter, loc, kernelShape[3]);
    // Conv kernel mapping (f,h,w,c) -> (h*w*c, f)
    // Calculate: h * (W*C) + w * C + c
    Value tmp0 = arith::MulIOp::create(rewriter, loc, loopIvs[1], kernelWidth);
    tmp0 = arith::MulIOp::create(rewriter, loc, tmp0, inChannels);
    Value tmp1 = arith::MulIOp::create(rewriter, loc, loopIvs[2], inChannels);
    tmp0 = arith::AddIOp::create(rewriter, loc, tmp0, tmp1);
    tmp0 = arith::AddIOp::create(rewriter, loc, tmp0, loopIvs[3]);
    Value element = memref::LoadOp::create(rewriter, loc, kernel, loopIvs);
    SmallVector<Value, 2> indices = {tmp0, loopIvs[0]};
    memref::StoreOp::create(rewriter, loc, element, kernelMat, indices);
    rewriter.setInsertionPointAfter(loopOp);
    attr = rewriter.getI64IntegerAttr(outputShape[1]);
    Value outRowDim = arith::ConstantOp::create(rewriter, loc, attr);
    attr = rewriter.getI64IntegerAttr(outputShape[2]);
    Value outColDim = arith::ConstantOp::create(rewriter, loc, attr);
    Value kernelDim = arith::ConstantOp::create(rewriter, 
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(kernelShape[1]));
    gemmini::TileConvOp::create(rewriter, 
        loc, input, kernelMat, bias, outputMat, outRowDim, outColDim, kernelDim,
        llvm::APFloat(float(1.0)), strides, /*inputDilation=*/1,
        /*kernelDilation=*/dilations);
    // After the conv operation is completed, the data in outputMat needs to be
    // transferred into output (2-D to 4-D).
    loopIvs.clear();
    indices.clear();
    for (size_t i = 0; i < outputShape.size(); i++) {
      Value lowerBound = arith::ConstantIndexOp::create(rewriter, loc, 0);
      Value upperBound =
          arith::ConstantIndexOp::create(rewriter, loc, outputShape[i]);
      Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);
      auto loop =
          scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step);
      loopIvs.push_back(loop.getInductionVar());
      if (i == 0)
        loopOp = loop.getOperation();
      rewriter.setInsertionPointToStart(loop.getBody());
    }
    // Map output from 2D (N*H*W, C) back to NHWC (n,h,w,c)
    Value outH = arith::ConstantIndexOp::create(rewriter, loc, outputShape[1]);
    Value outW = arith::ConstantIndexOp::create(rewriter, loc, outputShape[2]);
    // Calculate the row index in the 2D matrix: n * (H*W) + h * W + w
    tmp0 = arith::MulIOp::create(rewriter, loc, loopIvs[0], outH);
    tmp0 = arith::MulIOp::create(rewriter, loc, tmp0, outW);
    tmp1 = arith::MulIOp::create(rewriter, loc, loopIvs[1], outW);
    tmp0 = arith::AddIOp::create(rewriter, loc, tmp0, tmp1);
    tmp0 = arith::AddIOp::create(rewriter, loc, tmp0, loopIvs[2]);
    // The index in the 2D matrix is [n*H*W + h*W + w, c]
    indices.assign({tmp0, loopIvs[3]});
    tmp0 = memref::LoadOp::create(rewriter, loc, outputMat, indices);
    memref::StoreOp::create(rewriter, loc, tmp0, output, loopIvs);
    rewriter.setInsertionPointAfter(loopOp);
    memref::DeallocOp::create(rewriter, loc, kernelMat);
    memref::DeallocOp::create(rewriter, loc, outputMat);
    memref::DeallocOp::create(rewriter, loc, bias);
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
    Value inputMat = memref::AllocOp::create(rewriter, loc, inputMatType);
    Value weightsMat = memref::AllocOp::create(rewriter, loc, weightsMatType);
    MemRefType biasType =
        MemRefType::get(weightsShape[0], rewriter.getI32Type());
    if (accType == "f32")
      biasType = MemRefType::get(weightsShape[0], rewriter.getF32Type());
    SmallVector<int64_t, 2> outputMatShape = {
        inputShape[0] * outputShape[2] * outputShape[3], outputShape[1]};
    MemRefType outputMatType = MemRefType::get(outputMatShape, outputElemType);
    Value bias = memref::AllocOp::create(rewriter, loc, biasType);
    Value outputMat = memref::AllocOp::create(rewriter, loc, outputMatType);
    TypedAttr outDimAttr = rewriter.getI64IntegerAttr(outputShape[2]);
    Value outDim = arith::ConstantOp::create(rewriter, 
        loc, rewriter.getI64Type(), outDimAttr);
    Value kernelDim =
        arith::ConstantIndexOp::create(rewriter, loc, weightsShape[2]);
    Value inChannels =
        arith::ConstantIndexOp::create(rewriter, loc, inputShape[1]);
    SmallVector<Value, 4> loopIvs0;
    SmallVector<Value, 4> loopIvs1;
    Operation *loopOp = nullptr;
    for (unsigned i = 0, e = inputShape.size(); i != e; i++) {
      Value lowerBound = arith::ConstantIndexOp::create(rewriter, loc, 0);
      Value upperBound =
          arith::ConstantIndexOp::create(rewriter, loc, inputShape[i]);
      Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);
      auto loop =
          scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step);
      loopIvs0.push_back(loop.getInductionVar());
      rewriter.setInsertionPointToStart(loop.getBody());
      if (i == 0)
        loopOp = loop.getOperation();
    }
    loopIvs1.push_back(loopIvs0[0]);
    loopIvs1.push_back(loopIvs0[2]);
    loopIvs1.push_back(loopIvs0[3]);
    loopIvs1.push_back(loopIvs0[1]);
    Value element = memref::LoadOp::create(rewriter, loc, input0, loopIvs0);
    memref::StoreOp::create(rewriter, loc, element, inputMat, loopIvs1);
    rewriter.setInsertionPointAfter(loopOp);
    loopIvs0.clear();
    loopIvs1.clear();
    for (unsigned i = 0, e = weightsShape.size(); i != e; i++) {
      Value lowerBound = arith::ConstantIndexOp::create(rewriter, loc, 0);
      Value upperBound =
          arith::ConstantIndexOp::create(rewriter, loc, weightsShape[i]);
      Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);
      auto loop =
          scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step);
      loopIvs0.push_back(loop.getInductionVar());
      rewriter.setInsertionPointToStart(loop.getBody());
      if (i == 0)
        loopOp = loop.getOperation();
    }
    Value tmp0 =
        arith::MulIOp::create(rewriter, loc, /*krow*/ loopIvs0[2], kernelDim);
    tmp0 = arith::MulIOp::create(rewriter, loc, tmp0, inChannels);
    Value tmp1 =
        arith::MulIOp::create(rewriter, loc, /*kcol*/ loopIvs0[3], inChannels);
    tmp0 = arith::AddIOp::create(rewriter, loc, tmp0, tmp1);
    tmp0 = arith::AddIOp::create(rewriter, loc, tmp0, /*inchannel*/ loopIvs0[1]);
    tmp1 = memref::LoadOp::create(rewriter, loc, input1, loopIvs0);
    SmallVector<Value, 2> valueRange = {tmp0, loopIvs0[0]};
    memref::StoreOp::create(rewriter, loc, tmp1, weightsMat, valueRange);
    rewriter.setInsertionPointAfter(loopOp);
    kernelDim = arith::ConstantOp::create(rewriter, 
        loc, rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(weightsShape[2]));
    gemmini::TileConvOp::create(rewriter, 
        loc, inputMat, weightsMat, bias, outputMat, outDim, outDim, kernelDim,
        llvm::APFloat(float(1.0)), strides, /*inputDilation=*/1,
        /*kernelDilation=*/dilations);
    rewriter.eraseOp(convOp);
    loopIvs0.clear();
    loopIvs1.clear();
    for (unsigned i = 0, e = outputShape.size(); i != e; i++) {
      Value lowerBound = arith::ConstantIndexOp::create(rewriter, loc, 0);
      Value upperBound =
          arith::ConstantIndexOp::create(rewriter, loc, outputShape[i]);
      Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);
      auto loop =
          scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step);
      loopIvs0.push_back(loop.getInductionVar());
      rewriter.setInsertionPointToStart(loop.getBody());
      if (i == 0)
        loopOp = loop.getOperation();
    }
    outDim = arith::ConstantIndexOp::create(rewriter, loc, outputShape[2]);
    tmp0 = arith::MulIOp::create(rewriter, loc, loopIvs0[0], outDim);
    tmp0 = arith::MulIOp::create(rewriter, loc, tmp0, outDim);
    tmp1 = arith::MulIOp::create(rewriter, loc, loopIvs0[2], outDim);
    tmp0 = arith::AddIOp::create(rewriter, loc, tmp0, tmp1);
    tmp0 = arith::AddIOp::create(rewriter, loc, tmp0, loopIvs0[3]);
    loopIvs1.push_back(tmp0);
    loopIvs1.push_back(loopIvs0[1]);
    tmp1 = memref::LoadOp::create(rewriter, loc, outputMat, loopIvs1);
    memref::StoreOp::create(rewriter, loc, tmp1, output, loopIvs0);
    rewriter.setInsertionPointAfter(loopOp);
    memref::DeallocOp::create(rewriter, loc, inputMat);
    memref::DeallocOp::create(rewriter, loc, weightsMat);
    memref::DeallocOp::create(rewriter, loc, outputMat);
    memref::DeallocOp::create(rewriter, loc, bias);
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
    Value kernelMat = memref::AllocOp::create(rewriter, loc, kernelMatType);
    memRefShape.assign(
        {outputShape[0] * outputShape[1] * outputShape[2], outputShape[3]});
    MemRefType outputMatType = MemRefType::get(memRefShape, outputElemType);
    Value outputMat = memref::AllocOp::create(rewriter, loc, outputMatType);
    memRefShape.assign({outputShape[3]});
    MemRefType biasType = MemRefType::get(memRefShape, rewriter.getI32Type());
    if (accType == "f32")
      biasType = MemRefType::get(memRefShape, rewriter.getF32Type());
    Value bias = memref::AllocOp::create(rewriter, loc, biasType);
    TypedAttr attr = rewriter.getI32IntegerAttr(0);
    if (accType == "f32")
      attr = rewriter.getF32FloatAttr(0);
    Value constant0 = arith::ConstantOp::create(rewriter, loc, attr);
    SmallVector<Value, 1> inputs = {constant0};
    SmallVector<Value, 1> outputs = {bias};
    linalg::FillOp::create(rewriter, loc, inputs, outputs);
    // Transferring kernel data to kernelMat.
    Value lowerBound = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Operation *loopOp = nullptr;
    SmallVector<Value, 4> loopIvs;
    for (size_t i = 0; i != kernelShape.size(); i++) {
      Value upperBound =
          arith::ConstantIndexOp::create(rewriter, loc, kernelShape[i]);
      auto loop =
          scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step);
      loopIvs.push_back(loop.getInductionVar());
      if (i == 0)
        loopOp = loop.getOperation();
      rewriter.setInsertionPointToStart(loop.getBody());
    }
    Value kernelDim =
        arith::ConstantIndexOp::create(rewriter, loc, kernelShape[1]);
    Value inChannels =
        arith::ConstantIndexOp::create(rewriter, loc, kernelShape[2]);
    Value tmp0 = arith::MulIOp::create(rewriter, loc, loopIvs[0], kernelDim);
    tmp0 = arith::MulIOp::create(rewriter, loc, tmp0, inChannels);
    Value tmp1 = arith::MulIOp::create(rewriter, loc, loopIvs[1], inChannels);
    tmp0 = arith::AddIOp::create(rewriter, loc, tmp0, tmp1);
    tmp0 = arith::AddIOp::create(rewriter, loc, tmp0, loopIvs[2]);
    tmp1 = memref::LoadOp::create(rewriter, loc, kernel, loopIvs);
    SmallVector<Value, 2> indices = {tmp0, loopIvs[3]};
    memref::StoreOp::create(rewriter, loc, tmp1, kernelMat, indices);
    rewriter.setInsertionPointAfter(loopOp);
    attr = rewriter.getI64IntegerAttr(outputShape[1]);
    Value outDim = arith::ConstantOp::create(rewriter, loc, attr);
    attr = rewriter.getI64IntegerAttr(kernelShape[1]);
    kernelDim = arith::ConstantOp::create(rewriter, loc, attr);
    gemmini::TileConvOp::create(rewriter, 
        loc, input, kernelMat, bias, outputMat, outDim, outDim, kernelDim,
        llvm::APFloat(float(1.0)), strides, /*inputDilation=*/1,
        /*kernelDilation=*/dilations);
    // after the conv operation is completed, the data in outputmat needs to be
    // transferred into output.
    loopIvs.clear();
    indices.clear();
    for (size_t i = 0; i < outputShape.size(); i++) {
      Value upperBound =
          arith::ConstantIndexOp::create(rewriter, loc, outputShape[i]);
      auto loop =
          scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step);
      loopIvs.push_back(loop.getInductionVar());
      if (i == 0)
        loopOp = loop.getOperation();
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // Because outputRow is equal to outputCol,here you only need to use
    // outputRow.
    Value row = arith::ConstantIndexOp::create(rewriter, loc, outputShape[1]);
    tmp0 = arith::MulIOp::create(rewriter, loc, loopIvs[0], row);
    tmp0 = arith::MulIOp::create(rewriter, loc, tmp0, row);
    tmp1 = arith::MulIOp::create(rewriter, loc, row, loopIvs[1]);
    tmp0 = arith::AddIOp::create(rewriter, loc, tmp0, tmp1);
    tmp0 = arith::AddIOp::create(rewriter, loc, tmp0, loopIvs[2]);
    indices.assign({tmp0, loopIvs[3]});
    tmp0 = memref::LoadOp::create(rewriter, loc, outputMat, indices);
    memref::StoreOp::create(rewriter, loc, tmp0, output, loopIvs);
    rewriter.setInsertionPointAfter(loopOp);
    memref::DeallocOp::create(rewriter, loc, kernelMat);
    memref::DeallocOp::create(rewriter, loc, outputMat);
    memref::DeallocOp::create(rewriter, loc, bias);
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
    if (isa<linalg::BatchMatmulTransposeBOp, linalg::BatchMatmulTransposeAOp>(
            batchMatMulOp.getOperation()))
      return failure();
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
      Value subInput0 = memref::SubViewOp::create(rewriter, 
          loc, input0, staticOffsets, staticSizes, staticStrides);
      // If rank is 3 with leading 1, collapse to 2D [M,K]
      if (dyn_cast<MemRefType>(subInput0.getType()).getRank() == 3 &&
          dyn_cast<MemRefType>(subInput0.getType()).getShape()[0] == 1) {
        SmallVector<SmallVector<int64_t, 2>, 2> reassoc = {{0, 1}, {2}};
        subInput0 =
            memref::CollapseShapeOp::create(rewriter, loc, subInput0, reassoc);
      }
      staticSizes.assign({1, input1Shape[1], input1Shape[2]});
      Value subInput1 = memref::SubViewOp::create(rewriter, 
          loc, input1, staticOffsets, staticSizes, staticStrides);
      if (dyn_cast<MemRefType>(subInput1.getType()).getRank() == 3 &&
          dyn_cast<MemRefType>(subInput1.getType()).getShape()[0] == 1) {
        SmallVector<SmallVector<int64_t, 2>, 2> reassoc = {{0, 1}, {2}};
        subInput1 =
            memref::CollapseShapeOp::create(rewriter, loc, subInput1, reassoc);
      }
      staticSizes.assign({1, outputShape[1], outputShape[2]});
      Value subOutput = memref::SubViewOp::create(rewriter, 
          loc, output, staticOffsets, staticSizes, staticStrides);
      if (dyn_cast<MemRefType>(subOutput.getType()).getRank() == 3 &&
          dyn_cast<MemRefType>(subOutput.getType()).getShape()[0] == 1) {
        SmallVector<SmallVector<int64_t, 2>, 2> reassoc = {{0, 1}, {2}};
        subOutput =
            memref::CollapseShapeOp::create(rewriter, loc, subOutput, reassoc);
      }
      SmallVector<Value> inputs = {subInput0, subInput1};
      SmallVector<Value> output = {subOutput};
      linalg::MatmulOp::create(rewriter, batchMatMulOp.getLoc(), inputs, output);
    }
    rewriter.eraseOp(batchMatMulOp.getOperation());
    return success();
  }
};

class BatchMatMulTransposeBLowering : public OpRewritePattern<linalg::BatchMatmulOp> {
public:
  explicit BatchMatMulTransposeBLowering(MLIRContext *context,
                                         std::string accType)
      : OpRewritePattern(context, PatternBenefit(2)), accType(accType) {}
  using OpRewritePattern<linalg::BatchMatmulOp>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(linalg::BatchMatmulOp batchMatMulOp,
                  PatternRewriter &rewriter) const override {
    auto batchMatMulTransBOp =
        dyn_cast<linalg::BatchMatmulTransposeBOp>(batchMatMulOp.getOperation());
    if (!batchMatMulTransBOp)
      return failure();
    Location loc = batchMatMulTransBOp.getLoc();
    auto inputs = batchMatMulTransBOp.getInputs();
    Value input0 = inputs[0];
    Value input1 = inputs[1];
    Value output = batchMatMulTransBOp.getOutputs()[0];
    MemRefType input0Type = dyn_cast<MemRefType>(input0.getType());
    MemRefType input1Type = dyn_cast<MemRefType>(input1.getType());
    MemRefType outputType = dyn_cast<MemRefType>(output.getType());
    if (!input0Type || !input1Type || !outputType)
      return failure();
    ArrayRef<int64_t> input0Shape = input0Type.getShape();
    ArrayRef<int64_t> input1Shape = input1Type.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    Type elemType = input0Type.getElementType();

    // Process each batch
    for (unsigned i = 0; i != input0Shape[0]; i++) {
      SmallVector<int64_t> staticOffsets = {i, 0, 0};
      SmallVector<int64_t> staticSizes = {1, input0Shape[1], input0Shape[2]};
      SmallVector<int64_t> staticStrides = {1, 1, 1};
      Value subInput0 = memref::SubViewOp::create(rewriter, 
          loc, input0, staticOffsets, staticSizes, staticStrides);
      // If rank is 3 with leading 1, collapse to 2D [M,K]
      if (dyn_cast<MemRefType>(subInput0.getType()).getRank() == 3 &&
          dyn_cast<MemRefType>(subInput0.getType()).getShape()[0] == 1) {
        SmallVector<SmallVector<int64_t, 2>, 2> reassoc = {{0, 1}, {2}};
        subInput0 =
            memref::CollapseShapeOp::create(rewriter, loc, subInput0, reassoc);
      }
      // For BatchMatmulTransposeBOp, input1 has shape [batch, N, K]
      // where batch dimension index is 0, N is at index 1, K is at index 2
      staticSizes.assign({1, input1Shape[1], input1Shape[2]});
      Value subInput1 = memref::SubViewOp::create(rewriter, 
          loc, input1, staticOffsets, staticSizes, staticStrides);
      if (dyn_cast<MemRefType>(subInput1.getType()).getRank() == 3 &&
          dyn_cast<MemRefType>(subInput1.getType()).getShape()[0] == 1) {
        SmallVector<SmallVector<int64_t, 2>, 2> reassoc = {{0, 1}, {2}};
        subInput1 =
            memref::CollapseShapeOp::create(rewriter, loc, subInput1, reassoc);
      }
      staticSizes.assign({1, outputShape[1], outputShape[2]});
      Value subOutput = memref::SubViewOp::create(rewriter, 
          loc, output, staticOffsets, staticSizes, staticStrides);
      if (dyn_cast<MemRefType>(subOutput.getType()).getRank() == 3 &&
          dyn_cast<MemRefType>(subOutput.getType()).getShape()[0] == 1) {
        SmallVector<SmallVector<int64_t, 2>, 2> reassoc = {{0, 1}, {2}};
        subOutput =
            memref::CollapseShapeOp::create(rewriter, loc, subOutput, reassoc);
      }

      // Create bias memref
      MemRefType subInput0Type = dyn_cast<MemRefType>(subInput0.getType());
      MemRefType biasType =
          MemRefType::get(subInput0Type.getShape(), rewriter.getI32Type());
      TypedAttr fillOpInputAttr = rewriter.getI32IntegerAttr(0);
      Type fillOpInsType = rewriter.getI32Type();
      if (accType == "f32") {
        biasType =
            MemRefType::get(subInput0Type.getShape(), rewriter.getF32Type());
        fillOpInputAttr = rewriter.getF32FloatAttr(0);
        fillOpInsType = rewriter.getF32Type();
      }
      llvm::APFloat scale1((float)1.0);
      llvm::APFloat scale0((float)0.0);
      Value bias = memref::AllocOp::create(rewriter, loc, biasType);
      Value fillOpInputValue = arith::ConstantOp::create(rewriter, 
          loc, fillOpInsType, fillOpInputAttr);
      linalg::FillOp::create(rewriter, loc, fillOpInputValue, bias);

      // Create TileMatMulOp with bTranspose=true
      gemmini::TileMatMulOp::create(rewriter, 
          loc, subInput0, subInput1, subOutput, bias,
          /*aScaleFactor = */ scale1,
          /*bScaleFactor = */ scale1, /*dScaleFactor = */ scale1,
          /*act = */ 0, /*accScale = */ scale1, /*bertScale = */ scale0,
          /*repeatingBias = */ false, /*aTranspose = */ false,
          /*bTranspose = */ true, /*fullC = */ false, /*lowD = */ false,
          /*weightA = */ 0, /*dataflow = */ 1);

      memref::DeallocOp::create(rewriter, loc, bias);
    }
    rewriter.eraseOp(batchMatMulTransBOp.getOperation());
    return success();
  }

private:
  std::string accType;
};

} // namespace

void populateLowerLinalgToGemminiConversionPatterns(RewritePatternSet &patterns,
                                                    std::string accType) {
  patterns.add<MatmulLowering>(patterns.getContext(), accType);
  patterns.add<Conv2DNhwcFhwcLowering>(patterns.getContext(), accType);
  patterns.add<Conv2DNchwFchwLowering>(patterns.getContext(), accType);
  patterns.add<Conv2DNhwcHwcfLowering>(patterns.getContext(), accType);
  patterns.add<BatchMatMulOpLowering>(patterns.getContext());
  patterns.add<BatchMatMulTransposeBLowering>(patterns.getContext(), accType);
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
