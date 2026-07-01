//====- LowerLinalgToTile.cpp - Linalg Dialect Lowering Pass -----------===//
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
// This file defines Linalg dialect lowering pass to Tile dialect.
//
//===----------------------------------------------------------------------===//
#include <optional>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Tile/TileDialect.h"
#include "Tile/TileOps.h"
using namespace mlir;
using namespace buddy;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
static std::optional<int64_t> getUniformAttr(DenseIntElementsAttr attr) {
  if (!attr)
    return 1;
  if (attr.empty())
    return std::nullopt;

  int64_t value = (*attr.begin()).getSExtValue();
  for (llvm::APInt element : attr) {
    if (element.getSExtValue() != value)
      return std::nullopt;
  }
  return value;
}

static bool supportsTileConv(MemRefType inType, MemRefType filterType,
                             MemRefType outType) {
  if (inType.getRank() != 4 || filterType.getRank() != 4 ||
      outType.getRank() != 4)
    return false;
  if (!inType.getElementType().isF32() || !filterType.getElementType().isF32())
    return false;
  if (!inType.hasStaticShape() || !filterType.hasStaticShape() ||
      !outType.hasStaticShape())
    return false;

  constexpr int64_t bankWidth = 16;
  constexpr int64_t bankDepth = 4096;
  auto inShape = inType.getShape();
  auto fShape = filterType.getShape();
  auto outShape = outType.getShape();
  int64_t h = inShape[1], w = inShape[2], c = inShape[3];
  int64_t kh = fShape[0], kw = fShape[1], oc = fShape[3];
  int64_t oh = outShape[1], ow = outShape[2];
  if (inShape[0] <= 0 || h <= 0 || w <= 0 || c <= 0 || kh <= 0 || kw <= 0 ||
      oc <= 0 || oh <= 0 || ow <= 0)
    return false;

  int64_t cPad = c;
  while ((h * w * cPad) % bankWidth != 0)
    ++cPad;
  int64_t patchCols = kh * kw * cPad;
  return patchCols > 0 && patchCols <= bankDepth && kh <= 255 &&
         kw * cPad <= 255 && h <= 255 && w * cPad <= 255 && oh <= 255 &&
         ow <= 255 && cPad <= 255;
}

class MatmulLowering : public OpRewritePattern<linalg::MatmulOp> {
public:
  explicit MatmulLowering(MLIRContext *context) : OpRewritePattern(context) {}
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
    Attribute indexingMaps = matMulOp->getAttr("indexing_maps");
    bool isDefaultMatmul =
        linalg::MatmulOp::isDefaultIndexingMaps(indexingMaps);
    bool isTransposeB =
        linalg::MatmulTransposeBOp::isDefaultIndexingMaps(indexingMaps);
    if (!isDefaultMatmul && !isTransposeB)
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
    Value aVal = input0;
    Value bVal = input1;
    Value oVal = output0;
    if (needCollapse) {
      SmallVector<SmallVector<int64_t, 2>, 2> reassoc = {{0, 1}, {2}};
      aVal = memref::CollapseShapeOp::create(rewriter, loc, input0, reassoc);
      bVal = memref::CollapseShapeOp::create(rewriter, loc, input1, reassoc);
      oVal = memref::CollapseShapeOp::create(rewriter, loc, output0, reassoc);
    }

    Value matmulInput1 = bVal;
    if (isTransposeB) {
      auto bValType = dyn_cast<MemRefType>(bVal.getType());
      if (!bValType || bValType.getRank() != 2 || !bValType.hasStaticShape())
        return failure();

      ArrayRef<int64_t> bValShape = bValType.getShape();
      int64_t n = bValShape[0];
      int64_t k = bValShape[1];
      auto transposedType = MemRefType::get({k, n}, bValType.getElementType());
      Value transposed = memref::AllocOp::create(rewriter, loc, transposedType);

      Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
      Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
      Value nUb = arith::ConstantIndexOp::create(rewriter, loc, n);
      Value kUb = arith::ConstantIndexOp::create(rewriter, loc, k);

      auto nLoop = scf::ForOp::create(rewriter, loc, zero, nUb, one);
      rewriter.setInsertionPointToStart(nLoop.getBody());
      Value nIv = nLoop.getInductionVar();
      auto kLoop = scf::ForOp::create(rewriter, loc, zero, kUb, one);
      rewriter.setInsertionPointToStart(kLoop.getBody());
      Value kIv = kLoop.getInductionVar();
      Value value =
          memref::LoadOp::create(rewriter, loc, bVal, ValueRange{nIv, kIv});
      memref::StoreOp::create(rewriter, loc, value, transposed,
                              ValueRange{kIv, nIv});

      rewriter.setInsertionPointAfter(nLoop);
      matmulInput1 = transposed;
    }

    tile::TileMatMulOp::create(rewriter, loc, aVal, matmulInput1, oVal);
    if (isTransposeB)
      memref::DeallocOp::create(rewriter, loc, matmulInput1);
    rewriter.eraseOp(matMulOp);
    return success();
  }

private:
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
    Attribute indexingMaps = batchMatMulOp->getAttr("indexing_maps");
    bool isDefaultBatchMatmul =
        linalg::BatchMatmulOp::isDefaultIndexingMaps(indexingMaps);
    bool isTransposeB =
        linalg::BatchMatmulTransposeBOp::isDefaultIndexingMaps(indexingMaps);
    if (!isDefaultBatchMatmul && !isTransposeB)
      return failure();

    ArrayRef<int64_t> input0Shape = input0Type.getShape();
    ArrayRef<int64_t> input1Shape = input1Type.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    Type elemType = input0Type.getElementType();
    for (unsigned i = 0; i != input0Shape[0]; i++) {
      SmallVector<int64_t> staticOffsets = {i, 0, 0};
      SmallVector<int64_t> staticSizes = {1, input0Shape[1], input0Shape[2]};
      SmallVector<int64_t> staticStrides = {1, 1, 1};
      Value subInput0 = memref::SubViewOp::create(
          rewriter, loc, input0, staticOffsets, staticSizes, staticStrides);
      if (dyn_cast<MemRefType>(subInput0.getType()).getRank() == 3 &&
          dyn_cast<MemRefType>(subInput0.getType()).getShape()[0] == 1) {
        SmallVector<SmallVector<int64_t, 2>, 2> reassoc = {{0, 1}, {2}};
        subInput0 =
            memref::CollapseShapeOp::create(rewriter, loc, subInput0, reassoc);
      }

      staticSizes.assign({1, input1Shape[1], input1Shape[2]});
      Value subInput1 = memref::SubViewOp::create(
          rewriter, loc, input1, staticOffsets, staticSizes, staticStrides);
      if (dyn_cast<MemRefType>(subInput1.getType()).getRank() == 3 &&
          dyn_cast<MemRefType>(subInput1.getType()).getShape()[0] == 1) {
        SmallVector<SmallVector<int64_t, 2>, 2> reassoc = {{0, 1}, {2}};
        subInput1 =
            memref::CollapseShapeOp::create(rewriter, loc, subInput1, reassoc);
      }
      Value matmulInput1 = subInput1;
      if (isTransposeB) {
        auto transposedType =
            MemRefType::get({input1Shape[2], input1Shape[1]}, elemType);
        Value transposed =
            memref::AllocOp::create(rewriter, loc, transposedType);

        Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
        Value nUb =
            arith::ConstantIndexOp::create(rewriter, loc, input1Shape[1]);
        Value kUb =
            arith::ConstantIndexOp::create(rewriter, loc, input1Shape[2]);

        auto nLoop = scf::ForOp::create(rewriter, loc, zero, nUb, one);
        rewriter.setInsertionPointToStart(nLoop.getBody());
        Value nIv = nLoop.getInductionVar();
        auto kLoop = scf::ForOp::create(rewriter, loc, zero, kUb, one);
        rewriter.setInsertionPointToStart(kLoop.getBody());
        Value kIv = kLoop.getInductionVar();
        Value value = memref::LoadOp::create(rewriter, loc, subInput1,
                                             ValueRange{nIv, kIv});
        memref::StoreOp::create(rewriter, loc, value, transposed,
                                ValueRange{kIv, nIv});

        rewriter.setInsertionPointAfter(nLoop);
        matmulInput1 = transposed;
      }

      staticSizes.assign({1, outputShape[1], outputShape[2]});
      Value subOutput = memref::SubViewOp::create(
          rewriter, loc, output, staticOffsets, staticSizes, staticStrides);
      if (dyn_cast<MemRefType>(subOutput.getType()).getRank() == 3 &&
          dyn_cast<MemRefType>(subOutput.getType()).getShape()[0] == 1) {
        SmallVector<SmallVector<int64_t, 2>, 2> reassoc = {{0, 1}, {2}};
        subOutput =
            memref::CollapseShapeOp::create(rewriter, loc, subOutput, reassoc);
      }
      SmallVector<Value> inputs = {subInput0, matmulInput1};
      SmallVector<Value> outputs = {subOutput};
      linalg::MatmulOp::create(rewriter, batchMatMulOp.getLoc(), inputs,
                               outputs);
      if (isTransposeB)
        memref::DeallocOp::create(rewriter, loc, matmulInput1);
    }
    rewriter.eraseOp(batchMatMulOp.getOperation());
    return success();
  }
};

class TransposeOpLowering : public OpRewritePattern<linalg::TransposeOp> {
public:
  explicit TransposeOpLowering(MLIRContext *context)
      : OpRewritePattern<linalg::TransposeOp>(context) {}

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {

    Value input = transposeOp.getInput();
    Value output = transposeOp.getInit();
    Location loc = transposeOp.getLoc();

    // Only handle 2D transpose; let non-2D cases fall through to generic loops
    auto inputType = dyn_cast<MemRefType>(input.getType());
    if (!inputType || inputType.getRank() != 2)
      return failure();

    rewriter.replaceOpWithNewOp<tile::TileTransposeOp>(transposeOp, input,
                                                       output);
    return success();
  }
};

class Conv2dNhwcHwcfLowering
    : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
public:
  using OpRewritePattern<linalg::Conv2DNhwcHwcfOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {
    auto inputs = convOp.getInputs();
    auto outputs = convOp.getOutputs();
    if (inputs.size() != 2 || outputs.size() != 1)
      return failure();
    auto stride = getUniformAttr(convOp.getStrides());
    auto dilation = getUniformAttr(convOp.getDilations());
    if (!stride || !dilation || *stride != 1 || *dilation != 1)
      return failure();

    Value input = inputs[0];
    Value filter = inputs[1];
    Value output = outputs[0];
    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto filterType = dyn_cast<MemRefType>(filter.getType());
    auto outputType = dyn_cast<MemRefType>(output.getType());
    if (!inputType || !filterType || !outputType)
      return failure();
    if (!supportsTileConv(inputType, filterType, outputType))
      return failure();
    rewriter.replaceOpWithNewOp<tile::TileConv2dOp>(convOp, input, filter,
                                                    output);
    return success();
  }
};

class Conv2dNhwcFhwcLowering
    : public OpRewritePattern<linalg::Conv2DNhwcFhwcOp> {
public:
  using OpRewritePattern<linalg::Conv2DNhwcFhwcOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::Conv2DNhwcFhwcOp convOp,
                                PatternRewriter &rewriter) const override {
    auto inputs = convOp.getInputs();
    auto outputs = convOp.getOutputs();
    if (inputs.size() != 2 || outputs.size() != 1)
      return failure();

    Value input = inputs[0];
    Value filter = inputs[1];
    Value output = outputs[0];
    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto filterType = dyn_cast<MemRefType>(filter.getType());
    auto outputType = dyn_cast<MemRefType>(output.getType());
    if (!inputType || !filterType || !outputType)
      return failure();
    if (inputType.getRank() != 4 || filterType.getRank() != 4 ||
        outputType.getRank() != 4)
      return failure();

    auto stride = getUniformAttr(convOp.getStrides());
    auto dilation = getUniformAttr(convOp.getDilations());
    if (!stride || !dilation || *stride != 1 || *dilation != 1)
      return failure();

    Location loc = convOp.getLoc();
    ArrayRef<int64_t> filterShape = filterType.getShape();
    int64_t oc = filterShape[0];
    int64_t kh = filterShape[1];
    int64_t kw = filterShape[2];
    int64_t c = filterShape[3];

    auto hwcfType =
        MemRefType::get({kh, kw, c, oc}, filterType.getElementType());
    if (!supportsTileConv(inputType, hwcfType, outputType))
      return failure();
    Value hwcf = memref::AllocOp::create(rewriter, loc, hwcfType);

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value ocUb = arith::ConstantIndexOp::create(rewriter, loc, oc);
    Value khUb = arith::ConstantIndexOp::create(rewriter, loc, kh);
    Value kwUb = arith::ConstantIndexOp::create(rewriter, loc, kw);
    Value cUb = arith::ConstantIndexOp::create(rewriter, loc, c);

    auto ocLoop = scf::ForOp::create(rewriter, loc, zero, ocUb, one);
    rewriter.setInsertionPointToStart(ocLoop.getBody());
    Value ocIv = ocLoop.getInductionVar();

    auto khLoop = scf::ForOp::create(rewriter, loc, zero, khUb, one);
    rewriter.setInsertionPointToStart(khLoop.getBody());
    Value khIv = khLoop.getInductionVar();

    auto kwLoop = scf::ForOp::create(rewriter, loc, zero, kwUb, one);
    rewriter.setInsertionPointToStart(kwLoop.getBody());
    Value kwIv = kwLoop.getInductionVar();

    auto cLoop = scf::ForOp::create(rewriter, loc, zero, cUb, one);
    rewriter.setInsertionPointToStart(cLoop.getBody());
    Value cIv = cLoop.getInductionVar();

    Value value = memref::LoadOp::create(rewriter, loc, filter,
                                         ValueRange{ocIv, khIv, kwIv, cIv});
    memref::StoreOp::create(rewriter, loc, value, hwcf,
                            ValueRange{khIv, kwIv, cIv, ocIv});

    rewriter.setInsertionPointAfter(ocLoop);
    tile::TileConv2dOp::create(rewriter, loc, input, hwcf, output);
    memref::DeallocOp::create(rewriter, loc, hwcf);
    rewriter.eraseOp(convOp);
    return success();
  }
};

class Conv2dNchwFchwLowering
    : public OpRewritePattern<linalg::Conv2DNchwFchwOp> {
public:
  using OpRewritePattern<linalg::Conv2DNchwFchwOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp convOp,
                                PatternRewriter &rewriter) const override {
    auto inputs = convOp.getInputs();
    auto outputs = convOp.getOutputs();
    if (inputs.size() != 2 || outputs.size() != 1)
      return failure();

    Value input = inputs[0];
    Value filter = inputs[1];
    Value output = outputs[0];
    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto filterType = dyn_cast<MemRefType>(filter.getType());
    auto outputType = dyn_cast<MemRefType>(output.getType());
    if (!inputType || !filterType || !outputType)
      return failure();
    if (inputType.getRank() != 4 || filterType.getRank() != 4 ||
        outputType.getRank() != 4)
      return failure();

    auto stride = getUniformAttr(convOp.getStrides());
    auto dilation = getUniformAttr(convOp.getDilations());
    if (!stride || !dilation || *stride != 1 || *dilation != 1)
      return failure();

    Location loc = convOp.getLoc();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> filterShape = filterType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    int64_t n = inputShape[0];
    int64_t c = inputShape[1];
    int64_t h = inputShape[2];
    int64_t w = inputShape[3];
    int64_t f = filterShape[0];
    int64_t kh = filterShape[2];
    int64_t kw = filterShape[3];
    int64_t oh = outputShape[2];
    int64_t ow = outputShape[3];

    auto nhwcType = MemRefType::get({n, h, w, c}, inputType.getElementType());
    auto hwcfType =
        MemRefType::get({kh, kw, c, f}, filterType.getElementType());
    auto outNhwcType =
        MemRefType::get({n, oh, ow, f}, outputType.getElementType());
    if (!supportsTileConv(nhwcType, hwcfType, outNhwcType))
      return failure();
    Value nhwc = memref::AllocOp::create(rewriter, loc, nhwcType);
    Value hwcf = memref::AllocOp::create(rewriter, loc, hwcfType);
    Value outNhwc = memref::AllocOp::create(rewriter, loc, outNhwcType);

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value nUb = arith::ConstantIndexOp::create(rewriter, loc, n);
    Value cUb = arith::ConstantIndexOp::create(rewriter, loc, c);
    Value hUb = arith::ConstantIndexOp::create(rewriter, loc, h);
    Value wUb = arith::ConstantIndexOp::create(rewriter, loc, w);
    Value fUb = arith::ConstantIndexOp::create(rewriter, loc, f);
    Value khUb = arith::ConstantIndexOp::create(rewriter, loc, kh);
    Value kwUb = arith::ConstantIndexOp::create(rewriter, loc, kw);
    Value ohUb = arith::ConstantIndexOp::create(rewriter, loc, oh);
    Value owUb = arith::ConstantIndexOp::create(rewriter, loc, ow);

    auto nLoop = scf::ForOp::create(rewriter, loc, zero, nUb, one);
    rewriter.setInsertionPointToStart(nLoop.getBody());
    Value nIv = nLoop.getInductionVar();
    auto cLoop = scf::ForOp::create(rewriter, loc, zero, cUb, one);
    rewriter.setInsertionPointToStart(cLoop.getBody());
    Value cIv = cLoop.getInductionVar();
    auto hLoop = scf::ForOp::create(rewriter, loc, zero, hUb, one);
    rewriter.setInsertionPointToStart(hLoop.getBody());
    Value hIv = hLoop.getInductionVar();
    auto wLoop = scf::ForOp::create(rewriter, loc, zero, wUb, one);
    rewriter.setInsertionPointToStart(wLoop.getBody());
    Value wIv = wLoop.getInductionVar();
    Value inputValue = memref::LoadOp::create(rewriter, loc, input,
                                              ValueRange{nIv, cIv, hIv, wIv});
    memref::StoreOp::create(rewriter, loc, inputValue, nhwc,
                            ValueRange{nIv, hIv, wIv, cIv});

    rewriter.setInsertionPointAfter(nLoop);
    auto fLoop = scf::ForOp::create(rewriter, loc, zero, fUb, one);
    rewriter.setInsertionPointToStart(fLoop.getBody());
    Value fIv = fLoop.getInductionVar();
    auto fcLoop = scf::ForOp::create(rewriter, loc, zero, cUb, one);
    rewriter.setInsertionPointToStart(fcLoop.getBody());
    Value fcIv = fcLoop.getInductionVar();
    auto khLoop = scf::ForOp::create(rewriter, loc, zero, khUb, one);
    rewriter.setInsertionPointToStart(khLoop.getBody());
    Value khIv = khLoop.getInductionVar();
    auto kwLoop = scf::ForOp::create(rewriter, loc, zero, kwUb, one);
    rewriter.setInsertionPointToStart(kwLoop.getBody());
    Value kwIv = kwLoop.getInductionVar();
    Value filterValue = memref::LoadOp::create(
        rewriter, loc, filter, ValueRange{fIv, fcIv, khIv, kwIv});
    memref::StoreOp::create(rewriter, loc, filterValue, hwcf,
                            ValueRange{khIv, kwIv, fcIv, fIv});

    rewriter.setInsertionPointAfter(fLoop);
    tile::TileConv2dOp::create(rewriter, loc, nhwc, hwcf, outNhwc);

    auto onLoop = scf::ForOp::create(rewriter, loc, zero, nUb, one);
    rewriter.setInsertionPointToStart(onLoop.getBody());
    Value onIv = onLoop.getInductionVar();
    auto ofLoop = scf::ForOp::create(rewriter, loc, zero, fUb, one);
    rewriter.setInsertionPointToStart(ofLoop.getBody());
    Value ofIv = ofLoop.getInductionVar();
    auto ohLoop = scf::ForOp::create(rewriter, loc, zero, ohUb, one);
    rewriter.setInsertionPointToStart(ohLoop.getBody());
    Value ohIv = ohLoop.getInductionVar();
    auto owLoop = scf::ForOp::create(rewriter, loc, zero, owUb, one);
    rewriter.setInsertionPointToStart(owLoop.getBody());
    Value owIv = owLoop.getInductionVar();
    Value outputValue = memref::LoadOp::create(
        rewriter, loc, outNhwc, ValueRange{onIv, ohIv, owIv, ofIv});
    memref::StoreOp::create(rewriter, loc, outputValue, output,
                            ValueRange{onIv, ofIv, ohIv, owIv});

    rewriter.setInsertionPointAfter(onLoop);
    memref::DeallocOp::create(rewriter, loc, nhwc);
    memref::DeallocOp::create(rewriter, loc, hwcf);
    memref::DeallocOp::create(rewriter, loc, outNhwc);
    rewriter.eraseOp(convOp);
    return success();
  }
};

} // namespace

void populateLowerLinalgToTileConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<MatmulLowering>(patterns.getContext());
  patterns.add<BatchMatMulOpLowering>(patterns.getContext());
  patterns.add<TransposeOpLowering>(patterns.getContext());
  patterns.add<Conv2dNhwcHwcfLowering>(patterns.getContext());
  patterns.add<Conv2dNhwcFhwcLowering>(patterns.getContext());
  patterns.add<Conv2dNchwFchwLowering>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// LowerLinalgToTile
//===----------------------------------------------------------------------===//

namespace {
class LowerLinalgToTilePass
    : public PassWrapper<LowerLinalgToTilePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLinalgToTilePass);
  LowerLinalgToTilePass() = default;
  LowerLinalgToTilePass(const LowerLinalgToTilePass &) {}
  StringRef getArgument() const final { return "convert-linalg-to-tile"; }
  StringRef getDescription() const final {
    return "convert linalg dialect to tile dialect";
  }
  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<tile::TileDialect, func::FuncDialect, memref::MemRefDialect,
                linalg::LinalgDialect, arith::ArithDialect, scf::SCFDialect>();
  }
};
} // namespace

void LowerLinalgToTilePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  ConversionTarget target(*context);
  target.addLegalDialect<memref::MemRefDialect, tile::TileDialect,
                         arith::ArithDialect, scf::SCFDialect>();
  target.addLegalOp<linalg::FillOp, linalg::YieldOp>();
  RewritePatternSet patterns(context);
  populateLowerLinalgToTileConversionPatterns(patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerLinalgToTilePass() {
  PassRegistration<LowerLinalgToTilePass>();
}
} // namespace buddy
} // namespace mlir
