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
      aVal = rewriter.create<memref::CollapseShapeOp>(loc, input0, reassoc);
      bVal = rewriter.create<memref::CollapseShapeOp>(loc, input1, reassoc);
      oVal = rewriter.create<memref::CollapseShapeOp>(loc, output0, reassoc);
    }
    rewriter.replaceOpWithNewOp<tile::TileMatMulOp>(matMulOp, aVal, bVal, oVal);
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
      SmallVector<Value> outputs = {subOutput};
      rewriter.create<linalg::MatmulOp>(batchMatMulOp.getLoc(), inputs,
                                        outputs);
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
    if (!isa<MemRefType>(input.getType()) ||
        !isa<MemRefType>(filter.getType()) ||
        !isa<MemRefType>(output.getType()))
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
    Value hwcf = rewriter.create<memref::AllocOp>(loc, hwcfType);

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value ocUb = rewriter.create<arith::ConstantIndexOp>(loc, oc);
    Value khUb = rewriter.create<arith::ConstantIndexOp>(loc, kh);
    Value kwUb = rewriter.create<arith::ConstantIndexOp>(loc, kw);
    Value cUb = rewriter.create<arith::ConstantIndexOp>(loc, c);

    auto ocLoop = rewriter.create<scf::ForOp>(loc, zero, ocUb, one);
    rewriter.setInsertionPointToStart(ocLoop.getBody());
    Value ocIv = ocLoop.getInductionVar();

    auto khLoop = rewriter.create<scf::ForOp>(loc, zero, khUb, one);
    rewriter.setInsertionPointToStart(khLoop.getBody());
    Value khIv = khLoop.getInductionVar();

    auto kwLoop = rewriter.create<scf::ForOp>(loc, zero, kwUb, one);
    rewriter.setInsertionPointToStart(kwLoop.getBody());
    Value kwIv = kwLoop.getInductionVar();

    auto cLoop = rewriter.create<scf::ForOp>(loc, zero, cUb, one);
    rewriter.setInsertionPointToStart(cLoop.getBody());
    Value cIv = cLoop.getInductionVar();

    Value value = rewriter.create<memref::LoadOp>(
        loc, filter, ValueRange{ocIv, khIv, kwIv, cIv});
    rewriter.create<memref::StoreOp>(loc, value, hwcf,
                                     ValueRange{khIv, kwIv, cIv, ocIv});

    rewriter.setInsertionPointAfter(ocLoop);
    rewriter.create<tile::TileConv2dOp>(loc, input, hwcf, output);
    rewriter.create<memref::DeallocOp>(loc, hwcf);
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
    Value nhwc = rewriter.create<memref::AllocOp>(loc, nhwcType);
    Value hwcf = rewriter.create<memref::AllocOp>(loc, hwcfType);
    Value outNhwc = rewriter.create<memref::AllocOp>(loc, outNhwcType);

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value nUb = rewriter.create<arith::ConstantIndexOp>(loc, n);
    Value cUb = rewriter.create<arith::ConstantIndexOp>(loc, c);
    Value hUb = rewriter.create<arith::ConstantIndexOp>(loc, h);
    Value wUb = rewriter.create<arith::ConstantIndexOp>(loc, w);
    Value fUb = rewriter.create<arith::ConstantIndexOp>(loc, f);
    Value khUb = rewriter.create<arith::ConstantIndexOp>(loc, kh);
    Value kwUb = rewriter.create<arith::ConstantIndexOp>(loc, kw);
    Value ohUb = rewriter.create<arith::ConstantIndexOp>(loc, oh);
    Value owUb = rewriter.create<arith::ConstantIndexOp>(loc, ow);

    auto nLoop = rewriter.create<scf::ForOp>(loc, zero, nUb, one);
    rewriter.setInsertionPointToStart(nLoop.getBody());
    Value nIv = nLoop.getInductionVar();
    auto cLoop = rewriter.create<scf::ForOp>(loc, zero, cUb, one);
    rewriter.setInsertionPointToStart(cLoop.getBody());
    Value cIv = cLoop.getInductionVar();
    auto hLoop = rewriter.create<scf::ForOp>(loc, zero, hUb, one);
    rewriter.setInsertionPointToStart(hLoop.getBody());
    Value hIv = hLoop.getInductionVar();
    auto wLoop = rewriter.create<scf::ForOp>(loc, zero, wUb, one);
    rewriter.setInsertionPointToStart(wLoop.getBody());
    Value wIv = wLoop.getInductionVar();
    Value inputValue = rewriter.create<memref::LoadOp>(
        loc, input, ValueRange{nIv, cIv, hIv, wIv});
    rewriter.create<memref::StoreOp>(loc, inputValue, nhwc,
                                     ValueRange{nIv, hIv, wIv, cIv});

    rewriter.setInsertionPointAfter(nLoop);
    auto fLoop = rewriter.create<scf::ForOp>(loc, zero, fUb, one);
    rewriter.setInsertionPointToStart(fLoop.getBody());
    Value fIv = fLoop.getInductionVar();
    auto fcLoop = rewriter.create<scf::ForOp>(loc, zero, cUb, one);
    rewriter.setInsertionPointToStart(fcLoop.getBody());
    Value fcIv = fcLoop.getInductionVar();
    auto khLoop = rewriter.create<scf::ForOp>(loc, zero, khUb, one);
    rewriter.setInsertionPointToStart(khLoop.getBody());
    Value khIv = khLoop.getInductionVar();
    auto kwLoop = rewriter.create<scf::ForOp>(loc, zero, kwUb, one);
    rewriter.setInsertionPointToStart(kwLoop.getBody());
    Value kwIv = kwLoop.getInductionVar();
    Value filterValue = rewriter.create<memref::LoadOp>(
        loc, filter, ValueRange{fIv, fcIv, khIv, kwIv});
    rewriter.create<memref::StoreOp>(loc, filterValue, hwcf,
                                     ValueRange{khIv, kwIv, fcIv, fIv});

    rewriter.setInsertionPointAfter(fLoop);
    rewriter.create<tile::TileConv2dOp>(loc, nhwc, hwcf, outNhwc);

    auto onLoop = rewriter.create<scf::ForOp>(loc, zero, nUb, one);
    rewriter.setInsertionPointToStart(onLoop.getBody());
    Value onIv = onLoop.getInductionVar();
    auto ofLoop = rewriter.create<scf::ForOp>(loc, zero, fUb, one);
    rewriter.setInsertionPointToStart(ofLoop.getBody());
    Value ofIv = ofLoop.getInductionVar();
    auto ohLoop = rewriter.create<scf::ForOp>(loc, zero, ohUb, one);
    rewriter.setInsertionPointToStart(ohLoop.getBody());
    Value ohIv = ohLoop.getInductionVar();
    auto owLoop = rewriter.create<scf::ForOp>(loc, zero, owUb, one);
    rewriter.setInsertionPointToStart(owLoop.getBody());
    Value owIv = owLoop.getInductionVar();
    Value outputValue = rewriter.create<memref::LoadOp>(
        loc, outNhwc, ValueRange{onIv, ohIv, owIv, ofIv});
    rewriter.create<memref::StoreOp>(loc, outputValue, output,
                                     ValueRange{onIv, ofIv, ohIv, owIv});

    rewriter.setInsertionPointAfter(onLoop);
    rewriter.create<memref::DeallocOp>(loc, nhwc);
    rewriter.create<memref::DeallocOp>(loc, hwcf);
    rewriter.create<memref::DeallocOp>(loc, outNhwc);
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
