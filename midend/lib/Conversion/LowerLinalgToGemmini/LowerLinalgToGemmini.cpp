//====- LowerLinalgToGemmini - Linalg Dialect Lowering Pass --------------===//
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
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::MatmulOp matMulOp,
                                PatternRewriter &rewriter) const override {
    auto inputs = matMulOp.getInputs();
    auto ouputs = matMulOp.getOutputs();
    Location loc = matMulOp.getLoc();
    Value input0 = inputs[0];
    Value input1 = inputs[1];
    Value output0 = ouputs[0];
    MemRefType input0Type = input0.getType().dyn_cast<MemRefType>();
    if (!input0Type.getElementType().isInteger(8))
      return failure();
    MemRefType input1Type = input1.getType().dyn_cast<MemRefType>();
    if (!input1Type.getElementType().isInteger(8))
      return failure();
    MemRefType output0Type = output0.getType().dyn_cast<MemRefType>();
    if (!output0Type.getElementType().isInteger(8))
      return failure();
    MemRefType biasType =
        MemRefType::get(input0Type.getShape(), rewriter.getI32Type());
    IntegerAttr alignment = rewriter.getI64IntegerAttr(64);
    llvm::APFloat scale1((float)1.0);
    llvm::APFloat scale0((float)0.0);
    Value bias = rewriter.create<memref::AllocOp>(loc, biasType, alignment);
    IntegerAttr fillOpInputAttr = rewriter.getI32IntegerAttr(0);
    Value fillOpInputValue = rewriter.create<arith::ConstantOp>(
        loc, fillOpInputAttr, rewriter.getI32Type());
    rewriter.create<linalg::FillOp>(loc, fillOpInputValue, bias);
    rewriter.replaceOpWithNewOp<gemmini::TileMatMulOp>(
        matMulOp, input0, input1, output0, bias, /*aScaleFactor = */ scale1,
        /*bScaleFactor = */ scale1, /*dScaleFactor = */ scale1, /*act = */ 0,
        /*accScale = */ scale1, /*bertScale = */ scale0);
    rewriter.create<memref::DeallocOp>(loc, bias);
    return success();
  }
};

class Conv2DHchwFchwLowering
    : public OpRewritePattern<linalg::Conv2DNchwFchwOp> {
  using OpRewritePattern<linalg::Conv2DNchwFchwOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp convOp,
                                PatternRewriter &rewriter) const override {
    auto inputs = convOp.getInputs();
    Value input0 = inputs[0];
    Value input1 = inputs[1];
    Value output = convOp.getOutputs()[0];
    Location loc = convOp.getLoc();
    MemRefType inputType = input0.getType().dyn_cast<MemRefType>();
    MemRefType weightsType = input1.getType().dyn_cast<MemRefType>();
    MemRefType outputType = output.getType().dyn_cast<MemRefType>();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    ArrayRef<int64_t> weightsShape = weightsType.getShape();
    SmallVector<int64_t> inputMatShape = {inputShape[0], inputShape[2],
                                          inputShape[3], inputShape[1]};
    SmallVector<int64_t> weightsMatShape = {
        weightsShape[1] * weightsShape[2] * weightsShape[3], weightsShape[0]};
    MemRefType inputMatType =
        MemRefType::get(inputMatShape, rewriter.getI8Type());
    MemRefType weightsMatType =
        MemRefType::get(weightsMatShape, rewriter.getI8Type());
    Value inputMat = rewriter.create<memref::AllocOp>(loc, inputMatType);
    Value weightsMat = rewriter.create<memref::AllocOp>(loc, weightsMatType);
    MemRefType biasType =
        MemRefType::get(weightsShape[0], rewriter.getI32Type());
    SmallVector<int64_t, 2> outputMatShape = {
        inputShape[0] * outputShape[2] * outputShape[3], outputShape[1]};
    MemRefType outputMatType =
        MemRefType::get(outputMatShape, rewriter.getI8Type());
    Value bias = rewriter.create<memref::AllocOp>(loc, biasType);
    Value outputMat = rewriter.create<memref::AllocOp>(loc, outputMatType);
    IntegerAttr outDimAttr = rewriter.getI64IntegerAttr(outputShape[2]);
    Value batchSize =
        rewriter.create<arith::ConstantIndexOp>(loc, inputShape[0]);
    Value outDim = rewriter.create<arith::ConstantOp>(loc, outDimAttr,
                                                      rewriter.getI64Type());
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
    rewriter.create<gemmini::TileConvOp>(loc, inputMat, weightsMat, bias,
                                         outputMat, outDim, kernelDim,
                                         llvm::APFloat(float(1.0)));
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
    tmp0 = rewriter.create<arith::MulIOp>(loc, loopIvs0[0], batchSize);
    tmp1 = rewriter.create<arith::MulIOp>(loc, loopIvs0[2], outDim);
    tmp0 = rewriter.create<arith::AddIOp>(loc, tmp0, tmp1);
    tmp0 = rewriter.create<arith::AddIOp>(loc, tmp0, loopIvs0[3]);
    loopIvs1.push_back(tmp0);
    loopIvs1.push_back(loopIvs0[1]);
    tmp1 = rewriter.create<memref::LoadOp>(loc, outputMat, loopIvs1);
    rewriter.create<memref::StoreOp>(loc, tmp1, output, loopIvs0);
    rewriter.setInsertionPointAfter(loopOp);
    rewriter.create<memref::DeallocOp>(loc, inputMat);
    rewriter.create<memref::DeallocOp>(loc, outputMat);
    rewriter.create<memref::DeallocOp>(loc, bias);
    return success();
  }
};

} // namespace

void populateLowerLinalgToGemminiConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<MatmulLowering>(patterns.getContext());
  patterns.add<Conv2DHchwFchwLowering>(patterns.getContext());
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
  populateLowerLinalgToGemminiConversionPatterns(patterns);
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
