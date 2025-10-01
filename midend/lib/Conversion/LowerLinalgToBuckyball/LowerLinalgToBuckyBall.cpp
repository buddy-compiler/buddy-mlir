//====- LowerLinalgToBuckyball.cpp - Linalg Dialect Lowering Pass -----------===//
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

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"
using namespace mlir;
using namespace buddy;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
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
    rewriter.replaceOpWithNewOp<buckyball::VecTileMatMulOp>(
        matMulOp, input0, input1, output0);
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
    MemRefType input0Type =  dyn_cast<MemRefType>(input0.getType());
    ArrayRef<int64_t> input0Shape = input0Type.getShape();
    MemRefType input1Type =  dyn_cast<MemRefType>(input1.getType());
    ArrayRef<int64_t> input1Shape = input1Type.getShape();
    MemRefType outputType =  dyn_cast<MemRefType>(output.getType());
    ArrayRef<int64_t> outputShape = outputType.getShape();
    Type elemType = input0Type.getElementType();
    for (unsigned i = 0; i != input0Shape[0]; i++) {
      SmallVector<int64_t> staticOffsets = {i, 0, 0};
      SmallVector<int64_t> staticSizes = {1, input0Shape[1], input0Shape[2]};
      SmallVector<int64_t> staticStrides = {1, 1, 1};
      SmallVector<int64_t> resultShape = {input0Shape[1], input0Shape[2]};
      SmallVector<int64_t> layout = {input0Shape[2], 1};
      FailureOr<StridedLayoutAttr> computelayout =
          StridedLayoutAttr::get(batchMatMulOp.getContext(),
                                 i * input0Shape[1] * input0Shape[2], layout);
      MemRefType resultType =
          MemRefType::get(resultShape, elemType, *computelayout, 0);
      Value subInput0 = rewriter.create<memref::SubViewOp>(
          loc, resultType, input0, staticOffsets, staticSizes, staticStrides);

      staticSizes.assign({1, input1Shape[1], input1Shape[2]});
      resultShape.assign({input1Shape[1], input1Shape[2]});
      layout.assign({input1Shape[2], 1});
      computelayout =
          StridedLayoutAttr::get(batchMatMulOp.getContext(),
                                 i * input1Shape[1] * input1Shape[2], layout);
      resultType = MemRefType::get(resultShape, elemType, *computelayout, 0);
      Value subInput1 = rewriter.create<memref::SubViewOp>(
          loc, resultType, input1, staticOffsets, staticSizes, staticStrides);

      staticSizes.assign({1, outputShape[1], outputShape[2]});
      resultShape.assign({outputShape[1], outputShape[2]});
      layout.assign({outputShape[2], 1});
      computelayout =
          StridedLayoutAttr::get(batchMatMulOp.getContext(),
                                 i * outputShape[1] * outputShape[2], layout);
      resultType = MemRefType::get(resultShape, elemType, *computelayout, 0);
      Value subOutput = rewriter.create<memref::SubViewOp>(
          loc, resultType, output, staticOffsets, staticSizes, staticStrides);
      SmallVector<Value> inputs = {subInput0, subInput1};
      SmallVector<Value> output = {subOutput};
      rewriter.create<linalg::MatmulOp>(batchMatMulOp.getLoc(), inputs, output);
    }
    rewriter.eraseOp(batchMatMulOp.getOperation());
    return success();
  }
};

} // namespace

void populateLowerLinalgToBuckyballConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<MatmulLowering>(patterns.getContext());
  patterns.add<BatchMatMulOpLowering>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// LowerLinalgToBuckyball
//===----------------------------------------------------------------------===//

namespace {
class LowerLinalgToBuckyballPass
    : public PassWrapper<LowerLinalgToBuckyballPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLinalgToBuckyballPass);
  LowerLinalgToBuckyballPass() = default;
  LowerLinalgToBuckyballPass(const LowerLinalgToBuckyballPass &) {}
  StringRef getArgument() const final { return "convert-linalg-to-buckyball"; }
  StringRef getDescription() const final {
    return "convert linalg dialect to buckyball dialect";
  }
  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<buckyball::BuckyballDialect, func::FuncDialect,
                    memref::MemRefDialect, linalg::LinalgDialect,
                    arith::ArithDialect, scf::SCFDialect>();
  }
};
} // namespace

void LowerLinalgToBuckyballPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  ConversionTarget target(*context);
  target.addLegalDialect<memref::MemRefDialect, 
                         buckyball::BuckyballDialect,
                         arith::ArithDialect, 
                         scf::SCFDialect>();
  target.addLegalOp<linalg::FillOp, linalg::YieldOp>();
  RewritePatternSet patterns(context);
  populateLowerLinalgToBuckyballConversionPatterns(patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerLinalgToBuckyballPass() {
  PassRegistration<LowerLinalgToBuckyballPass>();
}
} // namespace buddy
} // namespace mlir
