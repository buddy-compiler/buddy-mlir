//===- IM2COLConv2DNhwcHwcf.cpp - transfer Convolution to GEMM by IM2COL --===//
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
// This file implements the algorithm to transfer Convolution to GEMM by IM2COL.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/Pass/Pass.h"

#include <iostream>

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
class IM2COLConv2DNhwcHwcfPattern : public ConversionPattern {
public:
  explicit IM2COLConv2DNhwcHwcfPattern(MLIRContext *context)
      : ConversionPattern(linalg::Conv2DNhwcHwcfOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    // Get input, kernel and output.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    // Get shape of input and output
    ShapedType inputShapeTy = input.getType().cast<ShapedType>();
    ShapedType filterShapeTy = kernel.getType().cast<ShapedType>();
    ShapedType outputShapeTy = output.getType().cast<ShapedType>();

    auto inputShape = inputShapeTy.getShape();
    auto filterShape = filterShapeTy.getShape();
    auto outputShape = outputShapeTy.getShape();
    // Assertions
    if (inputShape[0] != 1)
      return failure();

    auto convOp = dyn_cast<linalg::Conv2DNhwcHwcfOp>(op);

    if (!llvm::all_of(convOp.strides(), [](APInt element) {
          return element.getSExtValue() == 1;
        }))
      return failure();

    if (!llvm::all_of(convOp.dilations(), [](APInt element) {
          return element.getSExtValue() == 1;
        }))
      return failure();

    // col tensor shape (n, d1, d1, k1, k2, ci)
    SmallVector<int64_t, 4> colTensorShape = {outputShape[0], outputShape[1],
                                              outputShape[2], filterShape[0],
                                              filterShape[1], filterShape[2]};

    Value colTensor = rewriter.create<linalg::InitTensorOp>(
        loc, colTensorShape, inputShapeTy.getElementType());
    
    auto n = rewriter.getAffineDimExpr(0);
    auto d = [&](int i) { return rewriter.getAffineDimExpr(i); };
    auto k = [&](int i) { return rewriter.getAffineDimExpr(i + 2); };
    auto ci = rewriter.getAffineDimExpr(5);
    auto s = [&](unsigned i) {
      return rewriter.getAffineConstantExpr(
          convOp.strides().getValues<int64_t>()[i]);
    };

    SmallVector<AffineExpr, 4> inputExprs = {n, d(1) * s(0) + k(1),
                                             d(2) * s(1) + k(2), ci};

    auto nloops = colTensorShape.size();  

    SmallVector<StringRef, 3> loopAttrTy(nloops, "parallel");
    
    SmallVector<AffineMap, 4> idxMaps = {
        AffineMap::get(nloops, 0, inputExprs, rewriter.getContext()),
        AffineMap::getMultiDimIdentityMap(nloops, rewriter.getContext())};
    
    auto img2ColTensor = rewriter.create<linalg::GenericOp>(
        loc, colTensor.getType(),
        /*inputs=*/input, /*outputs=*/colTensor, idxMaps,
        loopAttrTy,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
        });
    
    SmallVector<ReassociationIndices> img2ColTensorReassociationIdxs = {
        {0, 1, 2}, {3, 4, 5}};
    
    SmallVector<ReassociationIndices> filterAndOutputReassociationIdxs = {
        {0, 1, 2}, {3}};

    auto reshapedImg2ColTensorTy = RankedTensorType::get(
        {outputShape[1] * outputShape[2],
         filterShape[0] * filterShape[1] * filterShape[2]},
        inputShapeTy.getElementType());
    
    auto reshapedFilterTy = RankedTensorType::get(
        {filterShape[0] * filterShape[1] * filterShape[2], filterShape[3]},
        inputShapeTy.getElementType());
    
    auto reshapedOutputTy =
        RankedTensorType::get({outputShape[1] * outputShape[2], outputShape[3]},
                              outputShapeTy.getElementType());
    Value reshapedImg2ColTensor =
        rewriter.create<tensor::CollapseShapeOp>(
            loc, reshapedImg2ColTensorTy, img2ColTensor.getResult(0),
            img2ColTensorReassociationIdxs);
    
    Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedFilterTy, kernel, filterAndOutputReassociationIdxs);
    
    Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedOutputTy, output, filterAndOutputReassociationIdxs);
    
    auto matRes = rewriter.create<linalg::MatmulOp>(
        loc, reshapedOutputTy,
        ArrayRef<Value>{reshapedImg2ColTensor, reshapedFilter},
        ArrayRef<Value>{reshapedOutput});
    
    auto reshapedRes = rewriter.create<tensor::ExpandShapeOp>(
        loc, outputShapeTy, matRes.getResults()[0],
        filterAndOutputReassociationIdxs);

    rewriter.replaceOp(op, ArrayRef<Value>{reshapedRes});
    return success();                
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// IM2COLConv2DPass
//===----------------------------------------------------------------------===//

namespace {
class IM2COLConv2DPass
    : public PassWrapper<IM2COLConv2DPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "conv2d-to-im2col"; }
  StringRef getDescription() const final {
    return "Convolution to Im2col.";
  }
  IM2COLConv2DPass() = default;
  IM2COLConv2DPass(const IM2COLConv2DPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect, AffineDialect,
                    scf::SCFDialect, func::FuncDialect>();
  }
};
} // end anonymous namespace.

void IM2COLConv2DPass::runOnOperation() {
    MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithmeticDialect, scf::SCFDialect,
                         func::FuncDialect, memref::MemRefDialect,
                         tensor::TensorDialect>();
  target.addLegalOp<ModuleOp, FuncOp, linalg::YieldOp, linalg::GenericOp, linalg::InitTensorOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp, tensor::CollapseShapeOp, linalg::MatmulOp,
                    tensor::ExpandShapeOp>();
  RewritePatternSet patterns(context);

  patterns.add<IM2COLConv2DNhwcHwcfPattern>(context);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerIM2COLConv2DPass() {
  PassRegistration<IM2COLConv2DPass>();
}
} // namespace buddy
} // namespace mlir
