//===- IM2COLConv2DNhwcHwcf.cpp - transfer Convolution to GEMM by IM2COL --===//
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

    // printf("-----------------");

    // Get input, kernel and output.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    // Get shape of input and output
    ShapedType inputShapeType = input.getType().cast<ShapedType>();
    ShapedType filterShapeType = kernel.getType().cast<ShapedType>();
    ShapedType outputShapeType = output.getType().cast<ShapedType>();

    auto inputShape = inputShapeType.getShape();
    auto filterShape = filterShapeType.getShape();
    auto outputShape = outputShapeType.getShape();
    // Assertions
    if (filterShape[0] != 1 || filterShape[1] != 1)
      return failure();

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
        loc, colTensorShape, inputShapeType.getElementType());
    
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

    SmallVector<StringRef, 3> loopAttributeTypes(nloops, "parallel");
    
    SmallVector<AffineMap, 4> indexingMaps = {
        AffineMap::get(nloops, 0, inputExprs, rewriter.getContext()),
        AffineMap::getMultiDimIdentityMap(nloops, rewriter.getContext())};
    
    auto img2ColTensor = rewriter.create<linalg::GenericOp>(
        loc, colTensor.getType(),
        /*inputs=*/input, /*outputs=*/colTensor, indexingMaps,
        loopAttributeTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
        });
    
    SmallVector<ReassociationIndices> img2ColTensorReassociationIndices = {
        {0, 1, 2}, {3, 4, 5}};
    
    SmallVector<ReassociationIndices> filterAndOutputReassociationIndices = {
        {0, 1, 2}, {3}};

    auto reshapedImg2ColTensorType = RankedTensorType::get(
        {outputShape[1] * outputShape[2],
         filterShape[0] * filterShape[1] * filterShape[2]},
        inputShapeType.getElementType());
    
    auto reshapedFilterType = RankedTensorType::get(
        {filterShape[0] * filterShape[1] * filterShape[2], filterShape[3]},
        inputShapeType.getElementType());
    
    auto reshapedOutputType =
        RankedTensorType::get({outputShape[1] * outputShape[2], outputShape[3]},
                              outputShapeType.getElementType());
    Value reshapedImg2ColTensor =
        rewriter.create<tensor::CollapseShapeOp>(
            loc, reshapedImg2ColTensorType, img2ColTensor.getResult(0),
            img2ColTensorReassociationIndices);
    
    Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedFilterType, kernel, filterAndOutputReassociationIndices);
    
    Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedOutputType, output, filterAndOutputReassociationIndices);
    
    auto matmulResult = rewriter.create<linalg::MatmulOp>(
        loc, reshapedOutputType,
        ArrayRef<Value>{reshapedImg2ColTensor, reshapedFilter},
        ArrayRef<Value>{reshapedOutput});
    
    auto reshapedResult = rewriter.create<tensor::ExpandShapeOp>(
        loc, outputShapeType, matmulResult.getResults()[0],
        filterAndOutputReassociationIndices);

    rewriter.replaceOp(op, ArrayRef<Value>{reshapedResult});
    // rewriter.eraseOp(op);
    return success();                
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// IM2COLConv2DPass
//===----------------------------------------------------------------------===//

namespace {
class IM2COLConv2DPass
    : public PassWrapper<IM2COLConv2DPass,
                         OperationPass<ModuleOp>> {
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
