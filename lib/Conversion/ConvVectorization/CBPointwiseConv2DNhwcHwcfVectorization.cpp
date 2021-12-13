//===- CBSMPointwiseConvVectorization.cpp - Vectorize Convolution by CB-SM-===//
//
// This file implements the coefficients broadcasting algorthm with strip mining
// strategy (CB-SM) for Pointwise Convolution vectorization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/VectorOps.h"

#include "mlir/Pass/Pass.h"

#include <iostream>

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
class CBSMPointwiseConvVectorizationPattern : public ConversionPattern {
public:
  explicit CBSMPointwiseConvVectorizationPattern(MLIRContext *context,
                                                 int64_t strideParam)
      : ConversionPattern(linalg::Conv2DNhwcHwcfOp::getOperationName(), 1,
                          context) {
    stride = strideParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

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

    // TODO(Joe): Support conversion to linalg.batch_matmul.
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

    // start arrange
    SmallVector<ReassociationIndices, 4> reassociationIndices = {{0, 1, 2},
                                                                 {3}};

    auto reshapedInputType =
        RankedTensorType::get({inputShape[1] * inputShape[2], inputShape[3]},
                              inputShapeType.getElementType());
    auto reshapedFilterType = RankedTensorType::get(
        {filterShape[2], filterShape[3]}, filterShapeType.getElementType());

    auto reshapedOutputType =
        RankedTensorType::get({outputShape[1] * outputShape[2], outputShape[3]},
                              outputShapeType.getElementType());

    Value reshapedInput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedInputType, input, reassociationIndices);
    Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedFilterType, kernel, reassociationIndices);
    Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedOutputType, output, reassociationIndices);

    // Create MutmulOp
    auto matmulResult = rewriter.create<linalg::MatmulOp>(
        loc, reshapedOutputType, ArrayRef<Value>{reshapedInput, reshapedFilter},
        ArrayRef<Value>{reshapedOutput});

    auto reshapedResult = rewriter.create<tensor::ExpandShapeOp>(
        loc, outputShapeType, matmulResult.getResults()[0],
        reassociationIndices);

    // Remove the origin convolution operation.
    rewriter.replaceOp(op, ArrayRef<Value>{reshapedResult});
    // rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// PointwiseConvVectorizationPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg convolution to mixture of
/// Affine + Vector + Std operations.
namespace {
class PointwiseConvVectorizationPass
    : public PassWrapper<PointwiseConvVectorizationPass,
                         OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "pointwise-conv-vectorization"; }
  StringRef getDescription() const final {
    return "Pointwise Convolution vectorization.";
  }
  PointwiseConvVectorizationPass() = default;
  PointwiseConvVectorizationPass(const PointwiseConvVectorizationPass &) {}
  explicit PointwiseConvVectorizationPass(int64_t strideParam) {
    stride = strideParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, tensor::TensorDialect, scf::SCFDialect,
                AffineDialect, VectorDialect, StandardOpsDialect>();
  }

  Option<int64_t> stride{*this, "strip-mining",
                         llvm::cl::desc("Strip mining size."),
                         llvm::cl::init(32)};
};
} // end anonymous namespace.

void PointwiseConvVectorizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithmeticDialect, AffineDialect, scf::SCFDialect,
                       StandardOpsDialect, memref::MemRefDialect, VectorDialect,
                       tensor::TensorDialect>();
  target.addLegalOp<ModuleOp, FuncOp, ReturnOp>();
  target.addLegalOp<linalg::FillOp, tensor::CollapseShapeOp, linalg::MatmulOp,
                    tensor::ExpandShapeOp>();

  RewritePatternSet patterns(context);
  patterns.add<CBSMPointwiseConvVectorizationPattern>(context, stride);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerPointwiseConvVectorizationPass() {
  PassRegistration<PointwiseConvVectorizationPass>();
}
} // namespace buddy
} // namespace mlir
