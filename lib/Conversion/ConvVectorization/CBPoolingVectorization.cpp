//====- CBPoolingVectorization.cpp ----------------------------------------===//
//
// This file implements the pooling vectorization.
//
//===----------------------------------------------------------------------===//
#include <iostream>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/Vector/VectorOps.h.inc"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

// PoolingNhwcSum vectorization pattern
class CBPoolingNhwcSumVectorizationPattern : public ConversionPattern {
public:
  explicit CBPoolingNhwcSumVectorizationPattern(MLIRContext *context,
                                                int64_t stripParam)
      : ConversionPattern(linalg::PoolingNhwcSumOp::getOperationName(), 1,
                          context) {
    strip = stripParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();
    // Get input, kernel and output
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    // Element type
    MemRefType inputMemRefTy = input.getType().dyn_cast<MemRefType>();
    // Element type
    Type fTy = inputMemRefTy.getElementType();
    // Constants
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    Value c3 = rewriter.create<arith::ConstantIndexOp>(loc, 3);
    // Dimensions of the input.
    Value batch = rewriter.create<memref::DimOp>(loc, input, c0);
    Value height = rewriter.create<memref::DimOp>(loc, input, c1);
    Value width = rewriter.create<memref::DimOp>(loc, input, c2);
    Value channels = rewriter.create<memref::DimOp>(loc, input, c3);
    // TODO check the strides and the dilations.
    Value strides = rewriter.create<arith::ConstantOp>(
        loc, op->getAttrOfType<mlir::DenseIntElementsAttr>("strides"));
    // Dilations.
    Value dilations = rewriter.create<arith::ConstantOp>(
        loc, op->getAttrOfType<mlir::DenseIntElementsAttr>("dilations"));
    // Dimension of the kernel
    Value kernelHeight = rewriter.create<memref::DimOp>(loc, kernel, c0);
    Value kernelWidth = rewriter.create<memref::DimOp>(loc, kernel, c1);
    Value inputHeight0 =
        rewriter.create<arith::DivUIOp>(loc, height, kernelHeight);
    Value inputHeight = rewriter.create<arith::AddIOp>(loc, inputHeight0, c1);
    Value inputWidth0 =
        rewriter.create<arith::DivUIOp>(loc, width, kernelWidth);
    Value inputWidth = rewriter.create<arith::AddIOp>(loc, inputWidth0, c1);
    // Affine maps
    AffineExpr n, h, w, c;
    bindDims(ctx, n, h, w, c);
    AffineMap outputVectorMap = AffineMap::get(4, 0, {n, h, w, c}, ctx);
    // Subview vector shape
    MemRefType kernelTy = kernel.getType().dyn_cast<MemRefType>();
    SmallVector<int64_t> subviewVecShape{1};
    for (unsigned i = 0; i < kernelTy.getRank(); i++) {
      if (!kernelTy.isDynamicDim(i)) {
        subviewVecShape.push_back(kernelTy.getDimSize(i));
      } else {
        return failure();
      }
    }
    subviewVecShape.push_back(1);
    // subview vector type
    VectorType subviewVecTy = VectorType::get(subviewVecShape, fTy);
    // Flattened vector type
    SmallVector<int64_t> flattenedVecShape{1};
    for (auto &val : subviewVecShape)
      flattenedVecShape[0] *= val;
    VectorType flattenedVecTy = VectorType::get(flattenedVecShape, fTy);
    // Loop.
    SmallVector<Value, 4> lowerBounds(4, c0);
    SmallVector<Value, 4> upperBounds{batch, inputHeight, inputWidth, channels};
    SmallVector<int64_t, 4> steps(4, 1);
    // TODO Strip mining ?
    buildAffineLoopNest(
        rewriter, loc, lowerBounds, upperBounds, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          // Load input.
          SmallVector<OpFoldResult, 4> offset{ivs[0], ivs[1], ivs[2], ivs[3]};
          SmallVector<OpFoldResult, 4> sizes{c1, kernelHeight, kernelWidth, c1};
          SmallVector<OpFoldResult, 4> strides(4, c1);
          Value subview = rewriter.create<memref::SubViewOp>(loc, input, offset,
                                                             sizes, strides);
          // Read vector.
          Value vec = rewriter.create<vector::TransferReadOp>(
              loc, subviewVecTy, subview, ValueRange{c0, c0, c0, c0});
          // Flatten.
          Value flattenedVec =
              rewriter.create<vector::ShapeCastOp>(loc, flattenedVecTy, vec);
          // Reduce flattened vector.
          Value res = rewriter.create<vector::ReductionOp>(
              loc, vector::CombiningKind::ADD, flattenedVec);
          // Store value.
          rewriter.create<mlir::AffineStoreOp>(
              loc, res, output, outputVectorMap,
              ValueRange{ivs[0], ivs[1], ivs[2], ivs[3]});
        });

    // Remove the origin pooling operation.
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t strip;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// PoolingVectorizationPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg pooling operations to mixture of
/// Affine + Vector + Std operations.
namespace {
class PoolingVectorizationPass
    : public PassWrapper<PoolingVectorizationPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "pooling-vectorization"; }
  StringRef getDescription() const final { return "Pooling vectorization."; }
  PoolingVectorizationPass() = default;
  PoolingVectorizationPass(const PoolingVectorizationPass &) {}
  explicit PoolingVectorizationPass(int64_t stripParam) { strip = stripParam; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect, AffineDialect,
                    VectorDialect, StandardOpsDialect>();
  }

  Option<int64_t> strip{*this, "strip-mining",
                        llvm::cl::desc("Strip mining size."),
                        llvm::cl::init(32)};
};
} // end anonymous namespace.

void PoolingVectorizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithmeticDialect, AffineDialect,
                         scf::SCFDialect, StandardOpsDialect,
                         memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, FuncOp, ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<CBPoolingNhwcSumVectorizationPattern>(context, strip);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerPoolingVectorizationPass() {
  PassRegistration<PoolingVectorizationPass>();
}
} // namespace buddy
} // namespace mlir
