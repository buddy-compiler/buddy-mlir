//====- CBPoolingVectorization.cpp ----------------------------------------===//
//
// This file implements the pooling vectorization.
//
//===----------------------------------------------------------------------===//
#include <iostream>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Dialect/Vector/Transforms/VectorTransforms.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Metadata.h>

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
    // Get input, kernel and output
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    // Element type
    MemRefType inputMemRefTy = input.getType().dyn_cast<MemRefType>();
    // Element type
    FloatType fTy = inputMemRefTy.getElementType().dyn_cast<FloatType>();
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
    // Strides.
    auto strides = op->getAttrOfType<mlir::DenseIntElementsAttr>("strides")
                       .getValues<int64_t>();
    Value stridesHeight =
        rewriter.create<arith::ConstantIndexOp>(loc, strides[0]);
    Value stridesWidth =
        rewriter.create<arith::ConstantIndexOp>(loc, strides[1]);
    // Dilations.
    auto dilations = op->getAttrOfType<mlir::DenseIntElementsAttr>("dilations")
                         .getValues<int64_t>();
    // Kernel shape.
    MemRefType kernelTy = kernel.getType().dyn_cast<MemRefType>();
    SmallVector<int64_t> kernelShape;
    for (unsigned i = 0; i < kernelTy.getRank(); i++) {
      if (!kernelTy.isDynamicDim(i)) {
        kernelShape.push_back(kernelTy.getDimSize(i));
      } else {
        return failure();
      }
    }
    // Subview shape.
    SmallVector<int64_t> subviewShape{
        1, kernelShape[0] + (dilations[0] - 1) * (kernelShape[0] - 1),
        kernelShape[1] + (dilations[1] - 1) * (kernelShape[1] - 1), 1};
    Value subviewHeight =
        rewriter.create<arith::ConstantIndexOp>(loc, subviewShape[1]);
    Value subviewWidth =
        rewriter.create<arith::ConstantIndexOp>(loc, subviewShape[2]);
    // Subview vector type.
    VectorType subviewVecTy = VectorType::get(subviewShape, fTy);
    // Flattened vector type.
    SmallVector<int64_t> flattenedVecShape{1};
    for (auto &val : subviewShape)
      flattenedVecShape[0] *= val;
    VectorType flattenedVecTy = VectorType::get(flattenedVecShape, fTy);
    // Output shape.
    Value out0 = rewriter.create<arith::SubIOp>(loc, height, subviewHeight);
    Value outputHeight =
        rewriter.create<arith::AddIOp>(loc, out0, stridesHeight);
    Value out1 = rewriter.create<arith::SubIOp>(loc, width, subviewWidth);
    Value outputWidth = rewriter.create<arith::AddIOp>(loc, out1, stridesWidth);
    // Loop arguments.
    SmallVector<Value, 4> lowerBounds(4, c0);
    SmallVector<Value, 4> upperBounds{batch, outputHeight, outputWidth,
                                      channels};
    bool dilated = dilations[0] != 1 || dilations[1] != 1;
    if (dilated) {
      // Kernel width.
      Value kWidth =
          rewriter.create<arith::ConstantIndexOp>(loc, kernelShape[1]);
      // Kernel size
      int64_t kSize = kernelShape[0] * kernelShape[1];
      // Vector type
      VectorType vecTy = VectorType::get({kSize}, fTy);
      // Loop.
      SmallVector<Value, 4> steps{c1, stridesHeight, stridesWidth, c1};
      mlir::scf::buildLoopNest(
          rewriter, loc, lowerBounds, upperBounds, steps, {},
          [&](OpBuilder &builder, Location loc, ValueRange ivs,
              ValueRange args) -> scf::ValueVector {
            // Pooling window.
            Value window = rewriter.create<memref::AllocaOp>(
                loc, MemRefType::get({kSize}, fTy));
            // Dialtions.
            Value dilHeight =
                rewriter.create<arith::ConstantIndexOp>(loc, dilations[0]);
            Value dilWidth =
                rewriter.create<arith::ConstantIndexOp>(loc, dilations[1]);
            // Loop.
            SmallVector<Value, 2> nestedLowerBounds{ivs[1], ivs[2]};
            Value uHeight =
                rewriter.create<arith::AddIOp>(loc, ivs[1], subviewHeight);
            Value uWidth =
                rewriter.create<arith::AddIOp>(loc, ivs[2], subviewWidth);
            SmallVector<Value, 2> nestedUpperBounds{uHeight, uWidth};
            SmallVector<Value, 2> nestedSteps{dilHeight, dilWidth};
            mlir::scf::buildLoopNest(
                rewriter, loc, nestedLowerBounds, nestedUpperBounds,
                nestedSteps, {},
                [&](OpBuilder &nestedBuilder, Location nestedLoc,
                    ValueRange nestedIvs,
                    ValueRange nestedArgs) -> scf::ValueVector {
                  // Load value from pooling window.
                  Value val = builder.create<memref::LoadOp>(
                      loc, input,
                      ValueRange{ivs[0], nestedIvs[0], nestedIvs[1], ivs[3]});
                  // Compute index.
                  Value i0 =
                      builder.create<arith::SubIOp>(loc, nestedIvs[0], ivs[1]);
                  Value i = builder.create<arith::DivUIOp>(loc, i0, dilHeight);
                  Value j0 =
                      builder.create<arith::SubIOp>(loc, nestedIvs[1], ivs[2]);
                  Value j = builder.create<arith::DivUIOp>(loc, j0, dilWidth);
                  Value index0 = builder.create<arith::MulIOp>(loc, i, kWidth);
                  Value index = builder.create<arith::AddIOp>(loc, index0, j);
                  // Store value into memref.
                  builder.create<memref::StoreOp>(loc, val, window, index);
                  return {};
                });
            // Load into a vector.
            Value vec = rewriter.create<vector::TransferReadOp>(
                loc, vecTy, window, ValueRange{c0});
            // TODO remove
            rewriter.create<vector::PrintOp>(loc, vec);
            // Reduce vector.
            Value res = rewriter.create<vector::ReductionOp>(
                loc, vector::CombiningKind::ADD, vec);
            // Output indices.
            Value outputH =
                rewriter.create<arith::DivUIOp>(loc, ivs[1], stridesHeight);
            Value outputW =
                rewriter.create<arith::DivUIOp>(loc, ivs[2], stridesWidth);
            // Store value into output.
            rewriter.create<memref::StoreOp>(
                loc, res, output, ValueRange{ivs[0], outputH, outputW, ivs[3]});
            // Return
            return {};
          });
    } else {
      SmallVector<int64_t, 4> steps{1, strides[0], strides[1], 1};
      buildAffineLoopNest(
          rewriter, loc, lowerBounds, upperBounds, steps,
          [&](OpBuilder &builder, Location loc, ValueRange ivs) {
            // Load pooling window.
            SmallVector<OpFoldResult, 4> offset{ivs[0], ivs[1], ivs[2], ivs[3]};
            SmallVector<OpFoldResult, 4> sizes{c1, subviewHeight, subviewWidth,
                                               c1};
            SmallVector<OpFoldResult, 4> strides{c1, c1, c1, c1};
            Value subview = rewriter.create<memref::SubViewOp>(
                loc, input, offset, sizes, strides);
            // Read vector.
            Value vec = rewriter.create<vector::TransferReadOp>(
                loc, subviewVecTy, subview, ValueRange{c0, c0, c0, c0});
            // Flatten vector.
            Value flattenedVec =
                rewriter.create<vector::ShapeCastOp>(loc, flattenedVecTy, vec);
            // Reduce flattened vector.
            Value res = rewriter.create<vector::ReductionOp>(
                loc, vector::CombiningKind::ADD, flattenedVec);
            // Output indices.
            Value outputH =
                rewriter.create<arith::DivUIOp>(loc, ivs[1], stridesHeight);
            Value outputW =
                rewriter.create<arith::DivUIOp>(loc, ivs[2], stridesWidth);
            // Store value.
            rewriter.create<memref::StoreOp>(
                loc, res, output, ValueRange{ivs[0], outputH, outputW, ivs[3]});
          });
    }

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
