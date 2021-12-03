//====- CBConvVectorization.cpp - Coefficients Broadcasting Algorithm -----===//
//
// This file implements the coefficients broadcasting algorthm (CB) for
// convolution vectorization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
class CBConvVectorizationPattern : public ConversionPattern {
public:
  explicit CBConvVectorizationPattern(MLIRContext *context, int64_t strideParam)
      : ConversionPattern(linalg::Conv2DOp::getOperationName(), 1, context) {
    stride = strideParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();
    // Currently use f32 as the element type.
    // TODO: replace f32 with input type.
    FloatType f32 = mlir::FloatType::getF32(ctx);
    // Get i1 as the element type for mask vector.
    IntegerType i1 = IntegerType::get(ctx, 1);
    // Define `*Type`.
    VectorType vectorTy1 = mlir::VectorType::get({1}, f32);
    VectorType vectorTy32 = mlir::VectorType::get({stride}, f32);
    VectorType vectorMaskTy = VectorType::get({stride}, i1);
    // Create constant index.
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value cStride = rewriter.create<arith::ConstantIndexOp>(loc, stride);
    Value f0 = rewriter.create<arith::ConstantFloatOp>(
        loc, APFloat::getZero(f32.getFloatSemantics()), f32);
    // Create pass through vector.
    Value passThroughVec = rewriter.create<SplatOp>(loc, vectorTy32, f0);
    // Get input, kernel and output.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    // Create DimOp.
    Value kernelRow = rewriter.create<memref::DimOp>(loc, kernel, c0);
    Value kernelCol = rewriter.create<memref::DimOp>(loc, kernel, c1);
    Value outputRow = rewriter.create<memref::DimOp>(loc, output, c0);
    Value outputCol = rewriter.create<memref::DimOp>(loc, output, c1);
    // Size of strip mining.
    AffineExpr d0;
    bindDims(ctx, d0);
    AffineMap stripMap = AffineMap::get(1, 0, {d0.ceilDiv(stride)}, ctx);
    SmallVector<Value, 8> lowerBounds(3, c0);
    SmallVector<Value, 8> uperBounds{outputRow, kernelRow, kernelCol};
    SmallVector<int64_t, 8> steps(3, /*Value=*/1);
    buildAffineLoopNest(
        rewriter, loc, lowerBounds, uperBounds, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          // Create strip mining loop.
          builder.create<AffineForOp>(
              loc, ValueRange{c0}, builder.getDimIdentityMap(),
              ValueRange{outputCol}, stripMap, /*Step=*/1, llvm::None,
              [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
                  ValueRange itrArgs) {
                // Vectorize the kernel.
                // Broadcast element of the kernel.
                Value kernelValue = builder.create<AffineVectorLoadOp>(
                    loc, vectorTy1, kernel, ValueRange{ivs[1], ivs[2]});
                Value kernelVector =
                    builder.create<BroadcastOp>(loc, vectorTy32, kernelValue);
                // Load input vector from memref.
                AffineExpr m, n, k, j;
                bindDims(ctx, m, n, k, j);
                AffineMap inputVectorMap = AffineMap::get(
                    /*dimCount=*/4, /*symbolCount=*/0, {m + n, k + j * stride},
                    ctx);
                // Calculate the tail.
                Value currCol =
                    nestedBuilder.create<arith::MulIOp>(loc, iv, cStride);
                Value tail = nestedBuilder.create<arith::SubIOp>(loc, outputCol,
                                                                 currCol);
                Value tailCond = rewriter.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::sge, tail, cStride);
                // If the current column does not reach the tail.
                builder.create<scf::IfOp>(
                    loc, tailCond,
                    [&](OpBuilder &builder, Location loc) {
                      Value inputVector =
                          nestedBuilder.create<AffineVectorLoadOp>(
                              loc, vectorTy32, input, inputVectorMap,
                              ValueRange{ivs[0], ivs[1], ivs[2], iv});
                      // Define AffineMap.
                      // The `outputVector` and `resultVector` share the same
                      // AffineMap.
                      AffineExpr x, y;
                      bindDims(ctx, x, y);
                      AffineMap outputVectorMap = AffineMap::get(
                          /*dimCount=*/2, /*symbolCount=*/0, {x, y * stride},
                          ctx);
                      Value outputVector =
                          nestedBuilder.create<AffineVectorLoadOp>(
                              loc, vectorTy32, output, outputVectorMap,
                              ValueRange{ivs[0], iv});
                      // FMA = Fused Multiply + Add
                      Value resultVector = nestedBuilder.create<FMAOp>(
                          loc, inputVector, kernelVector, outputVector);
                      nestedBuilder.create<AffineVectorStoreOp>(
                          loc, resultVector, output, outputVectorMap,
                          ValueRange{ivs[0], iv});
                      builder.create<scf::YieldOp>(loc);
                    },
                    // The else branch (the current column reaches the tail).
                    [&](OpBuilder &builder, Location loc) {
                      // Create mask according to the tail.
                      Value tailMask =
                          builder.create<CreateMaskOp>(loc, vectorMaskTy, tail);
                      // Calculate the index of the input and output.
                      Value inputRow = nestedBuilder.create<arith::AddIOp>(
                          loc, ivs[0], ivs[1]);
                      Value outputCol =
                          nestedBuilder.create<arith::MulIOp>(loc, iv, cStride);
                      Value inputCol = nestedBuilder.create<arith::AddIOp>(
                          loc, ivs[2], outputCol);
                      // Masked load input and output.
                      Value maskedInputVec = builder.create<MaskedLoadOp>(
                          loc, vectorTy32, input,
                          ValueRange{inputRow, inputCol}, tailMask,
                          passThroughVec);
                      Value maskedOutputVec = builder.create<MaskedLoadOp>(
                          loc, vectorTy32, output,
                          ValueRange{ivs[0], outputCol}, tailMask,
                          passThroughVec);
                      // FMA.
                      Value resultVec = builder.create<FMAOp>(
                          loc, maskedInputVec, kernelVector, maskedOutputVec);
                      // Masked store the result to output.
                      builder.create<MaskedStoreOp>(
                          loc, output, ValueRange{ivs[0], outputCol}, tailMask,
                          resultVec);
                      builder.create<scf::YieldOp>(loc);
                    });
                nestedBuilder.create<AffineYieldOp>(nestedLoc);
              });
        });
    // Remove the origin convolution operation.
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ConvVectorizationPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg convolution to mixture of
/// Affine + Vector + Std operations.
namespace {
class ConvVectorizationPass
    : public PassWrapper<ConvVectorizationPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "conv-vectorization"; }
  StringRef getDescription() const final {
    return "Convolution vectorization.";
  }
  ConvVectorizationPass() = default;
  ConvVectorizationPass(const ConvVectorizationPass &) {}
  explicit ConvVectorizationPass(int64_t strideParam) { stride = strideParam; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect, AffineDialect,
                    VectorDialect, StandardOpsDialect>();
  }

  Option<int64_t> stride{*this, "strip-mining",
                         llvm::cl::desc("Strip mining size."),
                         llvm::cl::init(32)};
};
} // end anonymous namespace.

void ConvVectorizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithmeticDialect, AffineDialect,
                         scf::SCFDialect, StandardOpsDialect,
                         memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, FuncOp, ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<CBConvVectorizationPattern>(context, stride);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerConvVectorizationPass() {
  PassRegistration<ConvVectorizationPass>();
}
} // namespace buddy
} // namespace mlir
