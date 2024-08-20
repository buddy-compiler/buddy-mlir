//===--------PoolingNhwcMaxVectorization.cpp-------------------------------===//
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
// This file implements the Pooling Nhwc Max Vectorization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/ArrayRef.h"
#include <mlir/Dialect/Affine/Analysis/AffineAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Vector/Transforms/VectorTransforms.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

#include "Utils/Utils.h"

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class PoolingNhwcMaxVectorizationPattern : public ConversionPattern {
public:
  explicit PoolingNhwcMaxVectorizationPattern(MLIRContext *context,
                                              int64_t stripParam)
      : ConversionPattern(linalg::PoolingNhwcMaxOp::getOperationName(), 1,
                          context) {
    strip = stripParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Get f32 as the element type.
    FloatType f32 = mlir::FloatType::getF32(ctx);
    // Get i1 as the element type for mask vector.
    IntegerType i1 = IntegerType::get(ctx, 1);

    // Define `*Type`.
    VectorType vectorTy32 = mlir::VectorType::get({strip}, f32);
    VectorType vectorMaskTy = mlir::VectorType::get({strip}, i1);

    // Get input, kernel and output.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);

    // Get Constants.
    const Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    const Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    const Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    const Value c3 = rewriter.create<arith::ConstantIndexOp>(loc, 3);
    const Value c32 = rewriter.create<arith::ConstantIndexOp>(loc, strip);
    Value f0 = rewriter.create<arith::ConstantFloatOp>(
        loc, APFloat::getZero(f32.getFloatSemantics()), f32);

    // Create pass through vector.
    Value passThroughVec = rewriter.create<SplatOp>(loc, vectorTy32, f0);

    // TODO: add to get Strides and Dilations.

    // Get Dimensions of Kernel.
    Value kernelHeight = rewriter.create<memref::DimOp>(loc, kernel, c0);
    Value kernelWidth = rewriter.create<memref::DimOp>(loc, kernel, c1);

    // Get Dimensions of Outputs.
    Value batch = rewriter.create<memref::DimOp>(loc, output, c0);
    Value height = rewriter.create<memref::DimOp>(loc, output, c1);
    Value width = rewriter.create<memref::DimOp>(loc, output, c2);
    Value channels = rewriter.create<memref::DimOp>(loc, output, c3);

    // Size of strip mining.
    AffineExpr d0;
    bindDims(ctx, d0);
    AffineMap stripMap = AffineMap::get(1, 0, {d0.ceilDiv(strip)}, ctx);
    SmallVector<Value, 8> lowerBounds(5, c0);
    SmallVector<Value, 8> uperBounds{batch, channels, height, kernelHeight,
                                     kernelWidth};
    SmallVector<int64_t, 8> steps(5, /*Value=*/1);
    affine::buildAffineLoopNest(
        rewriter, loc, lowerBounds, uperBounds, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          // Create strip mining loop.
          builder.create<affine::AffineForOp>(
              loc, ValueRange{c0}, builder.getDimIdentityMap(),
              ValueRange{width}, stripMap, /*Step=*/1, std::nullopt,
              [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
                  ValueRange itrArgs) {
                // Vectorize the kernel.

                // Load input vector from memref.
                AffineExpr input0, input1, input2, input3, input4, input5;
                bindDims(ctx, input0, input1, input2, input3, input4, input5);
                AffineMap inputVectorMap = AffineMap::get(
                    /*dimCount=*/6, /*symbolCount=*/0,
                    {input0, input3 + input2, input4 + input5 * strip, input1},
                    ctx);

                // Calculate the tail.
                Value currWidth =
                    nestedBuilder.create<arith::MulIOp>(loc, iv, c32);
                Value tail =
                    nestedBuilder.create<arith::SubIOp>(loc, width, currWidth);
                Value tailCond = rewriter.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::sge, tail, c32);

                // If the current column does not reach the tail.
                builder.create<scf::IfOp>(
                    loc, tailCond,
                    [&](OpBuilder &builder, Location loc) {
                      Value inputVector =
                          nestedBuilder.create<affine::AffineVectorLoadOp>(
                              loc, vectorTy32, input, inputVectorMap,
                              ValueRange{ivs[0], ivs[1], ivs[2], ivs[3], ivs[4],
                                         iv});
                      // Define AffineMap.
                      // The `outputVector` and `resultVector` share the
                      // same AffineMap.
                      AffineExpr output0, output1, output2, output3;
                      bindDims(ctx, output0, output1, output2, output3);
                      AffineMap outputVectorMap = AffineMap::get(
                          /*dimCount=*/4, /*symbolCount=*/0,
                          {output0, output2, output3 * strip, output1}, ctx);
                      Value outputVector =
                          nestedBuilder.create<affine::AffineVectorLoadOp>(
                              loc, vectorTy32, output, outputVectorMap,
                              ValueRange{ivs[0], ivs[1], ivs[2], iv});
                      // FMA = Fused Multiply + Add
                      Value resultVector =
                          nestedBuilder.create<arith::MaximumFOp>(
                              loc, inputVector, outputVector);
                      nestedBuilder.create<affine::AffineVectorStoreOp>(
                          loc, resultVector, output, outputVectorMap,
                          ValueRange{ivs[0], ivs[1], ivs[2], iv});
                      builder.create<scf::YieldOp>(loc);
                    },
                    // The else branch (the current column reaches the
                    // tail).
                    [&](OpBuilder &builder, Location loc) {
                      // Create mask according to the tail.
                      Value tailMask =
                          builder.create<CreateMaskOp>(loc, vectorMaskTy, tail);
                      // Calculate the index of the input and output.
                      Value inputHeight = nestedBuilder.create<arith::AddIOp>(
                          loc, ivs[2], ivs[3]);
                      Value outputWidth =
                          nestedBuilder.create<arith::MulIOp>(loc, iv, c32);
                      Value inputWidth = nestedBuilder.create<arith::AddIOp>(
                          loc, ivs[4], outputWidth);
                      // Masked load input and output.
                      Value maskedInputVec = builder.create<MaskedLoadOp>(
                          loc, vectorTy32, input,
                          ValueRange{ivs[0], inputHeight, inputWidth, ivs[1]},
                          tailMask, passThroughVec);
                      Value maskedOutputVec = builder.create<MaskedLoadOp>(
                          loc, vectorTy32, output,
                          ValueRange{ivs[0], ivs[2], outputWidth, ivs[1]},
                          tailMask, passThroughVec);
                      // FMA.
                      Value resultVec = builder.create<arith::MaximumFOp>(
                          loc, maskedInputVec, maskedOutputVec);
                      // Masked store the result to output.
                      builder.create<MaskedStoreOp>(
                          loc, output,
                          ValueRange{ivs[0], ivs[2], outputWidth, ivs[1]},
                          tailMask, resultVec);
                      builder.create<scf::YieldOp>(loc);
                    });
                nestedBuilder.create<affine::AffineYieldOp>(nestedLoc);
              });
        });
    // Remove the origin convolution operation.
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t strip;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// PoolingNhwcMaxVectorizationPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg pooling max operations to mixture of
/// Arith + Vector operations.
namespace {
class PoolingNhwcMaxVectorizationPass
    : public PassWrapper<PoolingNhwcMaxVectorizationPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PoolingNhwcMaxVectorizationPass)
  StringRef getArgument() const final {
    return "pooling-nhwc-max-vectorization";
  }
  StringRef getDescription() const final {
    return "Pooling_Nhwc_Max vectorization.";
  }
  PoolingNhwcMaxVectorizationPass() = default;
  PoolingNhwcMaxVectorizationPass(const PoolingNhwcMaxVectorizationPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, VectorDialect, func::FuncDialect>();
  }
  Option<int64_t> strip{*this, "strip-mining",
                        llvm::cl::desc("Strip mining size."),
                        llvm::cl::init(32)};
};
} // end anonymous namespace.

void PoolingNhwcMaxVectorizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                         scf::SCFDialect, func::FuncDialect,
                         memref::MemRefDialect, VectorDialect,
                         math::MathDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<PoolingNhwcMaxVectorizationPattern>(context, strip);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerPoolingNhwcMaxVectorizationPass() {
  PassRegistration<PoolingNhwcMaxVectorizationPass>();
}
} // namespace buddy
} // namespace mlir
