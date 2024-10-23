//===--------PoolingNhwcMaxAdaptVectorization.cpp--------------------------===//
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
// This file implements the Pooling Nhwc Max Adapt Vectorization and suitable
// for use when the size of the C dimension is small.
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

class PoolingNhwcMaxAdaptVectorizationPattern : public ConversionPattern {
public:
  explicit PoolingNhwcMaxAdaptVectorizationPattern(MLIRContext *context)
      : ConversionPattern(linalg::PoolingNhwcMaxOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Get input, kernel and output.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);

    // Get Constants.
    const Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    const Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    const Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    // Get Dimensions of Kernel.
    Value kernelHeight = rewriter.create<memref::DimOp>(loc, kernel, c0);
    Value kernelWidth = rewriter.create<memref::DimOp>(loc, kernel, c1);

    // Get Dimensions of Outputs.
    Value batch = rewriter.create<memref::DimOp>(loc, output, c0);
    Value height = rewriter.create<memref::DimOp>(loc, output, c1);
    Value width = rewriter.create<memref::DimOp>(loc, output, c2);

    // Get ElementType of input and create pass through vector.
    ShapedType inputTy = input.getType().cast<ShapedType>();
    Type elementTy = inputTy.getElementType();

    // get size of input c-dimensions.
    int64_t c_dim = inputTy.getShape()[3];
    VectorType vectorTy = mlir::VectorType::get({c_dim}, elementTy);

    // Define AffineMap of inputVector.
    AffineExpr input0, input1, input2, input3, input4, input5;
    bindDims(ctx, input0, input1, input2, input3, input4, input5);
    AffineMap inputVectorMap = AffineMap::get(
        /*dimCount=*/6, /*symbolCount=*/0,
        {input0, input1 * 2 + input4, input2 * 2 + input5, input3 * 0}, ctx);

    // Define AffineMap of outputVector.
    AffineExpr output0, output1, output2, output3;
    bindDims(ctx, output0, output1, output2, output3);
    AffineMap outputVectorMap = AffineMap::get(
        /*dimCount=*/4, /*symbolCount=*/0,
        {output0, output1, output2, output3 * 0}, ctx);

    // Create loop nest in NHoWo-order.
    SmallVector<Value, 8> lowerBounds(3, c0);
    SmallVector<Value, 8> uperBounds{batch, height, width};
    SmallVector<int64_t, 8> steps(3, /*Value=*/1);
    affine::buildAffineLoopNest(
        rewriter, loc, lowerBounds, uperBounds, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          Value outputVector = builder.create<affine::AffineVectorLoadOp>(
              loc, vectorTy, output, outputVectorMap,
              ValueRange{ivs[0], ivs[1], ivs[2], c0});

          auto tmp0 = builder.create<affine::AffineForOp>(
              loc, ValueRange{c0}, builder.getDimIdentityMap(),
              ValueRange{kernelHeight}, builder.getDimIdentityMap(),
              /*Step=*/1, ValueRange{outputVector},
              [&](OpBuilder &builder, Location loc, Value iv0,
                  ValueRange itrArgs0) {
                auto tmp1 = builder.create<affine::AffineForOp>(
                    loc, ValueRange{c0}, builder.getDimIdentityMap(),
                    ValueRange{kernelWidth}, builder.getDimIdentityMap(),
                    /*Step=*/1, ValueRange{itrArgs0[0]},
                    [&](OpBuilder &builder, Location loc, Value iv1,
                        ValueRange itrArgs1) {
                      Value inputVector =
                          builder.create<affine::AffineVectorLoadOp>(
                              loc, vectorTy, input, inputVectorMap,
                              ValueRange{ivs[0], ivs[1], ivs[2], c0, iv0, iv1});
                      // Max
                      Value resultVector;
                      if (auto ty = llvm::dyn_cast<IntegerType>(elementTy)) {
                        resultVector = builder.create<arith::MaxSIOp>(
                            loc, inputVector, itrArgs1[0]);
                      } else {
                        resultVector = builder.create<arith::MaximumFOp>(
                            loc, inputVector, itrArgs1[0]);
                      }
                      builder.create<affine::AffineYieldOp>(loc, resultVector);
                    });
                builder.create<affine::AffineYieldOp>(loc, tmp1.getResult(0));
              });
          builder.create<affine::AffineVectorStoreOp>(
              loc, tmp0.getResult(0), output, outputVectorMap,
              ValueRange{ivs[0], ivs[1], ivs[2], c0});
        });
    // Remove the origin convolution operation.
    rewriter.eraseOp(op);
    return success();
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// PoolingNhwcMaxAdaptVectorizationPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg pooling max operations to mixture of
/// Arith + Vector operations.
namespace {
class PoolingNhwcMaxAdaptVectorizationPass
    : public PassWrapper<PoolingNhwcMaxAdaptVectorizationPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      PoolingNhwcMaxAdaptVectorizationPass)
  StringRef getArgument() const final {
    return "pooling-nhwc-max-adapt-vectorization";
  }
  StringRef getDescription() const final {
    return "Pooling_Nhwc_Max Adapt Vectorization.";
  }
  PoolingNhwcMaxAdaptVectorizationPass() = default;
  PoolingNhwcMaxAdaptVectorizationPass(
      const PoolingNhwcMaxAdaptVectorizationPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, VectorDialect, func::FuncDialect>();
  }
};
} // end anonymous namespace.

void PoolingNhwcMaxAdaptVectorizationPass::runOnOperation() {
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
  patterns.add<PoolingNhwcMaxAdaptVectorizationPattern>(context);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerPoolingNhwcMaxAdaptVectorizationPass() {
  PassRegistration<PoolingNhwcMaxAdaptVectorizationPass>();
}
} // namespace buddy
} // namespace mlir
