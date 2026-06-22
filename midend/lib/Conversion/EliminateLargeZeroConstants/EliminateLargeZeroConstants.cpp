//===- EliminateLargeZeroConstants.cpp - Replace large zero constants -----===//
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
// This pass replaces large arith.constant dense<0> operations with
// tensor.empty + linalg.fill to avoid embedding large zero constants in
// the .rodata section, which can cause linking issues on some models.
//
// Pattern matched:
//   %cst = arith.constant dense<0.0> : tensor<10x4096x4096xf32>
//
// Transformed to:
//   %empty = tensor.empty() : tensor<10x4096x4096xf32>
//   %zero = arith.constant 0.0 : f32
//   %cst = linalg.fill ins(%zero : f32) outs(%empty : tensor<10x4096x4096xf32>)
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

// Check if a DenseElementsAttr is a splat of zero
static bool isZeroSplat(DenseElementsAttr attr) {
  if (!attr.isSplat())
    return false;

  auto elemType = attr.getElementType();
  if (isa<FloatType>(elemType) || isa<BFloat16Type>(elemType)) {
    return attr.getSplatValue<APFloat>().isZero();
  } else if (isa<IntegerType>(elemType)) {
    return attr.getSplatValue<APInt>().isZero();
  }
  return false;
}

// Calculate tensor size in bytes
static int64_t calculateTensorSizeBytes(RankedTensorType tensorType) {
  if (!tensorType.hasStaticShape())
    return -1;

  int64_t numElements = tensorType.getNumElements();
  auto elemType = tensorType.getElementType();

  int64_t elementSize = 0;
  if (auto floatType = dyn_cast<FloatType>(elemType)) {
    elementSize = floatType.getWidth() / 8;
  } else if (isa<BFloat16Type>(elemType)) {
    elementSize = 2;
  } else if (auto intType = dyn_cast<IntegerType>(elemType)) {
    elementSize = intType.getWidth() / 8;
  } else {
    return -1;
  }

  return numElements * elementSize;
}

// Pattern to replace large zero constants with tensor.empty + linalg.fill
struct ReplaceLargeZeroConstant
    : public OpRewritePattern<arith::ConstantOp> {
  int64_t sizeThresholdBytes;

  ReplaceLargeZeroConstant(MLIRContext *context, int64_t threshold = 1024 * 1024)
      : OpRewritePattern<arith::ConstantOp>(context),
        sizeThresholdBytes(threshold) {}

  LogicalResult matchAndRewrite(arith::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto tensorType = dyn_cast<RankedTensorType>(op.getType());
    if (!tensorType)
      return failure();

    auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!denseAttr)
      return failure();

    if (!isZeroSplat(denseAttr))
      return failure();

    int64_t sizeBytes = calculateTensorSizeBytes(tensorType);
    if (sizeBytes < 0 || sizeBytes < sizeThresholdBytes)
      return failure();

    Location loc = op.getLoc();
    auto elemType = tensorType.getElementType();

    SmallVector<Value> dynamicSizes; 
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, tensorType.getShape(), elemType, dynamicSizes);

    TypedAttr zeroAttr;
    if (isa<FloatType>(elemType) || isa<BFloat16Type>(elemType)) {
      zeroAttr = rewriter.getFloatAttr(elemType, 0.0);
    } else if (isa<IntegerType>(elemType)) {
      zeroAttr = rewriter.getIntegerAttr(elemType, 0);
    } else {
      return failure();
    }
    Value zeroScalar = rewriter.create<arith::ConstantOp>(loc, zeroAttr);

    Value filledTensor = rewriter.create<linalg::FillOp>(
        loc, zeroScalar, emptyTensor).getResult(0);

    rewriter.replaceOp(op, filledTensor);
    return success();
  }
};

class EliminateLargeZeroConstantsPass
    : public PassWrapper<EliminateLargeZeroConstantsPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EliminateLargeZeroConstantsPass)

  EliminateLargeZeroConstantsPass() = default;
  EliminateLargeZeroConstantsPass(const EliminateLargeZeroConstantsPass &) {}

  StringRef getArgument() const final {
    return "eliminate-large-zero-constants";
  }
  StringRef getDescription() const final {
    return "Replace large arith.constant dense<0> with tensor.empty + "
           "linalg.fill";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = func.getContext();

    RewritePatternSet patterns(ctx);
    // Currently, we use 1 MB threshold
    patterns.add<ReplaceLargeZeroConstant>(ctx, 1024 * 1024);

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);

    (void)applyPatternsGreedily(func, std::move(patterns), config);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, tensor::TensorDialect,
                    linalg::LinalgDialect>();
  }
};

} // namespace

namespace mlir {
namespace buddy {
void registerEliminateLargeZeroConstantsPass() {
  PassRegistration<EliminateLargeZeroConstantsPass>();
}
} // namespace buddy
} // namespace mlir
