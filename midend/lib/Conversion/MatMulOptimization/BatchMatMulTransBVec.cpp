//===- BatchMatMulOptimize.cpp --------------------------------------------===//
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
// This file implements the batchmatmul optimization.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <mlir/Dialect/Affine/Analysis/AffineAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/Transforms/VectorTransforms.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class BatchMatMulTransVecPattern : public ConversionPattern {
public:
  explicit BatchMatMulTransVecPattern(MLIRContext *context,
                                      int64_t vecSizeParam)
      : ConversionPattern(linalg::BatchMatmulTransposeBOp::getOperationName(),
                          1, context) {
    vecSize = vecSizeParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Retrieve input tensors A, B, and C.
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);

    // Get i1 as the element type for mask vector.
    IntegerType i1 = IntegerType::get(ctx, 1);
    VectorType vectorMaskTy = mlir::VectorType::get({vecSize}, i1);
    // Acquire the element type of input tensors.
    Type elementType = A.getType().cast<MemRefType>().getElementType();
    VectorType vectorTy = mlir::VectorType::get({vecSize}, elementType);

    // Define constants.
    const Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    const Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    const Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    const Value vlStep = rewriter.create<arith::ConstantIndexOp>(loc, vecSize);
    const Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));

    // Create pass through vector.
    Value passThroughVec = rewriter.create<SplatOp>(loc, vectorTy, zero);

    // Get dimensions of input tensors.
    Value batch = rewriter.create<memref::DimOp>(loc, A, c0);
    Value aRow = rewriter.create<memref::DimOp>(loc, A, c1);
    Value bRow = rewriter.create<memref::DimOp>(loc, A, c2);
    Value bCol = rewriter.create<memref::DimOp>(loc, B, c1);

    // Calculate the upper bound for vectorized processing
    // - Subtract `vlStep` is to avoid overflow at the vectorization tail.
    // - Add 1 to ensure the final loop runs when the workload length
    //   is divisible by the vector size.
    Value upperBoundTmp = rewriter.create<arith::SubIOp>(loc, bRow, vlStep);
    Value upperBound = rewriter.create<arith::AddIOp>(loc, upperBoundTmp, c1);

    affine::buildAffineLoopNest(
        rewriter, loc, {c0, c0, c0}, {batch, aRow, bCol}, /*Step=*/{1, 1, 1},
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          // Get a vector of the output memref.
          Value val = builder.create<memref::LoadOp>(
              loc, elementType, C, ValueRange{ivs[0], ivs[1], ivs[2]});
          auto iterValues = builder.create<scf::ForOp>(
              loc, c0, upperBound, /*Step=*/vlStep, ValueRange{c0, val},
              [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
                  ValueRange itrArgs) {
                Value aVec = builder.create<vector::LoadOp>(
                    loc, vectorTy, A, ValueRange{ivs[0], ivs[1], iv});
                Value bVec = builder.create<vector::LoadOp>(
                    loc, vectorTy, B, ValueRange{ivs[0], ivs[2], iv});
                // Compute the result vector either through integer
                // multiplication and addition or fused multiply-add
                // based on the element type.
                Value tmpVec;
                if (isa<IntegerType>(elementType)) {
                  tmpVec = builder.create<arith::MulIOp>(loc, aVec, bVec);
                } else {
                  tmpVec = builder.create<arith::MulFOp>(loc, aVec, bVec);
                }
                Value tmpVal = builder.create<vector::ReductionOp>(
                    loc, vector::CombiningKind::ADD, tmpVec, itrArgs[1],
                    ::mlir::arith::FastMathFlags::reassoc);
                Value idx =
                    nestedBuilder.create<arith::AddIOp>(nestedLoc, iv, vlStep);
                builder.create<scf::YieldOp>(loc, ValueRange{idx, tmpVal});
              });
          // Compute the tail size and Process the remaining elements
          // using masked vector operations.
          Value idx = iterValues.getResult(0);
          Value tailSize = builder.create<arith::SubIOp>(loc, bRow, idx);
          Value tailMask =
              builder.create<CreateMaskOp>(loc, vectorMaskTy, tailSize);
          Value maskedAVec = builder.create<MaskedLoadOp>(
              loc, vectorTy, A, ValueRange{ivs[0], ivs[1], idx}, tailMask,
              passThroughVec);
          Value maskedBVec = builder.create<MaskedLoadOp>(
              loc, vectorTy, B, ValueRange{ivs[0], ivs[2], idx}, tailMask,
              passThroughVec);
          // Compute the result vector either through integer
          // multiplication and addition or fused multiply-add
          // based on the element type.
          Value tmpVec;
          if (isa<IntegerType>(elementType)) {
            tmpVec = builder.create<arith::MulIOp>(loc, maskedAVec, maskedBVec);
          } else {
            tmpVec = builder.create<arith::MulFOp>(loc, maskedAVec, maskedBVec);
          }
          Value tmpVal = builder.create<vector::ReductionOp>(
              loc, vector::CombiningKind::ADD, tmpVec, iterValues.getResult(1),
              ::mlir::arith::FastMathFlags::reassoc);
          builder.create<affine::AffineStoreOp>(
              loc, tmpVal, C, ValueRange{ivs[0], ivs[1], ivs[2]});
        });
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t vecSize;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// BatchMatMulTransVecPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg pooling operations to mixture of
/// Affine + Vector operations.
namespace {
class BatchMatMulTransVecPass
    : public PassWrapper<BatchMatMulTransVecPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BatchMatMulTransVecPass)
  StringRef getArgument() const final {
    return "batchmatmul-transpose-b-vectorization";
  }
  StringRef getDescription() const final {
    return "BatchMatMulTransposeBOp vectorization.";
  }
  BatchMatMulTransVecPass() = default;
  BatchMatMulTransVecPass(const BatchMatMulTransVecPass &) {}
  explicit BatchMatMulTransVecPass(int64_t vecSizeParam) {
    vecSize = vecSizeParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, VectorDialect>();
  }

  Option<int64_t> vecSize{*this, "vector-size",
                          llvm::cl::desc("Affine Vector size."),
                          llvm::cl::init(32)};
};
} // end anonymous namespace.

void BatchMatMulTransVecPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                       scf::SCFDialect, memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<BatchMatMulTransVecPattern>(context, vecSize);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerBatchMatMulTransVecPass() {
  PassRegistration<BatchMatMulTransVecPass>();
}
} // namespace buddy
} // namespace mlir
