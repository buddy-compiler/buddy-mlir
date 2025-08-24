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
using namespace affine;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class BatchMatMulOptimizePattern : public ConversionPattern {
public:
  explicit BatchMatMulOptimizePattern(MLIRContext *context,
                                      int64_t vecSizeParam)
      : ConversionPattern(linalg::BatchMatmulOp::getOperationName(), 1,
                          context) {
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
    Type elementType = llvm::cast<MemRefType>(A.getType()).getElementType();
    VectorType vectorTy = mlir::VectorType::get({vecSize}, elementType);

    const AffineExpr d0 = rewriter.getAffineDimExpr(0);
    const AffineExpr d1 = rewriter.getAffineDimExpr(1);
    const AffineExpr d2 = rewriter.getAffineDimExpr(2);

    // Define constants.
    const Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    const Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    const Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    const Value vlStep = rewriter.create<arith::ConstantIndexOp>(loc, vecSize);
    const Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    const AffineExpr zeroAffine = rewriter.getAffineConstantExpr(0);

    // Create pass through vector.
    Value passThroughVec = rewriter.create<SplatOp>(loc, vectorTy, zero);

    // Get dimensions of input tensors.
    Value batch = rewriter.create<memref::DimOp>(loc, A, c0);
    Value aRow = rewriter.create<memref::DimOp>(loc, A, c1);
    Value bCol = rewriter.create<memref::DimOp>(loc, B, c2);
    Value bRow = rewriter.create<memref::DimOp>(loc, B, c1);

    // Calculate the upper bound for vectorized processing
    // - Subtract `vlStep` is to avoid overflow at the vectorization tail.
    // - Add 1 to ensure the final loop runs when the workload length
    //   is divisible by the vector size.
    Value upperBound_tmp = rewriter.create<arith::SubIOp>(loc, bCol, vlStep);
    Value upperBound = rewriter.create<arith::AddIOp>(loc, upperBound_tmp, c1);

    // Calculate the length of the tail, which might not fit in a vector.
    Value tailSize = rewriter.create<affine::AffineApplyOp>(
        loc, AffineMap::get(1, 0, d0 % vecSize), ValueRange{bCol});

    // Generate a mask vector based on the tail length.
    Value tailMask = rewriter.create<CreateMaskOp>(loc, vectorMaskTy, tailSize);

    SmallVector<Value, 4U> reducedValues = llvm::to_vector<4>(
        llvm::map_range(ArrayRef<LoopReduction>{},
                        [](const LoopReduction &red) { return red.value; }));

    // Create the primary parallel batch level loop.
    AffineParallelOp parallelBatchLoop =
        rewriter.create<affine::AffineParallelOp>(
            loc, ValueRange(reducedValues).getTypes(), ValueRange{batch},
            ArrayRef<NamedAttribute>{
                rewriter.getNamedAttr("lowerBoundsGroups",
                                      rewriter.getI32TensorAttr({1})),
                rewriter.getNamedAttr("upperBoundsGroups",
                                      rewriter.getI32TensorAttr({1})),
                rewriter.getNamedAttr(
                    "lowerBoundsMap",
                    AffineMapAttr::get(AffineMap::get(0, 0, {zeroAffine},
                                                      rewriter.getContext()))),
                rewriter.getNamedAttr("upperBoundsMap",
                                      AffineMapAttr::get(AffineMap::get(
                                          1, 0, {d0}, rewriter.getContext()))),
                rewriter.getNamedAttr("reductions", rewriter.getArrayAttr({})),
                rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr({1}))});

    // Create the loop body for the parallel loop.
    Block *loopBody = new Block();
    rewriter.setInsertionPointToStart(loopBody);
    loopBody->addArgument(rewriter.getIndexType(), loc);
    Value loopVarBatchIdx = loopBody->getArguments()[0];

    // Prefetching data from tensor 'A' for better cache utilization.
    rewriter.create<affine::AffinePrefetchOp>(
        loc, A, AffineMap::get(3, 0, {d0, d1, d2}, ctx),
        ArrayRef<Value>{loopVarBatchIdx, aRow, bRow}, false, 3, true);
    auto iter_idx = rewriter.create<affine::AffineForOp>(
        loc, ValueRange{c0}, rewriter.getDimIdentityMap(),
        ValueRange{upperBound}, rewriter.getDimIdentityMap(),
        /*Step=*/vecSize, ValueRange{c0},
        [&](OpBuilder &builder, Location loc, Value iv1, ValueRange itrArgs0) {
          builder.create<affine::AffineForOp>(
              loc, ValueRange{c0}, builder.getDimIdentityMap(),
              ValueRange{bRow}, builder.getDimIdentityMap(),
              /*Step=*/1, std::nullopt,
              [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv2,
                  ValueRange itrArgs0) {
                Value bVec = builder.create<affine::AffineVectorLoadOp>(
                    loc, vectorTy, B, ValueRange{loopVarBatchIdx, iv2, iv1});
                nestedBuilder.create<affine::AffineForOp>(
                    nestedLoc, ValueRange{c0}, builder.getDimIdentityMap(),
                    ValueRange{aRow}, builder.getDimIdentityMap(), /*Step=*/1,
                    std::nullopt,
                    [&](OpBuilder &builder, Location loc, Value iv3,
                        ValueRange itrArgs1) {
                      Value aValue = builder.create<memref::LoadOp>(
                          loc, elementType, A,
                          ValueRange{loopVarBatchIdx, iv3, iv2});
                      Value aVec = builder.create<vector::BroadcastOp>(
                          loc, vectorTy, aValue);
                      Value cVec = builder.create<affine::AffineVectorLoadOp>(
                          loc, vectorTy, C,
                          ValueRange{loopVarBatchIdx, iv3, iv1});
                      // Compute the result vector either through integer
                      // multiplication and addition or fused multiply-add
                      // based on the element type.
                      Value computedVec;
                      if (isa<IntegerType>(elementType)) {
                        Value mulVec =
                            builder.create<arith::MulIOp>(loc, aVec, bVec);
                        computedVec =
                            builder.create<arith::AddIOp>(loc, mulVec, cVec);
                      } else {
                        computedVec = builder.create<vector::FMAOp>(loc, aVec,
                                                                    bVec, cVec);
                      }
                      builder.create<affine::AffineVectorStoreOp>(
                          loc, computedVec, C,
                          ValueRange{loopVarBatchIdx, iv3, iv1});
                      builder.create<affine::AffineYieldOp>(loc);
                    });
                builder.create<affine::AffineYieldOp>(loc);
              });
          Value idx = builder.create<arith::AddIOp>(loc, iv1, vlStep);
          builder.create<affine::AffineYieldOp>(loc, idx);
        });
    // Compute the tail size and Process the remaining elements
    // using masked vector operations.
    Value idx = iter_idx.getResult(0);
    rewriter.create<affine::AffineForOp>(
        loc, ValueRange{c0}, rewriter.getDimIdentityMap(), ValueRange{bRow},
        rewriter.getDimIdentityMap(),
        /*Step=*/1, std::nullopt,
        [&](OpBuilder &builder, Location loc, Value iv1, ValueRange itrArgs0) {
          Value maskedBVec = builder.create<vector::MaskedLoadOp>(
              loc, vectorTy, B, ValueRange{loopVarBatchIdx, iv1, idx}, tailMask,
              passThroughVec);
          builder.create<affine::AffineForOp>(
              loc, ValueRange{c0}, builder.getDimIdentityMap(),
              ValueRange{aRow}, builder.getDimIdentityMap(), /*Step=*/1,
              std::nullopt,
              [&](OpBuilder &builder, Location loc, Value iv2,
                  ValueRange itrArgs1) {
                Value aValue = builder.create<memref::LoadOp>(
                    loc, A, ValueRange{loopVarBatchIdx, iv2, iv1});
                Value aVec =
                    builder.create<vector::BroadcastOp>(loc, vectorTy, aValue);
                Value maskedCVec = builder.create<MaskedLoadOp>(
                    loc, vectorTy, C, ValueRange{loopVarBatchIdx, iv2, idx},
                    tailMask, passThroughVec);
                // Compute the result vector either through integer
                // multiplication and addition or fused multiply-add
                // based on the element type.
                Value computedVec;
                if (isa<IntegerType>(elementType)) {
                  Value mulVec =
                      builder.create<arith::MulIOp>(loc, aVec, maskedBVec);
                  computedVec =
                      builder.create<arith::AddIOp>(loc, mulVec, maskedCVec);
                } else {
                  computedVec = builder.create<vector::FMAOp>(
                      loc, aVec, maskedBVec, maskedCVec);
                }
                builder.create<MaskedStoreOp>(
                    loc, C, ValueRange{loopVarBatchIdx, iv2, idx}, tailMask,
                    computedVec);
                builder.create<affine::AffineYieldOp>(loc);
              });

          builder.create<affine::AffineYieldOp>(loc);
        });
    rewriter.create<affine::AffineYieldOp>(loc);
    // Finalize the loop and erase the original operation.
    parallelBatchLoop.getRegion().push_back(loopBody);
    rewriter.setInsertionPointAfter(parallelBatchLoop);

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t vecSize;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// BatchMatMulOptimizePass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg pooling operations to mixture of
/// Affine + Vector operations.
namespace {
class BatchMatMulOptimizePass
    : public PassWrapper<BatchMatMulOptimizePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BatchMatMulOptimizePass)
  StringRef getArgument() const final { return "batchmatmul-optimize"; }
  StringRef getDescription() const final { return "BatchMatMul Optimization."; }
  BatchMatMulOptimizePass() = default;
  BatchMatMulOptimizePass(const BatchMatMulOptimizePass &) {}
  explicit BatchMatMulOptimizePass(int64_t vecSizeParam) {
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

void BatchMatMulOptimizePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                       scf::SCFDialect, memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<BatchMatMulOptimizePattern>(context, vecSize);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerBatchMatMulOptimizePass() {
  PassRegistration<BatchMatMulOptimizePass>();
}
} // namespace buddy
} // namespace mlir
