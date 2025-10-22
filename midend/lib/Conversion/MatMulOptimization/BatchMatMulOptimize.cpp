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

class BatchMatMulOptimizeOldPattern : public ConversionPattern {
public:
  explicit BatchMatMulOptimizeOldPattern(MLIRContext *context,
                                         int64_t vecSizeParam,
                                         int64_t kBlockSizeParam)
      : ConversionPattern(linalg::BatchMatmulOp::getOperationName(), 1,
                          context) {
    vecSize = vecSizeParam;
    kBlockSize = kBlockSizeParam;
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
    // Acquire the element type of input tensors.
    Type elementType = mlir::cast<MemRefType>(A.getType()).getElementType();
    VectorType vectorTy = mlir::VectorType::get({vecSize}, elementType);

    const AffineExpr d0 = rewriter.getAffineDimExpr(0);
    const AffineExpr d1 = rewriter.getAffineDimExpr(1);
    const AffineExpr d2 = rewriter.getAffineDimExpr(2);
    const AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
    const AffineExpr s1 = rewriter.getAffineSymbolExpr(1);
    const AffineExpr c0expr = rewriter.getAffineConstantExpr(0);

    // Define constants.
    const Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    const Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    const Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    const Value vl_step = rewriter.create<arith::ConstantIndexOp>(loc, vecSize);
    const Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));

    // Create pass through vector.
    Value passThroughVec = rewriter.create<SplatOp>(loc, vectorTy, zero);

    // Get dimensions of input tensors.
    Value batch = rewriter.create<memref::DimOp>(loc, A, c0);
    Value aRow = rewriter.create<memref::DimOp>(loc, A, c1);
    Value bCol = rewriter.create<memref::DimOp>(loc, B, c2);
    Value bRow = rewriter.create<memref::DimOp>(loc, B, c1);

    // Calculate the upper bound for vectorized processing
    // - Subtract `vl_step` is to avoid overflow at the vectorization tail.
    // - Add 1 to ensure the final loop runs when the workload length
    //   is divisible by the vector size.
    // Value upperBound_tmp = rewriter.create<arith::SubIOp>(loc, bCol,
    // vl_step); Value upperBound = rewriter.create<arith::AddIOp>(loc,
    // upperBound_tmp, c1);

    auto map = AffineMap::get(/*numDims=*/1, /*numSymbols=*/2, d0 - s0 + s1);
    Value upperBound = rewriter.create<affine::AffineApplyOp>(
        loc, map, ValueRange{bCol, vl_step, c1});

    auto zeroI32Tensor = rewriter.getI32TensorAttr({1});
    auto emptyReductions = rewriter.getArrayAttr({});
    auto lowerMapAttr = AffineMapAttr::get(
        AffineMap::get(/*numDims=*/0, /*numSymbols=*/0, {c0expr, c0expr}, ctx));
    auto upperMapAttr = AffineMapAttr::get(
        AffineMap::get(/*numDims=*/2, /*numSymbols=*/0, {d0, d1}, ctx));
    auto lowerBoundsGroups = rewriter.getI32TensorAttr({1, 1});
    auto upperBoundsGroups = rewriter.getI32TensorAttr({1, 1});
    auto stepsAttr =
        rewriter.getI64ArrayAttr({1, static_cast<int64_t>(vecSize)});

    AffineParallelOp parallelBatchLoop =
        rewriter.create<affine::AffineParallelOp>(
            loc,
            /*resultTypes=*/TypeRange{}, // no reductions / results
            /*operands=*/ValueRange{batch, upperBound},
            /*attributes=*/
            ArrayRef<NamedAttribute>{
                rewriter.getNamedAttr("lowerBoundsGroups", lowerBoundsGroups),
                rewriter.getNamedAttr("upperBoundsGroups", upperBoundsGroups),
                rewriter.getNamedAttr("lowerBoundsMap", lowerMapAttr),
                rewriter.getNamedAttr("upperBoundsMap", upperMapAttr),
                rewriter.getNamedAttr("reductions", emptyReductions),
                rewriter.getNamedAttr("steps", stepsAttr),
            });

    Block *loopBody = new Block();
    loopBody->addArgument(rewriter.getIndexType(), loc);
    loopBody->addArgument(rewriter.getIndexType(), loc);
    parallelBatchLoop.getRegion().push_back(loopBody);

    OpBuilder builder = OpBuilder::atBlockBegin(loopBody);
    Value batchIdx = loopBody->getArguments()[0];
    Value loopVarColOfB = loopBody->getArguments()[1];

    affine::buildAffineLoopNest(
        builder, loc, {c0}, {bRow}, /*Step=*/{kBlockSize},
        [&](OpBuilder &builder2, Location loc2, ValueRange ivRange) {
          Value kLow = ivRange.front();
          Value kHigh = builder2.create<affine::AffineMinOp>(
              loc2,
              AffineMap::get(1, 1, {d0 + kBlockSize, s0},
                             builder2.getContext()),
              SmallVector<Value>{kLow, bRow});
          affine::buildAffineLoopNest(
              builder2, loc2, {c0}, {aRow}, 1,
              [&](OpBuilder &builder3, Location loc3, ValueRange ivRange) {
                Value loopVarRowOfA = ivRange.front();
                Value cVec = builder3.create<vector::LoadOp>(
                    loc3, vectorTy, C,
                    ValueRange{batchIdx, loopVarRowOfA, loopVarColOfB});
                auto iter_vec = builder3.create<scf::ForOp>(
                    loc3, kLow, kHigh, /*Step=*/c1, ValueRange{cVec},
                    [&](OpBuilder &builder4, Location loc4, Value iv1,
                        ValueRange itrArgs0) {
                      Value aValue = builder4.create<memref::LoadOp>(
                          loc4, elementType, A,
                          ValueRange{batchIdx, loopVarRowOfA, iv1});
                      Value aVec = builder4.create<vector::BroadcastOp>(
                          loc4, vectorTy, aValue);
                      Value bVec = builder4.create<vector::LoadOp>(
                          loc4, vectorTy, B,
                          ValueRange{batchIdx, iv1, loopVarColOfB});
                                   // Compute the result vector either through integer
                                  // multiplication and addition or fused multiply-add
                                  // based on the element type.
                      Value computedVec;
                      if (isa<IntegerType>(elementType)) {
                        Value mulVec =
                            builder4.create<arith::MulIOp>(loc4, aVec, bVec);
                        computedVec = builder4.create<arith::AddIOp>(
                            loc4, mulVec, itrArgs0[0]);
                      } else {
                        computedVec = builder4.create<vector::FMAOp>(
                            loc4, aVec, bVec, itrArgs0[0]);
                      }
                      builder4.create<scf::YieldOp>(loc4, computedVec);
                    });
                builder3.create<vector::StoreOp>(
                    loc3, iter_vec.getResult(0), C,
                    ValueRange{batchIdx, loopVarRowOfA, loopVarColOfB});
              });
        });

    builder.create<affine::AffineYieldOp>(loc);
    rewriter.setInsertionPointAfter(parallelBatchLoop);

    if (B.getType().cast<MemRefType>().isDynamicDim(2) or
        B.getType().cast<MemRefType>().getDimSize(2) % vecSize != 0) {
      AffineParallelOp parallelBatchLoop2 =
          rewriter.create<affine::AffineParallelOp>(
              loc,
              /*resultTypes=*/TypeRange{},
              /*operands=*/ValueRange{batch},
              /*attributes=*/
              ArrayRef<NamedAttribute>{
                  rewriter.getNamedAttr("lowerBoundsGroups", zeroI32Tensor),
                  rewriter.getNamedAttr("upperBoundsGroups", zeroI32Tensor),
                  rewriter.getNamedAttr(
                      "lowerBoundsMap",
                      AffineMapAttr::get(AffineMap::get(0, 0, {c0expr}, ctx))),
                  rewriter.getNamedAttr(
                      "upperBoundsMap",
                      AffineMapAttr::get(AffineMap::get(1, 0, {d0}, ctx))),
                  rewriter.getNamedAttr("reductions", emptyReductions),
                  rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr({1})),
              });

      Block *loopBody2 = new Block();
      loopBody2->addArgument(rewriter.getIndexType(), loc);
      parallelBatchLoop2.getRegion().push_back(loopBody2);

      OpBuilder parallelBuilder2 = OpBuilder::atBlockBegin(loopBody2);
      Value parallelBatchIdx2 = loopBody2->getArguments()[0];

      affine::AffineIfOp branchingOp =
          parallelBuilder2.create<affine::AffineIfOp>(
              loc, IntegerSet::get(1, 0, {d0 % vecSize - 1}, {false}),
              ValueRange{bCol}, /*hasElse=*/false);

      // Branch handling operations on the tail.
      OpBuilder trueBranchBuilder = branchingOp.getThenBodyBuilder();
      Value tailSize = trueBranchBuilder.create<affine::AffineApplyOp>(
          loc, AffineMap::get(1, 0, d0 % vecSize), ValueRange{bCol});
      Value maskVector = trueBranchBuilder.create<vector::CreateMaskOp>(
          loc, VectorType::get({vecSize}, rewriter.getI1Type()),
          ValueRange{tailSize});
      Value loopVarColOfBTail =
          trueBranchBuilder.create<arith::SubIOp>(loc, bCol, tailSize);

      affine::buildAffineLoopNest(
          trueBranchBuilder, loc, {c0}, {bRow}, {kBlockSize},
          [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
            Value kLow = ivRange.front();
            Value kHigh = builder.create<affine::AffineMinOp>(
                loc,
                AffineMap::get(1, 1, {d0 + kBlockSize, s0},
                               builder.getContext()),
                SmallVector<Value>{kLow, bRow});
            affine::buildAffineLoopNest(
                builder, loc, {c0}, {aRow}, 1,
                [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
                  Value loopVarRowOfA = ivRange.front();
                  Value cVec = builder.create<vector::MaskedLoadOp>(
                      loc, vectorTy, C,
                      ValueRange{parallelBatchIdx2, loopVarRowOfA,
                                 loopVarColOfBTail},
                      maskVector, passThroughVec);
                  auto iter_vec = builder.create<scf::ForOp>(
                      loc, kLow, kHigh, /*Step=*/c1, ValueRange{cVec},
                      [&](OpBuilder &builder, Location loc, Value iv1,
                          ValueRange itrArgs0) {
                        Value aValue = builder.create<memref::LoadOp>(
                            loc, A,
                            ValueRange{parallelBatchIdx2, loopVarRowOfA, iv1});
                        Value aVec = builder.create<vector::BroadcastOp>(
                            loc, vectorTy, aValue);
                        Value maskedBVec = builder.create<MaskedLoadOp>(
                            loc, vectorTy, B,
                            ValueRange{parallelBatchIdx2, iv1,
                                       loopVarColOfBTail},
                            maskVector, passThroughVec);

                        Value computedVec;
                        if (isa<IntegerType>(elementType)) {
                          Value mulVec = builder.create<arith::MulIOp>(
                              loc, aVec, maskedBVec);
                          computedVec = builder.create<arith::AddIOp>(
                              loc, mulVec, itrArgs0[0]);
                        } else {
                          computedVec = builder.create<vector::FMAOp>(
                              loc, aVec, maskedBVec, itrArgs0[0]);
                        }
                        builder.create<scf::YieldOp>(loc, computedVec);
                      });
                  builder.create<MaskedStoreOp>(
                      loc, C,
                      ValueRange{parallelBatchIdx2, loopVarRowOfA,
                                 loopVarColOfBTail},
                      maskVector, iter_vec.getResult(0));
                });
          });

      parallelBuilder2.create<affine::AffineYieldOp>(loc);
      rewriter.setInsertionPointAfter(parallelBatchLoop2);
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t vecSize;
  int64_t kBlockSize;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// BatchMatMulOptimizeOldPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg pooling operations to mixture of
/// Affine + Vector operations.
namespace {
class BatchMatMulOptimizeOldPass
    : public PassWrapper<BatchMatMulOptimizeOldPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BatchMatMulOptimizeOldPass)
  StringRef getArgument() const final { return "batchmatmul-optimize-old"; }
  StringRef getDescription() const final {
    return "BatchMatMul Optimization old.";
  }
  BatchMatMulOptimizeOldPass() = default;
  BatchMatMulOptimizeOldPass(const BatchMatMulOptimizeOldPass &) {}
  explicit BatchMatMulOptimizeOldPass(int64_t vecSizeParam,
                                      int64_t kBlockSizeParam) {
    vecSize = vecSizeParam;
    kBlockSize = kBlockSizeParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, VectorDialect>();
  }

  Option<int64_t> vecSize{*this, "vector-size",
                          llvm::cl::desc("Affine Vector size."),
                          llvm::cl::init(32)};
  Option<int64_t> kBlockSize{*this, "k-block-size",
                             llvm::cl::desc("K block size."),
                             llvm::cl::init(32)};
};
} // end anonymous namespace.

void BatchMatMulOptimizeOldPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                       scf::SCFDialect, memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<BatchMatMulOptimizeOldPattern>(context, vecSize, kBlockSize);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerBatchMatMulOptimizeOldPass() {
  PassRegistration<BatchMatMulOptimizeOldPass>();
}
} // namespace buddy
} // namespace mlir
