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
// This file implements the batchmatmul scf vectorization optimization.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <mlir/Dialect/Affine/Analysis/AffineAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
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

class BatchMatMuSCFOptimizePattern : public ConversionPattern {
private:
  int64_t vecSize;

public:
  explicit BatchMatMuSCFOptimizePattern(MLIRContext *context,
                                        int64_t vecSizeParam)
      : ConversionPattern(linalg::BatchMatmulOp::getOperationName(), 1,
                          context) {
    vecSize = vecSizeParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // Retrieve input tensors A, B, and C.
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);

    // Acquire the element type of input tensors.
    Type elementType = A.getType().cast<MemRefType>().getElementType();

    // Define constants.
    const Value c0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value c1 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    const Value cVecSize =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(vecSize));
    const AffineExpr d0 = rewriter.getAffineDimExpr(0);
    // TODO: remove the following values?
    // const AffineExpr d1 = rewriter.getAffineDimExpr(1);
    // const AffineExpr d2 = rewriter.getAffineDimExpr(2);
    // const AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
    // const AffineExpr zeroAffine = rewriter.getAffineConstantExpr(0);

    const Value zeroElementType = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));

    // Get dimensions of input tensors.
    Value batch = rewriter.create<memref::DimOp>(loc, A, 0);
    Value aRow = rewriter.create<memref::DimOp>(loc, A, 1);
    Value bCol = rewriter.create<memref::DimOp>(loc, B, 2);
    Value bRow = rewriter.create<memref::DimOp>(loc, B, 1);

    VectorType vecTy = VectorType::get({vecSize}, elementType);
    Value zeroElementTypeVec;
    if (isa<IntegerType>(elementType))
      zeroElementTypeVec =
          rewriter.create<vector::BroadcastOp>(loc, vecTy, zeroElementType);
    else
      zeroElementTypeVec =
          rewriter.create<vector::SplatOp>(loc, vecTy, zeroElementType);
    // Calculate the length of the tail, which might not fit in a
    // vector.
    Value tailLength = rewriter.create<affine::AffineApplyOp>(
        loc, AffineMap::get(1, 0, d0 % vecSize), ValueRange{bCol});

    // Generate a mask vector based on the tail length.
    Value maskVector = rewriter.create<vector::CreateMaskOp>(
        loc, VectorType::get({vecSize}, rewriter.getI1Type()),
        ValueRange{tailLength});

    Value ApplyBCol = rewriter.create<affine::AffineApplyOp>(
        loc, AffineMap::get(1, 0, d0.floorDiv(vecSize) * vecSize), bCol);

    rewriter.create<scf::ForallOp>(
        loc, SmallVector<OpFoldResult, 1>({c0}),
        SmallVector<OpFoldResult, 1>({batch}),
        SmallVector<OpFoldResult, 1>({c1}), ValueRange{},
        std::nullopt, // No mapping specified in this example
        [&](OpBuilder &builder, Location loc, ValueRange loopIndices) {
          Value loopVarBatchIdx = loopIndices[0];
          builder.create<scf::ForOp>(
              loc, c0, aRow, c1, ValueRange{std::nullopt},
              [&](OpBuilder &builder, Location loc, Value loopVarRowOfA,
                  ValueRange iargs) {
                builder.create<scf::ForOp>(
                    loc, c0, bRow, c1, ValueRange{std::nullopt},
                    [&](OpBuilder &builder, Location loc, Value loopVarRowOfB,
                        ValueRange iargs) {
                      Value aElement = builder.create<memref::LoadOp>(
                          loc, A,
                          ValueRange{loopVarBatchIdx, loopVarRowOfA,
                                     loopVarRowOfB});
                      Value aVec = builder.create<vector::BroadcastOp>(
                          loc, vecTy, aElement);
                      builder.create<scf::ForOp>(
                          loc, c0, ApplyBCol, cVecSize,
                          ValueRange{std::nullopt},
                          [&](OpBuilder &builder, Location loc,
                              Value loopVarColOfB, ValueRange iargs) {
                            Value bVec = builder.create<vector::LoadOp>(
                                loc, vecTy, B,
                                ValueRange{loopVarBatchIdx, loopVarRowOfB,
                                           loopVarColOfB});

                            Value cVec = builder.create<vector::LoadOp>(
                                loc, vecTy, C,
                                ValueRange{loopVarBatchIdx, loopVarRowOfA,
                                           loopVarColOfB});
                            Value computedVec;

                            if (isa<IntegerType>(elementType)) {
                              Value mulVec = builder.create<arith::MulIOp>(
                                  loc, aVec, bVec);
                              computedVec = builder.create<arith::AddIOp>(
                                  loc, mulVec, cVec);
                            } else {
                              computedVec = builder.create<vector::FMAOp>(
                                  loc, aVec, bVec, cVec);
                            }
                            builder.create<vector::StoreOp>(
                                loc, computedVec, C,
                                ValueRange{loopVarBatchIdx, loopVarRowOfA,
                                           loopVarColOfB});
                            builder.create<scf::YieldOp>(
                                loc, ValueRange{std::nullopt});
                          });
                      Value condition = builder.create<arith::CmpIOp>(
                          loc, arith::CmpIPredicate::sgt, tailLength, c0);
                      builder.create<scf::IfOp>(
                          loc, condition,
                          [&](OpBuilder &builder, Location loc) {
                            Value bVec = builder.create<vector::MaskedLoadOp>(
                                loc, vecTy, B,
                                ValueRange{loopVarBatchIdx, loopVarRowOfB,
                                           ApplyBCol},
                                maskVector, zeroElementTypeVec);

                            Value cVec = builder.create<vector::MaskedLoadOp>(
                                loc, vecTy, C,
                                ValueRange{loopVarBatchIdx, loopVarRowOfA,
                                           ApplyBCol},
                                maskVector, zeroElementTypeVec);

                            Value computedVec;

                            if (isa<IntegerType>(elementType)) {
                              Value mulVec = builder.create<arith::MulIOp>(
                                  loc, aVec, bVec);
                              computedVec = builder.create<arith::AddIOp>(
                                  loc, mulVec, cVec);
                            } else {
                              computedVec = builder.create<vector::FMAOp>(
                                  loc, aVec, bVec, cVec);
                            }

                            builder.create<vector::MaskedStoreOp>(
                                loc, C,
                                ValueRange{loopVarBatchIdx, loopVarRowOfA,
                                           ApplyBCol},
                                maskVector, computedVec);
                            builder.create<scf::YieldOp>(loc);
                          });
                      builder.create<scf::YieldOp>(loc,
                                                   ValueRange{std::nullopt});
                    });
                builder.create<scf::YieldOp>(loc, ValueRange{std::nullopt});
              });

          builder.create<scf::InParallelOp>(loc);
        });

    rewriter.eraseOp(op);
    return success();
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// BatchMatMuSCFOptimize
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg pooling operations to mixture of
/// Affine + Vector operations.
namespace {
class BatchMatMuSCFOptimize
    : public PassWrapper<BatchMatMuSCFOptimize, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BatchMatMuSCFOptimize)
  StringRef getArgument() const final { return "batchmatmul-scf-optimize"; }
  StringRef getDescription() const final {
    return "BatchMatMul SCF Optimization.";
  }
  BatchMatMuSCFOptimize() = default;
  BatchMatMuSCFOptimize(const BatchMatMuSCFOptimize &) {}
  explicit BatchMatMuSCFOptimize(int64_t vecSizeParam) {
    vecSize = vecSizeParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, VectorDialect>();
  }

  Option<int64_t> vecSize{*this, "vector-size",
                          llvm::cl::desc("Strip mining size."),
                          llvm::cl::init(16)};
};
} // end anonymous namespace.

void BatchMatMuSCFOptimize::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                       scf::SCFDialect, memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<BatchMatMuSCFOptimizePattern>(context, vecSize);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
// add to buddy-opt.cpp
namespace mlir {
namespace buddy {
void registerBatchMatMuSCFOptimize() {
  PassRegistration<BatchMatMuSCFOptimize>();
}
} // namespace buddy
} // namespace mlir
