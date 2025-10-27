//===--------------BatchMatMulOptimize.cpp---------------------------------===//
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
// This file implements the BatchMatMul optimization.
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
    Type elementType = A.getType().cast<MemRefType>().getElementType();
    VectorType vectorTy = mlir::VectorType::get({vecSize}, elementType);

    const AffineExpr d0 = rewriter.getAffineDimExpr(0);
    const AffineExpr d1 = rewriter.getAffineDimExpr(1);
    const AffineExpr d2 = rewriter.getAffineDimExpr(2);

    // Define constants.
    const Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    const Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    const Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    const Value c3 = rewriter.create<arith::ConstantIndexOp>(loc, 3);
    const Value c4 = rewriter.create<arith::ConstantIndexOp>(loc, 4);
    const Value c5 = rewriter.create<arith::ConstantIndexOp>(loc, 5);
    const Value c6 = rewriter.create<arith::ConstantIndexOp>(loc, 6);
    const Value c7 = rewriter.create<arith::ConstantIndexOp>(loc, 7);
    const Value unroll = rewriter.create<arith::ConstantIndexOp>(loc, 8);
    const Value vlStep = rewriter.create<arith::ConstantIndexOp>(loc, vecSize);

    // Get dimensions of input tensors.
    Value batch = rewriter.create<memref::DimOp>(loc, A, c0);
    Value aRow = rewriter.create<memref::DimOp>(loc, A, c1);
    Value aCol = rewriter.create<memref::DimOp>(loc, A, c2);
    Value bCol = rewriter.create<memref::DimOp>(loc, C, c2);

    Value upperBound_tmp = rewriter.create<arith::SubIOp>(loc, bCol, vlStep);
    Value upperBound = rewriter.create<arith::AddIOp>(loc, upperBound_tmp, c1);

    // Create the primary parallel batch level loop.
    auto parOp = rewriter.create<scf::ParallelOp>(
        loc,
        /*lowerBounds=*/ValueRange{c0, c0},
        /*upperBounds=*/ValueRange{batch, aRow},
        /*steps=*/ValueRange{c1, unroll},
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          auto aRowIdx1 = rewriter.create<arith::AddIOp>(loc, ivs[1], c1);
          auto aRowIdx2 = rewriter.create<arith::AddIOp>(loc, ivs[1], c2);
          auto aRowIdx3 = rewriter.create<arith::AddIOp>(loc, ivs[1], c3);
          auto aRowIdx4 = rewriter.create<arith::AddIOp>(loc, ivs[1], c4);
          auto aRowIdx5 = rewriter.create<arith::AddIOp>(loc, ivs[1], c5);
          auto aRowIdx6 = rewriter.create<arith::AddIOp>(loc, ivs[1], c6);
          auto aRowIdx7 = rewriter.create<arith::AddIOp>(loc, ivs[1], c7);

          auto iter_idx = rewriter.create<scf::ForOp>(
              loc, c0, upperBound,
              /*Step=*/vlStep, ValueRange{c0},
              [&](OpBuilder &builder, Location loc, Value iv0,
                  ValueRange iterArgs0) {
                auto cVec0 = rewriter.create<vector::LoadOp>(
                    loc, vectorTy, C, ValueRange{ivs[0], ivs[1], iv0});
                auto cVec1 = rewriter.create<vector::LoadOp>(
                    loc, vectorTy, C, ValueRange{ivs[0], aRowIdx1, iv0});
                auto cVec2 = rewriter.create<vector::LoadOp>(
                    loc, vectorTy, C, ValueRange{ivs[0], aRowIdx2, iv0});
                auto cVec3 = rewriter.create<vector::LoadOp>(
                    loc, vectorTy, C, ValueRange{ivs[0], aRowIdx3, iv0});
                auto cVec4 = rewriter.create<vector::LoadOp>(
                    loc, vectorTy, C, ValueRange{ivs[0], aRowIdx4, iv0});
                auto cVec5 = rewriter.create<vector::LoadOp>(
                    loc, vectorTy, C, ValueRange{ivs[0], aRowIdx5, iv0});
                auto cVec6 = rewriter.create<vector::LoadOp>(
                    loc, vectorTy, C, ValueRange{ivs[0], aRowIdx6, iv0});
                auto cVec7 = rewriter.create<vector::LoadOp>(
                    loc, vectorTy, C, ValueRange{ivs[0], aRowIdx7, iv0});
                auto sumIterVecs = builder.create<scf::ForOp>(
                    loc, c0, aCol, /*Step=*/c1,
                    ValueRange{cVec0, cVec1, cVec2, cVec3, cVec4, cVec5, cVec6,
                               cVec7},
                    [&](OpBuilder &builder, Location loc, Value iv1,
                        ValueRange iterArgs1) {
                      auto aEle0 = builder.create<memref::LoadOp>(
                          loc, A, ValueRange{ivs[0], ivs[1], iv1});
                      auto aEle1 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{ivs[0], aRowIdx1, iv1});
                      auto aEle2 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{ivs[0], aRowIdx2, iv1});
                      auto aEle3 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{ivs[0], aRowIdx3, iv1});
                      auto aEle4 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{ivs[0], aRowIdx4, iv1});
                      auto aEle5 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{ivs[0], aRowIdx5, iv1});
                      auto aEle6 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{ivs[0], aRowIdx6, iv1});
                      auto aEle7 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{ivs[0], aRowIdx7, iv1});
                      auto aVec0 = rewriter.create<vector::BroadcastOp>(
                          loc, vectorTy, aEle0);
                      auto aVec1 = rewriter.create<vector::BroadcastOp>(
                          loc, vectorTy, aEle1);
                      auto aVec2 = rewriter.create<vector::BroadcastOp>(
                          loc, vectorTy, aEle2);
                      auto aVec3 = rewriter.create<vector::BroadcastOp>(
                          loc, vectorTy, aEle3);
                      auto aVec4 = rewriter.create<vector::BroadcastOp>(
                          loc, vectorTy, aEle4);
                      auto aVec5 = rewriter.create<vector::BroadcastOp>(
                          loc, vectorTy, aEle5);
                      auto aVec6 = rewriter.create<vector::BroadcastOp>(
                          loc, vectorTy, aEle6);
                      auto aVec7 = rewriter.create<vector::BroadcastOp>(
                          loc, vectorTy, aEle7);
                      auto bVec = rewriter.create<vector::LoadOp>(
                          loc, vectorTy, B, ValueRange{ivs[0], iv1, iv0});
                      Value computedVec0;
                      Value computedVec1;
                      Value computedVec2;
                      Value computedVec3;
                      Value computedVec4;
                      Value computedVec5;
                      Value computedVec6;
                      Value computedVec7;
                      if (isa<IntegerType>(elementType)) {
                        Value mulVec0 =
                            builder.create<arith::MulIOp>(loc, aVec0, bVec);
                        computedVec0 = builder.create<arith::AddIOp>(
                            loc, mulVec0, iterArgs1[0]);
                        Value mulVec1 =
                            builder.create<arith::MulIOp>(loc, aVec1, bVec);
                        computedVec1 = builder.create<arith::AddIOp>(
                            loc, mulVec1, iterArgs1[1]);
                        Value mulVec2 =
                            builder.create<arith::MulIOp>(loc, aVec2, bVec);
                        computedVec2 = builder.create<arith::AddIOp>(
                            loc, mulVec2, iterArgs1[2]);
                        Value mulVec3 =
                            builder.create<arith::MulIOp>(loc, aVec3, bVec);
                        computedVec3 = builder.create<arith::AddIOp>(
                            loc, mulVec3, iterArgs1[3]);
                        Value mulVec4 =
                            builder.create<arith::MulIOp>(loc, aVec4, bVec);
                        computedVec4 = builder.create<arith::AddIOp>(
                            loc, mulVec4, iterArgs1[4]);
                        Value mulVec5 =
                            builder.create<arith::MulIOp>(loc, aVec5, bVec);
                        computedVec5 = builder.create<arith::AddIOp>(
                            loc, mulVec5, iterArgs1[5]);
                        Value mulVec6 =
                            builder.create<arith::MulIOp>(loc, aVec6, bVec);
                        computedVec6 = builder.create<arith::AddIOp>(
                            loc, mulVec6, iterArgs1[6]);
                        Value mulVec7 =
                            builder.create<arith::MulIOp>(loc, aVec7, bVec);
                        computedVec7 = builder.create<arith::AddIOp>(
                            loc, mulVec7, iterArgs1[7]);
                      } else {
                        computedVec0 = builder.create<vector::FMAOp>(
                            loc, aVec0, bVec, iterArgs1[0]);
                        computedVec1 = builder.create<vector::FMAOp>(
                            loc, aVec1, bVec, iterArgs1[1]);
                        computedVec2 = builder.create<vector::FMAOp>(
                            loc, aVec2, bVec, iterArgs1[2]);
                        computedVec3 = builder.create<vector::FMAOp>(
                            loc, aVec3, bVec, iterArgs1[3]);
                        computedVec4 = builder.create<vector::FMAOp>(
                            loc, aVec4, bVec, iterArgs1[4]);
                        computedVec5 = builder.create<vector::FMAOp>(
                            loc, aVec5, bVec, iterArgs1[5]);
                        computedVec6 = builder.create<vector::FMAOp>(
                            loc, aVec6, bVec, iterArgs1[6]);
                        computedVec7 = builder.create<vector::FMAOp>(
                            loc, aVec7, bVec, iterArgs1[7]);
                      }
                      builder.create<scf::YieldOp>(
                          loc,
                          ValueRange{computedVec0, computedVec1, computedVec2,
                                     computedVec3, computedVec4, computedVec5,
                                     computedVec6, computedVec7});
                    });
                auto sumIterVec0 = rewriter.create<vector::StoreOp>(
                    loc, sumIterVecs.getResult(0), C,
                    ValueRange{ivs[0], ivs[1], iv0});
                auto sumIterVec1 = rewriter.create<vector::StoreOp>(
                    loc, sumIterVecs.getResult(1), C,
                    ValueRange{ivs[0], aRowIdx1, iv0});
                auto sumIterVec2 = rewriter.create<vector::StoreOp>(
                    loc, sumIterVecs.getResult(2), C,
                    ValueRange{ivs[0], aRowIdx2, iv0});
                auto sumIterVec3 = rewriter.create<vector::StoreOp>(
                    loc, sumIterVecs.getResult(3), C,
                    ValueRange{ivs[0], aRowIdx3, iv0});
                auto sumIterVec4 = rewriter.create<vector::StoreOp>(
                    loc, sumIterVecs.getResult(4), C,
                    ValueRange{ivs[0], aRowIdx4, iv0});
                auto sumIterVec5 = rewriter.create<vector::StoreOp>(
                    loc, sumIterVecs.getResult(5), C,
                    ValueRange{ivs[0], aRowIdx5, iv0});
                auto sumIterVec6 = rewriter.create<vector::StoreOp>(
                    loc, sumIterVecs.getResult(6), C,
                    ValueRange{ivs[0], aRowIdx6, iv0});
                auto sumIterVec7 = rewriter.create<vector::StoreOp>(
                    loc, sumIterVecs.getResult(7), C,
                    ValueRange{ivs[0], aRowIdx7, iv0});

                auto nextIdx = rewriter.create<arith::AddIOp>(loc, iv0, vlStep);
                builder.create<scf::YieldOp>(loc, ValueRange{nextIdx});
              });
          // Compute the tail size and Process the remaining elements
          // using masked vector operations.
          Value idx = iter_idx.getResult(0);
          rewriter.create<scf::ForOp>(
              loc, idx, bCol,
              /*Step=*/c1, std::nullopt,
              [&](OpBuilder &builder, Location loc, Value iv0,
                  ValueRange itrArgs0) {
                auto cEle0 = rewriter.create<memref::LoadOp>(
                    loc, C, ValueRange{ivs[0], ivs[1], iv0});
                auto cEle1 = rewriter.create<memref::LoadOp>(
                    loc, C, ValueRange{ivs[0], aRowIdx1, iv0});
                auto cEle2 = rewriter.create<memref::LoadOp>(
                    loc, C, ValueRange{ivs[0], aRowIdx2, iv0});
                auto cEle3 = rewriter.create<memref::LoadOp>(
                    loc, C, ValueRange{ivs[0], aRowIdx3, iv0});
                auto cEle4 = rewriter.create<memref::LoadOp>(
                    loc, C, ValueRange{ivs[0], aRowIdx4, iv0});
                auto cEle5 = rewriter.create<memref::LoadOp>(
                    loc, C, ValueRange{ivs[0], aRowIdx5, iv0});
                auto cEle6 = rewriter.create<memref::LoadOp>(
                    loc, C, ValueRange{ivs[0], aRowIdx6, iv0});
                auto cEle7 = rewriter.create<memref::LoadOp>(
                    loc, C, ValueRange{ivs[0], aRowIdx7, iv0});
                auto sumIterVecs = rewriter.create<scf::ForOp>(
                    loc, c0, aCol, c1,
                    ValueRange{cEle0, cEle1, cEle2, cEle3, cEle4, cEle5, cEle6,
                               cEle7},
                    [&](OpBuilder &builder, Location loc, Value iv1,
                        ValueRange iterArgs1) {
                      auto aEle0 = builder.create<memref::LoadOp>(
                          loc, A, ValueRange{ivs[0], ivs[1], iv1});
                      auto aEle1 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{ivs[0], aRowIdx1, iv1});
                      auto aEle2 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{ivs[0], aRowIdx2, iv1});
                      auto aEle3 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{ivs[0], aRowIdx3, iv1});
                      auto aEle4 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{ivs[0], aRowIdx4, iv1});
                      auto aEle5 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{ivs[0], aRowIdx5, iv1});
                      auto aEle6 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{ivs[0], aRowIdx6, iv1});
                      auto aEle7 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{ivs[0], aRowIdx7, iv1});
                      auto bEle = rewriter.create<memref::LoadOp>(
                          loc, B, ValueRange{ivs[0], iv1, iv0});

                      Value computedVal0;
                      Value computedVal1;
                      Value computedVal2;
                      Value computedVal3;
                      Value computedVal4;
                      Value computedVal5;
                      Value computedVal6;
                      Value computedVal7;
                      if (isa<IntegerType>(elementType)) {
                        auto tmpEle0 =
                            rewriter.create<arith::MulIOp>(loc, aEle0, bEle);
                        auto tmpEle1 =
                            rewriter.create<arith::MulIOp>(loc, aEle1, bEle);
                        auto tmpEle2 =
                            rewriter.create<arith::MulIOp>(loc, aEle2, bEle);
                        auto tmpEle3 =
                            rewriter.create<arith::MulIOp>(loc, aEle3, bEle);
                        auto tmpEle4 =
                            rewriter.create<arith::MulIOp>(loc, aEle4, bEle);
                        auto tmpEle5 =
                            rewriter.create<arith::MulIOp>(loc, aEle5, bEle);
                        auto tmpEle6 =
                            rewriter.create<arith::MulIOp>(loc, aEle6, bEle);
                        auto tmpEle7 =
                            rewriter.create<arith::MulIOp>(loc, aEle7, bEle);

                        computedVal0 = rewriter.create<arith::AddIOp>(
                            loc, tmpEle0, iterArgs1[0]);
                        computedVal1 = rewriter.create<arith::AddIOp>(
                            loc, tmpEle1, iterArgs1[1]);
                        computedVal2 = rewriter.create<arith::AddIOp>(
                            loc, tmpEle2, iterArgs1[2]);
                        computedVal3 = rewriter.create<arith::AddIOp>(
                            loc, tmpEle3, iterArgs1[3]);
                        computedVal4 = rewriter.create<arith::AddIOp>(
                            loc, tmpEle4, iterArgs1[4]);
                        computedVal5 = rewriter.create<arith::AddIOp>(
                            loc, tmpEle5, iterArgs1[5]);
                        computedVal6 = rewriter.create<arith::AddIOp>(
                            loc, tmpEle6, iterArgs1[6]);
                        computedVal7 = rewriter.create<arith::AddIOp>(
                            loc, tmpEle7, iterArgs1[7]);
                      } else {
                        auto tmpEle0 =
                            rewriter.create<arith::MulFOp>(loc, aEle0, bEle);
                        auto tmpEle1 =
                            rewriter.create<arith::MulFOp>(loc, aEle1, bEle);
                        auto tmpEle2 =
                            rewriter.create<arith::MulFOp>(loc, aEle2, bEle);
                        auto tmpEle3 =
                            rewriter.create<arith::MulFOp>(loc, aEle3, bEle);
                        auto tmpEle4 =
                            rewriter.create<arith::MulFOp>(loc, aEle4, bEle);
                        auto tmpEle5 =
                            rewriter.create<arith::MulFOp>(loc, aEle5, bEle);
                        auto tmpEle6 =
                            rewriter.create<arith::MulFOp>(loc, aEle6, bEle);
                        auto tmpEle7 =
                            rewriter.create<arith::MulFOp>(loc, aEle7, bEle);

                        computedVal0 = rewriter.create<arith::AddFOp>(
                            loc, tmpEle0, iterArgs1[0]);
                        computedVal1 = rewriter.create<arith::AddFOp>(
                            loc, tmpEle1, iterArgs1[1]);
                        computedVal2 = rewriter.create<arith::AddFOp>(
                            loc, tmpEle2, iterArgs1[2]);
                        computedVal3 = rewriter.create<arith::AddFOp>(
                            loc, tmpEle3, iterArgs1[3]);
                        computedVal4 = rewriter.create<arith::AddFOp>(
                            loc, tmpEle4, iterArgs1[4]);
                        computedVal5 = rewriter.create<arith::AddFOp>(
                            loc, tmpEle5, iterArgs1[5]);
                        computedVal6 = rewriter.create<arith::AddFOp>(
                            loc, tmpEle6, iterArgs1[6]);
                        computedVal7 = rewriter.create<arith::AddFOp>(
                            loc, tmpEle7, iterArgs1[7]);
                      }

                      builder.create<scf::YieldOp>(
                          loc,
                          ValueRange{computedVal0, computedVal1, computedVal2,
                                     computedVal3, computedVal4, computedVal5,
                                     computedVal6, computedVal7});
                    });

                auto sumIter0 = rewriter.create<memref::StoreOp>(
                    loc, sumIterVecs.getResult(0), C,
                    ValueRange{ivs[0], ivs[1], iv0});
                auto sumIter1 = rewriter.create<memref::StoreOp>(
                    loc, sumIterVecs.getResult(1), C,
                    ValueRange{ivs[0], aRowIdx1, iv0});
                auto sumIter2 = rewriter.create<memref::StoreOp>(
                    loc, sumIterVecs.getResult(2), C,
                    ValueRange{ivs[0], aRowIdx2, iv0});
                auto sumIter3 = rewriter.create<memref::StoreOp>(
                    loc, sumIterVecs.getResult(3), C,
                    ValueRange{ivs[0], aRowIdx3, iv0});
                auto sumIter4 = rewriter.create<memref::StoreOp>(
                    loc, sumIterVecs.getResult(4), C,
                    ValueRange{ivs[0], aRowIdx4, iv0});
                auto sumIter5 = rewriter.create<memref::StoreOp>(
                    loc, sumIterVecs.getResult(5), C,
                    ValueRange{ivs[0], aRowIdx5, iv0});
                auto sumIter6 = rewriter.create<memref::StoreOp>(
                    loc, sumIterVecs.getResult(6), C,
                    ValueRange{ivs[0], aRowIdx6, iv0});
                auto sumIter7 = rewriter.create<memref::StoreOp>(
                    loc, sumIterVecs.getResult(7), C,
                    ValueRange{ivs[0], aRowIdx7, iv0});

                builder.create<scf::YieldOp>(loc);
              });
        });

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
