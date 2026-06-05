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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;
using namespace vector;
using namespace affine;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

static bool hasDefaultBatchMatmulIndexingMaps(linalg::BatchMatmulOp op) {
  return linalg::BatchMatmulOp::isDefaultIndexingMaps(
      op->getAttr("indexing_maps"));
}

static bool isSupportedBatchMatmulOp(linalg::BatchMatmulOp op) {
  if (!op.hasPureBufferSemantics())
    return false;
  if (!hasDefaultBatchMatmulIndexingMaps(op))
    return false;

  Value A = op.getInputs()[0];
  Value B = op.getInputs()[1];
  Value C = op.getOutputs()[0];
  auto aType = dyn_cast<MemRefType>(A.getType());
  auto bType = dyn_cast<MemRefType>(B.getType());
  auto cType = dyn_cast<MemRefType>(C.getType());
  if (!aType || !bType || !cType)
    return false;
  if (aType.getRank() != 3 || bType.getRank() != 3 || cType.getRank() != 3)
    return false;

  Type elementType = aType.getElementType();
  if (bType.getElementType() != elementType ||
      cType.getElementType() != elementType)
    return false;

  return isa<IntegerType, FloatType>(elementType);
}

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
    auto batchMatmulOp = cast<linalg::BatchMatmulOp>(op);
    if (!isSupportedBatchMatmulOp(batchMatmulOp))
      return failure();

    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Retrieve input tensors A, B, and C.
    Value A = batchMatmulOp.getInputs()[0];
    Value B = batchMatmulOp.getInputs()[1];
    Value C = batchMatmulOp.getOutputs()[0];

    // Acquire the element type of input tensors.
    Type elementType =
        mlir::cast<mlir::MemRefType>(A.getType()).getElementType();
    VectorType vectorTy = mlir::VectorType::get({vecSize}, elementType);

    AffineExpr d0 = rewriter.getAffineDimExpr(0);
    AffineExpr d1 = rewriter.getAffineDimExpr(1);

    // Define constants.
    llvm::SmallVector<Value, 8> constantVals;
    for (int i = 0; i <= 8; ++i) {
      auto val =
          arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(i));
      constantVals.push_back(val);
    }
    Value vlStep = arith::ConstantIndexOp::create(rewriter, loc, vecSize);

    // Get dimensions of input tensors.
    Value batch = memref::DimOp::create(rewriter, loc, A, constantVals[0]);
    Value aRow = memref::DimOp::create(rewriter, loc, A, constantVals[1]);
    Value aCol = memref::DimOp::create(rewriter, loc, A, constantVals[2]);
    Value bCol = memref::DimOp::create(rewriter, loc, C, constantVals[2]);

    AffineMap unrollMap = AffineMap::get(
        /*dimCount=*/1,
        /*symbolCount=*/0, {d0 % rewriter.getAffineConstantExpr(8)}, ctx);
    AffineMap tailMap = AffineMap::get(
        /*dimCount=*/2,
        /*symbolCount=*/0, {d0 - d1}, ctx);
    AffineMap affineMapVec = AffineMap::get(
        /*dimCount=*/2,
        /*symbolCount=*/0, {d0 - d1 + rewriter.getAffineConstantExpr(1)}, ctx);

    Value tailSize =
        AffineApplyOp::create(rewriter, loc, unrollMap, ValueRange{aRow});
    Value parallelSize = AffineApplyOp::create(rewriter, loc, tailMap,
                                               ValueRange{aRow, tailSize});

    Value nUpperBound = AffineApplyOp::create(rewriter, loc, affineMapVec,
                                              ValueRange{bCol, vlStep});

    auto createUnrollParallel = [&](int unrollSize, Value lowerBound,
                                    Value upperBound) {
      // Create the primary parallel batch level loop.
      rewriter.create<scf::ParallelOp>(
          loc,
          /*lowerBounds=*/ValueRange{constantVals[0], lowerBound},
          /*upperBounds=*/ValueRange{batch, upperBound},
          /*steps=*/ValueRange{constantVals[1], constantVals[unrollSize]},
          [&](OpBuilder &builder, Location loc, ValueRange ivs) {
            llvm::SmallVector<Value, 8> mIndices;
            for (int i = 0; i < unrollSize; ++i) {
              auto mIndex =
                  arith::AddIOp::create(rewriter, loc, ivs[1], constantVals[i]);
              mIndices.push_back(mIndex);
            }

            auto iter_idx = scf::ForOp::create(
                rewriter, loc, constantVals[0], nUpperBound,
                /*Step=*/vlStep, ValueRange{constantVals[0]},
                [&](OpBuilder &builder, Location loc, Value iv0,
                    ValueRange iterArgs0) {
                  SmallVector<Value> cVecs;
                  for (auto mIndex : mIndices) {
                    auto cInitVec =
                        vector::LoadOp::create(rewriter, loc, vectorTy, C,
                                               ValueRange{ivs[0], mIndex, iv0});
                    cVecs.push_back(cInitVec);
                  }
                  auto sumIterVecs = scf::ForOp::create(
                      builder, loc, constantVals[0], aCol,
                      /*Step=*/constantVals[1], ValueRange{cVecs},
                      [&](OpBuilder &builder, Location loc, Value iv1,
                          ValueRange iterArgs1) {
                        auto bVec = vector::LoadOp::create(
                            rewriter, loc, vectorTy, B,
                            ValueRange{ivs[0], iv1, iv0});
                        SmallVector<Value> resSumVecs;
                        if (isa<IntegerType>(elementType)) {
                          for (int i = 0; i < unrollSize; ++i) {
                            auto aEle = memref::LoadOp::create(
                                rewriter, loc, A,
                                ValueRange{ivs[0], mIndices[i], iv1});
                            auto aVec = vector::BroadcastOp::create(
                                rewriter, loc, vectorTy, aEle);
                            Value mulVec =
                                builder.create<arith::MulIOp>(loc, aVec, bVec);
                            Value resSumVec = builder.create<arith::AddIOp>(
                                loc, mulVec, iterArgs1[i]);
                            resSumVecs.push_back(resSumVec);
                          }
                        } else {
                          for (int i = 0; i < unrollSize; ++i) {
                            auto aEle = memref::LoadOp::create(
                                rewriter, loc, A,
                                ValueRange{ivs[0], mIndices[i], iv1});
                            auto aVec = vector::BroadcastOp::create(
                                rewriter, loc, vectorTy, aEle);
                            Value resSumVec = vector::FMAOp::create(
                                rewriter, loc, aVec, bVec, iterArgs1[i]);
                            resSumVecs.push_back(resSumVec);
                          }
                        }
                        scf::YieldOp::create(builder, loc,
                                             ValueRange{resSumVecs});
                      });
                  for (int i = 0; i < unrollSize; ++i) {
                    vector::StoreOp::create(
                        rewriter, loc, sumIterVecs.getResult(i), C,
                        ValueRange{ivs[0], mIndices[i], iv0});
                  }

                  auto nextIdx =
                      arith::AddIOp::create(rewriter, loc, iv0, vlStep);
                  scf::YieldOp::create(builder, loc, ValueRange{nextIdx});
                });
            // Compute the tail size and Process the remaining elements
            // using masked vector operations.
            Value idx = iter_idx.getResult(0);
            scf::ForOp::create(
                rewriter, loc, idx, bCol,
                /*Step=*/constantVals[1], ValueRange{},
                [&](OpBuilder &builder, Location loc, Value iv0,
                    ValueRange itrArgs0) {
                  SmallVector<Value> cEles;
                  for (auto mIndex : mIndices) {
                    auto cInit = memref::LoadOp::create(
                        rewriter, loc, C, ValueRange{ivs[0], mIndex, iv0});
                    cEles.push_back(cInit);
                  }
                  auto sumIterVecs = scf::ForOp::create(
                      rewriter, loc, constantVals[0], aCol, constantVals[1],
                      ValueRange{cEles},
                      [&](OpBuilder &builder, Location loc, Value iv1,
                          ValueRange iterArgs1) {
                        auto bEle = memref::LoadOp::create(
                            rewriter, loc, B, ValueRange{ivs[0], iv1, iv0});
                        SmallVector<Value> resSums;
                        if (isa<IntegerType>(elementType)) {
                          for (int i = 0; i < unrollSize; ++i) {
                            auto aEle = memref::LoadOp::create(
                                rewriter, loc, A,
                                ValueRange{ivs[0], mIndices[i], iv1});
                            auto tmpEle = arith::MulIOp::create(rewriter, loc,
                                                                aEle, bEle);
                            auto resSum = arith::AddIOp::create(
                                rewriter, loc, tmpEle, iterArgs1[i]);
                            resSums.push_back(resSum);
                          }
                        } else {
                          for (int i = 0; i < unrollSize; ++i) {
                            auto aEle = memref::LoadOp::create(
                                rewriter, loc, A,
                                ValueRange{ivs[0], mIndices[i], iv1});
                            auto tmpEle = arith::MulFOp::create(rewriter, loc,
                                                                aEle, bEle);
                            auto resSum = arith::AddFOp::create(
                                rewriter, loc, tmpEle, iterArgs1[i]);
                            resSums.push_back(resSum);
                          }
                        }

                        scf::YieldOp::create(builder, loc, ValueRange{resSums});
                      });

                  for (int i = 0; i < unrollSize; i++) {
                    memref::StoreOp::create(
                        rewriter, loc, sumIterVecs.getResult(i), C,
                        ValueRange{ivs[0], mIndices[i], iv0});
                  }
                  scf::YieldOp::create(builder, loc);
                });
          });
    };
    createUnrollParallel(8, constantVals[0], parallelSize);
    createUnrollParallel(1, parallelSize, aRow);

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
  target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                         scf::SCFDialect, memref::MemRefDialect, VectorDialect,
                         linalg::LinalgDialect>();
  target.addDynamicallyLegalOp<linalg::BatchMatmulOp>(
      [](linalg::BatchMatmulOp op) { return !isSupportedBatchMatmulOp(op); });
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();

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
