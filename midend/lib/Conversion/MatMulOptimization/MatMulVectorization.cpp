//===- MatMulVectorization.cpp --------------------------------------------===//
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
// This file implements the matmul vectorization.
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

#include "Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class MatMulVectorizationPattern : public ConversionPattern {
public:
  explicit MatMulVectorizationPattern(MLIRContext *context,
                                      int64_t vecSizeParam, bool scalableParam)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), 1, context) {
    vecSize = vecSizeParam;
    scalable = scalableParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // Create constant 0-7 values.
    const Value c0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value c1 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    // unroll size
    // Note: the unroll factor is not the same as the vector size.
    const Value c8 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(8));

    // Get input A, B, C.
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);

    // Create DimOp for m, n, k.
    const Value m = rewriter.create<memref::DimOp>(loc, A, c0);
    const Value n = rewriter.create<memref::DimOp>(loc, C, c1);
    const Value k = rewriter.create<memref::DimOp>(loc, A, c1);

    // Get element type and create vector type.
    ShapedType ATy = mlir::cast<mlir::ShapedType>(A.getType());
    Type eleTy = ATy.getElementType();
    VectorType vectorTy = VectorType::get({vecSize}, eleTy, {scalable});
    
    // Define step.
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, vecSize);
    if (scalable) {
      Value vscale = rewriter.create<vector::VectorScaleOp>(loc);
      step = rewriter.create<arith::MulIOp>(loc, step, vscale);
    }
    FloatType eleFloatTy =
        eleTy.isF32()
            ? static_cast<FloatType>(Float32Type::get(rewriter.getContext()))
            : static_cast<FloatType>(Float64Type::get(rewriter.getContext()));

    auto tail_size = rewriter.create<arith::RemUIOp>(loc, m, c8);
    auto parallel_size = rewriter.create<arith::SubIOp>(loc, m, tail_size);

    auto createUnrollParallel = [&](int unroll_size, Value lowerBound,
                                    Value upperBound) {
      const Value unroll = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexAttr(unroll_size));
      auto parOp = rewriter.create<scf::ParallelOp>(
          loc,
          /*lowerBounds=*/ValueRange{lowerBound},
          /*upperBounds=*/ValueRange{upperBound},
          /*steps=*/ValueRange{unroll},
          [&](OpBuilder &builder, Location loc, ValueRange mIdx) {
            llvm::SmallVector<Value, 8> mIndices;
            for (int i = 0; i < unroll_size; i++) {
              auto offset = rewriter.create<arith::ConstantOp>(
                  loc, rewriter.getIndexAttr(i));
              auto m_index =
                  rewriter.create<arith::AddIOp>(loc, mIdx[0], offset);
              mIndices.push_back(m_index);
            }

            auto nBodyBoundTmp = rewriter.create<arith::SubIOp>(loc, n, step);
            auto nBodyBound =
                rewriter.create<arith::AddIOp>(loc, nBodyBoundTmp, c1);

            auto nIterIdx = rewriter.create<scf::ForOp>(
                loc,
                /*lowerBound=*/c0,
                /*upperBound=*/nBodyBound,
                /*step=*/step,
                /*initArgs=*/ValueRange{c0},
                [&](OpBuilder &builder, Location loc, Value iv,
                    ValueRange iterArgs) {
                  SmallVector<Value> sumInitVecs;
                  for (auto mIndex : mIndices) {
                    auto sumInitVec = rewriter.create<vector::LoadOp>(
                        loc, vectorTy, C, ValueRange{mIndex, iv});
                    sumInitVecs.push_back(sumInitVec);
                  }

                  auto sumIterVecs = rewriter.create<scf::ForOp>(
                      loc,
                      /*lowerBound=*/c0,
                      /*upperBound=*/k,
                      /*step=*/c1,
                      /*initArgs=*/
                      ValueRange(sumInitVecs),
                      [&](OpBuilder &builder, Location loc, Value kIdx,
                          ValueRange iterArgs) {
                        auto bVec = rewriter.create<vector::LoadOp>(
                            loc, vectorTy, B, ValueRange{kIdx, iv});

                        SmallVector<Value> resSumVecs;
                        for (int i = 0; i < unroll_size; i++) {
                          auto aEle = rewriter.create<memref::LoadOp>(
                              loc, A, ValueRange{mIndices[i], kIdx});
                          auto aVec = rewriter.create<vector::BroadcastOp>(
                              loc, vectorTy, aEle);
                          auto resSumVec = rewriter.create<vector::FMAOp>(
                              loc, aVec, bVec, iterArgs[i]);
                          resSumVecs.push_back(resSumVec);
                        }
                        builder.create<scf::YieldOp>(loc,
                                                     ValueRange(resSumVecs));
                      });

                  for (int i = 0; i < unroll_size; i++) {
                    rewriter.create<vector::StoreOp>(
                        loc, sumIterVecs.getResult(i), C,
                        ValueRange{mIndices[i], iv});
                  }

                  auto kNext = rewriter.create<arith::AddIOp>(loc, iv, step);
                  builder.create<scf::YieldOp>(loc, ValueRange{kNext});
                });

            // Tail processing.
            builder.create<scf::ForOp>(
                loc, nIterIdx.getResult(0), n, c1, ValueRange(),
                [&](OpBuilder &builder, Location loc, Value iv,
                    ValueRange iterArgs) {
                  SmallVector<Value> sumInits;
                  for (auto mIndex : mIndices) {
                    auto sumInit = rewriter.create<memref::LoadOp>(
                        loc, eleFloatTy, C, ValueRange{mIndex, iv});
                    sumInits.push_back(sumInit);
                  }
                  auto sumIterVecs = rewriter.create<scf::ForOp>(
                      loc,
                      /*lowerBound=*/c0,
                      /*upperBound=*/k,
                      /*step=*/c1,
                      /*initArgs=*/
                      ValueRange(sumInits),
                      [&](OpBuilder &builder, Location loc, Value kIdx,
                          ValueRange iterArgs) {
                        SmallVector<Value> resSums;
                        auto bEle = rewriter.create<memref::LoadOp>(
                            loc, B, ValueRange{kIdx, iv});
                        for (int i = 0; i < unroll_size; i++) {
                          auto aEle = rewriter.create<memref::LoadOp>(
                              loc, A, ValueRange{mIndices[i], kIdx});
                          auto tmpEle =
                              rewriter.create<arith::MulFOp>(loc, aEle, bEle);
                          auto resSum = rewriter.create<arith::AddFOp>(
                              loc, tmpEle, iterArgs[i]);
                          resSums.push_back(resSum);
                        }

                        builder.create<scf::YieldOp>(loc, ValueRange(resSums));
                      });

                  for (int i = 0; i < unroll_size; i++) {
                    rewriter.create<memref::StoreOp>(
                        loc, sumIterVecs.getResult(i), C,
                        ValueRange{mIndices[i], iv});
                  }
                  builder.create<scf::YieldOp>(loc);
                });
          });
    };

    // Create parallel op for m dimension with unroll factor equal to 8.
    createUnrollParallel(8, c0, parallel_size);

    // Create parallel op for the tail of m dimension.
    createUnrollParallel(1, parallel_size, m);

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t vecSize;
  bool scalable;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// MatMulVectorizationPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg matmul operations to mixture of
/// Affine + Vector operations.
namespace {
class MatMulVectorizationPass
    : public PassWrapper<MatMulVectorizationPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulVectorizationPass)
  StringRef getArgument() const final { return "matmul-vectorization"; }
  StringRef getDescription() const final { return "MatMul Vectorization."; }
  MatMulVectorizationPass() = default;
  MatMulVectorizationPass(const MatMulVectorizationPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, VectorDialect>();
  }
  Option<int64_t> vecSize{*this, "vector-size",
                          llvm::cl::desc("Specify vector type size."),
                          llvm::cl::init(32)};
  Option<std::string> vectorType{
      *this, "vector-type",
      llvm::cl::desc("Specify vector type: fixed or scalable."),
      llvm::cl::init("fixed")};
};
} // end anonymous namespace.

void MatMulVectorizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                       scf::SCFDialect, memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  // Determine if scalable vectors are requested
  bool isScalable = (vectorType == "scalable");

  RewritePatternSet patterns(context);
  patterns.add<MatMulVectorizationPattern>(context, vecSize, isScalable);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerMatMulVectorizationPass() {
  PassRegistration<MatMulVectorizationPass>();
}
} // namespace buddy
} // namespace mlir
