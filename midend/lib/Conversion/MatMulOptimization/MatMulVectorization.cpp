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

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class MatMulVectorizationPattern : public ConversionPattern {
public:
  explicit MatMulVectorizationPattern(MLIRContext *context,
                                      int64_t vecSizeParam)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), 1, context) {
    vecSize = vecSizeParam;
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
    const Value c2 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(2));
    const Value c3 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(3));
    const Value c4 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(4));
    const Value c5 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(5));
    const Value c6 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(6));
    const Value c7 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(7));

    // Note: the unroll factor is not the same as the vector size.
    const Value unroll =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(8));

    // Get input A, B, C.
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);

    // Create DimOp for m, n, k.
    const Value m = rewriter.create<memref::DimOp>(loc, A, c0);
    const Value n = rewriter.create<memref::DimOp>(loc, C, c1);
    const Value k = rewriter.create<memref::DimOp>(loc, A, c1);

    // Define step.
    const Value step = rewriter.create<arith::ConstantIndexOp>(loc, vecSize);

    // Get element type and create vector type.
    ShapedType ATy = cast<ShapedType>(A.getType());
    Type eleTy = ATy.getElementType();
    VectorType vectorTy = VectorType::get({vecSize}, eleTy);
    FloatType eleFloatTy =
        eleTy.isF32()
            ? static_cast<FloatType>(Float32Type::get(rewriter.getContext()))
            : static_cast<FloatType>(Float64Type::get(rewriter.getContext()));

    // Create parallel op for m dimension with unroll factor equal to 8.
    auto parOp = rewriter.create<scf::ParallelOp>(
        loc,
        /*lowerBounds=*/ValueRange{c0},
        /*upperBounds=*/ValueRange{m},
        /*steps=*/ValueRange{unroll},
        [&](OpBuilder &builder, Location loc, ValueRange mIdx) {
          auto mIdx1 = rewriter.create<arith::AddIOp>(loc, mIdx[0], c1);
          auto mIdx2 = rewriter.create<arith::AddIOp>(loc, mIdx[0], c2);
          auto mIdx3 = rewriter.create<arith::AddIOp>(loc, mIdx[0], c3);
          auto mIdx4 = rewriter.create<arith::AddIOp>(loc, mIdx[0], c4);
          auto mIdx5 = rewriter.create<arith::AddIOp>(loc, mIdx[0], c5);
          auto mIdx6 = rewriter.create<arith::AddIOp>(loc, mIdx[0], c6);
          auto mIdx7 = rewriter.create<arith::AddIOp>(loc, mIdx[0], c7);

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
                auto sumInitVec0 = rewriter.create<vector::LoadOp>(
                    loc, vectorTy, C, ValueRange{mIdx[0], iv});
                auto sumInitVec1 = rewriter.create<vector::LoadOp>(
                    loc, vectorTy, C, ValueRange{mIdx1, iv});
                auto sumInitVec2 = rewriter.create<vector::LoadOp>(
                    loc, vectorTy, C, ValueRange{mIdx2, iv});
                auto sumInitVec3 = rewriter.create<vector::LoadOp>(
                    loc, vectorTy, C, ValueRange{mIdx3, iv});
                auto sumInitVec4 = rewriter.create<vector::LoadOp>(
                    loc, vectorTy, C, ValueRange{mIdx4, iv});
                auto sumInitVec5 = rewriter.create<vector::LoadOp>(
                    loc, vectorTy, C, ValueRange{mIdx5, iv});
                auto sumInitVec6 = rewriter.create<vector::LoadOp>(
                    loc, vectorTy, C, ValueRange{mIdx6, iv});
                auto sumInitVec7 = rewriter.create<vector::LoadOp>(
                    loc, vectorTy, C, ValueRange{mIdx7, iv});

                auto sumIterVecs = rewriter.create<scf::ForOp>(
                    loc,
                    /*lowerBound=*/c0,
                    /*upperBound=*/k,
                    /*step=*/c1,
                    /*initArgs=*/
                    ValueRange{sumInitVec0, sumInitVec1, sumInitVec2,
                               sumInitVec3, sumInitVec4, sumInitVec5,
                               sumInitVec6, sumInitVec7},
                    [&](OpBuilder &builder, Location loc, Value kIdx,
                        ValueRange iterArgs) {
                      auto aEle0 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{mIdx[0], kIdx});
                      auto aEle1 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{mIdx1, kIdx});
                      auto aEle2 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{mIdx2, kIdx});
                      auto aEle3 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{mIdx3, kIdx});
                      auto aEle4 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{mIdx4, kIdx});
                      auto aEle5 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{mIdx5, kIdx});
                      auto aEle6 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{mIdx6, kIdx});
                      auto aEle7 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{mIdx7, kIdx});

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
                          loc, vectorTy, B, ValueRange{kIdx, iv});

                      auto resSumVec0 = rewriter.create<vector::FMAOp>(
                          loc, aVec0, bVec, iterArgs[0]);
                      auto resSumVec1 = rewriter.create<vector::FMAOp>(
                          loc, aVec1, bVec, iterArgs[1]);
                      auto resSumVec2 = rewriter.create<vector::FMAOp>(
                          loc, aVec2, bVec, iterArgs[2]);
                      auto resSumVec3 = rewriter.create<vector::FMAOp>(
                          loc, aVec3, bVec, iterArgs[3]);
                      auto resSumVec4 = rewriter.create<vector::FMAOp>(
                          loc, aVec4, bVec, iterArgs[4]);
                      auto resSumVec5 = rewriter.create<vector::FMAOp>(
                          loc, aVec5, bVec, iterArgs[5]);
                      auto resSumVec6 = rewriter.create<vector::FMAOp>(
                          loc, aVec6, bVec, iterArgs[6]);
                      auto resSumVec7 = rewriter.create<vector::FMAOp>(
                          loc, aVec7, bVec, iterArgs[7]);

                      builder.create<scf::YieldOp>(
                          loc, ValueRange{resSumVec0, resSumVec1, resSumVec2,
                                          resSumVec3, resSumVec4, resSumVec5,
                                          resSumVec6, resSumVec7});
                    });

                auto sumIterVec0 = rewriter.create<vector::StoreOp>(
                    loc, sumIterVecs.getResult(0), C, ValueRange{mIdx[0], iv});
                auto sumIterVec1 = rewriter.create<vector::StoreOp>(
                    loc, sumIterVecs.getResult(1), C, ValueRange{mIdx1, iv});
                auto sumIterVec2 = rewriter.create<vector::StoreOp>(
                    loc, sumIterVecs.getResult(2), C, ValueRange{mIdx2, iv});
                auto sumIterVec3 = rewriter.create<vector::StoreOp>(
                    loc, sumIterVecs.getResult(3), C, ValueRange{mIdx3, iv});
                auto sumIterVec4 = rewriter.create<vector::StoreOp>(
                    loc, sumIterVecs.getResult(4), C, ValueRange{mIdx4, iv});
                auto sumIterVec5 = rewriter.create<vector::StoreOp>(
                    loc, sumIterVecs.getResult(5), C, ValueRange{mIdx5, iv});
                auto sumIterVec6 = rewriter.create<vector::StoreOp>(
                    loc, sumIterVecs.getResult(6), C, ValueRange{mIdx6, iv});
                auto sumIterVec7 = rewriter.create<vector::StoreOp>(
                    loc, sumIterVecs.getResult(7), C, ValueRange{mIdx7, iv});

                auto kNext = rewriter.create<arith::AddIOp>(loc, iv, step);
                builder.create<scf::YieldOp>(loc, ValueRange{kNext});
              });

          // Tail processing.
          builder.create<scf::ForOp>(
              loc, nIterIdx.getResult(0), n, c1, std::nullopt,
              [&](OpBuilder &builder, Location loc, Value iv,
                  ValueRange iterArgs) {
                auto sumInit0 = rewriter.create<memref::LoadOp>(
                    loc, eleFloatTy, C, ValueRange{mIdx[0], iv});
                auto sumInit1 = rewriter.create<memref::LoadOp>(
                    loc, eleFloatTy, C, ValueRange{mIdx1, iv});
                auto sumInit2 = rewriter.create<memref::LoadOp>(
                    loc, eleFloatTy, C, ValueRange{mIdx2, iv});
                auto sumInit3 = rewriter.create<memref::LoadOp>(
                    loc, eleFloatTy, C, ValueRange{mIdx3, iv});
                auto sumInit4 = rewriter.create<memref::LoadOp>(
                    loc, eleFloatTy, C, ValueRange{mIdx4, iv});
                auto sumInit5 = rewriter.create<memref::LoadOp>(
                    loc, eleFloatTy, C, ValueRange{mIdx5, iv});
                auto sumInit6 = rewriter.create<memref::LoadOp>(
                    loc, eleFloatTy, C, ValueRange{mIdx6, iv});
                auto sumInit7 = rewriter.create<memref::LoadOp>(
                    loc, eleFloatTy, C, ValueRange{mIdx7, iv});
                auto sumIterVecs = rewriter.create<scf::ForOp>(
                    loc,
                    /*lowerBound=*/c0,
                    /*upperBound=*/k,
                    /*step=*/c1,
                    /*initArgs=*/
                    ValueRange{sumInit0, sumInit1, sumInit2, sumInit3, sumInit4,
                               sumInit5, sumInit6, sumInit7},
                    [&](OpBuilder &builder, Location loc, Value kIdx,
                        ValueRange iterArgs) {
                      auto aEle0 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{mIdx[0], kIdx});
                      auto aEle1 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{mIdx1, kIdx});
                      auto aEle2 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{mIdx2, kIdx});
                      auto aEle3 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{mIdx3, kIdx});
                      auto aEle4 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{mIdx4, kIdx});
                      auto aEle5 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{mIdx5, kIdx});
                      auto aEle6 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{mIdx6, kIdx});
                      auto aEle7 = rewriter.create<memref::LoadOp>(
                          loc, A, ValueRange{mIdx7, kIdx});

                      auto bEle = rewriter.create<memref::LoadOp>(
                          loc, B, ValueRange{kIdx, iv});

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

                      auto resSum0 = rewriter.create<arith::AddFOp>(
                          loc, tmpEle0, iterArgs[0]);
                      auto resSum1 = rewriter.create<arith::AddFOp>(
                          loc, tmpEle1, iterArgs[1]);
                      auto resSum2 = rewriter.create<arith::AddFOp>(
                          loc, tmpEle2, iterArgs[2]);
                      auto resSum3 = rewriter.create<arith::AddFOp>(
                          loc, tmpEle3, iterArgs[3]);
                      auto resSum4 = rewriter.create<arith::AddFOp>(
                          loc, tmpEle4, iterArgs[4]);
                      auto resSum5 = rewriter.create<arith::AddFOp>(
                          loc, tmpEle5, iterArgs[5]);
                      auto resSum6 = rewriter.create<arith::AddFOp>(
                          loc, tmpEle6, iterArgs[6]);
                      auto resSum7 = rewriter.create<arith::AddFOp>(
                          loc, tmpEle7, iterArgs[7]);

                      builder.create<scf::YieldOp>(
                          loc, ValueRange{resSum0, resSum1, resSum2, resSum3,
                                          resSum4, resSum5, resSum6, resSum7});
                    });

                auto sumIter0 = rewriter.create<memref::StoreOp>(
                    loc, sumIterVecs.getResult(0), C, ValueRange{mIdx[0], iv});
                auto sumIter1 = rewriter.create<memref::StoreOp>(
                    loc, sumIterVecs.getResult(1), C, ValueRange{mIdx1, iv});
                auto sumIter2 = rewriter.create<memref::StoreOp>(
                    loc, sumIterVecs.getResult(2), C, ValueRange{mIdx2, iv});
                auto sumIter3 = rewriter.create<memref::StoreOp>(
                    loc, sumIterVecs.getResult(3), C, ValueRange{mIdx3, iv});
                auto sumIter4 = rewriter.create<memref::StoreOp>(
                    loc, sumIterVecs.getResult(4), C, ValueRange{mIdx4, iv});
                auto sumIter5 = rewriter.create<memref::StoreOp>(
                    loc, sumIterVecs.getResult(5), C, ValueRange{mIdx5, iv});
                auto sumIter6 = rewriter.create<memref::StoreOp>(
                    loc, sumIterVecs.getResult(6), C, ValueRange{mIdx6, iv});
                auto sumIter7 = rewriter.create<memref::StoreOp>(
                    loc, sumIterVecs.getResult(7), C, ValueRange{mIdx7, iv});

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

  RewritePatternSet patterns(context);
  patterns.add<MatMulVectorizationPattern>(context, vecSize);

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
