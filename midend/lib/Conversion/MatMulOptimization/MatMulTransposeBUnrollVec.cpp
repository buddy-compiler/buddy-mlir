//===- MatMulTransposeBUnrollVec.cpp
//--------------------------------------------===//
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
#include <iostream>
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

class MatMulTransposeBUnrollVecPattern : public ConversionPattern {
public:
  explicit MatMulTransposeBUnrollVecPattern(MLIRContext *context,
                                            int64_t vecSizeParam)
      : ConversionPattern(linalg::MatmulTransposeBOp::getOperationName(), 1,
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

    // Define constants.
    const Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    const Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    const Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    const Value c3 = rewriter.create<arith::ConstantIndexOp>(loc, 3);
    const Value c4 = rewriter.create<arith::ConstantIndexOp>(loc, 4);
    const Value c5 = rewriter.create<arith::ConstantIndexOp>(loc, 5);
    const Value c6 = rewriter.create<arith::ConstantIndexOp>(loc, 6);
    const Value c7 = rewriter.create<arith::ConstantIndexOp>(loc, 7);
    const Value vlStep = rewriter.create<arith::ConstantIndexOp>(loc, vecSize);
    const Value initVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    const Value unroll =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(8));
    // const Value initVec = rewriter.create<arith::ConstantOp>(
    //     loc, vectorTy, DenseElementsAttr::get(vectorTy, APFloat(0.0f)));
    Value initVec = rewriter.create<SplatOp>(loc, vectorTy, initVal);

    // Get dimensions of input tensors.
    Value aRow = rewriter.create<memref::DimOp>(loc, A, c0);
    Value aCol = rewriter.create<memref::DimOp>(loc, A, c1);
    Value bCol = rewriter.create<memref::DimOp>(loc, C, c1);

    const AffineExpr d0 = rewriter.getAffineDimExpr(0);
    const AffineExpr d1 = rewriter.getAffineDimExpr(1);

    AffineMap map1 = AffineMap::get(2, 0, d0 + d1, ctx);
    AffineMap map2 = AffineMap::get(2, 0, d0 - d1, ctx);

    Value upperBound_tmp =
        rewriter.create<AffineApplyOp>(loc, map2, ValueRange{aCol, vlStep});
    Value upperBound = rewriter.create<AffineApplyOp>(
        loc, map1, ValueRange{upperBound_tmp, c1});

    auto parOp = rewriter.create<scf::ParallelOp>(
        loc,
        /*lowerBounds=*/ValueRange{c0},
        /*upperBounds=*/ValueRange{aRow},
        /*steps=*/ValueRange{unroll},
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          auto aRowIdx1 = builder.create<affine::AffineApplyOp>(
              loc, map1, ValueRange{ivs[0], c1});
          auto aRowIdx2 = builder.create<affine::AffineApplyOp>(
              loc, map1, ValueRange{ivs[0], c2});
          auto aRowIdx3 = builder.create<affine::AffineApplyOp>(
              loc, map1, ValueRange{ivs[0], c3});
          auto aRowIdx4 = builder.create<affine::AffineApplyOp>(
              loc, map1, ValueRange{ivs[0], c4});
          auto aRowIdx5 = builder.create<affine::AffineApplyOp>(
              loc, map1, ValueRange{ivs[0], c5});
          auto aRowIdx6 = builder.create<affine::AffineApplyOp>(
              loc, map1, ValueRange{ivs[0], c6});
          auto aRowIdx7 = builder.create<affine::AffineApplyOp>(
              loc, map1, ValueRange{ivs[0], c7});

          builder.create<scf::ForOp>(
              loc, c0, bCol,
              /*Step=*/c1, std::nullopt,
              [&](OpBuilder &builder, Location loc, Value iv0,
                  ValueRange itrArgs0) {
                auto iterVals = builder.create<scf::ForOp>(
                    loc, c0, upperBound,
                    /*Step=*/vlStep,
                    ValueRange{initVec, initVec, initVec, initVec, initVec,
                               initVec, initVec, initVec, c0},
                    [&](OpBuilder &builder, Location loc, Value iv1,
                        ValueRange itrArgs) {
                      Value aVec0 = builder.create<vector::LoadOp>(
                          loc, vectorTy, A, ValueRange{ivs[0], iv1});
                      Value aVec1 = builder.create<vector::LoadOp>(
                          loc, vectorTy, A, ValueRange{aRowIdx1, iv1});
                      Value aVec2 = builder.create<vector::LoadOp>(
                          loc, vectorTy, A, ValueRange{aRowIdx2, iv1});
                      Value aVec3 = builder.create<vector::LoadOp>(
                          loc, vectorTy, A, ValueRange{aRowIdx3, iv1});
                      Value aVec4 = builder.create<vector::LoadOp>(
                          loc, vectorTy, A, ValueRange{aRowIdx4, iv1});
                      Value aVec5 = builder.create<vector::LoadOp>(
                          loc, vectorTy, A, ValueRange{aRowIdx5, iv1});
                      Value aVec6 = builder.create<vector::LoadOp>(
                          loc, vectorTy, A, ValueRange{aRowIdx6, iv1});
                      Value aVec7 = builder.create<vector::LoadOp>(
                          loc, vectorTy, A, ValueRange{aRowIdx7, iv1});
                      Value bVec = builder.create<vector::LoadOp>(
                          loc, vectorTy, B, ValueRange{iv0, iv1});
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
                            loc, mulVec0, itrArgs[0]);
                        Value mulVec1 =
                            builder.create<arith::MulIOp>(loc, aVec1, bVec);
                        computedVec1 = builder.create<arith::AddIOp>(
                            loc, mulVec1, itrArgs[1]);
                        Value mulVec2 =
                            builder.create<arith::MulIOp>(loc, aVec2, bVec);
                        computedVec2 = builder.create<arith::AddIOp>(
                            loc, mulVec2, itrArgs[2]);
                        Value mulVec3 =
                            builder.create<arith::MulIOp>(loc, aVec3, bVec);
                        computedVec3 = builder.create<arith::AddIOp>(
                            loc, mulVec3, itrArgs[3]);
                        Value mulVec4 =
                            builder.create<arith::MulIOp>(loc, aVec4, bVec);
                        computedVec4 = builder.create<arith::AddIOp>(
                            loc, mulVec4, itrArgs[4]);
                        Value mulVec5 =
                            builder.create<arith::MulIOp>(loc, aVec5, bVec);
                        computedVec5 = builder.create<arith::AddIOp>(
                            loc, mulVec5, itrArgs[5]);
                        Value mulVec6 =
                            builder.create<arith::MulIOp>(loc, aVec6, bVec);
                        computedVec6 = builder.create<arith::AddIOp>(
                            loc, mulVec6, itrArgs[6]);
                        Value mulVec7 =
                            builder.create<arith::MulIOp>(loc, aVec7, bVec);
                        computedVec7 = builder.create<arith::AddIOp>(
                            loc, mulVec7, itrArgs[7]);
                      } else {
                        computedVec0 = builder.create<vector::FMAOp>(
                            loc, aVec0, bVec, itrArgs[0]);
                        computedVec1 = builder.create<vector::FMAOp>(
                            loc, aVec1, bVec, itrArgs[1]);
                        computedVec2 = builder.create<vector::FMAOp>(
                            loc, aVec2, bVec, itrArgs[2]);
                        computedVec3 = builder.create<vector::FMAOp>(
                            loc, aVec3, bVec, itrArgs[3]);
                        computedVec4 = builder.create<vector::FMAOp>(
                            loc, aVec4, bVec, itrArgs[4]);
                        computedVec5 = builder.create<vector::FMAOp>(
                            loc, aVec5, bVec, itrArgs[5]);
                        computedVec6 = builder.create<vector::FMAOp>(
                            loc, aVec6, bVec, itrArgs[6]);
                        computedVec7 = builder.create<vector::FMAOp>(
                            loc, aVec7, bVec, itrArgs[7]);
                      }
                      Value idx = builder.create<affine::AffineApplyOp>(
                          loc, map1, ValueRange{iv1, vlStep});
                      builder.create<scf::YieldOp>(
                          loc,
                          ValueRange{computedVec0, computedVec1, computedVec2,
                                     computedVec3, computedVec4, computedVec5,
                                     computedVec6, computedVec7, idx});
                    });
                auto tmpVals = iterVals.getResults();
                Value reduction0 = builder.create<vector::ReductionOp>(
                    loc, CombiningKind::ADD, tmpVals[0], initVal,
                    arith::FastMathFlags::reassoc);
                Value reduction1 = builder.create<vector::ReductionOp>(
                    loc, CombiningKind::ADD, tmpVals[1], initVal,
                    arith::FastMathFlags::reassoc);
                Value reduction2 = builder.create<vector::ReductionOp>(
                    loc, CombiningKind::ADD, tmpVals[2], initVal,
                    arith::FastMathFlags::reassoc);
                Value reduction3 = builder.create<vector::ReductionOp>(
                    loc, CombiningKind::ADD, tmpVals[3], initVal,
                    arith::FastMathFlags::reassoc);
                Value reduction4 = builder.create<vector::ReductionOp>(
                    loc, CombiningKind::ADD, tmpVals[4], initVal,
                    arith::FastMathFlags::reassoc);
                Value reduction5 = builder.create<vector::ReductionOp>(
                    loc, CombiningKind::ADD, tmpVals[5], initVal,
                    arith::FastMathFlags::reassoc);
                Value reduction6 = builder.create<vector::ReductionOp>(
                    loc, CombiningKind::ADD, tmpVals[6], initVal,
                    arith::FastMathFlags::reassoc);
                Value reduction7 = builder.create<vector::ReductionOp>(
                    loc, CombiningKind::ADD, tmpVals[7], initVal,
                    arith::FastMathFlags::reassoc);
                Value idx = tmpVals[8];
                auto sumIters = builder.create<scf::ForOp>(
                    loc, idx, aCol,
                    /*Step=*/c1,
                    ValueRange{reduction0, reduction1, reduction2, reduction3,
                               reduction4, reduction5, reduction6, reduction7},
                    [&](OpBuilder &builder, Location loc, Value iv1,
                        ValueRange iterArgs) {
                      auto aEle0 = builder.create<memref::LoadOp>(
                          loc, A, ValueRange{ivs[0], iv1});
                      auto aEle1 = builder.create<memref::LoadOp>(
                          loc, A, ValueRange{aRowIdx1, iv1});
                      auto aEle2 = builder.create<memref::LoadOp>(
                          loc, A, ValueRange{aRowIdx2, iv1});
                      auto aEle3 = builder.create<memref::LoadOp>(
                          loc, A, ValueRange{aRowIdx3, iv1});
                      auto aEle4 = builder.create<memref::LoadOp>(
                          loc, A, ValueRange{aRowIdx4, iv1});
                      auto aEle5 = builder.create<memref::LoadOp>(
                          loc, A, ValueRange{aRowIdx5, iv1});
                      auto aEle6 = builder.create<memref::LoadOp>(
                          loc, A, ValueRange{aRowIdx6, iv1});
                      auto aEle7 = builder.create<memref::LoadOp>(
                          loc, A, ValueRange{aRowIdx7, iv1});

                      auto bEle = builder.create<memref::LoadOp>(
                          loc, B, ValueRange{iv0, iv1});

                      auto tmpEle0 =
                          builder.create<arith::MulFOp>(loc, aEle0, bEle);
                      auto tmpEle1 =
                          builder.create<arith::MulFOp>(loc, aEle1, bEle);
                      auto tmpEle2 =
                          builder.create<arith::MulFOp>(loc, aEle2, bEle);
                      auto tmpEle3 =
                          builder.create<arith::MulFOp>(loc, aEle3, bEle);
                      auto tmpEle4 =
                          builder.create<arith::MulFOp>(loc, aEle4, bEle);
                      auto tmpEle5 =
                          builder.create<arith::MulFOp>(loc, aEle5, bEle);
                      auto tmpEle6 =
                          builder.create<arith::MulFOp>(loc, aEle6, bEle);
                      auto tmpEle7 =
                          builder.create<arith::MulFOp>(loc, aEle7, bEle);

                      auto resSum0 = builder.create<arith::AddFOp>(loc, tmpEle0,
                                                                   iterArgs[0]);
                      auto resSum1 = builder.create<arith::AddFOp>(loc, tmpEle1,
                                                                   iterArgs[1]);
                      auto resSum2 = builder.create<arith::AddFOp>(loc, tmpEle2,
                                                                   iterArgs[2]);
                      auto resSum3 = builder.create<arith::AddFOp>(loc, tmpEle3,
                                                                   iterArgs[3]);
                      auto resSum4 = builder.create<arith::AddFOp>(loc, tmpEle4,
                                                                   iterArgs[4]);
                      auto resSum5 = builder.create<arith::AddFOp>(loc, tmpEle5,
                                                                   iterArgs[5]);
                      auto resSum6 = builder.create<arith::AddFOp>(loc, tmpEle6,
                                                                   iterArgs[6]);
                      auto resSum7 = builder.create<arith::AddFOp>(loc, tmpEle7,
                                                                   iterArgs[7]);

                      builder.create<scf::YieldOp>(
                          loc, ValueRange{resSum0, resSum1, resSum2, resSum3,
                                          resSum4, resSum5, resSum6, resSum7});
                    });
                auto sumIter0 = builder.create<memref::StoreOp>(
                    loc, sumIters.getResult(0), C, ValueRange{ivs[0], iv0});
                auto sumIter1 = builder.create<memref::StoreOp>(
                    loc, sumIters.getResult(1), C, ValueRange{aRowIdx1, iv0});
                auto sumIter2 = builder.create<memref::StoreOp>(
                    loc, sumIters.getResult(2), C, ValueRange{aRowIdx2, iv0});
                auto sumIter3 = builder.create<memref::StoreOp>(
                    loc, sumIters.getResult(3), C, ValueRange{aRowIdx3, iv0});
                auto sumIter4 = builder.create<memref::StoreOp>(
                    loc, sumIters.getResult(4), C, ValueRange{aRowIdx4, iv0});
                auto sumIter5 = builder.create<memref::StoreOp>(
                    loc, sumIters.getResult(5), C, ValueRange{aRowIdx5, iv0});
                auto sumIter6 = builder.create<memref::StoreOp>(
                    loc, sumIters.getResult(6), C, ValueRange{aRowIdx6, iv0});
                auto sumIter7 = builder.create<memref::StoreOp>(
                    loc, sumIters.getResult(7), C, ValueRange{aRowIdx7, iv0});

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
// MatMulTransposeBUnrollVecPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg pooling operations to mixture of
/// Affine + Vector operations.
namespace {
class MatMulTransposeBUnrollVecPass
    : public PassWrapper<MatMulTransposeBUnrollVecPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulTransposeBUnrollVecPass)
  StringRef getArgument() const final {
    return "matmul-transpose-b-unroll-vec";
  }
  StringRef getDescription() const final {
    return "MatMul TransposeB Unroll Vectorization.";
  }
  MatMulTransposeBUnrollVecPass() = default;
  MatMulTransposeBUnrollVecPass(const MatMulTransposeBUnrollVecPass &) {}
  explicit MatMulTransposeBUnrollVecPass(int64_t vecSizeParam) {
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

void MatMulTransposeBUnrollVecPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                       scf::SCFDialect, memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<MatMulTransposeBUnrollVecPattern>(context, vecSize);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerMatMulTransposeBUnrollVecPass() {
  PassRegistration<MatMulTransposeBUnrollVecPass>();
}
} // namespace buddy
} // namespace mlir
