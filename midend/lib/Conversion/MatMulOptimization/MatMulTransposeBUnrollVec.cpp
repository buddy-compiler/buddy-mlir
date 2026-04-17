//===- MatMulTransposeBUnrollVec.cpp---------------------------------------===//
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
// This file implements the MatMulTransposeB optimization.
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
    Type elementType =
        mlir::cast<mlir::MemRefType>(A.getType()).getElementType();
    VectorType vectorTy = mlir::VectorType::get({vecSize}, elementType);

    // Define constants.
    const Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    const Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    const Value c2 = arith::ConstantIndexOp::create(rewriter, loc, 2);
    const Value c3 = arith::ConstantIndexOp::create(rewriter, loc, 3);
    const Value c4 = arith::ConstantIndexOp::create(rewriter, loc, 4);
    const Value c5 = arith::ConstantIndexOp::create(rewriter, loc, 5);
    const Value c6 = arith::ConstantIndexOp::create(rewriter, loc, 6);
    const Value c7 = arith::ConstantIndexOp::create(rewriter, loc, 7);
    const Value vlStep = arith::ConstantIndexOp::create(rewriter, loc, vecSize);
    const Value initVal = arith::ConstantOp::create(rewriter, 
        loc, rewriter.getZeroAttr(elementType));
    const Value unroll =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(8));
    // const Value initVec = arith::ConstantOp::create(rewriter, 
    //     loc, vectorTy, DenseElementsAttr::get(vectorTy, APFloat(0.0f)));
    Value initVec = vector::BroadcastOp::create(rewriter, loc, vectorTy, initVal);

    // Get dimensions of input tensors.
    Value aRow = memref::DimOp::create(rewriter, loc, A, c0);
    Value aCol = memref::DimOp::create(rewriter, loc, A, c1);
    Value bCol = memref::DimOp::create(rewriter, loc, C, c1);

    const AffineExpr d0 = rewriter.getAffineDimExpr(0);
    const AffineExpr d1 = rewriter.getAffineDimExpr(1);

    AffineMap map1 = AffineMap::get(2, 0, d0 + d1, ctx);
    AffineMap map2 = AffineMap::get(2, 0, d0 - d1, ctx);

    Value upperBound_tmp =
        AffineApplyOp::create(rewriter, loc, map2, ValueRange{aCol, vlStep});
    Value upperBound = AffineApplyOp::create(rewriter, 
        loc, map1, ValueRange{upperBound_tmp, c1});

    auto parOp = scf::ParallelOp::create(rewriter, 
        loc,
        /*lowerBounds=*/ValueRange{c0},
        /*upperBounds=*/ValueRange{aRow},
        /*steps=*/ValueRange{unroll},
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          auto aRowIdx1 = affine::AffineApplyOp::create(builder, 
              loc, map1, ValueRange{ivs[0], c1});
          auto aRowIdx2 = affine::AffineApplyOp::create(builder, 
              loc, map1, ValueRange{ivs[0], c2});
          auto aRowIdx3 = affine::AffineApplyOp::create(builder, 
              loc, map1, ValueRange{ivs[0], c3});
          auto aRowIdx4 = affine::AffineApplyOp::create(builder, 
              loc, map1, ValueRange{ivs[0], c4});
          auto aRowIdx5 = affine::AffineApplyOp::create(builder, 
              loc, map1, ValueRange{ivs[0], c5});
          auto aRowIdx6 = affine::AffineApplyOp::create(builder, 
              loc, map1, ValueRange{ivs[0], c6});
          auto aRowIdx7 = affine::AffineApplyOp::create(builder, 
              loc, map1, ValueRange{ivs[0], c7});

          scf::ForOp::create(builder, 
              loc, c0, bCol,
              /*Step=*/c1, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value iv0,
                  ValueRange itrArgs0) {
                auto iterVals = scf::ForOp::create(builder, 
                    loc, c0, upperBound,
                    /*Step=*/vlStep,
                    ValueRange{initVec, initVec, initVec, initVec, initVec,
                               initVec, initVec, initVec, c0},
                    [&](OpBuilder &builder, Location loc, Value iv1,
                        ValueRange itrArgs) {
                      Value aVec0 = vector::LoadOp::create(builder, 
                          loc, vectorTy, A, ValueRange{ivs[0], iv1});
                      Value aVec1 = vector::LoadOp::create(builder, 
                          loc, vectorTy, A, ValueRange{aRowIdx1, iv1});
                      Value aVec2 = vector::LoadOp::create(builder, 
                          loc, vectorTy, A, ValueRange{aRowIdx2, iv1});
                      Value aVec3 = vector::LoadOp::create(builder, 
                          loc, vectorTy, A, ValueRange{aRowIdx3, iv1});
                      Value aVec4 = vector::LoadOp::create(builder, 
                          loc, vectorTy, A, ValueRange{aRowIdx4, iv1});
                      Value aVec5 = vector::LoadOp::create(builder, 
                          loc, vectorTy, A, ValueRange{aRowIdx5, iv1});
                      Value aVec6 = vector::LoadOp::create(builder, 
                          loc, vectorTy, A, ValueRange{aRowIdx6, iv1});
                      Value aVec7 = vector::LoadOp::create(builder, 
                          loc, vectorTy, A, ValueRange{aRowIdx7, iv1});
                      Value bVec = vector::LoadOp::create(builder, 
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
                            arith::MulIOp::create(builder, loc, aVec0, bVec);
                        computedVec0 = arith::AddIOp::create(builder, 
                            loc, mulVec0, itrArgs[0]);
                        Value mulVec1 =
                            arith::MulIOp::create(builder, loc, aVec1, bVec);
                        computedVec1 = arith::AddIOp::create(builder, 
                            loc, mulVec1, itrArgs[1]);
                        Value mulVec2 =
                            arith::MulIOp::create(builder, loc, aVec2, bVec);
                        computedVec2 = arith::AddIOp::create(builder, 
                            loc, mulVec2, itrArgs[2]);
                        Value mulVec3 =
                            arith::MulIOp::create(builder, loc, aVec3, bVec);
                        computedVec3 = arith::AddIOp::create(builder, 
                            loc, mulVec3, itrArgs[3]);
                        Value mulVec4 =
                            arith::MulIOp::create(builder, loc, aVec4, bVec);
                        computedVec4 = arith::AddIOp::create(builder, 
                            loc, mulVec4, itrArgs[4]);
                        Value mulVec5 =
                            arith::MulIOp::create(builder, loc, aVec5, bVec);
                        computedVec5 = arith::AddIOp::create(builder, 
                            loc, mulVec5, itrArgs[5]);
                        Value mulVec6 =
                            arith::MulIOp::create(builder, loc, aVec6, bVec);
                        computedVec6 = arith::AddIOp::create(builder, 
                            loc, mulVec6, itrArgs[6]);
                        Value mulVec7 =
                            arith::MulIOp::create(builder, loc, aVec7, bVec);
                        computedVec7 = arith::AddIOp::create(builder, 
                            loc, mulVec7, itrArgs[7]);
                      } else {
                        computedVec0 = vector::FMAOp::create(builder, 
                            loc, aVec0, bVec, itrArgs[0]);
                        computedVec1 = vector::FMAOp::create(builder, 
                            loc, aVec1, bVec, itrArgs[1]);
                        computedVec2 = vector::FMAOp::create(builder, 
                            loc, aVec2, bVec, itrArgs[2]);
                        computedVec3 = vector::FMAOp::create(builder, 
                            loc, aVec3, bVec, itrArgs[3]);
                        computedVec4 = vector::FMAOp::create(builder, 
                            loc, aVec4, bVec, itrArgs[4]);
                        computedVec5 = vector::FMAOp::create(builder, 
                            loc, aVec5, bVec, itrArgs[5]);
                        computedVec6 = vector::FMAOp::create(builder, 
                            loc, aVec6, bVec, itrArgs[6]);
                        computedVec7 = vector::FMAOp::create(builder, 
                            loc, aVec7, bVec, itrArgs[7]);
                      }
                      Value idx = affine::AffineApplyOp::create(builder, 
                          loc, map1, ValueRange{iv1, vlStep});
                      scf::YieldOp::create(builder, 
                          loc,
                          ValueRange{computedVec0, computedVec1, computedVec2,
                                     computedVec3, computedVec4, computedVec5,
                                     computedVec6, computedVec7, idx});
                    });
                auto tmpVals = iterVals.getResults();
                Value reduction0 = vector::ReductionOp::create(builder, 
                    loc, CombiningKind::ADD, tmpVals[0], initVal,
                    arith::FastMathFlags::reassoc);
                Value reduction1 = vector::ReductionOp::create(builder, 
                    loc, CombiningKind::ADD, tmpVals[1], initVal,
                    arith::FastMathFlags::reassoc);
                Value reduction2 = vector::ReductionOp::create(builder, 
                    loc, CombiningKind::ADD, tmpVals[2], initVal,
                    arith::FastMathFlags::reassoc);
                Value reduction3 = vector::ReductionOp::create(builder, 
                    loc, CombiningKind::ADD, tmpVals[3], initVal,
                    arith::FastMathFlags::reassoc);
                Value reduction4 = vector::ReductionOp::create(builder, 
                    loc, CombiningKind::ADD, tmpVals[4], initVal,
                    arith::FastMathFlags::reassoc);
                Value reduction5 = vector::ReductionOp::create(builder, 
                    loc, CombiningKind::ADD, tmpVals[5], initVal,
                    arith::FastMathFlags::reassoc);
                Value reduction6 = vector::ReductionOp::create(builder, 
                    loc, CombiningKind::ADD, tmpVals[6], initVal,
                    arith::FastMathFlags::reassoc);
                Value reduction7 = vector::ReductionOp::create(builder, 
                    loc, CombiningKind::ADD, tmpVals[7], initVal,
                    arith::FastMathFlags::reassoc);
                Value idx = tmpVals[8];
                auto sumIters = scf::ForOp::create(builder, 
                    loc, idx, aCol,
                    /*Step=*/c1,
                    ValueRange{reduction0, reduction1, reduction2, reduction3,
                               reduction4, reduction5, reduction6, reduction7},
                    [&](OpBuilder &builder, Location loc, Value iv1,
                        ValueRange iterArgs) {
                      auto aEle0 = memref::LoadOp::create(builder, 
                          loc, A, ValueRange{ivs[0], iv1});
                      auto aEle1 = memref::LoadOp::create(builder, 
                          loc, A, ValueRange{aRowIdx1, iv1});
                      auto aEle2 = memref::LoadOp::create(builder, 
                          loc, A, ValueRange{aRowIdx2, iv1});
                      auto aEle3 = memref::LoadOp::create(builder, 
                          loc, A, ValueRange{aRowIdx3, iv1});
                      auto aEle4 = memref::LoadOp::create(builder, 
                          loc, A, ValueRange{aRowIdx4, iv1});
                      auto aEle5 = memref::LoadOp::create(builder, 
                          loc, A, ValueRange{aRowIdx5, iv1});
                      auto aEle6 = memref::LoadOp::create(builder, 
                          loc, A, ValueRange{aRowIdx6, iv1});
                      auto aEle7 = memref::LoadOp::create(builder, 
                          loc, A, ValueRange{aRowIdx7, iv1});

                      auto bEle = memref::LoadOp::create(builder, 
                          loc, B, ValueRange{iv0, iv1});

                      auto tmpEle0 =
                          arith::MulFOp::create(builder, loc, aEle0, bEle);
                      auto tmpEle1 =
                          arith::MulFOp::create(builder, loc, aEle1, bEle);
                      auto tmpEle2 =
                          arith::MulFOp::create(builder, loc, aEle2, bEle);
                      auto tmpEle3 =
                          arith::MulFOp::create(builder, loc, aEle3, bEle);
                      auto tmpEle4 =
                          arith::MulFOp::create(builder, loc, aEle4, bEle);
                      auto tmpEle5 =
                          arith::MulFOp::create(builder, loc, aEle5, bEle);
                      auto tmpEle6 =
                          arith::MulFOp::create(builder, loc, aEle6, bEle);
                      auto tmpEle7 =
                          arith::MulFOp::create(builder, loc, aEle7, bEle);

                      auto resSum0 = arith::AddFOp::create(builder, loc, tmpEle0,
                                                                   iterArgs[0]);
                      auto resSum1 = arith::AddFOp::create(builder, loc, tmpEle1,
                                                                   iterArgs[1]);
                      auto resSum2 = arith::AddFOp::create(builder, loc, tmpEle2,
                                                                   iterArgs[2]);
                      auto resSum3 = arith::AddFOp::create(builder, loc, tmpEle3,
                                                                   iterArgs[3]);
                      auto resSum4 = arith::AddFOp::create(builder, loc, tmpEle4,
                                                                   iterArgs[4]);
                      auto resSum5 = arith::AddFOp::create(builder, loc, tmpEle5,
                                                                   iterArgs[5]);
                      auto resSum6 = arith::AddFOp::create(builder, loc, tmpEle6,
                                                                   iterArgs[6]);
                      auto resSum7 = arith::AddFOp::create(builder, loc, tmpEle7,
                                                                   iterArgs[7]);

                      scf::YieldOp::create(builder, 
                          loc, ValueRange{resSum0, resSum1, resSum2, resSum3,
                                          resSum4, resSum5, resSum6, resSum7});
                    });
                auto sumIter0 = memref::StoreOp::create(builder, 
                    loc, sumIters.getResult(0), C, ValueRange{ivs[0], iv0});
                auto sumIter1 = memref::StoreOp::create(builder, 
                    loc, sumIters.getResult(1), C, ValueRange{aRowIdx1, iv0});
                auto sumIter2 = memref::StoreOp::create(builder, 
                    loc, sumIters.getResult(2), C, ValueRange{aRowIdx2, iv0});
                auto sumIter3 = memref::StoreOp::create(builder, 
                    loc, sumIters.getResult(3), C, ValueRange{aRowIdx3, iv0});
                auto sumIter4 = memref::StoreOp::create(builder, 
                    loc, sumIters.getResult(4), C, ValueRange{aRowIdx4, iv0});
                auto sumIter5 = memref::StoreOp::create(builder, 
                    loc, sumIters.getResult(5), C, ValueRange{aRowIdx5, iv0});
                auto sumIter6 = memref::StoreOp::create(builder, 
                    loc, sumIters.getResult(6), C, ValueRange{aRowIdx6, iv0});
                auto sumIter7 = memref::StoreOp::create(builder, 
                    loc, sumIters.getResult(7), C, ValueRange{aRowIdx7, iv0});

                scf::YieldOp::create(builder, loc);
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
