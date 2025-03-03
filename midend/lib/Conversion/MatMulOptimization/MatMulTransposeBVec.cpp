//===- MatMulTransposeBVec.cpp --------------------------------------------===//
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
// This file implements the Matmul_TransposeB vectorization.
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>

#include "Utils/Utils.h"

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
class MatMulTransposeBVecPattern : public ConversionPattern {
public:
  explicit MatMulTransposeBVecPattern(MLIRContext *context, int64_t vfParam,
                                      bool scalableParam)
      : ConversionPattern(linalg::MatmulTransposeBOp::getOperationName(), 1,
                          context) {
    vf = vfParam;
    scalable = scalableParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();
    // Get input A, B, C.
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);

    // Get shape of input and output.
    ShapedType ATy = A.getType().cast<ShapedType>();
    Type eleTy = ATy.getElementType();

    // the element type for mask vector.
    IntegerType i1 = IntegerType::get(ctx, 1);

    VectorType vectorTy = mlir::VectorType::get({vf}, eleTy, {scalable});
    VectorType vectorMaskTy = VectorType::get({vf}, i1, {scalable});

    const Value c0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value c1 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, vf);
    if (scalable) {
      Value vscale = rewriter.create<vector::VectorScaleOp>(loc);
      step = rewriter.create<arith::MulIOp>(loc, step, vscale);
    }

    const Value c0Ele = buddy::insertZeroConstantOp(ctx, rewriter, loc, eleTy);
    Value passthruVec = rewriter.create<SplatOp>(loc, vectorTy, c0Ele);

    const Value aRow = rewriter.create<memref::DimOp>(loc, A, c0);
    const Value bRow = rewriter.create<memref::DimOp>(loc, B, c0);
    const Value bCol = rewriter.create<memref::DimOp>(loc, B, c1);

    AffineMap vecTailMap;
    AffineExpr d0;
    bindDims(ctx, d0);
    if (scalable) {
      auto s0 = getAffineSymbolExpr(0, ctx);
      vecTailMap = AffineMap::get(1, 1, {d0.ceilDiv(s0)}, ctx);
    } else {
      vecTailMap = AffineMap::get(1, 0, {d0.ceilDiv(vf)}, ctx);
    }
    SmallVector<Value> innerUpperBoundOperands{bCol};
    if (scalable) {
      innerUpperBoundOperands.push_back(step);
    }

    SmallVector<Value, 8> lowerBounds(2, c0);
    SmallVector<Value, 8> uperBounds{aRow, bRow};
    SmallVector<int64_t, 8> steps(2, 1);

    affine::buildAffineLoopNest(
        rewriter, loc, lowerBounds, uperBounds, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          // Create loop based on vector size.
          auto innerLoop = builder.create<affine::AffineForOp>(
              loc, ValueRange{c0}, builder.getDimIdentityMap(),
              innerUpperBoundOperands, vecTailMap, 1, ValueRange{passthruVec},
              [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
                  ValueRange itrArgs) {
                Value acc = itrArgs[0];

                AffineExpr a, b;
                bindDims(ctx, a, b);
                AffineMap AVectorMap;
                if (scalable) {
                  auto s0 = getAffineSymbolExpr(0, ctx);
                  AVectorMap = AffineMap::get(
                      /*dimCount=*/2, /*symbolCount=*/1, {a, b * s0}, ctx);
                } else {
                  AVectorMap = AffineMap::get(
                      /*dimCount=*/2, /*symbolCount=*/0, {a, b * vf}, ctx);
                }
                // Check tail.
                AffineExpr m, k;
                bindDims(ctx, m, k);
                AffineMap BVectorMap;
                if (scalable) {
                  auto s0 = getAffineSymbolExpr(0, ctx);
                  BVectorMap = AffineMap::get(
                      /*dimCount=*/2, /*symbolCount=*/1, {m, k * s0}, ctx);
                } else {
                  BVectorMap = AffineMap::get(
                      /*dimCount=*/2, /*symbolCount=*/0, {m, k * vf}, ctx);
                }
                // Calculate the tail.
                Value bColCur = builder.create<arith::MulIOp>(loc, iv, step);
                Value tailLen =
                    builder.create<arith::SubIOp>(loc, bCol, bColCur);
                Value tailFlag = rewriter.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::sge, tailLen, step);
                // If the current column does not reach the tail.
                auto ifOp = builder.create<scf::IfOp>(
                    loc, tailFlag,
                    [&](OpBuilder &builder, Location loc) {
                      SmallVector<Value> aVecMapOperands{ivs[0], iv};
                      SmallVector<Value> bVecMapOperands{ivs[1], iv};
                      if (scalable) {
                        aVecMapOperands.push_back(step);
                        bVecMapOperands.push_back(step);
                      }
                      Value aVec = builder.create<affine::AffineVectorLoadOp>(
                          loc, vectorTy, A, AVectorMap, aVecMapOperands);
                      Value bVec = builder.create<affine::AffineVectorLoadOp>(
                          loc, vectorTy, B, BVectorMap, bVecMapOperands);
                      Value resvec =
                          builder.create<arith::MulFOp>(loc, aVec, bVec);
                      Value newAcc =
                          builder.create<arith::AddFOp>(loc, acc, resvec);
                      builder.create<scf::YieldOp>(loc, newAcc);
                    },
                    // The else branch
                    [&](OpBuilder &builder, Location loc) {
                      // Create mask according to the tail.
                      Value maskVec = builder.create<CreateMaskOp>(
                          loc, vectorMaskTy, tailLen);
                      Value ColIdxTail =
                          builder.create<arith::MulIOp>(loc, iv, step);

                      Value aVecTail = builder.create<MaskedLoadOp>(
                          loc, vectorTy, A, ValueRange{ivs[0], ColIdxTail},
                          maskVec, passthruVec);

                      Value bVecTail = builder.create<MaskedLoadOp>(
                          loc, vectorTy, B, ValueRange{ivs[1], ColIdxTail},
                          maskVec, passthruVec);

                      Value resvec = builder.create<arith::MulFOp>(
                          loc, aVecTail, bVecTail);
                      Value newAcc =
                          builder.create<arith::AddFOp>(loc, acc, resvec);
                      builder.create<scf::YieldOp>(loc, newAcc);
                    });
                builder.create<affine::AffineYieldOp>(loc, ifOp.getResult(0));
              });

          Value load = builder.create<memref::LoadOp>(
              loc, C, ValueRange{ivs[0], ivs[1]});
          Value reduction = builder.create<vector::ReductionOp>(
              loc, CombiningKind::ADD, innerLoop->getResult(0), load,
              arith::FastMathFlags::reassoc);
          builder.create<memref::StoreOp>(loc, reduction, C,
                                          ValueRange{ivs[0], ivs[1]});
        });

    rewriter.eraseOp(op);
    return success();
  }

private:
  /// Vectorization factor. This is the vector length when not scalable, and
  /// the minimum vector length when scalable.
  int64_t vf;
  /// If use scalable vector.
  bool scalable;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// MatMulVectorizationPass
//===----------------------------------------------------------------------===//

namespace {
class MatMulTransposeBVecPass
    : public PassWrapper<MatMulTransposeBVecPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulTransposeBVecPass)
  StringRef getArgument() const final {
    return "matmul-transpose-b-vectorization";
  }
  StringRef getDescription() const final {
    return "vectorize linalg MatmulTransposeBOp";
  }
  MatMulTransposeBVecPass() = default;
  MatMulTransposeBVecPass(const MatMulTransposeBVecPass &) {}
  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, VectorDialect>();
  }
  Option<int64_t> vf{*this, "vf",
                     llvm::cl::desc("Specify vectorization factor."),
                     llvm::cl::init(32)};
  Option<bool> scalable{
      *this, "scalable",
      llvm::cl::desc("Specify whether the vectorization factor is scalable."),
      llvm::cl::init(false)};
};
} // namespace

void MatMulTransposeBVecPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                       scf::SCFDialect, memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<MatMulTransposeBVecPattern>(context, vf, scalable);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerMatMulTransposeBVecPass() {
  PassRegistration<MatMulTransposeBVecPass>();
}
} // namespace buddy
} // namespace mlir
