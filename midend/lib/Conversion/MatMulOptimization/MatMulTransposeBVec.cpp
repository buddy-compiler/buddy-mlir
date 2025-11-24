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
#include <mlir/Dialect/SCF/IR/SCF.h>
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
    ShapedType ATy = cast<ShapedType>(A.getType());
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

    // Create permutation map for transfer_read: (d0, d1) -> (d1)
    AffineExpr d0, d1;
    bindDims(ctx, d0, d1);
    AffineMap permMap1D = AffineMap::get(2, 0, {d1}, ctx);

    // Create outer parallel loop for row dimension using scf.parallel
    auto outerParallelLoop = rewriter.create<scf::ParallelOp>(
        loc,
        /*lowerBounds=*/ValueRange{c0},
        /*upperBounds=*/ValueRange{aRow},
        /*steps=*/ValueRange{c1},
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          Value rowIdx = ivs[0];

          // Create inner parallel loop for column dimension
          auto innerParallelLoop = builder.create<scf::ParallelOp>(
              loc,
              /*lowerBounds=*/ValueRange{c0},
              /*upperBounds=*/ValueRange{bRow},
              /*steps=*/ValueRange{c1},
              [&](OpBuilder &builder, Location loc, ValueRange ivs) {
                Value colIdx = ivs[0];

                // Create inner vectorization loop with iter_args for
                // accumulation
                Value stepValueForScf =
                    scalable ? step
                             : builder.create<arith::ConstantIndexOp>(loc, vf);

                auto innerLoop = builder.create<scf::ForOp>(
                    loc, c0, bCol, stepValueForScf, ValueRange{passthruVec},
                    [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
                        ValueRange itrArgs) {
                      Value acc = itrArgs[0];
                      auto aVec = nestedBuilder.create<vector::TransferReadOp>(
                          nestedLoc, vectorTy, A, ValueRange{rowIdx, iv},
                          permMap1D);
                      auto bVec = nestedBuilder.create<vector::TransferReadOp>(
                          nestedLoc, vectorTy, B, ValueRange{colIdx, iv},
                          permMap1D);
                      Value newAcc = nestedBuilder.create<vector::FMAOp>(
                          nestedLoc, aVec, bVec, acc);
                      nestedBuilder.create<scf::YieldOp>(nestedLoc, newAcc);
                    });
                Value load = builder.create<memref::LoadOp>(
                    loc, C, ValueRange{rowIdx, colIdx});
                // Reduction directly uses load as accumulator, no need to add
                // again
                Value result = builder.create<vector::ReductionOp>(
                    loc, CombiningKind::ADD, innerLoop->getResult(0), load,
                    arith::FastMathFlags::reassoc);
                builder.create<memref::StoreOp>(loc, result, C,
                                                ValueRange{rowIdx, colIdx});
              });
        });

    // TODO: Add tile processing for the inner loop.

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
