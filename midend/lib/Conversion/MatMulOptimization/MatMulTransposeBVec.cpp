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
// Optimizations applied:
//   - K-accumulator unrolling (--unroll=N): use N independent FMA accumulator
//     vectors per output element to hide FMA latency (default 4).
//   - N-column tiling (--n-tile=N): process N output columns per parallel task,
//     sharing A loads across columns to improve register reuse (default 1).
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
#include <optional>

#include "Utils/Utils.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
class MatMulTransposeBVecPattern : public ConversionPattern {
public:
  explicit MatMulTransposeBVecPattern(MLIRContext *context, int64_t vfParam,
                                      bool scalableParam, int64_t unrollParam,
                                      int64_t nTileParam)
      : ConversionPattern(linalg::MatmulTransposeBOp::getOperationName(), 1,
                          context) {
    vf = vfParam;
    scalable = scalableParam;
    unroll = unrollParam;
    nTile = nTileParam;
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
    ShapedType ATy = mlir::cast<mlir::ShapedType>(A.getType());
    Type eleTy = ATy.getElementType();

    // The element type for mask vector.
    IntegerType i1 = IntegerType::get(ctx, 1);

    VectorType vectorTy = mlir::VectorType::get({vf}, eleTy, {scalable});
    VectorType vectorMaskTy = VectorType::get({vf}, i1, {scalable});

    const Value c0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value c1 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    // Single-vf step used for scalable offset computation inside the loop.
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

    // Permutation map for transfer_read: (d0, d1) -> (d1)
    AffineExpr d0, d1;
    bindDims(ctx, d0, d1);
    AffineMap permMap1D = AffineMap::get(2, 0, {d1}, ctx);
    AffineMapAttr permMapAttr = AffineMapAttr::get(permMap1D);
    ArrayAttr inBoundsAttr = rewriter.getBoolArrayAttr({true});

    // ── K-loop step = vf * unroll ─────────────────────────────────────────
    Value kStep;
    if (scalable) {
      Value ufVal = rewriter.create<arith::ConstantIndexOp>(loc, unroll);
      kStep = rewriter.create<arith::MulIOp>(loc, step, ufVal);
    } else {
      kStep = rewriter.create<arith::ConstantIndexOp>(loc, vf * unroll);
    }

    // ── N-column parallel step ────────────────────────────────────────────
    Value nStep = rewriter.create<arith::ConstantIndexOp>(loc, nTile);

    // nTile * unroll initial accumulator vectors (all zero).
    llvm::SmallVector<Value> initAccs(nTile * unroll, passthruVec);

    // ── Outer parallel loop: rows of A ────────────────────────────────────
    auto outerParallelLoop = rewriter.create<scf::ParallelOp>(
        loc,
        /*lowerBounds=*/ValueRange{c0},
        /*upperBounds=*/ValueRange{aRow},
        /*steps=*/ValueRange{c1},
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          Value rowIdx = ivs[0];

          // ── Inner parallel loop: columns of B (step = nTile) ───────────
          auto innerParallelLoop = builder.create<scf::ParallelOp>(
              loc,
              /*lowerBounds=*/ValueRange{c0},
              /*upperBounds=*/ValueRange{bRow},
              /*steps=*/ValueRange{nStep},
              [&](OpBuilder &builder, Location loc, ValueRange ivs) {
                Value colBase = ivs[0];

                // Column indices: colBase, colBase+1, ..., colBase+(nTile-1)
                llvm::SmallVector<Value> colIdxs(nTile);
                colIdxs[0] = colBase;
                for (int j = 1; j < nTile; ++j) {
                  Value jConst = builder.create<arith::ConstantIndexOp>(loc, j);
                  colIdxs[j] =
                      builder.create<arith::AddIOp>(loc, colBase, jConst);
                }

                // ── K reduction loop with nTile * unroll accumulators ────
                // Opt1 (K-unroll): unroll independent accumulator chains to
                //   hide FMA latency (4–5 cycles on modern x86).
                // Opt2 (N-tile):   share A loads across nTile columns.
                auto kLoop = builder.create<scf::ForOp>(
                    loc, c0, bCol, kStep, ValueRange(initAccs),
                    [&](OpBuilder &nb, Location nl, Value iv,
                        ValueRange itrArgs) {
                      // K offsets within the unrolled body
                      llvm::SmallVector<Value> ki(unroll);
                      ki[0] = iv;
                      for (int i = 1; i < unroll; ++i) {
                        Value offset;
                        if (scalable) {
                          // offset = i * (vscale * vf)  [runtime]
                          Value iConst =
                              nb.create<arith::ConstantIndexOp>(nl, i);
                          offset = nb.create<arith::MulIOp>(nl, iConst, step);
                        } else {
                          offset =
                              nb.create<arith::ConstantIndexOp>(nl, i * vf);
                        }
                        ki[i] = nb.create<arith::AddIOp>(nl, iv, offset);
                      }

                      // Load A vectors – shared across all nTile columns
                      llvm::SmallVector<Value> aVecs(unroll);
                      for (int i = 0; i < unroll; ++i) {
                        aVecs[i] = nb.create<vector::TransferReadOp>(
                            nl, vectorTy, A, ValueRange{rowIdx, ki[i]}, c0Ele,
                            permMapAttr, inBoundsAttr);
                      }

                      // For each column: load B chunks and FMA independently
                      llvm::SmallVector<Value> newAccs(nTile * unroll);
                      for (int j = 0; j < nTile; ++j) {
                        for (int i = 0; i < unroll; ++i) {
                          auto bVec = nb.create<vector::TransferReadOp>(
                              nl, vectorTy, B, ValueRange{colIdxs[j], ki[i]},
                              c0Ele, permMapAttr, inBoundsAttr);
                          newAccs[j * unroll + i] = nb.create<vector::FMAOp>(
                              nl, aVecs[i], bVec, itrArgs[j * unroll + i]);
                        }
                      }
                      nb.create<scf::YieldOp>(nl, ValueRange(newAccs));
                    });

                // For each column: sum unroll accumulators, reduce, store
                for (int j = 0; j < nTile; ++j) {
                  // Tree-reduce the unroll accumulator vectors
                  Value sumVec = kLoop->getResult(j * unroll);
                  for (int i = 1; i < unroll; ++i) {
                    sumVec = builder.create<arith::AddFOp>(
                        loc, sumVec, kLoop->getResult(j * unroll + i));
                  }
                  Value load = builder.create<memref::LoadOp>(
                      loc, C, ValueRange{rowIdx, colIdxs[j]});
                  Value result = builder.create<vector::ReductionOp>(
                      loc, CombiningKind::ADD, sumVec, load,
                      arith::FastMathFlags::reassoc);
                  builder.create<memref::StoreOp>(
                      loc, result, C, ValueRange{rowIdx, colIdxs[j]});
                }
              });
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
  /// Number of independent K-accumulator chains per output element.
  /// Hides FMA latency by allowing out-of-order execution.
  int64_t unroll;
  /// Number of output columns processed per parallel task.
  /// Shares A loads across nTile columns, reducing A bandwidth usage.
  int64_t nTile;
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
  Option<int64_t> unroll{
      *this, "unroll",
      llvm::cl::desc("Number of independent K-accumulator chains per output "
                     "element (hides FMA latency)."),
      llvm::cl::init(1)};
  Option<int64_t> nTile{
      *this, "n-tile",
      llvm::cl::desc("Number of output columns processed per parallel task "
                     "(shares A loads across columns)."),
      llvm::cl::init(1)};
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
  patterns.add<MatMulTransposeBVecPattern>(context, vf, scalable, unroll,
                                           nTile);

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
