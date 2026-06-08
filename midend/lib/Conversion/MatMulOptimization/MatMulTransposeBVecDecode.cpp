//===- MatMulTransposeBVecDecode.cpp -------------------------------------===//
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
// This file implements the Matmul_TransposeB vectorization for decode phase.
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
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
class MatMulTransposeBVecDecodePattern : public ConversionPattern {
public:
  explicit MatMulTransposeBVecDecodePattern(MLIRContext *context,
                                            int64_t vfParam,
                                            int64_t unrollParam,
                                            int64_t nTileParam)
      : ConversionPattern(linalg::MatmulTransposeBOp::getOperationName(), 1,
                          context),
        vf(vfParam), unroll(unrollParam), nTile(nTileParam) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    Value A = op->getOperand(0); // [M, K]
    Value B = op->getOperand(1); // [N, K]
    Value C = op->getOperand(2); // [M, N]

    auto aTy = dyn_cast<MemRefType>(A.getType());
    auto bTy = dyn_cast<MemRefType>(B.getType());
    auto cTy = dyn_cast<MemRefType>(C.getType());

    if (!aTy || !bTy || !cTy)
      return failure();

    if (!aTy.hasRank() || !bTy.hasRank() || !cTy.hasRank())
      return failure();

    if (aTy.getRank() != 2 || bTy.getRank() != 2 || cTy.getRank() != 2)
      return failure();

    // Currently this pass only handles f32.
    // Shape is fully general, but dtype is still f32-only.
    auto elemTy = dyn_cast<Float32Type>(aTy.getElementType());
    if (!elemTy || bTy.getElementType() != elemTy ||
        cTy.getElementType() != elemTy)
      return failure();

    // Static consistency checks when dimensions are known.
    //
    // A: [M, K]
    // B: [N, K]
    // C: [M, N]
    if (!ShapedType::isDynamic(aTy.getShape()[0]) &&
        !ShapedType::isDynamic(cTy.getShape()[0]) &&
        aTy.getShape()[0] != cTy.getShape()[0])
      return failure();

    if (!ShapedType::isDynamic(aTy.getShape()[1]) &&
        !ShapedType::isDynamic(bTy.getShape()[1]) &&
        aTy.getShape()[1] != bTy.getShape()[1])
      return failure();

    if (!ShapedType::isDynamic(bTy.getShape()[0]) &&
        !ShapedType::isDynamic(cTy.getShape()[1]) &&
        bTy.getShape()[0] != cTy.getShape()[1])
      return failure();

    if (vf <= 0 || unroll <= 0 || nTile <= 0)
      return failure();

    // Constants.
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value cVf = arith::ConstantIndexOp::create(rewriter, loc, vf);
    Value cStep = arith::ConstantIndexOp::create(rewriter, loc, vf * unroll);
    Value cNTile = arith::ConstantIndexOp::create(rewriter, loc, nTile);

    Value f0 = arith::ConstantFloatOp::create(
        rewriter, loc, rewriter.getF32Type(), APFloat(0.0f));

    auto vecTy = VectorType::get({vf}, rewriter.getF32Type());
    auto maskTy = VectorType::get({vf}, rewriter.getI1Type());
    Value vzero = vector::BroadcastOp::create(rewriter, loc, vecTy, f0);

    // Runtime dimensions.
    Value M = memref::DimOp::create(rewriter, loc, A, c0);
    Value K = memref::DimOp::create(rewriter, loc, A, c1);
    Value N = memref::DimOp::create(rewriter, loc, B, c0);

    // mainBound = floor(K / (vf * unroll)) * (vf * unroll)
    Value kDiv = arith::DivUIOp::create(rewriter, loc, K, cStep);
    Value mainBound = arith::MulIOp::create(rewriter, loc, kDiv, cStep);

    // nMainBound = floor(N / nTile) * nTile
    Value nDiv = arith::DivUIOp::create(rewriter, loc, N, cNTile);
    Value nMainBound = arith::MulIOp::create(rewriter, loc, nDiv, cNTile);

    //===------------------------------------------------------------------===//
    // Main loop:
    //
    // for m in [0, M)
    //   for nBase in [0, nMainBound), step nTile
    //
    // Each parallel task computes:
    //   C[m, nBase + 0]
    //   C[m, nBase + 1]
    //   ...
    //   C[m, nBase + nTile - 1]
    //
    // A[m, k:k+vf] is shared across nTile output columns.
    //===------------------------------------------------------------------===//

    SmallVector<Value> initTileAccs(nTile * unroll, vzero);

    scf::ParallelOp::create(
        rewriter, loc,
        /*lowerBounds=*/ValueRange{c0, c0},
        /*upperBounds=*/ValueRange{M, nMainBound},
        /*steps=*/ValueRange{c1, cNTile},
        [&](OpBuilder &pb, Location ploc, ValueRange ivs) {
          Value m = ivs[0];
          Value nBase = ivs[1];

          SmallVector<Value> colIdxs(nTile);
          colIdxs[0] = nBase;
          for (int64_t j = 1; j < nTile; ++j) {
            Value cj = arith::ConstantIndexOp::create(pb, ploc, j);
            colIdxs[j] = arith::AddIOp::create(pb, ploc, nBase, cj);
          }

          // Main K loop.
          //
          // iterArgs layout:
          //   iterArgs[j * unroll + u]
          auto mainFor = scf::ForOp::create(
              pb, ploc, c0, mainBound, cStep, ValueRange(initTileAccs),
              [&](OpBuilder &fb, Location forLoc, Value kBase,
                  ValueRange iterArgs) {
                SmallVector<Value> newAccs;
                newAccs.reserve(nTile * unroll);

                SmallVector<Value> aVecs(unroll);
                SmallVector<Value> kOffs(unroll);

                // Load A once for each unroll lane.
                // These A vectors are reused by all nTile B rows.
                for (int64_t u = 0; u < unroll; ++u) {
                  Value kOff;
                  if (u == 0) {
                    kOff = kBase;
                  } else {
                    Value cOff =
                        arith::ConstantIndexOp::create(fb, forLoc, u * vf);
                    kOff = arith::AddIOp::create(fb, forLoc, kBase, cOff);
                  }

                  kOffs[u] = kOff;

                  aVecs[u] = vector::LoadOp::create(fb, forLoc, vecTy, A,
                                                    ValueRange{m, kOff});
                }

                // For each tiled N column, load B and FMA with shared A.
                for (int64_t j = 0; j < nTile; ++j) {
                  for (int64_t u = 0; u < unroll; ++u) {
                    Value bVec = vector::LoadOp::create(
                        fb, forLoc, vecTy, B, ValueRange{colIdxs[j], kOffs[u]});

                    Value acc = vector::FMAOp::create(
                        fb, forLoc, aVecs[u], bVec, iterArgs[j * unroll + u]);

                    newAccs.push_back(acc);
                  }
                }

                scf::YieldOp::create(fb, forLoc, newAccs);
              });

          // Sum unroll accumulators for each output column.
          SmallVector<Value> sumVecs;
          sumVecs.reserve(nTile);

          for (int64_t j = 0; j < nTile; ++j) {
            Value sumVec = mainFor.getResult(j * unroll);
            for (int64_t u = 1; u < unroll; ++u) {
              sumVec = arith::AddFOp::create(pb, ploc, sumVec,
                                             mainFor.getResult(j * unroll + u));
            }
            sumVecs.push_back(sumVec);
          }

          // Tail K loop.
          //
          // Handles K % (vf * unroll) != 0.
          //
          // iterArgs layout:
          //   iterArgs[j]
          auto tailFor = scf::ForOp::create(
              pb, ploc, mainBound, K, cVf, ValueRange(sumVecs),
              [&](OpBuilder &tb, Location tailLoc, Value kBase,
                  ValueRange iterArgs) {
                Value remaining = arith::SubIOp::create(tb, tailLoc, K, kBase);

                Value gt = arith::CmpIOp::create(
                    tb, tailLoc, arith::CmpIPredicate::ugt, remaining, cVf);

                Value valid =
                    arith::SelectOp::create(tb, tailLoc, gt, cVf, remaining);

                Value mask = vector::CreateMaskOp::create(tb, tailLoc, maskTy,
                                                          ValueRange{valid});

                // A tail vector is shared across nTile output columns.
                Value aVec = vector::MaskedLoadOp::create(
                    tb, tailLoc, vecTy, A, ValueRange{m, kBase}, mask, vzero);

                SmallVector<Value> newTailAccs;
                newTailAccs.reserve(nTile);

                for (int64_t j = 0; j < nTile; ++j) {
                  Value bVec = vector::MaskedLoadOp::create(
                      tb, tailLoc, vecTy, B, ValueRange{colIdxs[j], kBase},
                      mask, vzero);

                  Value acc = vector::FMAOp::create(tb, tailLoc, aVec, bVec,
                                                    iterArgs[j]);

                  newTailAccs.push_back(acc);
                }

                scf::YieldOp::create(tb, tailLoc, newTailAccs);
              });

          // Reduce vector accumulators and store C[m, nBase + j].
          for (int64_t j = 0; j < nTile; ++j) {
            Value finalVec = tailFor.getResult(j);

            Value oldVal =
                memref::LoadOp::create(pb, ploc, C, ValueRange{m, colIdxs[j]});

            Value result = vector::ReductionOp::create(
                pb, ploc, vector::CombiningKind::ADD, finalVec, oldVal,
                arith::FastMathFlags::reassoc);

            memref::StoreOp::create(pb, ploc, result, C,
                                    ValueRange{m, colIdxs[j]});
          }
        });

    //===------------------------------------------------------------------===//
    // Tail N loop:
    //
    // for m in [0, M)
    //   for n in [nMainBound, N)
    //
    // Handles N % nTile != 0.
    //===------------------------------------------------------------------===//

    SmallVector<Value> initAccs(unroll, vzero);

    scf::ParallelOp::create(
        rewriter, loc,
        /*lowerBounds=*/ValueRange{c0, nMainBound},
        /*upperBounds=*/ValueRange{M, N},
        /*steps=*/ValueRange{c1, c1},
        [&](OpBuilder &pb, Location ploc, ValueRange ivs) {
          Value m = ivs[0];
          Value n = ivs[1];

          // Main K loop for one tail output column.
          auto mainFor = scf::ForOp::create(
              pb, ploc, c0, mainBound, cStep, ValueRange(initAccs),
              [&](OpBuilder &fb, Location forLoc, Value kBase,
                  ValueRange iterArgs) {
                SmallVector<Value> newAccs;
                newAccs.reserve(unroll);

                for (int64_t u = 0; u < unroll; ++u) {
                  Value kOff;
                  if (u == 0) {
                    kOff = kBase;
                  } else {
                    Value cOff =
                        arith::ConstantIndexOp::create(fb, forLoc, u * vf);
                    kOff = arith::AddIOp::create(fb, forLoc, kBase, cOff);
                  }

                  Value aVec = vector::LoadOp::create(fb, forLoc, vecTy, A,
                                                      ValueRange{m, kOff});

                  Value bVec = vector::LoadOp::create(fb, forLoc, vecTy, B,
                                                      ValueRange{n, kOff});

                  Value acc = vector::FMAOp::create(fb, forLoc, aVec, bVec,
                                                    iterArgs[u]);

                  newAccs.push_back(acc);
                }

                scf::YieldOp::create(fb, forLoc, newAccs);
              });

          // Sum unroll accumulators.
          Value sumVec = mainFor.getResult(0);
          for (int64_t u = 1; u < unroll; ++u) {
            sumVec =
                arith::AddFOp::create(pb, ploc, sumVec, mainFor.getResult(u));
          }

          // Tail K loop for one tail output column.
          auto tailFor = scf::ForOp::create(
              pb, ploc, mainBound, K, cVf, ValueRange{sumVec},
              [&](OpBuilder &tb, Location tailLoc, Value kBase,
                  ValueRange iterArgs) {
                Value remaining = arith::SubIOp::create(tb, tailLoc, K, kBase);

                Value gt = arith::CmpIOp::create(
                    tb, tailLoc, arith::CmpIPredicate::ugt, remaining, cVf);

                Value valid =
                    arith::SelectOp::create(tb, tailLoc, gt, cVf, remaining);

                Value mask = vector::CreateMaskOp::create(tb, tailLoc, maskTy,
                                                          ValueRange{valid});

                Value aVec = vector::MaskedLoadOp::create(
                    tb, tailLoc, vecTy, A, ValueRange{m, kBase}, mask, vzero);

                Value bVec = vector::MaskedLoadOp::create(
                    tb, tailLoc, vecTy, B, ValueRange{n, kBase}, mask, vzero);

                Value acc =
                    vector::FMAOp::create(tb, tailLoc, aVec, bVec, iterArgs[0]);

                scf::YieldOp::create(tb, tailLoc, acc);
              });

          Value finalVec = tailFor.getResult(0);

          Value oldVal = memref::LoadOp::create(pb, ploc, C, ValueRange{m, n});

          Value result = vector::ReductionOp::create(
              pb, ploc, vector::CombiningKind::ADD, finalVec, oldVal,
              arith::FastMathFlags::reassoc);

          memref::StoreOp::create(pb, ploc, result, C, ValueRange{m, n});
        });

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t vf;
  int64_t unroll;
  int64_t nTile;
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
class MatMulTransposeBVecDecodePass
    : public PassWrapper<MatMulTransposeBVecDecodePass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulTransposeBVecDecodePass)

  StringRef getArgument() const final {
    return "matmul-transpose-b-vectorization-decode";
  }

  StringRef getDescription() const final {
    return "Vectorize linalg.matmul_transpose_b for decode-oriented "
           "MxK times NxK shapes with N tiling and K-tail handling";
  }

  MatMulTransposeBVecDecodePass() = default;
  MatMulTransposeBVecDecodePass(const MatMulTransposeBVecDecodePass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, scf::SCFDialect, affine::AffineDialect,
                vector::VectorDialect, memref::MemRefDialect,
                arith::ArithDialect, func::FuncDialect>();
  }

  Option<int64_t> vf{*this, "vector-size",
                     llvm::cl::desc("Vectorization factor on K dimension."),
                     llvm::cl::init(32)};

  Option<int64_t> unroll{
      *this, "unroll",
      llvm::cl::desc("Number of independent vector accumulator chains."),
      llvm::cl::init(1)};

  Option<int64_t> nTile{
      *this, "n-tile",
      llvm::cl::desc("Number of output columns processed per parallel task. "
                     "A vector loads are shared across these columns."),
      llvm::cl::init(1)};
};
} // namespace

void MatMulTransposeBVecDecodePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);

  target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                         scf::SCFDialect, memref::MemRefDialect,
                         vector::VectorDialect, func::FuncDialect,
                         linalg::LinalgDialect>();

  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp, func::CallOp>();

  // This pass supports MxK x NxK -> MxN f32 memref shapes and rewrites
  // linalg.matmul_transpose_b into explicit vector/scf/memref operations.
  target.addIllegalOp<linalg::MatmulTransposeBOp>();

  RewritePatternSet patterns(context);
  patterns.add<MatMulTransposeBVecDecodePattern>(
      context, static_cast<int64_t>(vf), static_cast<int64_t>(unroll),
      static_cast<int64_t>(nTile));

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerMatMulTransposeBVecDecodePass() {
  PassRegistration<MatMulTransposeBVecDecodePass>();
}
} // namespace buddy
} // namespace mlir
