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
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

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

    // Support f32 and f16 (accumulating f16 in f32 for precision and RVV
    // widening FMA efficiency).
    Type elemTy = aTy.getElementType();
    bool isF16 = isa<Float16Type>(elemTy);
    if ((!isa<Float32Type>(elemTy) && !isF16) ||
        bTy.getElementType() != elemTy || cTy.getElementType() != elemTy)
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

    Type accElemTy = rewriter.getF32Type();
    Value f0 = arith::ConstantFloatOp::create(
        rewriter, loc, rewriter.getF32Type(), APFloat(0.0f));
    Value loadZero = arith::ConstantOp::create(rewriter, loc, elemTy,
                                               rewriter.getZeroAttr(elemTy));

    auto loadVecTy = VectorType::get({vf}, elemTy);
    auto accVecTy = VectorType::get({vf}, accElemTy);
    auto maskTy = VectorType::get({vf}, rewriter.getI1Type());
    Value vzero = vector::BroadcastOp::create(rewriter, loc, accVecTy, f0);
    Value vzeroLoad =
        vector::BroadcastOp::create(rewriter, loc, loadVecTy, loadZero);

    auto loadAsAcc = [&](OpBuilder &b, Location l, Value mem,
                         ValueRange idx) -> Value {
      Value v = vector::LoadOp::create(b, l, loadVecTy, mem, idx);
      return isF16 ? Value(arith::ExtFOp::create(b, l, accVecTy, v)) : v;
    };
    auto maskedLoadAsAcc = [&](OpBuilder &b, Location l, Value mem,
                               ValueRange idx, Value mask) -> Value {
      Value v = vector::MaskedLoadOp::create(b, l, loadVecTy, mem, idx, mask,
                                             vzeroLoad);
      return isF16 ? Value(arith::ExtFOp::create(b, l, accVecTy, v)) : v;
    };
    auto reduceAndStore = [&](OpBuilder &b, Location l, Value accVec, Value row,
                              Value col) {
      Value oldVal = memref::LoadOp::create(b, l, C, ValueRange{row, col});
      Value oldAcc = isF16
                         ? Value(arith::ExtFOp::create(b, l, accElemTy, oldVal))
                         : oldVal;
      Value resAcc =
          vector::ReductionOp::create(b, l, vector::CombiningKind::ADD, accVec,
                                      oldAcc, arith::FastMathFlags::reassoc);
      Value resOut =
          isF16 ? Value(arith::TruncFOp::create(b, l, elemTy, resAcc)) : resAcc;
      memref::StoreOp::create(b, l, resOut, C, ValueRange{row, col});
    };

    // Runtime dimensions.
    int64_t staticM = aTy.getShape()[0];
    int64_t staticK = aTy.getShape()[1];
    int64_t staticN = bTy.getShape()[0];
    bool isStaticM1 = !ShapedType::isDynamic(staticM) && staticM == 1;
    bool hasKTail =
        ShapedType::isDynamic(staticK) || (staticK % (vf * unroll) != 0);
    bool hasNTail = ShapedType::isDynamic(staticN) || (staticN % nTile != 0);

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

    auto buildMainBody = [&](OpBuilder &pb, Location ploc, Value m,
                             Value nBase) {
      SmallVector<Value> colIdxs(nTile);
      colIdxs[0] = nBase;
      for (int64_t j = 1; j < nTile; ++j) {
        Value cj = arith::ConstantIndexOp::create(pb, ploc, j);
        colIdxs[j] = arith::AddIOp::create(pb, ploc, nBase, cj);
      }

      auto mainFor = scf::ForOp::create(
          pb, ploc, c0, mainBound, cStep, ValueRange(initTileAccs),
          [&](OpBuilder &fb, Location forLoc, Value kBase,
              ValueRange iterArgs) {
            SmallVector<Value> newAccs;
            newAccs.reserve(nTile * unroll);

            SmallVector<Value> aVecs(unroll);
            SmallVector<Value> kOffs(unroll);

            for (int64_t u = 0; u < unroll; ++u) {
              Value kOff;
              if (u == 0) {
                kOff = kBase;
              } else {
                Value cOff = arith::ConstantIndexOp::create(fb, forLoc, u * vf);
                kOff = arith::AddIOp::create(fb, forLoc, kBase, cOff);
              }
              kOffs[u] = kOff;
              aVecs[u] = loadAsAcc(fb, forLoc, A, ValueRange{m, kOff});
            }

            for (int64_t j = 0; j < nTile; ++j) {
              for (int64_t u = 0; u < unroll; ++u) {
                Value bVec =
                    loadAsAcc(fb, forLoc, B, ValueRange{colIdxs[j], kOffs[u]});
                Value acc = vector::FMAOp::create(fb, forLoc, aVecs[u], bVec,
                                                  iterArgs[j * unroll + u]);
                newAccs.push_back(acc);
              }
            }

            scf::YieldOp::create(fb, forLoc, newAccs);
          });

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

      SmallVector<Value> finalVecs = sumVecs;
      if (hasKTail) {
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
              Value aVec =
                  maskedLoadAsAcc(tb, tailLoc, A, ValueRange{m, kBase}, mask);
              SmallVector<Value> newTailAccs;
              newTailAccs.reserve(nTile);
              for (int64_t j = 0; j < nTile; ++j) {
                Value bVec = maskedLoadAsAcc(
                    tb, tailLoc, B, ValueRange{colIdxs[j], kBase}, mask);
                Value acc =
                    vector::FMAOp::create(tb, tailLoc, aVec, bVec, iterArgs[j]);
                newTailAccs.push_back(acc);
              }
              scf::YieldOp::create(tb, tailLoc, newTailAccs);
            });
        finalVecs.assign(tailFor.getResults().begin(),
                         tailFor.getResults().end());
      }

      for (int64_t j = 0; j < nTile; ++j)
        reduceAndStore(pb, ploc, finalVecs[j], m, colIdxs[j]);
    };

    if (isStaticM1) {
      scf::ParallelOp::create(
          rewriter, loc, ValueRange{c0}, ValueRange{nMainBound},
          ValueRange{cNTile},
          [&](OpBuilder &pb, Location ploc, ValueRange ivs) {
            buildMainBody(pb, ploc, c0, ivs[0]);
          });
    } else {
      scf::ParallelOp::create(
          rewriter, loc, ValueRange{c0, c0}, ValueRange{M, nMainBound},
          ValueRange{c1, cNTile},
          [&](OpBuilder &pb, Location ploc, ValueRange ivs) {
            buildMainBody(pb, ploc, ivs[0], ivs[1]);
          });
    }

    //===------------------------------------------------------------------===//
    // Tail N loop: handles N % nTile != 0.
    //===------------------------------------------------------------------===//
    if (hasNTail) {
      SmallVector<Value> initAccs(unroll, vzero);
      auto buildTailNBody = [&](OpBuilder &pb, Location ploc, Value m,
                                Value n) {
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
                Value aVec = loadAsAcc(fb, forLoc, A, ValueRange{m, kOff});
                Value bVec = loadAsAcc(fb, forLoc, B, ValueRange{n, kOff});
                Value acc =
                    vector::FMAOp::create(fb, forLoc, aVec, bVec, iterArgs[u]);
                newAccs.push_back(acc);
              }
              scf::YieldOp::create(fb, forLoc, newAccs);
            });

        Value sumVec = mainFor.getResult(0);
        for (int64_t u = 1; u < unroll; ++u) {
          sumVec =
              arith::AddFOp::create(pb, ploc, sumVec, mainFor.getResult(u));
        }

        Value finalVec = sumVec;
        if (hasKTail) {
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
                Value aVec =
                    maskedLoadAsAcc(tb, tailLoc, A, ValueRange{m, kBase}, mask);
                Value bVec =
                    maskedLoadAsAcc(tb, tailLoc, B, ValueRange{n, kBase}, mask);
                Value acc =
                    vector::FMAOp::create(tb, tailLoc, aVec, bVec, iterArgs[0]);
                scf::YieldOp::create(tb, tailLoc, acc);
              });
          finalVec = tailFor.getResult(0);
        }

        reduceAndStore(pb, ploc, finalVec, m, n);
      };

      if (isStaticM1) {
        scf::ParallelOp::create(
            rewriter, loc, ValueRange{nMainBound}, ValueRange{N},
            ValueRange{c1}, [&](OpBuilder &pb, Location ploc, ValueRange ivs) {
              buildTailNBody(pb, ploc, c0, ivs[0]);
            });
      } else {
        scf::ParallelOp::create(
            rewriter, loc, ValueRange{c0, nMainBound}, ValueRange{M, N},
            ValueRange{c1, c1},
            [&](OpBuilder &pb, Location ploc, ValueRange ivs) {
              buildTailNBody(pb, ploc, ivs[0], ivs[1]);
            });
      }
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t vf;
  int64_t unroll;
  int64_t nTile;
};

static bool isUnitTrailingStride3D(MemRefType ty) {
  if (ty.getLayout().isIdentity())
    return true;
  SmallVector<int64_t> strides;
  int64_t offset;
  return succeeded(ty.getStridesAndOffset(strides, offset)) &&
         strides.size() == 3 && strides[2] == 1;
}

static bool isBatchMatmulTransposeBMaps(linalg::BatchMatmulOp op) {
  auto maps = op.getIndexingMapsArray();
  if (maps.size() != 3)
    return false;
  auto ctx = op.getContext();
  auto b = getAffineDimExpr(0, ctx);
  auto m = getAffineDimExpr(1, ctx);
  auto n = getAffineDimExpr(2, ctx);
  auto k = getAffineDimExpr(3, ctx);
  return maps == SmallVector<AffineMap>{AffineMap::get(4, 0, {b, m, k}, ctx),
                                        AffineMap::get(4, 0, {b, n, k}, ctx),
                                        AffineMap::get(4, 0, {b, m, n}, ctx)};
}

static bool isBatchMatmulStandardMaps(linalg::BatchMatmulOp op) {
  auto maps = op.getIndexingMapsArray();
  if (maps.size() != 3)
    return false;
  auto ctx = op.getContext();
  auto b = getAffineDimExpr(0, ctx);
  auto m = getAffineDimExpr(1, ctx);
  auto n = getAffineDimExpr(2, ctx);
  auto k = getAffineDimExpr(3, ctx);
  return maps == SmallVector<AffineMap>{AffineMap::get(4, 0, {b, m, k}, ctx),
                                        AffineMap::get(4, 0, {b, k, n}, ctx),
                                        AffineMap::get(4, 0, {b, m, n}, ctx)};
}

static bool isSupportedDecodeBatchMatmul(linalg::BatchMatmulOp op) {
  if (!op.hasPureBufferSemantics())
    return false;
  auto aTy = dyn_cast<MemRefType>(op.getInputs()[0].getType());
  auto bTy = dyn_cast<MemRefType>(op.getInputs()[1].getType());
  auto cTy = dyn_cast<MemRefType>(op.getOutputs()[0].getType());
  if (!aTy || !bTy || !cTy || !aTy.hasStaticShape() || !bTy.hasStaticShape() ||
      !cTy.hasStaticShape())
    return false;
  if (aTy.getRank() != 3 || bTy.getRank() != 3 || cTy.getRank() != 3)
    return false;
  if (!aTy.getElementType().isF16() || !bTy.getElementType().isF16() ||
      !cTy.getElementType().isF16())
    return false;
  if (aTy.getDimSize(1) != 1 || cTy.getDimSize(1) != 1)
    return false;
  if (isBatchMatmulTransposeBMaps(op)) {
    int64_t K = aTy.getDimSize(2);
    return K >= 32 && K % 32 == 0 && isUnitTrailingStride3D(aTy) &&
           isUnitTrailingStride3D(bTy);
  }
  if (isBatchMatmulStandardMaps(op)) {
    int64_t N = bTy.getDimSize(2);
    return N >= 32 && N % 32 == 0 && isUnitTrailingStride3D(bTy) &&
           isUnitTrailingStride3D(cTy);
  }
  return false;
}

// Vectorize M=1 decode linalg.batch_matmul (both Q@K^T transpose-B and P@V
// standard) in FP32 accumulation, parallelizing across the batch (head) dim.
class BatchMatmulDecodeVecPattern : public ConversionPattern {
public:
  explicit BatchMatmulDecodeVecPattern(MLIRContext *context)
      : ConversionPattern(linalg::BatchMatmulOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *rawOp, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto op = dyn_cast<linalg::BatchMatmulOp>(rawOp);
    if (!op || !isSupportedDecodeBatchMatmul(op))
      return failure();
    Location loc = op.getLoc();
    Value A = op.getInputs()[0];
    Value B = op.getInputs()[1];
    Value C = op.getOutputs()[0];
    auto aTy = cast<MemRefType>(A.getType());
    auto bTy = cast<MemRefType>(B.getType());
    int64_t Batch = aTy.getDimSize(0);
    Type f16Ty = rewriter.getF16Type();
    Type f32Ty = rewriter.getF32Type();
    constexpr int64_t VF = 32;
    auto loadVecTy = VectorType::get({VF}, f16Ty);
    auto accVecTy = VectorType::get({VF}, f32Ty);

    auto idx = [&](int64_t v) -> Value {
      return arith::ConstantIndexOp::create(rewriter, loc, v);
    };
    Value c0 = idx(0), c1 = idx(1), cBatch = idx(Batch);
    Value f0 = arith::ConstantFloatOp::create(
        rewriter, loc, rewriter.getF32Type(), APFloat(0.0f));
    Value vZero = vector::BroadcastOp::create(rewriter, loc, accVecTy, f0);

    if (isBatchMatmulTransposeBMaps(op)) {
      int64_t K = aTy.getDimSize(2);
      int64_t N = bTy.getDimSize(1);
      int64_t kChunks = K / VF;
      int64_t nTileLocal = (N >= 4) ? 4 : 1;
      int64_t nMain = (N / nTileLocal) * nTileLocal;

      auto emitHead = [&](OpBuilder &pb, Location pl, Value bIdx) {
        // Preload A[b, 0, :] once per head when K is small (e.g. K=64 -> 2
        // vecs).
        SmallVector<Value> aVecs;
        if (kChunks <= 4) {
          aVecs.reserve(kChunks);
          for (int64_t kc = 0; kc < kChunks; ++kc) {
            Value raw = vector::LoadOp::create(
                pb, pl, loadVecTy, A,
                ValueRange{bIdx, c0,
                           arith::ConstantIndexOp::create(pb, pl, kc * VF)});
            aVecs.push_back(arith::ExtFOp::create(pb, pl, accVecTy, raw));
          }
        }

        auto computeCols = [&](OpBuilder &nb, Location nl, Value nBase,
                               int64_t count) {
          SmallVector<Value> nIdxs(count);
          nIdxs[0] = nBase;
          for (int64_t j = 1; j < count; ++j)
            nIdxs[j] = arith::AddIOp::create(
                nb, nl, nBase, arith::ConstantIndexOp::create(nb, nl, j));

          SmallVector<Value> accs(count, vZero);
          if (!aVecs.empty()) {
            for (int64_t kc = 0; kc < kChunks; ++kc) {
              Value kOff = arith::ConstantIndexOp::create(nb, nl, kc * VF);
              for (int64_t j = 0; j < count; ++j) {
                Value bRaw = vector::LoadOp::create(
                    nb, nl, loadVecTy, B, ValueRange{bIdx, nIdxs[j], kOff});
                Value bV = arith::ExtFOp::create(nb, nl, accVecTy, bRaw);
                accs[j] = vector::FMAOp::create(nb, nl, aVecs[kc], bV, accs[j]);
              }
            }
          } else {
            auto kFor = scf::ForOp::create(
                nb, nl, c0, idx(K), idx(VF), ValueRange(accs),
                [&](OpBuilder &fb, Location fl, Value kOff, ValueRange it) {
                  Value aRaw = vector::LoadOp::create(
                      fb, fl, loadVecTy, A, ValueRange{bIdx, c0, kOff});
                  Value aV = arith::ExtFOp::create(fb, fl, accVecTy, aRaw);
                  SmallVector<Value> nextAccs;
                  nextAccs.reserve(count);
                  for (int64_t j = 0; j < count; ++j) {
                    Value bRaw = vector::LoadOp::create(
                        fb, fl, loadVecTy, B, ValueRange{bIdx, nIdxs[j], kOff});
                    Value bV = arith::ExtFOp::create(fb, fl, accVecTy, bRaw);
                    nextAccs.push_back(
                        vector::FMAOp::create(fb, fl, aV, bV, it[j]));
                  }
                  scf::YieldOp::create(fb, fl, nextAccs);
                });
            accs.assign(kFor.getResults().begin(), kFor.getResults().end());
          }
          for (int64_t j = 0; j < count; ++j) {
            Value cur = memref::LoadOp::create(nb, nl, C,
                                               ValueRange{bIdx, c0, nIdxs[j]});
            Value curF32 = arith::ExtFOp::create(nb, nl, f32Ty, cur);
            Value sum = vector::ReductionOp::create(
                nb, nl, vector::CombiningKind::ADD, accs[j], curF32,
                arith::FastMathFlags::reassoc);
            Value out = arith::TruncFOp::create(nb, nl, f16Ty, sum);
            memref::StoreOp::create(nb, nl, out, C,
                                    ValueRange{bIdx, c0, nIdxs[j]});
          }
        };

        if (nMain > 0) {
          scf::ForOp::create(
              pb, pl, c0, arith::ConstantIndexOp::create(pb, pl, nMain),
              arith::ConstantIndexOp::create(pb, pl, nTileLocal), ValueRange{},
              [&](OpBuilder &nb, Location nl, Value nBase, ValueRange) {
                computeCols(nb, nl, nBase, nTileLocal);
                scf::YieldOp::create(nb, nl);
              });
        }
        if (nMain < N) {
          scf::ForOp::create(
              pb, pl, arith::ConstantIndexOp::create(pb, pl, nMain),
              arith::ConstantIndexOp::create(pb, pl, N), c1, ValueRange{},
              [&](OpBuilder &nb, Location nl, Value nIdx, ValueRange) {
                computeCols(nb, nl, nIdx, 1);
                scf::YieldOp::create(nb, nl);
              });
        }
      };

      if (Batch > 1) {
        scf::ParallelOp::create(
            rewriter, loc, ValueRange{c0}, ValueRange{cBatch}, ValueRange{c1},
            [&](OpBuilder &pb, Location pl, ValueRange ivs) {
              emitHead(pb, pl, ivs[0]);
            });
      } else {
        emitHead(rewriter, loc, c0);
      }
      rewriter.eraseOp(op);
      return success();
    }

    // Standard M=1 batch_matmul: C[b, 0, 0:N] += sum_k A[b, 0, k] * B[b, k,
    // 0:N]
    int64_t K = aTy.getDimSize(2);
    int64_t N = bTy.getDimSize(2);
    int64_t nChunks = N / VF;

    auto emitStandardHead = [&](OpBuilder &pb, Location pl, Value bIdx) {
      if (nChunks <= 4) {
        SmallVector<Value> nOffs, initAccs;
        nOffs.reserve(nChunks);
        initAccs.reserve(nChunks);
        for (int64_t nc = 0; nc < nChunks; ++nc) {
          Value nOff = arith::ConstantIndexOp::create(pb, pl, nc * VF);
          nOffs.push_back(nOff);
          Value cRaw = vector::LoadOp::create(pb, pl, loadVecTy, C,
                                              ValueRange{bIdx, c0, nOff});
          initAccs.push_back(arith::ExtFOp::create(pb, pl, accVecTy, cRaw));
        }
        auto kFor = scf::ForOp::create(
            pb, pl, c0, arith::ConstantIndexOp::create(pb, pl, K), c1,
            ValueRange(initAccs),
            [&](OpBuilder &fb, Location fl, Value kIdx, ValueRange it) {
              Value aRaw =
                  memref::LoadOp::create(fb, fl, A, ValueRange{bIdx, c0, kIdx});
              Value aS = arith::ExtFOp::create(fb, fl, f32Ty, aRaw);
              Value aV = vector::BroadcastOp::create(fb, fl, accVecTy, aS);
              SmallVector<Value> nextAccs;
              nextAccs.reserve(nChunks);
              for (int64_t nc = 0; nc < nChunks; ++nc) {
                Value bRaw = vector::LoadOp::create(
                    fb, fl, loadVecTy, B, ValueRange{bIdx, kIdx, nOffs[nc]});
                Value bV = arith::ExtFOp::create(fb, fl, accVecTy, bRaw);
                nextAccs.push_back(
                    vector::FMAOp::create(fb, fl, aV, bV, it[nc]));
              }
              scf::YieldOp::create(fb, fl, nextAccs);
            });
        for (int64_t nc = 0; nc < nChunks; ++nc) {
          Value out =
              arith::TruncFOp::create(pb, pl, loadVecTy, kFor.getResult(nc));
          vector::StoreOp::create(pb, pl, out, C,
                                  ValueRange{bIdx, c0, nOffs[nc]});
        }
      } else {
        scf::ForOp::create(
            pb, pl, c0, arith::ConstantIndexOp::create(pb, pl, N),
            arith::ConstantIndexOp::create(pb, pl, VF), ValueRange{},
            [&](OpBuilder &nb, Location nl, Value nOff, ValueRange) {
              Value cRaw = vector::LoadOp::create(nb, nl, loadVecTy, C,
                                                  ValueRange{bIdx, c0, nOff});
              Value initAcc = arith::ExtFOp::create(nb, nl, accVecTy, cRaw);
              auto kFor = scf::ForOp::create(
                  nb, nl, c0, arith::ConstantIndexOp::create(nb, nl, K), c1,
                  ValueRange{initAcc},
                  [&](OpBuilder &fb, Location fl, Value kIdx, ValueRange it) {
                    Value aRaw = memref::LoadOp::create(
                        fb, fl, A, ValueRange{bIdx, c0, kIdx});
                    Value aS = arith::ExtFOp::create(fb, fl, f32Ty, aRaw);
                    Value aV =
                        vector::BroadcastOp::create(fb, fl, accVecTy, aS);
                    Value bRaw = vector::LoadOp::create(
                        fb, fl, loadVecTy, B, ValueRange{bIdx, kIdx, nOff});
                    Value bV = arith::ExtFOp::create(fb, fl, accVecTy, bRaw);
                    Value acc = vector::FMAOp::create(fb, fl, aV, bV, it[0]);
                    scf::YieldOp::create(fb, fl, acc);
                  });
              Value out =
                  arith::TruncFOp::create(nb, nl, loadVecTy, kFor.getResult(0));
              vector::StoreOp::create(nb, nl, out, C,
                                      ValueRange{bIdx, c0, nOff});
              scf::YieldOp::create(nb, nl);
            });
      }
    };

    if (Batch > 1) {
      scf::ParallelOp::create(rewriter, loc, ValueRange{c0}, ValueRange{cBatch},
                              ValueRange{c1},
                              [&](OpBuilder &pb, Location pl, ValueRange ivs) {
                                emitStandardHead(pb, pl, ivs[0]);
                              });
    } else {
      emitStandardHead(rewriter, loc, c0);
    }
    rewriter.eraseOp(op);
    return success();
  }
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

  // This pass supports MxK x NxK -> MxN f32/f16 memref shapes and rewrites
  // linalg.matmul_transpose_b into explicit vector/scf/memref operations.
  target.addDynamicallyLegalOp<linalg::MatmulTransposeBOp>(
      [](linalg::MatmulTransposeBOp op) {
        if (!op.hasPureBufferSemantics())
          return true;
        auto aTy = dyn_cast<MemRefType>(op.getInputs()[0].getType());
        auto bTy = dyn_cast<MemRefType>(op.getInputs()[1].getType());
        auto cTy = dyn_cast<MemRefType>(op.getOutputs()[0].getType());
        if (!aTy || !bTy || !cTy || aTy.getRank() != 2 || bTy.getRank() != 2 ||
            cTy.getRank() != 2)
          return true;
        Type elemTy = aTy.getElementType();
        if ((!isa<Float32Type>(elemTy) && !isa<Float16Type>(elemTy)) ||
            bTy.getElementType() != elemTy || cTy.getElementType() != elemTy)
          return true;
        return false;
      });

  target.addDynamicallyLegalOp<linalg::BatchMatmulOp>(
      [](linalg::BatchMatmulOp op) {
        return !isSupportedDecodeBatchMatmul(op);
      });

  RewritePatternSet patterns(context);
  patterns.add<MatMulTransposeBVecDecodePattern>(
      context, static_cast<int64_t>(vf), static_cast<int64_t>(unroll),
      static_cast<int64_t>(nTile));
  patterns.add<BatchMatmulDecodeVecPattern>(context);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace {
struct FuseMatMulTransposeBPattern : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (!matmulOp.hasPureTensorSemantics())
      return failure();
    auto maps = matmulOp.getIndexingMapsArray();
    auto ctx = rewriter.getContext();
    auto m = rewriter.getAffineDimExpr(0);
    auto n = rewriter.getAffineDimExpr(1);
    auto k = rewriter.getAffineDimExpr(2);
    if (maps != SmallVector<AffineMap>{AffineMap::get(3, 0, {m, k}, ctx),
                                       AffineMap::get(3, 0, {k, n}, ctx),
                                       AffineMap::get(3, 0, {m, n}, ctx)})
      return failure();

    Value A = matmulOp.getInputs()[0];
    Value B = matmulOp.getInputs()[1];
    Value C = matmulOp.getOutputs()[0];
    auto transposeOp = B.getDefiningOp<linalg::TransposeOp>();
    if (!transposeOp || !B.hasOneUse())
      return failure();
    ArrayRef<int64_t> perm = transposeOp.getPermutation();
    if (perm.size() != 2 || perm[0] != 1 || perm[1] != 0)
      return failure();

    auto newOp = linalg::MatmulTransposeBOp::create(
        rewriter, matmulOp.getLoc(), matmulOp.getResultTypes(),
        ValueRange{A, transposeOp.getInput()}, ValueRange{C});
    rewriter.replaceOp(matmulOp, newOp.getResults());
    rewriter.eraseOp(transposeOp);
    return success();
  }
};

class FuseMatMulTransposeBPass
    : public PassWrapper<FuseMatMulTransposeBPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseMatMulTransposeBPass)
  StringRef getArgument() const final { return "fuse-matmul-transpose-b"; }
  StringRef getDescription() const final {
    return "Fuse single-use 2D linalg.transpose on RHS into "
           "linalg.matmul_transpose_b before bufferization";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(func.getContext());
    patterns.add<FuseMatMulTransposeBPattern>(func.getContext());
    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

namespace mlir {
namespace buddy {
void registerMatMulTransposeBVecDecodePass() {
  PassRegistration<MatMulTransposeBVecDecodePass>();
}
void registerFuseMatMulTransposeBPass() {
  PassRegistration<FuseMatMulTransposeBPass>();
}
} // namespace buddy
} // namespace mlir
