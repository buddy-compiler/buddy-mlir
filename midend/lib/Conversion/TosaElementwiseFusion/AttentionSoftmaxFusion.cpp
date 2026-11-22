//===- AttentionSoftmaxFusion.cpp - Attention softmax and GELU -------===//
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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cmath>
#include <limits>

using namespace mlir;

namespace {

static bool isZeroFloatSplat(Value v) {
  auto cst = v.getDefiningOp<arith::ConstantOp>();
  if (!cst)
    return false;
  auto dense = dyn_cast<DenseElementsAttr>(cst.getValue());
  if (!dense || !dense.isSplat())
    return false;
  if (!isa<FloatType>(dense.getElementType()))
    return false;
  return dense.getSplatValue<APFloat>().isZero();
}

static bool isUnitTrailingStride(MemRefType ty) {
  if (ty.getLayout().isIdentity())
    return true;
  SmallVector<int64_t> strides;
  int64_t offset;
  return succeeded(ty.getStridesAndOffset(strides, offset)) &&
         !strides.empty() && strides.back() == 1;
}

static bool isPureParallelGeneric(linalg::GenericOp op) {
  return llvm::all_of(op.getIteratorTypesArray(), linalg::isParallelIterator);
}

static Operation *getSingleBodyOp(linalg::GenericOp op) {
  Block *body = op.getBody();
  if (!body || body->getOperations().size() != 2)
    return nullptr;
  auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return nullptr;
  Operation &front = body->front();
  if (front.getNumResults() != 1 || yield.getOperand(0) != front.getResult(0))
    return nullptr;
  return &front;
}

// Match an f16 clamp generic: min(x, +65504) then max(..., -65504).
static bool isF16FiniteClamp(linalg::GenericOp op, Value &src) {
  if (!op || !op.hasPureTensorSemantics() || op.getNumDpsInputs() != 1 ||
      op.getNumDpsInits() != 1 || !isPureParallelGeneric(op))
    return false;
  auto maps = op.getIndexingMapsArray();
  if (maps.size() != 2 || !maps[0].isIdentity() || !maps[1].isIdentity())
    return false;
  Block *body = op.getBody();
  if (!body || body->getOperations().size() != 3)
    return false;
  auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return false;
  auto it = body->begin();
  auto minOp = dyn_cast<arith::MinimumFOp>(&*it++);
  auto maxOp = dyn_cast<arith::MaximumFOp>(&*it++);
  if (!minOp || !maxOp || yield.getOperand(0) != maxOp.getResult())
    return false;
  if (minOp.getLhs() != body->getArgument(0) ||
      maxOp.getLhs() != minOp.getResult())
    return false;
  auto hiCst = minOp.getRhs().getDefiningOp<arith::ConstantOp>();
  auto loCst = maxOp.getRhs().getDefiningOp<arith::ConstantOp>();
  if (!hiCst || !loCst)
    return false;
  auto hiAttr = dyn_cast<FloatAttr>(hiCst.getValue());
  auto loAttr = dyn_cast<FloatAttr>(loCst.getValue());
  if (!hiAttr || !loAttr || !hiAttr.getType().isF16() ||
      !loAttr.getType().isF16())
    return false;
  if (hiAttr.getValueAsDouble() != 65504.0 ||
      loAttr.getValueAsDouble() != -65504.0)
    return false;
  src = op.getDpsInputs()[0];
  return true;
}

// Fold elementwise addf with a broadcast zero constant tensor: add(x, 0) -> x.
struct FoldAddZeroTensorPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics() || op.getNumDpsInputs() != 2 ||
        op.getNumDpsInits() != 1 || op.getNumResults() != 1 ||
        !isPureParallelGeneric(op))
      return failure();
    auto addOp = dyn_cast_or_null<arith::AddFOp>(getSingleBodyOp(op));
    if (!addOp)
      return failure();
    Block *body = op.getBody();
    Value arg0 = body->getArgument(0);
    Value arg1 = body->getArgument(1);
    if (!((addOp.getLhs() == arg0 && addOp.getRhs() == arg1) ||
          (addOp.getLhs() == arg1 && addOp.getRhs() == arg0)))
      return failure();
    auto maps = op.getIndexingMapsArray();
    if (maps.size() != 3 || !maps[2].isIdentity())
      return failure();

    Value in0 = op.getDpsInputs()[0];
    Value in1 = op.getDpsInputs()[1];
    Type outTy = op.getResult(0).getType();
    if (isZeroFloatSplat(in1) && in0.getType() == outTy &&
        maps[0].isIdentity()) {
      rewriter.replaceOp(op, in0);
      return success();
    }
    if (isZeroFloatSplat(in0) && in1.getType() == outTy &&
        maps[1].isIdentity()) {
      rewriter.replaceOp(op, in1);
      return success();
    }
    return failure();
  }
};

// Match PyTorch/TOSA decomposed softmax:
//   max  = reduce_max(x, dim=last)
//   sub1 = x - expand(max)
//   exp1 = math.exp(sub1)
//   sum  = reduce_add(exp1, dim=last)
//   logS = math.log(expand(sum))
//   lse  = expand(max) + logS
//   sub2 = x - lse
//   out  = math.exp(sub2)
// and rewrite `out` directly into `linalg.softmax dimension(last) ins(x)`.
struct FoldLogSoftmaxExpPattern : public OpRewritePattern<math::ExpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::ExpOp exp2Op,
                                PatternRewriter &rewriter) const override {
    auto resTy = dyn_cast<RankedTensorType>(exp2Op.getType());
    if (!resTy || !resTy.hasStaticShape() || resTy.getRank() < 2)
      return failure();
    Type elemTy = resTy.getElementType();
    if (!elemTy.isF16() && !elemTy.isF32())
      return failure();
    int64_t rank = resTy.getRank();
    int64_t lastDim = rank - 1;

    auto matchBroadcastLastDimSub = [&](linalg::GenericOp subOp, Value &lhs,
                                        Value &rhs) -> bool {
      if (!subOp || !subOp.hasPureTensorSemantics() ||
          subOp.getNumDpsInputs() != 2 || subOp.getNumDpsInits() != 1 ||
          !isPureParallelGeneric(subOp))
        return false;
      auto sub = dyn_cast_or_null<arith::SubFOp>(getSingleBodyOp(subOp));
      if (!sub || sub.getLhs() != subOp.getBody()->getArgument(0) ||
          sub.getRhs() != subOp.getBody()->getArgument(1))
        return false;
      auto maps = subOp.getIndexingMapsArray();
      if (maps.size() != 3 || !maps[0].isIdentity() || !maps[2].isIdentity())
        return false;
      // RHS must broadcast along the last dimension: (d0, ..., d_{r-2}, 0).
      auto rhsTy =
          dyn_cast<RankedTensorType>(subOp.getDpsInputs()[1].getType());
      if (!rhsTy || rhsTy.getRank() != rank || rhsTy.getDimSize(lastDim) != 1)
        return false;
      AffineMap rhsMap = maps[1];
      if (rhsMap.getNumResults() != rank)
        return false;
      for (int64_t d = 0; d < lastDim; ++d) {
        if (rhsMap.getResult(d) != rewriter.getAffineDimExpr(d))
          return false;
      }
      auto zeroExpr = dyn_cast<AffineConstantExpr>(rhsMap.getResult(lastDim));
      if (!zeroExpr || zeroExpr.getValue() != 0)
        return false;
      lhs = subOp.getDpsInputs()[0];
      rhs = subOp.getDpsInputs()[1];
      return lhs.getType() == resTy;
    };

    // 1. exp2(sub2(x, lse))
    auto sub2Op = exp2Op.getOperand().getDefiningOp<linalg::GenericOp>();
    Value x, lse;
    if (!matchBroadcastLastDimSub(sub2Op, x, lse))
      return failure();

    // 2. lse = add(maxExp, logSum) (either operand order)
    auto lseOp = lse.getDefiningOp<linalg::GenericOp>();
    if (!lseOp || !lseOp.hasPureTensorSemantics() ||
        lseOp.getNumDpsInputs() != 2 || !isPureParallelGeneric(lseOp))
      return failure();
    auto lseAdd = dyn_cast_or_null<arith::AddFOp>(getSingleBodyOp(lseOp));
    if (!lseAdd)
      return failure();
    Value lseIn0 = lseOp.getDpsInputs()[0];
    Value lseIn1 = lseOp.getDpsInputs()[1];

    auto matchLogSum = [&](Value cand, Value &sumExpOut) -> bool {
      auto logOp = cand.getDefiningOp<linalg::GenericOp>();
      if (!logOp || !logOp.hasPureTensorSemantics() ||
          logOp.getNumDpsInputs() != 1 || !isPureParallelGeneric(logOp))
        return false;
      auto log = dyn_cast_or_null<math::LogOp>(getSingleBodyOp(logOp));
      if (!log || log.getOperand() != logOp.getBody()->getArgument(0))
        return false;
      sumExpOut = logOp.getDpsInputs()[0];
      return true;
    };

    Value maxExp, sumExp;
    if (matchLogSum(lseIn1, sumExp)) {
      maxExp = lseIn0;
    } else if (matchLogSum(lseIn0, sumExp)) {
      maxExp = lseIn1;
    } else {
      return failure();
    }

    // 3. sumExp = expand_shape(reduce_add(exp1))
    auto sumExpandOp = sumExp.getDefiningOp<tensor::ExpandShapeOp>();
    if (!sumExpandOp)
      return failure();
    auto sumRedOp = sumExpandOp.getSrc().getDefiningOp<linalg::ReduceOp>();
    if (!sumRedOp || sumRedOp.getNumDpsInputs() != 1 ||
        sumRedOp.getDimensions() != ArrayRef<int64_t>{lastDim})
      return failure();
    Block *sumBody = sumRedOp.getBody();
    if (!sumBody || sumBody->getOperations().size() != 2 ||
        !isa<arith::AddFOp>(sumBody->front()))
      return failure();

    // 4. exp1 = math.exp(sub1(x, maxExp))
    auto exp1Op = sumRedOp.getDpsInputs()[0].getDefiningOp<math::ExpOp>();
    if (!exp1Op)
      return failure();
    auto sub1Op = exp1Op.getOperand().getDefiningOp<linalg::GenericOp>();
    Value x1, maxExp1;
    if (!matchBroadcastLastDimSub(sub1Op, x1, maxExp1) || x1 != x ||
        maxExp1 != maxExp)
      return failure();

    // 5. maxExp = expand_shape(reduce_max(x))
    auto maxExpandOp = maxExp.getDefiningOp<tensor::ExpandShapeOp>();
    if (!maxExpandOp)
      return failure();
    auto maxRedOp = maxExpandOp.getSrc().getDefiningOp<linalg::ReduceOp>();
    if (!maxRedOp || maxRedOp.getNumDpsInputs() != 1 ||
        maxRedOp.getDpsInputs()[0] != x ||
        maxRedOp.getDimensions() != ArrayRef<int64_t>{lastDim})
      return failure();
    Block *maxBody = maxRedOp.getBody();
    if (!maxBody || maxBody->getOperations().size() != 2 ||
        !isa<arith::MaximumFOp>(maxBody->front()))
      return failure();

    // Optional: look through f16 [-65504, 65504] clamp feeding x when all uses
    // of x stay inside this softmax chain (our f16 softmax clamps in f32).
    Value softmaxInput = x;
    if (elemTy.isF16()) {
      if (auto clampOp = x.getDefiningOp<linalg::GenericOp>()) {
        Value preClamp;
        if (isF16FiniteClamp(clampOp, preClamp) &&
            preClamp.getType() == resTy) {
          bool onlySoftmaxUses = llvm::all_of(x.getUsers(), [&](Operation *u) {
            return u == maxRedOp || u == sub1Op || u == sub2Op;
          });
          if (onlySoftmaxUses)
            softmaxInput = preClamp;
        }
      }
    }

    Location loc = exp2Op.getLoc();
    Value emptyOut =
        tensor::EmptyOp::create(rewriter, loc, resTy.getShape(), elemTy);
    auto softmax =
        linalg::SoftmaxOp::create(rewriter, loc, TypeRange{resTy}, softmaxInput,
                                  emptyOut, static_cast<uint64_t>(lastDim));
    rewriter.replaceOp(exp2Op, softmax.getResults());
    return success();
  }
};

static bool isSplatConst(Value v, double expected) {
  auto matchScalar = [&](Value x) -> bool {
    auto cst = x.getDefiningOp<arith::ConstantOp>();
    if (!cst)
      return false;
    if (auto fa = dyn_cast<FloatAttr>(cst.getValue()))
      return std::abs(fa.getValueAsDouble() - expected) < 1e-4;
    return false;
  };
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    auto dense = dyn_cast<DenseElementsAttr>(cst.getValue());
    return dense && dense.isSplat() && isa<FloatType>(dense.getElementType()) &&
           std::abs(dense.getSplatValue<APFloat>().convertToDouble() -
                    expected) < 1e-4;
  }
  if (auto fill = v.getDefiningOp<linalg::FillOp>())
    return matchScalar(fill.getInputs()[0]);
  return false;
}

// Match generic { out = mulf(in0, const c) } with in0 identity-mapped.
static bool matchScaleGeneric(Value v, double c, Value &src) {
  auto g = v.getDefiningOp<linalg::GenericOp>();
  if (!g || !g.hasPureTensorSemantics() || g.getNumDpsInputs() != 2 ||
      !isPureParallelGeneric(g))
    return false;
  auto mul = dyn_cast_or_null<arith::MulFOp>(getSingleBodyOp(g));
  if (!mul)
    return false;
  Block *body = g.getBody();
  Value a0 = body->getArgument(0), a1 = body->getArgument(1);
  if (!((mul.getLhs() == a0 && mul.getRhs() == a1) ||
        (mul.getLhs() == a1 && mul.getRhs() == a0)))
    return false;
  auto maps = g.getIndexingMapsArray();
  if (!maps[0].isIdentity() || !maps[2].isIdentity())
    return false;
  if (!isSplatConst(g.getDpsInputs()[1], c))
    return false;
  src = g.getDpsInputs()[0];
  return true;
}

// Fuse the decomposed exact GELU
//   trunc_f16( (x_f32*0.5) * (erf(x_f32*0.70710677) + 1) ), x_f32 = ext(x_f16)
// into one elementwise generic that keeps the intermediate in registers.
struct FuseGeluPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp truncOp,
                                PatternRewriter &rewriter) const override {
    if (!truncOp.hasPureTensorSemantics() || truncOp.getNumDpsInputs() != 1 ||
        truncOp.getNumResults() != 1 || !isPureParallelGeneric(truncOp))
      return failure();
    auto trunc = dyn_cast_or_null<arith::TruncFOp>(getSingleBodyOp(truncOp));
    auto outTy = dyn_cast<RankedTensorType>(truncOp.getResult(0).getType());
    if (!trunc || !outTy || !outTy.hasStaticShape() ||
        !outTy.getElementType().isF16() ||
        !truncOp.getIndexingMapsArray()[0].isIdentity())
      return failure();
    auto mulOut = truncOp.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
    if (!mulOut || mulOut.getNumDpsInputs() != 2 ||
        !isPureParallelGeneric(mulOut) ||
        !isa_and_nonnull<arith::MulFOp>(getSingleBodyOp(mulOut)))
      return failure();
    auto mmaps = mulOut.getIndexingMapsArray();
    if (!mmaps[0].isIdentity() || !mmaps[1].isIdentity())
      return failure();

    Value extX;
    auto tryOrder = [&](Value halfV, Value plus1V) -> bool {
      Value srcHalf;
      if (!matchScaleGeneric(halfV, 0.5, srcHalf))
        return false;
      auto add = plus1V.getDefiningOp<linalg::GenericOp>();
      if (!add || add.getNumDpsInputs() != 2 || !isPureParallelGeneric(add) ||
          !isa_and_nonnull<arith::AddFOp>(getSingleBodyOp(add)))
        return false;
      auto amaps = add.getIndexingMapsArray();
      if (!amaps[0].isIdentity() || !amaps[1].isIdentity())
        return false;
      Value erfV;
      if (isSplatConst(add.getDpsInputs()[1], 1.0))
        erfV = add.getDpsInputs()[0];
      else if (isSplatConst(add.getDpsInputs()[0], 1.0))
        erfV = add.getDpsInputs()[1];
      else
        return false;
      auto erf = erfV.getDefiningOp<math::ErfOp>();
      if (!erf)
        return false;
      Value srcScaled;
      if (!matchScaleGeneric(erf.getOperand(), 0.70710677, srcScaled) ||
          srcScaled != srcHalf)
        return false;
      extX = srcHalf;
      return true;
    };
    Value m0 = mulOut.getDpsInputs()[0], m1 = mulOut.getDpsInputs()[1];
    if (!tryOrder(m0, m1) && !tryOrder(m1, m0))
      return failure();

    auto ext = extX.getDefiningOp<linalg::GenericOp>();
    if (!ext || ext.getNumDpsInputs() != 1 || !isPureParallelGeneric(ext) ||
        !isa_and_nonnull<arith::ExtFOp>(getSingleBodyOp(ext)) ||
        !ext.getIndexingMapsArray()[0].isIdentity())
      return failure();
    Value srcF16 = ext.getDpsInputs()[0];
    if (srcF16.getType() != outTy)
      return failure();

    Location loc = truncOp.getLoc();
    int64_t rank = outTy.getRank();
    Value empty = tensor::EmptyOp::create(rewriter, loc, outTy.getShape(),
                                          rewriter.getF16Type());
    SmallVector<AffineMap> maps(2, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    auto fused = linalg::GenericOp::create(
        rewriter, loc, TypeRange{outTy}, ValueRange{srcF16}, ValueRange{empty},
        maps, iters, [&](OpBuilder &b, Location l, ValueRange args) {
          FloatType f32 = b.getF32Type();
          Value half = arith::ConstantFloatOp::create(b, l, f32, APFloat(0.5f));
          Value inv =
              arith::ConstantFloatOp::create(b, l, f32, APFloat(0.70710677f));
          Value one = arith::ConstantFloatOp::create(b, l, f32, APFloat(1.0f));
          Value x = arith::ExtFOp::create(b, l, f32, args[0]);
          Value h = arith::MulFOp::create(b, l, x, half);
          Value sc = arith::MulFOp::create(b, l, x, inv);
          Value e = math::ErfOp::create(b, l, sc);
          Value p1 = arith::AddFOp::create(b, l, e, one);
          Value pr = arith::MulFOp::create(b, l, h, p1);
          Value o = arith::TruncFOp::create(b, l, b.getF16Type(), pr);
          linalg::YieldOp::create(b, l, o);
        });
    rewriter.replaceOp(truncOp, fused.getResults());
    return success();
  }
};

class FuseAttentionSoftmaxPass
    : public PassWrapper<FuseAttentionSoftmaxPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseAttentionSoftmaxPass)
  StringRef getArgument() const final { return "fuse-attention-softmax"; }
  StringRef getDescription() const final {
    return "Fold broadcast-zero mask adds and fuse decomposed "
           "log_softmax+exp into linalg.softmax";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    math::MathDialect, tensor::TensorDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldAddZeroTensorPattern, FoldLogSoftmaxExpPattern,
                 FuseGeluPattern>(&getContext());
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

// Decompose a flat row index into the outer (all but last) indices.
static SmallVector<Value> outerIndices(OpBuilder &b, Location loc,
                                       ArrayRef<int64_t> shape, Value flatRow) {
  int64_t rank = shape.size();
  SmallVector<Value> indices(rank - 1);
  Value rem = flatRow;
  for (int64_t d = rank - 2; d >= 0; --d) {
    if (d == 0) {
      indices[d] = rem;
    } else {
      Value dimCst = arith::ConstantIndexOp::create(b, loc, shape[d]);
      indices[d] = arith::RemUIOp::create(b, loc, rem, dimCst);
      rem = arith::DivUIOp::create(b, loc, rem, dimCst);
    }
  }
  return indices;
}

static SmallVector<Value> withCol(ArrayRef<Value> outer, Value col) {
  SmallVector<Value> r(outer.begin(), outer.end());
  r.push_back(col);
  return r;
}

// Lower bufferized linalg.softmax (last dim, static shape, f16/f32) into a
// parallel L1-cache-resident RVV FP32 3-pass kernel with vector math.exp.
struct LowerSoftmaxToVectorPattern
    : public OpRewritePattern<linalg::SoftmaxOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::SoftmaxOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumResults() != 0)
      return failure();
    Value input = op.getInput();
    Value output = op.getOutput();
    auto inTy = dyn_cast<MemRefType>(input.getType());
    auto outTy = dyn_cast<MemRefType>(output.getType());
    if (!inTy || !outTy || !inTy.hasStaticShape() || !outTy.hasStaticShape())
      return failure();
    if (inTy.getShape() != outTy.getShape() ||
        inTy.getElementType() != outTy.getElementType())
      return failure();
    int64_t rank = inTy.getRank();
    if (rank < 2 || static_cast<int64_t>(op.getDimension()) != rank - 1)
      return failure();
    Type elemTy = inTy.getElementType();
    bool isF16 = elemTy.isF16();
    if (!isF16 && !elemTy.isF32())
      return failure();
    if (!isUnitTrailingStride(inTy) || !isUnitTrailingStride(outTy))
      return failure();

    ArrayRef<int64_t> shape = inTy.getShape();
    int64_t N = shape.back();
    if (N < 16 || N > 8192)
      return failure();
    int64_t outerRows = 1;
    for (int64_t d = 0; d < rank - 1; ++d)
      outerRows *= shape[d];
    if (outerRows < 1)
      return failure();

    constexpr int64_t VF = 32;
    int64_t mainN = (N / VF) * VF;
    int64_t tailN = N - mainN;
    int64_t bufSize = ((N + VF - 1) / VF) * VF;

    Location loc = op.getLoc();
    Type f32Ty = rewriter.getF32Type();
    auto idxCst = [&](int64_t v) -> Value {
      return arith::ConstantIndexOp::create(rewriter, loc, v);
    };
    Value c0 = idxCst(0), c1 = idxCst(1), cVF = idxCst(VF);
    Value cOuter = idxCst(outerRows), cMainN = idxCst(mainN);
    Value fZero = arith::ConstantFloatOp::create(
        rewriter, loc, rewriter.getF32Type(), APFloat(0.0f));
    Value fOne = arith::ConstantFloatOp::create(
        rewriter, loc, rewriter.getF32Type(), APFloat(1.0f));
    Value fNegInf = arith::ConstantFloatOp::create(
        rewriter, loc, rewriter.getF32Type(),
        APFloat::getInf(APFloat::IEEEsingle(), /*Negative=*/true));
    Value fClampLo = arith::ConstantFloatOp::create(
        rewriter, loc, rewriter.getF32Type(), APFloat(-65504.0f));
    Value fClampHi = arith::ConstantFloatOp::create(
        rewriter, loc, rewriter.getF32Type(), APFloat(65504.0f));
    Value elemZero = arith::ConstantOp::create(rewriter, loc, elemTy,
                                               rewriter.getZeroAttr(elemTy));

    auto loadVecTy = VectorType::get({VF}, elemTy);
    auto f32VecTy = VectorType::get({VF}, f32Ty);
    auto maskTy = VectorType::get({VF}, rewriter.getI1Type());

    Value vZeroF32 =
        vector::BroadcastOp::create(rewriter, loc, f32VecTy, fZero);
    Value vNegInf =
        vector::BroadcastOp::create(rewriter, loc, f32VecTy, fNegInf);
    Value vClampLo =
        vector::BroadcastOp::create(rewriter, loc, f32VecTy, fClampLo);
    Value vClampHi =
        vector::BroadcastOp::create(rewriter, loc, f32VecTy, fClampHi);
    Value vZeroElem =
        vector::BroadcastOp::create(rewriter, loc, loadVecTy, elemZero);

    auto emitRow = [&](OpBuilder &b, Location l, Value rowIdx) {
      SmallVector<Value> outer = outerIndices(b, l, shape, rowIdx);
      Value buf =
          memref::AllocaOp::create(b, l, MemRefType::get({bufSize}, f32Ty));

      // Pass 1: Load row into L1 stack buffer (clamping f16 infs in f32) + max.
      Value rowMax;
      if (mainN > 0) {
        auto maxFor = scf::ForOp::create(
            b, l, c0, cMainN, cVF, ValueRange{vNegInf},
            [&](OpBuilder &fb, Location fl, Value k, ValueRange it) {
              auto idxs = withCol(outer, k);
              Value raw =
                  vector::LoadOp::create(fb, fl, loadVecTy, input, idxs);
              Value vf =
                  isF16 ? Value(arith::ExtFOp::create(fb, fl, f32VecTy, raw))
                        : raw;
              if (isF16) {
                vf = arith::MinNumFOp::create(fb, fl, vf, vClampHi);
                vf = arith::MaxNumFOp::create(fb, fl, vf, vClampLo);
              }
              vector::StoreOp::create(fb, fl, vf, buf, ValueRange{k});
              Value vmax = arith::MaxNumFOp::create(fb, fl, it[0], vf);
              scf::YieldOp::create(fb, fl, vmax);
            });
        rowMax = vector::ReductionOp::create(
            b, l, vector::CombiningKind::MAXNUMF, maxFor.getResult(0), fNegInf);
      } else {
        rowMax = fNegInf;
      }

      Value tailMask;
      if (tailN > 0) {
        tailMask = vector::CreateMaskOp::create(
            b, l, maskTy,
            ValueRange{arith::ConstantIndexOp::create(b, l, tailN)});
        auto idxs = withCol(outer, cMainN);
        Value raw = vector::MaskedLoadOp::create(b, l, loadVecTy, input, idxs,
                                                 tailMask, vZeroElem);
        Value vf =
            isF16 ? Value(arith::ExtFOp::create(b, l, f32VecTy, raw)) : raw;
        if (isF16) {
          vf = arith::MinNumFOp::create(b, l, vf, vClampHi);
          vf = arith::MaxNumFOp::create(b, l, vf, vClampLo);
        }
        Value maskedForMax =
            arith::SelectOp::create(b, l, tailMask, vf, vNegInf);
        vector::StoreOp::create(b, l, maskedForMax, buf, ValueRange{cMainN});
        rowMax = vector::ReductionOp::create(
            b, l, vector::CombiningKind::MAXNUMF, maskedForMax, rowMax);
      }

      // Pass 2: exp(buf - rowMax) via vector math.exp + sum.
      Value vRowMax = vector::BroadcastOp::create(b, l, f32VecTy, rowMax);
      Value rowSum;
      if (mainN > 0) {
        auto sumFor = scf::ForOp::create(
            b, l, c0, cMainN, cVF, ValueRange{vZeroF32},
            [&](OpBuilder &fb, Location fl, Value k, ValueRange it) {
              Value vf =
                  vector::LoadOp::create(fb, fl, f32VecTy, buf, ValueRange{k});
              Value diff = arith::SubFOp::create(fb, fl, vf, vRowMax);
              Value ex = math::ExpOp::create(fb, fl, diff);
              vector::StoreOp::create(fb, fl, ex, buf, ValueRange{k});
              Value vsum = arith::AddFOp::create(fb, fl, it[0], ex);
              scf::YieldOp::create(fb, fl, vsum);
            });
        rowSum = vector::ReductionOp::create(b, l, vector::CombiningKind::ADD,
                                             sumFor.getResult(0), fZero,
                                             arith::FastMathFlags::reassoc);
      } else {
        rowSum = fZero;
      }

      if (tailN > 0) {
        Value vf =
            vector::LoadOp::create(b, l, f32VecTy, buf, ValueRange{cMainN});
        Value diff = arith::SubFOp::create(b, l, vf, vRowMax);
        Value ex = math::ExpOp::create(b, l, diff);
        Value maskedEx = arith::SelectOp::create(b, l, tailMask, ex, vZeroF32);
        vector::StoreOp::create(b, l, maskedEx, buf, ValueRange{cMainN});
        rowSum = vector::ReductionOp::create(b, l, vector::CombiningKind::ADD,
                                             maskedEx, rowSum,
                                             arith::FastMathFlags::reassoc);
      }

      // Pass 3: Normalize by invSum and store to output.
      Value invSum = arith::DivFOp::create(b, l, fOne, rowSum);
      Value vInvSum = vector::BroadcastOp::create(b, l, f32VecTy, invSum);
      if (mainN > 0) {
        scf::ForOp::create(
            b, l, c0, cMainN, cVF, ValueRange{},
            [&](OpBuilder &fb, Location fl, Value k, ValueRange) {
              Value ex =
                  vector::LoadOp::create(fb, fl, f32VecTy, buf, ValueRange{k});
              Value prob = arith::MulFOp::create(fb, fl, ex, vInvSum);
              Value outV =
                  isF16
                      ? Value(arith::TruncFOp::create(fb, fl, loadVecTy, prob))
                      : prob;
              auto idxs = withCol(outer, k);
              vector::StoreOp::create(fb, fl, outV, output, idxs);
              scf::YieldOp::create(fb, fl);
            });
      }
      if (tailN > 0) {
        Value ex =
            vector::LoadOp::create(b, l, f32VecTy, buf, ValueRange{cMainN});
        Value prob = arith::MulFOp::create(b, l, ex, vInvSum);
        Value outV = isF16
                         ? Value(arith::TruncFOp::create(b, l, loadVecTy, prob))
                         : prob;
        auto idxs = withCol(outer, cMainN);
        vector::MaskedStoreOp::create(b, l, output, idxs, tailMask, outV);
      }
    };

    if (outerRows > 1) {
      scf::ParallelOp::create(rewriter, loc, ValueRange{c0}, ValueRange{cOuter},
                              ValueRange{c1},
                              [&](OpBuilder &pb, Location pl, ValueRange ivs) {
                                emitRow(pb, pl, ivs[0]);
                              });
    } else {
      emitRow(rewriter, loc, c0);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// Lower contiguous FP32 linalg.generic containing a single math.erf into a
// parallel vector<32xf32> loop (so VectorExpApproximationPass can lower it to
// pure RVV polynomial instructions).
struct LowerGenericErfToVectorPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics() || op.getNumDpsInputs() != 1 ||
        op.getNumDpsInits() != 1 || !isPureParallelGeneric(op))
      return failure();
    auto erfOp = dyn_cast_or_null<math::ErfOp>(getSingleBodyOp(op));
    if (!erfOp || erfOp.getOperand() != op.getBody()->getArgument(0))
      return failure();
    auto maps = op.getIndexingMapsArray();
    if (maps.size() != 2 || !maps[0].isIdentity() || !maps[1].isIdentity())
      return failure();

    Value input = op.getDpsInputs()[0];
    Value output = op.getDpsInits()[0];
    auto inTy = dyn_cast<MemRefType>(input.getType());
    auto outTy = dyn_cast<MemRefType>(output.getType());
    if (!inTy || !outTy || !inTy.hasStaticShape() || !outTy.hasStaticShape())
      return failure();
    if (!inTy.getElementType().isF32() || !outTy.getElementType().isF32() ||
        inTy.getShape() != outTy.getShape())
      return failure();
    if (!isUnitTrailingStride(inTy) || !isUnitTrailingStride(outTy))
      return failure();

    ArrayRef<int64_t> shape = inTy.getShape();
    int64_t rank = shape.size();
    int64_t N = shape.back();
    if (rank < 2 || N < 32)
      return failure();
    int64_t outerRows = 1;
    for (int64_t d = 0; d < rank - 1; ++d)
      outerRows *= shape[d];

    constexpr int64_t VF = 32;
    int64_t mainN = (N / VF) * VF;
    int64_t tailN = N - mainN;

    Location loc = op.getLoc();
    Type f32Ty = rewriter.getF32Type();
    auto vecTy = VectorType::get({VF}, f32Ty);
    auto maskTy = VectorType::get({VF}, rewriter.getI1Type());
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value cVF = arith::ConstantIndexOp::create(rewriter, loc, VF);
    Value cOuter = arith::ConstantIndexOp::create(rewriter, loc, outerRows);
    Value cMainN = arith::ConstantIndexOp::create(rewriter, loc, mainN);
    Value fZero = arith::ConstantFloatOp::create(
        rewriter, loc, rewriter.getF32Type(), APFloat(0.0f));
    Value vZero = vector::BroadcastOp::create(rewriter, loc, vecTy, fZero);

    auto emitRow = [&](OpBuilder &b, Location l, Value rowIdx) {
      SmallVector<Value> outer = outerIndices(b, l, shape, rowIdx);
      scf::ForOp::create(b, l, c0, cMainN, cVF, ValueRange{},
                         [&](OpBuilder &fb, Location fl, Value k, ValueRange) {
                           auto idxs = withCol(outer, k);
                           Value v = vector::LoadOp::create(fb, fl, vecTy,
                                                            input, idxs);
                           Value r = math::ErfOp::create(fb, fl, v);
                           vector::StoreOp::create(fb, fl, r, output, idxs);
                           scf::YieldOp::create(fb, fl);
                         });
      if (tailN > 0) {
        Value mask = vector::CreateMaskOp::create(
            b, l, maskTy,
            ValueRange{arith::ConstantIndexOp::create(b, l, tailN)});
        auto idxs = withCol(outer, cMainN);
        Value v =
            vector::MaskedLoadOp::create(b, l, vecTy, input, idxs, mask, vZero);
        Value r = math::ErfOp::create(b, l, v);
        vector::MaskedStoreOp::create(b, l, output, idxs, mask, r);
      }
    };

    if (outerRows > 1) {
      scf::ParallelOp::create(rewriter, loc, ValueRange{c0}, ValueRange{cOuter},
                              ValueRange{c1},
                              [&](OpBuilder &pb, Location pl, ValueRange ivs) {
                                emitRow(pb, pl, ivs[0]);
                              });
    } else {
      emitRow(rewriter, loc, c0);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// Lower a bufferized fused-GELU generic (f16 in/out) to RVV vector loops.
struct LowerFusedGeluToVectorPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics() || op.getNumDpsInputs() != 1 ||
        op.getNumDpsInits() != 1 || !isPureParallelGeneric(op))
      return failure();
    Block *body = op.getBody();
    if (body->getOperations().size() != 11) // 3 consts + 7 ops + yield
      return failure();
    bool hasErf = false;
    for (Operation &o : *body)
      hasErf |= isa<math::ErfOp>(o);
    if (!hasErf)
      return failure();
    auto maps = op.getIndexingMapsArray();
    if (maps.size() != 2 || !maps[0].isIdentity() || !maps[1].isIdentity())
      return failure();
    Value input = op.getDpsInputs()[0];
    Value output = op.getDpsInits()[0];
    auto inTy = dyn_cast<MemRefType>(input.getType());
    auto outTy = dyn_cast<MemRefType>(output.getType());
    if (!inTy || !outTy || !inTy.hasStaticShape() || !outTy.hasStaticShape() ||
        !inTy.getElementType().isF16() || !outTy.getElementType().isF16() ||
        inTy.getShape() != outTy.getShape() || !isUnitTrailingStride(inTy) ||
        !isUnitTrailingStride(outTy))
      return failure();

    ArrayRef<int64_t> shape = inTy.getShape();
    int64_t rank = shape.size();
    int64_t N = shape.back();
    if (rank < 2 || N < 32)
      return failure();
    int64_t outerRows = 1;
    for (int64_t d = 0; d < rank - 1; ++d)
      outerRows *= shape[d];
    constexpr int64_t VF = 32;
    int64_t mainN = (N / VF) * VF, tailN = N - mainN;

    Location loc = op.getLoc();
    FloatType f32Ty = rewriter.getF32Type(), f16Ty = rewriter.getF16Type();
    auto f16Vec = VectorType::get({VF}, f16Ty);
    auto f32Vec = VectorType::get({VF}, f32Ty);
    auto maskTy = VectorType::get({VF}, rewriter.getI1Type());
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value cVF = arith::ConstantIndexOp::create(rewriter, loc, VF);
    Value cOuter = arith::ConstantIndexOp::create(rewriter, loc, outerRows);
    Value cMainN = arith::ConstantIndexOp::create(rewriter, loc, mainN);
    auto splat = [&](float v) {
      Value s =
          arith::ConstantFloatOp::create(rewriter, loc, f32Ty, APFloat(v));
      return vector::BroadcastOp::create(rewriter, loc, f32Vec, s).getResult();
    };
    Value vHalf = splat(0.5f), vInv = splat(0.70710677f), vOne = splat(1.0f);
    Value hZero = arith::ConstantOp::create(rewriter, loc, f16Ty,
                                            rewriter.getZeroAttr(f16Ty));
    Value vHZero = vector::BroadcastOp::create(rewriter, loc, f16Vec, hZero);

    auto compute = [&](OpBuilder &b, Location l, Value raw) -> Value {
      Value x = arith::ExtFOp::create(b, l, f32Vec, raw);
      Value h = arith::MulFOp::create(b, l, x, vHalf);
      Value sc = arith::MulFOp::create(b, l, x, vInv);
      Value e = math::ErfOp::create(b, l, sc);
      Value p1 = arith::AddFOp::create(b, l, e, vOne);
      Value pr = arith::MulFOp::create(b, l, h, p1);
      return arith::TruncFOp::create(b, l, f16Vec, pr);
    };

    auto emitRow = [&](OpBuilder &b, Location l, Value rowIdx) {
      SmallVector<Value> outer = outerIndices(b, l, shape, rowIdx);
      scf::ForOp::create(
          b, l, c0, cMainN, cVF, ValueRange{},
          [&](OpBuilder &fb, Location fl, Value k, ValueRange) {
            auto idxs = withCol(outer, k);
            Value raw = vector::LoadOp::create(fb, fl, f16Vec, input, idxs);
            vector::StoreOp::create(fb, fl, compute(fb, fl, raw), output, idxs);
            scf::YieldOp::create(fb, fl);
          });
      if (tailN > 0) {
        Value mask = vector::CreateMaskOp::create(
            b, l, maskTy,
            ValueRange{arith::ConstantIndexOp::create(b, l, tailN)});
        auto idxs = withCol(outer, cMainN);
        Value raw = vector::MaskedLoadOp::create(b, l, f16Vec, input, idxs,
                                                 mask, vHZero);
        vector::MaskedStoreOp::create(b, l, output, idxs, mask,
                                      compute(b, l, raw));
      }
    };
    if (outerRows > 1) {
      scf::ParallelOp::create(rewriter, loc, ValueRange{c0}, ValueRange{cOuter},
                              ValueRange{c1},
                              [&](OpBuilder &pb, Location pl, ValueRange ivs) {
                                emitRow(pb, pl, ivs[0]);
                              });
    } else {
      emitRow(rewriter, loc, c0);
    }
    rewriter.eraseOp(op);
    return success();
  }
};
class LowerVectorSoftmaxErfPass
    : public PassWrapper<LowerVectorSoftmaxErfPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerVectorSoftmaxErfPass)
  StringRef getArgument() const final { return "lower-vector-softmax-erf"; }
  StringRef getDescription() const final {
    return "Lower bufferized linalg.softmax and elementwise math.erf to "
           "parallel RVV-friendly vector loops";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    math::MathDialect, memref::MemRefDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerSoftmaxToVectorPattern, LowerGenericErfToVectorPattern,
                 LowerFusedGeluToVectorPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

namespace mlir::buddy {
void registerFuseAttentionSoftmaxPass() {
  PassRegistration<FuseAttentionSoftmaxPass>();
}
void registerLowerVectorSoftmaxErfPass() {
  PassRegistration<LowerVectorSoftmaxErfPass>();
}
} // namespace mlir::buddy
