//===- PolynomialApproximateMathPass.cpp ----------------------------------===//
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

// Replace selected scalar and vector math operations with MLIR polynomial
// approximations. This avoids per-lane libm calls on targets without a vector
// math library, at the cost of a small numerical difference.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>
#include <type_traits>

using namespace mlir;

namespace {
// All constants follow the operand's shape, including scalable dimensions.
// No rule depends on a target, vector length, or kernel name.
struct MathBuilder {
  ImplicitLocOpBuilder b;
  Type ft, it;
  MathBuilder(Operation *op, PatternRewriter &rewriter)
      : b(op->getLoc(), rewriter), ft(op->getResult(0).getType()),
        it(isa<VectorType>(ft)
               ? cast<VectorType>(ft).clone(rewriter.getI32Type())
               : Type(rewriter.getI32Type())) {}
  Value splat(Value v, Type t) {
    return isa<VectorType>(t) ? vector::BroadcastOp::create(b, t, v) : v;
  }
  Value f(double x) {
    return splat(arith::ConstantOp::create(b, b.getF32FloatAttr(x)), ft);
  }
  Value i(int32_t x) {
    return splat(arith::ConstantOp::create(b, b.getI32IntegerAttr(x)), it);
  }
  Value add(Value x, Value y) { return arith::AddFOp::create(b, x, y); }
  Value sub(Value x, Value y) { return arith::SubFOp::create(b, x, y); }
  Value mul(Value x, Value y) { return arith::MulFOp::create(b, x, y); }
  Value div(Value x, Value y) { return arith::DivFOp::create(b, x, y); }
  Value neg(Value x) { return arith::NegFOp::create(b, x); }
  Value fma(Value x, Value y, Value z) {
    return math::FmaOp::create(b, x, y, z);
  }
  Value cmp(arith::CmpFPredicate p, Value x, Value y) {
    return arith::CmpFOp::create(b, p, x, y);
  }
  Value sel(Value p, Value x, Value y) {
    return arith::SelectOp::create(b, p, x, y);
  }
  Value bits(Value x) { return arith::BitcastOp::create(b, it, x); }
  Value asFloat(Value x) { return arith::BitcastOp::create(b, ft, x); }
  Value andi(Value x, Value y) { return arith::AndIOp::create(b, x, y); }
  Value ori(Value x, Value y) { return arith::OrIOp::create(b, x, y); }
};

// Cephes polynomial and reconstruction from LLVM's log approximation, with
// exact power-of-two normalization instead of clamping subnormal inputs.
template <typename Op> struct LogFix : OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final {
    if (!getElementTypeOrSelf(op.getOperand()).isF32())
      return failure();
    MathBuilder m(op, rewriter);
    Value input = op.getOperand(), zero = m.f(0), one = m.f(1);
    Value small = m.cmp(arith::CmpFPredicate::OLT, input, m.f(0x1p-126));
    Value a = m.sel(small, m.mul(input, m.f(0x1p23)), input);
    Value bits = m.bits(a);
    Value x = m.asFloat(m.ori(m.andi(bits, m.i(0x7fffff)), m.i(0x3f000000)));
    Value exponent = arith::ShRUIOp::create(m.b, bits, m.i(23));
    Value e = m.sub(arith::SIToFPOp::create(m.b, m.ft, exponent), m.f(126));
    e = m.sub(e, m.sel(small, m.f(23), zero));
    Value mask =
        m.cmp(arith::CmpFPredicate::OLT, x, m.f(0.707106781186547524f));
    Value tmp = m.sel(mask, x, zero);
    x = m.add(m.sub(x, one), tmp);
    e = m.sub(e, m.sel(mask, one, zero));
    Value x2 = m.mul(x, x), x3 = m.mul(x2, x);
    Value y0 = m.fma(m.f(7.0376836292E-2f), x, m.f(-1.1514610310E-1f));
    Value y1 = m.fma(m.f(-1.2420140846E-1f), x, m.f(1.4249322787E-1f));
    Value y2 = m.fma(m.f(2.0000714765E-1f), x, m.f(-2.4999993993E-1f));
    y0 = m.fma(y0, x, m.f(1.1676998740E-1f));
    y1 = m.fma(y1, x, m.f(-1.6668057665E-1f));
    y2 = m.fma(y2, x, m.f(3.3333331174E-1f));
    y0 = m.fma(y0, x3, y1);
    y0 = m.fma(y0, x3, y2);
    y0 = m.fma(m.f(-0.5), x2, m.mul(y0, x3));
    x = m.add(x, y0);
    if constexpr (std::is_same_v<Op, math::Log2Op>)
      x = m.fma(x, m.f(1.4426950408889634074), e);
    else
      x = m.fma(e, m.f(0.6931471805599453094), x);
    Value inf = m.f(std::numeric_limits<float>::infinity());
    Value invalid = m.cmp(arith::CmpFPredicate::ULT, input, zero);
    Value isZero = m.cmp(arith::CmpFPredicate::OEQ, input, zero);
    Value isInf = m.cmp(arith::CmpFPredicate::OEQ, input, inf);
    x = m.sel(isInf, inf, x);
    x = m.sel(invalid, m.f(std::numeric_limits<float>::quiet_NaN()), x);
    rewriter.replaceOp(op, m.sel(isZero, m.neg(inf), x));
    return success();
  }
};

// Three-piece Cody-Waite reduction, followed by polynomials on [-pi/4, pi/4].
// Cody-Waite is used for |x| <= 1e5. A block-wide test guards the integer
// Payne-Hanek reduction, so ordinary vectors never evaluate the wide product.
template <typename Op> struct TrigFix : OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final {
    if (!getElementTypeOrSelf(op.getOperand()).isF32())
      return failure();
    if (auto vt = dyn_cast<VectorType>(op.getResult().getType()))
      if (llvm::count(vt.getScalableDims(), true) > 1)
        return failure();
    constexpr bool sine = std::is_same_v<Op, math::SinOp>;
    MathBuilder m(op, rewriter);
    Value input = op.getOperand(), zero = m.f(0), one = m.f(1);
    Value abs = math::AbsFOp::create(m.b, input);
    // Nonnegative IEEE bits have the same ordering as finite values, and
    // also send NaNs/infinities to the guarded path. One comparison suffices.
    Value large = arith::CmpIOp::create(m.b, arith::CmpIPredicate::ugt,
                                        m.bits(abs), m.i(0x47c35000));
    Value anyLarge = large;
    if (auto vt = dyn_cast<VectorType>(large.getType())) {
      if (vt.getRank() != 1) {
        // Flatten fixed or singly scalable shapes for a one-dimensional OR.
        Type flat = VectorType::get({vt.getNumElements()}, rewriter.getI1Type(),
                                    {vt.isScalable()});
        anyLarge = vector::ShapeCastOp::create(m.b, flat, large);
      }
      anyLarge =
          vector::ReductionOp::create(m.b, vector::CombiningKind::OR, anyLarge);
    }
    auto reduceCW = [&](Value x) -> std::pair<Value, Value> {
      Value k =
          math::RoundEvenOp::create(m.b, m.mul(x, m.f(0.636619772367581343)));
      Value nk = m.neg(k);
      Value r = m.fma(nk, m.f(1.5707962512969970703125), x);
      r = m.fma(nk, m.f(7.54978941586159635335e-8), r);
      r = m.fma(nk, m.f(5.39030252995776476554e-15), r);
      return {r, arith::FPToSIOp::create(m.b, m.it, k)};
    };
    auto branch = scf::IfOp::create(m.b, TypeRange{m.ft, m.it}, anyLarge, true);
    {
      OpBuilder::InsertionGuard guard(m.b);
      m.b.setInsertionPointToStart(branch.thenBlock());
      // For x = M * 2^(e-23), form the exact 24-by-192-bit product M*C,
      // C = floor((2/pi)*2^192). Its binary point is at S = 215-e.
      // Extract bits [S-62, S+1] to obtain two quadrant bits and 62 fraction
      // bits. Discarding the tail of C contributes less than 2^-64 turns even
      // at FLT_MAX. All shifts are bounded, including in small lanes of a mixed
      // vector.
      Type wt = isa<VectorType>(m.ft)
                    ? cast<VectorType>(m.ft).clone(rewriter.getI64Type())
                    : Type(rewriter.getI64Type());
      auto wi = [&](int64_t v) -> Value {
        return m.splat(arith::ConstantOp::create(m.b, m.b.getI64IntegerAttr(v)),
                       wt);
      };
      auto shl = [&](Value a, Value n) -> Value {
        return arith::ShLIOp::create(m.b, a, n);
      };
      auto shr = [&](Value a, Value n) -> Value {
        return arith::ShRUIOp::create(m.b, a, n);
      };
      Value abits = m.bits(abs);
      Value mantissa = arith::ExtUIOp::create(
          m.b, wt, m.ori(m.andi(abits, m.i(0x7fffff)), m.i(0x800000)));
      // Keep the six limb constants in a small table in the slow region.
      // A loop carries only the three words needed for the binary window.
      // The compact IR needs only a carry and the selected words. Backends may
      // unroll this loop; all its arithmetic remains inside the slow branch.
      auto tableTy = VectorType::get({6}, rewriter.getI32Type());
      Value table = arith::ConstantOp::create(
          m.b, tableTy,
          DenseIntElementsAttr::get(
              tableTy,
              ArrayRef<int32_t>{0x3c439041, int32_t(0xdb629599),
                                int32_t(0xf534ddc0), int32_t(0xfc2757d1),
                                0x4e441529, int32_t(0xa2f9836e)}));
      Value start = arith::SubIOp::create(m.b, m.i(280), shr(abits, m.i(23)));
      Value index = shr(start, m.i(5));
      Value shift = arith::ExtUIOp::create(m.b, wt, m.andi(start, m.i(31)));
      Value indices[] = {index, arith::AddIOp::create(m.b, index, m.i(1)),
                         arith::AddIOp::create(m.b, index, m.i(2))};
      Value izero = arith::ConstantIndexOp::create(m.b, 0);
      Value isix = arith::ConstantIndexOp::create(m.b, 6);
      Value ione = arith::ConstantIndexOp::create(m.b, 1);
      auto loop = scf::ForOp::create(m.b, izero, isix, ione,
                                     ValueRange{wi(0), m.i(0), m.i(0), m.i(0)});
      {
        OpBuilder::InsertionGuard guard(m.b);
        m.b.setInsertionPointToStart(loop.getBody());
        Value j = loop.getInductionVar();
        Value limb = vector::ExtractOp::create(m.b, table, OpFoldResult(j));
        limb = arith::ExtUIOp::create(m.b, rewriter.getI64Type(), limb);
        limb = m.splat(limb, wt);
        Value product = arith::MulIOp::create(m.b, mantissa, limb);
        product =
            arith::AddIOp::create(m.b, product, loop.getRegionIterArgs()[0]);
        Value lo = arith::TruncIOp::create(m.b, m.it, product);
        Value j32 = arith::IndexCastOp::create(m.b, rewriter.getI32Type(), j);
        j32 = m.splat(j32, m.it);
        SmallVector<Value> next{shr(product, wi(32))};
        for (int offset = 0; offset < 3; ++offset) {
          Value use = arith::CmpIOp::create(m.b, arith::CmpIPredicate::eq,
                                            indices[offset], j32);
          next.push_back(m.sel(use, lo, loop.getRegionIterArgs()[offset + 1]));
        }
        scf::YieldOp::create(m.b, next);
      }
      auto word = [&](int offset) -> Value {
        Value useCarry = arith::CmpIOp::create(m.b, arith::CmpIPredicate::eq,
                                               indices[offset], m.i(6));
        Value carry = arith::TruncIOp::create(m.b, m.it, loop.getResult(0));
        Value v = m.sel(useCarry, carry, loop.getResult(offset + 1));
        return arith::ExtUIOp::create(m.b, wt, v);
      };
      Value low = m.ori(shl(word(1), wi(32)), word(0));
      Value high = shl(word(2), arith::SubIOp::create(m.b, wi(32), shift));
      Value window = m.ori(shr(low, shift), shl(high, wi(32)));
      Value fraction = m.andi(window, wi((int64_t(1) << 62) - 1));
      Value roundUp = arith::CmpIOp::create(m.b, arith::CmpIPredicate::uge,
                                            fraction, wi(int64_t(1) << 61));
      Value phq = arith::AddIOp::create(m.b, shr(window, wi(62)),
                                        m.sel(roundUp, wi(1), wi(0)));
      phq = arith::TruncIOp::create(m.b, m.it, phq);
      fraction = arith::SubIOp::create(
          m.b, fraction, m.sel(roundUp, wi(int64_t(1) << 62), wi(0)));
      Value phr = arith::SIToFPOp::create(m.b, m.ft, fraction);
      phr = m.mul(m.mul(phr, m.f(0x1p-62)), m.f(1.57079632679489661923));
      Value inputNegative = arith::CmpIOp::create(
          m.b, arith::CmpIPredicate::slt, m.bits(input), m.i(0));
      phq = m.sel(inputNegative, arith::SubIOp::create(m.b, m.i(0), phq), phq);
      phr = m.sel(inputNegative, m.neg(phr), phr);

      auto [r, q] = reduceCW(m.sel(large, zero, input));
      r = m.sel(large, phr, r);
      q = m.sel(large, phq, q);
      scf::YieldOp::create(m.b, ValueRange{r, q});
    }
    {
      OpBuilder::InsertionGuard guard(m.b);
      m.b.setInsertionPointToStart(branch.elseBlock());
      auto [r, q] = reduceCW(input);
      scf::YieldOp::create(m.b, ValueRange{r, q});
    }
    Value r = branch.getResult(0), q = branch.getResult(1);
    if constexpr (!sine)
      q = arith::AddIOp::create(m.b, q, m.i(1));
    Value odd = arith::CmpIOp::create(m.b, arith::CmpIPredicate::ne,
                                      m.andi(q, m.i(1)), m.i(0));
    Value negative = arith::CmpIOp::create(m.b, arith::CmpIPredicate::ne,
                                           m.andi(q, m.i(2)), m.i(0));
    Value z = m.mul(r, r);
    // Share powers and accumulate from the smallest terms. Keeping constants
    // in FMA multiplier positions permits vector-scalar instructions, rather
    // than loop-invariant coefficient vectors spilled by the slow branch.
    // Round-to-even above also avoids a persistent vector constant for 0.5.
    SmallVector<Value> powers{z};
    for (int j = 1; j < 5; ++j)
      powers.push_back(m.mul(powers.back(), z));
    auto polynomial = [&](ArrayRef<double> coefficients) -> Value {
      Value p = m.mul(powers.back(), m.f(coefficients.back()));
      for (int j = 3; j >= 0; --j)
        p = m.fma(powers[j], m.f(coefficients[j]), p);
      return m.add(p, one);
    };
    Value sp = polynomial(
        {-1.0 / 6, 1.0 / 120, -1.0 / 5040, 1.0 / 362880, -1.0 / 39916800});
    Value cp =
        polynomial({-0.5, 1.0 / 24, -1.0 / 720, 1.0 / 40320, -1.0 / 3628800});
    Value p = m.sel(odd, cp, m.mul(r, sp));
    p = m.sel(negative, m.neg(p), p);
    if constexpr (sine) {
      // Also preserves the sign of -0, which the range-reduction FMAs lose.
      Value tiny = m.cmp(arith::CmpFPredicate::OLT, abs, m.f(0x1p-12));
      p = m.sel(tiny, input, p);
    }
    Value nonfinite = m.cmp(arith::CmpFPredicate::UGE, abs,
                            m.f(std::numeric_limits<float>::infinity()));
    rewriter.replaceOp(
        op, m.sel(nonfinite, m.f(std::numeric_limits<float>::quiet_NaN()), p));
    return success();
  }
};

struct Atan2Fix : OpRewritePattern<math::Atan2Op> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(math::Atan2Op op,
                                PatternRewriter &rewriter) const final {
    if (!getElementTypeOrSelf(op.getResult()).isF32())
      return failure();
    MathBuilder m(op, rewriter);
    Value y = op.getOperand(0), x = op.getOperand(1);
    Value ax = math::AbsFOp::create(m.b, x), ay = math::AbsFOp::create(m.b, y);
    Value swap = m.cmp(arith::CmpFPredicate::OGT, ay, ax);
    Value lo = m.sel(swap, ax, ay), hi = m.sel(swap, ay, ax);
    Value zero = m.f(0), inf = m.f(std::numeric_limits<float>::infinity());
    Value hiZero = m.cmp(arith::CmpFPredicate::OEQ, hi, zero);
    Value ratio = m.div(lo, m.sel(hiZero, m.f(1), hi));
    Value bothInf = m.cmp(arith::CmpFPredicate::OEQ, lo, inf);
    ratio = m.sel(bothInf, m.f(1), ratio);
    // LLVM's atan rational approximation, specialized to ratio in [0,1].
    // Build it here so selecting atan2 does not rewrite unrelated atan ops.
    Value mapped = m.cmp(arith::CmpFPredicate::OGT, ratio, m.f(0.66));
    Value numerator = m.sel(mapped, m.sub(ratio, m.f(1)), ratio);
    Value denominator = m.sel(mapped, m.add(ratio, m.f(1)), m.f(1));
    Value t = m.div(numerator, denominator), tt = m.mul(t, t);
    Value n = m.f(-8.750608600031904122785e-1);
    n = m.fma(tt, n, m.f(-1.615753718733365076637e1));
    n = m.fma(tt, n, m.f(-7.500855792314704667340e1));
    n = m.fma(tt, n, m.f(-1.228866684490136173410e2));
    n = m.fma(tt, n, m.f(-6.485021904942025371773e1));
    n = m.mul(n, tt);
    // Cephes uses p1evl: the denominator has a leading coefficient of one.
    // Omitting tt here loses about 42 ulp near a ratio of 0.66.
    Value d = m.add(tt, m.f(2.485846490142306297962e1));
    d = m.fma(tt, d, m.f(1.650270098316988542046e2));
    d = m.fma(tt, d, m.f(4.328810604912902668951e2));
    d = m.fma(tt, d, m.f(4.853903996359136964868e2));
    d = m.fma(tt, d, m.f(1.945506571482613964425e2));
    Value a = m.fma(m.div(n, d), t, t);
    a = m.sel(mapped, m.add(m.f(0.78539816339744830962), a), a);
    a = m.sel(swap, m.sub(m.f(1.57079632679489661923), a), a);
    Value xNegative = arith::CmpIOp::create(m.b, arith::CmpIPredicate::slt,
                                            m.bits(x), m.i(0));
    a = m.sel(xNegative, m.sub(m.f(3.14159265358979323846), a), a);
    a = math::CopySignOp::create(m.b, a, y);
    Value nan = m.cmp(arith::CmpFPredicate::UNO, x, y);
    rewriter.replaceOp(op, m.sel(nan, m.add(x, y), a));
    return success();
  }
};

struct CbrtFix : OpRewritePattern<math::CbrtOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(math::CbrtOp op,
                                PatternRewriter &rewriter) const final {
    if (!getElementTypeOrSelf(op.getResult()).isF32())
      return failure();
    MathBuilder m(op, rewriter);
    Value x = op.getOperand(), a = math::AbsFOp::create(m.b, x);
    Value zero = m.f(0), inf = m.f(std::numeric_limits<float>::infinity());
    Value small = m.cmp(arith::CmpFPredicate::OLT, a, m.f(0x1p-126));
    Value scaled = m.sel(small, m.mul(a, m.f(0x1p24)), a);
    Value special = m.ori(m.cmp(arith::CmpFPredicate::OEQ, a, zero),
                          m.cmp(arith::CmpFPredicate::UGE, a, inf));
    a = m.sel(special, m.f(1), scaled);
    // Hacker's Delight seed, as in LLVM, on a positive normal input only.
    Value ix = m.bits(a);
    Value seed =
        arith::AddIOp::create(m.b, arith::ShRUIOp::create(m.b, ix, m.i(2)),
                              arith::ShRUIOp::create(m.b, ix, m.i(4)));
    seed = arith::AddIOp::create(m.b, seed,
                                 arith::ShRUIOp::create(m.b, seed, m.i(4)));
    seed = arith::AddIOp::create(m.b, seed,
                                 arith::ShRUIOp::create(m.b, seed, m.i(8)));
    seed = arith::AddIOp::create(m.b, seed, m.i(0x2a5137a0));
    Value t = m.asFloat(seed);
    // Three Newton corrections, expressed to avoid rounding 2*t + a/t^2.
    for (int j = 0; j < 3; ++j)
      t = m.fma(m.sub(m.div(a, m.mul(t, t)), t), m.f(1.0 / 3), t);
    t = m.mul(t, m.sel(small, m.f(0x1p-8), m.f(1)));
    t = math::CopySignOp::create(m.b, t, x);
    rewriter.replaceOp(op, m.sel(special, x, t));
    return success();
  }
};

class PolynomialApproximateMathPass
    : public PassWrapper<PolynomialApproximateMathPass, OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PolynomialApproximateMathPass)
  PolynomialApproximateMathPass() = default;
  PolynomialApproximateMathPass(const PolynomialApproximateMathPass &) {}

  StringRef getArgument() const final { return "polynomial-approximate-math"; }
  StringRef getDescription() const final {
    return "Replace selected math ops (default: exp) by portable f32 "
           "approximations with guarded full-range sin/cos reduction.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, math::MathDialect,
                    vector::VectorDialect, scf::SCFDialect>();
  }

  ListOption<std::string> ops{*this, "ops",
                              llvm::cl::desc("Math op mnemonics to approximate "
                                             "(default: exp)")};

  void runOnOperation() override {
    SmallVector<std::string> selected;
    for (StringRef op : ops) {
      op.consume_front("math.");
      selected.push_back(op.str());
    }
    if (selected.empty())
      selected.push_back("exp");
    RewritePatternSet patterns(&getContext());
    auto predicate = [&](StringRef name) {
      name.consume_front("math.");
      return llvm::is_contained(selected, name.str());
    };
    // f16/bf16 math is extended to f32 first, so it is approximated as well.
    populateMathF32ExpansionPatterns(patterns, predicate);
    populateMathPolynomialApproximationPatterns(patterns, predicate);
    if (predicate("log"))
      patterns.add<LogFix<math::LogOp>>(&getContext(), PatternBenefit(2));
    if (predicate("log2"))
      patterns.add<LogFix<math::Log2Op>>(&getContext(), PatternBenefit(2));
    if (predicate("sin"))
      patterns.add<TrigFix<math::SinOp>>(&getContext(), PatternBenefit(2));
    if (predicate("cos"))
      patterns.add<TrigFix<math::CosOp>>(&getContext(), PatternBenefit(2));
    if (predicate("atan2"))
      patterns.add<Atan2Fix>(&getContext(), PatternBenefit(2));
    if (predicate("cbrt"))
      patterns.add<CbrtFix>(&getContext(), PatternBenefit(2));
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

namespace mlir {
namespace buddy {
void registerPolynomialApproximateMathPass() {
  PassRegistration<PolynomialApproximateMathPass>();
}
} // namespace buddy
} // namespace mlir
