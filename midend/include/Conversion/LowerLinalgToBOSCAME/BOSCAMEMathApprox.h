//===- BOSCAMEMathApprox.h - Explicitly authorized f32 approximations -----===//
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
// Explicitly authorized f32 math approximations (erf / exp). Recognition via
// `isSupportedMathApprox` is separate from authorization: callers must verify
// the Kernel Contract and the planner recipe first. Builders emit separate
// multiplies/adds with no fast-math flags for scalar and fixed-vector f32.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEMATHAPPROX_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEMATHAPPROX_H

#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEMathApproxCoefficients.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include <cassert>
#include <cstdint>
#include <limits>

namespace buddy::boscame {

// Recognition is separate from authorization: the caller must first verify
// the neutral contract and the recipe recorded by the planner. No generic
// math rewrite calls this helper.
inline bool isSupportedMathApprox(mlir::Operation &op) {
  return mlir::isa<mlir::math::ErfOp, mlir::math::ExpOp>(op) &&
         op.getNumOperands() == 1 && op.getOperand(0).getType().isF32() &&
         op.getNumResults() == 1 && op.getResult(0).getType().isF32();
}

namespace mathapprox {

// These builders deliberately set no fast-math flags and emit separate
// multiplies/adds. Both scalar and fixed-vector f32 inputs are supported.
struct Expression {
  mlir::OpBuilder &b;
  mlir::Location loc;
  mlir::Type type;

  mlir::Type integerType() const {
    if (auto vector = mlir::dyn_cast<mlir::VectorType>(type))
      return mlir::VectorType::get(vector.getShape(), b.getI32Type());
    return b.getI32Type();
  }
  mlir::Value f(float value) const {
    auto scalar = b.getF32FloatAttr(value);
    if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type))
      return mlir::arith::ConstantOp::create(
          b, loc, type, mlir::DenseElementsAttr::get(shaped, scalar));
    return mlir::arith::ConstantOp::create(b, loc, scalar);
  }
  mlir::Value i(int32_t value) const {
    auto scalar = b.getI32IntegerAttr(value);
    if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(integerType()))
      return mlir::arith::ConstantOp::create(
          b, loc, integerType(), mlir::DenseElementsAttr::get(shaped, scalar));
    return mlir::arith::ConstantOp::create(b, loc, scalar);
  }
  mlir::Value add(mlir::Value a, mlir::Value c) const {
    return mlir::arith::AddFOp::create(b, loc, a, c);
  }
  mlir::Value sub(mlir::Value a, mlir::Value c) const {
    return mlir::arith::SubFOp::create(b, loc, a, c);
  }
  mlir::Value mul(mlir::Value a, mlir::Value c) const {
    return mlir::arith::MulFOp::create(b, loc, a, c);
  }
  mlir::Value cmp(mlir::arith::CmpFPredicate pred, mlir::Value a,
                  mlir::Value c) const {
    return mlir::arith::CmpFOp::create(b, loc, pred, a, c);
  }
  mlir::Value select(mlir::Value condition, mlir::Value a,
                     mlir::Value c) const {
    return mlir::arith::SelectOp::create(b, loc, condition, a, c);
  }
  mlir::Value bits(mlir::Value a) const {
    return mlir::arith::BitcastOp::create(b, loc, integerType(), a);
  }
  mlir::Value floating(mlir::Value a) const {
    return mlir::arith::BitcastOp::create(b, loc, type, a);
  }
  mlir::Value pow2(mlir::Value exponent) const {
    auto biased = mlir::arith::AddIOp::create(b, loc, exponent, i(127));
    auto encoded = mlir::arith::ShLIOp::create(b, loc, biased, i(23));
    return floating(encoded);
  }
};

inline mlir::Value emitExp(Expression e, mlir::Value input) {
  using mlir::arith::CmpFPredicate;
  // Replace NaNs before conversion, then bound *every* eagerly evaluated lane.
  // The final selects restore IEEE special values and exact overflow/zero
  // thresholds for binary32 inputs, independently of the polynomial result.
  auto unordered = e.cmp(CmpFPredicate::UNO, input, input);
  auto x = e.select(unordered, e.f(0), input);
  x = e.select(e.cmp(CmpFPredicate::OLT, x, e.f(-104)), e.f(-104), x);
  x = e.select(e.cmp(CmpFPredicate::OGT, x, e.f(89)), e.f(89), x);
  auto offset =
      e.select(e.cmp(CmpFPredicate::OLT, x, e.f(0)), e.f(-0.5f), e.f(0.5f));
  auto rounded = e.add(e.mul(x, e.f(kLog2E)), offset);
  auto k = mlir::arith::FPToSIOp::create(e.b, e.loc, e.integerType(), rounded);
  auto kf = mlir::arith::SIToFPOp::create(e.b, e.loc, e.type, k);
  auto r = e.sub(e.sub(x, e.mul(kf, e.f(kLn2Hi))), e.mul(kf, e.f(kLn2Lo)));
  auto p = e.f(kExpCoefficients[7]);
  for (int n = 6; n >= 0; --n)
    p = e.add(e.mul(p, r), e.f(kExpCoefficients[n]));

  // Each constructed exponent is normal: k1 in [-126,127], k2 in [-24,1].
  // Scaling by k2 first keeps the intermediate normal. The final multiply is
  // the sole rounding to subnormal; there is no flush-to-zero approximation.
  auto k1 =
      e.select(mlir::arith::CmpIOp::create(
                   e.b, e.loc, mlir::arith::CmpIPredicate::slt, k, e.i(-126)),
               e.i(-126), k);
  k1 = e.select(mlir::arith::CmpIOp::create(
                    e.b, e.loc, mlir::arith::CmpIPredicate::sgt, k1, e.i(127)),
                e.i(127), k1);
  auto k2 = mlir::arith::SubIOp::create(e.b, e.loc, k, k1);
  auto result = e.mul(e.mul(p, e.pow2(k2)), e.pow2(k1));
  result =
      e.select(e.cmp(CmpFPredicate::OLE, input, e.f(-103.97208404541015625f)),
               e.f(0), result);
  result = e.select(e.cmp(CmpFPredicate::OGE, input, e.f(88.72283935546875f)),
                    e.f(std::numeric_limits<float>::infinity()), result);
  return e.select(unordered, input, result);
}

inline mlir::Value emitErf(Expression e, mlir::Value input) {
  using mlir::arith::CmpFPredicate;
  auto magnitude = e.floating(
      mlir::arith::AndIOp::create(e.b, e.loc, e.bits(input), e.i(0x7fffffff)));
  auto unordered = e.cmp(CmpFPredicate::UNO, input, input);
  auto x = e.select(unordered, e.f(0), magnitude);
  x = e.select(e.cmp(CmpFPredicate::OGT, x, e.f(4)), e.f(4), x);

  // The odd branch preserves very small values without cancellation, including
  // gradual underflow. Its only division is embodied in generated constants.
  auto smallX = e.select(e.cmp(CmpFPredicate::OGT, x, e.f(0.5f)), e.f(0.5f), x);
  auto z = e.mul(smallX, smallX);
  auto small = e.f(kErfSmallCoefficients[7]);
  for (int n = 6; n >= 0; --n)
    small = e.add(e.mul(small, z), e.f(kErfSmallCoefficients[n]));
  small = e.mul(smallX, small);

  // Select coefficients per lane, then evaluate a single degree-11 polynomial.
  // All centers are dyadic and subtraction is exact by Sterbenz's lemma on
  // each selected interval. Clamp unused lanes so no speculative overflow is
  // introduced by vector selects.
  auto largeX = e.select(e.cmp(CmpFPredicate::OLT, x, e.f(0.5f)), e.f(0.5f), x);
  mlir::Value center = e.f(kErfCenters[0]);
  mlir::SmallVector<mlir::Value, 12> coefficients;
  for (int n = 0; n < 12; ++n)
    coefficients.push_back(e.f(kErfCoefficients[0][n]));
  for (int segment = 1; segment < 7; ++segment) {
    auto choose = e.cmp(CmpFPredicate::OGE, largeX, e.f(0.5f + 0.5f * segment));
    center = e.select(choose, e.f(kErfCenters[segment]), center);
    for (int n = 0; n < 12; ++n)
      coefficients[n] =
          e.select(choose, e.f(kErfCoefficients[segment][n]), coefficients[n]);
  }
  auto h = e.sub(largeX, center);
  auto large = coefficients[11];
  for (int n = 10; n >= 0; --n)
    large = e.add(e.mul(large, h), coefficients[n]);
  auto positive =
      e.select(e.cmp(CmpFPredicate::OLT, x, e.f(0.5f)), small, large);
  positive = e.select(e.cmp(CmpFPredicate::OGE, x, e.f(4)), e.f(1), positive);
  auto sign = mlir::arith::AndIOp::create(
      e.b, e.loc, e.bits(input), e.i(std::numeric_limits<int32_t>::min()));
  auto result = e.floating(
      mlir::arith::OrIOp::create(e.b, e.loc, e.bits(positive), sign));
  // Preserve signed zero exactly; NaN payload preservation is incidental and
  // is not part of the numerical contract.
  result = e.select(e.cmp(CmpFPredicate::OEQ, input, e.f(0)), input, result);
  return e.select(unordered, input, result);
}

} // namespace mathapprox

// Emit the authorized erf/exp approximation for `op` applied to `input`.
// Preconditions: `isSupportedMathApprox(op)` and a verified planner recipe.
inline mlir::Value emitMathApprox(mlir::OpBuilder &builder, mlir::Operation &op,
                                  mlir::Value input) {
  assert(isSupportedMathApprox(op) && "unsupported approximation");
  mathapprox::Expression expression{builder, op.getLoc(), input.getType()};
  if (mlir::isa<mlir::math::ExpOp>(op))
    return mathapprox::emitExp(expression, input);
  return mathapprox::emitErf(expression, input);
}

} // namespace buddy::boscame

#endif
