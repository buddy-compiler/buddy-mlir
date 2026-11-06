//===- BOSCAMEPointwise.h - Supported lane-wise scalar expressions --------===//
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
// Closed set of lane-wise scalar / vector pointwise expressions the BOSCAME
// planner may lower (arith + selected math.erf/exp recipes). Couples recipe
// names to MathAccuracy budgets; recognition alone never authorizes approx
// emission without a matching contract.math promise.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEPOINTWISE_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEPOINTWISE_H

#include "Analysis/MathAccuracy.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include <algorithm>

namespace buddy::boscame {

// Named approximation recipe for supported math ops; empty if unrecognized.
inline llvm::StringRef pointwiseMathRecipe(mlir::Operation &op) {
  if (mlir::isa<mlir::math::ErfOp>(op))
    return "erf.f32.piecewise-taylor.v1";
  if (mlir::isa<mlir::math::ExpOp>(op))
    return "exp.f32.range-reduced-taylor.v1";
  return {};
}

// Absolute error budget paired with `pointwiseMathRecipe`.
inline double pointwiseMathAbsError(mlir::Operation &op) {
  return mlir::isa<mlir::math::ErfOp>(op) ? 2e-6 : 0x1p-146;
}

// Relative error budget paired with `pointwiseMathRecipe`.
inline double pointwiseMathRelError(mlir::Operation &op) {
  return mlir::isa<mlir::math::ExpOp>(op) ? 2e-6 : 0.0;
}

// True when `op` is a recognized f32 math approx with a matching accuracy
// promise and no fast-math flags.
inline bool supportedPointwiseMath(mlir::Operation &op) {
  using namespace mlir;
  if (pointwiseMathRecipe(op).empty() || op.getNumOperands() != 1 ||
      op.getNumResults() != 1 || !op.getOperand(0).getType().isF32() ||
      !op.getResult(0).getType().isF32())
    return false;
  if (auto erf = dyn_cast<math::ErfOp>(op))
    if (erf.getFastmath() != arith::FastMathFlags::none)
      return false;
  if (auto exp = dyn_cast<math::ExpOp>(op))
    if (exp.getFastmath() != arith::FastMathFlags::none)
      return false;
  return ::buddy::contract::permitsMathError(op, pointwiseMathAbsError(op),
                                             pointwiseMathRelError(op));
}

// Element types admitted in lane-wise pointwise payloads.
inline bool supportedPointwiseScalar(mlir::Type type) {
  return type.isF32() || type.isInteger(1) || type.isInteger(8) ||
         type.isInteger(16) || type.isInteger(32) || type.isInteger(64);
}

// True when `op` is either supported math or a closed arith/math pointwise.
inline bool supportedPointwiseOperation(mlir::Operation &op) {
  using namespace mlir;
  return supportedPointwiseMath(op) ||
         (op.getNumResults() == 1 && op.getNumRegions() == 0 &&
          llvm::all_of(op.getOperandTypes(), supportedPointwiseScalar) &&
          supportedPointwiseScalar(op.getResult(0).getType()) &&
          isa<arith::ConstantOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
              arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
              arith::NegFOp, arith::CmpFOp, arith::CmpIOp, arith::SelectOp,
              arith::MaxNumFOp, arith::MinNumFOp, arith::FPToSIOp,
              arith::SIToFPOp, arith::TruncIOp, arith::ExtSIOp, arith::ExtUIOp,
              math::AbsFOp>(op));
}

// Use one common lane count for the whole expression. Narrow results and
// masks occupy smaller vectors; their memory footprint is never promoted to
// the widest operand's footprint. All intermediate operations stay ordered.
inline int64_t pointwiseLanes(mlir::linalg::GenericOp generic) {
  // The restricted FPGA profile has no verified vector implementations for
  // all predicates and exponent construction required by these polynomials.
  // Keep their arithmetic scalar, with explicit single-lane RVV memory access.
  for (mlir::Operation &op : generic.getBody()->without_terminator())
    if (!pointwiseMathRecipe(op).empty())
      return 1;
  int64_t widest = 8;
  for (mlir::Type type : generic.getBody()->getArgumentTypes())
    if (supportedPointwiseScalar(type))
      widest = std::max<int64_t>(widest, type.getIntOrFloatBitWidth());
  for (mlir::Operation &op : generic.getBody()->without_terminator()) {
    for (mlir::Type type : op.getOperandTypes())
      if (supportedPointwiseScalar(type))
        widest = std::max<int64_t>(widest, type.getIntOrFloatBitWidth());
    for (mlir::Type type : op.getResultTypes())
      if (supportedPointwiseScalar(type))
        widest = std::max<int64_t>(widest, type.getIntOrFloatBitWidth());
  }
  return 512 / widest;
}

} // namespace buddy::boscame

#endif
