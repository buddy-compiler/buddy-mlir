//===- SimplifyTosaMatmulScalar.cpp - Replace scalar-like matmul ----------===//
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
// This pass canonicalizes a special case of tosa.matmul where K == 1 and J == 1
// (i.e., multiplying by a batch-aligned scalar), into an elementwise tosa.mul
// with broadcasting. This is cheaper and more optimizable downstream.
//
// Preconditions to rewrite:
// - lhs:  [..., I, 1]
// - rhs:  [..., 1, 1]
// - out:  [..., I, 1]
// - same number of batch dims, and each batch dim equal (conservative)
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

// Match tosa.matmul with K == 1 and J == 1; replace with tosa.mul
struct ReplaceScalarLikeMatmul : public OpRewritePattern<tosa::MatMulOp> {
  using OpRewritePattern<tosa::MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsTy = dyn_cast<RankedTensorType>(op.getA().getType());
    auto rhsTy = dyn_cast<RankedTensorType>(op.getB().getType());
    auto outTy = dyn_cast<RankedTensorType>(op.getType());
    if (!lhsTy || !rhsTy || !outTy)
      return failure();

    if (lhsTy.getRank() < 2 || rhsTy.getRank() < 2 || outTy.getRank() < 2)
      return failure();

    // Require same number of batch dims.
    int64_t lhsRank = lhsTy.getRank();
    int64_t rhsRank = rhsTy.getRank();
    int64_t outRank = outTy.getRank();
    if (lhsRank != rhsRank || lhsRank != outRank)
      return failure();
    int64_t batchRank = lhsRank - 2;

    // Check last-2 dims: lhs [..., I, K], rhs [..., K, J], out [..., I, J]
    ArrayRef<int64_t> lhsShape = lhsTy.getShape();
    ArrayRef<int64_t> rhsShape = rhsTy.getShape();
    ArrayRef<int64_t> outShape = outTy.getShape();

    int64_t I = lhsShape[lhsRank - 2];
    int64_t K_lhs = lhsShape[lhsRank - 1];
    int64_t K_rhs = rhsShape[rhsRank - 2];
    int64_t J = rhsShape[rhsRank - 1];
    int64_t J_out = outShape[outRank - 1];
    int64_t I_out = outShape[outRank - 2];

    // Static shape only for now to be conservative.
    if (ShapedType::isDynamic(I) || ShapedType::isDynamic(K_lhs) ||
        ShapedType::isDynamic(K_rhs) || ShapedType::isDynamic(J) ||
        ShapedType::isDynamic(I_out) || ShapedType::isDynamic(J_out))
      return failure();

    if (K_lhs != 1 || K_rhs != 1 || J != 1)
      return failure();
    if (I_out != I || J_out != 1)
      return failure();

    // Batch dims must match exactly (no broadcast in tosa.matmul).
    for (int64_t i = 0; i < batchRank; ++i) {
      int64_t a = lhsShape[i], b = rhsShape[i], c = outShape[i];
      if (ShapedType::isDynamic(a) || ShapedType::isDynamic(b) ||
          ShapedType::isDynamic(c))
        return failure();
      if (!(a == b && b == c))
        return failure();
    }

    // Replace matmul with elementwise mul; types are already aligned so
    // broadcasting of rhs [...,1,1] to lhs [...,I,1] is valid.
    rewriter.replaceOpWithNewOp<tosa::MulOp>(op, outTy, op.getA(), op.getB(),
                                             Value());
    return success();
  }
};

class SimplifyTosaMatmulScalarPass
    : public PassWrapper<SimplifyTosaMatmulScalarPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimplifyTosaMatmulScalarPass)
  StringRef getArgument() const final { return "simplify-tosa-matmul-scalar"; }
  StringRef getDescription() const final {
    return "Replace scalar-like tosa.matmul (K=1,J=1) with tosa.mul.";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = func.getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ReplaceScalarLikeMatmul>(ctx);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::AnyOp; // allow creating mul
    (void)applyPatternsGreedily(func, std::move(patterns), config);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tosa::TosaDialect>();
  }
};
} // namespace

namespace mlir {
namespace buddy {
void registerSimplifyTosaMatmulScalarPass() {
  PassRegistration<SimplifyTosaMatmulScalarPass>();
}
} // namespace buddy
} // namespace mlir
