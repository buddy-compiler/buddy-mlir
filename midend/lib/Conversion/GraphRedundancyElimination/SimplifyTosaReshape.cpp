//===- SimplifyTosaReshape.cpp - TOSA reshape simplification --------------===//
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
// This file defines SimplifyTosaReshapePass, which eliminates consecutive
// redundant reshape operations and performs minimal.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

// Collapse reshape chains by wiring the outer reshape directly to the
// innermost non-reshape input, i.e. reshape(reshape(...reshape(x))) ->
// reshape(x).
struct CollapseReshapeChain : public OpRewritePattern<tosa::ReshapeOp> {
  using OpRewritePattern<tosa::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    // Chase through successive reshape defs to find the base input.
    Value base = op.getInput1();
    bool changed = false;
    while (auto inner = base.getDefiningOp<tosa::ReshapeOp>()) {
      base = inner.getInput1();
      changed = true;
    }
    if (!changed)
      return failure();

    rewriter.modifyOpInPlace(op, [&]() { op.getInput1Mutable().assign(base); });
    return success();
  }
};

// Remove identity reshape when safe: if input/output types are equal and
// the input has <2 dynamic dims (same condition as upstream fold).
struct RemoveIdentityReshape : public OpRewritePattern<tosa::ReshapeOp> {
  using OpRewritePattern<tosa::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto inTy = dyn_cast<RankedTensorType>(op.getInput1().getType());
    auto outTy = dyn_cast<RankedTensorType>(op.getType());
    if (!inTy || !outTy)
      return failure();
    if (inTy != outTy)
      return failure();
    if (inTy.getNumDynamicDims() >= 2)
      return failure();

    rewriter.replaceOp(op, op.getInput1());
    return success();
  }
};

// Erase reshape ops that became dead after rewiring or folding.
struct EraseUnusedReshape : public OpRewritePattern<tosa::ReshapeOp> {
  using OpRewritePattern<tosa::ReshapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->use_empty())
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

class SimplifyTosaReshapePass
    : public PassWrapper<SimplifyTosaReshapePass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimplifyTosaReshapePass)
  StringRef getArgument() const final { return "simplify-tosa-reshape"; }
  StringRef getDescription() const final {
    return "Minimally simplify TOSA reshape chains and identity reshapes.";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = func.getContext();
    RewritePatternSet patterns(ctx);
    patterns
        .add<CollapseReshapeChain, RemoveIdentityReshape, EraseUnusedReshape>(
            ctx);

    // Collect a stable snapshot of reshape ops and only rewrite those.
    SmallVector<Operation *> reshapeOps;
    func.walk(
        [&](tosa::ReshapeOp rop) { reshapeOps.push_back(rop.getOperation()); });

    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    config.enableFolding(false);        // avoid materializing constants
    config.enableConstantCSE(false);    // avoid constant CSE reordering
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);

    bool changed = false;
    (void)applyOpPatternsGreedily(reshapeOps, std::move(patterns), config,
                                  &changed);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tosa::TosaDialect>();
  }
};
} // namespace

namespace mlir {
namespace buddy {
void registerSimplifyTosaReshapePass() {
  PassRegistration<SimplifyTosaReshapePass>();
}
} // namespace buddy
} // namespace mlir
