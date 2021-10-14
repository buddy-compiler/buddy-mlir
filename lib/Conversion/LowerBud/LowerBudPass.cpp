//====- LowerBudPass.cpp - Bud Dialect Lowering Pass  ---------------------===//
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
// This file defines bud dialect lowering pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Bufferize.h"

#include "Bud/BudDialect.h"
#include "Bud/BudOps.h"

using namespace mlir;
using namespace buddy;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
class BudTestConstantLowering : public OpRewritePattern<bud::TestConstantOp> {
public:
  using OpRewritePattern<bud::TestConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(bud::TestConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // Get type from the origin operation.
    Type resultType = op.getResult().getType();
    // Create constant operation.
    Attribute zeroAttr = rewriter.getZeroAttr(resultType);
    Value c0 = rewriter.create<arith::ConstantOp>(loc, resultType, zeroAttr);

    rewriter.replaceOp(op, c0);
    return success();
  }
};
} // end anonymous namespace

void populateLowerBudConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<BudTestConstantLowering>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// LowerBudPass
//===----------------------------------------------------------------------===//

namespace {
class LowerBudPass : public PassWrapper<LowerBudPass, OperationPass<ModuleOp>> {
public:
  LowerBudPass() = default;
  LowerBudPass(const LowerBudPass &) {}

  StringRef getArgument() const final { return "lower-bud"; }
  StringRef getDescription() const final { return "Lower Bud Dialect."; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<buddy::bud::BudDialect, StandardOpsDialect>();
  }
};
} // end anonymous namespace.

void LowerBudPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithmeticDialect, StandardOpsDialect>();
  target.addLegalOp<ModuleOp, FuncOp, ReturnOp>();

  RewritePatternSet patterns(context);
  populateLowerBudConversionPatterns(patterns);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerBudPass() { PassRegistration<LowerBudPass>(); }
} // namespace buddy
} // namespace mlir
