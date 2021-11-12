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
#include "mlir/Dialect/Vector/VectorOps.h"
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

class BudTestPrintLowering : public OpRewritePattern<bud::TestPrintOp> {
public:
  using OpRewritePattern<bud::TestPrintOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(bud::TestPrintOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // Get type from the origin operation.
    Type resultType = op.getResult().getType();
    // Create constant operation.
    Attribute zeroAttr = rewriter.getZeroAttr(resultType);
    Value c0 = rewriter.create<arith::ConstantOp>(loc, resultType, zeroAttr);
    // Create print operation for the scalar value.
    rewriter.create<vector::PrintOp>(loc, c0);
    VectorType vectorTy4 =
        VectorType::get({4 /*number of elements in the vector*/}, resultType);
    // Broadcast element of the kernel.
    Value broadcastVector =
        rewriter.create<vector::BroadcastOp>(loc, vectorTy4, c0);
    // Create print operation for the vector value.
    rewriter.create<vector::PrintOp>(loc, broadcastVector);

    rewriter.eraseOp(op);
    return success();
  }
};
} // end anonymous namespace

void populateLowerBudConversionPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
      BudTestConstantLowering,
      BudTestPrintLowering>(patterns.getContext());
  // clang-format on
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
    // clang-format off
    registry.insert<
        buddy::bud::BudDialect,
        StandardOpsDialect,
        vector::VectorDialect>();
    // clang-format on
  }
};
} // end anonymous namespace.

void LowerBudPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  // clang-format off
  target.addLegalDialect<
      arith::ArithmeticDialect,
      StandardOpsDialect,
      vector::VectorDialect>();
  // clang-format on
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
