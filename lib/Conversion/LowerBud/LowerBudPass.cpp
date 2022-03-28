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

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

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

class BudTestStrAttrLowering : public OpRewritePattern<bud::TestStrAttrOp> {
public:
  using OpRewritePattern<bud::TestStrAttrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(bud::TestStrAttrOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // Get type from the origin operation.
    Type resultType = op.getResult().getType();
    // Get the attribute.
    auto arithAttr = op.arith();
    // Get the lhs and rhs.
    Value lhs = op.lhs();
    Value rhs = op.rhs();
    Value result;
    // Lowering to different ops according to the attribute.
    if (arithAttr) {
      if (*arithAttr == "add")
        // Create addi operation.
        result = rewriter.create<arith::AddIOp>(loc, resultType, lhs, rhs);
      if (*arithAttr == "sub")
        // Create subi operation.
        result = rewriter.create<arith::SubIOp>(loc, resultType, lhs, rhs);
      rewriter.replaceOp(op, result);
    } else {
      // Default attribute is "add".
      result = rewriter.create<arith::AddIOp>(loc, resultType, lhs, rhs);
      rewriter.replaceOp(op, result);
    }
    return success();
  }
};

class BudTestArrayAttrLowering : public OpRewritePattern<bud::TestArrayAttrOp> {
public:
  using OpRewritePattern<bud::TestArrayAttrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(bud::TestArrayAttrOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // Get the attribute and the value.
    ArrayAttr coordinateAttr = op.coordinate();
    int64_t valX = coordinateAttr[0].cast<IntegerAttr>().getInt();
    int64_t valY = coordinateAttr[1].cast<IntegerAttr>().getInt();
    // Get the index attribute and constant value.
    IntegerAttr attrX = rewriter.getIntegerAttr(rewriter.getIndexType(), valX);
    IntegerAttr attrY = rewriter.getIntegerAttr(rewriter.getIndexType(), valY);
    Value idxX = rewriter.create<arith::ConstantOp>(loc, attrX);
    Value idxY = rewriter.create<arith::ConstantOp>(loc, attrY);
    SmallVector<Value, 2> memrefIdx = {idxX, idxY};
    // Get base memref.
    Value memref = op.base();
    // Create memref load operation.
    Value result = rewriter.create<memref::LoadOp>(loc, memref, memrefIdx);
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // end anonymous namespace

void populateLowerBudConversionPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
      BudTestConstantLowering,
      BudTestPrintLowering,
      BudTestStrAttrLowering,
      BudTestArrayAttrLowering>(patterns.getContext());
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
        func::FuncDialect,
        vector::VectorDialect,
        memref::MemRefDialect>();
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
      func::FuncDialect,
      vector::VectorDialect,
      memref::MemRefDialect>();
  // clang-format on
  target.addLegalOp<ModuleOp, FuncOp, func::ReturnOp>();

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
