//====- LowerVectorExpPass.cpp - Vector Experiment Dialect Lowering Pass  -===//
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
// This file defines vector experiment dialect lowering pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

#include "VectorExp/VectorExpDialect.h"
#include "VectorExp/VectorExpOps.h"

using namespace mlir;
using namespace buddy;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
class VectorExpPredicationLowering
    : public OpRewritePattern<vector_exp::PredicationOp> {
public:
  using OpRewritePattern<vector_exp::PredicationOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector_exp::PredicationOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    mlir::Region &configRegion = op.getRegion();
    mlir::Block &configBlock = configRegion.front();
    for (mlir::Operation &innerOp : configBlock.getOperations()) {
      if (isa<arith::AddFOp>(innerOp)) {
        Type resultType = cast<arith::AddFOp>(innerOp).getResult().getType();
        Value result = rewriter.create<LLVM::VPFAddOp>(
            loc, resultType, cast<arith::AddFOp>(innerOp).getLhs(),
            cast<arith::AddFOp>(innerOp).getRhs(), op.getMask(), op.getVl());
        rewriter.replaceOp(op, result);
      }
    }
    return success();
  }
};
} // end anonymous namespace

void populateLowerVectorExpConversionPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<VectorExpPredicationLowering>(patterns.getContext());
  // clang-format on
}

//===----------------------------------------------------------------------===//
// LowerVectorExpPass
//===----------------------------------------------------------------------===//

namespace {
class LowerVectorExpPass
    : public PassWrapper<LowerVectorExpPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerVectorExpPass)
  LowerVectorExpPass() = default;
  LowerVectorExpPass(const LowerVectorExpPass &) {}

  StringRef getArgument() const final { return "lower-vector-exp"; }
  StringRef getDescription() const final {
    return "Lower Vector Experiment Dialect.";
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<
        buddy::vector_exp::VectorExpDialect,
        func::FuncDialect,
        vector::VectorDialect,
        memref::MemRefDialect,
        LLVM::LLVMDialect>();
    // clang-format on
  }
};
} // end anonymous namespace.

void LowerVectorExpPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  // clang-format off
  target.addLegalDialect<
      arith::ArithDialect,
      func::FuncDialect,
      vector::VectorDialect,
      memref::MemRefDialect,
      LLVM::LLVMDialect>();
  // clang-format on
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();

  RewritePatternSet patterns(context);
  populateLowerVectorExpConversionPatterns(patterns);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerVectorExpPass() { PassRegistration<LowerVectorExpPass>(); }
} // namespace buddy
} // namespace mlir
