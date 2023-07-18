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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

#include "Bud/BudDialect.h"
#include "Bud/BudOps.h"
#include "Sche/ScheDialect.h"
#include "Sche/ScheOps.h"

#include <unordered_map>


using namespace mlir;
using namespace buddy;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class LaunchFuncOpScheLowering : public OpRewritePattern<sche::LaunchFuncOp>  {
public:
  using OpRewritePattern<sche::LaunchFuncOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(sche::LaunchFuncOp op, PatternRewriter &rewriter) const override {
    printf("begin\n");
    auto loc = op.getLoc();

    printf("finish\n");
    return success();

  }
};

class OnDeviceOpScheLowering : public OpRewritePattern<sche::OnDeviceOp>  {
public:
  using OpRewritePattern<sche::OnDeviceOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(sche::OnDeviceOp op, PatternRewriter &rewriter) const override {
    printf("begin\n");
    auto loc = op.getLoc();

    printf("finish\n");
    return success();

  }
};

} // end anonymous namespace

void populateLowerScheConversionPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<LowerSchePattern>(patterns.getContext());
  // clang-format on
}

//===----------------------------------------------------------------------===//
// LowerSchePass
//===----------------------------------------------------------------------===//

namespace {
class LowerSchePass : public PassWrapper<LowerSchePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerSchePass)
  LowerSchePass() = default;
  LowerSchePass(const LowerSchePass &) {}

  StringRef getArgument() const final { return "device-schedule"; }
  StringRef getDescription() const final { return "schedule on devices."; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<
        buddy::bud::BudDialect,
        func::FuncDialect,
        vector::VectorDialect,
        memref::MemRefDialect,
        LLVM::LLVMDialect,
        buddy::sche::ScheDialect>();
    // clang-format on
  }
};
} // end anonymous namespace.

void LowerSchePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  // clang-format off
  target.addLegalDialect<
      arith::ArithDialect,
      func::FuncDialect,
      vector::VectorDialect,
      memref::MemRefDialect,
      LLVM::LLVMDialect,
      buddy::sche::ScheDialect>();
  // clang-format on
  target.addLegalOp<ModuleOp, func::ReturnOp>();

  target.addIllegalOp<func::FuncOp>();

  RewritePatternSet patterns(context);
  populateLowerScheConversionPatterns(patterns);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerSchePass() { PassRegistration<LowerSchePass>(); }
} // namespace buddy
} // namespace mlir
