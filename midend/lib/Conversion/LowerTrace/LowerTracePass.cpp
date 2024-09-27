//====- LowerTracePass.cpp - Trace Dialect Lowering Pass  -----------------===//
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
// This file defines trace dialect lowering pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/CommandLine.h"
#include <vector>

#include "Trace/TraceDialect.h"
#include "Trace/TraceOps.h"

using namespace mlir;
using namespace buddy;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class TraceStartLowering : public OpRewritePattern<trace::TimeStartOp> {
public:
  using OpRewritePattern<trace::TimeStartOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(trace::TimeStartOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Ensure that the symbol for the rtclock function exists
    auto module = op->getParentOfType<mlir::ModuleOp>();

    auto timingStartFunc =
        module.lookupSymbol<mlir::func::FuncOp>("timingStart");

    // If the function does not exist
    if (!timingStartFunc) {
      op.emitError("Don't have timingStart declaration.");
    }

    mlir::FlatSymbolRefAttr funcSymbol =
        mlir::SymbolRefAttr::get(op.getContext(), timingStartFunc.getName());

    // create a call op
    auto timingStart =
        rewriter.create<mlir::func::CallOp>(loc, funcSymbol, ValueRange{});

    rewriter.replaceOp(op, timingStart.getResults());

    return success();
  }
};

class TraceEndLowering : public OpRewritePattern<trace::TimeEndOp> {
public:
  using OpRewritePattern<trace::TimeEndOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(trace::TimeEndOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Ensure that the symbol for the rtclock function exists
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto timingEndFunc = module.lookupSymbol<mlir::func::FuncOp>("timingEnd");

    // If the function does not exist
    if (!timingEndFunc) {
      op.emitError("Don't have timingEnd declaration.");
    }

    // Now retrieve the symbol reference attribute
    mlir::FlatSymbolRefAttr funcSymbol =
        mlir::SymbolRefAttr::get(op.getContext(), timingEndFunc.getName());

    // create a rtclock op
    auto timingEnd =
        rewriter.create<mlir::func::CallOp>(loc, funcSymbol, ValueRange{});

    rewriter.replaceOp(op, timingEnd.getResults());

    return success();
  }
};

} // end anonymous namespace

void populateLowerTraceConversionPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
      TraceStartLowering,
      TraceEndLowering>(patterns.getContext());
  // clang-format on
}

//===----------------------------------------------------------------------===//
// LowerTracePass
//===----------------------------------------------------------------------===//

namespace {
class LowerTracePass
    : public PassWrapper<LowerTracePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTracePass)
  LowerTracePass() = default;
  LowerTracePass(const LowerTracePass &) {}

  StringRef getArgument() const final { return "lower-trace"; }
  StringRef getDescription() const final { return "Lower Trace Dialect."; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<
        trace::TraceDialect,
        func::FuncDialect,
        vector::VectorDialect,
        memref::MemRefDialect,
        LLVM::LLVMDialect>();
    // clang-format on
  }
};

} // end anonymous namespace.

void LowerTracePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  OpBuilder builder(context);

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

  builder.setInsertionPointToStart(module.getBody());

  auto FuncType = builder.getFunctionType({}, {});

  // declare func.func private @timingStart()
  auto timingStartFunc = builder.create<mlir::func::FuncOp>(
      module.getLoc(), "timingStart", FuncType);
  timingStartFunc.setPrivate();

  // declare func.func private @timingEnd()
  auto timingEndFunc = builder.create<mlir::func::FuncOp>(
      module.getLoc(), "timingEnd", FuncType);
  timingEndFunc.setPrivate();

  RewritePatternSet patterns(context);
  populateLowerTraceConversionPatterns(patterns);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerTracePass() { PassRegistration<LowerTracePass>(); }

} // namespace buddy
} // namespace mlir
