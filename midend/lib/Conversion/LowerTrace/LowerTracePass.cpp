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
    mlir::FlatSymbolRefAttr funcSymbol;
    auto rtclockFunc = module.lookupSymbol<mlir::func::FuncOp>("rtclock");
    if (!rtclockFunc) {
      // If the function does not exist, it is declared
      auto funcType = rewriter.getFunctionType({}, rewriter.getF64Type());
      rtclockFunc = rewriter.create<mlir::func::FuncOp>(module.getLoc(),
                                                        "rtclock", funcType);
      rtclockFunc.setPrivate();
      funcSymbol = mlir::SymbolRefAttr::get(op.getContext(), "rtclock");
    } else {
      funcSymbol =
          mlir::SymbolRefAttr::get(op.getContext(), rtclockFunc.getName());
    }

    // create a rtclock op
    auto callOp = rewriter.create<mlir::func::CallOp>(
        loc, rewriter.getF64Type(), funcSymbol, ValueRange());

    rewriter.replaceOp(op, callOp.getResults());

    return success();
  }
};

class TraceEndLowering : public OpRewritePattern<trace::TimeEndOp> {
public:
  using OpRewritePattern<trace::TimeEndOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(trace::TimeEndOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // make sure rtclock exist
    auto module = op->getParentOfType<mlir::ModuleOp>();
    mlir::FlatSymbolRefAttr funcSymbol;
    auto rtclockFunc = module.lookupSymbol<mlir::func::FuncOp>("rtclock");
    if (!rtclockFunc) {
      // if rtclock not exist , than create it
      auto funcType = rewriter.getFunctionType({}, rewriter.getF64Type());
      rtclockFunc = rewriter.create<mlir::func::FuncOp>(module.getLoc(),
                                                        "rtclock", funcType);
      rtclockFunc.setPrivate();
      funcSymbol = mlir::SymbolRefAttr::get(op.getContext(), "rtclock");
    } else {
      funcSymbol =
          mlir::SymbolRefAttr::get(op.getContext(), rtclockFunc.getName());
    }

    // Create an operation that calls rtclock
    auto callOp = rewriter.create<mlir::func::CallOp>(
        loc, rewriter.getF64Type(), funcSymbol, ValueRange());

    // Replace EndOp with new call operation
    rewriter.replaceOp(op, callOp.getResults());

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
        buddy::trace::TraceDialect,
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
  populateLowerTraceConversionPatterns(patterns);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerTracePass() { PassRegistration<LowerTracePass>(); }

} // namespace buddy
} // namespace mlir
