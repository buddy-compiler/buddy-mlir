//===- StaticizeMemRefLayout.cpp ------------------------------------------===//
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
// This file implements a pass that finds memref.copy operations with dynamic
// layouts, traces back to their source reinterpret_cast operations, and
// converts them to use static layouts when the shape is fully static.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {
struct ReinterpretCastFolder
    : public OpRewritePattern<memref::ReinterpretCastOp> {
  using OpRewritePattern<memref::ReinterpretCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::ReinterpretCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getType().hasStaticShape()) {
      return failure();
    }
    SmallVector<OpFoldResult> mixedOffsets(op.getMixedOffsets());
    SmallVector<OpFoldResult> mixedStrides(op.getMixedStrides());

    // No constant operands were folded, just return.
    if (failed(foldDynamicIndexList(mixedOffsets, /*onlyNonNegative=*/true)) ||
        failed(foldDynamicIndexList(mixedStrides))) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, op.getSource(), mixedOffsets[0], op.getMixedSizes(), mixedStrides);

    return success();
  }
};

class StaticizeMemRefLayoutPass
    : public PassWrapper<StaticizeMemRefLayoutPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StaticizeMemRefLayoutPass)

  StringRef getArgument() const final { return "staticize-memref-layout"; }

  StringRef getDescription() const final {
    return "Convert dynamic layouts in memref.reinterpret_cast operations used"
           "by memref.copy to static layouts when shapes are fully static.";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    bool changed = false;

    RewritePatternSet patterns(context);
    patterns.add<ReinterpretCastFolder>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     GreedyRewriteConfig(), &changed))) {
      return signalPassFailure();
    }

    if (!changed)
      markAllAnalysesPreserved();
  }
};

} // namespace

namespace mlir {
namespace buddy {

void registerStaticizeMemRefLayoutPass() {
  PassRegistration<StaticizeMemRefLayoutPass>();
}

} // namespace buddy
} // namespace mlir
