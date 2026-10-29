//===- VectorExpApproximation.cpp - Prefill elementwise optimization
//-------===//
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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
class PrefillElementwiseFusionPass
    : public PassWrapper<PrefillElementwiseFusionPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrefillElementwiseFusionPass)
  StringRef getArgument() const final { return "fuse-prefill-elementwise"; }
  StringRef getDescription() const final {
    return "Fuse parallel generic kernels while preserving named contractions";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    linalg::populateElementwiseOpsFusionPatterns(
        patterns, [](OpOperand *operand) {
          auto producer = operand->get().getDefiningOp<linalg::GenericOp>();
          auto consumer = dyn_cast<linalg::GenericOp>(operand->getOwner());
          if (!producer || !consumer || !producer->hasOneUse())
            return false;
          auto parallel = [](linalg::GenericOp op) {
            return llvm::none_of(op.getIteratorTypesArray(),
                                 linalg::isReductionIterator);
          };
          return parallel(producer) && parallel(consumer);
        });
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

class VectorExpApproximationPass
    : public PassWrapper<VectorExpApproximationPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorExpApproximationPass)
  StringRef getArgument() const final { return "approximate-vector-exp"; }
  StringRef getDescription() const final {
    return "Use MLIR's FP32 polynomial exp approximation for vector operands";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, math::MathDialect,
                    vector::VectorDialect>();
  }
  void runOnOperation() override {
    SmallVector<Operation *> selected;
    getOperation().walk([&](math::ExpOp op) {
      auto type = dyn_cast<VectorType>(op.getType());
      if (type && type.getElementType().isF32())
        selected.push_back(op);
    });
    if (selected.empty())
      return;
    RewritePatternSet patterns(&getContext());
    populateMathPolynomialApproximationPatterns(patterns, [](StringRef name) {
      return name == math::ExpOp::getOperationName();
    });
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
    if (failed(applyOpPatternsGreedily(selected, std::move(patterns), config)))
      signalPassFailure();
  }
};
} // namespace

namespace mlir::buddy {
void registerPrefillElementwiseFusionPass() {
  PassRegistration<PrefillElementwiseFusionPass>();
}
void registerVectorExpApproximationPass() {
  PassRegistration<VectorExpApproximationPass>();
}
} // namespace mlir::buddy
