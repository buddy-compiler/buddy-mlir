//===- NestParallelLoops.cpp ----------------------------------------------===//
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
// Split scf.parallel loops with >3 dimensions into nested loops so that
// gpu-map-parallel-loops can map outer dims to blocks and inner dims to
// threads. Without this, dims beyond the 3rd become sequential scf.for.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

class NestParallelLoopsPass
    : public PassWrapper<NestParallelLoopsPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NestParallelLoopsPass)
  StringRef getArgument() const final { return "nest-parallel-loops-for-gpu"; }
  StringRef getDescription() const final {
    return "Split >3D scf.parallel into nested 3D(block)+remaining(thread).";
  }
  NestParallelLoopsPass() = default;
  NestParallelLoopsPass(const NestParallelLoopsPass &) {}

  void runOnOperation() override {
    SmallVector<scf::ParallelOp> targets;
    getOperation()->walk([&](scf::ParallelOp op) {
      if (op.getNumLoops() > 3)
        targets.push_back(op);
    });

    for (auto ploop : targets) {
      unsigned numLoops = ploop.getNumLoops();
      // Split into first 3 dims (block) and remaining dims (thread).
      unsigned outerDims = 3;
      unsigned innerDims = numLoops - outerDims;

      auto lbs = ploop.getLowerBound();
      auto ubs = ploop.getUpperBound();
      auto steps = ploop.getStep();

      OpBuilder builder(ploop);
      auto loc = ploop.getLoc();

      // Create outer scf.parallel with first 3 dims.
      SmallVector<Value> outerLBs(lbs.begin(), lbs.begin() + outerDims);
      SmallVector<Value> outerUBs(ubs.begin(), ubs.begin() + outerDims);
      SmallVector<Value> outerSteps(steps.begin(), steps.begin() + outerDims);

      auto outerLoop =
          builder.create<scf::ParallelOp>(loc, outerLBs, outerUBs, outerSteps);
      builder.setInsertionPointToStart(outerLoop.getBody());

      // Create inner scf.parallel with remaining dims.
      SmallVector<Value> innerLBs(lbs.begin() + outerDims, lbs.end());
      SmallVector<Value> innerUBs(ubs.begin() + outerDims, ubs.end());
      SmallVector<Value> innerSteps(steps.begin() + outerDims, steps.end());

      auto innerLoop =
          builder.create<scf::ParallelOp>(loc, innerLBs, innerUBs, innerSteps);

      // Move the original loop body into the inner loop.
      // The original body uses induction vars [0..numLoops-1].
      // Map them: [0..2] → outer IVs, [3..N-1] → inner IVs.
      Block *origBody = ploop.getBody();
      Block *innerBody = innerLoop.getBody();

      // Build mapping from old IVs to new IVs.
      IRMapping mapping;
      for (unsigned i = 0; i < outerDims; ++i)
        mapping.map(origBody->getArgument(i), outerLoop.getInductionVars()[i]);
      for (unsigned i = 0; i < innerDims; ++i)
        mapping.map(origBody->getArgument(outerDims + i),
                    innerLoop.getInductionVars()[i]);

      // Clone all ops from the original body (except the terminator) into
      // the inner loop body (before its terminator).
      builder.setInsertionPoint(innerBody->getTerminator());
      for (auto &op : llvm::make_early_inc_range(*origBody)) {
        if (isa<scf::ReduceOp>(&op))
          continue;
        builder.clone(op, mapping);
      }

      ploop->erase();
    }
  }
};

} // namespace

namespace mlir {
namespace buddy {
void registerNestParallelLoopsPass() {
  PassRegistration<NestParallelLoopsPass>();
}
} // namespace buddy
} // namespace mlir
