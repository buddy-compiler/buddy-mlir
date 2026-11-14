//===- BOSCAMEZeroInit.h - Proof for deferred accumulator initialization --===//
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
// Proof that a matmul's private accumulator may defer zero-initialization
// (for example via `zero_acc`) because a dominating fill covers the full
// buffer and no alias observes deferred contents before the first product.
// Re-derived from ordinary IR; independent of plan.* attributes.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEZEROINIT_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEZEROINIT_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include <optional>

namespace buddy::boscame {

// Dominating zero fill (and optional enclosing K loop) that justifies deferred
// accumulator initialization for a private matmul C tile.
struct ZeroAccumulatorInit {
  mlir::linalg::FillOp fill;
  // Absent for a same-block product. Otherwise only the first iteration uses
  // zero_acc; every subsequent iteration loads the memory-resident result.
  mlir::scf::ForOp loop;
};

// Re-prove the optimization from standard IR, independently of plan attrs.
// The private accumulator must be covered in full by both operations. No
// alias can observe its deferred contents before the first matmul completes,
// or modify/capture them during the reduction. Post-reduction copies may
// observe any valid subset (including a padded tile's valid tail).
inline std::optional<ZeroAccumulatorInit>
getZeroAccumulatorInit(mlir::linalg::MatmulOp matmul) {
  using namespace mlir;
  if (matmul.getOutputs().size() != 1)
    return std::nullopt;
  Value storage = matmul.getOutputs()[0];
  auto allocation = storage.getDefiningOp();
  auto type = dyn_cast<MemRefType>(storage.getType());
  if (!isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(allocation) ||
      !type || !type.hasStaticShape() || type.getRank() != 2 ||
      !type.getElementType().isInteger(32) ||
      llvm::is_contained(matmul.getInputs(), storage))
    return std::nullopt;
  linalg::FillOp fill;
  for (Operation *user : storage.getUsers()) {
    auto candidate = dyn_cast<linalg::FillOp>(user);
    if (!candidate)
      continue;
    IntegerAttr zero;
    if (fill || candidate.getOutputs()[0] != storage ||
        !matchPattern(candidate.getInputs()[0], m_Constant(&zero)) ||
        !zero.getValue().isZero())
      return std::nullopt;
    fill = candidate;
  }
  if (!fill || fill->getBlock() != allocation->getBlock())
    return std::nullopt;
  scf::ForOp loop;
  Operation *scope = matmul;
  if (matmul->getBlock() != fill->getBlock()) {
    loop = dyn_cast<scf::ForOp>(matmul->getParentOp());
    IntegerAttr lower, upper, step;
    if (!loop || loop->getBlock() != fill->getBlock() ||
        !matchPattern(loop.getLowerBound(), m_Constant(&lower)) ||
        !matchPattern(loop.getUpperBound(), m_Constant(&upper)) ||
        !matchPattern(loop.getStep(), m_Constant(&step)) ||
        step.getValue().getSExtValue() <= 0 ||
        lower.getValue().getSExtValue() >= upper.getValue().getSExtValue())
      return std::nullopt;
    scope = loop;
  }
  if (!fill->isBeforeInBlock(scope))
    return std::nullopt;

  auto afterReduction = [&](Operation *op) {
    while (op && op->getBlock() != scope->getBlock())
      op = op->getParentOp();
    return op && op != scope && scope->isBeforeInBlock(op);
  };
  SmallVector<Value> work{storage};
  llvm::DenseSet<Value> seen;
  while (!work.empty()) {
    Value value = work.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    for (Operation *user : value.getUsers()) {
      if (user == fill || user == matmul) {
        // No input aliases of C, or extra access through a partial view.
        if (value != storage ||
            (user == matmul && llvm::is_contained(matmul.getInputs(), value)))
          return std::nullopt;
        continue;
      }
      if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
        if (view.getViewSource() != value || user->getNumResults() != 1 ||
            user->getNumRegions() || !isMemoryEffectFree(user))
          return std::nullopt;
        work.push_back(user->getResult(0));
        continue;
      }
      if (isa<memref::DimOp>(user))
        continue;
      if (isa<memref::DeallocOp>(user) && afterReduction(user))
        continue;
      if (auto copy = dyn_cast<memref::CopyOp>(user);
          copy && copy.getSource() == value && copy.getTarget() != value &&
          afterReduction(user))
        continue;
      return std::nullopt;
    }
  }
  return ZeroAccumulatorInit{fill, loop};
}

// Inverse lookup: the unique matmul that may consume `fill` as zero-init.
inline mlir::linalg::MatmulOp
getZeroAccumulatorConsumer(mlir::linalg::FillOp fill) {
  for (mlir::Operation *user : fill.getOutputs()[0].getUsers())
    if (auto matmul = mlir::dyn_cast<mlir::linalg::MatmulOp>(user))
      if (auto init = getZeroAccumulatorInit(matmul);
          init && init->fill == fill)
        return matmul;
  return {};
}

} // namespace buddy::boscame

#endif
