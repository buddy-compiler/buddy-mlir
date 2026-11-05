//===- BOSCAMEHotAccumulator.h - Closed hot accumulator reuse proof -------===//
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
// Closed proof that a private accumulator stays "hot" across loop trips and
// may reuse SPM / register placement. Facts are reconstructed from ordinary
// IR (bytes, trips, boundary traffic), never from placement attributes alone.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEHOTACCUMULATOR_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEHOTACCUMULATOR_H

#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEPointwise.h"
#include "Dialect/BOSCAME/DmaAlignment.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

namespace buddy::boscame {

// Logical traffic facts for a repeatedly reused private C tile.
struct HotAccumulator {
  int64_t bytes;         // Static byte size of the accumulator tile.
  int64_t trips;         // Constant trip count of the enclosing K loop.
  int64_t loopBytes;     // Conservative bytes charged across loop iterations.
  int64_t boundaryBytes; // Pre/post consumer traffic outside the K loop.
};

// Reconstruct from ordinary IR, never from placement/schedule attributes.
// A repeated private C tile is useful only if every alias has a legal
// synchronous AME/RVV/transfer consumer. In particular scalar SPM accesses,
// captures, parallel scopes and observations during the K loop are excluded.
inline std::optional<HotAccumulator>
analyzeHotAccumulator(mlir::Value storage) {
  using namespace mlir;
  auto type = dyn_cast<MemRefType>(storage.getType());
  Operation *allocation = storage.getDefiningOp();
  auto function =
      allocation ? allocation->getParentOfType<func::FuncOp>() : func::FuncOp();
  SmallVector<int64_t> strides;
  int64_t offset;
  if (!isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(allocation) ||
      !function || !type || !type.hasStaticShape() || type.getRank() != 2 ||
      !type.getElementType().isInteger(32) || type.getMemorySpace() ||
      type.getDimSize(0) <= 0 || type.getDimSize(0) > 64 ||
      type.getDimSize(1) <= 0 || type.getDimSize(1) > 64 ||
      type.getDimSize(1) % 16 ||
      failed(type.getStridesAndOffset(strides, offset)) || offset != 0 ||
      strides[1] != 1 || strides[0] != type.getDimSize(1) ||
      !SymbolTable::symbolKnownUseEmpty(function,
                                        function->getParentOfType<ModuleOp>()))
    return std::nullopt;
  bool calls = false;
  function.walk([&](Operation *op) {
    calls |= isa<func::CallOp, func::CallIndirectOp>(op);
  });
  if (calls)
    return std::nullopt;
  linalg::MatmulOp matmul;
  for (Operation *user : storage.getUsers())
    if (auto candidate = dyn_cast<linalg::MatmulOp>(user)) {
      if (matmul || candidate.getOutputs().size() != 1 ||
          candidate.getOutputs()[0] != storage ||
          llvm::is_contained(candidate.getInputs(), storage) ||
          candidate.getCast() != linalg::TypeFn::cast_signed)
        return std::nullopt;
      matmul = candidate;
    }
  auto loop =
      matmul ? dyn_cast<scf::ForOp>(matmul->getParentOp()) : scf::ForOp();
  IntegerAttr lower, upper, step;
  if (!loop || allocation->getBlock() != loop->getBlock() ||
      !allocation->isBeforeInBlock(loop) ||
      !matchPattern(loop.getLowerBound(), m_Constant(&lower)) ||
      !matchPattern(loop.getUpperBound(), m_Constant(&upper)) ||
      !matchPattern(loop.getStep(), m_Constant(&step)))
    return std::nullopt;
  int64_t distance, stride = step.getValue().getSExtValue();
  if (stride <= 0 ||
      llvm::SubOverflow(upper.getValue().getSExtValue(),
                        lower.getValue().getSExtValue(), distance) ||
      distance <= 0)
    return std::nullopt;
  int64_t trips = llvm::divideCeil(uint64_t(distance), uint64_t(stride));
  if (trips <= 1)
    return std::nullopt;
  HotAccumulator result{type.getNumElements() * 4, trips, 0, 0};
  if (llvm::MulOverflow(result.bytes, trips, result.loopBytes) ||
      llvm::MulOverflow(result.loopBytes, int64_t(2), result.loopBytes))
    return std::nullopt;
  bool initialized = false, observed = false;
  SmallVector<Value> work{storage};
  llvm::DenseSet<Value> seen;
  llvm::DenseSet<Operation *> consumers;
  while (!work.empty()) {
    Value value = work.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    auto viewType = dyn_cast<MemRefType>(value.getType());
    auto origin = spmdma::getStaticPointerOrigin(value);
    auto descriptorOffset = spmdma::getStaticDescriptorOffset(value);
    SmallVector<int64_t> aliasStrides;
    int64_t ignored, begin, end;
    if (!viewType || !viewType.hasStaticShape() ||
        !viewType.getElementType().isInteger(32) || !origin ||
        origin->root != storage || !descriptorOffset ||
        failed(viewType.getStridesAndOffset(aliasStrides, ignored)) ||
        llvm::MulOverflow(*descriptorOffset, int64_t(4), begin) ||
        llvm::AddOverflow(begin, origin->byteShift, begin) || begin < 0)
      return std::nullopt;
    end = begin;
    for (auto [dim, s] : llvm::zip(viewType.getShape(), aliasStrides)) {
      int64_t delta;
      if (dim <= 0 || s <= 0 || llvm::MulOverflow(dim - 1, s, delta) ||
          llvm::MulOverflow(delta, int64_t(4), delta) ||
          llvm::AddOverflow(end, delta, end))
        return std::nullopt;
    }
    if (llvm::AddOverflow(end, int64_t(4), end) || end > result.bytes)
      return std::nullopt;
    for (Operation *user : value.getUsers()) {
      for (Operation *parent = user->getParentOp(); parent != function;
           parent = parent->getParentOp())
        if (!parent || !isa<scf::ForOp, scf::IfOp>(parent))
          return std::nullopt;
      if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
        if (view.getViewSource() != value || user->getNumResults() != 1 ||
            user->getNumRegions() || !isMemoryEffectFree(user))
          return std::nullopt;
        work.push_back(user->getResult(0));
        continue;
      }
      if (isa<memref::DimOp, memref::DeallocOp>(user))
        continue;
      if (user == matmul) {
        if (value != storage)
          return std::nullopt;
        continue;
      }
      if (loop->isAncestor(user))
        return std::nullopt;
      if (!consumers.insert(user).second)
        continue;
      Operation *scope = user;
      while (scope && scope->getBlock() != loop->getBlock())
        scope = scope->getParentOp();
      if (!scope)
        return std::nullopt;
      bool before = scope->isBeforeInBlock(loop);
      if (auto fill = dyn_cast<linalg::FillOp>(user)) {
        if (!before || fill.getOutputs()[0] != storage)
          return std::nullopt;
        initialized = true;
      } else if (auto copy = dyn_cast<memref::CopyOp>(user)) {
        if (copy.getSource() == copy.getTarget() ||
            (before && copy.getTarget() != storage) ||
            (!before && copy.getSource() != value))
          return std::nullopt;
        initialized |= before;
        observed |= !before;
      } else if (auto generic = dyn_cast<linalg::GenericOp>(user)) {
        if (before || viewType.getRank() != 1 || generic.getNumLoops() != 1 ||
            generic.getNumParallelLoops() != 1 ||
            generic.getNumDpsInits() != 1 ||
            viewType.getDimSize(0) % pointwiseLanes(generic))
          return std::nullopt;
        for (auto map : generic.getIndexingMapsArray())
          if (!map.isIdentity() && map.getNumResults() != 0)
            return std::nullopt;
        for (Operation &payload : generic.getBody()->without_terminator())
          if (!supportedPointwiseOperation(payload))
            return std::nullopt;
        observed = true;
      } else {
        return std::nullopt;
      }
      // Charge every pre/post consumer once, including conservative reads of
      // a generic's destination. This is logical traffic, not bus counters.
      int64_t traffic = viewType.getNumElements() * 4;
      if (isa<linalg::GenericOp>(user) &&
          llvm::MulOverflow(traffic, int64_t(3), traffic))
        return std::nullopt;
      if (llvm::AddOverflow(result.boundaryBytes, traffic,
                            result.boundaryBytes))
        return std::nullopt;
    }
  }
  return initialized && observed ? std::optional(result) : std::nullopt;
}

} // namespace buddy::boscame
#endif
