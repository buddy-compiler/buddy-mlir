//===- SPMBroadcasts.cpp - Forward private fills to registers -------------===//
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

#include "Conversion/BOSCAMEHelpers/BOSCAMEHelpers.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/SpmDma.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace buddy::boscame::spmdma;

namespace buddy::boscame::lowering {
namespace {

static Value spmRoot(Value value) {
  while (Operation *definition = value.getDefiningOp()) {
    if (isa<SpmViewOp>(definition))
      return value;
    if (!isa<memref::SubViewOp, memref::ReinterpretCastOp, memref::CastOp,
             memref::CollapseShapeOp, memref::ExpandShapeOp>(definition))
      return {};
    value = definition->getOperand(0);
  }
  return {};
}

// SPM views describe fixed physical addresses, so distinct SSA roots can
// alias. Require a complete set of known roots and disjoint physical ranges.
static bool hasPrivatePhysicalRange(SpmViewOp view) {
  auto function = view->getParentOfType<func::FuncOp>();
  auto bytes =
      getContiguousZeroOffsetByteSize(cast<MemRefType>(view.getType()));
  if (!function || !bytes)
    return false;
  bool disjoint = true;
  function.walk([&](Operation *operation) {
    for (Value value : operation->getOperands())
      if (getSpmBank(value.getType()) && !spmRoot(value))
        disjoint = false;
    for (Value value : operation->getResults())
      if (getSpmBank(value.getType()) && !spmRoot(value))
        disjoint = false;
    auto other = dyn_cast<SpmViewOp>(operation);
    if (!other || other == view || other.getBank() != view.getBank())
      return;
    auto otherBytes =
        getContiguousZeroOffsetByteSize(cast<MemRefType>(other.getType()));
    if (!otherBytes || (view.getOffset() < other.getOffset() + *otherBytes &&
                        other.getOffset() < view.getOffset() + *bytes))
      disjoint = false;
  });
  return disjoint;
}

static bool isReadOnlyGeneric(linalg::GenericOp generic, Value buffer) {
  if (!generic.hasPureBufferSemantics() || generic.getNumDpsInits() != 1 ||
      generic->hasAttr("bosc_ame.zero_tail_select_prefix") ||
      !llvm::all_of(generic.getIteratorTypesArray(),
                    [](utils::IteratorType t) {
                      return t == utils::IteratorType::parallel;
                    }) ||
      !llvm::all_of(generic.getBody()->without_terminator(), [](Operation &op) {
        return isMemoryEffectFree(&op) &&
               !isa<arith::SelectOp, linalg::IndexOp>(op);
      }))
    return false;
  auto type = cast<MemRefType>(buffer.getType());
  auto outputType = dyn_cast<MemRefType>(generic.getOutputs()[0].getType());
  if (!outputType || outputType.getShape() != type.getShape() ||
      getSpmElementByteWidth(outputType.getElementType()) !=
          getSpmElementByteWidth(type.getElementType()))
    return false;
  auto maps = generic.getIndexingMapsArray();
  for (auto [index, operand] : llvm::enumerate(generic.getOperands())) {
    if (operand != buffer)
      continue;
    if (index >= static_cast<size_t>(generic.getNumDpsInputs()) ||
        !maps[index].isIdentity())
      return false;
  }
  return true;
}

// Unrecognized calls or effects may access fixed SPM addresses implicitly.
// Known memory operations have explicit operands whose roots were checked.
static bool hasKnownEffects(Operation &operation) {
  if (isMemoryEffectFree(&operation) ||
      operation.hasTrait<OpTrait::HasRecursiveMemoryEffects>())
    return true;
  if (auto linalg = dyn_cast<linalg::LinalgOp>(operation))
    return linalg.hasPureBufferSemantics() &&
           llvm::all_of(
               operation.getRegion(0).front().without_terminator(),
               [](Operation &inner) { return isMemoryEffectFree(&inner); });
  return isa<memref::LoadOp, memref::StoreOp, memref::CopyOp, memref::AllocOp,
             memref::AllocaOp, memref::DeallocOp, vector::LoadOp,
             vector::StoreOp, DmaStartOp, DmaWaitOp>(operation);
}

struct BroadcastPlan {
  SpmViewOp view;
  SmallVector<linalg::FillOp> fills;
  SmallVector<std::pair<linalg::GenericOp, linalg::FillOp>> readers;
};

static std::optional<BroadcastPlan> planBroadcast(SpmViewOp view) {
  Value buffer = view.getResult();
  auto type = cast<MemRefType>(buffer.getType());
  unsigned bytes = getSpmElementByteWidth(type.getElementType());
  if (!bytes || bytes > kDmaAlignmentBytes || !type.hasStaticShape() ||
      type.getRank() < 1 || type.getRank() > 2 || type.getShape().back() <= 0 ||
      type.getShape().back() % (kDmaAlignmentBytes / bytes) ||
      !hasPrivatePhysicalRange(view))
    return std::nullopt;

  // An implicit access through a fixed address can observe the buffer even
  // after its last explicit reader. Require known effects throughout the
  // function, including nested regions and successor blocks.
  auto function = view->getParentOfType<func::FuncOp>();
  if (function
          .walk([&](Operation *operation) {
            return operation == function.getOperation() ||
                           hasKnownEffects(*operation)
                       ? WalkResult::advance()
                       : WalkResult::interrupt();
          })
          .wasInterrupted())
    return std::nullopt;

  llvm::SmallPtrSet<Operation *, 8> users;
  for (Operation *user : buffer.getUsers()) {
    if (user->getBlock() != view->getBlock())
      return std::nullopt;
    if (auto fill = dyn_cast<linalg::FillOp>(user)) {
      if (!fill.hasPureBufferSemantics() || fill.getOutputs().size() != 1 ||
          fill.getOutputs()[0] != buffer || fill.getInputs().size() != 1 ||
          fill.getInputs()[0].getType() != type.getElementType())
        return std::nullopt;
    } else if (auto generic = dyn_cast<linalg::GenericOp>(user)) {
      if (!isReadOnlyGeneric(generic, buffer))
        return std::nullopt;
    } else {
      // Aliases, escaping buffers, DMA uses and non-fill writers stay in SPM.
      return std::nullopt;
    }
    users.insert(user);
  }

  BroadcastPlan plan{view, {}, {}};
  linalg::FillOp latestFill;
  for (Operation &operation : *view->getBlock()) {
    if (!users.contains(&operation))
      continue;
    if (auto fill = dyn_cast<linalg::FillOp>(operation)) {
      latestFill = fill;
      plan.fills.push_back(fill);
    } else {
      if (!latestFill)
        return std::nullopt;
      plan.readers.emplace_back(cast<linalg::GenericOp>(operation), latestFill);
    }
  }
  if (plan.fills.empty() || plan.readers.empty())
    return std::nullopt;
  return plan;
}

} // namespace

LogicalResult
forwardSpmBroadcasts(ModuleOp module, SpmBroadcastMap &broadcasts,
                     llvm::function_ref<LogicalResult(ModuleOp)> verifyGraph) {
  SmallVector<BroadcastPlan> plans;
  module.walk([&](SpmViewOp view) {
    if (auto plan = planBroadcast(view))
      plans.push_back(std::move(*plan));
  });
  for (BroadcastPlan &plan : plans) {
    auto type = cast<MemRefType>(plan.view.getType());
    auto vectorType = VectorType::get(
        {kDmaAlignmentBytes / getSpmElementByteWidth(type.getElementType())},
        type.getElementType());
    DenseMap<Operation *, Value> fillValues;
    for (linalg::FillOp fill : plan.fills) {
      OpBuilder builder(fill);
      auto broadcast = vector::BroadcastOp::create(
          builder, fill.getLoc(), vectorType, fill.getInputs()[0]);
      if (verifyGraph)
        for (NamedAttribute attribute : fill->getAttrs())
          if (attribute.getName().getValue().starts_with("bosc_ame."))
            broadcast->setAttr(attribute.getName(), attribute.getValue());
      fillValues[fill] = broadcast;
      // Reader associations use operation pointers only as opaque map keys.
      fill->erase();
      if (verifyGraph && failed(verifyGraph(module)))
        return failure();
    }
    for (auto [reader, fill] : plan.readers) {
      auto &inputs = broadcasts[reader];
      inputs.resize(reader.getNumDpsInputs());
      for (auto [index, operand] : llvm::enumerate(reader.getInputs()))
        if (operand == plan.view.getResult())
          inputs[index] = fillValues.lookup(fill);
    }
  }
  return success();
}

} // namespace buddy::boscame::lowering
