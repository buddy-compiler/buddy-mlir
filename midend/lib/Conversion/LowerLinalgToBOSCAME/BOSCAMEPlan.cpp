//===- BOSCAMEPlan.cpp - Neutral kernel contract to explicit plan ---------===//
//----------===//
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
// Planning changes the storage graph before emission. Content versions are
// rebuilt from the resulting standard operations; physical windows and access
// routes remain separate, explicit decisions on those operations.
//
//===----------------------------------------------------------------------===//

#include "Analysis/KernelContract.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEAccumulatorPlacement.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEDDRPlan.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEFloatSchedule.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEHotAccumulator.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEPlanIR.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEPointwise.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMERegisterSchedule.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEStreamPlan.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMETargetInfo.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEZeroInit.h"
#include "Dialect/BOSCAME/DmaAlignment.h"
#include "Dialect/BOSCAME/Endpoint.h"
#include "Dialect/BOSCAME/MatrixTransfer.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <functional>
#include <optional>

using namespace mlir;

namespace buddy::boscame {
namespace {

Value storageRoot(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    if (auto view = dyn_cast<ViewLikeOpInterface>(op))
      value = view.getViewSource();
    else
      break;
  }
  return value;
}

bool privateRoot(Value value) {
  return isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(
      storageRoot(value).getDefiningOp());
}

// Earlier accesses may retain an address beyond their apparent position in
// the block. Only these synchronous, non-capturing users permit forwarding a
// later producer away from its private root.
bool nonCapturingMemoryUse(Operation *op) {
  if (isa<memref::CopyOp, memref::LoadOp, memref::StoreOp, memref::DimOp>(op))
    return true;
  if (!isa<linalg::LinalgOp>(op))
    return false;
  for (Region &region : op->getRegions()) {
    bool pure = true;
    region.walk([&](Operation *payload) {
      if (!isMemoryEffectFree(payload))
        pure = false;
    });
    if (!pure)
      return false;
  }
  return true;
}

bool orderedReference(Operation *op) {
  auto compute = op->getAttrOfType<DictionaryAttr>("contract.compute");
  return compute && compute.getAs<StringAttr>("order");
}

int64_t elementBytes(Value value) {
  auto type = cast<BaseMemRefType>(value.getType()).getElementType();
  return type.isIntOrFloat() ? (type.getIntOrFloatBitWidth() + 7) / 8 : 0;
}

std::optional<int64_t> bytes(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || !type.hasStaticShape())
    return std::nullopt;
  int64_t result = elementBytes(value);
  for (int64_t dim : type.getShape())
    if (dim <= 0 || llvm::MulOverflow(result, dim, result))
      return std::nullopt;
  return result > 0 ? std::optional(result) : std::nullopt;
}

std::optional<int64_t> storageBytes(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  SmallVector<int64_t> strides;
  int64_t offset;
  if (!type || !type.hasStaticShape() ||
      failed(type.getStridesAndOffset(strides, offset)) || offset < 0)
    return std::nullopt;
  int64_t span;
  if (llvm::AddOverflow(offset, int64_t(1), span))
    return std::nullopt;
  for (auto [dim, stride] : llvm::zip(type.getShape(), strides)) {
    int64_t delta;
    if (dim <= 0 || stride <= 0 || llvm::MulOverflow(dim - 1, stride, delta) ||
        llvm::AddOverflow(span, delta, span))
      return std::nullopt;
  }
  int64_t result;
  if (elementBytes(value) <= 0 ||
      llvm::MulOverflow(span, elementBytes(value), result))
    return std::nullopt;
  return result;
}

bool fullRoot(Value value) {
  // A view's byte shift changes its base separately from the descriptor's
  // element offset. Until the forwarding proof shares the full byte-address
  // normalizer, keep these snapshots instead of mistaking offset zero for a
  // full-root copy.
  for (Value current = value; Operation *op = current.getDefiningOp();) {
    if (isa<memref::ViewOp>(op))
      return false;
    auto view = dyn_cast<ViewLikeOpInterface>(op);
    if (!view)
      break;
    current = view.getViewSource();
  }
  auto type = dyn_cast<MemRefType>(value.getType());
  auto root = dyn_cast<MemRefType>(storageRoot(value).getType());
  SmallVector<int64_t> strides;
  int64_t offset;
  if (!type || !root || !type.hasStaticShape() || !root.hasStaticShape() ||
      failed(type.getStridesAndOffset(strides, offset)) ||
      spmdma::getStaticDescriptorOffset(value) != std::optional<int64_t>(0) ||
      bytes(value) != bytes(storageRoot(value)))
    return false;
  int64_t span = 1;
  for (auto [size, stride] : llvm::zip(type.getShape(), strides)) {
    int64_t delta;
    if (stride <= 0 || llvm::MulOverflow(size - 1, stride, delta) ||
        llvm::AddOverflow(span, delta, span))
      return false;
  }
  auto rootBytes = bytes(storageRoot(value));
  int64_t spanBytes;
  return rootBytes &&
         !llvm::MulOverflow(span, elementBytes(value), spanBytes) &&
         spanBytes == *rootBytes;
}

struct Attrs {
  Builder b;
  explicit Attrs(MLIRContext *context) : b(context) {}
  NamedAttribute i(StringRef name, int64_t value) {
    return b.getNamedAttr(name, b.getI64IntegerAttr(value));
  }
  NamedAttribute s(StringRef name, StringRef value) {
    return b.getNamedAttr(name, b.getStringAttr(value));
  }
  NamedAttribute flag(StringRef name, bool value) {
    return b.getNamedAttr(name, b.getBoolAttr(value));
  }
  DictionaryAttr d(ArrayRef<NamedAttribute> values) {
    return b.getDictionaryAttr(values);
  }
};

struct Geometry {
  int64_t m, n, k, tm, tn, tk;
};

FailureOr<Geometry> geometry(linalg::MatmulOp op) {
  auto a = dyn_cast<MemRefType>(op.getInputs()[0].getType());
  auto b = dyn_cast<MemRefType>(op.getInputs()[1].getType());
  auto c = dyn_cast<MemRefType>(op.getOutputs()[0].getType());
  if (!a || !b || !c || !a.hasStaticShape() || !b.hasStaticShape() ||
      !c.hasStaticShape() || a.getRank() != 2 || b.getRank() != 2 ||
      c.getRank() != 2 || !a.getElementType().isInteger(8) ||
      !b.getElementType().isInteger(8) || !c.getElementType().isInteger(32) ||
      op.getCast() != linalg::TypeFn::cast_signed) {
    op.emitError(
        "AME planning requires buffer-form signed i8 x i8 -> i32 matmul");
    return failure();
  }
  Geometry g{a.getDimSize(0),
             b.getDimSize(1),
             a.getDimSize(1),
             std::min<int64_t>(a.getDimSize(0), 64),
             std::min<int64_t>(b.getDimSize(1), 64),
             64};
  if (g.m <= 0 || g.m > 64 || g.n <= 0 || g.k <= 0 || g.n % g.tn ||
      g.k % g.tk || (g.tn * 4) % 64) {
    op.emitError("AME plan requires frontend tiling/padding: M in [1,64], "
                 "K divisible by 64, and full 64-byte N subtiles");
    return failure();
  }
  return g;
}

struct OldTask {
  int64_t id;
  Operation *operation;
  StringRef action = "retained";
};

class Planner {
public:
  explicit Planner(ModuleOp module, ArrayRef<FixedSPMBuffer> fixed,
                   PlanOptions options)
      : module(module), a(module.getContext()), fixedInputs(fixed),
        options(options) {}

  LogicalResult run();

private:
  ModuleOp module;
  Attrs a;
  llvm::DenseMap<Value, DictionaryAttr> placements;
  llvm::DenseMap<Value, int64_t> preferredBank;
  llvm::DenseMap<Operation *, Geometry> geometries;
  llvm::DenseSet<Operation *> orderedMatmuls;
  llvm::DenseMap<Operation *, SmallVector<OldTask>> oldTasks;
  llvm::DenseSet<Operation *> capacityForwarded;
  llvm::DenseSet<Operation *> terminalForwarded;
  llvm::DenseSet<Operation *> sharedBranches;
  ArrayRef<FixedSPMBuffer> fixedInputs;
  llvm::DenseMap<Value, FixedSPMBuffer> fixedBuffers;
  PlanOptions options;

  bool alignedArgument(Value value, int64_t alignmentBytes = 64) const;
  bool alignedRoot(Value value, int64_t alignmentBytes = 64) const;
  bool disjoint(Value first, Value second) const;
  bool directLegal(Value value, unsigned role) const;
  bool sourceStable(Value source, Operation *begin, Operation *end) const;
  bool destinationUnobserved(Value destination, Operation *begin,
                             Operation *end) const;
  bool isolatedRoot(Value value, ArrayRef<Operation *> users) const;
  void folded(Operation *from, Operation *to, StringRef action);
  void cleanupRoot(Value root);
  bool fuseAccumulator(linalg::MatmulOp op);
  void flattenPointwise();
  void forwardTerminalStores();
  LogicalResult forwardSnapshots(linalg::MatmulOp op, const Geometry &g,
                                 bool capacityFallback = false,
                                 bool *changed = nullptr);
  LogicalResult stageExternalOperands(linalg::MatmulOp op, const Geometry &g);
  LogicalResult placeBlock(Block &block, std::array<int64_t, 2> cursors,
                           std::array<int64_t, 2> &highWater,
                           bool commit = true);
  DictionaryAttr binding(Value value, unsigned index, StringRef reason);
  LogicalResult decisions(func::FuncOp function);
  LogicalResult transfer(memref::CopyOp op);
  LogicalResult internalTransfer(memref::CopyOp op, DictionaryAttr source,
                                 DictionaryAttr target);
  LogicalResult placeLegacyBlock(Block &block, std::array<int64_t, 2> cursors,
                                 std::array<int64_t, 2> &highWater,
                                 bool commit = true);
  std::array<int64_t, 2> reserved(func::FuncOp function) const;
};

bool Planner::alignedRoot(Value value, int64_t alignmentBytes) const {
  if (alignmentBytes <= 0)
    return false;
  Value root = storageRoot(value);
  if (!isa<BlockArgument>(root))
    return spmdma::isEndpointPointerAlignmentProven(root, alignmentBytes);
  if (!contract::hasEntryArgumentABI(
          root.getParentRegion()->getParentOfType<func::FuncOp>()))
    return false;
  auto abi = module->getAttrOfType<DictionaryAttr>("contract.abi");
  auto alignment =
      abi ? abi.getAs<IntegerAttr>("args_aligned_bytes") : IntegerAttr();
  return alignment && alignment.getInt() >= alignmentBytes &&
         alignment.getInt() % alignmentBytes == 0;
}

bool Planner::alignedArgument(Value value, int64_t alignmentBytes) const {
  return alignedRoot(value, alignmentBytes) &&
         spmdma::isEndpointPointerAlignmentProven(value, alignmentBytes);
}

bool Planner::disjoint(Value first, Value second) const {
  Value x = storageRoot(first), y = storageRoot(second);
  if (x != y) {
    if (privateRoot(x) || privateRoot(y))
      return true;
    auto abi = module->getAttrOfType<DictionaryAttr>("contract.abi");
    auto noalias = abi ? abi.getAs<BoolAttr>("args_disjoint") : BoolAttr();
    auto function = x.getParentRegion()->getParentOfType<func::FuncOp>();
    return noalias && noalias.getValue() && isa<BlockArgument>(x) &&
           isa<BlockArgument>(y) &&
           y.getParentRegion()->getParentOfType<func::FuncOp>() == function &&
           contract::hasEntryArgumentABI(function);
  }
  return false;
}

bool Planner::directLegal(Value value, unsigned role) const {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || !type.hasStaticShape() || type.getRank() != 2 ||
      privateRoot(value) || !alignedArgument(value))
    return false;
  auto layout = role == 0   ? AmeTileLayout::MxKWithKContiguous
                : role == 1 ? AmeTileLayout::KxNWithPhysicalNxK
                            : AmeTileLayout::MxNWithNContiguous;
  if (!hasAmeTileLayout(type, layout))
    return false;
  auto storage =
      computeTileStorageLayout(type, role == 0   ? TileRole::A
                                     : role == 1 ? TileRole::B
                                                 : TileRole::Accumulator);
  return storage && storage->rowStrideBytes % 64 == 0;
}

bool Planner::sourceStable(Value source, Operation *begin,
                           Operation *end) const {
  if (begin->getBlock() != end->getBlock() || !begin->isBeforeInBlock(end))
    return false;
  for (Operation *op = begin->getNextNode(); op != end;
       op = op->getNextNode()) {
    if (!op)
      return false;
    if (auto copy = dyn_cast<memref::CopyOp>(op)) {
      if (!disjoint(source, copy.getTarget()))
        return false;
    } else if (auto lin = dyn_cast<linalg::LinalgOp>(op)) {
      if (!nonCapturingMemoryUse(op))
        return false;
      if (llvm::any_of(lin.getDpsInits(),
                       [&](Value out) { return !disjoint(source, out); }))
        return false;
    } else if (isa<memref::AllocOp, memref::AllocaOp>(op)) {
      continue;
    } else if (!isMemoryEffectFree(op)) {
      return false;
    }
  }
  return true;
}

bool Planner::destinationUnobserved(Value destination, Operation *begin,
                                    Operation *end) const {
  if (begin->getBlock() != end->getBlock() || !begin->isBeforeInBlock(end))
    return false;
  for (Operation *op = begin->getNextNode(); op != end;
       op = op->getNextNode()) {
    if (!op)
      return false;
    if (isa<memref::AllocOp, memref::AllocaOp>(op) ||
        (op->getNumRegions() == 0 && isMemoryEffectFree(op)))
      continue;
    // Moving the terminal write to its producer must not change any intervening
    // observation, including reads. A stable source alone only excludes writes.
    if (!isa<memref::CopyOp, memref::LoadOp, memref::StoreOp, memref::DeallocOp,
             linalg::LinalgOp>(op))
      return false;
    if (isa<linalg::LinalgOp>(op) && !nonCapturingMemoryUse(op))
      return false;
    if (llvm::any_of(op->getOperands(), [&](Value operand) {
          return isa<BaseMemRefType>(operand.getType()) &&
                 !disjoint(destination, operand);
        }))
      return false;
  }
  return true;
}

bool Planner::isolatedRoot(Value value, ArrayRef<Operation *> allowed) const {
  Value root = storageRoot(value);
  bool isolated = true;
  auto function = root.getDefiningOp()->getParentOfType<func::FuncOp>();
  function.walk([&](Operation *op) {
    if (isa<ViewLikeOpInterface, memref::DeallocOp>(op) ||
        llvm::is_contained(allowed, op))
      return;
    if (llvm::any_of(op->getOperands(), [&](Value operand) {
          return isa<BaseMemRefType>(operand.getType()) &&
                 storageRoot(operand) == root;
        }))
      isolated = false;
  });
  return isolated;
}

void Planner::folded(Operation *from, Operation *to, StringRef action) {
  auto previous = to->getAttrOfType<DictionaryAttr>("contract.provenance");
  NamedAttrList provenance(previous);
  auto mergeOrigins = [&](StringRef key, ArrayRef<int64_t> added) {
    SmallVector<int64_t> origins(added);
    if (auto values =
            dyn_cast_if_present<DenseI64ArrayAttr>(provenance.get(key)))
      llvm::append_range(origins, values.asArrayRef());
    llvm::sort(origins);
    origins.erase(std::unique(origins.begin(), origins.end()), origins.end());
    provenance.set(key, a.b.getDenseI64ArrayAttr(origins));
  };
  if (auto source = from->getAttrOfType<DictionaryAttr>("contract.provenance"))
    for (StringRef key : {"source_load_origin_ids", "source_store_origin_ids"})
      if (auto origins = source.getAs<DenseI64ArrayAttr>(key))
        mergeOrigins(key, origins.asArrayRef());
  auto access = from->getAttrOfType<DictionaryAttr>("contract.access");
  if (access) {
    auto origin = access.getAs<IntegerAttr>("origin_id");
    if (origin) {
      StringRef key = action == "folded-store" ? "source_store_origin_ids"
                                               : "source_load_origin_ids";
      mergeOrigins(key, {origin.getInt()});
    }
  }
  if (!provenance.empty())
    to->setAttr("contract.provenance",
                provenance.getDictionary(module.getContext()));
  auto function = from->getParentOfType<func::FuncOp>();
  for (OldTask &task : oldTasks[function])
    if (task.operation == from) {
      task.operation = to;
      task.action = action;
    }
}

bool Planner::fuseAccumulator(linalg::MatmulOp op) {
  if (orderedReference(op))
    return false;
  if (!op.hasPureBufferSemantics() ||
      op.getCast() != linalg::TypeFn::cast_signed)
    return false;
  Value product = op.getOutputs()[0];
  auto type = dyn_cast<MemRefType>(product.getType());
  if (!type || !type.getElementType().isInteger(32) || !privateRoot(product) ||
      !fullRoot(product) ||
      !cast<MemRefType>(op.getInputs()[0].getType())
           .getElementType()
           .isInteger(8) ||
      !cast<MemRefType>(op.getInputs()[1].getType())
           .getElementType()
           .isInteger(8))
    return false;
  // Only cross pure, single-region-free address/scalar operations. Memory
  // accesses, calls and control flow retain their original observation points.
  auto harmless = [](Operation *candidate) {
    return candidate->getNumRegions() == 0 && isMemoryEffectFree(candidate);
  };
  Operation *next = op->getNextNode();
  while (next && harmless(next))
    next = next->getNextNode();
  auto add = dyn_cast_or_null<linalg::GenericOp>(next);
  if (!add || add.getNumDpsInputs() != 2 || add.getNumDpsInits() != 1 ||
      add.getNumLoops() != 2 ||
      !llvm::all_of(add.getIteratorTypesArray(),
                    [](utils::IteratorType iterator) {
                      return iterator == utils::IteratorType::parallel;
                    }) ||
      !llvm::all_of(add.getIndexingMapsArray(),
                    [](AffineMap map) { return map.isIdentity(); }))
    return false;
  Block &body = add.getRegion().front();
  if (body.getOperations().size() != 2)
    return false;
  auto sum = dyn_cast<arith::AddIOp>(body.front());
  auto yield = dyn_cast<linalg::YieldOp>(body.back());
  if (!sum || !yield ||
      sum.getOverflowFlags() != arith::IntegerOverflowFlags::none ||
      !((sum.getLhs() == body.getArgument(0) &&
         sum.getRhs() == body.getArgument(1)) ||
        (sum.getLhs() == body.getArgument(1) &&
         sum.getRhs() == body.getArgument(0))) ||
      yield.getValues().size() != 1 || yield.getValues()[0] != sum.getResult())
    return false;
  Value accumulator;
  if (add.getInputs()[0] == product)
    accumulator = add.getInputs()[1];
  else if (add.getInputs()[1] == product)
    accumulator = add.getInputs()[0];
  else
    return false;
  auto accumulatorType = dyn_cast<MemRefType>(accumulator.getType());
  if (!accumulatorType || accumulatorType.getShape() != type.getShape() ||
      accumulatorType.getElementType() != type.getElementType() ||
      add.getOutputs()[0] != accumulator || !disjoint(product, accumulator) ||
      !disjoint(accumulator, op.getInputs()[0]) ||
      !disjoint(accumulator, op.getInputs()[1]))
    return false;
  Operation *previous = op->getPrevNode();
  while (previous && harmless(previous))
    previous = previous->getPrevNode();
  auto fill = dyn_cast_or_null<linalg::FillOp>(previous);
  if (!fill || fill.getOutputs()[0] != product ||
      !matchPattern(fill.getInputs()[0], m_Zero()) ||
      !isolatedRoot(product, {fill, op, add}))
    return false;
  // The initial zero product is private. Modulo-i32 addition lets the AME
  // operation read C directly, independent of its later storage route.
  op.getOutputsMutable()[0].set(accumulator);
  folded(fill, op, "eliminated");
  folded(add, op, "eliminated");
  add.erase();
  fill.erase();
  cleanupRoot(storageRoot(product));
  return true;
}

void Planner::flattenPointwise() {
  SmallVector<linalg::GenericOp> candidates;
  module.walk([&](linalg::GenericOp op) { candidates.push_back(op); });
  for (auto op : candidates) {
    if (isFloatScheduleMember(op) || isRegisterScheduleMember(op))
      continue;
    if (!op.hasPureBufferSemantics() || op.getNumDpsInits() != 1 ||
        !nonCapturingMemoryUse(op) || op.getNumLoops() != 2 ||
        !llvm::all_of(op.getIteratorTypesArray(),
                      [](utils::IteratorType it) {
                        return it == utils::IteratorType::parallel;
                      }) ||
        !llvm::all_of(op.getIndexingMapsArray(),
                      [](AffineMap map) { return map.isIdentity(); }))
      continue;
    auto output = dyn_cast<MemRefType>(op.getOutputs()[0].getType());
    if (!output || !output.hasStaticShape() || output.getRank() != 2)
      continue;
    bool contiguous = llvm::all_of(op->getOperands(), [&](Value operand) {
      auto type = dyn_cast<MemRefType>(operand.getType());
      SmallVector<int64_t> strides;
      int64_t offset;
      return type && type.getShape() == output.getShape() &&
             succeeded(type.getStridesAndOffset(strides, offset)) &&
             strides[1] == 1 && strides[0] == type.getDimSize(1);
    });
    if (!contiguous)
      continue;
    OpBuilder builder(op);
    SmallVector<Value> inputs, outputs;
    for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
      Value flat = memref::CollapseShapeOp::create(
          builder, op.getLoc(), operand,
          ArrayRef<ReassociationIndices>{ReassociationIndices{0, 1}});
      (i < static_cast<size_t>(op.getNumDpsInputs()) ? inputs : outputs)
          .push_back(flat);
    }
    SmallVector<AffineMap> maps(
        op->getNumOperands(),
        AffineMap::getMultiDimIdentityMap(1, module.getContext()));
    auto flattened = linalg::GenericOp::create(
        builder, op.getLoc(), TypeRange{}, inputs, outputs, maps,
        ArrayRef<utils::IteratorType>{utils::IteratorType::parallel},
        [&](OpBuilder &nested, Location, ValueRange arguments) {
          IRMapping mapping;
          for (auto [source, target] :
               llvm::zip(op.getBody()->getArguments(), arguments))
            mapping.map(source, target);
          for (Operation &payload : op.getBody()->getOperations())
            nested.clone(payload, mapping);
        });
    for (NamedAttribute attr : op->getAttrs())
      if (attr.getName().getValue().starts_with("contract."))
        flattened->setAttr(attr.getName(), attr.getValue());
    auto function = op->getParentOfType<func::FuncOp>();
    for (OldTask &task : oldTasks[function])
      if (task.operation == op)
        task.operation = flattened;
    op.erase();
  }
}

void Planner::forwardTerminalStores() {
  SmallVector<linalg::GenericOp> candidates;
  module.walk([&](linalg::GenericOp op) { candidates.push_back(op); });
  auto compact = [](Value value) {
    auto type = dyn_cast<MemRefType>(value.getType());
    SmallVector<int64_t> strides;
    int64_t offset;
    if (!type || !type.hasStaticShape() || type.getRank() < 1 ||
        type.getRank() > 2 || failed(type.getStridesAndOffset(strides, offset)))
      return false;
    return strides.back() == 1 &&
           (type.getRank() == 1 || strides[0] == type.getDimSize(1));
  };
  for (auto op : candidates) {
    if (isFloatScheduleMember(op) || isRegisterScheduleMember(op))
      continue;
    if (!op.hasPureBufferSemantics() || op.getNumDpsInits() != 1 ||
        !nonCapturingMemoryUse(op) || op.getNumLoops() != 1 ||
        op.getNumParallelLoops() != 1 ||
        !llvm::all_of(op.getIndexingMapsArray(),
                      [](AffineMap map) { return map.isIdentity(); }))
      continue;
    Value output = op.getOutputs()[0];
    if (!privateRoot(output) || !fullRoot(output) || !compact(output) ||
        fixedBuffers.contains(storageRoot(output)))
      continue;
    Value root = storageRoot(output);
    auto function = op->getParentOfType<func::FuncOp>();
    memref::CopyOp store;
    bool terminal = true;
    function.walk([&](Operation *user) {
      if (user == op || isa<ViewLikeOpInterface, memref::DeallocOp>(user))
        return;
      if (!llvm::any_of(user->getOperands(), [&](Value operand) {
            return isa<BaseMemRefType>(operand.getType()) &&
                   storageRoot(operand) == root;
          }))
        return;
      if (user->getBlock() == op->getBlock() && user->isBeforeInBlock(op)) {
        if (!nonCapturingMemoryUse(user))
          terminal = false;
        return;
      }
      auto copy = dyn_cast<memref::CopyOp>(user);
      if (!store && copy && storageRoot(copy.getSource()) == root &&
          !privateRoot(copy.getTarget()) &&
          copy->getBlock() == op->getBlock() && op->isBeforeInBlock(copy))
        store = copy;
      else
        terminal = false;
    });
    if (!terminal || !store || !fullRoot(store.getSource()) ||
        !compact(store.getSource()) || !compact(store.getTarget()) ||
        bytes(output) != bytes(store.getSource()) ||
        !alignedArgument(store.getTarget()) ||
        !destinationUnobserved(store.getTarget(), op, store) ||
        llvm::any_of(op->getOperands(), [&](Value value) {
          return !disjoint(store.getTarget(), value);
        }))
      continue;
    auto targetType = cast<MemRefType>(store.getTarget().getType());
    if (targetType.getElementType() !=
            cast<MemRefType>(output.getType()).getElementType() ||
        bytes(output) != bytes(store.getTarget()))
      continue;

    DominanceInfo dominance(function);
    SmallVector<Operation *> addressOps;
    llvm::SmallPtrSet<Operation *, 8> visited;
    std::function<bool(Value)> materializable = [&](Value value) {
      if (dominance.properlyDominates(value, op))
        return true;
      Operation *def = value.getDefiningOp();
      if (!def || def->getNumRegions() != 0 || !isMemoryEffectFree(def))
        return false;
      if (!visited.insert(def).second)
        return true;
      for (Value operand : def->getOperands())
        if (!materializable(operand))
          return false;
      addressOps.push_back(def);
      return true;
    };
    if (!materializable(store.getTarget()))
      continue;

    OpBuilder builder(op);
    IRMapping addressMapping;
    for (Operation *address : addressOps)
      builder.clone(*address, addressMapping);
    Value destination = addressMapping.lookupOrDefault(store.getTarget());
    if (targetType.getRank() == 2)
      destination = memref::CollapseShapeOp::create(
          builder, op.getLoc(), destination,
          ArrayRef<ReassociationIndices>{ReassociationIndices{0, 1}});
    // An in-place generic may read its old outs. Preserve that private version
    // as a separate input; rebinding outs must never substitute the DDR init.
    SmallVector<Value> inputs(op.getInputs());
    bool readsOutput = !op.getBody()->getArguments().back().use_empty();
    if (readsOutput)
      inputs.push_back(output);
    SmallVector<AffineMap> maps(
        inputs.size() + 1,
        AffineMap::getMultiDimIdentityMap(1, module.getContext()));
    auto forwarded = linalg::GenericOp::create(
        builder, op.getLoc(), TypeRange{}, inputs, ValueRange{destination},
        maps, ArrayRef<utils::IteratorType>{utils::IteratorType::parallel},
        [&](OpBuilder &nested, Location, ValueRange arguments) {
          IRMapping mapping;
          unsigned inputCount = op.getNumDpsInputs();
          for (unsigned i = 0; i < inputCount; ++i)
            mapping.map(op.getBody()->getArgument(i), arguments[i]);
          mapping.map(op.getBody()->getArgument(inputCount),
                      readsOutput ? arguments[inputCount] : arguments.back());
          for (Operation &payload : op.getBody()->getOperations())
            nested.clone(payload, mapping);
        });
    for (NamedAttribute attr : op->getAttrs())
      if (attr.getName().getValue().starts_with("contract."))
        forwarded->setAttr(attr.getName(), attr.getValue());
    for (OldTask &task : oldTasks[function])
      if (task.operation == op)
        task.operation = forwarded;
    folded(store, forwarded, "folded-store");
    terminalForwarded.insert(forwarded);
    store.erase();
    op.erase();
    if (isolatedRoot(root, {}))
      cleanupRoot(root);
  }
}

void Planner::cleanupRoot(Value root) {
  auto function = root.getDefiningOp()->getParentOfType<func::FuncOp>();
  SmallVector<memref::DeallocOp> deallocs;
  function.walk([&](memref::DeallocOp op) {
    if (storageRoot(op.getMemref()) == root)
      deallocs.push_back(op);
  });
  for (auto op : deallocs)
    op.erase();
  bool changed;
  do {
    changed = false;
    SmallVector<Operation *> dead;
    function.walk([&](Operation *op) {
      if (isa<ViewLikeOpInterface>(op) && op->getNumResults() == 1 &&
          op->getResult(0).use_empty() && storageRoot(op->getResult(0)) == root)
        dead.push_back(op);
    });
    for (Operation *op : dead) {
      op->erase();
      changed = true;
    }
  } while (changed);
  if (root.use_empty()) {
    fixedBuffers.erase(root);
    root.getDefiningOp()->erase();
  }
}

LogicalResult Planner::forwardSnapshots(linalg::MatmulOp op, const Geometry &g,
                                        bool capacityFallback, bool *changed) {
  for (unsigned role = 0; role != 3; ++role) {
    if (capacityFallback && role != 0)
      continue;
    Value operand = op->getOperand(role);
    if (!privateRoot(operand) || !fullRoot(operand) ||
        fixedBuffers.contains(storageRoot(operand)))
      continue;
    Value root = storageRoot(operand);
    SmallVector<memref::CopyOp> loads, stores;
    auto function = op->getParentOfType<func::FuncOp>();
    function.walk([&](memref::CopyOp copy) {
      if (storageRoot(copy.getTarget()) == root &&
          !privateRoot(copy.getSource()))
        loads.push_back(copy);
      if (storageRoot(copy.getSource()) == root &&
          !privateRoot(copy.getTarget()))
        stores.push_back(copy);
    });
    if (loads.size() != 1 || (role != 2 && !stores.empty()) ||
        (role == 2 && stores.size() != 1))
      continue;
    auto load = loads.front();
    Value source = load.getSource();
    if (!fullRoot(load.getTarget()) || source.getType() != operand.getType()) {
      auto sourceType = dyn_cast<MemRefType>(source.getType());
      auto operandType = dyn_cast<MemRefType>(operand.getType());
      if (!fullRoot(load.getTarget()) || !sourceType || !operandType ||
          sourceType.getShape() != operandType.getShape() ||
          sourceType.getElementType() != operandType.getElementType())
        continue;
    }
    if (!directLegal(source, role) || !sourceStable(source, load, op))
      continue;
    // The footprint already includes all K slices. A repeats across N tiles;
    // B repeats across M tiles. Do not multiply either by K tile count.
    int64_t repeats = role == 0 ? g.n / g.tn : 1;
    if (options.version == 1 && !capacityFallback && role != 2 && repeats != 1)
      continue;
    // Repetition alone is not a proof that SPM is profitable. Reuse is
    // handled by the register schedule or a future measured reuse cost model.
    SmallVector<Operation *> allowed{load, op};
    if (role == 2) {
      auto store = stores.front();
      if (!fullRoot(store.getSource()) || store.getTarget() != source ||
          store->getBlock() != op->getBlock() || !op->isBeforeInBlock(store) ||
          !destinationUnobserved(source, op, store) ||
          !disjoint(source, op.getInputs()[0]) ||
          !disjoint(source, op.getInputs()[1]))
        continue;
      allowed.push_back(store);
    }
    if (!isolatedRoot(root, allowed))
      continue;
    // Rebinding removes the snapshot task that supplied this operand's
    // lineage. Preserve that actual read per operand, replacing any facts
    // left from the product's former zero accumulator before fusion.
    StringRef prefix = role == 0   ? "a_source"
                       : role == 1 ? "b_source"
                                   : "accumulator_source";
    NamedAttrList provenance(
        op->getAttrOfType<DictionaryAttr>("contract.provenance"));
    auto sourceFacts =
        load->getAttrOfType<DictionaryAttr>("contract.provenance");
    std::string originsKey = (prefix + "_load_origin_ids").str();
    std::string completeKey = (prefix + "_lineage_complete").str();
    provenance.erase(originsKey);
    provenance.erase(completeKey);
    if (sourceFacts) {
      if (auto origins =
              sourceFacts.getAs<DenseI64ArrayAttr>("source_load_origin_ids"))
        provenance.set(originsKey, origins);
      if (auto complete =
              sourceFacts.getAs<BoolAttr>("source_lineage_complete"))
        provenance.set(completeKey, complete);
    }
    op->setAttr("contract.provenance",
                provenance.getDictionary(module.getContext()));
    op->setOperand(role, source);
    if (capacityFallback)
      capacityForwarded.insert(op);
    if (changed)
      *changed = true;
    folded(load, op, "folded-load");
    load.erase();
    if (role == 2) {
      folded(stores.front(), op, "folded-store");
      stores.front().erase();
    }
    cleanupRoot(root);
  }
  return success();
}

LogicalResult Planner::stageExternalOperands(linalg::MatmulOp op,
                                             const Geometry &g) {
  for (unsigned role = 0; role != 3; ++role) {
    Value value = op->getOperand(role);
    if (privateRoot(value))
      continue;
    bool direct = directLegal(value, role);
    bool legacyTraffic = options.version == 1 && role == 0 && g.n / g.tn > 1;
    if (direct && !legacyTraffic)
      continue;
    auto type = cast<MemRefType>(value.getType());
    int64_t minor = role == 1 ? 0 : 1;
    SmallVector<int64_t> strides;
    int64_t offset, rowBytes, pitch;
    if (!alignedArgument(value, elementBytes(value)) ||
        failed(type.getStridesAndOffset(strides, offset)) ||
        strides[minor] != 1 || strides[1 - minor] <= 0 ||
        llvm::MulOverflow(type.getDimSize(minor), elementBytes(value),
                          rowBytes) ||
        llvm::MulOverflow(strides[1 - minor], elementBytes(value), pitch) ||
        rowBytes <= 0 || pitch <= 0 ||
        (type.getDimSize(1 - minor) > 1 && pitch < rowBytes))
      return op.emitError("no legal direct or staging route: external operand "
                          "requires natural endpoint alignment and a bounded "
                          "nonoverlapping contiguous-row layout");
    OpBuilder builder(op);
    SmallVector<int64_t> packed =
        role == 1 ? SmallVector<int64_t>{1, type.getDimSize(0)}
                  : SmallVector<int64_t>{type.getDimSize(1), 1};
    auto stagedType =
        MemRefType::get(type.getShape(), type.getElementType(),
                        StridedLayoutAttr::get(module.getContext(), 0, packed));
    auto buffer = memref::AllocOp::create(builder, op.getLoc(), stagedType,
                                          builder.getI64IntegerAttr(64));
    buffer->setAttr(
        "contract.buffer",
        a.d({a.s("role", "tile"), a.s("kind", "temporary"),
             a.b.getNamedAttr("logical_shape",
                              a.b.getDenseI64ArrayAttr(type.getShape()))}));
    auto load = memref::CopyOp::create(builder, op.getLoc(), value, buffer);
    auto provenance = op->getAttrOfType<DictionaryAttr>("contract.provenance");
    auto accessFor = [&](bool store) {
      NamedAttrList fields;
      fields.append(a.s("kind", store ? "store" : "load"));
      fields.append(a.s("boundary", "external_ddr"));
      fields.append(a.b.getNamedAttr("mask", a.d({a.s("kind", "none")})));
      StringRef key = store       ? "source_store_origin_ids"
                      : role == 0 ? "a_source_load_origin_ids"
                      : role == 1 ? "b_source_load_origin_ids"
                                  : "accumulator_source_load_origin_ids";
      if (provenance)
        if (auto origins = provenance.getAs<DenseI64ArrayAttr>(key))
          if (!origins.empty())
            fields.append(a.i("origin_id", origins.asArrayRef().front()));
      return fields.getDictionary(module.getContext());
    };
    load->setAttr("contract.access", accessFor(false));
    NamedAttrList sourceFacts;
    sourceFacts.append(a.s("generated_by", "planner-staging"));
    StringRef sourceKey = role == 0   ? "a_source_load_origin_ids"
                          : role == 1 ? "b_source_load_origin_ids"
                                      : "accumulator_source_load_origin_ids";
    StringRef completeKey = role == 0   ? "a_source_lineage_complete"
                            : role == 1 ? "b_source_lineage_complete"
                                        : "accumulator_source_lineage_complete";
    if (provenance) {
      if (auto origins = provenance.getAs<DenseI64ArrayAttr>(sourceKey))
        sourceFacts.append("source_load_origin_ids", origins);
      if (auto complete = provenance.getAs<BoolAttr>(completeKey))
        sourceFacts.append("source_lineage_complete", complete);
    }
    load->setAttr("contract.provenance",
                  sourceFacts.getDictionary(module.getContext()));
    op->setOperand(role, buffer);
    if (role == 2) {
      builder.setInsertionPointAfter(op);
      auto store = memref::CopyOp::create(builder, op.getLoc(), buffer, value);
      store->setAttr("contract.access", accessFor(true));
      NamedAttrList storeFacts;
      storeFacts.append(a.s("generated_by", "planner-staging"));
      if (provenance)
        for (StringRef key :
             {"source_load_origin_ids", "source_lineage_complete",
              "source_store_origin_ids", "source_store_lineage_complete"})
          if (auto fact = provenance.get(key))
            storeFacts.append(key, fact);
      store->setAttr("contract.provenance",
                     storeFacts.getDictionary(module.getContext()));
    }
  }
  return success();
}

LogicalResult Planner::placeLegacyBlock(Block &block,
                                        std::array<int64_t, 2> cursors,
                                        std::array<int64_t, 2> &highWater,
                                        bool commit) {
  for (Operation &op : block) {
    if (isa<memref::AllocOp, memref::AllocaOp>(op)) {
      Value root = op.getResult(0);
      auto size = storageBytes(root);
      if (!size) {
        if (commit)
          op.emitError(
              "automatic placement requires a bounded static private root");
        return failure();
      }
      auto type = cast<MemRefType>(root.getType());
      DictionaryAttr decision;
      auto fixed = fixedBuffers.find(root);
      if (fixed != fixedBuffers.end()) {
        const auto &window = fixed->second;
        decision =
            a.d({a.s("level", "spm"), a.i("bank", window.bank),
                 a.i("offset", window.offset), a.i("capacity", window.capacity),
                 a.s("role", "fixed-input")});
        highWater[window.bank] =
            std::max(highWater[window.bank], window.offset + window.capacity);
      } else if (type.getRank() == 0) {
        decision = a.d({a.s("level", "host"), a.i("capacity", *size),
                        a.s("role", "reduction")});
      } else {
        if (!options.automaticPlacement) {
          if (commit)
            op.emitError("buffer-placement debug override disables required "
                         "automatic placement of this private SPM root");
          return failure();
        }
        int64_t bank = preferredBank.count(root)
                           ? preferredBank.lookup(root)
                           : (cursors[0] <= cursors[1] ? 0 : 1);
        int64_t offset = llvm::alignTo(cursors[bank], int64_t(64));
        int64_t end;
        if (llvm::AddOverflow(offset, *size, end) || end > 131072) {
          int64_t otherOffset = llvm::alignTo(cursors[1 - bank], int64_t(64));
          if (llvm::AddOverflow(otherOffset, *size, end) || end > 131072) {
            if (commit)
              op.emitError("SPM workspace capacity exceeded after legal "
                           "direct/staging alternatives: private root needs ")
                  << *size << " bytes";
            return failure();
          }
          bank = 1 - bank;
          offset = otherOffset;
        }
        cursors[bank] = end;
        highWater[bank] = std::max(highWater[bank], end);
        decision =
            a.d({a.s("level", "spm"), a.i("bank", bank), a.i("offset", offset),
                 a.i("capacity", *size), a.s("role", "private")});
      }
      if (commit) {
        placements[root] = decision;
        op.setAttr("plan.buffer", decision);
      }
      continue;
    }
    if (isa<linalg::LinalgOp>(op))
      continue;
    bool shareArms = false;
    if (auto branch = dyn_cast<scf::IfOp>(op)) {
      SmallVector<Value> arms[2];
      for (auto [index, region] : llvm::enumerate(branch->getRegions()))
        region.walk([&](Operation *nested) {
          if (isa<memref::AllocOp, memref::AllocaOp>(nested)) {
            auto type = dyn_cast<MemRefType>(nested->getResult(0).getType());
            if (type && type.getRank() != 0)
              arms[index].push_back(nested->getResult(0));
          }
        });
      shareArms = llvm::all_of(arms[0], [&](Value first) {
        return llvm::all_of(arms[1], [&](Value second) {
          return canShareExclusiveSPMWindow(first, second);
        });
      });
    }
    if (shareArms) {
      if (commit)
        sharedBranches.insert(&op);
      std::array<int64_t, 2> branchEnd = cursors;
      for (Region &region : op.getRegions())
        for (Block &nested : region) {
          std::array<int64_t, 2> armEnd = cursors;
          if (failed(placeLegacyBlock(nested, cursors, armEnd, commit)))
            return failure();
          for (unsigned bank = 0; bank != 2; ++bank)
            branchEnd[bank] = std::max(branchEnd[bank], armEnd[bank]);
        }
      for (unsigned bank = 0; bank != 2; ++bank) {
        cursors[bank] = branchEnd[bank];
        highWater[bank] = std::max(highWater[bank], cursors[bank]);
      }
      continue;
    }
    for (Region &region : op.getRegions())
      for (Block &nested : region) {
        std::array<int64_t, 2> nestedEnd = cursors;
        if (failed(placeLegacyBlock(nested, cursors, nestedEnd, commit)))
          return failure();
        for (unsigned bank = 0; bank != 2; ++bank) {
          cursors[bank] = nestedEnd[bank];
          highWater[bank] = std::max(highWater[bank], cursors[bank]);
        }
      }
  }
  return success();
}

LogicalResult Planner::placeBlock(Block &block, std::array<int64_t, 2> cursors,
                                  std::array<int64_t, 2> &highWater,
                                  bool commit) {
  if (options.version == 1)
    return placeLegacyBlock(block, cursors, highWater, commit);
  for (Operation &op : block) {
    if (isa<memref::AllocOp, memref::AllocaOp>(op)) {
      Value root = op.getResult(0);
      auto size = storageBytes(root);
      if (!size) {
        if (commit)
          op.emitError(
              "automatic placement requires a bounded static private root");
        return failure();
      }
      DictionaryAttr decision;
      auto fixed = fixedBuffers.find(root);
      if (isStreamVirtual(root) || isRegisterScheduleVirtual(&op) ||
          isFloatScheduleVirtual(&op)) {
        decision = a.d({a.s("level", "virtual"), a.i("capacity", *size),
                        a.s("role", "logical-value")});
      } else if (fixed != fixedBuffers.end()) {
        const auto &window = fixed->second;
        decision =
            a.d({a.s("level", "spm"), a.i("bank", window.bank),
                 a.i("offset", window.offset), a.i("capacity", window.capacity),
                 a.s("role", "fixed-input")});
        highWater[window.bank] =
            std::max(highWater[window.bank], window.offset + window.capacity);
      } else {
        decision = a.d({a.s("level", "ddr"), a.i("capacity", *size),
                        a.i("alignment", 64), a.s("role", "workspace"),
                        a.s("reason", "reuse-benefit-unproven")});
        // Exact complete-adapter measurements qualify only the closed source
        // signature and bank0/offset0 envelope. Every fixed window has already
        // contributed to cursors, including windows later in source order.
        auto fixedEnds = reserved(op.getParentOfType<func::FuncOp>());
        if (auto hot = getHotAccumulatorPlacement(
                root, options.automaticPlacement,
                cursors[0] == 0 && fixedEnds[0] == 0 && fixedEnds[1] == 0)) {
          decision = hot;
          if (hot.getAs<StringAttr>("level").getValue() == "spm") {
            cursors[0] = *size;
            highWater[0] = std::max(highWater[0], *size);
          }
        }
        if (commit)
          op.setAttr("alignment", a.b.getI64IntegerAttr(64));
      }
      if (commit) {
        placements[root] = decision;
        op.setAttr("plan.buffer", decision);
      }
      continue;
    }
    if (isa<linalg::LinalgOp>(op))
      continue;
    bool shareArms = false;
    if (auto branch = dyn_cast<scf::IfOp>(op)) {
      SmallVector<Value> arms[2];
      for (auto [index, region] : llvm::enumerate(branch->getRegions()))
        region.walk([&](Operation *nested) {
          if (isa<memref::AllocOp, memref::AllocaOp>(nested)) {
            auto type = dyn_cast<MemRefType>(nested->getResult(0).getType());
            if (type && type.getRank() != 0)
              arms[index].push_back(nested->getResult(0));
          }
        });
      shareArms = llvm::all_of(arms[0], [&](Value first) {
        return llvm::all_of(arms[1], [&](Value second) {
          return canShareExclusiveSPMWindow(first, second);
        });
      });
    }
    if (shareArms) {
      if (commit)
        sharedBranches.insert(&op);
      std::array<int64_t, 2> branchEnd = cursors;
      for (Region &region : op.getRegions())
        for (Block &nested : region) {
          std::array<int64_t, 2> armEnd = cursors;
          if (failed(placeBlock(nested, cursors, armEnd, commit)))
            return failure();
          for (unsigned bank = 0; bank != 2; ++bank)
            branchEnd[bank] = std::max(branchEnd[bank], armEnd[bank]);
        }
      for (unsigned bank = 0; bank != 2; ++bank) {
        cursors[bank] = branchEnd[bank];
        highWater[bank] = std::max(highWater[bank], cursors[bank]);
      }
      continue;
    }
    for (Region &region : op.getRegions())
      for (Block &nested : region) {
        std::array<int64_t, 2> nestedEnd = cursors;
        if (failed(placeBlock(nested, cursors, nestedEnd, commit)))
          return failure();
        for (unsigned bank = 0; bank != 2; ++bank) {
          cursors[bank] = nestedEnd[bank];
          highWater[bank] = std::max(highWater[bank], cursors[bank]);
        }
      }
  }
  return success();
}

std::array<int64_t, 2> Planner::reserved(func::FuncOp function) const {
  std::array<int64_t, 2> ends{0, 0};
  for (const auto &entry : fixedBuffers) {
    const auto &window = entry.second;
    if (window.root.getDefiningOp()->getParentOfType<func::FuncOp>() ==
        function)
      ends[window.bank] =
          std::max(ends[window.bank], window.offset + window.capacity);
  }
  return ends;
}

DictionaryAttr Planner::binding(Value value, unsigned index, StringRef reason) {
  NamedAttrList fields;
  fields.append(a.i("operand", index));
  if (!isa<BaseMemRefType>(value.getType()))
    return a.d({a.i("operand", index), a.s("level", "scalar"),
                a.s("reason", "explicit-scalar-input")});
  Value root = storageRoot(value);
  auto placement = placements.lookup(root);
  fields.append(
      a.s("level",
          placement ? placement.getAs<StringAttr>("level").getValue() : "ddr"));
  if (placement && placement.getAs<StringAttr>("level").getValue() == "spm") {
    fields.append(a.i("bank", placement.getAs<IntegerAttr>("bank").getInt()));
    fields.append(
        a.i("offset", placement.getAs<IntegerAttr>("offset").getInt()));
  }
  fields.append(a.s("reason", reason));
  if (auto footprint = bytes(value))
    fields.append(a.i("bytes", *footprint));
  return fields.getDictionary(module.getContext());
}

LogicalResult Planner::internalTransfer(memref::CopyOp op,
                                        DictionaryAttr sourcePlacement,
                                        DictionaryAttr targetPlacement) {
  auto source = cast<MemRefType>(op.getSource().getType());
  auto target = cast<MemRefType>(op.getTarget().getType());
  auto axis = matrixtransfer::getCommonUnitAxis(source, target);
  if (!axis || source.getShape() != target.getShape() ||
      sourcePlacement.getAs<StringAttr>("level").getValue() != "spm" ||
      targetPlacement.getAs<StringAttr>("level").getValue() != "spm")
    return op.emitError(
        "no legal internal copy recipe: requires matching static rank-two "
        "SPM storage of signless i8/i16/i32/i64 or f32/f64 with a common "
        "physical contiguous axis; physical TR/ACC conversion is unsupported");
  auto sourceLayout = *matrixtransfer::getLayout(source, *axis);
  auto targetLayout = *matrixtransfer::getLayout(target, *axis);
  for (MemRefType type : {source, target}) {
    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(type.getStridesAndOffset(strides, offset)) ||
        offset == ShapedType::kDynamic)
      return op.emitError(
          "no legal internal copy recipe: requires static descriptor offsets");
  }
  int64_t count = sourceLayout.rows, width = sourceLayout.rowBytes;
  if (width % 4 || sourceLayout.pitch % 4 || targetLayout.pitch % 4 ||
      !spmdma::isEndpointPointerAlignmentProven(op.getSource(), 4) ||
      !spmdma::isEndpointPointerAlignmentProven(op.getTarget(), 4))
    return op.emitError("no legal internal copy recipe: raw ACC32 requires "
                        "exact word-aligned rows, pitches and endpoints");
  auto endpoint = [&](Value value, DictionaryAttr placement,
                      const matrixtransfer::Layout &layout,
                      unsigned operand) -> DictionaryAttr {
    auto origin = spmdma::getStaticPointerOrigin(value);
    auto offset = spmdma::getStaticDescriptorOffset(value);
    auto field = [&](StringRef name) {
      return placement.getAs<IntegerAttr>(name).getInt();
    };
    int64_t begin, end, relative, span;
    if (!origin || !offset || *offset < 0 || origin->byteShift < 0 ||
        origin->root != storageRoot(value) ||
        llvm::MulOverflow(*offset, layout.elementBytes, relative) ||
        llvm::AddOverflow(relative, origin->byteShift, relative) ||
        relative < 0 || llvm::MulOverflow(count - 1, layout.pitch, span) ||
        llvm::AddOverflow(span, width, span) ||
        llvm::AddOverflow(relative, span, end) || end > field("capacity") ||
        llvm::AddOverflow(field("offset"), relative, begin) ||
        llvm::AddOverflow(field("offset"), end, end))
      return {};
    return a.d({a.i("operand", operand), a.s("level", "spm"),
                a.i("bank", field("bank")),
                a.i("window_offset", field("offset")),
                a.i("capacity", field("capacity")), a.i("begin", begin),
                a.i("end", end)});
  };
  auto src = endpoint(op.getSource(), sourcePlacement, sourceLayout, 0);
  auto dst = endpoint(op.getTarget(), targetPlacement, targetLayout, 1);
  if (!src || !dst)
    return op.emitError("internal copy endpoint exceeds its planned SPM "
                        "window or lacks an exact static byte range");
  auto integer = [](DictionaryAttr p, StringRef name) {
    return p.getAs<IntegerAttr>(name).getInt();
  };
  if (storageRoot(op.getSource()) == storageRoot(op.getTarget()) ||
      (integer(src, "bank") == integer(dst, "bank") &&
       integer(src, "begin") < integer(dst, "end") &&
       integer(dst, "begin") < integer(src, "end")))
    return op.emitError("internal copy requires distinct nonaliasing roots "
                        "and nonoverlapping physical SPM ranges");
  auto layout = [&](MemRefType type, const matrixtransfer::Layout &geometry) {
    SmallVector<int64_t> strides;
    int64_t offset;
    (void)type.getStridesAndOffset(strides, offset);
    return a.d(
        {a.b.getNamedAttr("shape", a.b.getDenseI64ArrayAttr(type.getShape())),
         a.b.getNamedAttr("strides", a.b.getDenseI64ArrayAttr(strides)),
         a.i("element_bits", geometry.elementBytes * 8),
         a.i("descriptor_offset", offset), a.i("contiguous_axis", *axis),
         a.i("rows", count), a.i("row_bytes", width),
         a.i("row_pitch", geometry.pitch), a.s("origin", "effective-pointer")});
  };
  SmallVector<Attribute> tiles;
  for (int64_t row = 0; row < count; row += matrixtransfer::kMaxRows)
    for (int64_t column = 0; column < width;
         column += matrixtransfer::kMaxRowBytes) {
      int64_t id = tiles.size();
      tiles.push_back(
          a.d({a.i("id", id), a.i("predecessor", id - 1),
               a.i("source_offset", row * sourceLayout.pitch + column),
               a.i("target_offset", row * targetLayout.pitch + column),
               a.i("rows", std::min(matrixtransfer::kMaxRows, count - row)),
               a.i("row_bytes",
                   std::min(matrixtransfer::kMaxRowBytes, width - column)),
               a.i("source_pitch", sourceLayout.pitch),
               a.i("target_pitch", targetLayout.pitch),
               a.i("instruction_bits", matrixtransfer::kInstructionBits),
               a.i("register_count", matrixtransfer::kRegisterCount),
               a.s("register_class", matrixtransfer::kRegisterClass),
               a.i("mtype", matrixtransfer::kMtype)}));
    }
  int64_t total;
  if (llvm::MulOverflow(count, width, total))
    return op.emitError("internal copy byte accounting overflows");
  op->setAttr(
      "plan.transfer",
      a.d({a.s("role", "internal"), a.s("engine", "matrix"),
           a.s("recipe", "spm-acc32-spm"), a.s("from", "spm"), a.s("to", "spm"),
           a.s("temporary_window", "none"),
           a.s("ordering", "synchronous-fenced-tiles"),
           a.b.getNamedAttr("source_endpoint", src),
           a.b.getNamedAttr("target_endpoint", dst),
           a.b.getNamedAttr("source_layout", layout(source, sourceLayout)),
           a.b.getNamedAttr("target_layout", layout(target, targetLayout)),
           a.b.getNamedAttr("chunks",
                            a.d({a.i("count", count), a.i("bytes", width),
                                 a.i("source_stride", sourceLayout.pitch),
                                 a.i("target_stride", targetLayout.pitch)})),
           a.b.getNamedAttr("tiles", a.b.getArrayAttr(tiles)),
           a.i("total_bytes", total), a.i("temp_bytes", 0),
           a.i("completion", tiles.size() - 1), a.s("wait", "after-each-tile"),
           a.s("reason", "nonoverlapping-spm-exact-acc32-rectangle")}));
  return success();
}

LogicalResult Planner::transfer(memref::CopyOp op) {
  Value source = op.getSource(), target = op.getTarget();
  auto src = dyn_cast<MemRefType>(source.getType());
  auto dst = dyn_cast<MemRefType>(target.getType());
  if (!src || !dst)
    return op.emitError("DMA planning requires ranked endpoint views");
  auto srcPlacement = placements.lookup(storageRoot(source));
  auto dstPlacement = placements.lookup(storageRoot(target));
  auto isSPM = [](DictionaryAttr p) {
    auto level = p ? p.getAs<StringAttr>("level") : StringAttr();
    return level && level.getValue() == "spm";
  };
  bool srcSPM = isSPM(srcPlacement), dstSPM = isSPM(dstPlacement);
  if ((srcSPM && dstSPM) ||
      (options.version == 1 && srcPlacement && dstPlacement))
    return internalTransfer(op, srcPlacement, dstPlacement);
  bool load = !srcSPM && dstSPM;
  bool store = srcSPM && !dstSPM;
  if (!load && !store) {
    if (options.version == 1)
      return op.emitError("v1 has no independent DDR-to-DDR copy recipe; a "
                          "boundary copy requires a planned private endpoint");
    return op.emitError("no legal boundary route: boundary copy requires "
                        "exactly one SPM endpoint");
  }
  Value external = load ? source : target, local = load ? target : source;
  auto placement = load ? dstPlacement : srcPlacement;
  auto ddrPlacement = load ? srcPlacement : dstPlacement;
  bool privateDDR =
      options.version == 2 && ddrPlacement &&
      ddrPlacement.getAs<StringAttr>("level").getValue() == "ddr" &&
      classifyEndpoint(external) == EndpointKind::PrivateAllocation;
  if (privateDDR && failed(verifyPrivateEndpoint(op, external)))
    return failure();
  auto ddrRootAligned = [&](int64_t alignment = 64) {
    return privateDDR
               ? ddrPlacement.getAs<IntegerAttr>("alignment").getInt() >=
                         alignment &&
                     ddrPlacement.getAs<IntegerAttr>("alignment").getInt() %
                             alignment ==
                         0
               : alignedRoot(external, alignment);
  };
  auto ddrAligned = [&](int64_t alignment = 64) {
    return ddrRootAligned(alignment) &&
           spmdma::isEndpointPointerAlignmentProven(external, alignment);
  };
  int64_t element = spmdma::getSpmElementByteWidth(src.getElementType());
  if (!element || src.getElementType() != dst.getElementType() ||
      !ddrAligned(element) ||
      !spmdma::isEndpointPointerAlignmentProven(local, element))
    return op.emitError(
        "no legal boundary route: natural endpoint alignment, element type, "
        "or SPM placement is unproven");
  SmallVector<int64_t> ss, ds;
  int64_t so, destOffset;
  if (!src.hasStaticShape() || src.getShape() != dst.getShape() ||
      elementBytes(source) != elementBytes(target) ||
      failed(src.getStridesAndOffset(ss, so)) ||
      failed(dst.getStridesAndOffset(ds, destOffset)) || src.getRank() < 1 ||
      src.getRank() > 2 || llvm::any_of(ss, [](int64_t s) { return s < 0; }) ||
      llvm::any_of(ds, [](int64_t s) { return s < 0; }))
    return op.emitError(
        "DMA plan needs matching static rank-one/rank-two rectangles");
  unsigned minor = src.getRank() - 1;
  if ((ss[minor] != 1 || ds[minor] != 1) && src.getRank() == 2)
    minor = 0;
  if (ss[minor] != 1 || ds[minor] != 1)
    return op.emitError("no legal DMA route: source/destination physical "
                        "unit-stride axes differ");
  int64_t width;
  int64_t count = src.getRank() == 1 ? 1 : src.getDimSize(1 - minor);
  int64_t sourceStride, targetStride;
  if (llvm::MulOverflow(src.getDimSize(minor), elementBytes(source), width) ||
      llvm::MulOverflow(src.getRank() == 1 ? src.getDimSize(0) : ss[1 - minor],
                        elementBytes(source), sourceStride) ||
      llvm::MulOverflow(dst.getRank() == 1 ? dst.getDimSize(0) : ds[1 - minor],
                        elementBytes(target), targetStride))
    return op.emitError("DMA geometry exceeds the supported byte range");
  int64_t last;
  if (count <= 0 || width <= 0 || sourceStride < 0 || targetStride < 0 ||
      (count > 1 && (sourceStride < width || targetStride < width)) ||
      llvm::MulOverflow(count - 1, sourceStride, last) ||
      llvm::AddOverflow(last, width, last) ||
      llvm::MulOverflow(count - 1, targetStride, last) ||
      llvm::AddOverflow(last, width, last))
    return op.emitError("boundary geometry exceeds the supported byte range "
                        "or has overlapping rows");
  auto chunks = a.d({a.i("count", count), a.i("bytes", width),
                     a.i("ddr_stride", load ? sourceStride : targetStride),
                     a.i("spm_stride", load ? targetStride : sourceStride)});
  if (width % 64 || !ddrAligned() ||
      !spmdma::isEndpointPointerAlignmentProven(local, 64) ||
      (count > 1 && (sourceStride % 64 || targetStride % 64))) {
    // Ordinary ACC32 memory instructions have their own natural-alignment
    // contract. They copy raw words, without interpreting the logical element
    // type, and do not inherit DMA's cache-line pitch/length restriction.
    bool matrixType =
        matrixtransfer::isSupportedElementType(src.getElementType());
    StringRef matrixRejection = matrixType
                                    ? "matrix-requires-rank-two-rectangle"
                                    : "matrix-logical-element-type-unsupported";
    bool matrixLayout = matrixType && src.getRank() == 2 &&
                        matrixtransfer::getLayout(src, minor) &&
                        matrixtransfer::getLayout(dst, minor);
    if (matrixType && src.getRank() == 2 && !matrixLayout)
      matrixRejection = "matrix-physical-layout-unsupported";
    if (matrixLayout) {
      matrixRejection = "matrix-requires-exact-four-byte-rows-and-pitches";
      if (width % 4 == 0 && sourceStride % 4 == 0 && targetStride % 4 == 0) {
        matrixRejection = "matrix-effective-four-byte-alignment-unproven";
        if (ddrAligned(4) &&
            spmdma::isEndpointPointerAlignmentProven(local, 4)) {
          auto layout = [&](MemRefType type, ArrayRef<int64_t> strides,
                            int64_t offset, int64_t pitch) {
            return a.d(
                {a.b.getNamedAttr("shape",
                                  a.b.getDenseI64ArrayAttr(type.getShape())),
                 a.b.getNamedAttr("strides", a.b.getDenseI64ArrayAttr(strides)),
                 a.i("element_bits", element * 8),
                 a.i("descriptor_offset", offset),
                 a.i("contiguous_axis", minor), a.i("rows", count),
                 a.i("row_bytes", width), a.i("row_pitch", pitch),
                 a.s("origin", "effective-pointer")});
          };
          SmallVector<Attribute> tiles;
          for (int64_t row = 0; row < count; row += matrixtransfer::kMaxRows)
            for (int64_t column = 0; column < width;
                 column += matrixtransfer::kMaxRowBytes) {
              int64_t rows = std::min(matrixtransfer::kMaxRows, count - row);
              int64_t rowBytes =
                  std::min(matrixtransfer::kMaxRowBytes, width - column);
              int64_t id = tiles.size();
              tiles.push_back(a.d(
                  {a.i("id", id), a.i("predecessor", id - 1),
                   a.i("source_offset", row * sourceStride + column),
                   a.i("target_offset", row * targetStride + column),
                   a.i("rows", rows), a.i("row_bytes", rowBytes),
                   a.i("source_pitch", sourceStride),
                   a.i("target_pitch", targetStride),
                   a.i("instruction_bits", matrixtransfer::kInstructionBits),
                   a.i("register_count", matrixtransfer::kRegisterCount),
                   a.s("register_class", matrixtransfer::kRegisterClass),
                   a.i("mtype", matrixtransfer::kMtype)}));
            }
          op->setAttr(
              "plan.transfer",
              a.d({a.s("role", load ? "input" : "output"),
                   a.s("engine", "matrix"),
                   a.i("channel",
                       placement.getAs<IntegerAttr>("bank").getInt()),
                   a.s("from", load ? "ddr" : "spm"),
                   a.s("to", load ? "spm" : "ddr"),
                   a.b.getNamedAttr("chunks", chunks),
                   a.b.getNamedAttr("source_layout",
                                    layout(src, ss, so, sourceStride)),
                   a.b.getNamedAttr("target_layout",
                                    layout(dst, ds, destOffset, targetStride)),
                   a.b.getNamedAttr("tiles", a.b.getArrayAttr(tiles)),
                   a.i("total_bytes", count * width), a.i("temp_bytes", 0),
                   a.i("completion", tiles.size() - 1),
                   a.s("wait", "after-each-tile"),
                   a.s("reason", "dma-illegal-exact-acc32-rectangle")}));
          return success();
        }
      }
    }
    // The NR target currently permits ordinary vle/vse8,32,64, not 16-bit
    // vector memory instructions. A wholly DMA copy remains type agnostic.
    if (element == 2)
      return op.emitError("no legal boundary route: nr-fpga has no exact "
                          "16-bit RVV boundary-copy recipe");
    auto sourceResidue = spmdma::getEndpointByteRemainder(source);
    auto targetResidue = spmdma::getEndpointByteRemainder(target);
    // A weaker naturally aligned ABI permits RVV but cannot authorize DMA.
    if (!ddrRootAligned()) {
      if (load)
        sourceResidue.reset();
      else
        targetResidue.reset();
    }
    SmallVector<Attribute> segments;
    auto append = [&](StringRef engine, int64_t srcOffset, int64_t dstOffset,
                      int64_t length, StringRef reason,
                      int64_t vectorBytes = 0) {
      int64_t id = segments.size();
      NamedAttrList fields;
      fields.append(a.i("id", id));
      fields.append(a.i("predecessor", id - 1));
      fields.append(a.s("engine", engine));
      fields.append(a.i("source_offset", srcOffset));
      fields.append(a.i("target_offset", dstOffset));
      fields.append(a.i("bytes", length));
      fields.append(a.s("reason", reason));
      if (vectorBytes)
        fields.append(a.i("vector_bytes", vectorBytes));
      segments.push_back(fields.getDictionary(module.getContext()));
    };
    auto rvv = [&](int64_t srcOffset, int64_t dstOffset, int64_t length,
                   StringRef reason) {
      if (int64_t full = length / 64 * 64) {
        append("rvv", srcOffset, dstOffset, full, reason, 64);
        srcOffset += full;
        dstOffset += full;
        length -= full;
      }
      if (length)
        append("rvv", srcOffset, dstOffset, length, reason, length);
    };
    for (int64_t row = 0; row < count; ++row) {
      int64_t srcOffset = row * sourceStride, dstOffset = row * targetStride;
      auto sr = sourceResidue, tr = targetResidue;
      if (sr)
        *sr = (*sr + srcOffset % 64) % 64;
      if (tr)
        *tr = (*tr + dstOffset % 64) % 64;
      if (!sr || !tr || sr != tr) {
        rvv(srcOffset, dstOffset, width,
            !sr || !tr ? "dma-endpoint-residue-unproven"
                       : "dma-endpoint-residues-differ");
        continue;
      }
      int64_t head = std::min(width, (64 - *sr) % 64);
      int64_t body = (width - head) / 64 * 64;
      rvv(srcOffset, dstOffset, head, "exact-unaligned-prefix");
      if (body)
        append("dma", srcOffset + head, dstOffset + head, body,
               "both-endpoints-cache-line-aligned");
      rvv(srcOffset + head + body, dstOffset + head + body, width - head - body,
          "exact-short-suffix");
    }
    op->setAttr(
        "plan.transfer",
        a.d({a.s("role", load ? "input" : "output"), a.s("engine", "segmented"),
             a.i("channel", placement.getAs<IntegerAttr>("bank").getInt()),
             a.s("from", load ? "ddr" : "spm"), a.s("to", load ? "spm" : "ddr"),
             a.b.getNamedAttr("chunks", chunks),
             a.b.getNamedAttr("segments", a.b.getArrayAttr(segments)),
             a.i("completion", segments.size() - 1),
             a.s("wait", "after-each-segment"),
             a.s("matrix_rejection", matrixRejection),
             a.s("reason", "exact-boundary-segments")}));
    return success();
  }
  op->setAttr(
      "plan.transfer",
      a.d({a.s("role", load ? "input" : "output"), a.s("engine", "dma"),
           a.i("channel", placement.getAs<IntegerAttr>("bank").getInt()),
           a.s("from", load ? "ddr" : "spm"), a.s("to", load ? "spm" : "ddr"),
           a.b.getNamedAttr("chunks", chunks), a.s("wait", "after-each-chunk"),
           a.s("reason", "bounded-external-boundary")}));
  return success();
}

LogicalResult Planner::decisions(func::FuncOp function) {
  WalkResult status = function.walk([&](Operation *op) -> WalkResult {
    if (isStreamMember(op) || isRegisterScheduleMember(op) ||
        isFloatScheduleMember(op))
      return WalkResult::advance();
    if (auto recipe = getDDRRecipe(op);
        options.version == 2 && succeeded(recipe)) {
      op->setAttr("plan.cpu", *recipe);
      return WalkResult::advance();
    }
    if (auto copy = dyn_cast<memref::CopyOp>(op))
      return failed(transfer(copy)) ? WalkResult::interrupt()
                                    : WalkResult::advance();
    if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
      if (orderedMatmuls.contains(op)) {
        SmallVector<Attribute> operands;
        for (auto [index, value] : llvm::enumerate(op->getOperands()))
          operands.push_back(binding(value, index, "ordered-matmul-operand"));
        auto lhs = cast<MemRefType>(matmul.getInputs()[0].getType());
        auto rhs = cast<MemRefType>(matmul.getInputs()[1].getType());
        op->setAttr(
            "plan.compute",
            a.d({a.s("engine", "scalar"), a.s("kind", "matmul"),
                 a.s("order", "mnk"), a.s("fp_policy", "preserve"),
                 a.b.getNamedAttr("shape",
                                  a.b.getDenseI64ArrayAttr(
                                      {lhs.getDimSize(0), rhs.getDimSize(1),
                                       lhs.getDimSize(1)})),
                 a.b.getNamedAttr("operands", a.b.getArrayAttr(operands))}));
        return WalkResult::advance();
      }
      const Geometry &g = geometries.find(op)->second;
      op->setAttr(
          "plan.route",
          a.d({a.s("engine", "ame"),
               a.b.getNamedAttr("tile", a.d({a.i("m", g.tm), a.i("n", g.tn),
                                             a.i("k", g.tk)})),
               a.b.getNamedAttr("subtiles", a.d({a.i("n", g.n / g.tn),
                                                 a.i("k", g.k / g.tk)})),
               a.b.getNamedAttr(
                   "a", binding(op->getOperand(0), 0,
                                capacityForwarded.contains(op)
                                    ? "capacity-fallback-safe-direct"
                                    : "legal-minimum-external-traffic")),
               a.b.getNamedAttr("b", binding(op->getOperand(1), 1,
                                             "legal-minimum-external-traffic")),
               a.b.getNamedAttr("accumulator",
                                binding(op->getOperand(2), 2,
                                        privateRoot(op->getOperand(2))
                                            ? "private-accumulator"
                                            : "safe-terminal-forwarding")),
               a.b.getNamedAttr(
                   "output",
                   binding(op->getOperand(2), 2, "same-accumulator-window"))}));
      if (auto init = getZeroAccumulatorInit(matmul)) {
        NamedAttrList route(op->getAttrOfType<DictionaryAttr>("plan.route"));
        auto task = init->fill->getAttrOfType<DictionaryAttr>("contract.task");
        // The fill and consumer share one execution decision. In particular,
        // V2 may already have selected a real CPU DDR fill above; its source
        // zero proof does not authorize silently deferring that selected fill.
        auto fillPlan = init->fill->getAttrOfType<DictionaryAttr>("plan.fill");
        auto engine =
            fillPlan ? fillPlan.getAs<StringAttr>("engine") : StringAttr();
        auto kind =
            fillPlan ? fillPlan.getAs<StringAttr>("kind") : StringAttr();
        if (task && op->hasAttr("contract.task") && engine && kind &&
            engine.getValue() == "ame" && kind.getValue() == "deferred-zero") {
          route.set(
              "initialization",
              a.d({a.s("kind", init->loop ? "first-iteration-zero" : "zero"),
                   a.i("task", task.getAs<IntegerAttr>("id").getInt())}));
          op->setAttr("plan.route", route.getDictionary(module.getContext()));
        }
      }
    } else if (auto fill = dyn_cast<linalg::FillOp>(op)) {
      Value out = fill.getOutputs()[0];
      auto type = cast<MemRefType>(out.getType());
      auto placement = placements.lookup(storageRoot(out));
      if (!placement)
        return op->emitError("unplanned external fill"),
               WalkResult::interrupt();
      if (auto consumer = getZeroAccumulatorConsumer(fill);
          consumer && geometries.contains(consumer) &&
          consumer->hasAttr("contract.task") && op->hasAttr("contract.task")) {
        auto task = consumer->getAttrOfType<DictionaryAttr>("contract.task");
        op->setAttr(
            "plan.fill",
            a.d({a.s("level", "spm"), a.s("engine", "ame"),
                 a.s("kind", "deferred-zero"),
                 a.i("consumer", task.getAs<IntegerAttr>("id").getInt())}));
        return WalkResult::advance();
      }
      if (type.getRank() == 0) {
        op->setAttr("plan.fill",
                    a.d({a.s("level", "host"), a.s("engine", "scalar"),
                         a.s("kind", "init")}));
      } else {
        auto size = bytes(out);
        int64_t rows = type.getRank() == 2 ? type.getDimSize(0) : 1;
        if (!size || type.getRank() > 2)
          return op->emitError("RVV fill requires complete static rows"),
                 WalkResult::interrupt();
        int64_t vectorBytes =
            int64_t{1} << llvm::Log2_64(std::min<int64_t>(64, *size / rows));
        if (*size / rows % vectorBytes || vectorBytes < elementBytes(out))
          return op->emitError("RVV fill row has no exact vector width"),
                 WalkResult::interrupt();
        auto padding = op->getAttrOfType<DictionaryAttr>("contract.fill");
        bool isPadding = padding && padding.getAs<BoolAttr>("padding") &&
                         padding.getAs<BoolAttr>("padding").getValue();
        StringRef kind =
            isPadding ? "padding"
            : fill.getInputs()[0].getDefiningOp<arith::ConstantOp>()
                ? "overwrite"
                : "broadcast";
        op->setAttr("plan.fill",
                    a.d({a.s("level", "spm"), a.s("engine", "rvv"),
                         a.i("vector_bytes", vectorBytes), a.i("rows", rows),
                         a.i("row_bytes", *size / rows), a.s("kind", kind)}));
      }
    } else if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
      auto output = dyn_cast<MemRefType>(generic.getOutputs()[0].getType());
      if (!output || !output.hasStaticShape() || output.getRank() != 1 ||
          generic.getNumLoops() != 1 || generic.getNumParallelLoops() != 1)
        return op->emitError(
                   "v1 vector plan requires rank-one identity pointwise"),
               WalkResult::interrupt();
      SmallVector<Attribute> operands;
      SmallVector<Attribute> scalarOperands;
      SmallVector<Attribute> mathOperations;
      Block &body = generic.getRegion().front();
      for (auto [operationIndex, payload] : llvm::enumerate(body)) {
        // A generic may yield an external constant directly. Its use needs
        // the same explicit broadcast as a scalar arithmetic operand.
        if (!pointwiseMathRecipe(payload).empty() &&
            !supportedPointwiseMath(payload))
          return payload.emitError(
                     "BOSCAME math approximation requires an explicit "
                     "contract.math accuracy budget covering the recipe, "
                     "f32 operands and no fastmath flags"),
                 WalkResult::interrupt();
        if (!pointwiseMathRecipe(payload).empty())
          mathOperations.push_back(
              a.d({a.i("operation", operationIndex),
                   a.s("function", payload.getName().stripDialect()),
                   a.s("recipe", pointwiseMathRecipe(payload)),
                   a.s("evaluation", "scalar-separate-mul-add-rne"),
                   a.b.getNamedAttr(
                       "max_abs_error",
                       a.b.getF64FloatAttr(pointwiseMathAbsError(payload))),
                   a.b.getNamedAttr(
                       "max_rel_error",
                       a.b.getF64FloatAttr(pointwiseMathRelError(payload)))}));
        if (!isa<linalg::YieldOp>(payload) &&
            !supportedPointwiseOperation(payload))
          return op->emitError("unsupported lane-wise scalar expression"),
                 WalkResult::interrupt();
        for (auto [index, value] : llvm::enumerate(payload.getOperands())) {
          auto argument = dyn_cast<BlockArgument>(value);
          if ((argument && argument.getOwner() == &body) ||
              (value.getDefiningOp() &&
               value.getDefiningOp()->getBlock() == &body))
            continue;
          auto constant = value.getDefiningOp<arith::ConstantOp>();
          if (!constant || constant->getBlock() == &body ||
              !isa<IntegerAttr, FloatAttr>(constant.getValue()))
            return op->emitError("pointwise scalar capture requires a proven "
                                 "scalar constant broadcast"),
                   WalkResult::interrupt();
          NamedAttrList scalar(
              a.d({a.i("operand", index), a.s("kind", "constant-broadcast"),
                   a.b.getNamedAttr("value", constant.getValue())}));
          if (operationIndex)
            scalar.set("operation", a.b.getI64IntegerAttr(operationIndex));
          scalarOperands.push_back(scalar.getDictionary(module.getContext()));
        }
      }
      int64_t elements = output.getDimSize(0);
      int64_t lanes = pointwiseLanes(generic);
      if (elements <= 0 || elements % lanes)
        return op->emitError(
                   "pointwise plan requires complete common-lane vectors"),
               WalkResult::interrupt();
      for (auto [index, pair] : llvm::enumerate(
               llvm::zip(op->getOperands(), generic.getIndexingMapsArray()))) {
        auto [value, map] = pair;
        auto type = dyn_cast<MemRefType>(value.getType());
        if (!type && supportedPointwiseScalar(value.getType()) &&
            map.getNumResults() == 0) {
          operands.push_back(binding(value, index, "explicit-scalar-input"));
          continue;
        }
        if (!type || !type.hasStaticShape() || type.getRank() != 1 ||
            type.getDimSize(0) != elements || !map.isIdentity() ||
            !supportedPointwiseScalar(type.getElementType()))
          return op->emitError("pointwise plan requires matching contiguous "
                               "memrefs or explicit scalar inputs"),
                 WalkResult::interrupt();
        operands.push_back(binding(
            value, index,
            terminalForwarded.contains(op) &&
                    index == static_cast<size_t>(generic.getNumDpsInputs())
                ? "safe-terminal-forwarding"
                : "pointwise-memory-operand"));
      }
      op->setAttr("plan.vector",
                  a.d({a.s("engine", mathOperations.empty() ? "rvv" : "scalar"),
                       a.i("vector_bytes", mathOperations.empty() ? 64 : 4),
                       a.i("elements", elements), a.i("lanes", lanes),
                       a.s("tail", "none"),
                       a.b.getNamedAttr("operands", a.b.getArrayAttr(operands)),
                       a.s("fp_policy", mathOperations.empty()
                                            ? "preserve"
                                            : "explicit-approximation")}));
      if (!mathOperations.empty()) {
        NamedAttrList fields(op->getAttrOfType<DictionaryAttr>("plan.vector"));
        fields.set("math", a.b.getArrayAttr(mathOperations));
        op->setAttr("plan.vector", fields.getDictionary(module.getContext()));
      }
      if (!scalarOperands.empty()) {
        NamedAttrList fields(op->getAttrOfType<DictionaryAttr>("plan.vector"));
        fields.set("scalar_operands", a.b.getArrayAttr(scalarOperands));
        op->setAttr("plan.vector", fields.getDictionary(module.getContext()));
      }
    } else if (auto reduce = dyn_cast<linalg::ReduceOp>(op)) {
      Block &body = reduce.getRegion().front();
      Operation &combiner = body.front();
      bool supported =
          llvm::TypeSwitch<Operation *, bool>(&combiner)
              .Case<arith::AddFOp, arith::MaximumFOp, arith::MaxNumFOp>(
                  [](auto op) {
                    return op.getFastmath() == arith::FastMathFlags::none;
                  })
              .Default(false);
      if (!supported)
        return op->emitError("v1 reduction plan requires ordered addf, "
                             "maximumf, or maxnumf"),
               WalkResult::interrupt();
      bool isSum = isa<arith::AddFOp>(combiner);
      bool isMaximum = isa<arith::MaximumFOp>(combiner);
      Value input = reduce.getInputs()[0];
      auto type = dyn_cast<MemRefType>(input.getType());
      if (!type || type.getRank() != 1 || !type.hasStaticShape() ||
          !type.getElementType().isF32() || type.getDimSize(0) % 16)
        return op->emitError("v1 reduction plan requires full rank-one f32 "
                             "chunks"),
               WalkResult::interrupt();
      op->setAttr(
          "plan.reduce",
          a.d({a.s("engine", "rvv"), a.i("vector_bytes", isMaximum ? 4 : 64),
               a.i("elements", type.getDimSize(0)), a.i("axis", 0),
               a.s("strategy", isSum       ? "ordered-vector-add"
                               : isMaximum ? "ordered-scalar-max"
                                           : "ordered-vector-maxnum"),
               a.b.getNamedAttr("input",
                                binding(input, 0, "ordered-reduction-input")),
               a.b.getNamedAttr("output", binding(reduce.getInits()[0], 1,
                                                  "host-rank-zero")),
               a.s("fp_policy", "preserve")}));
    } else if (isa<linalg::LinalgOp>(op)) {
      return op->emitError("unsupported Linalg operation in BOSCAME plan"),
             WalkResult::interrupt();
    } else if (auto load = dyn_cast<memref::LoadOp>(op)) {
      auto type = cast<MemRefType>(load.getMemRef().getType());
      if (type.getRank() == 1) {
        auto placement = placements.lookup(storageRoot(load.getMemRef()));
        if (type.getShape() != ArrayRef<int64_t>{1} ||
            !matchPattern(load.getIndices()[0], m_Zero()) || !placement ||
            placement.getAs<StringAttr>("level").getValue() != "spm")
          return op->emitError(
                     "scalar snapshot load requires a private SPM singleton"),
                 WalkResult::interrupt();
        op->setAttr(
            "plan.scalar",
            a.d({a.s("engine", "rvv"), a.s("kind", "load"), a.i("elements", 1),
                 a.b.getNamedAttr("input", binding(load.getMemRef(), 0,
                                                   "singleton-snapshot"))}));
      }
    } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
      auto p = placements.lookup(storageRoot(store.getMemRef()));
      if (!p || p.getAs<StringAttr>("level").getValue() != "host" ||
          cast<MemRefType>(store.getMemRef().getType()).getRank() != 0)
        return op->emitError(
                   "v1 has no independent external or SPM scalar-store "
                   "recipe; use a supported Linalg output and boundary copy"),
               WalkResult::interrupt();
    } else if (auto loop = dyn_cast<scf::ForOp>(op)) {
      auto cf = op->getAttrOfType<DictionaryAttr>("contract.cf");
      if (cf && cf.getAs<IntegerAttr>("trip_count"))
        op->setAttr(
            "plan.loop",
            a.d({a.i("trips", cf.getAs<IntegerAttr>("trip_count").getInt()),
                 a.flag("spm_window_reuse", true)}));
    } else if (isa<scf::IfOp>(op)) {
      op->setAttr("plan.scope", a.d({a.s("kind", "exclusive-arms"),
                                     a.flag("share_spm_windows",
                                            sharedBranches.contains(op))}));
    }
    return WalkResult::advance();
  });
  return failure(status.wasInterrupted());
}

LogicalResult Planner::run() {
  if (options.version != 1 && options.version != 2)
    return module.emitError("BOSCAME planner supports plan-version=1 or 2");
  WalkResult modules = module.walk([&](ModuleOp nested) -> WalkResult {
    if (nested != module) {
      nested.emitError(
          "BOSCAME planning requires one module with direct "
          "function children; split nested modules before planning");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (modules.wasInterrupted())
    return failure();
  if (module->hasAttr("bosc_ame.store_policy"))
    return module.emitError(
        "bosc_ame.store_policy is obsolete; remove it and replan");
  if (module->hasAttr("plan.version"))
    return module.emitError(
        "input is already planned; run bosc-ame-plan-verify or bosc-ame-emit");
  if (failed(::buddy::contract::verifyKernelContract(module)))
    return failure();
  for (const auto &window : fixedInputs) {
    auto *definition = window.root ? window.root.getDefiningOp() : nullptr;
    auto footprint = definition ? storageBytes(window.root) : std::nullopt;
    if (!definition || !isa<memref::AllocOp, memref::AllocaOp>(definition) ||
        definition->getParentOfType<ModuleOp>() != module || !footprint ||
        window.bank < 0 || window.bank >= 2 || window.offset < 0 ||
        window.offset % 64 || window.capacity < *footprint ||
        window.capacity > 131072 || window.offset > 131072 - window.capacity ||
        !fixedBuffers.try_emplace(window.root, window).second)
      return module.emitError(
          "invalid fixed SPM window from target input adapter");
  }
  for (auto [i, first] : llvm::enumerate(fixedInputs))
    for (const auto &second : fixedInputs.drop_front(i + 1))
      if (first.bank == second.bank &&
          first.root.getDefiningOp()->getParentOfType<func::FuncOp>() ==
              second.root.getDefiningOp()->getParentOfType<func::FuncOp>() &&
          first.offset < second.offset + second.capacity &&
          second.offset < first.offset + first.capacity &&
          !canShareExclusiveSPMWindow(first.root, second.root))
        return module.emitError("fixed SPM windows overlap without an "
                                "exclusive, non-escaping lifetime proof");
  module.walk([&](func::FuncOp function) {
    function.walk([&](Operation *op) {
      if (auto task = op->getAttrOfType<DictionaryAttr>("contract.task"))
        oldTasks[function].push_back(
            {task.getAs<IntegerAttr>("id").getInt(), op});
    });
  });
  // Fill is independent of logical axis order. Present a contiguous physical
  // row to the existing RVV recipe, including column-major B padding, without
  // changing the root, covered bytes or the scalar's bit pattern.
  module.walk([&](linalg::FillOp fill) {
    auto type = dyn_cast<MemRefType>(fill.getOutputs()[0].getType());
    SmallVector<int64_t> strides;
    int64_t offset;
    if (!type || type.getRank() != 2 ||
        failed(type.getStridesAndOffset(strides, offset)) || strides[0] != 1 ||
        strides[1] == 1)
      return;
    OpBuilder builder(fill);
    Value transposed = memref::TransposeOp::create(
        builder, fill.getLoc(), fill.getOutputs()[0],
        AffineMapAttr::get(AffineMap::getPermutationMap(
            ArrayRef<unsigned>{1, 0}, module.getContext())));
    fill.getOutputsMutable()[0].set(transposed);
  });
  SmallVector<linalg::MatmulOp> matmuls;
  module.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
  for (auto op : matmuls)
    (void)fuseAccumulator(op);
  SmallVector<Value> fixedRoots;
  for (const auto &fixed : fixedInputs)
    fixedRoots.push_back(fixed.root);
  if (options.version == 2 && failed(analyzeFloatSchedules(module, fixedRoots)))
    return failure();
  // Match the complete integer reduction and its terminal rank-two epilogue
  // before flattening can hide that ownership relationship.
  if (options.version == 2 &&
      failed(analyzeRegisterSchedules(module, fixedRoots)))
    return failure();
  flattenPointwise();
  for (auto op : matmuls) {
    if (isRegisterScheduleMember(op) || isFloatScheduleMember(op))
      continue;
    auto lhs = dyn_cast<MemRefType>(op.getInputs()[0].getType());
    auto rhs = dyn_cast<MemRefType>(op.getInputs()[1].getType());
    auto output = dyn_cast<MemRefType>(op.getOutputs()[0].getType());
    if (!lhs || !rhs || !output || !lhs.hasStaticShape() ||
        !rhs.hasStaticShape() || !output.hasStaticShape() ||
        lhs.getRank() != 2 || rhs.getRank() != 2 || output.getRank() != 2 ||
        llvm::any_of(lhs.getShape(), [](int64_t dim) { return dim <= 0; }) ||
        llvm::any_of(rhs.getShape(), [](int64_t dim) { return dim <= 0; }))
      return op.emitError(
          "matmul planning requires positive static rank-two buffers");
    bool ame = lhs.getElementType().isInteger(8) &&
               rhs.getElementType().isInteger(8) &&
               output.getElementType().isInteger(32) &&
               op.getCast() == linalg::TypeFn::cast_signed;
    if (!ame || orderedReference(op)) {
      orderedMatmuls.insert(op);
      continue;
    }
    auto g = geometry(op);
    if (failed(g))
      return failure();
    geometries.try_emplace(op, *g);
    if (failed(forwardSnapshots(op, *g)) ||
        failed(stageExternalOperands(op, *g)))
      return failure();
    for (unsigned role = 0; role != 3; ++role)
      if (privateRoot(op->getOperand(role)))
        preferredBank[storageRoot(op->getOperand(role))] = role == 2 ? 0 : 1;
  }
  forwardTerminalStores();
  if (options.version == 1) {
    // Capacity is a legality constraint, not a reason to demand a user policy.
    // Retry one proven direct alternative at a time in source order. A may read
    // more DDR bytes directly across N subtiles, but avoids its staging window.
    for (;;) {
      bool fits = true;
      for (auto function : module.getOps<func::FuncOp>()) {
        if (function.isExternal())
          continue;
        auto cursors = reserved(function);
        std::array<int64_t, 2> highWater = cursors;
        if (failed(placeBlock(function.getBody().front(), cursors, highWater,
                              false))) {
          fits = false;
          break;
        }
      }
      if (fits)
        break;
      bool changed = false;
      for (auto op : matmuls) {
        if (orderedMatmuls.contains(op))
          continue;
        if (failed(forwardSnapshots(op, geometries.find(op)->second, true,
                                    &changed)))
          return failure();
        if (changed)
          break;
      }
      if (!changed) {
        for (auto function : module.getOps<func::FuncOp>()) {
          if (function.isExternal())
            continue;
          auto cursors = reserved(function);
          std::array<int64_t, 2> highWater = cursors;
          if (failed(
                  placeBlock(function.getBody().front(), cursors, highWater)))
            return failure();
        }
        return module.emitError(
            "no legal storage route satisfies the SPM workspace capacity");
      }
    }
  }
  StringRef producer = "buddy-neutral";
  if (auto attr = module->getAttrOfType<StringAttr>("contract.producer"))
    producer = attr.getValue();
  if (failed(::buddy::contract::annotateKernelContract(module, producer, true)))
    return failure();
  if (options.version == 2 &&
      failed(analyzeRegisterSchedules(module, fixedRoots)))
    return failure();
  if (options.version == 2 &&
      (failed(analyzeFloatSchedules(module, fixedRoots)) ||
       failed(analyzeStreams(module, fixedRoots))))
    return failure();
  module->setAttr("plan.version", a.b.getI32IntegerAttr(options.version));
  module->setAttr("plan.target_profile", a.b.getStringAttr("nr-fpga"));
  if (options.version == 2) {
    module->setAttr("plan.automatic_spm",
                    a.b.getBoolAttr(options.automaticPlacement));
    // Preserve explicit physical-input constraints independently of the
    // chosen placement, so verification reconstructs all reservations first.
    for (const auto &entry : fixedBuffers) {
      const auto &window = entry.second;
      window.root.getDefiningOp()->setAttr(
          "plan.fixed_input",
          a.d({a.i("bank", window.bank), a.i("offset", window.offset),
               a.i("capacity", window.capacity)}));
    }
  }
  for (auto function : module.getOps<func::FuncOp>()) {
    if (function.isExternal())
      continue;
    auto cursors = reserved(function);
    std::array<int64_t, 2> highWater = cursors;
    if (failed(placeBlock(function.getBody().front(), cursors, highWater)))
      return failure();
    function->setAttr(
        "plan.budget",
        a.d({a.i("spm_banks", 2), a.i("bank_capacity", 131072),
             a.b.getNamedAttr("high_water",
                              a.b.getDenseI64ArrayAttr(highWater))}));
    SmallVector<Attribute> mapping;
    for (const OldTask &old : oldTasks[function]) {
      if (old.action == "eliminated") {
        mapping.push_back(
            a.d({a.i("old_id", old.id), a.s("action", old.action)}));
        continue;
      }
      auto task = old.operation->getAttrOfType<DictionaryAttr>("contract.task");
      if (!task)
        return function.emitError(
            "storage rewrite lost a source task's physical completion mapping");
      mapping.push_back(
          a.d({a.i("old_id", old.id),
               a.i("new_id", task.getAs<IntegerAttr>("id").getInt()),
               a.s("action", old.action)}));
    }
    function->setAttr("plan.source_task_count",
                      a.b.getI64IntegerAttr(oldTasks[function].size()));
    function->setAttr("plan.task_map", a.b.getArrayAttr(mapping));
    if (failed(decisions(function)))
      return failure();
  }
  return verifyBOSCAMEPlan(module);
}

class BOSCAMEPlanPass
    : public PassWrapper<BOSCAMEPlanPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BOSCAMEPlanPass)
  BOSCAMEPlanPass() = default;
  BOSCAMEPlanPass(const BOSCAMEPlanPass &) {}
  Option<bool> report{
      *this, "plan-report",
      llvm::cl::desc("Print explicit access decisions and workspace"),
      llvm::cl::init(false)};
  Option<int64_t> version{
      *this, "plan-version",
      llvm::cl::desc(
          "Plan version (1: frozen compatibility, 2: DDR/register default)"),
      llvm::cl::init(2)};
  StringRef getArgument() const override { return "bosc-ame-plan"; }
  StringRef getDescription() const override {
    return "Plan neutral kernel contracts into explicit BOSCAME storage routes";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    if (failed(planBOSCAME(getOperation(), {}, PlanOptions{true, version}))) {
      signalPassFailure();
      return;
    }
    if (report)
      printBOSCAMEPlan(getOperation(), llvm::errs());
  }
};

} // namespace

LogicalResult planBOSCAME(ModuleOp module,
                          ArrayRef<FixedSPMBuffer> fixedBuffers,
                          PlanOptions options) {
  return Planner(module, fixedBuffers, options).run();
}

void printBOSCAMEPlan(ModuleOp module, llvm::raw_ostream &stream) {
  module.walk([&](Operation *op) {
    for (StringRef attr :
         {"plan.budget", "plan.route", "plan.compute", "plan.transfer",
          "plan.buffer", "plan.vector", "plan.reduce", "plan.fill",
          "plan.scalar", "plan.cpu", "plan.stream", "plan.register_schedule",
          "plan.float_schedule"})
      if (Attribute decision = op->getAttr(attr))
        stream << op->getName() << " " << attr << "=" << decision << "\n";
  });
}

std::unique_ptr<Pass> createBOSCAMEPlanPass() {
  return std::make_unique<BOSCAMEPlanPass>();
}

void registerBOSCAMEPlanPasses() { PassRegistration<BOSCAMEPlanPass>(); }

} // namespace buddy::boscame
