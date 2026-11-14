//===- KernelContractCore.cpp - Buffer Contract annotation and verification core
//-===//
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
// Core implementation of the neutral Kernel Contract ABI (contract.* attrs).
// Public wrappers live in KernelContract.cpp; planners call the detail::
// APIs here through that layer with a shared SourcePolicy.
//
// Responsibilities
// ----------------
// * Normalize producer IR enough for a closed graph (explicit scalar captures,
//   reaching-zero copy folding) before assigning content versions.
// * Walk each func.func and build / verify per-scope task graphs: buffers,
//   boundary accesses, compute/fill tasks, loop-carried slots, and branch
//   exclusivity (scf.for / scf.if).
// * Enforce bidirectional consistency between declared contract.* facts and
//   actual memref roots, byte extents, effects, and program order.
// * Keep logical memory space at zero. Target placement (banks, DMA, plan.*)
//   is out of scope here; planned mode only accepts those attrs without
//   relaxing version or alias rules.
//
// SourcePolicy.isSyntheticTransfer marks Buddy-inserted staging copies. Such
// ops cannot supply a missing frontend boundary event during annotate/verify.
//
// Entry points at the bottom of this file:
//   annotateKernelContract / verifyKernelContract
//   rebuildPlannedContract / verifyPlannedContract
//   hasEntryArgumentABI / checkKernelContractVersions
//
//===----------------------------------------------------------------------===//

#include "Analysis/detail/KernelContractCore.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

using namespace mlir;

namespace buddy::contract::detail {
namespace {

//===----------------------------------------------------------------------===//
// Storage identity and producer normalization
//===----------------------------------------------------------------------===//

// Follow view/cast chains to the alloc/arg that owns the underlying storage.
static Value storageRoot(Value value) {
  llvm::SmallDenseSet<Value> seen;
  while (value && seen.insert(value).second) {
    Operation *def = value.getDefiningOp();
    if (!def)
      return value;
    if (auto view = dyn_cast<ViewLikeOpInterface>(def)) {
      value = view.getViewSource();
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    break;
  }
  return value;
}

static bool isPrivate(Value root) {
  return root && root.getDefiningOp() &&
         isa<memref::AllocOp, memref::AllocaOp>(root.getDefiningOp());
}

// Linalg fusion can turn a tensor splat into a scalar region capture. Lift
// dynamic captures to explicit operands before collecting task dependencies
// so the contract graph sees every real SSA input.
static void makeScalarInputsExplicit(ModuleOp module) {
  SmallVector<linalg::GenericOp> generics;
  module.walk([&](linalg::GenericOp generic) { generics.push_back(generic); });
  for (linalg::GenericOp generic : generics) {
    Block &body = generic.getRegion().front();
    llvm::SetVector<Value> captures;
    for (Operation &operation : body)
      for (Value value : operation.getOperands()) {
        auto argument = dyn_cast<BlockArgument>(value);
        if (!value.getType().isIntOrFloat() ||
            (argument && argument.getOwner() == &body) ||
            (value.getDefiningOp() &&
             value.getDefiningOp()->getBlock() == &body) ||
            value.getDefiningOp<arith::ConstantOp>())
          continue;
        captures.insert(value);
      }
    if (captures.empty())
      continue;
    OpBuilder builder(generic);
    SmallVector<Value> inputs(generic.getInputs());
    llvm::append_range(inputs, captures);
    auto maps = generic.getIndexingMapsArray();
    auto scalarMap =
        AffineMap::get(generic.getNumLoops(), 0, {}, module.getContext());
    maps.insert(maps.begin() + generic.getNumDpsInputs(), captures.size(),
                scalarMap);
    auto replacement = linalg::GenericOp::create(
        builder, generic.getLoc(), generic.getResultTypes(), inputs,
        generic.getOutputs(), maps, generic.getIteratorTypesArray(),
        [&](OpBuilder &nested, Location loc, ValueRange args) {
          IRMapping mapping;
          for (unsigned i = 0; i < generic.getNumDpsInputs(); ++i)
            mapping.map(body.getArgument(i), args[i]);
          for (unsigned i = 0; i < generic.getNumDpsInits(); ++i)
            mapping.map(body.getArgument(generic.getNumDpsInputs() + i),
                        args[inputs.size() + i]);
          for (auto [i, capture] : llvm::enumerate(captures))
            mapping.map(capture, args[generic.getNumDpsInputs() + i]);
          for (Operation &operation : body)
            nested.clone(operation, mapping);
        });
    for (NamedAttribute attr : generic->getDiscardableAttrs())
      if (attr.getName().strref() != "contract.task")
        replacement->setAttr(attr.getName(), attr.getValue());
    generic.replaceAllUsesWith(replacement.getResults());
    generic.erase();
  }
}

// Rematerialize an exact reaching zero before assigning content versions.
// The complete private source has one dominating initializer and no alias
// can modify or escape before any execution of the copy. A write may follow
// the copy's whole enclosing scope in the allocation's block: that scope
// includes every repeated iteration sharing this allocation. Merely following
// the copy inside its loop is insufficient, even through another nested loop.
static void normalizeReachingZeroCopies(ModuleOp module) {
  DominanceInfo dominance(module);
  SmallVector<memref::CopyOp> copies;
  module.walk([&](memref::CopyOp copy) { copies.push_back(copy); });
  for (memref::CopyOp copy : copies) {
    if (copy->hasAttr("contract.access"))
      continue;
    Value source = copy.getSource(), target = copy.getTarget();
    auto sourceType = dyn_cast<MemRefType>(source.getType());
    auto targetType = dyn_cast<MemRefType>(target.getType());
    if (!sourceType || !targetType || !sourceType.hasStaticShape() ||
        sourceType.getShape() != targetType.getShape() ||
        sourceType.getElementType() != targetType.getElementType() ||
        !isPrivate(source) || !isPrivate(target) || source == target)
      continue;
    Block *allocationBlock = source.getDefiningOp()->getBlock();
    Operation *readScope = copy;
    while (readScope && readScope->getBlock() != allocationBlock)
      readScope = readScope->getParentOp();
    if (!readScope)
      continue;
    auto afterReadLifetime = [&](Operation *operation) {
      // Lift both accesses to the allocation's dynamic instance. In
      // particular, an inner-loop copy cannot ignore a subsequent write in
      // the same outer loop: its next iteration still shares this storage.
      while (operation && operation->getBlock() != allocationBlock)
        operation = operation->getParentOp();
      return operation && operation != readScope &&
             readScope->isBeforeInBlock(operation);
    };
    SmallVector<Value> aliases{source};
    llvm::SmallDenseSet<Value> seen;
    linalg::FillOp initializer;
    bool immutable = true;
    while (!aliases.empty() && immutable) {
      Value alias = aliases.pop_back_val();
      if (!seen.insert(alias).second)
        continue;
      for (OpOperand &use : alias.getUses()) {
        Operation *owner = use.getOwner();
        if (afterReadLifetime(owner))
          continue;
        if (auto view = dyn_cast<ViewLikeOpInterface>(owner)) {
          if (view.getViewSource() != alias || owner->getNumResults() != 1) {
            immutable = false;
            break;
          }
          aliases.push_back(owner->getResult(0));
          continue;
        }
        if (auto fill = dyn_cast<linalg::FillOp>(owner)) {
          if (alias != source || fill.getOutputs()[0] != source ||
              initializer) {
            immutable = false;
            break;
          }
          Attribute value;
          if (!matchPattern(fill.getInputs()[0], m_Constant(&value)) ||
              !(isa<IntegerAttr>(value)
                    ? cast<IntegerAttr>(value).getValue().isZero()
                    : isa<FloatAttr>(value) &&
                          cast<FloatAttr>(value).getValue().isZero())) {
            immutable = false;
            break;
          }
          initializer = fill;
          continue;
        }
        if (isa<memref::LoadOp, memref::DimOp, memref::DeallocOp>(owner))
          continue;
        if (auto read = dyn_cast<memref::CopyOp>(owner);
            read && read.getSource() == alias && read.getTarget() != alias)
          continue;
        if (auto compute = dyn_cast<linalg::LinalgOp>(owner);
            compute && use.getOperandNumber() < compute.getNumDpsInputs())
          continue;
        immutable = false;
        break;
      }
    }
    if (!immutable || !initializer ||
        !dominance.dominates(initializer.getOperation(), copy.getOperation()))
      continue;
    // Unknown effects intentionally invalidate private content in contract
    // v1. Do not bypass that rule merely because the opaque operation has no
    // explicit SSA use of this allocation (including in another iteration).
    auto function = source.getDefiningOp()->getParentOfType<func::FuncOp>();
    if (!function)
      continue;
    WalkResult knownEffects = function.walk([&](Operation *operation) {
      if (isa<func::FuncOp, scf::ForOp, scf::IfOp>(operation) ||
          isMemoryEffectFree(operation))
        return WalkResult::advance();
      auto effect = dyn_cast<MemoryEffectOpInterface>(operation);
      if (!effect)
        return WalkResult::interrupt();
      SmallVector<MemoryEffects::EffectInstance> effects;
      effect.getEffects(effects);
      for (const auto &instance : effects)
        if (isa<MemoryEffects::Write, MemoryEffects::Free>(
                instance.getEffect()) &&
            !instance.getValue())
          return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (knownEffects.wasInterrupted())
      continue;
    OpBuilder builder(copy);
    // Reuse the original scalar, including the sign bit of floating zero.
    auto fill = linalg::FillOp::create(
        builder, copy.getLoc(), initializer.getInputs(), ValueRange{target});
    for (NamedAttribute attr : copy->getAttrs())
      if (attr.getName().strref() != "contract.task")
        fill->setAttr(attr.getName(), attr.getValue());
    copy.erase();
    bool unused = llvm::all_of(source.getUsers(), [&](Operation *user) {
      return user == initializer || isa<memref::DeallocOp>(user);
    });
    if (unused) {
      SmallVector<Operation *> users(source.getUsers());
      for (Operation *user : users)
        user->erase();
      source.getDefiningOp()->erase();
    }
  }
}

// Most view operations preserve the aligned pointer, while memref.view moves
// it by byte_shift. Descriptor offsets are in elements of the result type and
// must be added separately (reinterpret_cast replaces the descriptor offset).
static std::optional<int64_t> alignedByteBase(Value value) {
  Operation *op = value.getDefiningOp();
  if (!op)
    return int64_t(0);
  auto view = dyn_cast<ViewLikeOpInterface>(op);
  if (!view)
    return int64_t(0);
  auto base = alignedByteBase(view.getViewSource());
  if (!base)
    return std::nullopt;
  if (auto byteView = dyn_cast<memref::ViewOp>(op)) {
    auto shift = getConstantIntValue(byteView.getByteShift());
    if (!shift || llvm::AddOverflow(*base, *shift, *base))
      return std::nullopt;
  }
  return base;
}

struct ByteDescriptor {
  int64_t offset;
  SmallVector<int64_t> shape, strides;
  int64_t elementBytes;
};

static std::optional<ByteDescriptor> byteDescriptor(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  auto base = alignedByteBase(value);
  if (!type || !type.hasStaticShape() || !base ||
      !type.getElementType().isIntOrFloat())
    return std::nullopt;
  ByteDescriptor descriptor;
  if (failed(type.getStridesAndOffset(descriptor.strides, descriptor.offset)))
    return std::nullopt;
  if (auto cast = value.getDefiningOp<memref::ReinterpretCastOp>()) {
    auto offset = getConstantIntValue(cast.getMixedOffsets().front());
    if (offset)
      descriptor.offset = *offset;
    for (auto [index, stride] : llvm::enumerate(cast.getMixedStrides()))
      if (auto constant = getConstantIntValue(stride))
        descriptor.strides[index] = *constant;
  }
  if (ShapedType::isDynamic(descriptor.offset) ||
      llvm::any_of(descriptor.strides, [](int64_t stride) {
        return ShapedType::isDynamic(stride) || stride < 0;
      }))
    return std::nullopt;
  descriptor.shape = llvm::to_vector(type.getShape());
  descriptor.elementBytes = (type.getElementTypeBitWidth() + 7) / 8;
  if (llvm::MulOverflow(descriptor.offset, descriptor.elementBytes,
                        descriptor.offset) ||
      llvm::AddOverflow(descriptor.offset, *base, descriptor.offset))
    return std::nullopt;
  for (int64_t &stride : descriptor.strides)
    if (llvm::MulOverflow(stride, descriptor.elementBytes, stride))
      return std::nullopt;
  return descriptor;
}

static std::optional<int64_t> integer(DictionaryAttr dict, StringRef key) {
  if (auto attr = dict ? dict.getAs<IntegerAttr>(key) : IntegerAttr())
    return attr.getInt();
  return std::nullopt;
}

static bool boolean(DictionaryAttr dict, StringRef key, bool fallback = false) {
  if (auto attr = dict ? dict.getAs<BoolAttr>(key) : BoolAttr())
    return attr.getValue();
  return fallback;
}

static StringRef string(DictionaryAttr dict, StringRef key) {
  if (auto attr = dict ? dict.getAs<StringAttr>(key) : StringAttr())
    return attr.getValue();
  return {};
}

static SmallVector<int64_t> numbers(DictionaryAttr dict, StringRef key) {
  if (auto attr =
          dict ? dict.getAs<DenseI64ArrayAttr>(key) : DenseI64ArrayAttr())
    return llvm::to_vector(attr.asArrayRef());
  return {};
}

static bool referenceOrderedMatmul(Operation *op) {
  auto compute = op->getAttrOfType<DictionaryAttr>("contract.compute");
  return isa<linalg::MatmulOp>(op) && string(compute, "kind") == "matmul" &&
         string(compute, "order") == "mnk";
}

// Walk both attribute/type subelements and all regions, including attributes on
// function arguments. A target attribute hidden in a payload or dictionary is
// just as invalid as one on a task. Planned IR admits plan.* only.
//===----------------------------------------------------------------------===//
// Module / dialect representation guards
//===----------------------------------------------------------------------===//

// Reject unknown dialects, illegal memory spaces, and (unless planned) any
// plan.* or bosc_ame.* leakage into the neutral contract surface.
static LogicalResult checkRepresentation(ModuleOp module, bool planned) {
  if (Attribute attribute = module->getAttr("contract.abi")) {
    auto abi = dyn_cast<DictionaryAttr>(attribute);
    if (!abi)
      return module.emitOpError("kernel contract V2: ABI must be a dictionary");
    if (Attribute alignment = abi.get("args_aligned_bytes")) {
      auto value = dyn_cast<IntegerAttr>(alignment);
      if (!value || value.getInt() <= 0)
        return module.emitOpError(
            "kernel contract V2: ABI alignment must be a positive integer");
    }
    if (abi.get("args_disjoint") && !abi.getAs<BoolAttr>("args_disjoint"))
      return module.emitOpError(
          "kernel contract V2: ABI disjointness must be boolean");
  }
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    auto fail = [&](StringRef rule, const Twine &message) -> WalkResult {
      op->emitOpError() << "kernel contract " << rule << ": " << message;
      return WalkResult::interrupt();
    };
    if (Attribute declared = op->getAttr("contract.math")) {
      auto math = dyn_cast<DictionaryAttr>(declared);
      if (!isa<func::FuncOp>(op) || !math || math.empty())
        return fail("V26", "contract.math requires a nonempty function "
                           "accuracy dictionary");
      for (NamedAttribute entry : math) {
        auto accuracy = dyn_cast<DictionaryAttr>(entry.getValue());
        auto abs =
            accuracy ? accuracy.getAs<FloatAttr>("max_abs_error") : FloatAttr();
        auto rel =
            accuracy ? accuracy.getAs<FloatAttr>("max_rel_error") : FloatAttr();
        if ((entry.getName() != "erf" && entry.getName() != "exp") ||
            !accuracy || accuracy.size() != 4 ||
            string(accuracy, "domain") != "all-f32" ||
            string(accuracy, "special_values") != "ieee" || !abs || !rel ||
            !abs.getType().isF64() || !rel.getType().isF64() ||
            !std::isfinite(abs.getValueAsDouble()) ||
            !std::isfinite(rel.getValueAsDouble()) ||
            abs.getValueAsDouble() < 0 || rel.getValueAsDouble() < 0)
          return fail("V26", "math accuracy requires erf/exp with domain="
                             "all-f32, special_values=ieee and finite "
                             "nonnegative f64 absolute/relative budgets");
      }
    }
    AttrTypeWalker walker;
    walker.addWalk([&](DictionaryAttr dictionary) -> WalkResult {
      for (NamedAttribute attr : dictionary) {
        StringRef key = attr.getName().strref();
        if (key == "contract.math" &&
            (!isa<func::FuncOp>(op) || dictionary != op->getAttrDictionary()))
          return fail("V26", "math accuracy must be attached directly to a "
                             "function");
        if (key.starts_with("bosc_ame.") ||
            (!planned && key.starts_with("plan.")))
          return fail(
              "V14", "target attributes are forbidden in the neutral contract");
        if (key == "origin_id" || key.ends_with("_origin_id")) {
          auto id = dyn_cast<IntegerAttr>(attr.getValue());
          if (!id || id.getInt() < 0)
            return fail("V12", "origin IDs must be nonnegative integers");
        }
        if (key.ends_with("_origin_ids")) {
          auto ids = dyn_cast<DenseI64ArrayAttr>(attr.getValue());
          if (!ids ||
              llvm::any_of(ids.asArrayRef(), [](int64_t id) { return id < 0; }))
            return fail("V12",
                        "source origin IDs must be nonnegative i64 arrays");
        }
        if (key.ends_with("_lineage_complete")) {
          auto complete = dyn_cast<BoolAttr>(attr.getValue());
          if (!complete)
            return fail("V12", "lineage completeness must be boolean");
          if (complete.getValue()) {
            std::string prefix =
                key.drop_back(StringRef("_lineage_complete").size()).str();
            // source_lineage_complete in compute_graph is an aggregate fact,
            // whereas provenance source completeness describes an ID list.
            if (dictionary.get("root_scope"))
              continue;
            auto ids = dictionary.getAs<DenseI64ArrayAttr>(prefix +
                                                           "_load_origin_ids");
            if (!ids)
              ids = dictionary.getAs<DenseI64ArrayAttr>(prefix + "_origin_ids");
            if (!ids || ids.empty())
              return fail(
                  "V12",
                  "complete provenance requires a nonempty source origin list");
          }
        }
      }
      return WalkResult::advance();
    });
    walker.addWalk([&](Attribute attr) -> WalkResult {
      StringRef dialect = attr.getDialect().getNamespace();
      if (dialect == "bosc_ame" || dialect == "tt" || dialect == "tts" ||
          dialect == "tptr" || (!planned && dialect == "plan"))
        return fail("V14",
                    "target attributes are forbidden in the neutral contract");
      return WalkResult::advance();
    });
    walker.addWalk([&](Type type) -> WalkResult {
      if (isa<TensorType>(type))
        return fail("V3", "contract must be buffer form, without tensor types");
      if (type.getDialect().getNamespace() != "builtin")
        return fail("V3/V14", "nonstandard pointer or target types are "
                              "forbidden in the contract");
      Attribute memorySpace;
      if (auto memref = dyn_cast<BaseMemRefType>(type))
        memorySpace = memref.getMemorySpace();
      if (memorySpace && (!isa<IntegerAttr>(memorySpace) ||
                          cast<IntegerAttr>(memorySpace).getInt() != 0))
        return fail(
            "V3/V14",
            "target memory spaces are forbidden in the neutral contract");
      return WalkResult::advance();
    });
    for (NamedAttribute attr : op->getAttrs()) {
      StringRef name = attr.getName().strref();
      if (name.starts_with("bosc_ame.") ||
          (!planned && name.starts_with("plan.")))
        return fail("V14",
                    "target attributes are forbidden in the neutral contract");
      if (walker.walk(attr.getValue()).wasInterrupted())
        return WalkResult::interrupt();
    }
    for (Type type : op->getOperandTypes())
      if (walker.walk(type).wasInterrupted())
        return WalkResult::interrupt();
    for (Type type : op->getResultTypes())
      if (walker.walk(type).wasInterrupted())
        return WalkResult::interrupt();
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          if (walker.walk(argument.getType()).wasInterrupted())
            return WalkResult::interrupt();
    StringRef dialect = op->getName().getDialectNamespace();
    if (dialect != "builtin" && dialect != "affine" && dialect != "arith" &&
        dialect != "cf" && dialect != "complex" && dialect != "func" &&
        dialect != "index" && dialect != "linalg" && dialect != "math" &&
        dialect != "memref" && dialect != "scf" && dialect != "ub" &&
        dialect != "vector")
      return fail("V3/V14", "only standard buffer operations are permitted");
    if (isa<UnrealizedConversionCastOp>(op))
      return fail("V3", "unrealized casts are forbidden");
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

static bool uniformCondition(Value value,
                             llvm::SmallPtrSetImpl<Operation *> &seen) {
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
    return function &&
           function.getArgAttr(argument.getArgNumber(), "contract.program");
  }
  Operation *def = value.getDefiningOp();
  if (!def || !isMemoryEffectFree(def) || def->getNumRegions() != 0)
    return false;
  if (!seen.insert(def).second)
    return true;
  return llvm::all_of(def->getOperands(), [&](Value operand) {
    return !isa<ShapedType>(operand.getType()) &&
           uniformCondition(operand, seen);
  });
}

static std::optional<int64_t> denseBytes(const ByteDescriptor &descriptor) {
  SmallVector<std::pair<int64_t, int64_t>> dims;
  for (auto [size, stride] : llvm::zip(descriptor.shape, descriptor.strides)) {
    if (size <= 0)
      return std::nullopt;
    if (size > 1)
      dims.emplace_back(stride, size);
  }
  llvm::sort(dims);
  int64_t pitch = descriptor.elementBytes;
  for (auto [stride, size] : dims) {
    if (stride != pitch || llvm::MulOverflow(pitch, size, pitch))
      return std::nullopt;
  }
  return pitch;
}

static bool fullRoot(Value view) {
  Value root = storageRoot(view);
  if (view == root)
    return true;
  auto descriptor = byteDescriptor(view), rootDescriptor = byteDescriptor(root);
  if (!descriptor || !rootDescriptor ||
      descriptor->offset != rootDescriptor->offset)
    return false;
  auto size = denseBytes(*descriptor), rootSize = denseBytes(*rootDescriptor);
  return size && rootSize && *size == *rootSize;
}

struct Bounds {
  int64_t begin;
  int64_t end;
};

static std::optional<Bounds> bounds(Value value) {
  auto descriptor = byteDescriptor(value);
  if (!descriptor)
    return std::nullopt;
  int64_t end = descriptor->offset;
  for (auto [size, stride] :
       llvm::zip(descriptor->shape, descriptor->strides)) {
    int64_t delta;
    if (size <= 0 || llvm::MulOverflow(size - 1, stride, delta) ||
        llvm::AddOverflow(end, delta, end))
      return std::nullopt;
  }
  if (llvm::AddOverflow(end, descriptor->elementBytes, end))
    return std::nullopt;
  return Bounds{descriptor->offset, end};
}

static bool sameAddresses(Value a, Value b) {
  if (a == b)
    return true;
  if (storageRoot(a) != storageRoot(b))
    return false;
  auto ad = byteDescriptor(a), bd = byteDescriptor(b);
  return ad && bd && ad->shape == bd->shape && ad->offset == bd->offset &&
         ad->elementBytes == bd->elementBytes && ad->strides == bd->strides;
}

struct Effects {
  SmallVector<Value> reads;
  SmallVector<Value> writes;
  SmallVector<Value> scalars;
  StringRef kind;
  bool barrier = false;
};

static std::optional<Effects> effects(Operation *op) {
  Effects result;
  if (auto copy = dyn_cast<memref::CopyOp>(op)) {
    result.reads.push_back(copy.getSource());
    result.writes.push_back(copy.getTarget());
    result.kind =
        isPrivate(storageRoot(copy.getTarget())) ? "copy" : "memory-store";
    if (!isPrivate(storageRoot(copy.getSource())))
      result.kind = "memory-load";
    if (auto access = op->getAttrOfType<DictionaryAttr>("contract.access"))
      result.kind =
          string(access, "kind") == "store" ? "memory-store" : "memory-load";
    result.barrier = result.kind != "copy";
    return result;
  }
  if (auto load = dyn_cast<memref::LoadOp>(op)) {
    result.reads.push_back(load.getMemref());
    result.kind = "scalar-load";
    return result;
  }
  if (auto store = dyn_cast<memref::StoreOp>(op)) {
    result.scalars.push_back(store.getValue());
    result.writes.push_back(store.getMemref());
    result.kind = "memory-store";
    result.barrier = true;
    return result;
  }
  auto linalg = dyn_cast<linalg::LinalgOp>(op);
  if (!linalg || linalg.hasPureTensorSemantics())
    return std::nullopt;
  if (isa<linalg::FillOp>(op)) {
    result.kind = "fill";
  } else if (isa<linalg::MatmulOp>(op)) {
    result.kind = "dot";
    result.barrier = true;
  } else if (isa<linalg::ReduceOp>(op)) {
    result.kind = "reduce";
    result.barrier = true;
  } else if (isa<linalg::GenericOp>(op)) {
    result.kind = "elementwise";
  } else {
    return std::nullopt;
  }
  Block &payload = linalg->getRegion(0).front();
  unsigned index = 0;
  for (Value input : linalg.getDpsInputs()) {
    if (isa<BaseMemRefType>(input.getType())) {
      if (!isa<linalg::GenericOp>(op) ||
          !payload.getArgument(index).use_empty())
        result.reads.push_back(input);
    } else {
      result.scalars.push_back(input);
    }
    ++index;
  }
  for (Value output : linalg.getDpsInits()) {
    if (!isa<BaseMemRefType>(output.getType()))
      return std::nullopt;
    result.writes.push_back(output);
    if (isa<linalg::MatmulOp, linalg::ReduceOp>(op) ||
        (isa<linalg::GenericOp>(op) && !payload.getArgument(index).use_empty()))
      result.reads.push_back(output);
    ++index;
  }
  return result;
}

struct Scope {
  int64_t id = 0;
  int64_t parent = -1;
  Operation *owner = nullptr;
  unsigned arm = 0;
  bool complete = true;
  std::string limitation;
};

struct Environment {
  DenseMap<Value, int64_t> versions;
  DenseMap<Value, int64_t> writers;
  DenseMap<Value, SmallVector<int64_t>> readers;
  DenseMap<Value, SmallVector<Value>> externalWrites;
  llvm::SmallDenseSet<Value> unknownRoots;
  // A call/opaque operation may retain a private address or leave accesses in
  // flight. A later overwrite does not establish exclusive ownership again.
  llvm::SmallDenseSet<Value> escapedRoots;
};

struct Lineage {
  SmallVector<int64_t> origins;
  bool complete = true;
};

static void mergeLineage(Lineage &target, const Lineage &source) {
  llvm::append_range(target.origins, source.origins);
  llvm::sort(target.origins);
  target.origins.erase(
      std::unique(target.origins.begin(), target.origins.end()),
      target.origins.end());
  target.complete &= source.complete;
}

struct LoopSummary {
  scf::ForOp op;
  Value root;
  int64_t parent = -1, scope = -1;
  int64_t init = -1, iter = -1, yielded = -1, result = -1;
  SmallVector<int64_t> initProducers, iterConsumers, backedgeProducers,
      exitConsumers;
};

class FunctionContract {
public:
  FunctionContract(func::FuncOp function, bool annotate, bool planned,
                   SourcePolicy sourcePolicy)
      : function(function), context(function.getContext()), builder(context),
        annotate(annotate), planned(planned), sourcePolicy(sourcePolicy) {}

  LogicalResult run();

private:
  LogicalResult fail(Operation *op, StringRef rule, const Twine &message) {
    return op->emitOpError() << "kernel contract " << rule << ": " << message;
  }
  void incomplete(int64_t scope, StringRef reason) {
    scopes[scope].complete = false;
    if (scopes[scope].limitation.empty())
      scopes[scope].limitation = reason.str();
  }
  bool disjoint(Value a, Value b);
  bool opaqueReferenceMatmul(Operation *op);
  void invalidate(Value root, Environment &environment);
  void unknownEffect(Operation *op, Environment &environment);
  bool knownScalar(Value value, llvm::SmallPtrSetImpl<Operation *> &seen);
  bool passive(Operation *op);
  LogicalResult checkInPlace(Operation *op, const Effects &effect);
  LogicalResult processRegion(Region &region, int64_t scope,
                              Environment &environment);
  LogicalResult processTask(Operation *op, const Effects &effect, int64_t scope,
                            Environment &environment);
  LogicalResult processLoop(scf::ForOp loop, int64_t parent,
                            Environment &environment);
  LogicalResult processIf(scf::IfOp branch, int64_t parent,
                          Environment &environment);
  FailureOr<int64_t> enterScope(Operation *op, int64_t parent, unsigned arm);
  LogicalResult registerValue(Operation *op, int64_t id, Value root);
  void scalarPredecessors(Value value, SmallVectorImpl<int64_t> &preds,
                          llvm::SmallPtrSetImpl<Operation *> &seen);
  bool reaches(int64_t predecessor, int64_t task) const;
  LogicalResult checkSchema(Operation *op);
  LogicalResult checkDeclaredGraph();
  LogicalResult finish();
  DictionaryAttr scopeAttribute(const Scope &scope, StringRef kind);

  func::FuncOp function;
  MLIRContext *context;
  Builder builder;
  bool annotate, planned;
  SourcePolicy sourcePolicy;
  bool argsDisjoint = false;
  int64_t nextTask = 0, nextValue = 0, nextScope = 1, lastTask = -1;
  DenseMap<int64_t, Scope> scopes;
  DenseMap<int64_t, Value> valueRoots;
  DenseMap<int64_t, int64_t> producers;
  DenseMap<int64_t, Operation *> taskOps;
  DenseMap<Operation *, int64_t> taskIds;
  DenseMap<int64_t, int64_t> taskScopes, stages;
  DenseMap<int64_t, SmallVector<int64_t>> predecessors;
  DenseMap<int64_t, bool> barriers;
  DenseMap<int64_t, Lineage> valueLineages, taskLineages;
  bool sourceFactsComplete = true;
  size_t declaredTaskCount = 0;
  SmallVector<LoopSummary, 2> loops;
};

bool FunctionContract::disjoint(Value a, Value b) {
  Value ar = storageRoot(a), br = storageRoot(b);
  if (ar != br) {
    auto aa = dyn_cast<BlockArgument>(ar), ba = dyn_cast<BlockArgument>(br);
    bool distinctArguments = aa && ba && aa.getOwner() == &function.front() &&
                             ba.getOwner() == &function.front();
    return isPrivate(ar) || isPrivate(br) ||
           (argsDisjoint && distinctArguments);
  }
  auto ab = bounds(a), bb = bounds(b);
  return ab && bb && (ab->end <= bb->begin || bb->end <= ab->begin);
}

bool FunctionContract::opaqueReferenceMatmul(Operation *op) {
  if (!referenceOrderedMatmul(op))
    return false;
  auto matmul = cast<linalg::MatmulOp>(op);
  return llvm::any_of(matmul.getInputs(), [&](Value input) {
    return !disjoint(input, matmul.getOutputs().front());
  });
}

void FunctionContract::invalidate(Value root, Environment &environment) {
  environment.unknownRoots.insert(root);
  environment.versions.erase(root);
  environment.writers.erase(root);
  environment.readers.erase(root);
}

bool FunctionContract::passive(Operation *op) {
  return isa<memref::AllocOp, memref::AllocaOp, memref::DeallocOp,
             memref::DimOp>(op) ||
         isa<ViewLikeOpInterface>(op) ||
         op->hasTrait<OpTrait::IsTerminator>() || isMemoryEffectFree(op);
}

void FunctionContract::unknownEffect(Operation *op, Environment &environment) {
  sourceFactsComplete = false;
  SmallVector<Value> roots;
  for (auto [root, version] : environment.versions)
    roots.push_back(root);
  llvm::append_range(roots, environment.unknownRoots);
  llvm::append_range(roots, environment.escapedRoots);
  auto invalidateAliases = [&](Value value) {
    invalidate(storageRoot(value), environment);
    for (Value root : roots)
      if (!disjoint(root, value))
        invalidate(root, environment);
  };
  if (auto interface = dyn_cast<MemoryEffectOpInterface>(op)) {
    // Effect descriptions alone do not promise that an arbitrary operation
    // completes its accesses here. AtomicRMW is a known synchronous standard
    // operation; other unmodeled operations cannot prove private ownership or
    // the absence of conflicting outstanding accesses.
    bool synchronous = isa<memref::AtomicRMWOp>(op);
    SmallVector<MemoryEffects::EffectInstance> effects;
    interface.getEffects(effects);
    for (const auto &effect : effects) {
      if (synchronous && isa<MemoryEffects::Read>(effect.getEffect()))
        continue;
      if (Value value = effect.getValue();
          value && isa<BaseMemRefType>(value.getType())) {
        Value root = storageRoot(value);
        if (!synchronous && isPrivate(root))
          environment.escapedRoots.insert(root);
        invalidateAliases(value);
      } else {
        for (Value root : roots) {
          if (!synchronous && isPrivate(root))
            environment.escapedRoots.insert(root);
          invalidate(root, environment);
        }
      }
    }
    return;
  }
  // An opaque operation cannot discover an unescaped local allocation unless
  // its address is passed in. Public/previously escaped storage remains visible
  // even to a call without memref operands. Do not extend argument noalias to
  // a call's results or to other opaque roots.
  op->walk([&](Operation *nested) {
    for (Value operand : nested->getOperands()) {
      if (!isa<BaseMemRefType>(operand.getType()))
        continue;
      Value root = storageRoot(operand);
      if (isPrivate(root))
        environment.escapedRoots.insert(root);
      invalidateAliases(operand);
    }
  });
  for (Value root : roots)
    if (!isPrivate(root) || environment.escapedRoots.count(root))
      invalidate(root, environment);
  for (Value result : op->getResults())
    if (isa<BaseMemRefType>(result.getType()))
      invalidate(storageRoot(result), environment);
}

bool FunctionContract::knownScalar(Value value,
                                   llvm::SmallPtrSetImpl<Operation *> &seen) {
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Operation *owner = argument.getOwner()->getParentOp();
    return isa<func::FuncOp>(owner) ||
           (isa<scf::ForOp>(owner) && argument.getArgNumber() == 0);
  }
  Operation *def = value.getDefiningOp();
  if (!def || !seen.insert(def).second)
    return bool(def);
  if (taskIds.count(def))
    return true;
  // In particular, a load without a verified task captured unknown content.
  // Its pure scalar users cannot make that content known again.
  if (!isMemoryEffectFree(def) || def->getNumRegions() != 0)
    return false;
  return llvm::all_of(def->getOperands(), [&](Value operand) {
    return isa<BaseMemRefType>(operand.getType()) || knownScalar(operand, seen);
  });
}

LogicalResult FunctionContract::registerValue(Operation *op, int64_t id,
                                              Value root) {
  if (id < 0 || id == std::numeric_limits<int64_t>::max() ||
      !valueRoots.try_emplace(id, root).second)
    return fail(op, "V7/V21",
                "content version must have one nonnegative definition");
  nextValue = std::max(nextValue, id + 1);
  return success();
}

LogicalResult FunctionContract::checkInPlace(Operation *op,
                                             const Effects &effect) {
  for (Value output : effect.writes) {
    if (isa<linalg::FillOp>(op))
      continue;
    if (isa<linalg::MatmulOp, linalg::ReduceOp>(op)) {
      auto linalg = cast<linalg::LinalgOp>(op);
      for (Value input : linalg.getDpsInputs())
        if (isa<BaseMemRefType>(input.getType()) && !disjoint(input, output))
          return fail(
              op, "V25",
              "accumulator/init must not overlap a matrix/reduction input");
    }
    if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
      if (generic.getNumDpsInits() != 1 ||
          llvm::any_of(generic.getIteratorTypesArray(),
                       [](utils::IteratorType i) {
                         return i != utils::IteratorType::parallel;
                       }) ||
          llvm::any_of(
              llvm::zip(generic.getOperands(), generic.getIndexingMapsArray()),
              [](auto entry) {
                auto [value, map] = entry;
                return isa<BaseMemRefType>(value.getType())
                           ? !map.isIdentity()
                           : map.getNumResults() != 0;
              }))
        return fail(op, "V25",
                    "pointwise update requires parallel identity indexing");
      Block &body = generic.getRegion().front();
      for (Operation &scalar : body.without_terminator()) {
        StringRef dialect = scalar.getName().getDialectNamespace();
        if ((dialect != "arith" && dialect != "math") ||
            !isMemoryEffectFree(&scalar) || scalar.getNumRegions() ||
            llvm::any_of(scalar.getResultTypes(),
                         [](Type type) { return !type.isIntOrFloat(); }))
          return fail(op, "V25",
                      "generic requires pure scalar arithmetic operations");
        for (Value operand : scalar.getOperands()) {
          auto argument = dyn_cast<BlockArgument>(operand);
          if ((argument && argument.getOwner() == &body) ||
              (operand.getDefiningOp() &&
               operand.getDefiningOp()->getBlock() == &body))
            continue;
          auto constant = operand.getDefiningOp<arith::ConstantOp>();
          if (!constant || !isa<IntegerAttr, FloatAttr>(constant.getValue()))
            return fail(op, "V25",
                        "generic scalar captures require scalar constants; "
                        "other scalars must be explicit operands");
        }
      }
    }
    for (Value input : effect.reads) {
      if (disjoint(input, output))
        continue;
      if (isa<linalg::MatmulOp, linalg::ReduceOp>(op)) {
        if (sameAddresses(input, output))
          continue;
        return fail(
            op, "V25",
            "accumulator/init must not overlap a matrix/reduction input");
      }
      if (isa<linalg::GenericOp, memref::CopyOp>(op) &&
          sameAddresses(input, output))
        continue;
      if (isa<memref::StoreOp>(op))
        continue;
      return fail(op, "V25",
                  "overlapping views must have identical element addresses; "
                  "use a separate buffer");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Per-function contract walk (tasks, loops, branches, finish)
//===----------------------------------------------------------------------===//

// True when `predecessor` can reach `task` through recorded task edges.
bool FunctionContract::reaches(int64_t predecessor, int64_t task) const {
  llvm::SmallDenseSet<int64_t> seen;
  SmallVector<int64_t> worklist{task};
  while (!worklist.empty()) {
    int64_t current = worklist.pop_back_val();
    if (current == predecessor)
      return true;
    if (!seen.insert(current).second)
      continue;
    auto found = predecessors.find(current);
    if (found != predecessors.end())
      llvm::append_range(worklist, found->second);
  }
  return false;
}

void FunctionContract::scalarPredecessors(
    Value value, SmallVectorImpl<int64_t> &preds,
    llvm::SmallPtrSetImpl<Operation *> &seen) {
  Operation *def = value.getDefiningOp();
  if (!def || !seen.insert(def).second)
    return;
  if (auto found = taskIds.find(def); found != taskIds.end()) {
    preds.push_back(found->second);
    return;
  }
  if (!isMemoryEffectFree(def))
    return;
  for (Value operand : def->getOperands())
    if (!isa<ShapedType>(operand.getType()))
      scalarPredecessors(operand, preds, seen);
}

LogicalResult FunctionContract::processTask(Operation *op,
                                            const Effects &effect,
                                            int64_t scope,
                                            Environment &environment) {
  auto oldTask = op->getAttrOfType<DictionaryAttr>("contract.task");
  // A reference execution order is an explicit semantic promise, independent
  // of any target engine. Without proven operand independence its internal
  // reads do not form a snapshot task: preserve the operation as a synchronous
  // opaque boundary, and let the planner choose the mechanical ordered recipe.
  bool opaqueOrdered = opaqueReferenceMatmul(op);
  // A full, read-free private overwrite can establish a new version after an
  // unmodeled boundary. Reads, partial inheritance and possibly escaped memory
  // still require facts which the boundary invalidated.
  bool unknown = llvm::any_of(effect.reads, [&](Value value) {
    return environment.unknownRoots.count(storageRoot(value));
  });
  for (Value output : effect.writes) {
    Value root = storageRoot(output);
    unknown |= environment.escapedRoots.count(root) ||
               (isPrivate(root) && !fullRoot(output) &&
                environment.unknownRoots.count(root));
  }
  llvm::SmallPtrSet<Operation *, 16> knownScalars;
  for (Value scalar : effect.scalars)
    unknown |= !knownScalar(scalar, knownScalars);
  // Preserve the v1 external write/read restriction even when graph claims at
  // one of the events have to be omitted.
  for (Value input : effect.reads)
    for (const auto &entry : environment.externalWrites)
      for (Value written : entry.second)
        if (!disjoint(input, written))
          return fail(
              op, "V23/V24",
              "v1 does not support reading an external region after its "
              "terminal write");
  if (opaqueOrdered) {
    if (!annotate && oldTask)
      return fail(op, "V15/V25",
                  "may-alias reference-ordered matmul must omit snapshot task "
                  "claims");
    incomplete(scope,
               "reference-ordered matmul has no proven operand independence");
    sourceFactsComplete = false;
    SmallVector<Value> roots;
    for (auto [root, version] : environment.versions)
      roots.push_back(root);
    llvm::append_range(roots, environment.unknownRoots);
    for (Value output : effect.writes) {
      Value root = storageRoot(output);
      if (!isPrivate(root))
        environment.externalWrites[root].push_back(output);
      invalidate(root, environment);
      for (Value other : roots)
        if (!disjoint(output, other))
          invalidate(other, environment);
    }
    if (annotate) {
      op->removeAttr("contract.task");
      op->removeAttr("contract.value_users");
    }
    return success();
  }
  if ((!annotate && !oldTask && !scopes[scope].complete) ||
      (annotate && unknown)) {
    // An unmodeled boundary cannot manufacture old content IDs. Retain access
    // and provenance facts, but omit graph claims depending on that boundary.
    incomplete(scope, "task depends on an unmodeled memory boundary");
    sourceFactsComplete = false;
    for (Value value : effect.writes) {
      Value root = storageRoot(value);
      if (!isPrivate(root))
        environment.externalWrites[root].push_back(value);
      invalidate(root, environment);
    }
    if (annotate) {
      op->removeAttr("contract.task");
      op->removeAttr("contract.value_users");
    }
    return success();
  }
  if (unknown)
    return fail(op, "V15/V23",
                "declared task depends on an unmodeled memory boundary; omit "
                "unreliable graph claims");
  int64_t id = nextTask++;
  if (!annotate) {
    auto actual = integer(oldTask, "id");
    if (!actual || *actual < 0 || *actual <= lastTask || taskOps.count(*actual))
      return fail(op, "V6",
                  "task IDs must be unique, nonnegative and increasing");
    id = *actual;
    if (integer(oldTask, "scope") != scope ||
        string(oldTask, "kind") != effect.kind)
      return fail(op, "V6/V18",
                  "task kind/scope does not match the actual operation");
    if (!oldTask.getAs<BoolAttr>("rematerializable"))
      return fail(op, "V18", "task rematerializable fact is required");
    if (referenceOrderedMatmul(op) && boolean(oldTask, "rematerializable"))
      return fail(op, "V18/V25",
                  "reference-ordered matmul cannot be rematerializable");
  }
  lastTask = id;
  SmallVector<int64_t> inputValues, requiredPreds;
  DenseMap<Value, int64_t> previousVersions;
  for (Value output : effect.writes) {
    Value root = storageRoot(output);
    if (environment.versions.count(root))
      previousVersions[root] = environment.versions[root];
  }
  auto use = [&](Value view) -> LogicalResult {
    Value root = storageRoot(view);
    auto current = environment.versions.find(root);
    if (current == environment.versions.end())
      return fail(
          op, "V22/V23",
          "read or partial inheritance has no initialized content version");
    if (!llvm::is_contained(inputValues, current->second))
      inputValues.push_back(current->second);
    auto producer = producers.find(current->second);
    if (producer != producers.end() && producer->second >= 0)
      requiredPreds.push_back(producer->second);
    return success();
  };
  for (Value input : effect.reads) {
    if (failed(use(input)))
      return failure();
  }
  for (Value output : effect.writes) {
    Value root = storageRoot(output);
    if (isPrivate(root) && !fullRoot(output) && failed(use(output)))
      return failure();
    if (auto previous = environment.writers.find(root);
        previous != environment.writers.end())
      requiredPreds.push_back(previous->second);
    llvm::append_range(requiredPreds, environment.readers[root]);
    // Distinct external descriptors can name the same storage. Entry root IDs
    // alone do not establish independence for WAR/WAW ordering.
    for (auto [other, writer] : environment.writers)
      if (other != root && !disjoint(output, other))
        requiredPreds.push_back(writer);
    for (const auto &[other, readers] : environment.readers)
      if (other != root && !disjoint(output, other))
        llvm::append_range(requiredPreds, readers);
  }
  llvm::SmallPtrSet<Operation *, 16> scalarSeen;
  SmallVector<int64_t> scalarSources;
  for (Value scalar : effect.scalars)
    scalarPredecessors(scalar, scalarSources, scalarSeen);
  llvm::append_range(requiredPreds, scalarSources);
  llvm::sort(requiredPreds);
  requiredPreds.erase(std::unique(requiredPreds.begin(), requiredPreds.end()),
                      requiredPreds.end());
  if (failed(checkInPlace(op, effect)))
    return failure();

  Lineage lineage;
  for (int64_t input : inputValues)
    mergeLineage(lineage, valueLineages[input]);
  for (int64_t source : scalarSources)
    mergeLineage(lineage, taskLineages[source]);
  auto provenance = op->getAttrOfType<DictionaryAttr>("contract.provenance");
  auto access = op->getAttrOfType<DictionaryAttr>("contract.access");
  SmallVector<Lineage, 3> dotLineages;
  if (auto dot = dyn_cast<linalg::MatmulOp>(op)) {
    // A forwarded argument has no snapshot task from which to recover its
    // source access. Retained operand facts survive that physical rewrite.
    // A proven constant/fill has a complete, empty lineage and must never be
    // replaced with stale frontend provenance from a former accumulator.
    lineage = Lineage{};
    const StringRef operandNames[] = {"a", "b", "accumulator"};
    for (auto [name, operand] : llvm::zip(operandNames, dot->getOperands())) {
      Value root = storageRoot(operand);
      Lineage operandLineage = valueLineages[environment.versions.lookup(root)];
      if (!operandLineage.complete) {
        std::string prefix = name.str() + "_source";
        Lineage retained{numbers(provenance, prefix + "_load_origin_ids"),
                         boolean(provenance, prefix + "_lineage_complete")};
        if (!retained.origins.empty()) {
          bool retainedComplete = retained.complete;
          mergeLineage(operandLineage, retained);
          operandLineage.complete = retainedComplete;
        }
      }
      mergeLineage(lineage, operandLineage);
      dotLineages.push_back(std::move(operandLineage));
    }
  }
  bool externalLoad = effect.kind == "memory-load" &&
                      llvm::any_of(effect.reads, [](Value value) {
                        return !isPrivate(storageRoot(value));
                      });
  if (externalLoad) {
    // The boundary event identifies the new snapshot; its external entry
    // version does not itself stand for a source-language load.
    lineage = Lineage{numbers(provenance, "source_load_origin_ids"),
                      boolean(provenance, "source_lineage_complete")};
    if (!sourcePolicy.isSyntheticTransfer ||
        !sourcePolicy.isSyntheticTransfer(op))
      if (auto origin = integer(access, "origin_id")) {
        mergeLineage(lineage, Lineage{{*origin}, true});
        lineage.complete = true;
      }
    // A masked copy still inherits the destination's initialized padding.
    for (Value output : effect.writes)
      if (!fullRoot(output))
        if (auto old = previousVersions.find(storageRoot(output));
            old != previousVersions.end())
          mergeLineage(lineage, valueLineages[old->second]);
    sourceFactsComplete &= lineage.complete;
  } else if (llvm::any_of(effect.reads, [](Value value) {
               return !isPrivate(storageRoot(value));
             })) {
    sourceFactsComplete &= lineage.complete;
  }

  SmallVector<int64_t> preds = requiredPreds;
  if (!annotate) {
    if (!oldTask.getAs<DenseI64ArrayAttr>("input_values") ||
        numbers(oldTask, "input_values") != inputValues)
      return fail(op, "V22/V23",
                  "input_values must match actual reads, latest versions and "
                  "partial-write inheritance");
    preds = numbers(oldTask, "predecessors");
    if (!oldTask.getAs<DenseI64ArrayAttr>("predecessors"))
      return fail(op, "V9", "predecessors array is required");
    for (int64_t pred : preds)
      if (pred < 0 || pred >= id || !taskOps.count(pred))
        return fail(op, "V9", "predecessor is not an earlier defined task");
    predecessors[id] = preds;
    for (int64_t pred : requiredPreds)
      if (!reaches(pred, id))
        return fail(op, "V23/V24",
                    "missing RAW, WAR, WAW or scalar SSA dependency");
  }
  int64_t stage = 0;
  for (int64_t pred : preds)
    stage = std::max(stage, stages[pred] + int64_t(barriers[pred]));
  if (!annotate && (integer(oldTask, "stage") != stage ||
                    !oldTask.getAs<BoolAttr>("barrier_after") ||
                    boolean(oldTask, "barrier_after") != effect.barrier))
    return fail(op, "V18",
                "stage/barrier does not match task completion dependencies");

  SmallVector<int64_t> outputs;
  if (!annotate) {
    if (!oldTask.getAs<DenseI64ArrayAttr>("output_values"))
      return fail(op, "V21", "output_values array is required");
    outputs = numbers(oldTask, "output_values");
  }
  size_t outputIndex = 0;
  llvm::SmallDenseSet<Value> writtenRoots;
  for (Value output : effect.writes) {
    Value root = storageRoot(output);
    if (!writtenRoots.insert(root).second)
      return fail(op, "V21",
                  "one task cannot define two versions of the same root");
    if (isPrivate(root)) {
      if (!environment.versions.count(root) && !fullRoot(output))
        return fail(op, "V22",
                    "first private write must initialize the whole root");
      int64_t version = nextValue;
      if (annotate)
        outputs.push_back(version);
      else if (outputIndex >= outputs.size())
        return fail(op, "V21", "missing output content version");
      else
        version = outputs[outputIndex];
      ++outputIndex;
      if (failed(registerValue(op, version, root)))
        return failure();
      environment.versions[root] = version;
      environment.unknownRoots.erase(root);
      producers[version] = id;
      valueLineages[version] = lineage;
    } else {
      environment.externalWrites[root].push_back(output);
    }
    environment.writers[root] = id;
    environment.readers[root].clear();
  }
  if (outputs.size() != outputIndex)
    return fail(
        op, "V21",
        "output_values must correspond exactly to private written roots");
  for (Value input : effect.reads) {
    Value root = storageRoot(input);
    if (!writtenRoots.count(root) &&
        !llvm::is_contained(environment.readers[root], id))
      environment.readers[root].push_back(id);
  }
  for (LoopSummary &loop : loops) {
    auto previous = previousVersions.find(loop.root);
    bool writesBoundary = previous != previousVersions.end();
    if (scope == loop.scope &&
        (llvm::is_contained(inputValues, loop.iter) ||
         (writesBoundary && previous->second == loop.iter)))
      loop.iterConsumers.push_back(id);
    if (scope == loop.parent &&
        (llvm::is_contained(inputValues, loop.result) ||
         (writesBoundary && previous->second == loop.result)))
      loop.exitConsumers.push_back(id);
  }
  predecessors[id] = preds;
  taskOps[id] = op;
  taskIds[op] = id;
  taskScopes[id] = scope;
  stages[id] = stage;
  barriers[id] = effect.barrier;
  taskLineages[id] = lineage;
  if (annotate) {
    NamedAttrList attrs;
    attrs.set("id", builder.getI64IntegerAttr(id));
    attrs.set("kind", builder.getStringAttr(effect.kind));
    attrs.set("scope", builder.getI64IntegerAttr(scope));
    attrs.set("stage", builder.getI64IntegerAttr(stage));
    attrs.set("predecessors", builder.getDenseI64ArrayAttr(preds));
    attrs.set("input_values", builder.getDenseI64ArrayAttr(inputValues));
    attrs.set("output_values", builder.getDenseI64ArrayAttr(outputs));
    attrs.set("barrier_after", builder.getBoolAttr(effect.barrier));
    attrs.set("rematerializable", builder.getBoolAttr(false));
    if (oldTask && oldTask.get("origin_id"))
      attrs.set("origin_id", oldTask.get("origin_id"));
    else if (auto access = op->getAttrOfType<DictionaryAttr>("contract.access"))
      if (auto origin = access.get("origin_id"))
        attrs.set("origin_id", origin);
    op->setAttr("contract.task", attrs.getDictionary(context));
    NamedAttrList sourceAttrs(provenance);
    if (!lineage.origins.empty()) {
      sourceAttrs.set("source_load_origin_ids",
                      builder.getDenseI64ArrayAttr(lineage.origins));
      sourceAttrs.set("source_lineage_complete",
                      builder.getBoolAttr(lineage.complete));
    } else {
      sourceAttrs.erase("source_load_origin_ids");
      sourceAttrs.erase("source_lineage_complete");
      if (!lineage.complete)
        sourceAttrs.set("source_lineage_complete", builder.getBoolAttr(false));
    }
    if (!dotLineages.empty()) {
      const StringRef operandNames[] = {"a", "b", "accumulator"};
      for (auto [name, operandLineage] : llvm::zip(operandNames, dotLineages)) {
        std::string prefix = name.str() + "_source";
        if (!operandLineage.origins.empty())
          sourceAttrs.set(prefix + "_load_origin_ids",
                          builder.getDenseI64ArrayAttr(operandLineage.origins));
        else
          sourceAttrs.erase(prefix + "_load_origin_ids");
        if (operandLineage.complete && operandLineage.origins.empty())
          sourceAttrs.erase(prefix + "_lineage_complete");
        else
          sourceAttrs.set(prefix + "_lineage_complete",
                          builder.getBoolAttr(operandLineage.complete));
      }
    }
    if (!sourceAttrs.empty())
      op->setAttr("contract.provenance", sourceAttrs.getDictionary(context));
    op->removeAttr("contract.value_users");
    if (!isa<scf::ForOp>(op))
      op->removeAttr("contract.value_ids");
  }
  return success();
}

FailureOr<int64_t> FunctionContract::enterScope(Operation *op, int64_t parent,
                                                unsigned arm) {
  DictionaryAttr old;
  if (isa<scf::ForOp>(op)) {
    old = op->getAttrOfType<DictionaryAttr>("contract.scope");
  } else if (auto attrs = op->getAttrOfType<ArrayAttr>("contract.scopes")) {
    if (arm < attrs.size())
      old = dyn_cast<DictionaryAttr>(attrs[arm]);
  }
  int64_t id = nextScope++;
  if (!annotate) {
    auto actual = integer(old, "id");
    if (!actual || *actual <= 0 || scopes.count(*actual) ||
        integer(old, "parent") != parent)
      return fail(op, "V17",
                  "scope IDs/parent must describe the actual region tree");
    id = *actual;
    if (!old.getAs<BoolAttr>("complete"))
      return fail(op, "V17/V18", "scope completeness is required");
    if (string(old, "kind") != (isa<scf::ForOp>(op) ? "for-body" : "if-arm") ||
        (isa<scf::IfOp>(op) && (string(old, "arm") != (arm ? "else" : "then") ||
                                !boolean(old, "arms_exclusive"))))
      return fail(op, "V17",
                  "scope kind/arm/exclusivity does not match its region");
  }
  Scope scope{id, parent, op, arm, true, {}};
  if (old && !boolean(old, "complete", true)) {
    scope.complete = false;
    scope.limitation = string(old, "limitation").str();
    if (!annotate && scope.limitation.empty())
      return fail(op, "V15",
                  "incomplete scope must identify its unmodeled boundary");
  }
  scopes[id] = std::move(scope);
  return id;
}

LogicalResult FunctionContract::processLoop(scf::ForOp loop, int64_t parent,
                                            Environment &environment) {
  FailureOr<int64_t> scope = enterScope(loop, parent, 0);
  if (failed(scope))
    return failure();
  SmallVector<Value> carries;
  bool modeledBody = true;
  loop.getBodyRegion().walk([&](Operation *op) {
    if (opaqueReferenceMatmul(op))
      modeledBody = false;
    if (auto effect = effects(op))
      for (Value output : effect->writes) {
        Value root = storageRoot(output);
        if (isPrivate(root) && environment.versions.count(root) &&
            !llvm::is_contained(carries, root))
          carries.push_back(root);
      }
    else if (isa<scf::ForOp, scf::IfOp>(op) || !passive(op))
      modeledBody = false;
  });
  auto lower = getConstantIntValue(loop.getLowerBound());
  auto upper = getConstantIntValue(loop.getUpperBound());
  auto step = getConstantIntValue(loop.getStep());
  int64_t distance = 0, trips = 0;
  bool supported = modeledBody && carries.size() == 1 && lower && upper &&
                   step && *step > 0 && *upper > *lower &&
                   loop.getInitArgs().empty() &&
                   !llvm::SubOverflow(*upper, *lower, distance);
  if (supported) {
    trips = distance / *step;
    if (distance % *step)
      supported = !llvm::AddOverflow(trips, int64_t(1), trips);
  }
  if (!annotate && !scopes[*scope].complete &&
      !loop->hasAttr("contract.loop_carried_slots"))
    supported = false;
  if (!supported) {
    incomplete(*scope, "v1 requires a static serial loop with one private "
                       "buffer-carried slot");
    if (!annotate &&
        boolean(loop->getAttrOfType<DictionaryAttr>("contract.scope"),
                "complete"))
      return fail(
          loop, "V20",
          "unsupported loop cannot claim a complete single-slot summary");
    if (!annotate && (loop->hasAttr("contract.value_ids") ||
                      loop->hasAttr("contract.loop_carried_slots")))
      return fail(loop, "V15/V20",
                  "unmodeled loop must omit unreliable slot/value definitions");
    if (annotate) {
      loop->removeAttr("contract.value_ids");
      loop->removeAttr("contract.loop_carried_slots");
      loop->removeAttr("contract.cf");
    }
    Environment body = environment, boundary = environment;
    // Without a loop summary the final writer/iteration is unknown. Compute
    // the may-write set before inspecting one body instance, so a read cannot
    // accidentally use the initializer in every dynamic iteration. Unrelated
    // roots retain their parent-scope facts.
    loop.getBodyRegion().walk([&](Operation *op) {
      if (auto effect = effects(op)) {
        for (Value input : effect->reads) {
          Value root = storageRoot(input);
          if (isPrivate(root) && environment.versions.count(root))
            invalidate(root, body);
        }
        for (Value output : effect->writes) {
          Value root = storageRoot(output);
          if (environment.versions.count(root) ||
              environment.unknownRoots.count(root))
            invalidate(root, boundary);
        }
      } else if (!isa<scf::ForOp, scf::IfOp>(op) && !passive(op)) {
        unknownEffect(op, boundary);
      }
    });
    for (Value root : boundary.unknownRoots)
      invalidate(root, body);
    for (Value root : boundary.escapedRoots)
      body.escapedRoots.insert(root);
    if (failed(processRegion(loop.getRegion(), *scope, body)))
      return failure();
    for (Value root : boundary.unknownRoots)
      invalidate(root, environment);
    for (Value root : body.escapedRoots)
      environment.escapedRoots.insert(root);
    for (auto &[root, writes] : body.externalWrites)
      for (Value write : writes)
        if (!llvm::is_contained(environment.externalWrites[root], write))
          environment.externalWrites[root].push_back(write);
    return success();
  }
  if (!annotate) {
    auto cf = loop->getAttrOfType<DictionaryAttr>("contract.cf");
    if (string(cf, "kind") != "uniform_for" || !boolean(cf, "uniform") ||
        !boolean(cf, "serial") || integer(cf, "trip_count") != trips)
      return fail(loop, "V10/V20",
                  "loop summary must match its static serial trip count");
  }
  LoopSummary summary;
  summary.op = loop;
  summary.root = carries.front();
  summary.parent = parent;
  summary.scope = *scope;
  summary.init = environment.versions[summary.root];
  summary.iter = nextValue++;
  summary.result = nextValue++;
  if (!annotate) {
    auto values = loop->getAttrOfType<DenseI64ArrayAttr>("contract.value_ids");
    auto slots = loop->getAttrOfType<ArrayAttr>("contract.loop_carried_slots");
    if (!values || values.size() != 2 || !slots || slots.size() != 1)
      return fail(
          loop, "V20",
          "complete loop requires one slot and [iter,result] definitions");
    summary.iter = values[0];
    summary.result = values[1];
    auto slot = dyn_cast<DictionaryAttr>(slots[0]);
    if (!slot || integer(slot, "slot") != 0 ||
        integer(slot, "init_value_id") != summary.init ||
        integer(slot, "iter_arg_value_id") != summary.iter ||
        integer(slot, "result_value_id") != summary.result)
      return fail(
          loop, "V20/V26",
          "loop slot does not match its initializer and value definitions");
  }
  if (failed(registerValue(loop, summary.iter, summary.root)) ||
      failed(registerValue(loop, summary.result, summary.root)))
    return failure();
  valueLineages[summary.iter] = valueLineages[summary.init];
  if (auto producer = producers.find(summary.init);
      producer != producers.end()) {
    summary.initProducers.push_back(producer->second);
    producers[summary.iter] = producer->second;
  }
  size_t loopIndex = loops.size();
  loops.push_back(summary);
  Environment body = environment;
  body.versions[summary.root] = summary.iter;
  if (failed(processRegion(loop.getRegion(), *scope, body)))
    return failure();
  LoopSummary &completed = loops[loopIndex];
  completed.yielded = body.versions[completed.root];
  if (completed.yielded == completed.iter)
    return fail(loop, "V20", "loop slot has no backedge writer");
  completed.backedgeProducers.push_back(producers[completed.yielded]);
  producers[completed.result] = producers[completed.yielded];
  valueLineages[completed.result] = valueLineages[completed.yielded];
  environment.versions[completed.root] = completed.result;
  environment.writers[completed.root] = producers[completed.yielded];
  environment.readers[completed.root].clear();
  if (annotate) {
    loop->setAttr(
        "contract.cf",
        builder.getDictionaryAttr(
            {builder.getNamedAttr("kind", builder.getStringAttr("uniform_for")),
             builder.getNamedAttr("trip_count",
                                  builder.getI64IntegerAttr(trips)),
             builder.getNamedAttr("uniform", builder.getBoolAttr(true)),
             builder.getNamedAttr("serial", builder.getBoolAttr(true))}));
  }
  return success();
}

LogicalResult FunctionContract::processIf(scf::IfOp branch, int64_t parent,
                                          Environment &environment) {
  if (!branch.elseBlock())
    return fail(branch, "V17", "v1 if requires explicit then and else scopes");
  if (!annotate) {
    auto attrs = branch->getAttrOfType<ArrayAttr>("contract.scopes");
    if (!attrs || attrs.size() != 2)
      return fail(branch, "V17", "if requires exactly two scope records");
  }
  auto oldCF = branch->getAttrOfType<DictionaryAttr>("contract.cf");
  llvm::SmallPtrSet<Operation *, 16> seen;
  bool uniform = uniformCondition(branch.getCondition(), seen);
  if (annotate && !oldCF && uniform)
    branch->setAttr(
        "contract.cf",
        builder.getDictionaryAttr(
            {builder.getNamedAttr("kind", builder.getStringAttr("uniform_if")),
             builder.getNamedAttr("arms_exclusive",
                                  builder.getBoolAttr(true))}));
  llvm::SmallDenseSet<Value> changedRoots;
  DenseMap<Value, SmallVector<Value>> branchWrites;
  for (unsigned arm = 0; arm < 2; ++arm) {
    auto scope = enterScope(branch, parent, arm);
    if (failed(scope))
      return failure();
    if (!uniform)
      incomplete(
          *scope,
          "if condition is not derived solely from uniform program facts");
    if (!annotate && scopes[*scope].complete &&
        (string(oldCF, "kind") != "uniform_if" ||
         !boolean(oldCF, "arms_exclusive")))
      return fail(branch, "V10/V17",
                  "complete if requires uniform exclusive arms");
    Environment body = environment;
    branch->getRegion(arm).walk([&](Operation *op) {
      auto effect = effects(op);
      if (!effect)
        return;
      auto markParentPrivate = [&](Value value) {
        Value root = storageRoot(value);
        if (isPrivate(root) && environment.versions.count(root)) {
          invalidate(root, body);
          incomplete(*scope,
                     "parent private content crossing if is outside v1");
        }
      };
      for (Value value : effect->reads)
        markParentPrivate(value);
      for (Value value : effect->writes)
        markParentPrivate(value);
    });
    if (failed(processRegion(branch->getRegion(arm), *scope, body)))
      return failure();
    for (auto [root, version] : environment.versions)
      if (isPrivate(root) && (body.versions.lookup(root) != version ||
                              body.unknownRoots.count(root))) {
        changedRoots.insert(root);
        incomplete(*scope,
                   "private value merge through if is outside the v1 contract");
      }
    for (Value root : body.unknownRoots)
      if (environment.versions.count(root) ||
          environment.unknownRoots.count(root))
        changedRoots.insert(root);
    for (Value root : body.escapedRoots)
      environment.escapedRoots.insert(root);
    for (auto &[root, writes] : body.externalWrites)
      for (Value write : writes)
        if (!llvm::is_contained(environment.externalWrites[root], write) &&
            !llvm::is_contained(branchWrites[root], write))
          branchWrites[root].push_back(write);
  }
  for (Value root : changedRoots)
    invalidate(root, environment);
  for (auto &[root, writes] : branchWrites) {
    llvm::append_range(environment.externalWrites[root], writes);
    // Mutually exclusive stores do not name one parent-scope writer. Further
    // accesses require a memory merge model outside the v1 contract.
    environment.unknownRoots.insert(root);
  }
  return success();
}

LogicalResult FunctionContract::processRegion(Region &region, int64_t scope,
                                              Environment &environment) {
  if (!llvm::hasSingleElement(region)) {
    incomplete(scope, "multi-block control flow is not modeled");
    return success();
  }
  for (Operation &operation : region.front()) {
    Operation *op = &operation;
    if (failed(checkSchema(op)))
      return failure();
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      if (failed(processLoop(loop, scope, environment)))
        return failure();
    } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
      if (failed(processIf(branch, scope, environment)))
        return failure();
    } else if (auto effect = effects(op)) {
      if (failed(processTask(op, *effect, scope, environment)))
        return failure();
    } else if (passive(op)) {
      // Allocation/view/index/pure scalar ops are not dataflow tasks.
    } else {
      if (!annotate && op->hasAttr("contract.task"))
        return fail(op, "V15/V18",
                    "unmodeled operation cannot declare a verified task");
      incomplete(scope, "unmodeled operation or memory effect");
      unknownEffect(op, environment);
    }
  }
  return success();
}

LogicalResult FunctionContract::checkDeclaredGraph() {
  DenseMap<int64_t, Operation *> definitions, tasks;
  SmallVector<std::pair<Operation *, int64_t>> valueReferences, taskReferences;
  auto define = [&](Operation *op, int64_t id) -> LogicalResult {
    if (id < 0 || id == std::numeric_limits<int64_t>::max() ||
        !definitions.try_emplace(id, op).second)
      return fail(op, "V7/V21",
                  "content version must have one nonnegative definition");
    return success();
  };
  for (auto [index, argument] : llvm::enumerate(function.getArguments())) {
    Attribute attr = function.getArgAttr(index, "contract.value_id");
    if (!attr)
      continue;
    auto id = dyn_cast<IntegerAttr>(attr);
    if (!id || !isa<BaseMemRefType>(argument.getType()))
      return fail(function, "V7",
                  "only memref arguments define integer entry versions");
    if (failed(define(function, id.getInt())))
      return failure();
  }
  LogicalResult valid = success();
  int64_t previousTask = -1;
  function.walk([&](Operation *op) {
    if (failed(valid))
      return;
    if (Attribute attr = op->getAttr("contract.task")) {
      auto task = dyn_cast<DictionaryAttr>(attr);
      if (!task || !effects(op)) {
        valid = fail(op, "V18",
                     "task must be a dictionary on a modeled memory operation");
        return;
      }
      for (StringRef key : {"id", "scope", "stage"})
        if (!integer(task, key) || *integer(task, key) < 0) {
          valid = fail(op, "V6/V18",
                       "task id/scope/stage must be nonnegative integers");
          return;
        }
      int64_t id = *integer(task, "id");
      if (id <= previousTask || !tasks.try_emplace(id, op).second) {
        valid = fail(op, "V6",
                     "task IDs must be unique, nonnegative and increasing");
        return;
      }
      previousTask = id;
      for (StringRef key : {"kind"})
        if (!task.getAs<StringAttr>(key)) {
          valid = fail(op, "V18", "task kind must be a string");
          return;
        }
      for (StringRef key : {"barrier_after", "rematerializable"})
        if (!task.getAs<BoolAttr>(key)) {
          valid =
              fail(op, "V18",
                   "task completion/rematerialization fields must be boolean");
          return;
        }
      for (StringRef key : {"input_values", "output_values", "predecessors"}) {
        auto ids = task.getAs<DenseI64ArrayAttr>(key);
        if (!ids) {
          valid =
              fail(op, "V7/V9/V21",
                   "task inputs, outputs and predecessors must be i64 arrays");
          return;
        }
        for (int64_t value : ids.asArrayRef()) {
          if (key == "output_values") {
            if (failed(valid = define(op, value)))
              return;
          } else if (key == "input_values") {
            valueReferences.emplace_back(op, value);
          } else {
            taskReferences.emplace_back(op, value);
          }
        }
      }
    }
    if (Attribute attr = op->getAttr("contract.value_ids")) {
      auto values = dyn_cast<DenseI64ArrayAttr>(attr);
      if (!isa<scf::ForOp>(op) || !values || values.size() != 2) {
        valid =
            fail(op, "V20/V21", "only a modeled loop defines [iter,result]");
        return;
      }
      for (int64_t value : values.asArrayRef())
        if (failed(valid = define(op, value)))
          return;
    }
    if (Attribute attr = op->getAttr("contract.loop_carried_slots")) {
      auto slots = dyn_cast<ArrayAttr>(attr);
      if (!isa<scf::ForOp>(op) || !slots || slots.size() != 1) {
        valid = fail(op, "V20", "v1 loop summary requires one slot dictionary");
        return;
      }
      auto slot = dyn_cast<DictionaryAttr>(slots[0]);
      for (StringRef key : {"slot", "init_value_id", "iter_arg_value_id",
                            "yielded_value_id", "result_value_id"}) {
        auto id = integer(slot, key);
        if (!id || *id < 0) {
          valid = fail(op, "V11/V20",
                       "loop slot/value fields must be nonnegative integers");
          return;
        }
        if (key != "slot")
          valueReferences.emplace_back(op, *id);
      }
      for (StringRef key :
           {"init_producer_compute_ids", "iter_arg_consumer_compute_ids",
            "backedge_producer_compute_ids", "exit_consumer_compute_ids"}) {
        auto ids = slot.getAs<DenseI64ArrayAttr>(key);
        if (!ids) {
          valid = fail(op, "V11",
                       "loop producer/consumer fields must be i64 arrays");
          return;
        }
        for (int64_t id : ids.asArrayRef())
          taskReferences.emplace_back(op, id);
      }
    }
  });
  if (failed(valid))
    return failure();
  declaredTaskCount = tasks.size();
  for (auto [op, value] : valueReferences)
    if (!definitions.count(value))
      return fail(
          op, "V7/V15",
          "unresolved content version, including in an incomplete graph");
  for (auto [op, task] : taskReferences)
    if (!tasks.count(task))
      return fail(
          op, "V9/V15",
          "unresolved task reference, including in an incomplete graph");
  return success();
}

LogicalResult FunctionContract::checkSchema(Operation *op) {
  for (StringRef name :
       {"contract.access", "contract.buffer", "contract.compute",
        "contract.provenance", "contract.task", "contract.scope", "contract.cf",
        "contract.compute_graph"})
    if (op->hasAttr(name) && !op->getAttrOfType<DictionaryAttr>(name))
      return fail(op, "V15",
                  "declared contract dictionary has an invalid field type");
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (!planned &&
        (name.starts_with("plan.") || name.starts_with("bosc_ame.")))
      return fail(op, "V14",
                  "target attributes are forbidden in the neutral contract");
  }
  for (Type type : op->getOperandTypes())
    if (isa<TensorType>(type))
      return fail(op, "V3",
                  "contract must be buffer form, without tensor operands");
  for (Type type : op->getResultTypes())
    if (isa<TensorType>(type))
      return fail(op, "V3",
                  "contract must be buffer form, without tensor results");
  if (isa<UnrealizedConversionCastOp>(op))
    return fail(op, "V3", "unrealized casts are forbidden");
  if (auto store = dyn_cast<memref::StoreOp>(op))
    if (isPrivate(storageRoot(store.getMemref())) &&
        cast<MemRefType>(store.getMemref().getType()).getRank() != 0)
      return fail(op, "V22/V25",
                  "v1 private scalar stores require rank-zero storage; a "
                  "single-element write cannot initialize the whole root");
  if (auto compute = op->getAttrOfType<DictionaryAttr>("contract.compute")) {
    if (compute.get("order") || string(compute, "kind") == "matmul") {
      if (!referenceOrderedMatmul(op))
        return fail(op, "V18/V25",
                    "reference order requires matmul with kind=matmul and "
                    "order=mnk");
      for (Operation &payload :
           cast<linalg::MatmulOp>(op).getRegion().front().without_terminator())
        if (payload.getName().getDialectNamespace() != "arith" ||
            payload.getNumRegions() != 0 || !isMemoryEffectFree(&payload))
          return fail(op, "V18/V25",
                      "reference-ordered matmul requires a pure scalar "
                      "arithmetic payload");
    }
  }
  if (!annotate &&
      ((op->hasAttr("contract.value_ids") && !isa<scf::ForOp>(op)) ||
       (op->hasAttr("contract.scope") && !isa<scf::ForOp>(op)) ||
       (op->hasAttr("contract.scopes") && !isa<scf::IfOp>(op))))
    return fail(op, "V17/V21",
                "scope/loop definitions are attached to the wrong operation");
  if (auto access = op->getAttrOfType<DictionaryAttr>("contract.access")) {
    StringRef kind = string(access, "kind");
    auto mask = access.getAs<DictionaryAttr>("mask");
    StringRef maskKind = string(mask, "kind");
    if ((kind != "load" && kind != "store") ||
        string(access, "boundary") != "external_ddr" ||
        (maskKind != "none" && maskKind != "rectangular" &&
         maskKind != "opaque"))
      return fail(op, "V4", "invalid access kind/boundary/mask");
    if (boolean(mask, "dynamic"))
      return fail(op, "V16", "v1 only supports none/static/opaque masks");
    if (maskKind == "rectangular") {
      auto shape = mask.getAs<DenseI64ArrayAttr>("valid_shape");
      auto copy = dyn_cast<memref::CopyOp>(op);
      auto type = copy ? dyn_cast<MemRefType>(copy.getSource().getType())
                       : MemRefType();
      if (!shape || !type || shape.size() != type.getRank() ||
          llvm::any_of(shape.asArrayRef(),
                       [](int64_t size) { return size <= 0; }))
        return fail(
            op, "V4",
            "static valid shape must match the access rank and be positive");
      if (kind == "load" && !access.get("other"))
        return fail(op, "V4", "masked load must record its other value");
    }
    if (access.get("other") && kind != "load")
      return fail(op, "V4", "other is only valid on loads");
    if (Attribute other = access.get("other")) {
      auto typed = dyn_cast<TypedAttr>(other);
      auto copy = dyn_cast<memref::CopyOp>(op);
      auto type = copy ? dyn_cast<MemRefType>(copy.getSource().getType())
                       : MemRefType();
      if (!typed || !type || typed.getType() != type.getElementType())
        return fail(op, "V4", "other must have the load element type");
    }
  }
  if (auto buffer = op->getAttrOfType<DictionaryAttr>("contract.buffer")) {
    auto shape = buffer.getAs<DenseI64ArrayAttr>("logical_shape");
    if (!isa<memref::AllocOp, memref::AllocaOp>(op) || !shape ||
        shape.size() > 2 || llvm::any_of(shape.asArrayRef(), [](int64_t size) {
          return size <= 0;
        }))
      return fail(op, "V5",
                  "buffer must describe a static rank-0/1/2 allocation");
    int64_t count = 1;
    for (int64_t size : shape.asArrayRef())
      if (llvm::MulOverflow(count, size, count))
        return fail(op, "V5", "logical shape overflows");
    auto type = cast<MemRefType>(op->getResult(0).getType());
    if (!type.hasStaticShape() || count != type.getNumElements())
      return fail(op, "V5",
                  "logical shape and allocation element counts disagree");
    StringRef role = string(buffer, "role"), kind = string(buffer, "kind");
    if ((role != "tile" && role != "scratch") ||
        (kind != "accumulator" && kind != "input-a" && kind != "input-b" &&
         kind != "temporary" && kind != "padding"))
      return fail(op, "V5", "buffer role/kind is invalid");
  } else if (annotate && isa<memref::AllocOp, memref::AllocaOp>(op)) {
    auto type = cast<MemRefType>(op->getResult(0).getType());
    if (type.hasStaticShape() && type.getRank() <= 2)
      op->setAttr(
          "contract.buffer",
          builder.getDictionaryAttr(
              {builder.getNamedAttr(
                   "role",
                   builder.getStringAttr(type.getRank() ? "tile" : "scratch")),
               builder.getNamedAttr("kind", builder.getStringAttr("temporary")),
               builder.getNamedAttr(
                   "logical_shape",
                   builder.getDenseI64ArrayAttr(type.getShape()))}));
  }
  return success();
}

DictionaryAttr FunctionContract::scopeAttribute(const Scope &scope,
                                                StringRef kind) {
  NamedAttrList attrs;
  attrs.set("id", builder.getI64IntegerAttr(scope.id));
  attrs.set("parent", builder.getI64IntegerAttr(scope.parent));
  attrs.set("kind", builder.getStringAttr(kind));
  attrs.set("complete", builder.getBoolAttr(scope.complete));
  if (!scope.limitation.empty())
    attrs.set("limitation", builder.getStringAttr(scope.limitation));
  if (kind == "if-arm") {
    attrs.set("arm", builder.getStringAttr(scope.arm ? "else" : "then"));
    attrs.set("arms_exclusive", builder.getBoolAttr(true));
  }
  return attrs.getDictionary(context);
}

LogicalResult FunctionContract::finish() {
  bool changed;
  do {
    changed = false;
    for (auto &[id, scope] : scopes)
      if (id != 0 && !scope.complete && scopes[scope.parent].complete) {
        incomplete(scope.parent, "child scope has an unmodeled boundary");
        changed = true;
      }
  } while (changed);
  if (!annotate && taskOps.size() != declaredTaskCount)
    return fail(
        function, "V15/V18",
        "declared task lies in an unmodeled region and cannot be verified");
  bool complete = llvm::all_of(
      scopes, [](const auto &entry) { return entry.second.complete; });
  for (auto &[id, scope] : scopes) {
    if (id == 0)
      continue;
    if (annotate) {
      if (isa<scf::ForOp>(scope.owner))
        scope.owner->setAttr("contract.scope",
                             scopeAttribute(scope, "for-body"));
    } else {
      DictionaryAttr actual;
      if (isa<scf::ForOp>(scope.owner))
        actual = scope.owner->getAttrOfType<DictionaryAttr>("contract.scope");
      else if (auto records =
                   scope.owner->getAttrOfType<ArrayAttr>("contract.scopes"))
        if (scope.arm < records.size())
          actual = dyn_cast<DictionaryAttr>(records[scope.arm]);
      if (!actual || boolean(actual, "complete") != scope.complete)
        return fail(scope.owner, "V18",
                    "scope completeness must include its child scopes");
    }
  }
  if (annotate) {
    function.walk([&](scf::IfOp branch) {
      SmallVector<Attribute> attrs(2);
      for (auto &[id, scope] : scopes)
        if (scope.owner == branch)
          attrs[scope.arm] = scopeAttribute(scope, "if-arm");
      // A branch nested in an opaque/unmodeled region was never analyzed.
      // It has no scope records to attach, rather than two null attributes.
      if (!attrs[0] || !attrs[1]) {
        branch->removeAttr("contract.scopes");
        return;
      }
      branch->setAttr("contract.scopes", builder.getArrayAttr(attrs));
    });
  }
  for (LoopSummary &loop : loops) {
    auto slot = builder.getDictionaryAttr(
        {builder.getNamedAttr("slot", builder.getI64IntegerAttr(0)),
         builder.getNamedAttr("init_value_id",
                              builder.getI64IntegerAttr(loop.init)),
         builder.getNamedAttr("iter_arg_value_id",
                              builder.getI64IntegerAttr(loop.iter)),
         builder.getNamedAttr("yielded_value_id",
                              builder.getI64IntegerAttr(loop.yielded)),
         builder.getNamedAttr("result_value_id",
                              builder.getI64IntegerAttr(loop.result)),
         builder.getNamedAttr("init_producer_compute_ids",
                              builder.getDenseI64ArrayAttr(loop.initProducers)),
         builder.getNamedAttr("iter_arg_consumer_compute_ids",
                              builder.getDenseI64ArrayAttr(loop.iterConsumers)),
         builder.getNamedAttr(
             "backedge_producer_compute_ids",
             builder.getDenseI64ArrayAttr(loop.backedgeProducers)),
         builder.getNamedAttr(
             "exit_consumer_compute_ids",
             builder.getDenseI64ArrayAttr(loop.exitConsumers))});
    if (annotate) {
      loop.op->setAttr("contract.value_ids",
                       builder.getDenseI64ArrayAttr({loop.iter, loop.result}));
      loop.op->setAttr("contract.loop_carried_slots",
                       builder.getArrayAttr({slot}));
    } else {
      auto slots =
          loop.op->getAttrOfType<ArrayAttr>("contract.loop_carried_slots");
      auto actual = slots && slots.size() == 1
                        ? dyn_cast<DictionaryAttr>(slots[0])
                        : DictionaryAttr();
      if (!actual || llvm::any_of(slot, [&](NamedAttribute attr) {
            return actual.get(attr.getName()) != attr.getValue();
          }))
        return fail(
            loop.op, "V11/V20/V26",
            "loop slot must match actual init, backedge and exit consumers");
    }
  }
  for (auto [id, preds] : predecessors) {
    for (int64_t pred : preds) {
      if (taskScopes[pred] == taskScopes[id])
        continue;
      bool legal = llvm::any_of(loops, [&](const LoopSummary &loop) {
        return (llvm::is_contained(loop.initProducers, pred) &&
                llvm::is_contained(loop.iterConsumers, id)) ||
               (llvm::is_contained(loop.backedgeProducers, pred) &&
                llvm::is_contained(loop.exitConsumers, id));
      });
      if (!legal)
        return fail(taskOps[id], "V19",
                    "cross-scope edge is not a registered loop init/exit edge");
    }
  }
  if (!annotate) {
    DenseMap<int64_t, SmallVector<int64_t>> users;
    for (auto [id, op] : taskOps)
      for (int64_t value :
           numbers(op->getAttrOfType<DictionaryAttr>("contract.task"),
                   "input_values"))
        users[value].push_back(id);
    for (const LoopSummary &loop : loops) {
      for (int64_t id : loop.iterConsumers)
        if (llvm::is_contained(
                numbers(
                    taskOps[id]->getAttrOfType<DictionaryAttr>("contract.task"),
                    "input_values"),
                loop.iter))
          users[loop.init].push_back(id);
      for (int64_t id : loop.exitConsumers)
        if (llvm::is_contained(
                numbers(
                    taskOps[id]->getAttrOfType<DictionaryAttr>("contract.task"),
                    "input_values"),
                loop.result))
          users[loop.yielded].push_back(id);
    }
    for (auto &[value, ids] : users) {
      llvm::sort(ids);
      ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    }
    LogicalResult valid = success();
    function.walk([&](Operation *op) {
      auto actual = op->getAttrOfType<ArrayAttr>("contract.value_users");
      if (!actual || failed(valid))
        return;
      SmallVector<int64_t> values;
      if (auto loopValues =
              op->getAttrOfType<DenseI64ArrayAttr>("contract.value_ids"))
        values = llvm::to_vector(loopValues.asArrayRef());
      else
        values = numbers(op->getAttrOfType<DictionaryAttr>("contract.task"),
                         "output_values");
      if (actual.size() != values.size()) {
        valid = fail(
            op, "V8",
            "reverse users must correspond to the defined content versions");
        return;
      }
      for (auto [value, attr] : llvm::zip(values, actual)) {
        auto ids = dyn_cast<DenseI64ArrayAttr>(attr);
        if (!ids || ids.asArrayRef() != ArrayRef<int64_t>(users[value])) {
          valid =
              fail(op, "V8",
                   "reverse users disagree with task inputs and loop edges");
          return;
        }
      }
    });
    if (failed(valid))
      return failure();
  }
  auto graph =
      function->getAttrOfType<DictionaryAttr>("contract.compute_graph");
  if (annotate) {
    NamedAttrList attrs;
    attrs.set("version", builder.getI32IntegerAttr(1));
    attrs.set("root_scope", builder.getI64IntegerAttr(0));
    attrs.set("scope_count", builder.getI64IntegerAttr(scopes.size()));
    attrs.set("complete", builder.getBoolAttr(complete));
    attrs.set("source_lineage_complete",
              builder.getBoolAttr(sourceFactsComplete));
    if (graph && graph.get("limitation"))
      attrs.set("limitation", graph.get("limitation"));
    else if (!complete && !scopes[0].limitation.empty())
      attrs.set("limitation", builder.getStringAttr(scopes[0].limitation));
    else if (!complete)
      attrs.set(
          "limitation",
          builder.getStringAttr(
              "one or more scopes have unmodeled effects or control flow"));
    function->setAttr("contract.compute_graph", attrs.getDictionary(context));
  } else if (!graph || integer(graph, "version") != 1 ||
             integer(graph, "root_scope") != 0 ||
             integer(graph, "scope_count") != int64_t(scopes.size()) ||
             !graph.getAs<BoolAttr>("complete") ||
             boolean(graph, "complete") != complete) {
    return fail(
        function, "V18",
        "graph completeness/scope count does not match the actual scope tree");
  }
  if (!annotate &&
      (!graph.getAs<BoolAttr>("source_lineage_complete") ||
       boolean(graph, "source_lineage_complete") != sourceFactsComplete))
    return fail(function, "V12",
                "source lineage completeness does not match the retained "
                "source facts and actual memory boundaries");
  return success();
}

LogicalResult FunctionContract::run() {
  if (failed(checkSchema(function)))
    return failure();
  auto graph =
      function->getAttrOfType<DictionaryAttr>("contract.compute_graph");
  if (!annotate && failed(checkDeclaredGraph()))
    return failure();
  scopes[0] = Scope{};
  // Reannotation may remove tasks, but may never invent missing information.
  if (graph && !boolean(graph, "complete", true)) {
    scopes[0].complete = false;
    scopes[0].limitation = string(graph, "limitation").str();
    if (!annotate && scopes[0].limitation.empty())
      return fail(function, "V15",
                  "incomplete graph must identify its unmodeled boundary");
  }
  if (hasEntryArgumentABI(function))
    argsDisjoint = boolean(
        function->getParentOfType<ModuleOp>()->getAttrOfType<DictionaryAttr>(
            "contract.abi"),
        "args_disjoint");
  Environment environment;
  for (auto [index, argument] : llvm::enumerate(function.getArguments())) {
    if (auto program = function.getArgAttrOfType<DictionaryAttr>(
            index, "contract.program")) {
      auto axis = integer(program, "axis");
      StringRef role = string(program, "role");
      if (!axis || *axis < 0 || *axis >= 3 ||
          (role != "program_id" && role != "num_programs"))
        return fail(function, "V10", "invalid program argument role/axis");
    }
    if (!isa<BaseMemRefType>(argument.getType())) {
      if (!argument.getType().isIntOrIndexOrFloat())
        return fail(function, "V3",
                    "arguments must be memrefs or ordinary scalars");
      if (function.getArgAttr(index, "contract.value_id"))
        return fail(function, "V7",
                    "ordinary scalar arguments do not define content versions");
      continue;
    }
    auto old =
        function.getArgAttrOfType<IntegerAttr>(index, "contract.value_id");
    if (!annotate && !old)
      return fail(function, "V7",
                  "memref argument requires an entry content version");
    int64_t value = annotate ? nextValue : old.getInt();
    if (failed(registerValue(function, value, argument)))
      return failure();
    environment.versions[argument] = value;
    valueLineages[value] = Lineage{{}, false};
    if (annotate)
      function.setArgAttr(index, "contract.value_id",
                          builder.getI64IntegerAttr(value));
  }
  if (failed(processRegion(function.getBody(), 0, environment)))
    return failure();
  return finish();
}

} // namespace

//===----------------------------------------------------------------------===//
// Public detail:: API
//===----------------------------------------------------------------------===//

// True only for module-entry functions with no symbol uses in the module.
// Helpers reached by call / address-taken do not inherit contract.abi.
bool hasEntryArgumentABI(func::FuncOp function) {
  if (!function)
    return false;
  auto module = function->getParentOfType<ModuleOp>();
  return module && module->hasAttr("contract.abi") &&
         SymbolTable::symbolKnownUseEmpty(function, module);
}

LogicalResult checkKernelContractVersions(ModuleOp module) {
  return failure(
      module
          .walk([](Operation *op) -> WalkResult {
            if (Attribute attr = op->getAttr("contract.version")) {
              auto version = dyn_cast<IntegerAttr>(attr);
              if (!isa<ModuleOp>(op) || !version || version.getInt() != 1) {
                op->emitOpError("kernel contract V1: unsupported existing "
                                "contract.version");
                return WalkResult::interrupt();
              }
            }
            if (Attribute attr = op->getAttr("contract.compute_graph")) {
              auto graph = dyn_cast<DictionaryAttr>(attr);
              auto version =
                  graph ? graph.getAs<IntegerAttr>("version") : IntegerAttr();
              if (!version || version.getInt() != 1) {
                op->emitOpError(
                    "kernel contract V18: unsupported existing graph version");
                return WalkResult::interrupt();
              }
            }
            return WalkResult::advance();
          })
          .wasInterrupted());
}

// Full annotate path: version/representation guards, optional producer
// normalization, then per-function FunctionContract in annotate mode.
// Full annotate path: version/representation guards, optional producer
// normalization, then per-function FunctionContract in annotate mode.
static LogicalResult annotate(ModuleOp module, StringRef producer, bool planned,
                              SourcePolicy sourcePolicy) {
  // Never turn an unknown version into v1, even when normalization would
  // otherwise replace the operation carrying the version.
  if (failed(checkKernelContractVersions(module)) ||
      failed(checkRepresentation(module, planned)))
    return failure();
  if (!planned) {
    makeScalarInputsExplicit(module);
    normalizeReachingZeroCopies(module);
  }
  Builder builder(module.getContext());
  module->setAttr("contract.version", builder.getI32IntegerAttr(1));
  module->setAttr("contract.producer", builder.getStringAttr(producer));
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    if (!function.isExternal() &&
        failed(FunctionContract(function, true, planned, sourcePolicy).run()))
      return failure();
  return success();
}

static LogicalResult verify(ModuleOp module, bool planned,
                            SourcePolicy sourcePolicy) {
  if (failed(checkKernelContractVersions(module)) ||
      failed(checkRepresentation(module, planned)))
    return failure();
  auto version = module->getAttrOfType<IntegerAttr>("contract.version");
  if (!version || version.getInt() != 1)
    return module.emitOpError(
        "kernel contract V1: supported contract.version = 1 is required");
  if (!module->getAttrOfType<StringAttr>("contract.producer"))
    return module.emitOpError(
        "kernel contract V1: contract.producer is required");
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    if (!function.isExternal() &&
        failed(FunctionContract(function, false, planned, sourcePolicy).run()))
      return failure();
  return success();
}

LogicalResult annotateKernelContract(ModuleOp module, StringRef producer,
                                     SourcePolicy sourcePolicy) {
  return annotate(module, producer, false, sourcePolicy);
}
LogicalResult verifyKernelContract(ModuleOp module, SourcePolicy sourcePolicy) {
  return verify(module, false, sourcePolicy);
}
LogicalResult rebuildPlannedContract(ModuleOp module, StringRef producer,
                                     SourcePolicy sourcePolicy) {
  return annotate(module, producer, true, sourcePolicy);
}
LogicalResult verifyPlannedContract(ModuleOp module,
                                    SourcePolicy sourcePolicy) {
  return verify(module, true, sourcePolicy);
}

} // namespace buddy::contract::detail
