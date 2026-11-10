//===- BOSCAMEStreamPlan.cpp - Virtual values and ordered DDR streams -----===//
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
// Analyze, verify, and emit content-version streaming plans for pointwise and
// reduction sinks. Virtual values stay logical until emission expands verified
// recipes into ordered physical execution anchors.
//
//===----------------------------------------------------------------------===//

#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEStreamPlan.h"
#include "Analysis/KernelContract.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEMathApprox.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEPointwise.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <memory>
#include <optional>

using namespace mlir;

namespace buddy::boscame {
namespace {

constexpr StringLiteral kStream = "plan.stream";
constexpr StringLiteral kVirtual = "plan.stream_virtual";

Value root(Value value) {
  while (auto view =
             dyn_cast_or_null<ViewLikeOpInterface>(value.getDefiningOp()))
    value = view.getViewSource();
  return value;
}

bool privateRoot(Value value) {
  return isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(
      root(value).getDefiningOp());
}

bool smallRank(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && type.getRank() <= 1 && type.hasStaticShape() &&
         llvm::all_of(type.getShape(), [](int64_t n) { return n > 0; }) &&
         supportedPointwiseScalar(type.getElementType());
}

int64_t elements(Value value) {
  auto type = cast<MemRefType>(value.getType());
  return type.getRank() ? type.getDimSize(0) : 1;
}

std::optional<int64_t> taskId(Operation *op) {
  auto task = op->getAttrOfType<DictionaryAttr>("contract.task");
  auto id = task ? task.getAs<IntegerAttr>("id") : IntegerAttr();
  return id ? std::optional<int64_t>(id.getInt()) : std::nullopt;
}

struct View {
  Value storage;
  int64_t offset = 0;
  int64_t stride = 1;
  int64_t size = 1;
};

// Logical indices of a private alias, independently of its physical layout.
// Do not interpret descriptor offset zero as proof that a shifted view is a
// full-root snapshot.
std::optional<View> privateView(Value value) {
  if (!smallRank(value))
    return std::nullopt;
  if (isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(value.getDefiningOp()))
    return View{value, 0, 1, elements(value)};
  if (auto cast = value.getDefiningOp<memref::CastOp>()) {
    auto result = privateView(cast.getSource());
    if (result && result->size == elements(value))
      return result;
    return std::nullopt;
  }
  auto subview = value.getDefiningOp<memref::SubViewOp>();
  if (!subview || subview.getSourceType().getRank() != 1 ||
      subview.getType().getRank() != 1 ||
      subview.getStaticOffsets()[0] == ShapedType::kDynamic ||
      subview.getStaticStrides()[0] == ShapedType::kDynamic)
    return std::nullopt;
  auto parent = privateView(subview.getSource());
  if (!parent)
    return std::nullopt;
  int64_t offset, stride, delta;
  if (llvm::MulOverflow(subview.getStaticOffsets()[0], parent->stride, delta) ||
      llvm::AddOverflow(parent->offset, delta, offset) ||
      llvm::MulOverflow(subview.getStaticStrides()[0], parent->stride,
                        stride) ||
      offset < 0 || stride <= 0)
    return std::nullopt;
  return View{parent->storage, offset, stride, elements(value)};
}

bool supportedGeneric(linalg::GenericOp op) {
  if (!op.hasPureBufferSemantics() || op.getOutputs().size() != 1 ||
      !smallRank(op.getOutputs()[0]) ||
      cast<MemRefType>(op.getOutputs()[0].getType()).getRank() != 1 ||
      op.getNumLoops() != 1 || op.getNumParallelLoops() != 1)
    return false;
  auto maps = op.getIndexingMapsArray();
  for (auto [value, map] : llvm::zip(op.getOperands(), maps)) {
    if (isa<MemRefType>(value.getType())) {
      if (!smallRank(value) || !map.isIdentity() ||
          elements(value) != elements(op.getOutputs()[0]))
        return false;
    } else if (!supportedPointwiseScalar(value.getType()) ||
               map.getNumResults() != 0) {
      return false;
    }
  }
  return llvm::all_of(
      op.getBody()->without_terminator(),
      [](Operation &payload) { return supportedPointwiseOperation(payload); });
}

bool supportedReduce(linalg::ReduceOp op) {
  if (op.getInputs().size() != 1 || op.getInits().size() != 1 ||
      !smallRank(op.getInputs()[0]) || !smallRank(op.getInits()[0]))
    return false;
  auto input = cast<MemRefType>(op.getInputs()[0].getType());
  auto output = cast<MemRefType>(op.getInits()[0].getType());
  if (input.getRank() != 1 || output.getRank() != 0 ||
      !input.getElementType().isF32() || !output.getElementType().isF32() ||
      op.getDimensions() != ArrayRef<int64_t>{0} || input.getDimSize(0) % 16)
    return false;
  Block &body = op.getRegion().front();
  if (std::distance(body.begin(), body.end()) != 2)
    return false;
  Operation &arithmetic = body.front();
  auto yield = dyn_cast<linalg::YieldOp>(body.back());
  if ((!isa<arith::AddFOp, arith::MaxNumFOp>(arithmetic)) ||
      arithmetic.getNumOperands() != 2 ||
      arithmetic.getOperand(0) != body.getArgument(0) ||
      arithmetic.getOperand(1) != body.getArgument(1) || !yield ||
      yield.getValues()[0] != arithmetic.getResult(0))
    return false;
  if (auto add = dyn_cast<arith::AddFOp>(arithmetic))
    return add.getFastmath() == arith::FastMathFlags::none;
  return true;
}

bool compatibleUser(Operation *op, Block *block) {
  if (op->getBlock() != block || op->hasAttr("plan.register_member") ||
      op->hasAttr("plan.register_schedule") ||
      op->hasAttr("plan.float_member") || op->hasAttr("plan.float_schedule"))
    return false;
  if (auto copy = dyn_cast<memref::CopyOp>(op))
    return smallRank(copy.getSource()) && smallRank(copy.getTarget()) &&
           elements(copy.getSource()) == elements(copy.getTarget());
  if (auto fill = dyn_cast<linalg::FillOp>(op))
    return fill.getOutputs().size() == 1 && smallRank(fill.getOutputs()[0]);
  if (auto generic = dyn_cast<linalg::GenericOp>(op))
    return supportedGeneric(generic);
  if (auto reduce = dyn_cast<linalg::ReduceOp>(op))
    return supportedReduce(reduce);
  return isa<memref::LoadOp, memref::DeallocOp>(op);
}

bool eligibleRoot(Value value, Block *block) {
  if (!smallRank(value) ||
      value.getDefiningOp()->hasAttr("plan.register_virtual") ||
      value.getDefiningOp()->hasAttr("plan.float_virtual"))
    return false;
  SmallVector<Value> work{value};
  DenseSet<Value> seen;
  while (!work.empty()) {
    Value alias = work.pop_back_val();
    if (!seen.insert(alias).second)
      continue;
    for (Operation *user : alias.getUsers()) {
      if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
        if (user->getBlock() != block || view.getViewSource() != alias ||
            user->getNumResults() != 1 || !privateView(user->getResult(0)))
          return false;
        work.push_back(user->getResult(0));
      } else if (!compatibleUser(user, block)) {
        return false;
      }
    }
  }
  return true;
}

struct Expr;
using Expression = std::shared_ptr<Expr>;
struct Expr {
  enum Kind { Undefined, Scalar, Read, Apply, Patch, Slice, Reduction } kind;
  Operation *operation = nullptr;
  Value value;
  SmallVector<Expression> args;
  int64_t offset = 0;
  int64_t size = 0;
  int64_t stride = 1;
};

Expression expression(Expr::Kind kind, Operation *op = nullptr,
                      Value value = {}, ArrayRef<Expression> args = {}) {
  auto result = std::make_shared<Expr>();
  result->kind = kind;
  result->operation = op;
  result->value = value;
  result->args.append(args.begin(), args.end());
  return result;
}

struct Entry {
  Operation *op;
  StringRef kind;
  Expression value;
  Expression initial;
  int64_t size = 0;
  Value destination;
  SmallVector<int64_t> dependencies;
  bool sink() const { return kind != "virtual"; }
};

struct BlockPlan {
  Block *block;
  SmallVector<Value> roots;
  SmallVector<Entry> entries;
};

// Every vector loop starts at a multiple of the logical 16-lane chunk size.
// Split only at aligned content-version boundaries: a nonaligned boundary
// must be handled by a scalar store or by the materialized DDR fallback.
void collectVectorBounds(Expression expr, int64_t shift, int64_t size,
                         SmallVectorImpl<int64_t> &bounds) {
  if (!expr)
    return;
  if (expr->kind == Expr::Slice) {
    collectVectorBounds(expr->args[0], shift + expr->offset, size, bounds);
    return;
  }
  if (expr->kind == Expr::Patch) {
    for (int64_t edge : {expr->offset, expr->offset + expr->size}) {
      int64_t boundary = edge - shift;
      if (boundary > 0 && boundary < size && boundary % 16 == 0)
        bounds.push_back(boundary);
    }
    collectVectorBounds(expr->args[0], shift, size, bounds);
    collectVectorBounds(expr->args[1], shift - expr->offset, size, bounds);
    return;
  }
  for (Expression arg : expr->args)
    collectVectorBounds(arg, shift, size, bounds);
}

SmallVector<int64_t> vectorBounds(Expression expr, int64_t size) {
  if (!size)
    return {};
  SmallVector<int64_t> bounds{0, size};
  collectVectorBounds(expr, 0, size, bounds);
  llvm::sort(bounds);
  bounds.erase(std::unique(bounds.begin(), bounds.end()), bounds.end());
  return bounds;
}

bool vectorizable(Expression expr, int64_t begin, int64_t end) {
  if (!expr)
    return true;
  if (expr->kind == Expr::Undefined)
    return false;
  if (expr->kind == Expr::Read) {
    auto type = cast<MemRefType>(expr->value.getType());
    SmallVector<int64_t> strides;
    int64_t offset;
    return type.getRank() == 0 ||
           (succeeded(type.getStridesAndOffset(strides, offset)) &&
            strides[0] == 1);
  }
  if (expr->kind == Expr::Slice)
    return expr->stride == 1 &&
           vectorizable(expr->args[0], begin + expr->offset,
                        end + expr->offset);
  if (expr->kind == Expr::Patch) {
    if (begin >= expr->offset && end <= expr->offset + expr->size)
      return vectorizable(expr->args[1], begin - expr->offset,
                          end - expr->offset);
    if (end <= expr->offset || begin >= expr->offset + expr->size)
      return vectorizable(expr->args[0], begin, end);
    return false;
  }
  if (expr->kind == Expr::Apply) {
    auto generic = cast<linalg::GenericOp>(expr->operation);
    for (Operation &payload : generic.getBody()->without_terminator())
      if (!pointwiseMathRecipe(payload).empty())
        return false;
  }
  return llvm::all_of(expr->args, [&](Expression arg) {
    return vectorizable(arg, begin, end);
  });
}

bool vectorizable(Expression expr, ArrayRef<int64_t> bounds) {
  for (size_t i = 1; i < bounds.size(); ++i)
    if (!vectorizable(expr, bounds[i - 1], bounds[i]))
      return false;
  return true;
}

int64_t vectorPrefix(const Entry &entry) {
  if (entry.size < 16)
    return 0;
  SmallVector<int64_t> strides;
  int64_t offset;
  auto output = cast<MemRefType>(entry.destination.getType());
  if (failed(output.getStridesAndOffset(strides, offset)) ||
      strides.size() != 1 || strides[0] != 1)
    return 0;
  int64_t prefix = entry.size / 16 * 16;
  return vectorizable(entry.value, vectorBounds(entry.value, prefix)) ? prefix
                                                                      : 0;
}

class StreamAnalysis {
public:
  explicit StreamAnalysis(ModuleOp module, ArrayRef<Value> fixed,
                          bool recordedOnly = false,
                          bool checkSourceStability = true)
      : module(module), excluded(fixed.begin(), fixed.end()),
        recordedOnly(recordedOnly), checkSourceStability(checkSourceStability) {
  }
  SmallVector<BlockPlan> run();

private:
  ModuleOp module;
  DenseSet<Value> excluded;
  DenseSet<Value> candidates;
  DenseMap<Value, Expression> versions;
  DenseMap<Value, Expression> scalarLoads;
  bool recordedOnly;
  bool checkSourceStability;
  bool fixed(Value value);
  bool disjoint(Value first, Value second);
  bool stable(Expression expr, Operation *sink);
  bool defined(Expression expr, int64_t begin, int64_t end);
  Expression read(Value value, Operation *at);
  bool write(Value target, Expression value, Operation *op);
  bool build(BlockPlan &plan);
  void dependencies(Expression expr, DenseSet<int64_t> &ids,
                    DenseSet<const Expr *> &seen);
  void scalarDependencies(Value value, DenseSet<int64_t> &ids,
                          DenseSet<Value> &seen);
};

bool StreamAnalysis::fixed(Value value) {
  Value storage = root(value);
  if (excluded.contains(storage))
    return true;
  Operation *definition = storage.getDefiningOp();
  auto placement =
      definition ? definition->getAttrOfType<DictionaryAttr>("plan.buffer")
                 : DictionaryAttr();
  return placement && placement.getAs<StringAttr>("level") == "spm";
}

bool StreamAnalysis::disjoint(Value first, Value second) {
  first = root(first);
  second = root(second);
  if (first == second)
    return false;
  if (privateRoot(first) || privateRoot(second))
    return true;
  auto a = dyn_cast<BlockArgument>(first);
  auto b = dyn_cast<BlockArgument>(second);
  if (!a || !b || a.getOwner() != b.getOwner())
    return false;
  auto function = dyn_cast_or_null<func::FuncOp>(a.getOwner()->getParentOp());
  auto abi = module->getAttrOfType<DictionaryAttr>("contract.abi");
  auto promise = abi ? abi.getAs<BoolAttr>("args_disjoint") : BoolAttr();
  return function && contract::hasEntryArgumentABI(function) && promise &&
         promise.getValue();
}

bool StreamAnalysis::stable(Expression expr, Operation *sink) {
  if (!checkSourceStability)
    return true;
  if (!expr)
    return true;
  // SSA results have already been computed by their own sink. Traversing
  // their logical dependencies here would incorrectly prove a repeated read.
  if (expr->kind == Expr::Scalar || expr->kind == Expr::Reduction)
    return true;
  if (expr->kind == Expr::Read) {
    Operation *begin = expr->operation;
    if (begin->getBlock() != sink->getBlock())
      return false;
    for (Operation *op = begin; op; op = op->getNextNode()) {
      bool safe = true;
      if (auto copy = dyn_cast<memref::CopyOp>(op))
        safe = disjoint(expr->value, copy.getTarget());
      else if (auto linalg = dyn_cast<linalg::LinalgOp>(op)) {
        for (Value target : linalg.getDpsInits())
          safe &= disjoint(expr->value, target);
        for (Region &region : op->getRegions())
          region.walk(
              [&](Operation *payload) { safe &= isMemoryEffectFree(payload); });
      } else if (!isa<memref::LoadOp, memref::AllocOp, memref::AllocaOp>(op) &&
                 !isMemoryEffectFree(op)) {
        safe = false;
      }
      if (!safe)
        return false;
      if (op == sink)
        return true;
    }
    return false;
  }
  return llvm::all_of(expr->args,
                      [&](Expression arg) { return stable(arg, sink); });
}

bool StreamAnalysis::defined(Expression expr, int64_t begin, int64_t end) {
  if (!expr || begin >= end)
    return true;
  switch (expr->kind) {
  case Expr::Undefined:
    return false;
  case Expr::Read:
    return begin >= 0 && end <= elements(expr->value);
  case Expr::Scalar:
  case Expr::Reduction:
    return true;
  case Expr::Apply:
    return llvm::all_of(
        expr->args, [&](Expression arg) { return defined(arg, begin, end); });
  case Expr::Slice:
    return defined(expr->args[0], expr->offset + begin * expr->stride,
                   expr->offset + (end - 1) * expr->stride + 1);
  case Expr::Patch: {
    int64_t lo = std::max(begin, expr->offset);
    int64_t hi = std::min(end, expr->offset + expr->size);
    return (lo >= hi ||
            defined(expr->args[1], lo - expr->offset, hi - expr->offset)) &&
           (begin >= expr->offset ||
            defined(expr->args[0], begin, std::min(end, expr->offset))) &&
           (end <= expr->offset + expr->size ||
            defined(expr->args[0], std::max(begin, expr->offset + expr->size),
                    end));
  }
  }
  llvm_unreachable("unknown stream expression");
}

Expression StreamAnalysis::read(Value value, Operation *at) {
  if (!isa<MemRefType>(value.getType()))
    return expression(Expr::Scalar, nullptr, value);
  if (!candidates.contains(root(value))) {
    if (fixed(value))
      return {};
    return expression(Expr::Read, at, value);
  }
  auto view = privateView(value);
  if (!view)
    return {};
  Expression current = versions.lookup(view->storage);
  if (!current)
    return {};
  if (!view->offset && view->stride == 1)
    return current;
  auto result = expression(Expr::Slice, nullptr, {}, {current});
  result->offset = view->offset;
  result->stride = view->stride;
  return result;
}

bool StreamAnalysis::write(Value target, Expression value, Operation *op) {
  auto view = privateView(target);
  if (!view || view->stride != 1 || !candidates.contains(view->storage) ||
      view->offset > elements(view->storage) - view->size)
    return false;
  if (!view->offset && view->size == elements(view->storage)) {
    versions[view->storage] = value;
    return true;
  }
  auto result =
      expression(Expr::Patch, op, {}, {versions.lookup(view->storage), value});
  result->offset = view->offset;
  result->size = view->size;
  versions[view->storage] = result;
  return true;
}

void StreamAnalysis::scalarDependencies(Value value, DenseSet<int64_t> &ids,
                                        DenseSet<Value> &seen) {
  if (!value || !seen.insert(value).second)
    return;
  Operation *op = value.getDefiningOp();
  if (!op)
    return;
  if (scalarLoads.count(value)) {
    if (auto id = taskId(op))
      ids.insert(*id);
    return;
  }
  if (isMemoryEffectFree(op))
    for (Value operand : op->getOperands())
      scalarDependencies(operand, ids, seen);
}

void StreamAnalysis::dependencies(Expression expr, DenseSet<int64_t> &ids,
                                  DenseSet<const Expr *> &seen) {
  if (!expr || !seen.insert(expr.get()).second)
    return;
  if (expr->operation)
    if (auto id = taskId(expr->operation))
      ids.insert(*id);
  if (expr->kind == Expr::Reduction)
    return;
  if (expr->kind == Expr::Scalar) {
    DenseSet<Value> scalarSeen;
    scalarDependencies(expr->value, ids, scalarSeen);
    return;
  }
  for (Expression arg : expr->args)
    dependencies(arg, ids, seen);
}

bool StreamAnalysis::build(BlockPlan &plan) {
  versions.clear();
  scalarLoads.clear();
  for (Value storage : plan.roots)
    versions[storage] = expression(Expr::Undefined);
  for (Operation &operation : *plan.block) {
    Operation *op = &operation;
    if (auto fill = dyn_cast<linalg::FillOp>(op)) {
      if (!candidates.contains(root(fill.getOutputs()[0])))
        continue;
      auto value = expression(Expr::Scalar, op, fill.getInputs()[0]);
      if (!write(fill.getOutputs()[0], value, op))
        return false;
      plan.entries.push_back({op, "virtual", value});
    } else if (auto copy = dyn_cast<memref::CopyOp>(op)) {
      bool source = candidates.contains(root(copy.getSource()));
      bool target = candidates.contains(root(copy.getTarget()));
      if (!source && !target)
        continue;
      Expression value = read(copy.getSource(), op);
      if (!value)
        return false;
      // Keep the copy itself in logical provenance even if its source is an
      // already virtual expression rather than a fresh external read.
      auto copyValue = expression(Expr::Slice, op, {}, {value});
      if (target) {
        if (!write(copy.getTarget(), copyValue, op))
          return false;
        plan.entries.push_back({op, "virtual", copyValue});
      } else {
        if (fixed(copy.getTarget()) ||
            !defined(copyValue, 0, elements(copy.getSource())) ||
            !stable(copyValue, op))
          return false;
        plan.entries.push_back({op,
                                "store",
                                copyValue,
                                {},
                                elements(copy.getSource()),
                                copy.getTarget()});
      }
    } else if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
      bool target = candidates.contains(root(generic.getOutputs()[0]));
      bool source = llvm::any_of(generic.getInputs(), [&](Value value) {
        return isa<MemRefType>(value.getType()) &&
               candidates.contains(root(value));
      });
      if (!target && !source)
        continue;
      if (!supportedGeneric(generic))
        return false;
      auto value = expression(Expr::Apply, op);
      for (auto [operand, argument] : llvm::zip(
               generic.getOperands(), generic.getBody()->getArguments())) {
        Expression input =
            argument.use_empty() ? Expression() : read(operand, op);
        if (!argument.use_empty() && !input)
          return false;
        value->args.push_back(input);
      }
      if (target) {
        if (!write(generic.getOutputs()[0], value, op))
          return false;
        plan.entries.push_back({op, "virtual", value});
      } else {
        if (fixed(generic.getOutputs()[0]) ||
            !defined(value, 0, elements(generic.getOutputs()[0])) ||
            !stable(value, op))
          return false;
        plan.entries.push_back({op,
                                "store",
                                value,
                                {},
                                elements(generic.getOutputs()[0]),
                                generic.getOutputs()[0]});
      }
    } else if (auto reduce = dyn_cast<linalg::ReduceOp>(op)) {
      bool source = candidates.contains(root(reduce.getInputs()[0]));
      bool target = candidates.contains(root(reduce.getInits()[0]));
      if (!source && !target)
        continue;
      if (!target || !supportedReduce(reduce))
        return false;
      Expression value = read(reduce.getInputs()[0], op);
      Expression initial = read(reduce.getInits()[0], op);
      if (!value || !initial ||
          !defined(value, 0, elements(reduce.getInputs()[0])) ||
          !defined(initial, 0, 1) || !stable(value, op) || !stable(initial, op))
        return false;
      plan.entries.push_back(
          {op, "reduce", value, initial, elements(reduce.getInputs()[0])});
      if (!write(reduce.getInits()[0], expression(Expr::Reduction, op), op))
        return false;
    } else if (auto load = dyn_cast<memref::LoadOp>(op)) {
      if (!candidates.contains(root(load.getMemRef())))
        continue;
      Expression value = read(load.getMemRef(), op);
      if (!value || !defined(value, 0, elements(load.getMemRef())) ||
          !stable(value, op))
        return false;
      scalarLoads[load.getResult()] = value;
      plan.entries.push_back({op, "scalar-load", value, {}, 1});
    }
  }
  if (plan.entries.empty())
    return false;
  // A nonvirtual generic cannot keep reading a buffer that is about to vanish.
  DenseSet<Operation *> members;
  for (const Entry &entry : plan.entries)
    members.insert(entry.op);
  for (Value storage : plan.roots) {
    SmallVector<Value> aliases{storage};
    while (!aliases.empty()) {
      Value alias = aliases.pop_back_val();
      for (Operation *user : alias.getUsers()) {
        if (isa<ViewLikeOpInterface>(user))
          aliases.push_back(user->getResult(0));
        else if (!members.contains(user) && !isa<memref::DeallocOp>(user))
          return false;
      }
    }
  }
  for (Entry &entry : plan.entries) {
    // Ordered reductions keep the exact logical groups of 16 lanes. If a
    // group crosses a virtual-write boundary, do not invent vector inserts,
    // permutes or an unplanned packing buffer in the emitter.
    if (entry.kind == "reduce" &&
        !vectorizable(entry.value, vectorBounds(entry.value, entry.size)))
      return false;
    DenseSet<int64_t> ids;
    DenseSet<const Expr *> seen;
    dependencies(entry.value, ids, seen);
    dependencies(entry.initial, ids, seen);
    if (auto id = taskId(entry.op))
      ids.insert(*id);
    entry.dependencies.append(ids.begin(), ids.end());
    llvm::sort(entry.dependencies);
  }
  return true;
}

SmallVector<BlockPlan> StreamAnalysis::run() {
  SmallVector<BlockPlan> result;
  llvm::SetVector<Block *> blocks;
  module.walk([&](Operation *op) {
    if (isa<memref::AllocOp, memref::AllocaOp>(op))
      blocks.insert(op->getBlock());
  });
  for (Block *block : blocks) {
    BlockPlan plan{block};
    candidates.clear();
    for (Operation &op : *block) {
      if (!isa<memref::AllocOp, memref::AllocaOp>(op))
        continue;
      Value value = op.getResult(0);
      if (recordedOnly && !isStreamVirtual(&op))
        continue;
      auto placement = op.getAttrOfType<DictionaryAttr>("plan.buffer");
      auto level =
          placement ? placement.getAs<StringAttr>("level") : StringAttr();
      if (excluded.contains(value) || (level && level == "spm") ||
          !eligibleRoot(value, block))
        continue;
      candidates.insert(value);
      plan.roots.push_back(value);
    }
    if (!plan.roots.empty() && build(plan))
      result.push_back(std::move(plan));
  }
  return result;
}

DictionaryAttr recipe(Builder &b, const Entry &entry) {
  SmallVector<NamedAttribute> attrs{
      b.getNamedAttr("kind", b.getStringAttr(entry.kind)),
      b.getNamedAttr("logical_inputs",
                     b.getDenseI64ArrayAttr(entry.dependencies))};
  if (entry.sink())
    attrs.push_back(
        b.getNamedAttr("elements", b.getI64IntegerAttr(entry.size)));
  if (entry.kind == "reduce") {
    attrs.push_back(b.getNamedAttr("lanes", b.getI64IntegerAttr(16)));
    attrs.push_back(b.getNamedAttr("fp_policy", b.getStringAttr("preserve")));
    attrs.push_back(b.getNamedAttr("engine", b.getStringAttr("rvv")));
    attrs.push_back(
        b.getNamedAttr("input_generation", b.getStringAttr("vector")));
    attrs.push_back(b.getNamedAttr(
        "vector_bounds",
        b.getDenseI64ArrayAttr(vectorBounds(entry.value, entry.size))));
  } else if (entry.kind == "store") {
    int64_t prefix = vectorPrefix(entry);
    attrs.push_back(
        b.getNamedAttr("vector_elements", b.getI64IntegerAttr(prefix)));
    attrs.push_back(
        b.getNamedAttr("lanes", b.getI64IntegerAttr(prefix ? 16 : 1)));
    attrs.push_back(b.getNamedAttr(
        "engine", b.getStringAttr(!prefix                ? "scalar"
                                  : prefix == entry.size ? "rvv"
                                                         : "rvv-prefix")));
    attrs.push_back(b.getNamedAttr(
        "vector_bounds",
        b.getDenseI64ArrayAttr(vectorBounds(entry.value, prefix))));
  } else if (entry.kind == "scalar-load") {
    attrs.push_back(b.getNamedAttr("engine", b.getStringAttr("scalar")));
  }
  return b.getDictionaryAttr(attrs);
}

DictionaryAttr archive(Builder &b, Operation *op) {
  SmallVector<NamedAttribute> fields;
  for (StringRef key : {"task", "provenance", "access", "fill", "compute"})
    if (Attribute value = op->getAttr(("contract." + key).str()))
      fields.push_back(b.getNamedAttr(key, value));
  return b.getDictionaryAttr(fields);
}

Value index(OpBuilder &b, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(b, loc, value);
}

class StreamEmitter {
public:
  StreamEmitter(ModuleOp module, ArrayRef<BlockPlan> plans)
      : module(module), plans(plans) {}
  LogicalResult run();

private:
  ModuleOp module;
  ArrayRef<BlockPlan> plans;
  DenseMap<Value, Value> replacements;
  DenseMap<Operation *, Value> reductions;
  Value emit(Expression expr, Value position, OpBuilder &b, Location loc);
  Type type(Expression expr, OpBuilder &b);
  Value emitVector(Expression expr, Value position, int64_t begin, int64_t end,
                   OpBuilder &b, Location loc);
  Value scalar(Value value);
  Operation *sink(const Entry &entry);
};

Value StreamEmitter::scalar(Value value) {
  auto found = replacements.find(value);
  return found == replacements.end() ? value : found->second;
}

Type StreamEmitter::type(Expression expr, OpBuilder &b) {
  switch (expr->kind) {
  case Expr::Undefined:
    return {};
  case Expr::Scalar:
    return expr->value.getType();
  case Expr::Read:
    return cast<MemRefType>(expr->value.getType()).getElementType();
  case Expr::Apply:
    return cast<linalg::GenericOp>(expr->operation)
        .getBody()
        ->getTerminator()
        ->getOperand(0)
        .getType();
  case Expr::Patch:
    return type(expr->args[1], b);
  case Expr::Slice:
    return type(expr->args[0], b);
  case Expr::Reduction:
    return b.getF32Type();
  }
  llvm_unreachable("unknown stream expression");
}

Value StreamEmitter::emitVector(Expression expr, Value position, int64_t begin,
                                int64_t end, OpBuilder &b, Location loc) {
  auto broadcast = [&](Value scalar) -> Value {
    return vector::BroadcastOp::create(
        b, loc, VectorType::get({16}, scalar.getType()), scalar);
  };
  switch (expr->kind) {
  case Expr::Undefined:
    return {};
  case Expr::Scalar:
    return broadcast(scalar(expr->value));
  case Expr::Reduction:
    return broadcast(reductions.lookup(expr->operation));
  case Expr::Read: {
    auto memory = cast<MemRefType>(expr->value.getType());
    if (!memory.getRank())
      return broadcast(emit(expr, position, b, loc));
    auto loaded = vector::LoadOp::create(
        b, loc, VectorType::get({16}, memory.getElementType()), expr->value,
        ValueRange{position});
    loaded->setAttr("bosc_ame.direct_ddr", b.getUnitAttr());
    return loaded;
  }
  case Expr::Slice: {
    Value translated = position;
    if (expr->offset)
      translated = b.createOrFold<arith::AddIOp>(loc, translated,
                                                 index(b, loc, expr->offset));
    return emitVector(expr->args[0], translated, begin + expr->offset,
                      end + expr->offset, b, loc);
  }
  case Expr::Patch: {
    if (begin >= expr->offset && end <= expr->offset + expr->size) {
      Value local = b.createOrFold<arith::SubIOp>(loc, position,
                                                  index(b, loc, expr->offset));
      return emitVector(expr->args[1], local, begin - expr->offset,
                        end - expr->offset, b, loc);
    }
    assert((end <= expr->offset || begin >= expr->offset + expr->size) &&
           "verified vector interval crosses a content-version boundary");
    return emitVector(expr->args[0], position, begin, end, b, loc);
  }
  case Expr::Apply: {
    auto generic = cast<linalg::GenericOp>(expr->operation);
    IRMapping mapping;
    for (auto [argument, input] :
         llvm::zip(generic.getBody()->getArguments(), expr->args))
      if (input)
        mapping.map(argument, emitVector(input, position, begin, end, b, loc));
    for (Operation &payload : generic.getBody()->without_terminator()) {
      for (Value operand : payload.getOperands())
        if (!mapping.contains(operand))
          mapping.map(operand, broadcast(scalar(operand)));
      auto resultType = VectorType::get({16}, payload.getResult(0).getType());
      if (auto constant = dyn_cast<arith::ConstantOp>(payload)) {
        mapping.map(payload.getResult(0), broadcast(arith::ConstantOp::create(
                                              b, loc, constant.getValue())));
      } else if (isa<math::AbsFOp>(payload)) {
        auto bitsType = VectorType::get({16}, b.getI32Type());
        Value bits = arith::BitcastOp::create(
            b, loc, bitsType, mapping.lookup(payload.getOperand(0)));
        Value mask = arith::ConstantOp::create(
            b, loc, DenseElementsAttr::get(bitsType, int32_t{0x7fffffff}));
        Value positive = arith::AndIOp::create(b, loc, bits, mask);
        mapping.map(payload.getResult(0),
                    arith::BitcastOp::create(b, loc, resultType, positive));
      } else {
        OperationState state(loc, payload.getName());
        for (Value operand : payload.getOperands())
          state.addOperands(mapping.lookup(operand));
        state.addTypes(resultType);
        state.addAttributes(payload.getAttrs());
        mapping.map(payload.getResult(0), b.create(state)->getResult(0));
      }
    }
    Value result = cast<linalg::YieldOp>(generic.getBody()->getTerminator())
                       .getValues()[0];
    return mapping.lookupOrDefault(result);
  }
  }
  llvm_unreachable("unknown stream expression");
}

Value StreamEmitter::emit(Expression expr, Value position, OpBuilder &b,
                          Location loc) {
  switch (expr->kind) {
  case Expr::Undefined:
    // Reachability of every sink interval was checked before emission. Undef
    // can only occur in a statically unreachable branch of a partial write.
    return {};
  case Expr::Scalar:
    return scalar(expr->value);
  case Expr::Reduction:
    return reductions.lookup(expr->operation);
  case Expr::Read: {
    auto type = cast<MemRefType>(expr->value.getType());
    auto loaded = memref::LoadOp::create(b, loc, expr->value,
                                         type.getRank() ? ValueRange{position}
                                                        : ValueRange{});
    loaded->setAttr("bosc_ame.direct_ddr", b.getUnitAttr());
    return loaded;
  }
  case Expr::Slice: {
    Value translated = position;
    if (expr->stride != 1)
      translated = b.createOrFold<arith::MulIOp>(loc, translated,
                                                 index(b, loc, expr->stride));
    if (expr->offset)
      translated = b.createOrFold<arith::AddIOp>(loc, translated,
                                                 index(b, loc, expr->offset));
    return emit(expr->args[0], translated, b, loc);
  }
  case Expr::Patch: {
    // Full writes replace their predecessor during analysis. Partial writes
    // remain lazy control flow, so a virtual padded lane cannot perform an
    // out-of-range DDR load before a select discards it.
    auto lower = b.createOrFold<arith::CmpIOp>(
        loc, arith::CmpIPredicate::uge, position, index(b, loc, expr->offset));
    auto upper =
        b.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, position,
                                      index(b, loc, expr->offset + expr->size));
    Value condition = b.createOrFold<arith::AndIOp>(loc, lower, upper);
    Type resultType = type(expr, b);
    auto branch = scf::IfOp::create(b, loc, resultType, condition, true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&branch.getThenRegion().front());
      Value local = b.createOrFold<arith::SubIOp>(loc, position,
                                                  index(b, loc, expr->offset));
      Value value = emit(expr->args[1], local, b, loc);
      scf::YieldOp::create(b, loc, value);
      b.setInsertionPointToStart(&branch.getElseRegion().front());
      value = emit(expr->args[0], position, b, loc);
      // A partial uninitialized predecessor can only be reached outside the
      // sink's proved domain. Supply a benign value there, never a memory read.
      if (!value)
        value = arith::ConstantOp::create(b, loc, b.getZeroAttr(resultType));
      scf::YieldOp::create(b, loc, value);
    }
    return branch.getResult(0);
  }
  case Expr::Apply: {
    auto generic = cast<linalg::GenericOp>(expr->operation);
    IRMapping mapping;
    for (auto [argument, input] :
         llvm::zip(generic.getBody()->getArguments(), expr->args))
      if (input)
        mapping.map(argument, emit(input, position, b, loc));
    for (Operation &payload : generic.getBody()->without_terminator()) {
      // Constants and scalar captures need no buffer or vector broadcast.
      for (Value operand : payload.getOperands())
        if (!mapping.contains(operand))
          mapping.map(operand, scalar(operand));
      if (isSupportedMathApprox(payload)) {
        mapping.map(
            payload.getResult(0),
            emitMathApprox(b, payload, mapping.lookup(payload.getOperand(0))));
      } else if (isa<math::AbsFOp>(payload)) {
        Value bits = arith::BitcastOp::create(
            b, loc, b.getI32Type(), mapping.lookup(payload.getOperand(0)));
        Value mask = arith::ConstantIntOp::create(b, loc, 0x7fffffff, 32);
        Value positive = arith::AndIOp::create(b, loc, bits, mask);
        mapping.map(payload.getResult(0),
                    arith::BitcastOp::create(b, loc, b.getF32Type(), positive));
      } else {
        b.clone(payload, mapping);
      }
    }
    Value result = cast<linalg::YieldOp>(generic.getBody()->getTerminator())
                       .getValues()[0];
    return mapping.lookupOrDefault(result);
  }
  }
  llvm_unreachable("unknown stream expression");
}

Operation *StreamEmitter::sink(const Entry &entry) {
  OpBuilder b(entry.op);
  Location loc = entry.op->getLoc();
  if (entry.kind == "scalar-load") {
    auto load = cast<memref::LoadOp>(entry.op);
    auto region =
        scf::ExecuteRegionOp::create(b, loc, load.getResult().getType());
    region.getRegion().push_back(new Block());
    b.setInsertionPointToStart(&region.getRegion().front());
    Value position =
        load.getIndices().empty() ? index(b, loc, 0) : load.getIndices()[0];
    Value value = emit(entry.value, position, b, loc);
    scf::YieldOp::create(b, loc, value);
    replacements[load.getResult()] = region.getResult(0);
    load.getResult().replaceAllUsesWith(region.getResult(0));
    return region;
  }
  if (entry.kind == "store") {
    auto execution = scf::ExecuteRegionOp::create(b, loc, TypeRange{});
    execution.getRegion().push_back(new Block());
    b.setInsertionPointToStart(&execution.getRegion().front());
    auto recorded = entry.op->getAttrOfType<DictionaryAttr>(kStream);
    int64_t prefix = recorded.getAs<IntegerAttr>("vector_elements").getInt();
    auto bounds =
        recorded.getAs<DenseI64ArrayAttr>("vector_bounds").asArrayRef();
    for (size_t segment = 1; segment < bounds.size(); ++segment) {
      int64_t begin = bounds[segment - 1], end = bounds[segment];
      auto loop = scf::ForOp::create(b, loc, index(b, loc, begin),
                                     index(b, loc, end), index(b, loc, 16));
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(loop.getBody());
        Value value =
            emitVector(entry.value, loop.getInductionVar(), begin, end, b, loc);
        auto stored =
            vector::StoreOp::create(b, loc, value, entry.destination,
                                    ValueRange{loop.getInductionVar()});
        stored->setAttr("bosc_ame.direct_ddr", b.getUnitAttr());
      }
    }
    if (prefix == entry.size) {
      scf::YieldOp::create(b, loc);
      return execution;
    }
    auto loop = scf::ForOp::create(b, loc, index(b, loc, 0),
                                   index(b, loc, entry.size - prefix),
                                   index(b, loc, 1));
    b.setInsertionPointToStart(loop.getBody());
    Value position = b.createOrFold<arith::AddIOp>(loc, loop.getInductionVar(),
                                                   index(b, loc, prefix));
    Value value = emit(entry.value, position, b, loc);
    auto type = cast<MemRefType>(entry.destination.getType());
    auto stored = memref::StoreOp::create(b, loc, value, entry.destination,
                                          type.getRank() ? ValueRange{position}
                                                         : ValueRange{});
    stored->setAttr("bosc_ame.direct_ddr", b.getUnitAttr());
    b.setInsertionPointAfter(loop);
    scf::YieldOp::create(b, loc);
    return execution;
  }
  assert(entry.kind == "reduce");
  auto execution = scf::ExecuteRegionOp::create(b, loc, b.getF32Type());
  execution.getRegion().push_back(new Block());
  b.setInsertionPointToStart(&execution.getRegion().front());
  Value initial = emit(entry.initial, index(b, loc, 0), b, loc);
  auto recorded = entry.op->getAttrOfType<DictionaryAttr>(kStream);
  auto bounds = recorded.getAs<DenseI64ArrayAttr>("vector_bounds").asArrayRef();
  auto reduce = cast<linalg::ReduceOp>(entry.op);
  auto kind = isa<arith::AddFOp>(reduce.getRegion().front().front())
                  ? vector::CombiningKind::ADD
                  : vector::CombiningKind::MAXNUMF;
  for (size_t segment = 1; segment < bounds.size(); ++segment) {
    int64_t begin = bounds[segment - 1], end = bounds[segment];
    auto loop =
        scf::ForOp::create(b, loc, index(b, loc, begin), index(b, loc, end),
                           index(b, loc, 16), ValueRange{initial});
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      Value lanes =
          emitVector(entry.value, loop.getInductionVar(), begin, end, b, loc);
      Value next = vector::ReductionOp::create(b, loc, kind, lanes,
                                               loop.getRegionIterArgs()[0]);
      scf::YieldOp::create(b, loc, next);
    }
    initial = loop.getResult(0);
  }
  scf::YieldOp::create(b, loc, initial);
  reductions[entry.op] = execution.getResult(0);
  return execution;
}

LogicalResult StreamEmitter::run() {
  Builder attrs(module.getContext());
  DenseMap<Operation *, SmallVector<Attribute>> logicalTasks, taskMaps;
  DenseMap<Operation *, int64_t> nextId;
  module.walk([&](Operation *op) {
    auto id = op->getAttrOfType<IntegerAttr>("bosc_ame.execution_id");
    if (id) {
      Operation *function = op->getParentOfType<func::FuncOp>();
      nextId[function] = std::max(nextId[function], id.getInt() + 1);
    }
  });
  SmallVector<Operation *> erase;
  SmallVector<Value> virtualRoots;
  for (const BlockPlan &plan : plans) {
    Operation *function =
        plan.block->getParentOp()->getParentOfType<func::FuncOp>();
    if (auto directFunction = dyn_cast<func::FuncOp>(plan.block->getParentOp()))
      function = directFunction;
    DenseMap<int64_t, SmallVector<int64_t>> physical;
    for (const Entry &entry : plan.entries) {
      if (!entry.sink())
        continue;
      int64_t execution = nextId[function]++;
      Operation *completion = sink(entry);
      completion->setAttr("bosc_ame.execution_id",
                          attrs.getI64IntegerAttr(execution));
      SmallVector<Attribute> tasks;
      SmallVector<int64_t> predecessors;
      for (const Entry &source : plan.entries)
        if (auto id = taskId(source.op))
          if (llvm::is_contained(entry.dependencies, *id)) {
            if (source.sink() && source.op != entry.op) {
              auto previous = physical.lookup(*id);
              predecessors.append(previous.begin(), previous.end());
              continue;
            }
            tasks.push_back(archive(attrs, source.op));
            physical[*id].push_back(execution);
          }
      completion->setAttr("bosc_ame.stream_tasks", attrs.getArrayAttr(tasks));
      llvm::sort(predecessors);
      predecessors.erase(std::unique(predecessors.begin(), predecessors.end()),
                         predecessors.end());
      completion->setAttr("bosc_ame.execution_predecessors",
                          attrs.getDenseI64ArrayAttr(predecessors));
      if (Attribute provenance = entry.op->getAttr("contract.provenance"))
        completion->setAttr("contract.provenance", provenance);
    }
    for (const Entry &entry : plan.entries) {
      if (auto id = taskId(entry.op)) {
        logicalTasks[function].push_back(archive(attrs, entry.op));
        auto executions = physical.lookup(*id);
        StringRef action = executions.empty() ? "eliminated"
                           : entry.sink()     ? "executed"
                                              : "rematerialized";
        taskMaps[function].push_back(attrs.getDictionaryAttr(
            {attrs.getNamedAttr("logical_task_id",
                                attrs.getI64IntegerAttr(*id)),
             attrs.getNamedAttr("execution_ids",
                                attrs.getDenseI64ArrayAttr(executions)),
             attrs.getNamedAttr("action", attrs.getStringAttr(action))}));
      }
      erase.push_back(entry.op);
    }
    virtualRoots.append(plan.roots.begin(), plan.roots.end());
  }
  // Expr payloads and scalar values remain alive until every sink has been
  // emitted; destructive traversal would leave DAG references dangling.
  for (Operation *op : llvm::reverse(erase))
    op->erase();
  for (Value storage : virtualRoots) {
    SmallVector<Operation *> aliases, deallocs;
    SmallVector<Value> work{storage};
    while (!work.empty()) {
      Value value = work.pop_back_val();
      for (Operation *user : value.getUsers()) {
        if (isa<memref::DeallocOp>(user))
          deallocs.push_back(user);
        else if (isa<ViewLikeOpInterface>(user)) {
          aliases.push_back(user);
          work.push_back(user->getResult(0));
        }
      }
    }
    for (Operation *op : deallocs)
      op->erase();
    for (Operation *op : llvm::reverse(aliases))
      if (op->use_empty())
        op->erase();
    if (!storage.use_empty())
      return storage.getDefiningOp()->emitError(
          "stream virtual root retained a physical user");
    storage.getDefiningOp()->erase();
  }
  for (auto &[function, tasks] : logicalTasks) {
    function->setAttr("bosc_ame.stream_logical_tasks",
                      attrs.getArrayAttr(tasks));
    function->setAttr("bosc_ame.stream_task_map",
                      attrs.getArrayAttr(taskMaps[function]));
  }
  return success();
}

} // namespace

bool isStreamMember(Operation *op) { return op && op->hasAttr(kStream); }
bool isStreamVirtual(Operation *op) { return op && op->hasAttr(kVirtual); }
bool isStreamVirtual(Value value) {
  return value && isStreamVirtual(root(value).getDefiningOp());
}

LogicalResult analyzeStreams(ModuleOp module, ArrayRef<Value> fixedRoots) {
  auto plans = StreamAnalysis(module, fixedRoots).run();
  Builder b(module.getContext());
  for (const BlockPlan &plan : plans) {
    for (Value value : plan.roots)
      value.getDefiningOp()->setAttr(kVirtual, b.getUnitAttr());
    for (const Entry &entry : plan.entries)
      entry.op->setAttr(kStream, recipe(b, entry));
  }
  return success();
}

static LogicalResult verifyRecordedStreams(ModuleOp module,
                                           ArrayRef<BlockPlan> plans) {
  DenseMap<Operation *, DictionaryAttr> expected;
  DenseSet<Operation *> roots;
  Builder b(module.getContext());
  for (const BlockPlan &plan : plans) {
    for (Value value : plan.roots)
      roots.insert(value.getDefiningOp());
    for (const Entry &entry : plan.entries)
      expected[entry.op] = recipe(b, entry);
  }
  bool invalid = false;
  module.walk([&](Operation *op) {
    if (Attribute recorded = op->getAttr(kStream))
      if (!expected.count(op) || recorded != expected.lookup(op)) {
        op->emitError(
            "stream recipe does not match the proved content-version graph");
        invalid = true;
      }
    if (Attribute recorded = op->getAttr(kVirtual))
      if (!isa<UnitAttr>(recorded) || !roots.contains(op)) {
        op->emitError("stream virtual buffer lacks a complete non-escaping "
                      "content-version proof");
        invalid = true;
      }
  });
  // A recorded component is all-or-nothing; do not accept a missing recipe
  // that would cause the normal emitter to materialize one of its operations.
  for (const BlockPlan &plan : plans) {
    bool selected =
        llvm::any_of(plan.roots,
                     [](Value value) { return isStreamVirtual(value); }) ||
        llvm::any_of(plan.entries, [](const Entry &entry) {
          return isStreamMember(entry.op);
        });
    if (!selected)
      continue;
    for (Value value : plan.roots)
      if (!isStreamVirtual(value)) {
        value.getDefiningOp()->emitError(
            "incomplete stream component buffer selection");
        invalid = true;
      }
    for (const Entry &entry : plan.entries)
      if (entry.op->getAttr(kStream) != expected.lookup(entry.op)) {
        entry.op->emitError("missing recipe in selected stream component");
        invalid = true;
      }
  }
  return failure(invalid);
}

LogicalResult verifyStreams(ModuleOp module) {
  auto plans = StreamAnalysis(module, {}).run();
  return verifyRecordedStreams(module, plans);
}

LogicalResult emitStreams(ModuleOp module, bool inputVerified) {
  if (!inputVerified && failed(verifyStreams(module)))
    return failure();
  // Other helpers may already have replaced their logical computations by
  // physical execution groups. They must not create new stream candidates or
  // invalidate a source proof by merely changing the representation of a
  // disjoint write. Rebuild expressions only for the verified recorded roots;
  // retain exact recipe and complete non-escaping member checks below.
  auto plans = StreamAnalysis(module, {}, true, false).run();
  if (failed(verifyRecordedStreams(module, plans)))
    return failure();
  return StreamEmitter(module, plans).run();
}

} // namespace buddy::boscame
