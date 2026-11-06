//===- BOSCAMEFloatSchedule.cpp - Preserve partial rounding in FPRs -------===//
//---------===//
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
// Analyze, verify, and emit ordered scalar floating-point register microtile
// schedules that preserve partial rounding in FPRs.
//
//===----------------------------------------------------------------------===//

#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEFloatSchedule.h"
#include "Analysis/KernelContract.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMETargetInfo.h"
#include "Dialect/BOSCAME/DmaAlignment.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"
#include <functional>

using namespace mlir;
namespace buddy::boscame {
namespace {
constexpr StringLiteral kPlan = "plan.float_schedule";
constexpr StringLiteral kMember = "plan.float_member";
constexpr StringLiteral kVirtual = "plan.float_virtual";
Value root(Value v) {
  while (auto view = dyn_cast_or_null<ViewLikeOpInterface>(v.getDefiningOp()))
    v = view.getViewSource();
  return v;
}
bool local(Value v) {
  return isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(
      root(v).getDefiningOp());
}
SmallVector<Operation *> users(Value v) {
  SmallVector<Value> work{root(v)};
  DenseSet<Value> seen;
  DenseSet<Operation *> unique;
  SmallVector<Operation *> result;
  while (!work.empty()) {
    Value alias = work.pop_back_val();
    if (!seen.insert(alias).second)
      continue;
    for (Operation *op : alias.getUsers()) {
      if (auto view = dyn_cast<ViewLikeOpInterface>(op);
          view && view.getViewSource() == alias && op->getNumResults() == 1)
        work.push_back(op->getResult(0));
      else if (!isa<memref::DeallocOp>(op) && unique.insert(op).second)
        result.push_back(op);
    }
  }
  return result;
}
bool matrix(Value v, int64_t m = -1, int64_t n = -1) {
  auto t = dyn_cast<MemRefType>(v.getType());
  SmallVector<int64_t> strides;
  int64_t offset;
  if (!(t && t.getRank() == 2 && t.hasStaticShape() && !t.getMemorySpace() &&
        t.getElementType().isF32() && t.getDimSize(0) > 0 &&
        t.getDimSize(1) > 0 && (m < 0 || t.getDimSize(0) == m) &&
        (n < 0 || t.getDimSize(1) == n) &&
        succeeded(t.getStridesAndOffset(strides, offset)) && strides[0] > 0 &&
        strides[1] > 0))
    return false;
  // Distinct output lanes and private snapshot elements must not alias.
  // A positive stride by itself does not prove an injective descriptor.
  SmallVector<unsigned> order{0, 1};
  llvm::sort(order,
             [&](unsigned a, unsigned b) { return strides[a] < strides[b]; });
  int64_t span = 1;
  for (unsigned d : order) {
    int64_t delta;
    if ((t.getDimSize(d) > 1 && strides[d] < span) ||
        llvm::MulOverflow(t.getDimSize(d) - 1, strides[d], delta) ||
        llvm::AddOverflow(span, delta, span))
      return false;
  }
  return true;
}
// A dense footprint can be compared across the transpose introduced by the
// compatibility normalizer. Equal element counts alone do not prove coverage.
std::optional<std::pair<int64_t, int64_t>> denseFootprint(Value v) {
  if (!matrix(v))
    return std::nullopt;
  auto type = cast<MemRefType>(v.getType());
  auto origin = spmdma::getStaticPointerOrigin(v);
  auto offset = spmdma::getStaticDescriptorOffset(v);
  SmallVector<int64_t> strides;
  int64_t ignored, elements = 1, span = 1, begin, end;
  if (!origin || !offset || failed(type.getStridesAndOffset(strides, ignored)))
    return std::nullopt;
  for (unsigned d = 0; d < 2; ++d) {
    int64_t delta;
    if (llvm::MulOverflow(elements, type.getDimSize(d), elements) ||
        llvm::MulOverflow(type.getDimSize(d) - 1, strides[d], delta) ||
        llvm::AddOverflow(span, delta, span))
      return std::nullopt;
  }
  if (span != elements || llvm::MulOverflow(*offset, int64_t(4), begin) ||
      llvm::AddOverflow(begin, origin->byteShift, begin) || begin < 0 ||
      llvm::MulOverflow(span, int64_t(4), span) ||
      llvm::AddOverflow(begin, span, end))
    return std::nullopt;
  return std::pair(begin, end);
}
bool prefix(Value prefix, Value full) {
  if (!matrix(prefix) || !matrix(full) || root(prefix) != root(full))
    return false;
  auto p = cast<MemRefType>(prefix.getType());
  auto f = cast<MemRefType>(full.getType());
  SmallVector<int64_t> ps, fs;
  int64_t po, fo;
  auto pOrigin = spmdma::getStaticPointerOrigin(prefix);
  auto fOrigin = spmdma::getStaticPointerOrigin(full);
  return pOrigin && fOrigin && pOrigin->root == fOrigin->root &&
         pOrigin->byteShift == fOrigin->byteShift &&
         spmdma::getStaticDescriptorOffset(prefix) ==
             std::optional<int64_t>(0) &&
         spmdma::getStaticDescriptorOffset(full) == std::optional<int64_t>(0) &&
         succeeded(p.getStridesAndOffset(ps, po)) &&
         succeeded(f.getStridesAndOffset(fs, fo)) && ps == fs &&
         p.getDimSize(0) <= f.getDimSize(0) &&
         p.getDimSize(1) <= f.getDimSize(1);
}
bool positiveZero(Value v) {
  FloatAttr c;
  return matchPattern(v, m_Constant(&c)) && c.getValue().isZero() &&
         !c.getValue().isNegative();
}
bool exactAdd(linalg::GenericOp op, Value left, Value right, Value out) {
  if (!op || op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
      op.getInputs()[0] != left || op.getInputs()[1] != right ||
      op.getOutputs()[0] != out || op.getNumLoops() != 2 ||
      op.getNumParallelLoops() != 2 ||
      !llvm::all_of(op.getIndexingMapsArray(),
                    [](AffineMap m) { return m.isIdentity(); }))
    return false;
  Block &body = *op.getBody();
  auto add = dyn_cast<arith::AddFOp>(body.front());
  return std::distance(body.begin(), body.end()) == 2 && add &&
         add.getFastmath() == arith::FastMathFlags::none &&
         add.getLhs() == body.getArgument(0) &&
         add.getRhs() == body.getArgument(1) &&
         cast<linalg::YieldOp>(body.back()).getValues()[0] == add.getResult();
}
struct Input {
  Value logical, source;
  memref::CopyOp load;
  linalg::FillOp padding;
};
struct Recipe {
  linalg::MatmulOp matmul;
  scf::ForOp loop;
  linalg::GenericOp accumulate, finalAdd;
  linalg::FillOp initial, partialZero;
  Input a, b, previous;
  Value outer, partial, output;
  memref::CopyOp store;
  SmallVector<Operation *> members;
  SmallVector<Value> roots;
  int64_t k = 0, n = 0, logicalN = 0;
};
bool input(Value logical, Input &in, Operation *consumer) {
  if (!local(logical) || !matrix(logical))
    return false;
  in.logical = logical;
  for (Operation *op : users(logical)) {
    if (op == consumer)
      continue;
    if (auto copy = dyn_cast<memref::CopyOp>(op)) {
      if (in.load || !prefix(copy.getTarget(), logical) ||
          local(copy.getSource()))
        return false;
      in.load = copy;
      in.source = copy.getSource();
    } else if (auto fill = dyn_cast<linalg::FillOp>(op)) {
      // The compatibility normalizer may transpose a whole fill. Compare
      // storage coverage, not its logical axis order.
      if (in.padding || root(fill.getOutputs()[0]) != root(logical) ||
          !positiveZero(fill.getInputs()[0]))
        return false;
      auto filled = denseFootprint(fill.getOutputs()[0]);
      auto expected = denseFootprint(logical);
      if (!filled || !expected || *filled != *expected)
        return false;
      in.padding = fill;
    } else {
      return false;
    }
  }
  if (!in.load || in.load->getBlock() != consumer->getBlock() ||
      !in.load->isBeforeInBlock(consumer))
    return false;
  return !in.padding || (in.padding->getBlock() == in.load->getBlock() &&
                         in.padding->isBeforeInBlock(in.load));
}
bool externalABI(ModuleOp module, ArrayRef<Value> values) {
  auto abi = module->getAttrOfType<DictionaryAttr>("contract.abi");
  auto separate = abi ? abi.getAs<BoolAttr>("args_disjoint") : BoolAttr();
  auto alignment =
      abi ? abi.getAs<IntegerAttr>("args_aligned_bytes") : IntegerAttr();
  if (!separate || !separate.getValue() || !alignment ||
      alignment.getInt() < 4 || alignment.getInt() % 4)
    return false;
  Value first = values.front();
  auto fn = first.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!fn || fn.isExternal())
    return false;
  DenseSet<Value> roots;
  for (Value value : values) {
    auto arg = dyn_cast<BlockArgument>(root(value));
    if (!arg || arg.getOwner() != &fn.getBody().front() ||
        !contract::hasEntryArgumentABI(fn) || !roots.insert(arg).second ||
        !spmdma::isEndpointPointerAlignmentProven(value, 4))
      return false;
  }
  return true;
}
std::optional<Recipe> reconstruct(linalg::MatmulOp matmul) {
  if (!matmul.hasPureBufferSemantics() || !matrix(matmul.getInputs()[0], 1) ||
      !matrix(matmul.getInputs()[1]) || !matrix(matmul.getOutputs()[0], 1) ||
      matmul->hasAttr("plan.register_member") || matmul->hasAttr("plan.stream"))
    return std::nullopt;
  Recipe r;
  r.matmul = matmul;
  r.loop = dyn_cast<scf::ForOp>(matmul->getParentOp());
  if (!r.loop || r.loop.getNumRegionIterArgs() ||
      !getConstantIntValue(r.loop.getLowerBound()) ||
      !getConstantIntValue(r.loop.getUpperBound()) ||
      !getConstantIntValue(r.loop.getStep()) ||
      *getConstantIntValue(r.loop.getStep()) <= 0)
    return std::nullopt;
  int64_t tripsSpan;
  auto lower = *getConstantIntValue(r.loop.getLowerBound());
  auto upper = *getConstantIntValue(r.loop.getUpperBound());
  if (llvm::SubOverflow(upper, lower, tripsSpan) || tripsSpan < 0)
    return std::nullopt;
  auto aType = cast<MemRefType>(matmul.getInputs()[0].getType());
  auto bType = cast<MemRefType>(matmul.getInputs()[1].getType());
  auto cType = cast<MemRefType>(matmul.getOutputs()[0].getType());
  r.k = aType.getDimSize(1);
  r.logicalN = cType.getDimSize(1);
  if (r.k != bType.getDimSize(0) || r.logicalN != bType.getDimSize(1) ||
      getNrFpgaTargetInfo().scalarFloatRegisters < 19)
    return std::nullopt;
  Block &product = matmul.getRegion().front();
  if (std::distance(product.begin(), product.end()) != 3)
    return std::nullopt;
  auto mul = dyn_cast<arith::MulFOp>(product.front());
  auto add = dyn_cast<arith::AddFOp>(*std::next(product.begin()));
  if (!mul || !add || mul.getFastmath() != arith::FastMathFlags::none ||
      add.getFastmath() != arith::FastMathFlags::none ||
      mul.getLhs() != product.getArgument(0) ||
      mul.getRhs() != product.getArgument(1) ||
      add.getLhs() != product.getArgument(2) ||
      add.getRhs() != mul.getResult() ||
      cast<linalg::YieldOp>(product.back()).getValues()[0] != add.getResult())
    return std::nullopt;
  r.partial = matmul.getOutputs()[0];
  if (!local(r.partial))
    return std::nullopt;
  for (Operation *op : users(r.partial)) {
    if (op == matmul)
      continue;
    if (auto fill = dyn_cast<linalg::FillOp>(op)) {
      if (r.partialZero || fill.getOutputs()[0] != r.partial ||
          !positiveZero(fill.getInputs()[0]))
        return std::nullopt;
      r.partialZero = fill;
    } else if (auto sum = dyn_cast<linalg::GenericOp>(op)) {
      if (r.accumulate || sum.getNumDpsInputs() != 2)
        return std::nullopt;
      r.accumulate = sum;
      r.outer = sum.getInputs()[0];
    } else
      return std::nullopt;
  }
  if (!r.partialZero || !r.accumulate || !local(r.outer) ||
      !matrix(r.outer, 1, r.logicalN) ||
      !exactAdd(r.accumulate, r.outer, r.partial, r.outer) ||
      r.partialZero->getBlock() != matmul->getBlock() ||
      !r.partialZero->isBeforeInBlock(matmul) ||
      r.accumulate->getBlock() != matmul->getBlock() ||
      !matmul->isBeforeInBlock(r.accumulate))
    return std::nullopt;
  for (Operation *op : users(r.outer)) {
    if (op == r.accumulate)
      continue;
    if (auto fill = dyn_cast<linalg::FillOp>(op)) {
      if (r.initial || fill.getOutputs()[0] != r.outer ||
          !positiveZero(fill.getInputs()[0]))
        return std::nullopt;
      r.initial = fill;
    } else if (auto sum = dyn_cast<linalg::GenericOp>(op)) {
      if (r.finalAdd || sum.getNumDpsInputs() != 2)
        return std::nullopt;
      r.finalAdd = sum;
    } else if (auto copy = dyn_cast<memref::CopyOp>(op)) {
      if (r.store || !prefix(copy.getSource(), r.outer) ||
          local(copy.getTarget()))
        return std::nullopt;
      r.store = copy;
      r.output = copy.getTarget();
    } else
      return std::nullopt;
  }
  if (!r.initial || !r.finalAdd || !r.store ||
      !exactAdd(r.finalAdd, r.outer, r.finalAdd.getInputs()[1], r.outer) ||
      !input(matmul.getInputs()[0], r.a, matmul) ||
      !input(matmul.getInputs()[1], r.b, matmul) ||
      !input(r.finalAdd.getInputs()[1], r.previous, r.finalAdd))
    return std::nullopt;
  if (!matrix(r.a.source, 1, r.k) || !matrix(r.b.source) ||
      !matrix(r.previous.source, 1) || !matrix(r.output, 1))
    return std::nullopt;
  auto as = cast<MemRefType>(r.a.source.getType());
  auto bs = cast<MemRefType>(r.b.source.getType());
  auto cs = cast<MemRefType>(r.previous.source.getType());
  auto os = cast<MemRefType>(r.output.getType());
  r.n = os.getDimSize(1);
  if (as.getShape() != aType.getShape() || bs.getDimSize(0) != r.k ||
      bs.getDimSize(1) != r.n || cs.getShape() != os.getShape() ||
      os.getDimSize(0) != 1 || r.n > r.logicalN ||
      (r.n < r.logicalN && (!r.b.padding || !r.previous.padding)) ||
      !externalABI(matmul->getParentOfType<ModuleOp>(),
                   {r.a.source, r.b.source, r.previous.source}))
    return std::nullopt;
  // Previous C and the final store must denote the same effective address.
  // The frontend uses one SSA view (or its same-sized prefix) for both.
  if (r.previous.source != r.output)
    return std::nullopt;
  r.members = {r.initial,    r.partialZero, matmul,
               r.accumulate, r.finalAdd,    r.store};
  for (Input *in : {&r.a, &r.b, &r.previous}) {
    r.members.push_back(in->load);
    if (in->padding)
      r.members.push_back(in->padding);
    r.roots.push_back(root(in->logical));
  }
  r.roots.push_back(root(r.outer));
  r.roots.push_back(root(r.partial));
  for (Value logical :
       {r.a.logical, r.b.logical, r.previous.logical, r.outer, r.partial})
    if (!denseFootprint(logical))
      return std::nullopt;
  DenseSet<Operation *> members(r.members.begin(), r.members.end());
  DenseSet<Value> roots(r.roots.begin(), r.roots.end());
  if (members.size() != r.members.size() || roots.size() != r.roots.size())
    return std::nullopt;
  Block *block = r.loop->getBlock();
  if (r.initial->getBlock() != block || !r.initial->isBeforeInBlock(r.loop) ||
      r.finalAdd->getBlock() != block ||
      !r.loop->isBeforeInBlock(r.previous.load) ||
      !r.previous.load->isBeforeInBlock(r.finalAdd) ||
      !r.finalAdd->isBeforeInBlock(r.store))
    return std::nullopt;
  std::function<bool(Operation *)> safe = [&](Operation *op) {
    if (members.contains(op) || isa<scf::YieldOp>(op))
      return true;
    if (op == r.loop) {
      for (Operation &nested : r.loop.getBody()->getOperations())
        if (!safe(&nested))
          return false;
      return true;
    }
    if (isa<memref::AllocOp, memref::AllocaOp>(op))
      return roots.contains(op->getResult(0));
    if (auto d = dyn_cast<memref::DeallocOp>(op))
      return roots.contains(root(d.getMemref()));
    return op->getNumRegions() == 0 && isMemoryEffectFree(op);
  };
  for (Operation *op = r.initial;; op = op->getNextNode()) {
    if (!op || !safe(op))
      return std::nullopt;
    if (op == r.store)
      break;
  }
  for (Operation *member : r.members)
    if (member->hasAttr("plan.stream") ||
        member->hasAttr("plan.register_member"))
      return std::nullopt;
  return r;
}
DictionaryAttr describe(const Recipe &r, Builder &b) {
  SmallVector<int64_t> ids;
  for (Operation *op : r.members)
    if (auto task = op->getAttrOfType<DictionaryAttr>("contract.task"))
      ids.push_back(task.getAs<IntegerAttr>("id").getInt());
  return b.getDictionaryAttr(
      {b.getNamedAttr("kind", b.getStringAttr("ordered-f32-microtile")),
       b.getNamedAttr("valid_n", b.getI64IntegerAttr(r.n)),
       b.getNamedAttr("partial_k", b.getI64IntegerAttr(r.k)),
       b.getNamedAttr("micro_n", b.getI64IntegerAttr(8)),
       b.getNamedAttr("peak_fpr", b.getI64IntegerAttr(19)),
       b.getNamedAttr("fp_policy",
                      b.getStringAttr("preserve-partial-rounding")),
       b.getNamedAttr("tasks", b.getDenseI64ArrayAttr(ids))});
}
Value index(OpBuilder &b, Location loc, int64_t n) {
  return arith::ConstantIndexOp::create(b, loc, n);
}
Value cloneSum(OpBuilder &b, linalg::GenericOp sum, Value lhs, Value rhs) {
  IRMapping mapping;
  mapping.map(sum.getBody()->getArgument(0), lhs);
  mapping.map(sum.getBody()->getArgument(1), rhs);
  // Keep all source arithmetic properties, including an explicit rounding
  // mode. Creating a default AddFOp would silently change those properties.
  return b.clone(sum.getBody()->front(), mapping)->getResult(0);
}
Value cloneAddress(Value value, scf::ForOp loop, IRMapping &mapping,
                   OpBuilder &b) {
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;
  Operation *op = value.getDefiningOp();
  if (!op || !loop->isAncestor(op))
    return value;
  for (Value operand : op->getOperands())
    mapping.map(operand, cloneAddress(operand, loop, mapping, b));
  auto cloned = b.clone(*op, mapping);
  for (auto attr : llvm::make_early_inc_range(cloned->getAttrs()))
    if (attr.getName().getValue().starts_with("contract.") ||
        attr.getName().getValue().starts_with("plan."))
      cloned->removeAttr(attr.getName());
  return mapping.lookup(value);
}
void append(func::FuncOp fn, StringRef key, ArrayRef<Attribute> extra,
            Builder &b) {
  SmallVector<Attribute> values;
  if (auto old = fn->getAttrOfType<ArrayAttr>(key))
    llvm::append_range(values, old);
  llvm::append_range(values, extra);
  fn->setAttr(key, b.getArrayAttr(values));
}
void emit(Recipe &r, int64_t id) {
  OpBuilder b(r.store);
  Location loc = r.matmul.getLoc();
  auto fn = r.matmul->getParentOfType<func::FuncOp>();
  auto group = scf::ExecuteRegionOp::create(b, loc, TypeRange{});
  group.getRegion().push_back(new Block);
  b.setInsertionPointToStart(&group.getRegion().front());
  Value zero = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0.0));
  for (int64_t base = 0; base < r.n; base += 8) {
    int64_t width = std::min<int64_t>(8, r.n - base);
    SmallVector<Value> initial(width, zero);
    auto outer =
        scf::ForOp::create(b, loc, r.loop.getLowerBound(),
                           r.loop.getUpperBound(), r.loop.getStep(), initial);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(outer.getBody());
      IRMapping mapping;
      mapping.map(r.loop.getInductionVar(), outer.getInductionVar());
      Value a = cloneAddress(r.a.source, r.loop, mapping, b);
      Value rhs = cloneAddress(r.b.source, r.loop, mapping, b);
      auto inner =
          scf::ForOp::create(b, loc, index(b, loc, 0), index(b, loc, r.k),
                             index(b, loc, 1), initial);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(inner.getBody());
        Value k = inner.getInductionVar();
        Value lhs =
            memref::LoadOp::create(b, loc, a, ValueRange{index(b, loc, 0), k});
        SmallVector<Value> next;
        for (int64_t lane = 0; lane < width; ++lane) {
          Value bv = memref::LoadOp::create(
              b, loc, rhs, ValueRange{k, index(b, loc, base + lane)});
          IRMapping payload;
          payload.map(r.matmul.getBody()->getArguments(),
                      ValueRange{lhs, bv, inner.getRegionIterArgs()[lane]});
          for (Operation &op : r.matmul.getBody()->without_terminator())
            b.clone(op, payload);
          next.push_back(
              payload.lookup(cast<linalg::YieldOp>(r.matmul.getBody()->back())
                                 .getValues()[0]));
        }
        scf::YieldOp::create(b, loc, next);
      }
      SmallVector<Value> next;
      for (int64_t lane = 0; lane < width; ++lane)
        next.push_back(cloneSum(b, r.accumulate,
                                outer.getRegionIterArgs()[lane],
                                inner.getResult(lane)));
      scf::YieldOp::create(b, loc, next);
    }
    for (int64_t lane = 0; lane < width; ++lane) {
      SmallVector<Value> at{index(b, loc, 0), index(b, loc, base + lane)};
      Value previous = memref::LoadOp::create(b, loc, r.previous.source, at);
      Value result = cloneSum(b, r.finalAdd, outer.getResult(lane), previous);
      memref::StoreOp::create(b, loc, result, r.output, at);
    }
  }
  scf::YieldOp::create(b, loc);
  SmallVector<Attribute> archive, mappings;
  for (Operation *op : r.members) {
    NamedAttrList facts;
    for (auto attr : op->getAttrs())
      if (attr.getName().getValue().starts_with("contract."))
        facts.append(attr);
    if (auto task = op->getAttrOfType<DictionaryAttr>("contract.task")) {
      archive.push_back(facts.getDictionary(b.getContext()));
      mappings.push_back(b.getDictionaryAttr(
          {b.getNamedAttr("logical_task_id", task.getAs<IntegerAttr>("id")),
           b.getNamedAttr("execution_ids", b.getDenseI64ArrayAttr({id})),
           b.getNamedAttr("action", b.getStringAttr("executed"))}));
    }
  }
  group->setAttr("bosc_ame.execution_id", b.getI64IntegerAttr(id));
  group->setAttr("bosc_ame.float_schedule", describe(r, b));
  group->setAttr("bosc_ame.float_tasks", b.getArrayAttr(archive));
  NamedAttrList logicalLoop;
  for (auto attr : r.loop->getAttrs())
    if (attr.getName().getValue().starts_with("contract."))
      logicalLoop.append(attr);
  group->setAttr("bosc_ame.float_logical_loop",
                 logicalLoop.getDictionary(b.getContext()));
  append(fn, "bosc_ame.float_logical_tasks", archive, b);
  append(fn, "bosc_ame.float_task_map", mappings, b);
  SmallVector<Value> outside;
  for (Value v : r.roots)
    if (!r.loop->isAncestor(v.getDefiningOp()))
      outside.push_back(v);
  for (Operation *op : r.members)
    if (!r.loop->isAncestor(op))
      op->erase();
  r.loop->erase();
  for (Value v : outside) {
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> dead;
      v.getDefiningOp()->getParentOfType<func::FuncOp>().walk(
          [&](Operation *op) {
            if (isa<ViewLikeOpInterface>(op) && op->use_empty())
              dead.push_back(op);
          });
      for (Operation *op : llvm::reverse(dead)) {
        op->erase();
        changed = true;
      }
    }
    SmallVector<Operation *> ds;
    for (Operation *user : v.getUsers())
      if (isa<memref::DeallocOp>(user))
        ds.push_back(user);
    for (Operation *op : ds)
      op->erase();
    if (v.use_empty())
      v.getDefiningOp()->erase();
  }
}
} // namespace
bool isFloatScheduleMember(Operation *op) { return op && op->hasAttr(kMember); }
bool isFloatScheduleVirtual(Operation *op) {
  return op && op->hasAttr(kVirtual);
}
LogicalResult analyzeFloatSchedules(ModuleOp module,
                                    ArrayRef<Value> fixedRoots) {
  Builder b(module.getContext());
  module.walk([&](linalg::MatmulOp op) {
    if (auto r = reconstruct(op)) {
      if (llvm::any_of(r->roots, [&](Value v) {
            return llvm::is_contained(fixedRoots, v);
          }))
        return;
      op->setAttr(kPlan, describe(*r, b));
      for (Operation *member : r->members)
        member->setAttr(kMember, b.getUnitAttr());
      for (Value v : r->roots)
        v.getDefiningOp()->setAttr(kVirtual, b.getUnitAttr());
    }
  });
  return success();
}
LogicalResult verifyFloatSchedules(ModuleOp module) {
  Builder b(module.getContext());
  DenseSet<Operation *> members, roots;
  bool bad = false;
  module.walk([&](Operation *op) {
    if (!op->hasAttr(kPlan))
      return;
    auto matmul = dyn_cast<linalg::MatmulOp>(op);
    auto r = matmul ? reconstruct(matmul) : std::nullopt;
    if (!r || op->getAttr(kPlan) != describe(*r, b)) {
      op->emitError("invalid ordered floating register schedule");
      bad = true;
      return;
    }
    for (Operation *member : r->members)
      if (!members.insert(member).second ||
          !member->hasAttrOfType<UnitAttr>(kMember))
        bad = true;
    for (Value v : r->roots)
      if (!roots.insert(v.getDefiningOp()).second ||
          !v.getDefiningOp()->hasAttrOfType<UnitAttr>(kVirtual))
        bad = true;
  });
  module.walk([&](Operation *op) {
    if ((op->hasAttr(kMember) && !members.contains(op)) ||
        (op->hasAttr(kVirtual) && !roots.contains(op))) {
      op->emitError("orphaned floating register schedule decision");
      bad = true;
    }
  });
  return failure(bad);
}
LogicalResult emitFloatSchedules(ModuleOp module) {
  if (failed(verifyFloatSchedules(module)))
    return failure();
  SmallVector<Recipe, 0> recipes;
  int64_t id = 0;
  module.walk([&](Operation *op) {
    if (auto previous = op->getAttrOfType<IntegerAttr>("bosc_ame.execution_id"))
      id = std::max(id, previous.getInt() + 1);
    if (op->hasAttr(kPlan))
      recipes.push_back(*reconstruct(cast<linalg::MatmulOp>(op)));
  });
  for (auto &r : recipes)
    emit(r, id++);
  return success();
}
} // namespace buddy::boscame
