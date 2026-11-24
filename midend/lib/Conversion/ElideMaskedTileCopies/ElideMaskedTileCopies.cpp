//===- ElideMaskedTileCopies.cpp
//-------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Elide staging copies around a bufferized, pointwise linalg.generic when
// the input and output copies cover the same zero-based rectangle. Run the
// generic directly on subviews when their byte ranges are disjoint or the
// input and output have the same elementwise layout. Retain the staged path
// for possible overlap between different elements or unsupported layouts.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
namespace {
constexpr llvm::StringLiteral fallbackAttr = "buddy.masked_tile_copy_fallback";

bool constant(OpFoldResult v, int64_t n) {
  if (auto a = dyn_cast<Attribute>(v))
    return cast<IntegerAttr>(a).getInt() == n;
  APInt value;
  return matchPattern(cast<Value>(v), m_ConstantInt(&value)) &&
         value.getSExtValue() == n;
}
bool equal(OpFoldResult a, OpFoldResult b) {
  if (a == b)
    return true;
  auto number = [](OpFoldResult v) -> std::optional<int64_t> {
    if (auto a = dyn_cast<Attribute>(v))
      return cast<IntegerAttr>(a).getInt();
    APInt n;
    if (matchPattern(cast<Value>(v), m_ConstantInt(&n)))
      return n.getSExtValue();
    return std::nullopt;
  };
  auto x = number(a), y = number(b);
  return x && y && x == y;
}
bool equalSizes(ArrayRef<OpFoldResult> a, ArrayRef<OpFoldResult> b) {
  return a.size() == b.size() && llvm::all_of(llvm::zip(a, b), [](auto pair) {
           return equal(std::get<0>(pair), std::get<1>(pair));
         });
}
Value root(Value v) {
  while (auto s = v.getDefiningOp<memref::SubViewOp>()) {
    if (s.getType().getRank() != s.getSourceType().getRank() ||
        !llvm::all_of(s.getMixedOffsets(),
                      [](auto x) { return constant(x, 0); }) ||
        !llvm::all_of(s.getMixedStrides(),
                      [](auto x) { return constant(x, 1); }))
      return {};
    v = s.getSource();
  }
  return v;
}
bool local(Value v) {
  return v && v.getDefiningOp() &&
         isa<memref::AllocOp, memref::AllocaOp>(v.getDefiningOp());
}
SmallVector<OpFoldResult> sizes(Value v, OpBuilder &b) {
  if (auto s = v.getDefiningOp<memref::SubViewOp>())
    return s.getMixedSizes();
  SmallVector<OpFoldResult> result;
  auto type = cast<MemRefType>(v.getType());
  // A dynamic extent must be explicit in the copy subview.
  if (!type.hasStaticShape())
    return {};
  for (auto dim : type.getShape())
    result.push_back(b.getIndexAttr(dim));
  return result;
}
struct Copy {
  Value src, dst;
  SmallVector<OpFoldResult> shape;
  SmallVector<Operation *> ops;
};
// Match copies that were expanded into a vector loop and a scalar tail.
std::optional<Copy> loopCopy(scf::ForOp main, OpBuilder &b) {
  auto tail = dyn_cast_or_null<scf::ForOp>(main->getNextNode());
  if (!tail || main.getNumResults() || tail.getNumResults() ||
      !constant(main.getLowerBound(), 0) ||
      tail.getLowerBound() != main.getUpperBound() ||
      !constant(tail.getStep(), 1) ||
      main.getBody()->getOperations().size() != 3 ||
      tail.getBody()->getOperations().size() != 3)
    return std::nullopt;
  APInt width;
  if (!matchPattern(main.getStep(), m_ConstantInt(&width)) || width.ule(1))
    return std::nullopt;
  auto product = main.getUpperBound().getDefiningOp<arith::MulIOp>();
  if (!product || product.getRhs() != main.getStep())
    return std::nullopt;
  auto quotient = product.getLhs().getDefiningOp<arith::DivUIOp>();
  if (!quotient || quotient.getLhs() != tail.getUpperBound() ||
      quotient.getRhs() != main.getStep())
    return std::nullopt;
  auto store =
      dyn_cast<vector::StoreOp>(main.getBody()->getTerminator()->getPrevNode());
  auto scalarStore =
      dyn_cast<memref::StoreOp>(tail.getBody()->getTerminator()->getPrevNode());
  auto scalarLoad = dyn_cast<memref::LoadOp>(&tail.getBody()->front());
  if (!store || !scalarStore || !scalarLoad ||
      scalarStore.getValue() != scalarLoad.getResult() ||
      store.getIndices().size() != 1 || scalarLoad.getIndices().size() != 1 ||
      scalarStore.getIndices().size() != 1 ||
      store.getIndices()[0] != main.getInductionVar() ||
      scalarLoad.getIndices()[0] != tail.getInductionVar() ||
      scalarStore.getIndices()[0] != tail.getInductionVar())
    return std::nullopt;
  Value src;
  Operation *first = &main.getBody()->front();
  if (auto load = dyn_cast<vector::LoadOp>(first)) {
    if (store.getValueToStore() != load.getResult() ||
        load.getIndices().size() != 1 ||
        load.getIndices()[0] != main.getInductionVar())
      return std::nullopt;
    src = load.getBase();
  } else if (auto load = dyn_cast<vector::TransferReadOp>(first)) {
    if (store.getValueToStore() != load.getResult() || load.getMask() ||
        !load.isDimInBounds(0) || !load.getPermutationMap().isIdentity() ||
        load.getIndices().size() != 1 ||
        load.getIndices()[0] != main.getInductionVar())
      return std::nullopt;
    src = load.getBase();
  } else {
    return std::nullopt;
  }
  auto vt = cast<VectorType>(store.getValueToStore().getType());
  if (vt.isScalable() || vt.getRank() != 1 ||
      vt.getNumElements() != width.getSExtValue() ||
      scalarLoad.getMemRef() != src ||
      scalarStore.getMemRef() != store.getBase() ||
      !isa<MemRefType>(src.getType()) ||
      cast<MemRefType>(src.getType()).getRank() != 1 ||
      cast<MemRefType>(store.getBase().getType()).getRank() != 1)
    return std::nullopt;
  return Copy{src, store.getBase(), {tail.getUpperBound()}, {main, tail}};
}

bool hasAddressableLayout(Value view) {
  auto type = cast<MemRefType>(view.getType());
  return !type.getMemorySpace() && type.isStrided();
}

struct Match {
  linalg::GenericOp generic;
  Copy output;
  SmallVector<std::optional<Copy>> inputs;
  SmallVector<Value> buffers;
  llvm::SmallPtrSet<Operation *, 32> owned;
  SmallVector<Operation *> range;
};
std::optional<Match> match(linalg::GenericOp g) {
  if (g->hasAttr(fallbackAttr) || !g.hasPureBufferSemantics() ||
      g.getNumDpsInits() != 1 || g.getNumDpsInputs() == 0 ||
      llvm::any_of(g.getIteratorTypesArray(),
                   [](auto i) { return i != utils::IteratorType::parallel; }) ||
      llvm::any_of(g.getIndexingMapsArray(),
                   [](AffineMap m) { return !m.isIdentity(); }) ||
      !g.getBody()->getArguments().back().use_empty())
    return std::nullopt;
  for (Operation &op : g.getBody()->without_terminator())
    if (!isMemoryEffectFree(&op))
      return std::nullopt;
  OpBuilder b(g);
  SmallVector<Copy> copies;
  for (Operation &op : *g->getBlock()) {
    if (auto c = dyn_cast<memref::CopyOp>(op)) {
      auto srcSize = sizes(c.getSource(), b), dstSize = sizes(c.getTarget(), b);
      if (!srcSize.empty() && equalSizes(srcSize, dstSize))
        copies.push_back({c.getSource(), c.getTarget(), srcSize, {c}});
    } else if (auto loop = dyn_cast<scf::ForOp>(op)) {
      if (auto c = loopCopy(loop, b))
        copies.push_back(*c);
    }
  }
  Value outputBuffer = root(g.getDpsInits()[0]);
  if (!local(outputBuffer) || outputBuffer != g.getDpsInits()[0] ||
      outputBuffer.getDefiningOp()->getBlock() != g->getBlock())
    return std::nullopt;
  SmallVector<Copy> stores;
  for (auto &c : copies)
    if (root(c.src) == outputBuffer && g->isBeforeInBlock(c.ops.front()))
      stores.push_back(c);
  if (stores.size() != 1 || root(stores[0].dst) == outputBuffer)
    return std::nullopt;
  Match m;
  m.generic = g;
  m.output = stores[0];
  m.buffers.push_back(outputBuffer);
  m.owned.insert(g);
  for (auto *op : m.output.ops)
    m.owned.insert(op);
  bool hasInputCopy = false;
  for (Value in : g.getDpsInputs()) {
    Value buffer = root(in);
    SmallVector<Copy> loads;
    for (auto &c : copies)
      if (root(c.dst) == buffer && c.ops.back()->isBeforeInBlock(g))
        loads.push_back(c);
    if (!local(buffer) || loads.empty()) {
      m.inputs.push_back(std::nullopt);
      continue;
    }
    if (buffer.getDefiningOp()->getBlock() != g->getBlock() || in != buffer ||
        loads.size() != 1 || !equalSizes(loads[0].shape, m.output.shape) ||
        root(loads[0].src) == buffer)
      return std::nullopt;
    hasInputCopy = true;
    m.inputs.push_back(loads[0]);
    if (!llvm::is_contained(m.buffers, buffer))
      m.buffers.push_back(buffer);
    for (auto *op : loads[0].ops)
      m.owned.insert(op);
  }
  if (!hasInputCopy)
    return std::nullopt;
  // All uses of a staging allocation must belong to this sequence.
  for (Value buffer : m.buffers) {
    m.owned.insert(buffer.getDefiningOp());
    SmallVector<Value> work{buffer};
    while (!work.empty()) {
      Value value = work.pop_back_val();
      for (Operation *user : value.getUsers()) {
        Operation *top = user;
        while (top->getBlock() != g->getBlock() && top->getParentOp())
          top = top->getParentOp();
        if (m.owned.contains(top))
          continue;
        if (auto view = dyn_cast<memref::SubViewOp>(user)) {
          if (root(view.getResult()) != buffer)
            return std::nullopt;
          m.owned.insert(user);
          work.push_back(view.getResult());
        } else if (auto fill = dyn_cast<linalg::FillOp>(user)) {
          // An input fill must precede the copy into the staging buffer.
          if (buffer == outputBuffer || fill->getBlock() != g->getBlock() ||
              !fill->isBeforeInBlock(g))
            return std::nullopt;
          bool beforeLoad = false;
          for (const auto &input : m.inputs)
            if (input && root(input->dst) == buffer)
              beforeLoad |= fill->isBeforeInBlock(input->ops.front());
          if (!beforeLoad || fill.getOutputs()[0] != buffer)
            return std::nullopt;
          m.owned.insert(user);
        } else if (isa<memref::DeallocOp>(user)) {
          if (user->getBlock() != g->getBlock() ||
              !m.output.ops.back()->isBeforeInBlock(user))
            return std::nullopt;
          m.owned.insert(user);
        } else {
          return std::nullopt;
        }
      }
    }
  }
  Operation *first = g, *last = m.output.ops.back();
  for (Operation *op : m.owned) {
    if (op->getBlock() != g->getBlock())
      return std::nullopt;
    if (op->isBeforeInBlock(first))
      first = op;
  }
  for (Operation *op = first;; op = op->getNextNode()) {
    if (!op)
      return std::nullopt;
    if (!m.owned.contains(op) &&
        (!isMemoryEffectFree(op) || op->getNumRegions()))
      return std::nullopt;
    m.range.push_back(op);
    if (op == last)
      break;
  }
  llvm::SmallPtrSet<Operation *, 32> range(m.range.begin(), m.range.end());
  for (Operation *op : m.range)
    for (Value result : op->getResults())
      for (Operation *user : result.getUsers()) {
        Operation *top = user;
        while (top->getBlock() != g->getBlock() && top->getParentOp())
          top = top->getParentOp();
        if (!range.contains(top) && !m.owned.contains(top))
          return std::nullopt;
      }
  // Recreate view and size computations needed by the direct path.
  std::function<bool(Value)> available = [&](Value v) {
    Operation *op = v.getDefiningOp();
    if (!op || !range.contains(op))
      return true;
    return !m.owned.contains(op) && isMemoryEffectFree(op) &&
           !op->getNumRegions() && llvm::all_of(op->getOperands(), available);
  };
  llvm::SetVector<Value> captured;
  getUsedValuesDefinedAbove(g.getRegion(), captured);
  if (!llvm::all_of(captured, available))
    return std::nullopt;
  if (!available(m.output.dst) || !hasAddressableLayout(m.output.dst))
    return std::nullopt;
  for (auto &input : m.inputs)
    if (input && (!available(input->src) || !hasAddressableLayout(input->src)))
      return std::nullopt;
  for (auto [in, c] : llvm::zip(g.getDpsInputs(), m.inputs))
    if (!c && (!available(in) || !hasAddressableLayout(in)))
      return std::nullopt;
  for (auto dim : m.output.shape)
    if (auto v = dyn_cast<Value>(dim); v && !available(v))
      return std::nullopt;
  return m;
}

struct Interval {
  Value start, end, valid;
  SmallVector<Value> strides;
  unsigned bytes;
};
// A bounding byte interval is conservative for views with holes.
Interval interval(OpBuilder &b, Location loc, Value view) {
  auto ty = cast<MemRefType>(view.getType());
  auto meta = memref::ExtractStridedMetadataOp::create(b, loc, view);
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  unsigned bytes = ty.getElementType().getIntOrFloatBitWidth() / 8;
  Value scale = arith::ConstantIndexOp::create(b, loc, bytes);
  Value ptr = memref::ExtractAlignedPointerAsIndexOp::create(b, loc, view);
  Value offset = arith::MulIOp::create(b, loc, meta.getOffset(), scale);
  Value start = arith::AddIOp::create(b, loc, ptr, offset);
  Value span = one;
  Value valid = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge,
                                      meta.getOffset(), zero);
  SmallVector<Value> strides;
  for (auto [size, stride] : llvm::zip(meta.getSizes(), meta.getStrides())) {
    Value positive =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sgt, stride, zero);
    valid = arith::AndIOp::create(b, loc, valid, positive);
    Value count = arith::SubIOp::create(b, loc, size, one);
    count = arith::MaxSIOp::create(b, loc, count, zero);
    Value part = arith::MulIOp::create(b, loc, count, stride);
    span = arith::AddIOp::create(b, loc, span, part);
    strides.push_back(arith::MulIOp::create(b, loc, stride, scale));
  }
  Value byteSpan = arith::MulIOp::create(b, loc, span, scale);
  Value end = arith::AddIOp::create(b, loc, start, byteSpan);
  return {start, end, valid, strides, bytes};
}
Value safe(OpBuilder &b, Location loc, const Interval &src,
           const Interval &dst) {
  Value same = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                     src.start, dst.start);
  if (src.bytes != dst.bytes)
    same = arith::ConstantIntOp::create(b, loc, 0, 1);
  for (auto [x, y] : llvm::zip(src.strides, dst.strides)) {
    Value equal = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, x, y);
    same = arith::AndIOp::create(b, loc, same, equal);
  }
  Value before = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ule,
                                       src.end, dst.start);
  Value after = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ule,
                                      dst.end, src.start);
  Value disjoint = arith::OrIOp::create(b, loc, before, after);
  Value okay = arith::OrIOp::create(b, loc, same, disjoint);
  Value valid = arith::AndIOp::create(b, loc, src.valid, dst.valid);
  return arith::AndIOp::create(b, loc, okay, valid);
}

void rewrite(Match &m) {
  OpBuilder b(m.range.front());
  Location loc = m.generic.getLoc();
  llvm::SmallPtrSet<Operation *, 32> range(m.range.begin(), m.range.end());
  IRMapping mapping;
  std::function<Value(Value)> materialize = [&](Value v) -> Value {
    if (auto mapped = mapping.lookupOrNull(v))
      return mapped;
    Operation *op = v.getDefiningOp();
    if (!op || !range.contains(op))
      return v;
    for (Value operand : op->getOperands())
      mapping.map(operand, materialize(operand));
    b.clone(*op, mapping);
    return mapping.lookup(v);
  };
  SmallVector<OpFoldResult> shape;
  for (OpFoldResult dim : m.output.shape)
    shape.push_back(
        isa<Value>(dim) ? OpFoldResult(materialize(cast<Value>(dim))) : dim);
  auto tile = [&](Value view) -> Value {
    view = materialize(view);
    auto ty = cast<MemRefType>(view.getType());
    SmallVector<OpFoldResult> offsets(ty.getRank(), b.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(ty.getRank(), b.getIndexAttr(1));
    return memref::SubViewOp::create(b, loc, view, offsets, shape, strides);
  };
  Value out = tile(m.output.dst);
  SmallVector<Value> ins;
  for (auto [input, copy] : llvm::zip(m.generic.getDpsInputs(), m.inputs))
    ins.push_back(tile(copy ? copy->src : input));
  Interval dst = interval(b, loc, out);
  Value guard = dst.valid;
  for (Value in : ins)
    guard = arith::AndIOp::create(b, loc, guard,
                                  safe(b, loc, interval(b, loc, in), dst));
  // Require distinct addresses for distinct output indices.
  auto dstMeta = memref::ExtractStridedMetadataOp::create(b, loc, out);
  Value required = arith::ConstantIndexOp::create(b, loc, 1);
  for (int64_t i = cast<MemRefType>(out.getType()).getRank() - 1; i >= 0; --i) {
    Value enough = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge,
                                         dstMeta.getStrides()[i], required);
    guard = arith::AndIOp::create(b, loc, guard, enough);
    required = arith::MulIOp::create(b, loc, dstMeta.getStrides()[i],
                                     dstMeta.getSizes()[i]);
  }
  llvm::SetVector<Value> captured;
  getUsedValuesDefinedAbove(m.generic.getRegion(), captured);
  for (Value v : captured)
    mapping.map(v, materialize(v));
  auto branch = scf::IfOp::create(b, loc, guard, true);
  b.setInsertionPointToStart(&branch.getThenRegion().front());
  auto direct = cast<linalg::GenericOp>(b.clone(*m.generic, mapping));
  direct.getInputsMutable().assign(ins);
  direct.getDpsInitsMutable().assign(ValueRange{out});
  // Keep the original sequence for potentially overlapping views.
  m.generic->setAttr(fallbackAttr, b.getUnitAttr());
  Block *fallback = &branch.getElseRegion().front();
  for (Operation *op : m.range)
    op->moveBefore(fallback->getTerminator());
  SmallVector<Operation *> deallocs;
  for (Operation *op : m.owned)
    if (isa<memref::DeallocOp>(op) && !range.contains(op))
      deallocs.push_back(op);
  for (Operation *op : deallocs)
    op->moveBefore(fallback->getTerminator());
}

// The alias guard uses the byte sizes of ordinary scalar storage types.
bool supportedElement(Type type) {
  if (!type.isIntOrFloat())
    return false;
  unsigned bits = type.getIntOrFloatBitWidth();

  return (bits == 8 && isa<IntegerType>(type)) || bits == 16 || bits == 32 ||
         bits == 64;
}

struct ElideMaskedTileCopiesPass
    : PassWrapper<ElideMaskedTileCopiesPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ElideMaskedTileCopiesPass)
  StringRef getArgument() const final { return "elide-masked-tile-copies"; }
  StringRef getDescription() const final {
    return "Elide staging copies for masked elementwise tiles";
  }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<arith::ArithDialect, memref::MemRefDialect, linalg::LinalgDialect,
             scf::SCFDialect>();
  }
  void runOnOperation() override {
    SmallVector<linalg::GenericOp> candidates;
    getOperation().walk([&](linalg::GenericOp g) { candidates.push_back(g); });
    for (auto g : candidates) {
      if (auto m = match(g)) {
        auto outType = cast<MemRefType>(g.getDpsInits()[0].getType());
        bool supported = supportedElement(outType.getElementType());
        for (Value v : g.getDpsInputs()) {
          auto t = cast<MemRefType>(v.getType());
          supported &= supportedElement(t.getElementType());
        }
        if (supported)
          rewrite(*m);
      }
    }
  }
};
} // namespace
namespace mlir::buddy {
void registerElideMaskedTileCopiesPass() {
  PassRegistration<ElideMaskedTileCopiesPass>();
}
} // namespace mlir::buddy
