//===- ForwardMaskedLoads.cpp
//----------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Forward loads from a local staging tile to the original source when the
// loads are confined to a prefix filled by a recognized copy loop. Preserve
// the source indices and keep copies whose source can change before a read.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
bool literal(OpFoldResult term, int64_t expected) {
  if (auto attr = dyn_cast<Attribute>(term))
    return cast<IntegerAttr>(attr).getInt() == expected;
  APInt number;
  return matchPattern(cast<Value>(term), m_ConstantInt(&number)) &&
         number == expected;
}

bool unitStride(MemRefType type) {
  SmallVector<int64_t> strides;
  int64_t offset;
  return type.getRank() == 1 &&
         succeeded(type.getStridesAndOffset(strides, offset)) &&
         strides.front() == 1;
}

bool prefix(memref::SubViewOp view, Value count) {
  return view && unitStride(view.getType()) &&
         unitStride(view.getSourceType()) &&
         literal(view.getMixedOffsets().front(), 0) &&
         literal(view.getMixedStrides().front(), 1) &&
         dyn_cast<Value>(view.getMixedSizes().front()) == count;
}

Value root(Value value) {
  while (auto view = value.getDefiningOp<ViewLikeOpInterface>())
    value = view.getViewSource();
  return value;
}

bool independentLocal(Value value, Value sourceRoot) {
  Value allocation = root(value);
  return allocation != sourceRoot &&
         (allocation.getDefiningOp<memref::AllocOp>() ||
          allocation.getDefiningOp<memref::AllocaOp>());
}

bool safeUntilReads(Operation *operation, Value sourceRoot) {
  if (isMemoryEffectFree(operation))
    return true;
  if (auto effects = dyn_cast<MemoryEffectOpInterface>(operation)) {
    SmallVector<MemoryEffects::EffectInstance> instances;
    effects.getEffects(instances);
    for (const auto &instance : instances) {
      if (isa<MemoryEffects::Read, MemoryEffects::Allocate>(
              instance.getEffect()))
        continue;
      if (!instance.getValue() ||
          !independentLocal(instance.getValue(), sourceRoot))
        return false;
    }
  } else if (!operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    return false;
  }
  for (Region &region : operation->getRegions())
    for (Block &block : region)
      for (Operation &nested : block)
        if (!safeUntilReads(&nested, sourceRoot))
          return false;
  return true;
}

Value fillValue(Operation *user, Value tile, Operation *before, int64_t size,
                Operation *&fill) {
  auto store = dyn_cast<memref::StoreOp>(user);
  if (!store || store.getMemRef() != tile)
    return {};
  auto loop = dyn_cast_or_null<scf::ForOp>(store->getParentOp());
  if (!loop || loop->getBlock() != before->getBlock() ||
      !loop->isBeforeInBlock(before) || loop.getNumResults() ||
      loop.getBody()->getOperations().size() != 2 ||
      !literal(loop.getLowerBound(), 0) || !literal(loop.getStep(), 1) ||
      !literal(loop.getUpperBound(), size) ||
      store.getIndices().front() != loop.getInductionVar() ||
      !loop.isDefinedOutsideOfLoop(store.getValue()))
    return {};
  fill = loop.getOperation();
  return store.getValue();
}

bool forwardCopy(scf::ForOp main, IRRewriter &rewriter) {
  auto tail = dyn_cast_or_null<scf::ForOp>(main->getNextNode());
  if (!tail || main.getNumResults() || tail.getNumResults() ||
      !literal(main.getLowerBound(), 0) ||
      tail.getLowerBound() != main.getUpperBound() ||
      !literal(tail.getStep(), 1) ||
      main.getBody()->getOperations().size() != 3 ||
      tail.getBody()->getOperations().size() != 3)
    return false;
  Value count = tail.getUpperBound();
  auto product = main.getUpperBound().getDefiningOp<arith::MulIOp>();
  if (!product || product.getRhs() != main.getStep())
    return false;
  auto quotient = product.getLhs().getDefiningOp<arith::DivUIOp>();
  if (!quotient || quotient.getLhs() != count ||
      quotient.getRhs() != main.getStep())
    return false;
  auto vec = dyn_cast<vector::LoadOp>(main.getBody()->front());
  auto vecStore =
      dyn_cast<vector::StoreOp>(main.getBody()->back().getPrevNode());
  auto load = dyn_cast<memref::LoadOp>(tail.getBody()->front());
  auto store = dyn_cast<memref::StoreOp>(tail.getBody()->back().getPrevNode());
  if (!vec || !vecStore || !load || !store ||
      vec.getResult() != vecStore.getValueToStore() ||
      load.getResult() != store.getValue() ||
      vec.getBase() != load.getMemRef() ||
      vecStore.getBase() != store.getMemRef())
    return false;
  auto original = cast<VectorType>(vec.getResult().getType());
  if (original.getRank() != 1 || original.isScalable() ||
      !literal(main.getStep(), original.getDimSize(0)))
    return false;
  auto source = vec.getBase().getDefiningOp<memref::SubViewOp>();
  auto destination = vecStore.getBase().getDefiningOp<memref::SubViewOp>();
  if (!prefix(source, count) || !prefix(destination, count) ||
      vec.getIndices().front() != main.getInductionVar() ||
      vecStore.getIndices().front() != main.getInductionVar() ||
      load.getIndices().front() != tail.getInductionVar() ||
      store.getIndices().front() != tail.getInductionVar() ||
      !destination.getResult().hasNUses(2))
    return false;
  Value tile = destination.getSource();
  Operation *allocation = tile.getDefiningOp();
  Value sourceRoot = root(source.getSource());
  auto arg = dyn_cast<BlockArgument>(sourceRoot);
  if (!allocation || !isa<memref::AllocOp, memref::AllocaOp>(allocation) ||
      !arg || !isa_and_nonnull<func::FuncOp>(arg.getOwner()->getParentOp()))
    return false;
  auto type = cast<MemRefType>(tile.getType());
  Type element = type.getElementType();
  if (!type.hasStaticShape() || !element.isIntOrFloat() ||
      element.getIntOrFloatBitWidth() < 8)
    return false;
  SmallVector<Operation *> reads, frees;
  Value other;
  Operation *fill = nullptr;
  Operation *last = tail.getOperation();
  for (Operation *user : tile.getUsers()) {
    if (user == destination.getOperation())
      continue;
    if (isa<memref::DeallocOp>(user)) {
      frees.push_back(user);
      continue;
    }
    if (auto read = dyn_cast<vector::TransferReadOp>(user)) {
      if (read.getMask() || !read.getPermutationMap().isIdentity() ||
          !read.isDimInBounds(0) || read.getVectorType().getRank() != 1)
        return false;
    } else if (!isa<memref::LoadOp, vector::LoadOp>(user)) {
      if (other ||
          !(other = fillValue(user, tile, main, type.getDimSize(0), fill)))
        return false;
      continue;
    }
    if (auto vector = dyn_cast<VectorType>(user->getResult(0).getType());
        vector && vector.getRank() != 1)
      return false;
    Operation *container = main->getBlock()->findAncestorOpInBlock(*user);
    if (!container || !tail->isBeforeInBlock(container))
      return false;
    if (last->isBeforeInBlock(container))
      last = container;
    reads.push_back(user);
  }
  if (reads.empty())
    return false;
  for (Operation *between = tail->getNextNode(); between;
       between = between->getNextNode()) {
    if (!safeUntilReads(between, root(source.getSource())))
      return false;
    if (between == last)
      break;
  }
  if (!other) {
    rewriter.setInsertionPoint(main);
    other = arith::ConstantOp::create(
        rewriter, main.getLoc(),
        cast<TypedAttr>(rewriter.getZeroAttr(element)));
  }
  for (Operation *read : reads) {
    Location loc = read->getLoc();
    rewriter.setInsertionPoint(read);
    if (auto scalar = dyn_cast<memref::LoadOp>(read)) {
      Value iv = scalar.getIndices().front();
      Value valid = arith::CmpIOp::create(rewriter, loc,
                                          arith::CmpIPredicate::ult, iv, count);
      auto conditional =
          scf::IfOp::create(rewriter, loc, TypeRange{element}, valid, true);
      rewriter.setInsertionPointToStart(conditional.thenBlock());
      Value value = memref::LoadOp::create(rewriter, loc, source.getResult(),
                                           ValueRange{iv});
      scf::YieldOp::create(rewriter, loc, value);
      rewriter.setInsertionPointToStart(conditional.elseBlock());
      scf::YieldOp::create(rewriter, loc, other);
      rewriter.replaceOp(read, conditional.getResults());
    } else {
      Value iv = isa<vector::LoadOp>(read)
                     ? cast<vector::LoadOp>(read).getIndices().front()
                     : cast<vector::TransferReadOp>(read).getIndices().front();
      auto vecType = cast<VectorType>(read->getResult(0).getType());
      auto maskType = VectorType::get(vecType.getShape(), rewriter.getI1Type(),
                                      vecType.getScalableDims());
      Value remainder = arith::SubIOp::create(rewriter, loc, count, iv);
      Value mask = vector::CreateMaskOp::create(rewriter, loc, maskType,
                                                ValueRange{remainder});
      Value pad = vector::BroadcastOp::create(rewriter, loc, vecType, other);
      auto masked = vector::MaskedLoadOp::create(rewriter, loc, vecType,
                                                 source.getResult(),
                                                 ValueRange{iv}, mask, pad);
      rewriter.replaceOp(read, masked.getResult());
    }
  }
  for (Operation *op : frees)
    rewriter.eraseOp(op);
  if (fill)
    rewriter.eraseOp(fill);
  rewriter.eraseOp(tail);
  rewriter.eraseOp(main);
  rewriter.eraseOp(destination);
  rewriter.eraseOp(allocation);
  return true;
}

struct MaskedLoadsPass
    : PassWrapper<MaskedLoadsPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaskedLoadsPass)
  MaskedLoadsPass() = default;
  MaskedLoadsPass(const MaskedLoadsPass &other) : PassWrapper(other) {}
  StringRef getArgument() const final { return "forward-masked-loads"; }
  StringRef getDescription() const final {
    return "Forward reads of a copied prefix";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    SmallVector<scf::ForOp> candidates;
    getOperation().walk([&](scf::ForOp loop) {
      if (loop.getBody()->getOperations().size() == 3 &&
          isa<vector::LoadOp>(loop.getBody()->front()))
        candidates.push_back(loop);
    });
    for (scf::ForOp loop : candidates) {
      if (forwardCopy(loop, rewriter))
        ++hits;
    }
  }
  Statistic hits{this, "rewritten", "Number of staging buffers forwarded"};
};
} // namespace

namespace mlir::buddy {
void registerForwardMaskedLoadsPass() { PassRegistration<MaskedLoadsPass>(); }
} // namespace mlir::buddy
