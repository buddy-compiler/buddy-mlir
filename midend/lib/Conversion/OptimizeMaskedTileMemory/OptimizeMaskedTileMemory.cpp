//===- OptimizeMaskedTileMemory.cpp
//----------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Vectorize contiguous predicated stores and strided copies, and place
// small fixed-size nonescaping temporaries on the stack. The vector width
// of the store and copy passes is in bits and defaults to 128.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace {
bool isConstant(Value value, int64_t expected) {
  APInt number;
  return matchPattern(value, m_ConstantInt(&number)) && number == expected;
}

Value index(OpBuilder &builder, Location loc, int64_t number) {
  return arith::ConstantIndexOp::create(builder, loc, number);
}

int64_t elementBits(Type type) {
  if (type.isIntOrFloat())
    return type.getIntOrFloatBitWidth();
  return 0;
}

Block *simpleElementwise(linalg::GenericOp generic) {
  if (!generic || generic.getNumLoops() != 1 ||
      generic.getNumParallelLoops() != 1 || !generic.hasPureTensorSemantics() ||
      generic->getNumResults() != 1 || !isMemoryEffectFree(generic) ||
      !llvm::all_of(generic.getIndexingMapsArray(),
                    [](AffineMap map) { return map.isIdentity(); }))
    return nullptr;
  return generic.getBody();
}

bool isIota(Value value) {
  auto generic = value.getDefiningOp<linalg::GenericOp>();
  Block *body = simpleElementwise(generic);
  if (!body || !generic.getInputs().empty())
    return false;
  Value yielded = body->getTerminator()->getOperand(0);
  if (auto conversion = yielded.getDefiningOp<arith::IndexCastOp>())
    yielded = conversion.getIn();
  auto position = yielded.getDefiningOp<linalg::IndexOp>();
  return position && position->getBlock() == body && position.getDim() == 0;
}

Value baseOfIota(Value value) {
  auto generic = value.getDefiningOp<linalg::GenericOp>();
  Block *body = simpleElementwise(generic);
  if (!body || generic.getInputs().size() != 2)
    return {};
  auto add =
      body->getTerminator()->getOperand(0).getDefiningOp<arith::AddIOp>();
  if (!add || add->getBlock() != body ||
      add.getOverflowFlags() != arith::IntegerOverflowFlags::none)
    return {};
  if (!((add.getLhs() == body->getArgument(0) &&
         add.getRhs() == body->getArgument(1)) ||
        (add.getLhs() == body->getArgument(1) &&
         add.getRhs() == body->getArgument(0))))
    return {};
  auto operands = generic.getInputs();
  for (unsigned base : {0u, 1u}) {
    if (auto fill = operands[base].getDefiningOp<linalg::FillOp>();
        fill && isIota(operands[1 - base]))
      return fill.getInputs().front();
  }
  return {};
}

bool alignedBase(Value base, int64_t extent) {
  auto multiply = base.getDefiningOp<arith::MulIOp>();
  if (!multiply)
    return false;
  APInt factor;
  if (!matchPattern(multiply.getLhs(), m_ConstantInt(&factor)) &&
      !matchPattern(multiply.getRhs(), m_ConstantInt(&factor)))
    return false;
  return factor.countr_zero() >= llvm::Log2_64_Ceil(extent);
}

tensor::ExtractOp extraction(Value value, Value iv) {
  auto extract = value.getDefiningOp<tensor::ExtractOp>();
  if (extract && extract.getIndices().size() == 1 &&
      extract.getIndices()[0] == iv)
    return extract;
  return {};
}

bool independentTensor(Value value) {
  if (auto arg = dyn_cast<BlockArgument>(value))
    return isa_and_nonnull<func::FuncOp>(arg.getOwner()->getParentOp());
  if (value.getDefiningOp<tensor::EmptyOp>() ||
      value.getDefiningOp<arith::ConstantOp>())
    return true;
  if (auto view = value.getDefiningOp<ViewLikeOpInterface>())
    return independentTensor(view.getViewSource());
  if (auto insert = value.getDefiningOp<tensor::InsertSliceOp>())
    return independentTensor(insert.getDest());
  if (auto producer = value.getDefiningOp<DestinationStyleOpInterface>()) {
    auto result = cast<OpResult>(value);
    return producer.getNumDpsInits() > result.getResultNumber() &&
           independentTensor(
               producer.getDpsInitOperand(result.getResultNumber())->get());
  }
  return false;
}

struct Comparison {
  arith::CmpIPredicate predicate;
  int iotaOperand;
  Value limit;
};

std::optional<Comparison> offsetComparison(Value mask, Value offsets) {
  auto producer = mask.getDefiningOp<linalg::GenericOp>();
  auto *body = simpleElementwise(producer);
  if (!offsets || !body || producer.getInputs().size() != 2)
    return std::nullopt;
  auto cmp =
      body->getTerminator()->getOperand(0).getDefiningOp<arith::CmpIOp>();
  if (!cmp || !((cmp.getLhs() == body->getArgument(0) &&
                 cmp.getRhs() == body->getArgument(1)) ||
                (cmp.getLhs() == body->getArgument(1) &&
                 cmp.getRhs() == body->getArgument(0))))
    return std::nullopt;
  for (unsigned operand : {0u, 1u}) {
    if (producer.getInputs()[operand] != offsets)
      continue;
    auto fill =
        producer.getInputs()[1 - operand].getDefiningOp<linalg::FillOp>();
    if (fill)
      return Comparison{cmp.getPredicate(),
                        cmp.getLhs() == body->getArgument(operand) ? 0 : 1,
                        fill.getInputs().front()};
  }
  return std::nullopt;
}

bool vectorizeStore(scf::ForOp loop, IRRewriter &rewriter,
                    unsigned vectorBits) {
  APInt bound;
  if (!isConstant(loop.getLowerBound(), 0) || !isConstant(loop.getStep(), 1) ||
      loop.getNumResults() ||
      !matchPattern(loop.getUpperBound(), m_ConstantInt(&bound)) ||
      bound.isNegative() || bound.getActiveBits() > 30)
    return false;
  auto conditional = dyn_cast_or_null<scf::IfOp>(
      loop.getBody()->getTerminator()->getPrevNode());
  if (!conditional || !conditional.getElseRegion().empty() ||
      conditional.getNumResults() ||
      conditional.thenBlock()->getOperations().size() != 2)
    return false;
  auto store = dyn_cast<memref::StoreOp>(conditional.thenBlock()->front());
  if (!store || store.getMemRefType().getRank() != 1)
    return false;
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(store.getMemRefType().getStridesAndOffset(strides, offset)) ||
      strides[0] != 1)
    return false;
  Value iv = loop.getInductionVar();
  auto mask = extraction(conditional.getCondition(), iv);
  auto values = extraction(store.getValue(), iv);
  bool broadcast = loop.isDefinedOutsideOfLoop(store.getValue());
  if (!mask || (!values && !broadcast) ||
      !loop.isDefinedOutsideOfLoop(store.getMemRef()) ||
      !loop.isDefinedOutsideOfLoop(mask.getTensor()) ||
      !independentTensor(mask.getTensor()) ||
      (values && (!loop.isDefinedOutsideOfLoop(values.getTensor()) ||
                  !independentTensor(values.getTensor()))))
    return false;
  for (Operation &op : loop.getBody()->without_terminator())
    if (&op != conditional.getOperation() &&
        !isa<tensor::ExtractOp, arith::IndexCastOp>(op))
      return false;
  Type element = store.getValue().getType();
  int64_t bits = elementBits(element);
  if (bits < 8 || bits % 8)
    return false;
  int64_t width = vectorBits / bits;
  int64_t extent = bound.getZExtValue();
  if (width < 2 || extent < width)
    return false;
  Value destinationIndex = store.getIndices()[0];
  if (auto cast = destinationIndex.getDefiningOp<arith::IndexCastOp>())
    destinationIndex = cast.getIn();
  Value base;
  Value offsetsTensor;
  if (auto offsets = extraction(destinationIndex, iv)) {
    offsetsTensor = offsets.getTensor();
    base = baseOfIota(offsets.getTensor());
    if (!base && !isIota(offsets.getTensor()))
      return false;
  } else if (destinationIndex != iv) {
    return false;
  }
  Type indexType = destinationIndex.getType();
  if (auto integer = dyn_cast<IntegerType>(indexType)) {
    if (integer.getWidth() > 64 || integer.getWidth() < 2 ||
        APInt(64, extent - 1)
            .ugt(APInt::getSignedMaxValue(integer.getWidth()).zextOrTrunc(64)))
      return false;
  } else if (!indexType.isIndex() || base) {
    return false;
  }
  Location loc = loop.getLoc();
  auto comparison = offsetComparison(mask.getTensor(), offsetsTensor);
  rewriter.setInsertionPoint(loop);
  scf::IfOp guarded;
  if (base && !alignedBase(base, extent)) {
    auto type = cast<IntegerType>(base.getType());
    APInt safeLimit = APInt::getSignedMaxValue(type.getWidth()) - (extent - 1);
    Value limit = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(type, safeLimit));
    Value safe = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sle,
                                       base, limit);
    guarded = scf::IfOp::create(rewriter, loc, safe, true);
    rewriter.setInsertionPointToStart(guarded.elseBlock());
    rewriter.clone(*loop);
    rewriter.setInsertionPointToStart(guarded.thenBlock());
  }
  if (base)
    base = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                      base);
  Value upper = index(rewriter, loc, extent / width * width);
  auto valuesType = VectorType::get({width}, element);
  auto maskType = VectorType::get({width}, rewriter.getI1Type());
  Value zero = loop.getLowerBound();
  Value one = loop.getStep();
  Value step = index(rewriter, loc, width);
  scf::ForOp::create(
      rewriter, loc, zero, upper, step, ValueRange{},
      [&](OpBuilder &builder, Location loc, Value lane, ValueRange) {
        const bool inBounds[] = {true};
        Value start =
            base ? arith::AddIOp::create(builder, loc, base, lane).getResult()
                 : lane;
        Value pred;
        if (comparison) {
          auto indexVector = VectorType::get({width}, indexType);
          Value lane0 =
              indexType.isIndex()
                  ? start
                  : arith::IndexCastOp::create(builder, loc, indexType, start)
                        .getResult();
          Value stepVector = vector::StepOp::create(builder, loc, indexVector);
          Value initial =
              vector::BroadcastOp::create(builder, loc, indexVector, lane0);
          Value offsets =
              arith::AddIOp::create(builder, loc, initial, stepVector);
          Value limit = vector::BroadcastOp::create(builder, loc, indexVector,
                                                    comparison->limit);
          auto [lhs, rhs] = comparison->iotaOperand == 0
                                ? std::pair{offsets, limit}
                                : std::pair{limit, offsets};
          pred = arith::CmpIOp::create(builder, loc, comparison->predicate, lhs,
                                       rhs);
        } else {
          Value blank = arith::ConstantOp::create(
              builder, loc, cast<TypedAttr>(builder.getZeroAttr(maskType)));
          auto pack = scf::ForOp::create(
              builder, loc, zero, step, one, ValueRange{blank},
              [&](OpBuilder &nested, Location loc, Value position,
                  ValueRange prev) {
                Value coordinate =
                    arith::AddIOp::create(nested, loc, lane, position);
                Value active = tensor::ExtractOp::create(
                    nested, loc, mask.getTensor(), ValueRange{coordinate});
                Value next = vector::InsertOp::create(
                    nested, loc, active, prev.front(), OpFoldResult(position));
                scf::YieldOp::create(nested, loc, next);
              });
          pred = pack.getResult(0);
        }
        Value data = broadcast ? vector::BroadcastOp::create(
                                     builder, loc, valuesType, store.getValue())
                                     .getResult()
                               : vector::TransferReadOp::create(
                                     builder, loc, valuesType,
                                     values.getTensor(), ValueRange{lane},
                                     std::nullopt, ArrayRef<bool>(inBounds))
                                     .getResult();
        vector::MaskedStoreOp::create(builder, loc, store.getMemRef(),
                                      ValueRange{start}, pred, data);
        scf::YieldOp::create(builder, loc);
      });
  if (extent % width) {
    auto tail = cast<scf::ForOp>(rewriter.clone(*loop));
    tail.getLowerBoundMutable().assign(upper);
  }
  rewriter.eraseOp(loop);
  return true;
}

Value viewRoot(Value value) {
  while (auto view = value.getDefiningOp<ViewLikeOpInterface>())
    value = view.getViewSource();
  return value;
}

bool noOverlap(Value first, Value second) {
  first = viewRoot(first);
  second = viewRoot(second);
  if (first == second)
    return false;
  auto independent = [](Value value) {
    return value.getDefiningOp<memref::AllocOp>() ||
           value.getDefiningOp<memref::AllocaOp>();
  };
  auto known = [&](Value value) {
    auto arg = dyn_cast<BlockArgument>(value);
    return independent(value) || value.getDefiningOp<memref::GetGlobalOp>() ||
           (arg &&
            isa_and_nonnull<func::FuncOp>(arg.getOwner()->getParentOp()));
  };
  return (independent(first) && known(second)) ||
         (independent(second) && known(first));
}

bool stridedCopy(memref::CopyOp copy, IRRewriter &rewriter,
                 unsigned vectorBits) {
  auto sourceType = dyn_cast<MemRefType>(copy.getSource().getType());
  auto targetType = dyn_cast<MemRefType>(copy.getTarget().getType());
  auto contiguousRowMajor = [](MemRefType type) {
    return type.getLayout().isIdentity() ||
           (type.hasStaticShape() && type.getNumElements() > 0 &&
            memref::isStaticShapeAndContiguousRowMajor(type));
  };
  if (!sourceType || !targetType || !sourceType.getRank() ||
      (contiguousRowMajor(sourceType) && contiguousRowMajor(targetType)) ||
      !noOverlap(copy.getSource(), copy.getTarget()))
    return false;
  SmallVector<int64_t> sourceStrides, targetStrides;
  int64_t offset;
  if (failed(sourceType.getStridesAndOffset(sourceStrides, offset)) ||
      failed(targetType.getStridesAndOffset(targetStrides, offset)))
    return false;
  int64_t inner = sourceType.getRank() - 1;
  for (int64_t dim = inner; dim >= 0; --dim) {
    if (sourceStrides[dim] == 1 && targetStrides[dim] == 1) {
      inner = dim;
      break;
    }
  }
  bool contiguous = sourceStrides[inner] == 1 && targetStrides[inner] == 1;
  if (!contiguous && (llvm::is_contained(sourceStrides, ShapedType::kDynamic) ||
                      llvm::is_contained(targetStrides, ShapedType::kDynamic)))
    return false;
  int64_t bits = elementBits(sourceType.getElementType());
  if (bits < 8 || bits % 8)
    return false;
  int64_t width = contiguous ? vectorBits / bits : 1;
  Location loc = copy.getLoc();
  rewriter.setInsertionPoint(copy);
  Value source = copy.getSource();
  Value target = copy.getTarget();
  int64_t last = sourceType.getRank() - 1;
  if (width > 1 && inner != last) {
    SmallVector<AffineExpr> axes;
    for (int64_t dim = 0; dim <= last; ++dim)
      axes.push_back(rewriter.getAffineDimExpr(dim));
    std::swap(axes[inner], axes[last]);
    auto perm = AffineMapAttr::get(
        AffineMap::get(axes.size(), 0, axes, rewriter.getContext()));
    source = memref::TransposeOp::create(rewriter, loc, source, perm);
    target = memref::TransposeOp::create(rewriter, loc, target, perm);
    sourceType = cast<MemRefType>(source.getType());
    inner = last;
  }
  Value zero = index(rewriter, loc, 0);
  Value one = index(rewriter, loc, 1);
  SmallVector<Value> shape;
  for (int64_t dim = 0; dim < sourceType.getRank(); ++dim)
    shape.push_back(
        sourceType.isDynamicDim(dim)
            ? memref::DimOp::create(rewriter, loc, source, dim).getResult()
            : index(rewriter, loc, sourceType.getDimSize(dim)));
  SmallVector<Value> indices(shape.size());
  auto build = [&](auto &&self, OpBuilder &builder, unsigned depth) -> void {
    if (depth + 1 == shape.size()) {
      Value end = shape[inner];
      Value start = zero;
      if (width > 1) {
        Value step = index(builder, loc, width);
        Value rest = arith::RemUIOp::create(builder, loc, end, step);
        start = arith::SubIOp::create(builder, loc, end, rest);
        auto type = VectorType::get({width}, sourceType.getElementType());
        scf::ForOp::create(
            builder, loc, zero, start, step, ValueRange{},
            [&](OpBuilder &nested, Location loc, Value iv, ValueRange) {
              indices[inner] = iv;
              Value loaded =
                  vector::LoadOp::create(nested, loc, type, source, indices);
              vector::StoreOp::create(nested, loc, loaded, target, indices);
              scf::YieldOp::create(nested, loc);
            });
      }
      scf::ForOp::create(
          builder, loc, start, end, one, ValueRange{},
          [&](OpBuilder &nested, Location loc, Value iv, ValueRange) {
            indices[inner] = iv;
            Value loaded = memref::LoadOp::create(nested, loc, source, indices);
            memref::StoreOp::create(nested, loc, loaded, target, indices);
            scf::YieldOp::create(nested, loc);
          });
      return;
    }
    unsigned dim = depth < static_cast<unsigned>(inner) ? depth : depth + 1;
    scf::ForOp::create(
        builder, loc, zero, shape[dim], one, ValueRange{},
        [&](OpBuilder &nested, Location loc, Value iv, ValueRange) {
          indices[dim] = iv;
          self(self, nested, depth + 1);
          scf::YieldOp::create(nested, loc);
        });
  };
  build(build, rewriter, 0);
  rewriter.eraseOp(copy);
  return true;
}

int64_t bytes(MemRefType type) {
  int64_t bits = elementBits(type.getElementType());
  if (!type.hasStaticShape() || !type.getLayout().isIdentity() || bits < 8 ||
      bits % 8)
    return 0;
  int64_t size = bits / 8;
  for (int64_t extent : type.getShape()) {
    if (!extent || size > (64 * 1024) / extent)
      return 0;
    size *= extent;
  }
  return size;
}

bool nonescaping(Value value, Region &lifetime,
                 SmallVectorImpl<Operation *> &dead,
                 llvm::SmallPtrSetImpl<Operation *> &users) {
  for (OpOperand &use : value.getUses()) {
    Operation *operation = use.getOwner();
    if (!lifetime.isAncestor(operation->getParentRegion()))
      return false;
    if (auto view = dyn_cast<ViewLikeOpInterface>(operation)) {
      if (view.getViewSource() != value || operation->getNumResults() != 1 ||
          !isa<MemRefType>(operation->getResult(0).getType()) ||
          !nonescaping(operation->getResult(0), lifetime, dead, users))
        return false;
    } else if (isa<memref::DeallocOp>(operation)) {
      if (!value.getDefiningOp<memref::AllocOp>())
        return false;
      dead.push_back(operation);
      continue;
    } else if (auto store = dyn_cast<memref::StoreOp>(operation)) {
      if (store.getMemRef() != value || store.getValue() == value)
        return false;
    } else if (!isa<memref::LoadOp, memref::CopyOp, memref::DimOp,
                    linalg::LinalgOp, vector::LoadOp, vector::StoreOp,
                    vector::MaskableOpInterface>(operation)) {
      return false;
    }
    users.insert(operation);
  }
  return true;
}

enum class MemoryRewrite { MaskedStore, StridedCopy, StackTemporary };

template <MemoryRewrite Kind>
struct MemoryPass : PassWrapper<MemoryPass<Kind>, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemoryPass)
  using Base = PassWrapper<MemoryPass<Kind>, OperationPass<func::FuncOp>>;
  MemoryPass() = default;
  MemoryPass(const MemoryPass &other) : Base(other) {}

  StringRef getArgument() const final {
    if constexpr (Kind == MemoryRewrite::MaskedStore)
      return "vectorize-masked-stores";
    if constexpr (Kind == MemoryRewrite::StridedCopy)
      return "lower-strided-copies";
    return "stack-temporaries";
  }
  StringRef getDescription() const final {
    if constexpr (Kind == MemoryRewrite::MaskedStore)
      return "Vectorize contiguous predicated stores";
    if constexpr (Kind == MemoryRewrite::StridedCopy)
      return "Lower strided copies to vector loops";
    return "Place small nonescaping temporaries on the stack";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect,
                    tensor::TensorDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    IRRewriter rewriter(&this->getContext());
    func::FuncOp fn = this->getOperation();
    if (fn.isExternal())
      return;
    if constexpr (Kind == MemoryRewrite::MaskedStore)
      fn.walk([&](scf::ForOp loop) {
        if (vectorizeStore(loop, rewriter, vectorBits))
          ++hits;
      });
    if constexpr (Kind == MemoryRewrite::StridedCopy)
      fn.walk([&](memref::CopyOp copy) {
        if (stridedCopy(copy, rewriter, vectorBits))
          ++hits;
      });
    if constexpr (Kind == MemoryRewrite::StackTemporary) {
      if (!llvm::hasSingleElement(fn.getBody()))
        return;
      int64_t remaining = 64 * 1024;
      fn.walk(
          [&](memref::AllocaOp alloc) { remaining -= bytes(alloc.getType()); });
      struct Slot {
        memref::AllocaOp alloc;
        Operation *end;
      };
      SmallVector<Slot> slots;
      fn.walk([&](memref::AllocOp alloc) {
        auto size = bytes(alloc.getType());
        if (!size || size > remaining || alloc.getType().getMemorySpace())
          return;
        auto *lifetime = alloc->getParentRegion();
        SmallVector<Operation *> dead;
        llvm::SmallPtrSet<Operation *, 16> users;
        if (!nonescaping(alloc, *lifetime, dead, users))
          return;
        Block &entry = fn.getBody().front();
        Operation *start = entry.findAncestorOpInBlock(*alloc);
        Operation *end = start;
        for (Operation *use : users) {
          Operation *parent = entry.findAncestorOpInBlock(*use);
          if (end->isBeforeInBlock(parent))
            end = parent;
        }
        if (end == alloc.getOperation())
          return;
        auto slot = llvm::find_if(slots, [&](Slot &slot) {
          return slot.alloc.getType() == alloc.getType() &&
                 slot.alloc.getAlignment() == alloc.getAlignment() &&
                 slot.end->isBeforeInBlock(start);
        });
        memref::AllocaOp replacement;
        if (slot == slots.end()) {
          rewriter.setInsertionPointToStart(&entry);
          replacement = memref::AllocaOp::create(rewriter, alloc.getLoc(),
                                                 alloc.getType(),
                                                 alloc.getAlignmentAttr());
          remaining -= size;
          slots.push_back({replacement, end});
        } else {
          replacement = slot->alloc;
          slot->end = end;
        }
        for (Operation *operation : dead)
          rewriter.eraseOp(operation);
        rewriter.replaceOp(alloc, replacement.getResult());
        ++hits;
      });
    }
  }

  typename Pass::Option<unsigned> vectorBits{*this, "vector-bits",
                                             llvm::cl::init(128)};
  typename Pass::Statistic hits{this, "rewritten",
                                "Number of memory operations rewritten"};
};
} // namespace

namespace mlir::buddy {
void registerOptimizeMaskedTileMemoryPasses() {
  PassRegistration<MemoryPass<MemoryRewrite::MaskedStore>>();
  PassRegistration<MemoryPass<MemoryRewrite::StridedCopy>>();
  PassRegistration<MemoryPass<MemoryRewrite::StackTemporary>>();
}
} // namespace mlir::buddy
