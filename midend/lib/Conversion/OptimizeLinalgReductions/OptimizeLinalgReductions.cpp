//===- OptimizeLinalgReductions.cpp
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
// Optimize tensor reductions before bufferization. Fold transposes into
// indexing maps, crop reductions to a rectangular tile when the omitted
// padding is neutral, and fuse eligible elementwise producers. Keep producers
// that vary over the innermost reduced dimension separate for ordered
// floating-point addition so VIR can preserve the accumulator order.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace {
bool constantsEqual(OpFoldResult value, int64_t expected) {
  if (auto attr = dyn_cast<Attribute>(value))
    return cast<IntegerAttr>(attr).getInt() == expected;
  APInt number;
  return matchPattern(cast<Value>(value), m_ConstantInt(&number)) &&
         number == expected;
}

bool identity(Operation *op, Value value) {
  APInt integer;
  if (matchPattern(value, m_ConstantInt(&integer))) {
    if (isa<arith::AddIOp, arith::OrIOp, arith::XOrIOp, arith::MaxUIOp>(op))
      return integer.isZero();
    if (isa<arith::MulIOp>(op))
      return integer.isOne();
    if (isa<arith::AndIOp, arith::MinUIOp>(op))
      return integer.isAllOnes();
    if (isa<arith::MaxSIOp>(op))
      return integer.isMinSignedValue();
    if (isa<arith::MinSIOp>(op))
      return integer.isMaxSignedValue();
  }
  APFloat floating(0.0);
  if (!matchPattern(value, m_ConstantFloat(&floating)))
    return false;
  if (auto add = dyn_cast<arith::AddFOp>(op))
    return floating.isZero() &&
           bitEnumContainsAll(add.getFastmath(), arith::FastMathFlags::nsz) &&
           bitEnumContainsAll(add.getFastmath(), arith::FastMathFlags::nnan);
  if (!floating.isInfinity())
    return false;
  return floating.isNegative() ? isa<arith::MaximumFOp, arith::MaxNumFOp>(op)
                               : isa<arith::MinimumFOp, arith::MinNumFOp>(op);
}

Operation *singleCombiner(linalg::ReduceOp reduce) {
  Block *body = reduce.getBody();
  if (body->getNumArguments() != 2 || body->getOperations().size() != 2)
    return nullptr;
  Operation &combiner = body->front();
  auto yield = cast<linalg::YieldOp>(body->getTerminator());
  if (combiner.getNumOperands() != 2 || combiner.getNumResults() != 1 ||
      combiner.getOperand(0) == combiner.getOperand(1) ||
      !llvm::all_of(combiner.getOperands(),
                    [&](Value operand) {
                      return operand == body->getArgument(0) ||
                             operand == body->getArgument(1);
                    }) ||
      yield.getNumOperands() != 1 ||
      yield.getOperand(0) != combiner.getResult(0))
    return nullptr;
  return &combiner;
}

struct ReduceInput {
  Value value;
  SmallVector<int64_t> loopToInput;
};

bool sameSnapshot(Operation *producer, Operation *consumer) {
  if (producer->getBlock() != consumer->getBlock() ||
      !producer->isBeforeInBlock(consumer))
    return false;
  for (Operation *op = producer->getNextNode(); op != consumer;
       op = op->getNextNode())
    if (!isMemoryEffectFree(op))
      return false;
  return true;
}

ReduceInput traceTranspose(Value value, Operation *consumer) {
  auto type = cast<RankedTensorType>(value.getType());
  SmallVector<int64_t> perm =
      llvm::to_vector(llvm::seq<int64_t>(0, type.getRank()));
  if (consumer) {
    while (auto transpose = value.getDefiningOp<linalg::TransposeOp>()) {
      if (!sameSnapshot(transpose, consumer))
        break;
      for (int64_t &dim : perm)
        dim = transpose.getPermutation()[dim];
      value = transpose.getInput();
      consumer = transpose;
    }
  }
  return {value, std::move(perm)};
}

linalg::GenericOp buildReduction(OpBuilder &builder, linalg::ReduceOp reduce,
                                 ArrayRef<ReduceInput> inputs,
                                 ValueRange inits) {
  auto rank = inputs.front().loopToInput.size();
  auto *ctx = builder.getContext();
  SmallVector<AffineMap> maps;
  SmallVector<Value> values;
  for (const ReduceInput &input : inputs) {
    SmallVector<AffineExpr> inverse(rank);
    for (auto [loop, dim] : llvm::enumerate(input.loopToInput))
      inverse[dim] = builder.getAffineDimExpr(loop);
    maps.push_back(AffineMap::get(rank, 0, inverse, ctx));
    values.push_back(input.value);
  }
  SmallVector<AffineExpr> parallel;
  SmallVector<utils::IteratorType> iterators;
  for (unsigned dim = 0; dim < rank; ++dim) {
    bool reduced = llvm::is_contained(reduce.getDimensions(), dim);
    iterators.push_back(reduced ? utils::IteratorType::reduction
                                : utils::IteratorType::parallel);
    if (!reduced)
      parallel.push_back(builder.getAffineDimExpr(dim));
  }
  maps.append(inits.size(), AffineMap::get(rank, 0, parallel, ctx));
  return linalg::GenericOp::create(
      builder, reduce.getLoc(), TypeRange(inits), values, inits, maps,
      iterators, [&](OpBuilder &nested, Location, ValueRange args) {
        IRMapping mapping;
        mapping.map(reduce.getBody()->getArguments(), args);
        for (Operation &operation : *reduce.getBody())
          nested.clone(operation, mapping);
      });
}

void eraseUnused(Operation *operation, IRRewriter &rewriter) {
  if (operation && isOpTriviallyDead(operation))
    rewriter.eraseOp(operation);
}

bool cropReduction(linalg::ReduceOp reduce, IRRewriter &rewriter) {
  if (reduce.getInputs().size() != 1 || reduce.getInits().size() != 1)
    return false;
  auto input = traceTranspose(reduce.getInputs().front(), reduce);
  auto insert = input.value.getDefiningOp<tensor::InsertSliceOp>();
  if (!insert || !sameSnapshot(insert, reduce) ||
      insert.getSourceType().getRank() != insert.getType().getRank() ||
      !llvm::all_of(
          insert.getMixedOffsets(),
          [](OpFoldResult offset) { return constantsEqual(offset, 0); }) ||
      !llvm::all_of(insert.getMixedStrides(), [](OpFoldResult stride) {
        return constantsEqual(stride, 1);
      }))
    return false;
  Value init = reduce.getInits().front();
  auto pad = insert.getDest().getDefiningOp<linalg::FillOp>();
  auto seed = init.getDefiningOp<linalg::FillOp>();
  Operation *combiner = singleCombiner(reduce);
  if (!pad || !seed || !combiner ||
      !identity(combiner, pad.getInputs().front()) ||
      !identity(combiner, seed.getInputs().front()))
    return false;
  auto sizes = insert.getMixedSizes();
  SmallVector<OpFoldResult> kept;
  bool cutsParallel = false;
  bool crops = false;
  for (auto [loop, dim] : llvm::enumerate(input.loopToInput)) {
    int64_t extent = insert.getType().getDimSize(dim);
    bool cut =
        ShapedType::isDynamic(extent) || !constantsEqual(sizes[dim], extent);
    crops |= cut;
    if (!llvm::is_contained(reduce.getDimensions(), loop)) {
      kept.push_back(sizes[dim]);
      cutsParallel |= cut;
    }
  }
  if (!crops)
    return false;
  auto loc = reduce.getLoc();
  rewriter.setInsertionPoint(reduce);
  SmallVector<OpFoldResult> offsets(kept.size(), rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(kept.size(), rewriter.getIndexAttr(1));
  Value croppedInit = init;
  if (cutsParallel) {
    croppedInit = tensor::ExtractSliceOp::create(rewriter, loc, init, offsets,
                                                 kept, strides);
  }
  input.value = insert.getSource();
  Value result =
      buildReduction(rewriter, reduce, {input}, {croppedInit}).getResult(0);
  if (cutsParallel)
    result = tensor::InsertSliceOp::create(rewriter, loc, result, init, offsets,
                                           kept, strides);
  Operation *oldInput = reduce.getInputs().front().getDefiningOp();
  rewriter.replaceOp(reduce, result);
  eraseUnused(oldInput, rewriter);
  if (oldInput != insert.getOperation())
    eraseUnused(insert, rewriter);
  eraseUnused(pad, rewriter);
  return true;
}

bool foldTransposes(linalg::ReduceOp reduce, IRRewriter &rewriter) {
  SmallVector<Value> original(reduce.getInputs());
  SmallVector<ReduceInput> inputs;
  bool changed = false;
  for (Value value : original) {
    auto &input = inputs.emplace_back(traceTranspose(value, reduce));
    changed |= input.value != value;
  }
  if (!changed)
    return false;
  rewriter.setInsertionPoint(reduce);
  auto generic = buildReduction(rewriter, reduce, inputs, reduce.getInits());
  rewriter.replaceOp(reduce, generic.getResults());
  for (Value input :
       llvm::SmallSetVector<Value, 8>(original.begin(), original.end()))
    eraseUnused(input.getDefiningOp(), rewriter);
  return true;
}

bool hasInnermostOrderedFloatSum(linalg::GenericOp generic) {
  auto iterators = generic.getIteratorTypesArray();
  if (iterators.empty() || iterators.back() != utils::IteratorType::reduction)
    return false;
  auto *body = generic.getBody();
  auto accumulators = body->getArguments().take_back(generic.getNumDpsInits());
  auto yields = body->getTerminator()->getOperands();
  for (auto [accumulator, value] : llvm::zip_equal(accumulators, yields)) {
    auto add = value.getDefiningOp<arith::AddFOp>();
    if (add && (add.getLhs() == accumulator || add.getRhs() == accumulator) &&
        !arith::bitEnumContainsAll(add.getFastmath(),
                                   arith::FastMathFlags::reassoc))
      return true;
  }
  return false;
}

linalg::GenericOp fuseIntoGeneric(linalg::GenericOp generic,
                                  IRRewriter &rewriter, bool &fused,
                                  linalg::ReduceOp original = {}) {
  for (;;) {
    bool progress = false;
    bool orderedSum = hasInnermostOrderedFloatSum(generic);
    for (OpOperand *operand : generic.getDpsInputOperands()) {
      if (orderedSum && generic.getMatchingIndexingMap(operand).isFunctionOfDim(
                            generic.getNumLoops() - 1))
        continue;
      Value input = operand->get();
      if (!input.hasOneUse() &&
          !(original && input.hasNUses(2) &&
            llvm::is_contained(original.getInputs(), input)))
        continue;
      if (!linalg::areElementwiseOpsFusable(operand))
        continue;
      Operation *producer = input.getDefiningOp();
      if (!isMemoryEffectFree(producer) || !sameSnapshot(producer, generic))
        continue;
      rewriter.setInsertionPoint(generic);
      auto fusion = linalg::fuseElementwiseOps(rewriter, operand);
      if (failed(fusion))
        continue;
      auto replacement = cast<linalg::GenericOp>(fusion->fusedOp);
      for (auto [from, to] : fusion->replacements)
        rewriter.replaceAllUsesWith(from, to);
      rewriter.eraseOp(generic);
      if (isOpTriviallyDead(producer))
        eraseUnused(producer, rewriter);
      generic = replacement;
      progress = fused = true;
      break;
    }
    if (!progress)
      break;
  }
  return generic;
}

bool fuseProducers(linalg::ReduceOp reduce, IRRewriter &rewriter) {
  SmallVector<ReduceInput> inputs;
  bool candidate = false;
  for (Value value : reduce.getInputs()) {
    inputs.push_back(traceTranspose(value, nullptr));
    candidate |= value.hasOneUse() && value.getDefiningOp<linalg::GenericOp>();
  }
  if (!candidate)
    return false;
  rewriter.setInsertionPoint(reduce);
  auto generic = buildReduction(rewriter, reduce, inputs, reduce.getInits());
  bool changed = false;
  auto fused = fuseIntoGeneric(generic, rewriter, changed, reduce);
  if (!changed) {
    rewriter.eraseOp(generic);
    return false;
  }
  SmallVector<Value> oldInputs(reduce.getInputs());
  auto replacements = fused.getResults().take_back(reduce->getNumResults());
  rewriter.replaceOp(reduce, replacements);
  for (Value input :
       llvm::SmallSetVector<Value, 8>(oldInputs.begin(), oldInputs.end()))
    eraseUnused(input.getDefiningOp(), rewriter);
  return true;
}

enum class ReductionRewrite { FoldTranspose, CropNeutralPadding, FuseProducer };

template <ReductionRewrite Kind>
struct ReductionPass
    : PassWrapper<ReductionPass<Kind>, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReductionPass)
  using Base = PassWrapper<ReductionPass<Kind>, OperationPass<func::FuncOp>>;
  ReductionPass() = default;
  ReductionPass(const ReductionPass &other) : Base(other) {}

  StringRef getArgument() const final {
    if constexpr (Kind == ReductionRewrite::FoldTranspose)
      return "fold-reduction-transposes";
    if constexpr (Kind == ReductionRewrite::CropNeutralPadding)
      return "crop-neutral-reductions";
    return "fuse-reduction-producers";
  }
  StringRef getDescription() const final {
    if constexpr (Kind == ReductionRewrite::FoldTranspose)
      return "Fold transposes into reduction indexing maps";
    if constexpr (Kind == ReductionRewrite::CropNeutralPadding)
      return "Crop reductions whose padding is neutral";
    return "Fuse eligible elementwise producers into reductions";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    IRRewriter rewriter(&this->getContext());
    this->getOperation().walk([&](linalg::ReduceOp reduce) {
      if (!reduce.hasPureTensorSemantics() || reduce.getInputs().empty() ||
          reduce->getNumRegions() != 1 || !isMemoryEffectFree(reduce))
        return;
      bool changed = Kind == ReductionRewrite::FoldTranspose
                         ? foldTransposes(reduce, rewriter)
                     : Kind == ReductionRewrite::CropNeutralPadding
                         ? cropReduction(reduce, rewriter)
                         : fuseProducers(reduce, rewriter);
      if (changed)
        ++hits;
    });
    if constexpr (Kind == ReductionRewrite::FuseProducer) {
      SmallVector<linalg::GenericOp> candidates;
      this->getOperation().walk([&](linalg::GenericOp generic) {
        if (generic.hasPureTensorSemantics() &&
            generic.getNumReductionLoops() && isMemoryEffectFree(generic))
          candidates.push_back(generic);
      });
      for (auto generic : candidates) {
        bool changed = false;
        fuseIntoGeneric(generic, rewriter, changed);
        if (changed)
          ++hits;
      }
    }
  }

  typename Pass::Statistic hits{this, "rewritten",
                                "Number of reductions rewritten"};
};
} // namespace

namespace mlir::buddy {
void registerOptimizeLinalgReductionPasses() {
  PassRegistration<ReductionPass<ReductionRewrite::FoldTranspose>>();
  PassRegistration<ReductionPass<ReductionRewrite::CropNeutralPadding>>();
  PassRegistration<ReductionPass<ReductionRewrite::FuseProducer>>();
}
} // namespace mlir::buddy
