// Some code comes from
// compiler/lib/Dialect/SCF/Transforms/RemoveSingleIterationLoop.cpp
// Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");

#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"

#include "GPU/Transforms/RemoveReduntantLoops.h"
#include "PassDetail.h"
#include "Utils/GemmCodegenUtils.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <utility>

#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::buddy;

namespace {

using GetMinMaxExprFn =
    std::function<std::optional<std::pair<AffineExpr, AffineExpr>>(
        Value value, SmallVectorImpl<Value> &dims,
        SmallVectorImpl<Value> &symbols)>;

/// The pattern will detect single iteration loops
/// based on the range returned by the lambda
/// |getMinMaxFn| for some know values.

struct RemoveSingleIterationLoops : public OpRewritePattern<scf::ForOp> {
private:
  /// Compose map with apply affine ops and try to simplify it.
  static void combineAndSimplifyMap(AffineMap &map,
                                    SmallVectorImpl<Value> &dims,
                                    SmallVectorImpl<Value> &symbols) {
    SmallVector<Value> operands(dims.begin(), dims.end());
    operands.append(symbols.begin(), symbols.end());
    // Pull in affine.apply operations and compose them fully into the
    // result.
    affine::fullyComposeAffineMapAndOperands(&map, &operands);
    affine::canonicalizeMapAndOperands(&map, &operands);
    map = simplifyAffineMap(map);
    // Assign the results.
    dims.assign(operands.begin(), operands.begin() + map.getNumDims());
    symbols.assign(operands.begin() + map.getNumDims(), operands.end());
  }

  /// Replace dimensions and symbols with known range in the map expression.
  // TODO: Use core function once the interface using a lambda lands.
  static AffineMap substituteMin(AffineMap map, SmallVectorImpl<Value> &dims,
                                 SmallVectorImpl<Value> &symbols,
                                 GetMinMaxExprFn getMinMaxExpr) {
    combineAndSimplifyMap(map, dims, symbols);

    auto exprs = llvm::to_vector(map.getResults());
    for (AffineExpr &expr : exprs) {
      bool substituted = true;
      while (substituted) {
        substituted = false;
        for (unsigned dimIdx = 0; dimIdx < dims.size(); ++dimIdx) {
          Value dim = dims[dimIdx];
          auto minMax = getMinMaxExpr(dim, dims, symbols);
          if (!minMax)
            continue;
          AffineExpr dimExpr = getAffineDimExpr(dimIdx, expr.getContext());
          // Substitute occurrences of `dimExpr` by either the min expression or
          // the max expression depending on whether the value is used with a
          // positive or negative  coefficient.
          AffineExpr substitutedExpr = affine::substWithMin(
              expr, dimExpr, minMax->first, minMax->second);
          substituted = (substitutedExpr != expr);
          expr = substitutedExpr;
        }
        // Substitute symbols
        for (unsigned symIdx = 0; symIdx < symbols.size(); ++symIdx) {
          Value sym = symbols[symIdx];
          auto minMax = getMinMaxExpr(sym, dims, symbols);
          if (!minMax)
            continue;
          AffineExpr symExpr = getAffineSymbolExpr(symIdx, expr.getContext());
          AffineExpr substitutedExpr = affine::substWithMin(
              expr, symExpr, minMax->first, minMax->second);
          substituted = (substitutedExpr != expr);
          expr = substitutedExpr;
        }
      }
      map = AffineMap::get(dims.size(), symbols.size(), exprs,
                           exprs.front().getContext());
      // Cleanup and simplify the results.
      // This needs to happen outside of the loop iterating on dims.size() since
      // it modifies dims.
      combineAndSimplifyMap(map, dims, symbols);
      // Assign the results.
      exprs.assign(map.getResults().begin(), map.getResults().end());
    }

    assert(!exprs.empty() && "Unexpected empty exprs");
    return AffineMap::get(dims.size(), symbols.size(), exprs, map.getContext());
  }

  /// Replaces the given op with the contents of the given single-block region,
  /// using the operands of the block terminator to replace operation results.
  static void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                                  Region &region, ValueRange blockArgs = {}) {
    assert(llvm::hasSingleElement(region) && "expected single-block region");
    Block *block = &region.front();
    Operation *terminator = block->getTerminator();
    ValueRange results = terminator->getOperands();
    rewriter.inlineBlockBefore(block, op, blockArgs);
    rewriter.replaceOp(op, results);
    rewriter.eraseOp(terminator);
  }

  static void replaceOpWithBlocks(PatternRewriter &rewriter, Operation *op,
                                  Region &region, ValueRange blockArgs = {}) {}

  /// Return true if we can prove that the we always run at least the first
  /// iteration of the ForOp.
  static bool alwaysRunsFirstIteration(scf::ForOp op,
                                       GetMinMaxExprFn getMinMax) {
    // Calculate the minimum value of ub - lb. If it is strictly positive it
    // means the loop will always run at least once.
    MLIRContext *ctx = op->getContext();
    SmallVector<Value> dims;
    SmallVector<Value> symbols;
    AffineExpr lb = getAffineDimExpr(dims.size(), ctx);
    dims.push_back(op.getLowerBound());
    AffineExpr ub = getAffineDimExpr(dims.size(), ctx);
    dims.push_back(op.getUpperBound());
    AffineExpr iterZero = ub - lb;
    auto map = AffineMap::get(dims.size(), 0, iterZero);
    AffineMap simplifiedMap = substituteMin(map, dims, symbols, getMinMax);
    assert(simplifiedMap.getNumResults() == 1);
    auto cst = simplifiedMap.getResult(0).cast<AffineConstantExpr>();
    if (cst.getValue() > 0)
      return true;
    return false;
  }

  /// Return true if we can prove that the we never run more than one iteration
  /// of the ForOp.
  static bool neverRunsSecondIteration(scf::ForOp op,
                                       GetMinMaxExprFn getMinMax) {
    // Calculate the minimum of lb + step - ub. If it is positive it means the
    // loop never run more than once.
    MLIRContext *ctx = op->getContext();
    SmallVector<Value> dims;
    SmallVector<Value> symbols;
    AffineExpr lb = getAffineDimExpr(dims.size(), ctx);
    dims.push_back(op.getLowerBound());
    AffineExpr ub = getAffineDimExpr(dims.size(), ctx);
    dims.push_back(op.getUpperBound());
    AffineExpr step = getAffineDimExpr(dims.size(), ctx);
    dims.push_back(op.getStep());
    AffineExpr iterOne = lb + step - ub;
    auto map = AffineMap::get(dims.size(), 0, iterOne);
    AffineMap simplifiedMap = substituteMin(map, dims, symbols, getMinMax);
    assert(simplifiedMap.getNumResults() == 1);
    auto cst = simplifiedMap.getResult(0).cast<AffineConstantExpr>();
    if (cst.getValue() >= 0)
      return true;
    return false;
  }

public:
  RemoveSingleIterationLoops(MLIRContext *context, GetMinMaxExprFn getMinMax)
      : OpRewritePattern<scf::ForOp>(context, 1), getMinMax(getMinMax) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    if (!(alwaysRunsFirstIteration(forOp, getMinMax) &&
          neverRunsSecondIteration(forOp, getMinMax))) {
      return failure();
    }
    SmallVector<Value> blockArgs;
    blockArgs.reserve(forOp.getInitArgs().size() + 1);
    blockArgs.push_back(forOp.getLowerBound());
    llvm::append_range(blockArgs, forOp.getInitArgs());
    replaceOpWithRegion(rewriter, forOp, forOp.getRegion(), blockArgs);
    return success();
  }

private:
  GetMinMaxExprFn getMinMax;
};

} // namespace

namespace {

/// Converts a symbolic GPU processor dimension to its numeric one.
static unsigned gpuDimToIndex(gpu::Dimension dim) {
  switch (dim) {
  case gpu::Dimension::x:
    return 0;
  case gpu::Dimension::y:
    return 1;
  case gpu::Dimension::z:
    return 2;
  default:
    assert(false && "invalid dimension");
    return 0;
  }
}

/// If the value is a threadID return the range [0, blockGroup-1].
/// If the number of workgroup is known also return the range of workgroupId ad
/// workGroup.
/// As we only use this function in gemm codegen, we we can assume loop variable
/// is relavant to gpu.threadId or gpu.blockId.
static std::optional<std::pair<AffineExpr, AffineExpr>>
getWorkgroupRange(Value processorValue, SmallVectorImpl<Value> & /*dims*/,
                  SmallVectorImpl<Value> & /*symbols*/,
                  ArrayRef<int64_t> workGroup, ArrayRef<int64_t> blockGroup) {
  OpBuilder builder(processorValue.getContext());
  // If the value is a threadID return the range [0, blockDim.i - 1].
  if (auto idOp = processorValue.getDefiningOp<gpu::ThreadIdOp>()) {
    unsigned index = gpuDimToIndex(idOp.getDimension());
    AffineExpr zero = builder.getAffineConstantExpr(0);
    AffineExpr ubExpr = builder.getAffineConstantExpr(workGroup[index]);
    return std::make_pair(zero, ubExpr - 1);
  }
  // If the value is a blockDim return the range [blockGroup, blockGroup].
  if (auto dimOp = processorValue.getDefiningOp<gpu::BlockDimOp>()) {
    unsigned index = gpuDimToIndex(dimOp.getDimension());
    AffineExpr bound = builder.getAffineConstantExpr(workGroup[index]);
    return std::make_pair(bound, bound);
  }
  // If the value is a blockID return the range [0, blockGroupSize_i - 1].
  if (auto idOp = processorValue.getDefiningOp<gpu::BlockIdOp>()) {
    unsigned index = gpuDimToIndex(idOp.getDimension());
    AffineExpr zero = builder.getAffineConstantExpr(0);
    AffineExpr ubExpr = builder.getAffineConstantExpr(blockGroup[index]);
    return std::make_pair(zero, ubExpr - 1);
  }

  return std::nullopt;
}

LogicalResult removeReduntantLoops(func::FuncOp funcOp,
                                   SmallVector<int64_t, 3> workGroup,
                                   SmallVector<int64_t> blockGroup) {
  auto getParallelRangeFn = [=](Value processorValue,
                                SmallVectorImpl<Value> &dims,
                                SmallVectorImpl<Value> &symbols) {
    return getWorkgroupRange(processorValue, dims, symbols, workGroup,
                             blockGroup);
  };
  RewritePatternSet patterns(funcOp->getContext());
  patterns.add<RemoveSingleIterationLoops>(patterns.getContext(),
                                           getParallelRangeFn);
  return applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

struct RemoveReduntantLoopsPass
    : public RemoveReduntantLoopsBase<RemoveReduntantLoopsPass> {
public:
  RemoveReduntantLoopsPass() = default;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    if (!funcHasGemm(funcOp)) {
      return;
    }

    std::optional<SmallVector<int64_t, 3>> optionalworkGroup =
        getGemmBlockSize(funcOp);
    if (!optionalworkGroup.has_value()) {
      return;
    }
    SmallVector<int64_t, 3> workGroup = optionalworkGroup.value();

    std::optional<scf::ForallOp> optionalBlockForallOp =
        getForallOpMappedToBlock(funcOp);
    if (!optionalBlockForallOp.has_value()) {
      return;
    }
    scf::ForallOp forallOp = optionalBlockForallOp.value();
    auto optionalBlockGroups =
        getConstantIntValues(forallOp.getMixedLowerBound());
    SmallVector<int64_t> blockGroup = optionalBlockGroups.value();
    blockGroup.push_back(1);
    if (failed(removeReduntantLoops(funcOp, workGroup, blockGroup))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createRemoveReduntantLoops() {
  return std::make_unique<RemoveReduntantLoopsPass>();
}