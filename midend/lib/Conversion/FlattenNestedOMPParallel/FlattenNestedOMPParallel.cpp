//===- FlattenNestedOMPParallel.cpp - Flatten nested omp.parallel ---------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This pass strips omp.parallel/omp.wsloop wrappers nested inside another
// omp.parallel, replacing the inner omp.loop_nest with scf.for. Under the
// default OMP_NESTED=false runtime the inner parallel always serializes,
// so the rewrite preserves runtime semantics while drastically reducing
// the number of regions handed to OpenMPIRBuilder during translation.
//
// See docs/superpowers/specs/2026-04-21-flatten-nested-omp-parallel-design.md
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flatten-nested-omp-parallel"

using namespace mlir;

STATISTIC(numFlattened, "Number of nested omp.parallel folded into scf.for");
STATISTIC(numSkippedClauses, "Inner omp.parallel skipped: has clauses");
STATISTIC(numSkippedNonTrivialBody,
          "Inner omp.parallel skipped: body has non-wsloop ops");
STATISTIC(numSkippedNonStandardWsloop,
          "Inner omp.wsloop skipped: has clauses or non-loop_nest body");
STATISTIC(numSkippedMultiLoopNest,
          "Inner omp.loop_nest skipped: more than one induction variable");
STATISTIC(numSkippedNonLinearizableCFG,
          "Inner omp.loop_nest body skipped: CFG is not linearizable");

namespace {

// Returns true iff `parallelOp` is *immediately* nested inside another
// omp.parallel's region body (not crossing an op boundary like func or scf).
static bool isImmediatelyNestedInParallel(omp::ParallelOp parallelOp) {
  Operation *parent = parallelOp->getParentOp();
  while (parent) {
    if (isa<omp::ParallelOp>(parent))
      return true;
    // Stop at function or module boundaries.
    if (parent->hasTrait<OpTrait::IsIsolatedFromAbove>())
      return false;
    parent = parent->getParentOp();
  }
  return false;
}

// Returns the omp.wsloop that is the only meaningful child of `parallelOp`'s
// region, or nullptr if the body has reductions/private/etc. clauses or
// extra ops. Conservative: skip on any structural surprise.
static omp::WsloopOp matchSimpleParallelBody(omp::ParallelOp parallelOp) {
  // Reject if the inner parallel carries any clause beyond num_threads.
  if (!parallelOp.getReductionVars().empty() ||
      !parallelOp.getPrivateVars().empty() ||
      !parallelOp.getAllocateVars().empty() ||
      !parallelOp.getAllocatorVars().empty() ||
      parallelOp.getProcBindKind().has_value()) {
    return nullptr;
  }

  Region &region = parallelOp.getRegion();
  if (!region.hasOneBlock())
    return nullptr;

  Block &block = region.front();
  // Expect exactly two ops: omp.wsloop, omp.terminator.
  auto it = block.begin();
  if (it == block.end())
    return nullptr;
  auto wsloop = dyn_cast<omp::WsloopOp>(&*it);
  if (!wsloop)
    return nullptr;
  ++it;
  if (it == block.end() || !isa<omp::TerminatorOp>(&*it))
    return nullptr;
  ++it;
  if (it != block.end())
    return nullptr;
  return wsloop;
}

// Returns the omp.loop_nest that is the only meaningful child of `wsloop`,
// or nullptr if wsloop has reductions/schedule/etc. or extra body ops.
static omp::LoopNestOp matchSimpleWsloopBody(omp::WsloopOp wsloop) {
  if (!wsloop.getReductionVars().empty() || wsloop.getNowait() ||
      wsloop.getOrdered().has_value() || wsloop.getScheduleKind().has_value()) {
    return nullptr;
  }
  Region &region = wsloop.getRegion();
  if (!region.hasOneBlock())
    return nullptr;
  Block &block = region.front();
  auto it = block.begin();
  if (it == block.end())
    return nullptr;
  auto loopNest = dyn_cast<omp::LoopNestOp>(&*it);
  if (!loopNest)
    return nullptr;
  ++it;
  // Allow a single-block trailing terminator if present, otherwise expect end.
  if (it != block.end() && !isa<omp::TerminatorOp>(&*it))
    return nullptr;
  return loopNest;
}

// Without mutating, returns true iff `region`'s blocks can be collapsed to
// one block by repeatedly merging a block ending in an unconditional br
// (no successor operands) into a successor that has only that single
// predecessor and no block arguments. Mirrors linearizeRegionToSingleBlock
// so we can guard the rewrite (the greedy driver does NOT roll back on
// failure()).
static bool canLinearizeRegionToSingleBlock(Region &region) {
  // Simulate the merging by walking from the entry block forward,
  // following the unique foldable successor each time. The simulated
  // "current block" must reach the single terminating block (no successor
  // or a return-like terminator) covering every block in the region.
  if (region.empty())
    return false;
  llvm::SmallPtrSet<Block *, 8> visited;
  Block *cur = &region.front();
  visited.insert(cur);
  while (true) {
    Operation *terminator = cur->getTerminator();
    if (!terminator)
      return false;
    unsigned numSucc = terminator->getNumSuccessors();
    if (numSucc == 0)
      break; // terminating block reached
    if (numSucc != 1)
      return false;
    auto branchIface = dyn_cast<BranchOpInterface>(terminator);
    if (!branchIface)
      return false;
    if (branchIface.getSuccessorOperands(0).size() != 0)
      return false;
    Block *successor = terminator->getSuccessor(0);
    if (successor->getSinglePredecessor() != cur)
      return false;
    if (successor->getNumArguments() != 0)
      return false;
    if (!visited.insert(successor).second)
      return false; // cycle
    cur = successor;
  }
  // All blocks in the region must lie on this chain (otherwise we'd have
  // unreachable blocks left over after collapsing).
  return visited.size() == (size_t)std::distance(region.begin(), region.end());
}

// Merge sequential blocks in `region` connected by unconditional branches
// (e.g. llvm.br with no operands, cf.br) when the successor has only that
// single predecessor. This linearizes CFG-style bodies (qwen3.0 emits
// stacksave / llvm.br ^bb1 / work / llvm.br ^bb2 / omp.yield) into one
// block so they can fit in scf.for's SizedRegion<1>.
//
// Returns success if the region collapses to a single block; failure
// otherwise (e.g. real conditional branches, cycles, multi-successor).
static LogicalResult linearizeRegionToSingleBlock(Region &region,
                                                  PatternRewriter &rewriter) {
  bool changed = true;
  while (changed) {
    changed = false;
    Block *block = &region.front();
    while (block) {
      Operation *terminator = block->getTerminator();
      if (!terminator)
        return failure();
      if (terminator->getNumSuccessors() != 1) {
        block = block->getNextNode();
        continue;
      }
      auto branchIface = dyn_cast<BranchOpInterface>(terminator);
      if (!branchIface) {
        block = block->getNextNode();
        continue;
      }
      SuccessorOperands succOperands = branchIface.getSuccessorOperands(0);
      if (succOperands.size() != 0) {
        block = block->getNextNode();
        continue;
      }
      Block *successor = terminator->getSuccessor(0);
      if (successor == block)
        return failure(); // self-loop — not linearizable
      if (successor->getSinglePredecessor() != block) {
        block = block->getNextNode();
        continue;
      }
      if (successor->getNumArguments() != 0) {
        block = block->getNextNode();
        continue;
      }
      // Splice successor's ops into block (replacing the terminator),
      // then erase the now-empty successor block.
      rewriter.eraseOp(terminator);
      block->getOperations().splice(block->end(), successor->getOperations());
      rewriter.eraseBlock(successor);
      changed = true;
      // Re-examine the same block (it now has a new terminator that may
      // also be foldable).
    }
  }
  return region.hasOneBlock() ? success() : failure();
}

class FlattenInnerParallel : public OpRewritePattern<omp::ParallelOp> {
public:
  using OpRewritePattern<omp::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::ParallelOp innerParallel,
                                PatternRewriter &rewriter) const override {
    if (!isImmediatelyNestedInParallel(innerParallel))
      return failure();

    omp::WsloopOp wsloop = matchSimpleParallelBody(innerParallel);
    if (!wsloop) {
      // Distinguish reasons for the statistic counters.
      if (!innerParallel.getReductionVars().empty() ||
          !innerParallel.getPrivateVars().empty() ||
          !innerParallel.getAllocateVars().empty() ||
          !innerParallel.getAllocatorVars().empty() ||
          innerParallel.getProcBindKind().has_value()) {
        ++numSkippedClauses;
      } else {
        ++numSkippedNonTrivialBody;
      }
      LLVM_DEBUG(innerParallel->emitWarning(
          "flatten-nested-omp-parallel: skipping inner parallel "
          "(clauses or non-trivial body)"));
      return failure();
    }

    omp::LoopNestOp loopNest = matchSimpleWsloopBody(wsloop);
    if (!loopNest) {
      ++numSkippedNonStandardWsloop;
      LLVM_DEBUG(innerParallel->emitWarning(
          "flatten-nested-omp-parallel: skipping inner parallel "
          "(non-standard wsloop)"));
      return failure();
    }

    // Build scf.for to replace the inner parallel.
    // omp.loop_nest in this dialect form is single-iv; for multi-iv loop_nest
    // we'd skip — but qwen3.0 stg3.mlir uses single-iv exclusively.
    if (loopNest.getNumLoops() != 1) {
      ++numSkippedMultiLoopNest;
      return failure();
    }

    // Build scf.for to replace the inner parallel.
    Location loc = innerParallel.getLoc();
    Value lb = loopNest.getLoopLowerBounds().front();
    Value ub = loopNest.getLoopUpperBounds().front();
    Value step = loopNest.getLoopSteps().front();

    // Linearize the loop_nest body if it has multiple basic blocks (qwen3.0
    // emits CFG-style llvm.br between sequential blocks). scf.for requires
    // SizedRegion<1>, so we must collapse those blocks before moving them
    // in. Skip the rewrite when the body has real branching CFG; the
    // greedy driver does NOT roll back on failure(), so check feasibility
    // BEFORE mutating.
    Region &loopRegion = loopNest.getRegion();
    if (!loopRegion.hasOneBlock()) {
      if (!canLinearizeRegionToSingleBlock(loopRegion)) {
        ++numSkippedNonLinearizableCFG;
        return failure();
      }
      if (failed(linearizeRegionToSingleBlock(loopRegion, rewriter))) {
        // Should not happen given the dry-run check, but be defensive.
        ++numSkippedNonLinearizableCFG;
        return failure();
      }
    }

    auto scfFor = rewriter.create<scf::ForOp>(loc, lb, ub, step);

    // Replace omp.yield (loopRegion's terminator) with scf.yield BEFORE
    // moving so scf.for ends up with only scf.yield. Use a top-level walk
    // restricted to ops directly inside loopRegion's blocks — a deep walk
    // would also rewrite omp.yield of any inner (non-flattened) loop_nest
    // nested inside the body, breaking those.
    SmallVector<omp::YieldOp> yields;
    for (Block &b : loopRegion)
      if (auto y = dyn_cast<omp::YieldOp>(b.getTerminator()))
        yields.push_back(y);
    for (auto y : yields) {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(y);
      rewriter.replaceOpWithNewOp<scf::YieldOp>(y);
    }

    // Merge loop_nest's (now single) entry block INTO scf.for's auto-
    // generated entry block, mapping loopIV -> scfFor.getInductionVar().
    // The auto-generated scf.yield is removed first so the loop_nest's
    // (replaced) terminator becomes scfEntry's terminator.
    Block &scfEntry = scfFor.getRegion().front();
    rewriter.eraseOp(scfEntry.getTerminator());

    Block &loopEntry = loopRegion.front();
    rewriter.mergeBlocks(&loopEntry, &scfEntry, {scfFor.getInductionVar()});

    rewriter.eraseOp(innerParallel);
    ++numFlattened;
    return success();
  }
};

class FlattenNestedOMPParallelPass
    : public PassWrapper<FlattenNestedOMPParallelPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FlattenNestedOMPParallelPass)
  StringRef getArgument() const final { return "flatten-nested-omp-parallel"; }
  StringRef getDescription() const final {
    return "Flatten omp.parallel nested inside another omp.parallel into "
           "scf.for, since the runtime serializes nested parallels under "
           "OMP_NESTED=false anyway.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<FlattenInnerParallel>(ctx);

    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::AnyOp);
    if (failed(applyPatternsGreedily(module, std::move(patterns), config)))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<omp::OpenMPDialect, scf::SCFDialect>();
  }
};

} // namespace

namespace mlir {
namespace buddy {
void registerFlattenNestedOMPParallelPass() {
  PassRegistration<FlattenNestedOMPParallelPass>();
}
} // namespace buddy
} // namespace mlir
