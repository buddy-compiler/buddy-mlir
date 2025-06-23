//===- TileAndFuse.cpp - Polyhedral Tiling and Fuse Optimization ------===//
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
// This file implements the loop tiling then fuse optimization for
// linalg generic.
//
//===----------------------------------------------------------------------===//
#include "Utils/TileSizeSelection.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "polyhedral-tile-and-fuse"

using namespace mlir;

namespace {
/// Starting from `op` walk all operands backwards to find all potentially
/// fuseable operations, i.e. operations that implement the `TilingInterface`
void collectTiledAndFusedOps(Operation *rootOp,
                             llvm::SmallDenseSet<Operation *> &result) {
  SmallVector<Operation *> worklist;
  worklist.push_back(rootOp);
  result.insert(rootOp);
  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    for (OpOperand &operand : current->getOpOperands()) {
      Operation *producer = operand.get().getDefiningOp();
      if (!producer || !isa<TilingInterface>(producer) ||
          result.count(producer))
        continue;
      worklist.push_back(producer);
      result.insert(producer);
    }
  }
}

FailureOr<tensor::PadOp> foldIfGeneratedFromPadding(RewriterBase &rewriter,
                                                    tensor::PadOp untiledPadOp,
                                                    tensor::PadOp tiledPadOp) {
  auto ifOp = dyn_cast<scf::IfOp>(tiledPadOp->getParentOp());
  if (!ifOp)
    return failure();
  Block *block = tiledPadOp->getBlock();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.inlineBlockBefore(block, ifOp, {});
  rewriter.replaceOp(ifOp, results);
  rewriter.eraseOp(terminator);
  return tiledPadOp;
}

struct TileAndFusePass
    : public PassWrapper<TileAndFusePass, OperationPass<func::FuncOp>> {
  TileAndFusePass() = default;
  TileAndFusePass(int64_t tilingLevel) { this->tilingLevel = tilingLevel; }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileAndFusePass)
  StringRef getArgument() const final { return "polyhedral-tile-and-fuse"; }
  StringRef getDescription() const final { return "Tile and fuse"; }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, affine::AffineDialect,
                    linalg::LinalgDialect, vector::VectorDialect,
                    scf::SCFDialect>();
  }
  void runOnOperation() override;

  int64_t tilingLevel;
};

LogicalResult applyTileAndFuse(RewriterBase &rewriter, Operation *rootOp,
                               DominanceInfo &dominanceInfo,
                               scf::SCFTilingOptions options) {
  llvm::SmallDenseSet<Operation *> originTiledAndFuseOps;
  collectTiledAndFusedOps(rootOp, originTiledAndFuseOps);
  auto isIgnoredUser = [&](Operation *user, scf::ForOp outerMostTiledLoop) {
    return originTiledAndFuseOps.count(user) || isa<tensor::DimOp>(user);
  };

  // 1. Tile the consumer.
  SmallVector<OpResult> yieldedValuesToOrigValues;
  SmallVector<Operation *> tiledOps;
  FailureOr<scf::SCFTilingResult> tilingResult =
      scf::tileUsingSCFForOp(rewriter, cast<TilingInterface>(rootOp), options);
  if (failed(tilingResult)) {
    return failure();
  }
  auto forLoops = llvm::to_vector(llvm::map_range(
      tilingResult->loops, [](Operation *op) { return cast<scf::ForOp>(op); }));
  yieldedValuesToOrigValues.append(rootOp->result_begin(),
                                   rootOp->result_end());
  // A map from untiled value to scf.for iter_arg. The iter_arg is used for DPS
  // init operand of they use the same init operand
  llvm::DenseMap<Value, Value> mapToIterArg;

  if (auto rootPadOp = dyn_cast<tensor::PadOp>(rootOp)) {
    assert(tilingResult->tiledOps.size() == 1 &&
           "Expecting only one tiled op for tensor::PadOp");
    FailureOr<Operation *> replacementTiledOp = foldIfGeneratedFromPadding(
        rewriter, rootPadOp, cast<tensor::PadOp>(tilingResult->tiledOps[0]));
    if (!failed(replacementTiledOp)) {
      tilingResult->tiledOps[0] = replacementTiledOp.value();
    }
  } else if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(rootOp)) {
    for (auto [init, iterArg] : llvm::zip_equal(
             dpsOp.getDpsInitOperands(),
             cast<scf::ForOp>(forLoops.back()).getRegionIterArgs())) {
      mapToIterArg[init->get()] = iterArg;
    }
  }
  tiledOps.append(tilingResult->tiledOps);

  // 2. Tiling each operation results in generation of slices. The source of
  // these slices could be producers that can be fused into the tiled loops by
  // computing the slices of these producers in-place. This results in more
  // slices created for operands of the "fused producer". This open up more
  // opportunities for fusion. Use a worklist to fuse greedily.
  auto addCandidateSlices =
      [&](Operation *fusedOp, std::deque<tensor::ExtractSliceOp> &candidates) {
        for (OpOperand &operand : fusedOp->getOpOperands()) {
          auto sliceOp = operand.get().getDefiningOp<tensor::ExtractSliceOp>();
          if (!sliceOp)
            continue;
          candidates.push_back(sliceOp);

          auto dpsOp = dyn_cast<DestinationStyleOpInterface>(fusedOp);
          if (!dpsOp)
            continue;

          if (dpsOp.isDpsInit(&operand) &&
              mapToIterArg.contains(sliceOp.getSource())) {
            rewriter.startRootUpdate(sliceOp);
            sliceOp.getSourceMutable().assign(
                mapToIterArg[sliceOp.getSource()]);
            rewriter.finalizeRootUpdate(sliceOp);
          }
        }
      };

  std::deque<tensor::ExtractSliceOp> candidates;
  addCandidateSlices(tilingResult->tiledOps.back(), candidates);
  OpBuilder::InsertionGuard g(rewriter);
  while (!candidates.empty()) {
    // Traverse the slices in BFS fashion.
    tensor::ExtractSliceOp candidateSliceOp = candidates.front();
    candidates.pop_front();

    // Materialize the slice of the producer in place.
    std::optional<scf::SCFFuseProducerOfSliceResult> fusedProducer =
        scf::tileAndFuseProducerOfSlice(rewriter, candidateSliceOp, forLoops);
    if (!fusedProducer)
      continue;

    // Check if the fused producer has other uses that require the value
    // to be yielded from within the tiled loop.
    OpResult untiledProducer = fusedProducer->origProducer;
    if (llvm::any_of(untiledProducer.getUsers(), [&](Operation *user) {
          return !isIgnoredUser(user, forLoops.front()) &&
                 !forLoops.front()->isAncestor(user);
        })) {
      scf::yieldReplacementForFusedProducer(rewriter, candidateSliceOp,
                                            fusedProducer.value(), forLoops);
      yieldedValuesToOrigValues.push_back(untiledProducer);
    }

    // Add more fusion candidates to the worklist.
    for (auto tiledOp : fusedProducer->tiledOps) {
      addCandidateSlices(tiledOp, candidates);
      tiledOps.push_back(tiledOp);
    }
  }

  scf::ForOp outermostLoop = forLoops.front();
  for (auto [index, origVal] : llvm::enumerate(yieldedValuesToOrigValues)) {
    Value replacement = outermostLoop.getResult(index);
    rewriter.replaceUsesWithIf(origVal, replacement, [&](OpOperand &use) {
      return !isIgnoredUser(use.getOwner(), outermostLoop) &&
             dominanceInfo.properlyDominates(outermostLoop, use.getOwner());
    });
  }

  return success();
}

void TileAndFusePass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  TilingInterface consumerOp;
  funcOp.walk<WalkOrder::PostOrder, ReverseIterator>([&](TilingInterface op) {
    // Find the next consumer op if it does not have loops.
    if (op.getLoopIteratorTypes().empty())
      return WalkResult::advance();
    consumerOp = op;
    return WalkResult::interrupt();
  });
  if (!consumerOp) {
    LLVM_DEBUG(llvm::dbgs() << "No consumer op found, skip tiling\n");
    return;
  }

  SmallVector<int64_t> tileSizes;
  SmallVector<bool> tileScalableFlags;

  // todo: configure tile sizes and tile scalable flags

  if (llvm::all_of(tileSizes, [&](int64_t size) { return size == 0; })) {
    LLVM_DEBUG(llvm::dbgs() << "All tile sizes are 0, skip tiling\n");
    return;
  }

  scf::SCFTilingOptions options{};
  buddy::setSCFTileSizes(options, consumerOp, std::move(tileSizes),
                         std::move(tileScalableFlags));

  IRRewriter rewriter(context);
  DominanceInfo domainInfo(funcOp);
  if (failed(applyTileAndFuse(rewriter, consumerOp, domainInfo, options))) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to tile and fuse\n");
    return signalPassFailure();
  }

  RewritePatternSet patterns =
      linalg::getLinalgTilingCanonicalizationPatterns(context);
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
  context->getLoadedDialect<tensor::TensorDialect>()
      ->getCanonicalizationPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to canonicalize\n");
    return signalPassFailure();
  }
}

} // namespace

namespace mlir {
namespace buddy {
void registerPolyhedralTileAndFusePass() {
  PassRegistration<TileAndFusePass>();
}
} // namespace buddy
} // namespace mlir