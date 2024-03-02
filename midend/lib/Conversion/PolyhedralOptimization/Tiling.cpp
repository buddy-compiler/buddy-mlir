//===- Tiling.cpp - Polyhedral Tiling Optimization ------===//
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
// This file implements the loop tilig optimiztion for
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
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "polyhedral-tiling"

using namespace mlir;

namespace {

SmallVector<Operation *> getComputeOps(func::FuncOp funcOp) {
  SmallVector<Operation *> computeOps;
  funcOp.walk([&](Operation *op) {
    if (isa<linalg::GenericOp>(op)) {
      computeOps.push_back(op);
    }
  });
  return computeOps;
}

struct TilingPass
    : public PassWrapper<TilingPass, OperationPass<func::FuncOp>> {
  TilingPass() = default;
  TilingPass(const TilingPass &) {}
  TilingPass(ArrayRef<int64_t> tileParam) { 
    tile = tileParam; 
  }
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TilingPass)
  StringRef getArgument() const final { return "polyhedral-tiling"; }
  StringRef getDescription() const final {
    return "Tiling for polyhedral optimization";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, affine::AffineDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }
  void runOnOperation() override;

  ListOption<int64_t> tile{*this, "tile-sizes", llvm::cl::desc("Tile sizes."),
                           llvm::cl::ZeroOrMore};
};

void TilingPass::runOnOperation() {
  if (tile.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "tile sizes is zero, skip tiling\n");
    return;
  }
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);

  for (auto computeOp : computeOps) {
    auto op = cast<TilingInterface>(computeOp);
    if (op.getLoopIteratorTypes().empty())
      continue;

    SmallVector<int64_t> tileSizes;
    SmallVector<bool> tileScalableFlags = SmallVector(tile.size(), false);

    tileSizes.assign(tile.begin(), tile.end());

    if (llvm::all_of(tileSizes, [](int64_t v) { return v == 0; })) {
      LLVM_DEBUG(llvm::dbgs() << "tileSizes are all 0, skip tiling\n");
      return;
    }
    IRRewriter rewriter(context);
    scf::SCFTilingOptions options{};
    buddy::setSCFTileSizes(options, op, std::move(tileSizes),
                           std::move(tileScalableFlags));
    FailureOr<scf::SCFTilingResult> tiledResults =
        scf::tileUsingSCFForOp(rewriter, op, options);
    if (failed(tiledResults)) {
      continue;
    }
    rewriter.replaceOp(op, tiledResults->replacements);
  }

  RewritePatternSet patterns =
      linalg::getLinalgTilingCanonicalizationPatterns(context);
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
  context->getLoadedDialect<tensor::TensorDialect>()
      ->getCanonicalizationPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace

namespace mlir {
namespace buddy {
void registerPolyhedralTilingPass() { PassRegistration<TilingPass>(); }
} // namespace buddy
} // namespace mlir