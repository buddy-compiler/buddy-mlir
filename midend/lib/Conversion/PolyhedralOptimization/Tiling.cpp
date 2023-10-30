#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/SmallVector.h"
#include "Utils/TileSizeSelection.h"

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

struct TilingPass : public PassWrapper<TilingPass, OperationPass<func::FuncOp>> {
  TilingPass(int64_t tilingLevel = -1) {
    this->tilingLevel = tilingLevel;
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

  int64_t tilingLevel;
};

void TilingPass::runOnOperation() {
  if (tilingLevel == -1) {
    LLVM_DEBUG(llvm::dbgs() << "tilingLevel is -1, skip tiling\n");
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
    SmallVector<bool> tileScalableFlags;

    if (llvm::all_of(tileSizes, [](int64_t v) { return v == 0;})) {
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
void registerPolyhedralTilingPass() {
  PassRegistration<TilingPass>();
}
} // namespace buddy
} // namespace mlir