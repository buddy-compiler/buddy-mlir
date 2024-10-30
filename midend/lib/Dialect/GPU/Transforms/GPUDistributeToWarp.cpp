#include "GPU/Transforms/GPUDistributeToWarp.h"
#include "PassDetail.h"
#include "Utils/GemmCodegenUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cstdint>
#include <optional>

using namespace mlir;
using namespace mlir::buddy;

namespace {

static constexpr int64_t warpSize = 32;

// get the parallel dim of linalg loops
// if parallel dim = 1 remove it
SmallVector<unsigned>
getPartitionedLoopsFromLinalgOp(linalg::LinalgOp linalgOp) {
  SmallVector<unsigned> parallelLoops;
  // Return the dims that are parallel loops.
  linalgOp.getParallelDims(parallelLoops);

  // LoopRange is the size of each parallel's dim
  SmallVector<int64_t> loopRanges = linalgOp.getStaticLoopRanges();
  // remove the dimension in parallelLoops whose range is 1
  parallelLoops = llvm::to_vector(
      llvm::make_filter_range(parallelLoops, [=](unsigned loopDim) {
        return loopRanges[loopDim] != 1;
      }));
  return parallelLoops;
}

// LinalgTilingOptions need this function to compute tile size
// Set this func to LinalgTilingOptions.setTileSizeComputationFunction
// op is LinalgOp
// warp level tile only tile DIMM and DIMN
std::optional<SmallVector<Value>>
warpTileSizeComputationFunc(SmallVector<int64_t, 3> warpWorkGroup,
                            OpBuilder &builder, Operation *op) {
  func::FuncOp funcOp = op->getParentOfType<func::FuncOp>();
  // get Block Level tileConfig from the annotation in funcOp's attribute
  std::optional<SmallVector<int64_t, 3>> optionalGemmTileSize =
      getGemmTileSize(funcOp);
  if (!optionalGemmTileSize.has_value()) {
    return std::nullopt;
  }
  SmallVector<int64_t, 3> gemmTileSize = optionalGemmTileSize.value();

  SmallVector<int64_t> blockTileSize;
  auto linalgOp = cast<linalg::LinalgOp>(op);
  // blockTileSize gets gemmTileSize's M and N because dimK no need to be tiled
  if (linalgOp.getNumParallelLoops() == 3) { // BMM
    blockTileSize = {0, gemmTileSize[0], gemmTileSize[1]};
  } else { // Matmul
    blockTileSize = {gemmTileSize[0], gemmTileSize[1]};
  }

  SmallVector<Value> warpTileSize;

  auto partitionedLoops = getPartitionedLoopsFromLinalgOp(linalgOp);

  auto zero = builder.create<arith::ConstantIndexOp>(op->getLoc(), 0);
  warpTileSize.resize(cast<TilingInterface>(op).getLoopIteratorTypes().size(),
                      zero);
  // if M N is parallel remove the workGroup[2]
  warpWorkGroup.resize(partitionedLoops.size());
  // partitionedLoops is M N which needed to be mapped to WarpIdx.y WarpIdx.x
  std::reverse(warpWorkGroup.begin(), warpWorkGroup.end());

  unsigned idx = 0;
  for (auto depth : partitionedLoops) {
    if (depth >= blockTileSize.size()) {
      continue;
    }
    warpTileSize[depth] = builder.create<arith::ConstantIndexOp>(
        op->getLoc(),
        llvm::divideCeil(blockTileSize[depth], warpWorkGroup[idx++]));
  }
  return warpTileSize;
}

SmallVector<linalg::ProcInfo, 2> getWarpInfoDistributedToTile(
    OpBuilder &b, Location loc, ArrayRef<Range> parallelLoopRanges,
    unsigned warpSize, SmallVector<int64_t, 3> warpWorkGroup) {
  unsigned parallelDimNum = parallelLoopRanges.size();
  SmallVector<linalg::ProcInfo, 2> warpProcInfo(parallelDimNum);
  SmallVector<gpu::Dimension, 3> gpuDimAttrs = {
      gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
  for (unsigned i = 0; i < parallelDimNum; i++) {
    Value threadIdx_i = b.create<gpu::ThreadIdOp>(loc, gpuDimAttrs[i]);
    Value warpIdx_i;
    if (i == 0) { // warpIdx.x = threadIdx.x / 32
      mlir::AffineExpr d0 = b.getAffineDimExpr(0);
      warpIdx_i = affine::makeComposedAffineApply(
          b, loc, d0.floorDiv(b.getAffineConstantExpr(warpSize)),
          {threadIdx_i});
    } else {
      mlir::AffineExpr d0 = b.getAffineDimExpr(0);
      warpIdx_i = affine::makeComposedAffineApply(b, loc, d0, {threadIdx_i});
    }
    // warpProcInfo[i] mapped to the tiled scf dim
    warpProcInfo[parallelDimNum - 1 - i] = linalg::ProcInfo{
        warpIdx_i,
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(warpWorkGroup[i])),
        linalg::DistributionMethod::Cyclic};
  }
  return warpProcInfo;
}

LogicalResult distributeToWarpLevel(scf::ForallOp forallOp,
                                    SmallVector<int64_t, 3> &workGroup) {
  // Use LinalgTilingOptions to distribute to warp level

  // calculate WarpIdx
  // WarpIdx.x = threadIdx.x / 32
  // WarpIdx.y = threadIdx.y
  // WarpIdx.z = threadIdx.z
  if (workGroup[0] / warpSize == 0) {
    return failure();
  }
  SmallVector<int64_t, 3> warpWorkGroup = {workGroup[0] / warpSize,
                                           workGroup[1], workGroup[2]};

  linalg::LinalgTilingOptions linalgTilingOptions =
      linalg::LinalgTilingOptions();

  auto tileSizeComputationFunction = [=](OpBuilder &b, Operation *op) {
    return warpTileSizeComputationFunc(warpWorkGroup, b, op).value();
  };
  linalgTilingOptions.setTileSizeComputationFunction(
      tileSizeComputationFunction);

  linalg::LinalgLoopDistributionOptions distributionOptions;
  auto getWarpProcInfoFn = [=](OpBuilder &b, Location loc,
                               ArrayRef<Range> parallelLoopRanges) {
    return getWarpInfoDistributedToTile(b, loc, parallelLoopRanges, warpSize,
                                        warpWorkGroup);
  };
  distributionOptions.procInfo = getWarpProcInfoFn;
  linalgTilingOptions.setDistributionOptions(distributionOptions)
      .setLoopType(linalg::LinalgTilingLoopType::Loops);

  SmallVector<linalg::LinalgOp> candidates;
  forallOp.walk([&](linalg::LinalgOp linalgOp) {
    if (isa<linalg::FillOp>(linalgOp) || isLinalgMatmul(linalgOp)) {
      candidates.push_back(linalgOp);
    }
  });

  IRRewriter rewriter(forallOp->getContext());
  for (auto linalgOp : candidates) {
    FailureOr<linalg::TiledLinalgOp> res =
        linalg::tileLinalgOp(rewriter, linalgOp, linalgTilingOptions);
    if (failed(res)) {
      return failure();
    }
    setMarker(res->op, buddy::getVectorizeMarkerAttrName());
    if (res->tensorResults.empty()) {
      rewriter.eraseOp(linalgOp);
    } else {
      rewriter.replaceOp(linalgOp, res->tensorResults);
    }
  }

  return success();
}

struct GPUDistributeToWarpPass
    : public GPUDistributeToWarpBase<GPUDistributeToWarpPass> {
public:
  GPUDistributeToWarpPass() = default;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    if (!funcHasGemm(funcOp)) {
      return;
    }

    // get BlockSize from the annotation in funcOp's attribute
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
    if (failed(distributeToWarpLevel(forallOp, workGroup))) {
      return signalPassFailure();
    }

    {
      // Apply canonicalization patterns.
      RewritePatternSet threadTilingCanonicalizationPatterns =
          linalg::getLinalgTilingCanonicalizationPatterns(funcOp.getContext());
      // populateAffineMinSCFCanonicalizationPattern(
      //     threadTilingCanonicalizationPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(threadTilingCanonicalizationPatterns)))) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createGPUDistributeToWarpPass() {
  return std::make_unique<GPUDistributeToWarpPass>();
}

