//===- GPUDistributeSharedMemCopy.cpp --------------------------------------===//
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
// The process in this file references the IREE project,
// which is hereby acknowledged.
// For the license of the IREE project
// please see: https://github.com/iree-org/iree/blob/main/LICENSE
//
//===----------------------------------------------------------------------===//

#include "GPU/Transforms/GPUDistributeSharedMemCopy.h"
#include "PassDetail.h"
#include "Utils/GemmCodegenUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <memory>
#include <mlir/Support/LogicalResult.h>
#include <optional>

using namespace mlir;
using namespace llvm;
using namespace mlir::buddy;

namespace {

static constexpr int copyVectorNumBits = 128;

static std::optional<SmallVector<linalg::GenericOp>>
transformLinalgCopyOpsToGenericOps(ArrayRef<linalg::CopyOp> linalgCopyOps) {
  SmallVector<linalg::GenericOp> genericLinalgCopyOps;
  for (auto linalgCopyOp : linalgCopyOps) {
    auto attributeDict = linalgCopyOp->getAttrDictionary();
    IRRewriter rewriter(linalgCopyOp->getContext());
    rewriter.setInsertionPoint(linalgCopyOp);
    auto failureOrGenericLinalgCopyOp =
        linalg::generalizeNamedOp(rewriter, linalgCopyOp);
    if (failed(failureOrGenericLinalgCopyOp)) {
      linalgCopyOp.emitError("Cannot generalize Linalg Named Copy Op\n");
      return std::nullopt;
    } else {
      auto genericLinalgCopyOp = failureOrGenericLinalgCopyOp.value();
      // genericLinalgCopyOp->setDiscardableAttrs(attributeDict);
      genericLinalgCopyOp->setAttrs(attributeDict);
      genericLinalgCopyOps.push_back(genericLinalgCopyOp);
    }
  }
  return genericLinalgCopyOps;
}

// 1.
static inline unsigned getTypeBitWidth(Type type) {
  if (auto complexType = type.dyn_cast<ComplexType>()) {
    return 2 * complexType.getElementType().getIntOrFloatBitWidth();
  }
  if (auto vectorType = type.dyn_cast<VectorType>()) {
    return vectorType.getNumElements() *
           getTypeBitWidth(vectorType.getElementType());
  }
  return type.getIntOrFloatBitWidth();
}

static int getLoadVectorSize(linalg::GenericOp genericOp) {
  assert(genericOp.getNumDpsInits() == 1);
  unsigned resultBitWidth =
      llvm::cast<MemRefType>(genericOp.getDpsInitOperand(0)->get().getType())
          .getElementTypeBitWidth();

  unsigned operandBitWidth = std::numeric_limits<unsigned>::max();
  for (OpOperand *operand : genericOp.getDpsInputOperands()) {
    unsigned b = getTypeBitWidth(getElementTypeOrSelf(operand->get()));
    operandBitWidth = std::min(operandBitWidth, b);
  }
  // copyVectorNumbits = 128
  int vectorSize = copyVectorNumBits / resultBitWidth;
  if (operandBitWidth < resultBitWidth && operandBitWidth < 8) {
    vectorSize *= 8 / operandBitWidth;
  }
  return vectorSize;
}

static std::optional<SmallVector<int64_t>>
getCopyOpDistributedTileSize(linalg::GenericOp genericOp,
                             int64_t totalWorkGroupSize) {
  SmallVector<int64_t> copySharedMemShape = genericOp.getStaticLoopRanges();
  int loadVectorSize = getLoadVectorSize(genericOp);

  SmallVector<int64_t> unroll;
  assert(copySharedMemShape.back() % loadVectorSize == 0);
  for (auto [index, dim] : llvm::enumerate(llvm::reverse(copySharedMemShape))) {
    int64_t numElementPerThread = index == 0 ? loadVectorSize : 1;
    int64_t numThreads =
        std::min(dim / numElementPerThread, totalWorkGroupSize);

    unroll.push_back(numThreads * numElementPerThread);
    assert(totalWorkGroupSize % numThreads == 0);
    totalWorkGroupSize = totalWorkGroupSize / numThreads;
    if (totalWorkGroupSize == 1)
      break;
  }
  assert(totalWorkGroupSize == 1);
  unroll.resize(copySharedMemShape.size(), 1);
  std::reverse(unroll.begin(), unroll.end());

  return unroll;
}

static LogicalResult tileLinalgCopyOp(scf::ForallOp forallOp,
                                      int64_t totalWorkGroupSize) {
  linalg::TileSizeComputationFunction copyTileSizeFn = [=](OpBuilder &builder,
                                                           Operation *op) {
    SmallVector<Value> tileSizeVals;
    linalg::GenericOp genericCopyOp = dyn_cast<linalg::GenericOp>(op);
    if (!genericCopyOp) {
      return tileSizeVals;
    }
    std::optional<SmallVector<int64_t>> optionalStaticSize =
        getCopyOpDistributedTileSize(genericCopyOp, totalWorkGroupSize);
    SmallVector<int64_t> staticSize = optionalStaticSize.value();
    for (auto dim : staticSize) {
      tileSizeVals.push_back(
          builder.create<arith::ConstantIndexOp>(op->getLoc(), dim));
    }
    return tileSizeVals;
  };

  auto linalgTilingOptions =
      linalg::LinalgTilingOptions()
          .setLoopType(linalg::LinalgTilingLoopType::Loops)
          .setTileSizeComputationFunction(copyTileSizeFn);

  SmallVector<linalg::LinalgOp> candidates;
  forallOp.walk([&](linalg::LinalgOp linalgOp) {
    if (hasMarker(linalgOp, getCopyToSharedMemoryAMarker()) ||
        hasMarker(linalgOp, getCopyToSharedMemoryBMarker()) ||
        hasMarker(linalgOp, getCopyFromSharedMemoryAccMarker())) {
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

/// Return a flattened Id Value by combining the 3D gpu thread IDs.
static Value createFlatId(scf::ForallOp forallOp,
                          SmallVector<int64_t, 3> workgroupSize) {
  OpBuilder b = OpBuilder::atBlockBegin(forallOp.getBody());
  Type indexType = b.getIndexType();
  Value threadX = b.create<gpu::ThreadIdOp>(forallOp.getLoc(), indexType,
                                            gpu::Dimension::x);
  Value threadY = b.create<gpu::ThreadIdOp>(forallOp.getLoc(), indexType,
                                            gpu::Dimension::y);
  Value threadZ = b.create<gpu::ThreadIdOp>(forallOp.getLoc(), indexType,
                                            gpu::Dimension::z);
  AffineExpr d0 = getAffineDimExpr(0, b.getContext());
  AffineExpr d1 = getAffineDimExpr(1, b.getContext());
  AffineExpr d2 = getAffineDimExpr(2, b.getContext());
  Value flatThreadId = affine::makeComposedAffineApply(
      b, forallOp.getLoc(),
      d0 + workgroupSize[0] * d1 + (workgroupSize[0] * workgroupSize[1]) * d2,
      {threadX, threadY, threadZ});
  return flatThreadId;
}

static SmallVector<mlir::linalg::ProcInfo>
getThreadProcInfo(OpBuilder &b, Location loc,
                  ArrayRef<Range> parallelLoopRanges, Value flatThreadId) {
  SmallVector<linalg::ProcInfo> threadProcInfos;
  Value id = flatThreadId;
  AffineExpr d0 = b.getAffineConstantExpr(0);
  for (Range range : llvm::reverse(parallelLoopRanges)) {
    auto size = range.size.dyn_cast<Attribute>();
    auto offset = range.offset.dyn_cast<Attribute>();
    auto stride = range.stride.dyn_cast<Attribute>();

    int64_t numThreadsPerDim = (llvm::cast<IntegerAttr>(size).getInt() -
                                llvm::cast<IntegerAttr>(offset).getInt()) /
                               llvm::cast<IntegerAttr>(stride).getInt();
    Value dimId = id;
    linalg::ProcInfo procInfo;
    procInfo.procId = dimId;
    procInfo.nprocs = b.create<arith::ConstantIndexOp>(loc, numThreadsPerDim);
    procInfo.distributionMethod =
        linalg::DistributionMethod::CyclicNumProcsEqNumIters;
    threadProcInfos.push_back(procInfo);
  }
  std::reverse(threadProcInfos.begin(), threadProcInfos.end());
  return threadProcInfos;
}

static SmallVector<int64_t> getVectorizeDstShape(linalg::GenericOp copyOp) {
  SmallVector<int64_t> vectorDstShape;
  for (auto dim : copyOp.getStaticLoopRanges()) {
    vectorDstShape.push_back((dim == 1 ? 0 : 1));
  }
  vectorDstShape.back() = getLoadVectorSize(copyOp);
  return vectorDstShape;
}

static LogicalResult
distributeLinalgCopyOp(scf::ForallOp forallOp,
                       SmallVector<int64_t, 3> workGroupSize) {
  linalg::TileSizeComputationFunction wgCopyTileSizeFn =
      [](OpBuilder &builder, Operation *operation) {
        SmallVector<Value> tileSizesVal;
        auto copyOp = dyn_cast<linalg::GenericOp>(operation);
        if (!copyOp)
          return tileSizesVal;
        SmallVector<int64_t> staticSize = getVectorizeDstShape(copyOp);
        for (int64_t dim : staticSize) {
          tileSizesVal.push_back(
              builder.create<arith::ConstantIndexOp>(operation->getLoc(), dim));
        }
        return tileSizesVal;
      };

  Value flatThreadId = createFlatId(forallOp, workGroupSize);
  linalg::LinalgLoopDistributionOptions distributionOptions;
  auto getProcInfoFn = [flatThreadId](OpBuilder &b, Location loc,
                                      ArrayRef<Range> parallelLoopRanges) {
    return getThreadProcInfo(b, loc, parallelLoopRanges, flatThreadId);
  };
  distributionOptions.procInfo = getProcInfoFn;

  linalg::LinalgTilingOptions linalgTilingOptions;
  linalgTilingOptions.setLoopType(linalg::LinalgTilingLoopType::ParallelLoops)
      .setTileSizeComputationFunction(wgCopyTileSizeFn)
      .setDistributionOptions(distributionOptions);
  return success();
}

static LogicalResult
distributeSharedMemCopyToThread(scf::ForallOp forallOp,
                                SmallVector<int64_t, 3> workGroupSize) {
  SmallVector<linalg::CopyOp> linalgSharedMemCopyOps;
  forallOp.walk([&](linalg::CopyOp copyOp) {
    if (hasMarker(copyOp, getCopyToSharedMemoryAMarker()) ||
        hasMarker(copyOp, getCopyToSharedMemoryBMarker()) ||
        hasMarker(copyOp, getCopyFromSharedMemoryAccMarker())) {
      linalgSharedMemCopyOps.push_back(copyOp);
    }
  });

  if (linalgSharedMemCopyOps.empty()) {
    return success();
  }

  // step 1. transform Named Op to Generic Op
  auto optionalLinalgGenericOps =
      transformLinalgCopyOpsToGenericOps(linalgSharedMemCopyOps);
  if (!optionalLinalgGenericOps.has_value()) {
    return failure();
  }

  int64_t totalWorkGroupSize = 1;
  for (auto workGroup_i : workGroupSize) {
    totalWorkGroupSize = totalWorkGroupSize * workGroup_i;
  }
  // step 2. tile Linalg Generic Copy Op
  if (failed(tileLinalgCopyOp(forallOp, totalWorkGroupSize))) {
    return failure();
  }

  // step 3. distribute tile of Linalg Copy Op to threads
  if (failed(distributeLinalgCopyOp(forallOp, workGroupSize))) {
    return failure();
  }

  return success();
}

struct GPUDistributeSharedMemCopyPass
    : public GPUDistributeSharedMemCopyBase<GPUDistributeSharedMemCopyPass> {
public:
  GPUDistributeSharedMemCopyPass() = default;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    auto context = funcOp->getContext();
    if (!funcHasGemm(funcOp)) {
      return signalPassFailure();
    }

    std::optional<SmallVector<int64_t, 3>> optionalWorkGroupSize =
        getGemmBlockSize(funcOp);
    if (!optionalWorkGroupSize.has_value()) {
      return signalPassFailure();
    }
    SmallVector<int64_t, 3> workGroupSize = optionalWorkGroupSize.value();

    std::optional<scf::ForallOp> optionalForallOp =
        getForallOpMappedToBlock(funcOp);
    if (!optionalForallOp.has_value()) {
      return signalPassFailure();
    }
    auto forallOp = optionalForallOp.value();
    if (failed(distributeSharedMemCopyToThread(forallOp, workGroupSize))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createGPUDistributeSharedMemCopyPass() {
  return std::make_unique<GPUDistributeSharedMemCopyPass>();
}
