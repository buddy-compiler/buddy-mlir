#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "Pipelines/GPU/Utils.h"

#include <optional>

using namespace mlir;
using namespace llvm;

namespace mlir {
namespace buddy {

bool isLinalgMatmul(Operation *op) {
    if (!llvm::isa<linalg::LinalgOp>(op)) {
        return false;
    }

    linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);
    if (isa<linalg::MatmulOp>(linalgOp) || isa<linalg::BatchMatmulOp>(linalgOp)) {
        return true;
    } else {
        if (!(linalg::isaContractionOpInterface(linalgOp) &&
                linalgOp.getNumParallelLoops() >= 2 &&
                linalgOp.getNumParallelLoops() <= 3)) {
            return false;
        }
        Region &body = linalgOp->getRegion(0);
        Region::OpIterator it = body.op_begin();
        while (it != body.op_end() && isa<arith::ExtFOp>(*it)) 
            it++;
        if (it == body.op_end() || !isa<arith::MulFOp>(*(it++)))
            return false;
        if (it == body.op_end() || !isa<arith::AddFOp>(*(it++)))
            return false;
        if (it == body.op_end() || !isa<linalg::YieldOp>(*(it++)))
            return false;
        AffineMap outputMap = linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(0));
        if (outputMap.getNumResults() != outputMap.getNumDims() - 1)
            return false;
        OpBuilder b(linalgOp);
        for (unsigned i = 0; i < outputMap.getNumResults(); i++) {
            if (outputMap.getResult(i) != b.getAffineDimExpr(i))
                return false;
        }
        return true;
    }
}


std::optional<SmallVector<int64_t, 3>> getGemmTileSize(func::FuncOp funcOp) {
    if (funcOp->hasAttr(getGemmTileMConfigAttrName()) &&
        funcOp->hasAttr(getGemmTileNConfigAttrName()) &&
        funcOp->hasAttr(getGemmTileKConfigAttrName())) {
        auto tileConfigM = funcOp->getAttrOfType<IntegerAttr>(
                            getGemmTileMConfigAttrName()).getInt();
        auto tileConfigN = funcOp->getAttrOfType<IntegerAttr>(
                            getGemmTileNConfigAttrName()).getInt();
        auto tileConfigK = funcOp->getAttrOfType<IntegerAttr>(
                            getGemmTileKConfigAttrName()).getInt();

        llvm::SmallVector<int64_t, 3> configVec = {tileConfigM, tileConfigN, tileConfigK};

        return configVec;
    }
    return std::nullopt;
}

std::optional<SmallVector<int64_t, 3>> getGemmBlockSize(func::FuncOp funcOp) {
    if (funcOp->hasAttr(getGemmBlockXSizeAttrName()) &&
        funcOp->hasAttr(getGemmBlockYSizeAttrName()) &&
        funcOp->hasAttr(getGemmBlockZSizeAttrName())) {
        auto blockSizeX = funcOp->getAttrOfType<IntegerAttr>(
                            getGemmBlockXSizeAttrName()).getInt();
        auto blockSizeY = funcOp->getAttrOfType<IntegerAttr>(
                            getGemmBlockYSizeAttrName()).getInt();
        auto blockSizeZ = funcOp->getAttrOfType<IntegerAttr>(
                            getGemmBlockZSizeAttrName()).getInt();

        llvm::SmallVector<int64_t, 3> blockSizeVec = {blockSizeX, blockSizeY, blockSizeZ};

        return blockSizeVec;
    }
    return std::nullopt;
}

std::optional<int64_t> getGemmPipelineStages(func::FuncOp funcOp) {
    if (funcOp->hasAttr(getGemmPipelineStageAttrName())) {
        auto stages = funcOp->getAttrOfType<IntegerAttr>(
                        getGemmPipelineStageAttrName()).getInt();
        return stages;
    }
    return std::nullopt;
}

void setMarker(Operation *op, StringRef marker) {
  op->setAttr(marker, UnitAttr::get(op->getContext()));
}

bool hasMarker(Operation *op, StringRef marker) {
  return op->hasAttrOfType<UnitAttr>(marker);
}

} // namespace mlir::buddy
} // namespace mlir