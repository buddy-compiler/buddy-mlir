#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

#include "Utils/GemmCodegenUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

using namespace mlir;
using namespace llvm;

namespace mlir {
namespace buddy {

void setMarker(Operation *op, StringRef marker) {
  op->setAttr(marker, UnitAttr::get(op->getContext()));
}

bool hasMarker(Operation *op, StringRef marker) {
  return op->hasAttrOfType<UnitAttr>(marker);
}


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

bool funcHasGemm(func::FuncOp funcOp) {
    // TODO
    return true;
}

bool isMappedToGPUBlock(scf::ForallOp forallOp) {

    SmallVector<int64_t> mappingIdx{2, 1, 0}; 
    MLIRContext ctx;
    ctx.loadDialect<gpu::GPUDialect>();
    auto mapping = llvm::to_vector(llvm::map_range(
            mappingIdx, 
            [](int64_t i){return static_cast<gpu::MappingId>(i);}));
    auto mappingAttrs = llvm::to_vector(llvm::map_range(
            mapping,
            [&](gpu::MappingId dim) -> Attribute {
                return gpu::GPUBlockMappingAttr::get(&ctx, dim);}));
    ArrayAttr getMappingAttrs = forallOp->getAttrOfType<ArrayAttr>("mapping");

    if (!getMappingAttrs) {
        return false;
    } else {
        for (auto mappingAttr : getMappingAttrs) {
            if (mappingAttr.isa<gpu::GPUBlockMappingAttr>()) {
                return true;
            }
        }
        return false;
    }
}

std::optional<scf::ForallOp> getForallOpMappedToBlock(func::FuncOp funcOp) {
    SmallVector<scf::ForallOp> forallOps;
    funcOp->walk([&](scf::ForallOp forallOp){
        if (isMappedToGPUBlock(forallOp)) {
            forallOps.push_back(forallOp);
        }
    });
    // one func one kernel -> one func only have one forallOp mapped to block
    if (forallOps.size() != 1) {
        llvm::errs() << "this funcOp has no ForallOp MappedToBlock\n";
        return std::nullopt; 
    }
    return forallOps[0];
}

} // namespace mlir::buddy
} // namespace mlir