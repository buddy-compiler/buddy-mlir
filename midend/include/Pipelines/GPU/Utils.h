#ifndef PIPELINES_GPU_UTILS_H
#define PIPELINES_GPU_UTILS_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
class ModuleOp;

namespace func {
class FuncOp;
} // namespace mlir::func

namespace buddy {

template <typename OpClass = ModuleOp, typename Builder, typename... Args>
void invokeOpPassPipelineBuilder(Builder builder, OpPassManager &pm,
                                 Args &&...args) {
  if (pm.getOpAnchorName() != OpPassManager::getAnyOpAnchorName() &&
      pm.getOpAnchorName() != OpClass::getOperationName()) {
    if (pm.getNesting() == OpPassManager::Nesting::Implicit) {
      builder(pm.nest<OpClass>(), std::forward<Args>(args)...);
      return;
    }
    llvm::report_fatal_error(
        llvm::Twine("Can't build pass pipeline on expected op type ") +
        OpClass::getOperationName() + " but got " + pm.getOpAnchorName());
  } else {
    builder(pm, std::forward<Args>(args)...);
  }
}

bool isLinalgMatmul(Operation *op);


static constexpr StringRef getGemmTileMConfigAttrName() {
    return "__buddy_gemm_tile_config__M";
}

static constexpr StringRef getGemmTileNConfigAttrName() {
    return "__buddy_gemm_tile_config__N";
}

static constexpr StringRef getGemmTileKConfigAttrName() {
    return "__buddy_gemm_tile_config__K";
}

static constexpr StringRef getGemmBlockXSizeAttrName() {
    return "__buddy_gemm_block_size__X";
}

static constexpr StringRef getGemmBlockYSizeAttrName() {
    return "__buddy_gemm_block_size__Y";
}

static constexpr StringRef getGemmBlockZSizeAttrName() {
    return "__buddy_gemm_block_size__Z";
}

static constexpr StringRef getGemmPipelineStageAttrName() {
    return "__buddy_gemm_pipeline_stage__";
}

static constexpr StringRef getMatmulKMainLoopMarker() {
    return "__buddy_gemm_main_loopk__";
}

constexpr StringRef getLinalgMMALevelAttrName() {
  return "__buddy_mma_level__";
}

constexpr StringRef getMMAPatternAttrName() { return "__buddy_mma__"; }


std::optional<SmallVector<int64_t, 3>> getGemmTileSize(func::FuncOp funcOp);

std::optional<SmallVector<int64_t, 3>> getGemmBlockSize(func::FuncOp funcOp);

std::optional<int64_t> getGemmPipelineStages(func::FuncOp funcOp);

void setMarker(mlir::Operation *op, llvm::StringRef marker);

bool hasMarker(Operation *op, StringRef marker);


} // namespace buddy::pipelines
} // namespace buddy

#endif