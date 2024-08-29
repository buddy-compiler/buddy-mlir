#ifndef GPU_TRANSFORMS_UTILS_H
#define GPU_TRANSFORMS_UTILS_H

#include "mlir/Dialect/Linalg/Utils/Utils.h"

namespace mlir {

static constexpr StringRef getGemmTileConfigAttrName() {
    return "__buddy_gemm_tile_config__";
}

static constexpr StringRef getGemmBlockSizeAttrName() {
    return "__buddy_gemm_block_size__";
}

static constexpr StringRef getGemmPipelineStageAttrName() {
    return "__buddy_gemm_pipeline_stage__";
}

}

#endif