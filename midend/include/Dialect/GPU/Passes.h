#ifndef DIALECT_GPU_PASSES_H
#define DIALECT_GPU_PASSES_H

// Include the constructor of passes in GPU Dialect
#include "GPU/Transforms/GPUDistributeToWarp.h" 
#include "GPU/Transforms/RemoveReduntantLoops.h"

namespace mlir {
// Generate the definition of GPU Passes
#define GEN_PASS_DECL
#include "GPU/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "GPU/Passes.h.inc"

} // namespace mlir

#endif