#ifndef DIALECT_GPU_TRANSFORMS_PASSDETAIL_H
#define DIALECT_GPU_TRANSFORMS_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_CLASSES
#include "GPU/Passes.h.inc"

} // namespace mlir

#endif
