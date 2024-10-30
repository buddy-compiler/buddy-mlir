#ifndef TRANSFORM_PASSDETAIL_H
#define TRANSFORM_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_CLASSES
#include "Transform/Passes.h.inc"

} // namespace mlir

#endif
