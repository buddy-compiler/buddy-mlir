#ifndef DIALECT_LINALG_TRANSFORMS_PASSDETAIL_H
#define DIALECT_LINALG_TRANSFORMS_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_CLASSES
#include "Linalg/Passes.h.inc"

} // namespace mlir

#endif
