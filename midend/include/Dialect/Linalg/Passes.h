#ifndef DIALECT_LINALG_PASSES_H
#define DIALECT_LINALG_PASSES_H

// Include the constructor of passes in Linalg Dialect
#include "Linalg/Transforms/LinalgPromotion.h"

namespace mlir {
// Generate the definition of Linalg Passes
#define GEN_PASS_DECL
#include "Linalg/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Linalg/Passes.h.inc"

} // namespace mlir