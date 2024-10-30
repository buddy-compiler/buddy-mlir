#ifndef DIALECT_TRANSFORM_PASSES_H
#define DIALECT_TRANSFORM_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

// Include the constructor of passes in Transform Dialect
#include "Transform/Transforms/TransformDialectInterpreter.h"
#include "Transform/Transforms/TransformInsertion.h"

namespace mlir {
class ModuleOp;
// Generate the definition of Transform Passes
#define GEN_PASS_DECL
#include "Transform/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Transform/Passes.h.inc"

} // namespace mlir

#endif
