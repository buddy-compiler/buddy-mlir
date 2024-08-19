#pragma once
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
namespace mlir {
class ModuleOp;
// Generate the definition of Transform Passes
#define GEN_PASS_DECL
#include "Transform/Passes.h.inc"

// Include the constructor of passes in Transform Dialect
#include "Transform/Transforms/TransformDialectInterpreter.h"
#include "Transform/Transforms/TransformInsertion.h"

#define GEN_PASS_REGISTRATION
#include "Transform/Passes.h.inc"

}