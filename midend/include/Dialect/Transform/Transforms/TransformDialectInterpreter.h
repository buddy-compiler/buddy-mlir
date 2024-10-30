#ifndef TRANSFORM_TRANSFORMS_TRANSFORMDIALECTINTERPRETER_H
#define TRANSFORM_TRANSFORMS_TRANSFORMDIALECTINTERPRETER_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

class ModuleOp;

std::unique_ptr<OperationPass<ModuleOp>>
createTransformDialectInterpreter(bool eraseAfter = false);

} // namespace mlir

#endif
