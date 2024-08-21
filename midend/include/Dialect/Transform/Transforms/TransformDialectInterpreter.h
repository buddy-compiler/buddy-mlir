#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
    
class ModuleOp;

std::unique_ptr<OperationPass<ModuleOp>> 
createTransformDialectInterpreter(bool eraseAfter = false);

} // namespace mlir