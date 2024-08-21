#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir {

class ModuleOp;
class ImplicitLocOpBuilder;

struct TransformInsertionConfig {
    std::string funcAnchor;
    std::string matchPrefix;
    std::function<bool(Operation *)> opFilter;
    std::function<void(ImplicitLocOpBuilder &, Operation *, Value)> transformBuilder;
}

std::unique_ptr<OperationPass<ModuleOp>>
createGenericTransformInsertionPass(const TransformInsertionConfig &config);

}