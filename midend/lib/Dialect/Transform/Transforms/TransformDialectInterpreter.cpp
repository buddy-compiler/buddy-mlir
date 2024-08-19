#define GEN_PASS_DEF_TRANSFORMDIALECTINTERPRETER
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "Transform/Passes.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace {

struct TransformDialectInterpreterPass
    : public impl::TransformDialectInterpreterBase<TransformDialectInterpreterPass> {

    explicit TransformDialectInterpreterPass(bool erase)
        : TransformDialectInterpreterBase() {
        eraseAfter = erase;
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        for (auto op : module.getOps<transform::TransformOpInterface>()) {
            RaggedArray<transform::MappedValue> extraMappings;
            if (failed(transform::applyTransforms(
                module, op, extraMappings,
                transform::TransformOptions().enableExpensiveChecks(false)))) {
                return signalPassFailure();
            }
        }
        if (eraseAfter) {
            module.walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
                if (isa<transform::TransformOpInterface>(nestedOp)) {
                    nestedOp->erase();
                    return WalkResult::skip();
                }
                return WalkResult::advance();
            });
        }
    }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createTransformDialectInterpreter(bool eraseAfter = false) {
    return std::make_unique<TransformDialectInterpreterPass>(eraseAfter);
}