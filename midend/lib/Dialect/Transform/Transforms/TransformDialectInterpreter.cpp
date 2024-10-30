#include "Transform/Transforms/TransformDialectInterpreter.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/BuiltinOps.h"

#include "PassDetail.h"

using namespace mlir;

namespace {

struct TransformDialectInterpreterPass
    : public TransformDialectInterpreterBase<TransformDialectInterpreterPass> {

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
mlir::createTransformDialectInterpreter(bool eraseAfter) {
  return std::make_unique<TransformDialectInterpreterPass>(eraseAfter);
}