#ifndef PIPELINES_GPU_UTILS_H
#define PIPELINES_GPU_UTILS_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
class ModuleOp;
namespace buddy {
template <typename OpClass = ModuleOp, typename Builder, typename... Args>
void invokeOpPassPipelineBuilder(Builder builder, OpPassManager &pm,
                                 Args &&...args) {
  if (pm.getOpAnchorName() != OpPassManager::getAnyOpAnchorName() &&
      pm.getOpAnchorName() != OpClass::getOperationName()) {
    if (pm.getNesting() == OpPassManager::Nesting::Implicit) {
      builder(pm.nest<OpClass>(), std::forward<Args>(args)...);
      return;
    }
    llvm::report_fatal_error(
        llvm::Twine("Can't build pass pipeline on expected op type ") +
        OpClass::getOperationName() + " but got " + pm.getOpAnchorName());
  } else {
    builder(pm, std::forward<Args>(args)...);
  }
}

bool isLinalgMatmul(Operation *op);

} // namespace buddy::pipelines
} // namespace buddy

#endif