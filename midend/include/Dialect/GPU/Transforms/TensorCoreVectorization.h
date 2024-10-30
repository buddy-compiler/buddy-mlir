#ifndef DIALECT_GPU_TRANSFORMS_TENSORCOREVECTORIZATION_H
#define DIALECT_GPU_TRANSFORMS_TENSORCOREVECTORIZATION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mlir {

std::unique_ptr<OperationPass<func::FuncOp>>
createTensorCoreVectorizationPass();

} // namespace mlir

#endif
