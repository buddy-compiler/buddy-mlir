#ifndef DIALECT_GPU_TRANSFORMS_REMOVEREDUNTANTLOOPS_H
#define DIALECT_GPU_TRANSFORMS_REMOVEREDUNTANTLOOPS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mlir {

std::unique_ptr<OperationPass<func::FuncOp>> createRemoveReduntantLoops();

} // namespace mlir

#endif