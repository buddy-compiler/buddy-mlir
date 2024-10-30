#ifndef DIALECT_LINALG_TRANSFORMS_LINALGPROMOTION_H
#define DIALECT_LINALG_TRANSFORMS_LINALGPROMOTION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mlir {

std::unique_ptr<OperationPass<func::FuncOp>> createLinalgPromotionPass();

} // namespace mlir

#endif