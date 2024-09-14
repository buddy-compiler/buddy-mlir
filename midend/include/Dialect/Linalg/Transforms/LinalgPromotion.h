#ifndef LINALG_TRANSFORMS_LINALGPROMOTION_H
#define LINALG_TRANSFORMS_LINALGPROMOTION_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <memory>

namespace mlir {

std::unique_ptr<OperationPass<func::FuncOp>>
createLinalgPromotionPass();

} // namespace mlir

#endif