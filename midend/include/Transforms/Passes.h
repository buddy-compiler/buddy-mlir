#ifndef BUDDY_TRANSFORMS_PASSES_H
#define BUDDY_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace buddy {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Transforms/Passes.h.inc"

} // namespace buddy
} // namespce mlir
#endif // BUDDY_TRANSFORMS_PASSES_H
