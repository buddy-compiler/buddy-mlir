#include "RVV/RVVDialect.h"

using namespace mlir;
using namespace buddy::rvv;

namespace {
LogicalResult verifyFloatBinary(Operation *op) {
  Type left = op->getOperand(0).getType();
  Type right = op->getOperand(1).getType();
  if (isa<VectorType>(right) && left != right)
    return op->emitOpError("requires matching vector operand types");
  return success();
}
} // namespace

LogicalResult RVVFAddOp::verify() { return verifyFloatBinary(*this); }
LogicalResult RVVFSubOp::verify() { return verifyFloatBinary(*this); }
LogicalResult RVVFMulOp::verify() { return verifyFloatBinary(*this); }
LogicalResult RVVFDivOp::verify() { return verifyFloatBinary(*this); }
