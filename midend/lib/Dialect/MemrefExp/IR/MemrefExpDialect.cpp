#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

#include "MemrefExp/MemrefExpDialect.h"
#include "MemrefExp/MemrefExpOps.h"

using namespace mlir;
using namespace buddy::memref_exp;

#include "MemrefExp/MemrefExpOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "MemrefExp/MemrefExpOps.cpp.inc"

void MemrefExpDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MemrefExp/MemrefExpOps.cpp.inc"
      >();
}
