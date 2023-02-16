#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"


#include "Gemmini/GemminiOps.h"
#include "Gemmini/GemminiDialect.h"
using namespace mlir;
using namespace buddy::gemmini;

#include "Gemmini/GemminiDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Gemmini/Gemmini.cpp.inc"

void GemminiDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "Gemmini/Gemmini.cpp.inc"
      >();
}
