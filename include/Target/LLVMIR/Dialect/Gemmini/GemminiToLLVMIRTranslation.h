#ifndef TARGET_LLVMIR_DIALECT_GEMMINI_GEMMINITOLLVMIRTRANSLATION_H
#define TARGET_LLVMIR_DIALECT_GEMMINI_GEMMINITOLLVMIRTRANSLATION_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

namespace buddy {
  void registerGemminiDialectTranslation(mlir::DialectRegistry &registry);
  void registerGemminiDialectTranslation(mlir::MLIRContext &context);
}

#endif