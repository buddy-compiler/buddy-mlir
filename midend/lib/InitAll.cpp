//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "buddy-mlir-c/InitAll.h"

#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"

#include "Dialect/Gemmini/GemminiDialect.h"

namespace mlir {
namespace buddy {

void registerLowerGemminiPass();
void registerLowerLinalgToGemminiPass();
} // namespace buddy
} // namespace mlir


void mlir::buddy::registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<::buddy::gemmini::GemminiDialect>();
}

void mlir::buddy::registerAllPasses() {
  mlir::buddy::registerLowerGemminiPass();
  mlir::buddy::registerLowerLinalgToGemminiPass();
}