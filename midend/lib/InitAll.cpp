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

#include "Dialect/Bud/BudDialect.h"
#include "Dialect/DAP/DAPDialect.h"
#include "Dialect/DIP/DIPDialect.h"
#include "Dialect/Gemmini/GemminiDialect.h"
#include "Dialect/RVV/RVVDialect.h"
#include "Dialect/Sche/ScheDialect.h"
#include "Dialect/VectorExp/VectorExpDialect.h"

namespace mlir {
namespace buddy {
  void registerConvOptimizePass();
  void registerConvVectorizationPass();
  void registerPointwiseConvToGemmPass();
  void registerPoolingVectorizationPass();
  void registerLowerBudPass();
  void registerLowerDAPPass();
  void registerLowerDIPPass();
  void registerLowerGemminiPass();
  void registerLowerLinalgToGemminiPass();
  void registerLowerRVVPass();
  void registerLowerSchePass();
  void registerLowerVectorExpPass();
  void registerBatchMatMulOptimizePass();
  void registerMatMulOptimizePass();
  void registerMatMulParallelVectorizationPass();
  void registerMatMulVectorizationPass();
  void registerDeviceSchedulePass();
  void registerTransposeOptimizationPass();
} // namespace buddy
} // namespace mlir

void mlir::buddy::registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<::buddy::bud::BudDialect>();
  registry.insert<::buddy::dap::DAPDialect>();
  registry.insert<::buddy::dip::DIPDialect>();
  registry.insert<::buddy::gemmini::GemminiDialect>();
  registry.insert<::buddy::rvv::RVVDialect>();
  registry.insert<::buddy::sche::ScheDialect>();
  registry.insert<::buddy::vector_exp::VectorExpDialect>();
}

void mlir::buddy::registerAllPasses() {
  mlir::buddy::registerConvOptimizePass();
  mlir::buddy::registerConvVectorizationPass();
  mlir::buddy::registerPointwiseConvToGemmPass();
  mlir::buddy::registerPoolingVectorizationPass();
  mlir::buddy::registerLowerBudPass();
  mlir::buddy::registerLowerDAPPass();
  mlir::buddy::registerLowerDIPPass();
  mlir::buddy::registerLowerGemminiPass();
  mlir::buddy::registerLowerLinalgToGemminiPass();
  mlir::buddy::registerLowerRVVPass();
  mlir::buddy::registerLowerSchePass();
  mlir::buddy::registerLowerVectorExpPass();
  mlir::buddy::registerBatchMatMulOptimizePass();
  mlir::buddy::registerMatMulOptimizePass();
  mlir::buddy::registerMatMulParallelVectorizationPass();
  mlir::buddy::registerMatMulVectorizationPass();
  mlir::buddy::registerDeviceSchedulePass();
  mlir::buddy::registerTransposeOptimizationPass();
}