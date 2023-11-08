//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_MLIR_INITALL_H
#define BUDDY_MLIR_INITALL_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace buddy {

void registerAllDialects(mlir::DialectRegistry &registry);
void registerAllPasses();

} // namespace buddy
} // namespace mlir

#endif // BUDDY_MLIR_INITALL_H