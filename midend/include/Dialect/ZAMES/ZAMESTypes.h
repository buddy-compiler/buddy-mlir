//===- ZAMESTypes.h - ZAMES Type Declarations ----------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ZAMES_ZAMESTYPES_H
#define ZAMES_ZAMESTYPES_H

#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "ZAMES/ZAMESTypes.h.inc"

#endif // ZAMES_ZAMESTYPES_H
