//===- ZAMESOps.h - ZAMES Operation Declarations -------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ZAMES_ZAMESOPS_H
#define ZAMES_ZAMESOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "ZAMES/ZAMESTypes.h"

#define GET_OP_CLASSES
#include "ZAMES/ZAMES.h.inc"

#endif // ZAMES_ZAMESOPS_H
