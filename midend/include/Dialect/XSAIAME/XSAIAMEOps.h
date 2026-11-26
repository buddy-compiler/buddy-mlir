//===- XSAIAMEOps.h - XSAIAME Operation Declarations ------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef XSAIAME_XSAIAMEOPS_H
#define XSAIAME_XSAIAMEOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "XSAIAME/XSAIAMETypes.h"

#define GET_OP_CLASSES
#include "XSAIAME/XSAIAME.h.inc"

#endif // XSAIAME_XSAIAMEOPS_H
