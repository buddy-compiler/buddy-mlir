#ifndef GEMMINI_GEMMINIDIALECT_H
#define GEMMINI_GEMMINIDIALECT_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Gemmini/Gemmini.h.inc"
#endif // GEMMINI_GEMMINIDIALECT_H
