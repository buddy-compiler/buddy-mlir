#ifndef MEMREFEXP_MEMREFEXPDIALECT_H
#define MEMREFEXP_MEMREFEXPDIALECT_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "MemrefExp/MemrefExpOps.h.inc"
#endif // MEMREFEXP_MEMREFEXPDIALECT_H
