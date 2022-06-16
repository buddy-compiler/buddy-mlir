#ifndef BUDDY_EXAMPLES_TOY_DIALECT_H_
#define BUDDY_EXAMPLES_TOY_DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/IR/MLIRContext.h"

/// Include the auto-generated header file containing the declaration of the toy
/// dialect.
#include "toy/Dialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// toy operations.
#define GET_OP_CLASSES
#include "toy/Ops.h.inc"

#endif // BUDDY_EXAMPLES_TOY_DIALECT_H_
