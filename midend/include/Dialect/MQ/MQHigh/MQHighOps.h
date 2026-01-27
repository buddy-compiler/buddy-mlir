//===- MQHighOps.h - MQHigh dialect operations ------------*- C++ -*-===//
//
// This file declares the operations for the MQHigh dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_MQHIGH_OPS_H
#define BUDDY_MQHIGH_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"



// TODO
#include "MQ/MQHigh/MQHighOpsEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "MQ/MQHigh/MQHighOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "MQ/MQHigh/MQHighOps.h.inc"

#endif // BUDDY_MQHIGH_OPS_H
