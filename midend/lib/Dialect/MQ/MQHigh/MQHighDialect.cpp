//===- MQHighDialect.cpp - MQHigh dialect implementation ------------*- C++ -*-===//
//
// This file implements the MQHigh dialect for the MoonQuest accelerator.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Transforms/InliningUtils.h"

#include "MQ/MQHigh/MQHighDialect.h"
#include "MQ/MQHigh/MQHighOps.h"

using namespace mlir;
using namespace buddy::mqhigh;

#include "MQ/MQHigh/MQHighOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// dialect.
//===----------------------------------------------------------------------===//

void MQHighDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MQ/MQHigh/MQHighOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "MQ/MQHigh/MQHighOpsAttributes.cpp.inc"
      >();
//   addInterfaces<BudInlinerInterface>();
}