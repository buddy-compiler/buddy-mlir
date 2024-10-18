//===- TraceOps.cpp - Trace Dialect Ops --------------------------*- C++-*-===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines operations in the trace dialect.
//
//===----------------------------------------------------------------------===//

#include "Trace/TraceOps.h"
#include "Trace/TraceDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_OP_CLASSES
#include "Trace/TraceOps.cpp.inc"

//===----------------------------------------------------------------------===//
// ScopeOp
//===----------------------------------------------------------------------===//

using namespace mlir;
using namespace buddy::trace;

// ParseResult ScopeOp::parse(OpAsmParser &parser, OperationState &result) {
//   if (parser.parseOptionalArrowTypeList(result.types))
//     return failure();

//   // Introduce the body region and parse it.
//   Region *body = result.addRegion();
//   if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}) ||
//       parser.parseOptionalAttrDict(result.attributes))
//     return failure();

//   return success();
// }

// void ScopeOp::print(OpAsmPrinter &p) {
//   p.printOptionalArrowTypeList(getResultTypes());

//   p << ' ';
//   p.printRegion(getRegion(),
//                 /*printEntryBlockArgs=*/false,
//                 /*printBlockTerminators=*/true);

//   p.printOptionalAttrDict((*this)->getAttrs());
// }

LogicalResult ScopeOp::verify() {
  // if (getRegion().empty())
  //   return emitOpError("region needs to have at least one block");
  if (getRegion().front().getNumArguments() > 0)
    return emitOpError("region cannot have any arguments");
  return success();
}
