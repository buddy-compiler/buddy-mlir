//===- BudDialect.cpp - Bud Dialect Definition-------------------*- C++ -*-===//
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
// This file defines bud dialect.
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

#include "Bud/BudDialect.h"
#include "Bud/BudOps.h"

using namespace mlir;
using namespace buddy::bud;

#include "Bud/BudOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// BudDialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct BudInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  // We don't have any special restrictions on what can be inlined into
  // destination regions (e.g. while/conditional bodies). Always allow it.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const final {
    return true;
  }
  // Operations in bud dialect are always legal to inline since they are
  // pure.
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Bud dialect.
//===----------------------------------------------------------------------===//

void BudDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Bud/BudOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Bud/BudOpsAttributes.cpp.inc"
      >();
  addInterfaces<BudInlinerInterface>();
}

#include "Bud/BudOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Bud/BudOpsAttributes.cpp.inc"
