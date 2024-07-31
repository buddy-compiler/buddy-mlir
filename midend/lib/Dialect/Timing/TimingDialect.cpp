//===- TimingDialect.cpp - Timing Dialect Definition-------------------*- C++
//-*-===//
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
// This file defines Timing dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/SourceMgr.h"

#include "Timing/TimingDialect.h"
#include "Timing/TimingOps.h"

using namespace mlir;
using namespace buddy::timing;

#include "Timing/TimingOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TimingDialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct TimingInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  // We don't have any special restrictions on what can be inlined into
  // destination regions (e.g. while/conditional bodies). Always allow it.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
  // Operations in Timing dialect are always legal to inline since they are
  // pure.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Timing dialect.
//===----------------------------------------------------------------------===//

void TimingDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Timing/TimingOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Timing/TimingOpsAttributes.cpp.inc"
      >();
  addInterfaces<TimingInlinerInterface>();
}

#include "Timing/TimingOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Timing/TimingOpsAttributes.cpp.inc"
