//===- RVProfDialect.cpp - RVProf Dialect Definition ------------*- C++ -*-===//
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
// This file defines the RVProf dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/InliningUtils.h"

#include "RVProf/RVProfDialect.h"
#include "RVProf/RVProfOps.h"

using namespace mlir;
using namespace buddy::rvprof;

#include "RVProf/RVProfOpsDialect.cpp.inc"

namespace {
struct RVProfInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }

  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// RVProf dialect.
//===----------------------------------------------------------------------===//

void RVProfDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "RVProf/RVProfOps.cpp.inc"
      >();
  addInterfaces<RVProfInlinerInterface>();
}
