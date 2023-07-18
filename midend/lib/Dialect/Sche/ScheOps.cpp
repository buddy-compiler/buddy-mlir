//===- BudOps.cpp - Bud Dialect Ops -----------------------------*- C++ -*-===//
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
// This file defines operations in the sche dialect.
//
//===----------------------------------------------------------------------===//

#include "Sche/ScheOps.h"
#include "Sche/ScheDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "Sche/ScheOps.cpp.inc"

using namespace mlir;
using namespace buddy::sche;

namespace buddy::sche {

    using BodyBuilderFn = mlir::function_ref<void(OpBuilder&, Location, ValueRange)>;
    
    void OnDeviceOp::build(OpBuilder& builder, OperationState& result, llvm::StringRef targetId,
                         llvm::StringRef targetConfig, TypeRange resultTypes,
                         BodyBuilderFn bodyBuilder) {
    result.addAttribute("targetId", builder.getStringAttr(targetId));
    result.addAttribute("targetConfig", builder.getStringAttr(targetConfig));

    Region* bodyRegion = result.addRegion();
    bodyRegion->push_back(new Block());
    auto& bodyBlock = bodyRegion->front();
    result.addTypes(resultTypes);

    if (bodyBuilder) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&bodyBlock);
        bodyBuilder(builder, result.location, bodyBlock.getArguments().drop_front());
    } else {
        llvm_unreachable("no body builder given for OnDevice op builder.");
    }
}
} //namespace