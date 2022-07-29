//===- DIPOps.cpp - dip Dialect Ops -----------------------------*- C++ -*-===//
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
// This file defines operations in the DIP dialect.
//
//===----------------------------------------------------------------------===//

#include "DIP/DIPOps.h"
#include "DIP/DIPDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

#define GET_OP_CLASSES
#include "DIP/DIPOps.cpp.inc"

using namespace mlir;

LogicalResult buddy::dip::Corr2DOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {

  auto inputTy = operands[0].getType();
  inferredReturnTypes.assign({inputTy});
  return success();
}

bool buddy::dip::Corr2DOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  Type lhs = l.front();
  Type rhs = r.front();

  return lhs == rhs;
}
