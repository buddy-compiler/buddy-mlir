//===-- VIRAttrs.cpp - Dynamic Vector IR Attribute Implementation ---------===//
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
// This file implements the attribute system for the Dynamic Vector IR (VIR)
// dialect, including scaling factor attributes and their parsing/printing.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "VIR/VIRAttrs.h"
#include "VIR/VIRDialect.h"

using namespace mlir;
using namespace buddy::vir;

#define GET_ATTRDEF_CLASSES
#include "VIR/VIRAttrs.cpp.inc"

void VIRDialect::registerAttrs() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "VIR/VIRAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Custom parse/print for ScalingFactorAttr (hasCustomAssemblyFormat = 1).
//===----------------------------------------------------------------------===//

Attribute ScalingFactorAttr::parse(AsmParser &parser, Type) {
  if (parser.parseLess())
    return {};

  StringRef token;
  if (parser.parseKeyword(&token))
    return {};

  StringRef ratio = token.take_front(1);
  if (!(ratio == "m" || ratio == "f"))
    return {};

  int64_t value;
  if (token.drop_front(1).getAsInteger(10, value))
    return {};

  if (parser.parseGreater())
    return {};

  return get(parser.getContext(), ratio, value);
}

void ScalingFactorAttr::print(AsmPrinter &printer) const {
  printer << "<" << this->str() << ">";
}
