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
#include "llvm/Support/MathExtras.h"

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

LogicalResult
ScalingFactorAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                          StringRef ratio, int64_t value) {
  if (ratio != "m" && ratio != "f")
    return emitError() << "the ratio of a scaling factor is 'm' (multiplier) "
                          "or 'f' (fraction), not '"
                       << ratio << "'";
  if (value < 2 || value > 64 || !llvm::isPowerOf2_64(value))
    return emitError() << "the value of a scaling factor is 2, 4, 8, 16, 32 "
                          "or 64, not "
                       << value;
  return success();
}

ScalingFactorAttr ScalingFactorAttr::parseKeyword(AsmParser &parser) {
  SMLoc loc = parser.getCurrentLocation();
  StringRef token;
  if (parser.parseKeyword(&token))
    return {};
  int64_t value;
  if (token.drop_front(1).getAsInteger(10, value)) {
    parser.emitError(loc, "expected a scaling factor, 'm' or 'f' and its "
                          "value (e.g. m2), not '")
        << token << "'";
    return {};
  }
  return getChecked([&] { return parser.emitError(loc); }, parser.getContext(),
                    token.take_front(1), value);
}

Attribute ScalingFactorAttr::parse(AsmParser &parser, Type) {
  if (parser.parseLess())
    return {};
  ScalingFactorAttr attr = parseKeyword(parser);
  if (!attr || parser.parseGreater())
    return {};
  return attr;
}

void ScalingFactorAttr::print(AsmPrinter &printer) const {
  printer << "<" << this->str() << ">";
}
