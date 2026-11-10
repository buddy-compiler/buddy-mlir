//===-- VIRTypes.cpp - Dynamic Vector IR Type Implementation --------------===//
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
// This file implements the type system for the Dynamic Vector IR (VIR) dialect,
// including dynamic vector types and their parsing/printing functionality.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include "VIR/VIRAttrs.h"
#include "VIR/VIRDialect.h"
#include "VIR/VIRTypes.h"

using namespace buddy::vir;
using namespace mlir;

#define GET_TYPEDEF_CLASSES
#include "VIR/VIRTypes.cpp.inc"

void VIRDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "VIR/VIRTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// DynamicVectorType
//===----------------------------------------------------------------------===//

LogicalResult
DynamicVectorType::verify(function_ref<InFlightDiagnostic()> emitError,
                          ArrayRef<int64_t> shape, Type elementType,
                          ScalingFactorAttr scalingFactor) {
  // static dims, then the dynamic one
  if (shape.empty() || shape.back() != ShapedType::kDynamic)
    return emitError() << "the last dim of a dynamic vector is the dynamic "
                          "one";
  for (int64_t dim : shape.drop_back())
    if (dim <= 0)
      return emitError() << "the dims before the last one of a dynamic "
                            "vector are static and positive, not "
                         << dim;
  return success();
}

::mlir::Type DynamicVectorType::parse(::mlir::AsmParser &parser) {
  SMLoc loc = parser.getCurrentLocation();
  ::llvm::SmallVector<int64_t> shape;
  ::mlir::Type elementType;
  ScalingFactorAttr scalingFactor;

  // Parse the opening '<'.
  if (parser.parseLess())
    return {};

  // Parse static dimension list (optional)
  // VectorIR is currently tailored for SIMD/Vector architectures.
  // As a result, in the Dynamic Vector Type, only the lowest dimension
  // is allowed to be dynamic.
  while (true) {
    // Try to parse a static dimension.
    int64_t dim;
    auto result = parser.parseOptionalInteger(dim);
    if (result.has_value()) {
      // Successfully parsed an integer.
      shape.push_back(dim);
      // Check for 'x' separator.
      if (parser.parseXInDimensionList())
        continue;
      else
        break;
    } else {
      // No more static dimensions, break to parse dynamic dimension.
      break;
    }
  }

  // Parse dynamic dimension '?'.
  if (parser.parseQuestion())
    return {};

  // Add dynamic dimension to shape.
  shape.push_back(::mlir::ShapedType::kDynamic);

  // Parse 'x' separator.
  if (parser.parseXInDimensionList())
    return {};

  // Parse the element type.
  if (parser.parseType(elementType))
    return {};

  // Parse optional ", <scaling-factor>": m2, or the attribute #vir.sf<m2>.
  if (succeeded(parser.parseOptionalComma())) {
    SMLoc attrLoc = parser.getCurrentLocation();
    ::mlir::Attribute attr;
    auto attrResult = parser.parseOptionalAttribute(attr);
    if (attrResult.has_value()) {
      if (failed(*attrResult))
        return {};
      scalingFactor = llvm::dyn_cast<ScalingFactorAttr>(attr);
      if (!scalingFactor)
        return parser.emitError(attrLoc, "expected a scaling factor (e.g. m2 "
                                         "or #vir.sf<m2>)"),
               ::mlir::Type();
    } else {
      scalingFactor = ScalingFactorAttr::parseKeyword(parser);
      if (!scalingFactor)
        return {};
    }
  }

  // Parse the closing '>'.
  if (parser.parseGreater())
    return {};

  return DynamicVectorType::getChecked([&] { return parser.emitError(loc); },
                                       parser.getContext(), shape, elementType,
                                       scalingFactor);
}

void DynamicVectorType::print(::mlir::AsmPrinter &printer) const {
  printer << "<";

  // Print dimensions.
  bool first = true;
  for (int64_t dim : getShape()) {
    if (!first) {
      printer << "x";
    }
    first = false;
    if (dim == ::mlir::ShapedType::kDynamic) {
      printer << "?";
    } else {
      printer << dim;
    }
  }

  // Print 'x' separator.
  printer << "x";

  // Print element type.
  printer << getElementType();

  // Print the optional scaling factor as the syntax has it, m2 (not the
  // attribute, #vir.sf<m2>).
  if (ScalingFactorAttr sf = getScalingFactor()) {
    printer << ", " << sf.str();
  }

  printer << ">";
}
