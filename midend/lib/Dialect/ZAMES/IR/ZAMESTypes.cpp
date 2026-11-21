//===- ZAMESTypes.cpp - ZAMES Type Implementation ------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ZAMES/ZAMESDialect.h"
#include "ZAMES/ZAMESTypes.h"

using namespace mlir;
using namespace buddy::zames;

#define GET_TYPEDEF_CLASSES
#include "ZAMES/ZAMESTypes.cpp.inc"

void ZAMESDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "ZAMES/ZAMESTypes.cpp.inc"
      >();
}

LogicalResult
TileType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                 StringRef kind, ArrayRef<int64_t> shape, StringRef element) {
  if (kind != "a" && kind != "b" && kind != "c")
    return emitError() << "expected tile kind to be one of a, b, or c";
  if (shape.size() != 2)
    return emitError() << "expected tile shape to have exactly 2 dims";
  for (int64_t dim : shape)
    if (!ShapedType::isDynamic(dim) && dim <= 0)
      return emitError()
             << "expected tile dimensions to be positive or dynamic";
  bool supported = llvm::StringSwitch<bool>(element)
                       .Cases({"i4", "u4", "i8", "u8", "i32"}, true)
                       .Cases({"nvfp4", "mxfp4", "fp8e5m2", "fp8e4m3"}, true)
                       .Cases({"fp16", "bf16", "tf32", "fp32"}, true)
                       .Cases({"fp2pack4", "fp2pack5"}, true)
                       .Default(false);
  if (!supported)
    return emitError() << "unsupported ZAMES element token: " << element;
  return success();
}

Type TileType::parse(AsmParser &parser) {
  StringRef kind;
  SmallVector<int64_t> shape;
  StringRef element;
  if (parser.parseLess() || parser.parseKeyword(&kind) || parser.parseComma() ||
      parser.parseDimensionList(shape, /*allowDynamic=*/true,
                                /*withTrailingX=*/false) ||
      parser.parseComma() || parser.parseKeyword(&element) ||
      parser.parseGreater())
    return {};
  return parser.getChecked<TileType>(parser.getContext(), kind, shape, element);
}

void TileType::print(AsmPrinter &printer) const {
  auto printDim = [&](int64_t dim) {
    if (ShapedType::isDynamic(dim))
      printer << "?";
    else
      printer << dim;
  };
  printer << "<" << getKind() << ", ";
  ArrayRef<int64_t> shape = getShape();
  printDim(shape[0]);
  printer << "x";
  printDim(shape[1]);
  printer << ", " << getElement() << ">";
}
