//===- VectorExpOps.cpp - Vector Experiment Dialect Ops -------------------===//
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
// This file defines operations in the vector experiment dialect.
//
//===----------------------------------------------------------------------===//

#include "VectorExp/VectorExpOps.h"
#include "VectorExp/DynamicVLOpInterface.h"
#include "VectorExp/VectorExpDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "VectorExp/VectorExpOps.cpp.inc"

mlir::TypedValue<mlir::IntegerType> buddy::vector_exp::LoadOp::getVLValue() {
  auto operands = getODSOperands(1);
  return operands.empty() ? mlir::TypedValue<mlir::IntegerType>{}
                          : llvm::cast<mlir::TypedValue<mlir::IntegerType>>(
                                *operands.begin());
}

mlir::ParseResult
buddy::vector_exp::LoadOp::parseVL(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> vlOperands;
  llvm::SMLoc vlOperandsLoc;
  llvm::SmallVector<mlir::Type, 1> vlTypes;

  if (mlir::succeeded(parser.parseOptionalLParen())) {
    if (parser.parseKeyword("vl"))
      return mlir::failure();
    if (parser.parseEqual())
      return mlir::failure();
    {
      vlOperandsLoc = parser.getCurrentLocation();
      mlir::OpAsmParser::UnresolvedOperand operand;
      mlir::OptionalParseResult parseResult =
          parser.parseOptionalOperand(operand);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return mlir::failure();
        vlOperands.push_back(operand);
      }
    }
    if (parser.parseColon())
      return mlir::failure();
    {
      mlir::Type optionalType;
      mlir::OptionalParseResult parseResult =
          parser.parseOptionalType(optionalType);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return mlir::failure();
        vlTypes.push_back(optionalType);
      }
    }
    if (parser.parseRParen())
      return mlir::failure();
  }
  if (parser.resolveOperands(vlOperands, vlTypes, vlOperandsLoc,
                             result.operands))
    return mlir::failure();
  return mlir::success();
}

void buddy::vector_exp::LoadOp::printVL(mlir::OpAsmPrinter &_odsPrinter) {
  if (getVLValue()) {
    _odsPrinter << ' ' << "(";
    _odsPrinter << "vl";
    _odsPrinter << ' ' << "=";
    _odsPrinter << ' ';
    if (mlir::Value value = getVLValue())
      _odsPrinter << value;
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << (getVLValue()
                        ? llvm::ArrayRef<mlir::Type>(getVLValue().getType())
                        : llvm::ArrayRef<mlir::Type>());
    _odsPrinter << ")";
  }
}
