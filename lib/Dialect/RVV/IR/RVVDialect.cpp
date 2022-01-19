//===- RVVDialect.cpp - MLIR RVV dialect implementation -------------------===//
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
// This file implements the RVV dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

#include "RVV/RVVDialect.h"

using namespace mlir;
using namespace buddy;

#include "RVV/RVVDialect.cpp.inc"

#define GET_OP_CLASSES
#include "RVV/RVV.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "RVV/RVVTypes.cpp.inc"

void rvv::RVVDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "RVV/RVV.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "RVV/RVVTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// RVVLMULType
//===----------------------------------------------------------------------===//

RVVLMULType RVVLMULType::getMF8(MLIRContext *ctx) {
  return rvv::MF8Type::get(ctx);
}

RVVLMULType RVVLMULType::getMF4(MLIRContext *ctx) {
  return rvv::MF4Type::get(ctx);
}

RVVLMULType RVVLMULType::getMF2(MLIRContext *ctx) {
  return rvv::MF2Type::get(ctx);
}

RVVLMULType RVVLMULType::getM1(MLIRContext *ctx) {
  return rvv::M1Type::get(ctx);
}

RVVLMULType RVVLMULType::getM2(MLIRContext *ctx) {
  return rvv::M2Type::get(ctx);
}

RVVLMULType RVVLMULType::getM4(MLIRContext *ctx) {
  return rvv::M4Type::get(ctx);
}

RVVLMULType RVVLMULType::getM8(MLIRContext *ctx) {
  return rvv::M8Type::get(ctx);
}

//===----------------------------------------------------------------------===//
// RVVMaskType
//===----------------------------------------------------------------------===//

RVVMaskType RVVMaskType::getMask1(MLIRContext *ctx) {
  return rvv::Mask1Type::get(ctx);
}

RVVMaskType RVVMaskType::getMask2(MLIRContext *ctx) {
  return rvv::Mask2Type::get(ctx);
}

RVVMaskType RVVMaskType::getMask4(MLIRContext *ctx) {
  return rvv::Mask4Type::get(ctx);
}

RVVMaskType RVVMaskType::getMask8(MLIRContext *ctx) {
  return rvv::Mask8Type::get(ctx);
}

RVVMaskType RVVMaskType::getMask16(MLIRContext *ctx) {
  return rvv::Mask16Type::get(ctx);
}

RVVMaskType RVVMaskType::getMask32(MLIRContext *ctx) {
  return rvv::Mask32Type::get(ctx);
}

RVVMaskType RVVMaskType::getMask64(MLIRContext *ctx) {
  return rvv::Mask64Type::get(ctx);
}
