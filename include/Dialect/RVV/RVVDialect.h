//===- RVVDialect.h - MLIR Dialect for RISC-V vector extension ------------===//
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

#ifndef RVV_RVVDIALECT_H
#define RVV_RVVDIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// RVVLMULType
//===----------------------------------------------------------------------===//

/// This RVVLMULType represents the vector register group multiplier (LMUL)
/// setting. When the LMUL greater than 1, the multiplier (M1, M2, M4, M8)
/// represents the number of vector registers that are combined to form a
/// vector register group. The multiplier can also be fractional values (MF8,
/// MF4, MF2), which reduces the number of bits used in a vector register.
class RVVLMULType : public Type {
public:
  using Type::Type;

  static RVVLMULType getMF8(MLIRContext *ctx);
  static RVVLMULType getMF4(MLIRContext *ctx);
  static RVVLMULType getMF2(MLIRContext *ctx);
  static RVVLMULType getM1(MLIRContext *ctx);
  static RVVLMULType getM2(MLIRContext *ctx);
  static RVVLMULType getM4(MLIRContext *ctx);
  static RVVLMULType getM8(MLIRContext *ctx);
};

//===----------------------------------------------------------------------===//
// RVVMaskType
//===----------------------------------------------------------------------===//

/// This RVVMaskType represents the mask length setting. The mask length
/// setting is equal to the ratio of SEW and LMUL (n = SEW/LMUL).
class RVVMaskType : public Type {
public:
  using Type::Type;

  static RVVMaskType getMask1(MLIRContext *ctx);
  static RVVMaskType getMask2(MLIRContext *ctx);
  static RVVMaskType getMask4(MLIRContext *ctx);
  static RVVMaskType getMask8(MLIRContext *ctx);
  static RVVMaskType getMask16(MLIRContext *ctx);
  static RVVMaskType getMask32(MLIRContext *ctx);
  static RVVMaskType getMask64(MLIRContext *ctx);
};

} // end namespace mlir

#include "RVV/RVVDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "RVV/RVVTypes.h.inc"

#define GET_OP_CLASSES
#include "RVV/RVV.h.inc"

#endif // RVV_RVVDIALECT_H
