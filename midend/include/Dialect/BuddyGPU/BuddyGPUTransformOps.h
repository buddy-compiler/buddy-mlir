//====- BuddyGPUTransformOps.h -------------------------------------------===//
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
//===---------------------------------------------------------------------===//
//
// This file defines transform ops for code generation.
//
//===---------------------------------------------------------------------===//

// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef BUDDYGPU_TRANSFORM_OPS_H
#define BUDDYGPU_TRANSFORM_OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"


namespace mlir {
class DialectRegistry;

namespace func {
class FuncOp;
}

namespace scf {
class ForallOp;
class IfOp;
class ForOp;
} // namespace scf

namespace vector {
class VectorDialect;
class WarpExecuteOnLane0Op;
} // namespace vector

}

namespace mlir{
namespace buddy{
void registerBuddyTransformOps(mlir::DialectRegistry &registry);

namespace buddygpu::transform_dialect {

class BuddyGPUTransformExtensions
    : public mlir::transform::TransformDialectExtension<
          BuddyGPUTransformExtensions> {
public:
  BuddyGPUTransformExtensions();
};
} // namespace buddygpu::transform_dialect
} // namespace buddy
} // namespace mlir

#define GET_OP_CLASSES
#include "BuddyGPU/BuddyGPUTransformOps.h.inc"

#endif // BUDDYGPU_TRANSFORM_OPS_H
