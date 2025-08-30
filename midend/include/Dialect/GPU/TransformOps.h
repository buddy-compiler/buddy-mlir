//===- TransformOps.h -----------------------------------------------------===//
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
// The process in this file references the IREE project,
// which is hereby acknowledged.
// For the license of the IREE project
// please see: https://github.com/iree-org/iree/blob/main/LICENSE
//
//===----------------------------------------------------------------------===//
//
// This file defines GPU transform ops for code generation.
//
//===----------------------------------------------------------------------===//

#ifndef TRANSFORM_OPS_H
#define TRANSFORM_OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

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

} // namespace mlir

namespace mlir {
namespace buddy {
void registerBuddyGPUTransformOps(mlir::DialectRegistry &registry);

namespace gpu {

class TransformExtensions
    : public mlir::transform::TransformDialectExtension<TransformExtensions> {
public:
  TransformExtensions();
};
} // namespace gpu
} // namespace buddy
} // namespace mlir

#define GET_OP_CLASSES
#include "GPU/TransformOps.h.inc"

#endif // TRANSFORM_OPS_H
