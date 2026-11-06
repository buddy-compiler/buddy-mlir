//===- BOSCAMEOrderedMatmul.h - Ordered compatibility recipe --------------===//
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
// Compatibility emitter for a verified scalar matmul recipe in serial M/N/K
// order. Used when the planner selects ordered CPU/scalar expansion instead
// of AME tiles. Memory spaces are already materialized from operand bindings.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEORDEREDMATMUL_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEORDEREDMATMUL_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace buddy::boscame {

// Mechanically expand `matmul` in serial M/N/K order, cloning the original
// scalar payload (casts and floating flags). SPM accesses use one-element
// vectors rather than prohibited scalar loads. The caller moves completion
// facts and erases the original operation.
mlir::Operation *emitOrderedMatmul(mlir::linalg::MatmulOp matmul);

} // namespace buddy::boscame

#endif // BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEORDEREDMATMUL_H
