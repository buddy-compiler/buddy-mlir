//===- Matmul.h -----------------------------------------------------------===//
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
// Standard buddy runtime matmul ABI:
// MemRef<float, 2> from buddy/Core/Container.h.
// Graph lowering uses func name buddy_matmul_f32 with llvm.emit_c_interface
// -> symbol _mlir_ciface_buddy_matmul_f32. Backends (e.g. TORQ-Tile) link one
// implementation.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_MATMUL_H
#define BUDDY_RUNTIME_MATMUL_H

#include "buddy/Core/Container.h"

#ifdef __cplusplus
extern "C" {
#endif

/// 2D FP32 matrix multiply: result = A * B (row-major MemRef descriptors).
void _mlir_ciface_buddy_matmul_f32(MemRef<float, 2> *result,
                                   MemRef<float, 2> *A, MemRef<float, 2> *B);

#ifdef __cplusplus
}
#endif

#endif // BUDDY_RUNTIME_MATMUL_H
