//===- BOSCAMEBufferNormalization.h - Compatibility buffer forms ----------===//
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
// Compatibility adapter that rewrites supported handwritten bufferized
// generics into ordinary copies, explicit scalar broadcasts, and one
// arithmetic operation per generic. Does not select hardware routes, move
// boundary operations, or change floating evaluation order. Unsupported
// payloads are left unchanged for later diagnostics.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEBUFFERNORMALIZATION_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEBUFFERNORMALIZATION_H

#include "mlir/IR/BuiltinOps.h"

namespace buddy::boscame {

// Normalize supported buffer generics in `module` in place.
void normalizeBOSCAMEBufferInput(mlir::ModuleOp module);

} // namespace buddy::boscame

#endif // BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEBUFFERNORMALIZATION_H
