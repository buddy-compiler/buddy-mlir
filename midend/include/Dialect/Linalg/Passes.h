//===- Passes.h -----------------------------------------------------------===//
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

#ifndef DIALECT_LINALG_PASSES_H
#define DIALECT_LINALG_PASSES_H

// Include the constructor of passes in Linalg Dialect
#include "Linalg/Transforms/LinalgPromotion.h"

namespace mlir {
// Generate the definition of Linalg Passes
#define GEN_PASS_DECL
#include "Linalg/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Linalg/Passes.h.inc"

} // namespace mlir

#endif
