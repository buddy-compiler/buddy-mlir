//===----------- Registration.h - Register all dialects and passes --------===//
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

#ifndef BUDDYMLIR_C_REGISTRATION_H
#define BUDDYMLIR_C_REGISTRATION_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Registers all dialects with a context.
 * This is needed before creating IR for these Dialects.
 */
MLIR_CAPI_EXPORTED void buddyMlirRegisterAllDialects(MlirContext context);

/** Registers all passes for symbolic access with the global registry. */
MLIR_CAPI_EXPORTED void buddyMlirRegisterAllPasses();

#ifdef __cplusplus
}
#endif

#endif // BUDDYMLIR_C_REGISTRATION_H
