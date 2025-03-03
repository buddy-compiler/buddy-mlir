//===----------------- Dialects.h - CAPI for dialects ---------------------===//
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

#ifndef BUDDYMLIR_C_DIALECTS_H
#define BUDDYMLIR_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Bud, bud);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(DAP, dap);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(DIP, dip);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Gemmini, gemmini);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(RVV, rvv);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Sche, sche);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(VectorExp, vector_exp);

#ifdef __cplusplus
}
#endif

#endif // BUDDYMLIR_C_DIALECTS_H
