//===----------- ZAMES.h - C Interface for the ZAMES dialect --------------===//
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
// The types of the ZAMES matrix tile dialect, for the Python bindings.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDYMLIR_C_DIALECT_ZAMES_H
#define BUDDYMLIR_C_DIALECT_ZAMES_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(ZAMES, zames);

//===----------------------------------------------------------------------===//
// TileType
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAZAMESTileType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirZAMESTileTypeGetTypeID(void);

/// `!zames.tile<kind, shape, element>` in the context of `loc`: kind "a", "b"
/// or "c", positive or dynamic (mlirShapedTypeGetDynamicSize()) dims and an
/// element token. Parameters the type verifier rejects emit its error at `loc`
/// and give a null type.
MLIR_CAPI_EXPORTED MlirType mlirZAMESTileTypeGetChecked(MlirLocation loc,
                                                        MlirStringRef kind,
                                                        intptr_t rank,
                                                        const int64_t *shape,
                                                        MlirStringRef element);

MLIR_CAPI_EXPORTED MlirStringRef mlirZAMESTileTypeGetKind(MlirType type);

MLIR_CAPI_EXPORTED intptr_t mlirZAMESTileTypeGetRank(MlirType type);

MLIR_CAPI_EXPORTED int64_t mlirZAMESTileTypeGetDimSize(MlirType type,
                                                       intptr_t dim);

MLIR_CAPI_EXPORTED MlirStringRef mlirZAMESTileTypeGetElement(MlirType type);

//===----------------------------------------------------------------------===//
// SyncType
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAZAMESSyncType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirZAMESSyncTypeGetTypeID(void);

/// `!zames.sync` in `ctx`.
MLIR_CAPI_EXPORTED MlirType mlirZAMESSyncTypeGet(MlirContext ctx);

#ifdef __cplusplus
}
#endif

#endif // BUDDYMLIR_C_DIALECT_ZAMES_H
