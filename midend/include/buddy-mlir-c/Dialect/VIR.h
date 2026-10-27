//===------------- VIR.h - C Interface for the VIR dialect ----------------===//
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
// The types and attributes of the Dynamic Vector IR (VIR) dialect, for the
// Python bindings.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDYMLIR_C_DIALECT_VIR_H
#define BUDDYMLIR_C_DIALECT_VIR_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(VIR, vir);

//===----------------------------------------------------------------------===//
// ScalingFactorAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool
mlirAttributeIsAVIRScalingFactorAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID mlirVIRScalingFactorAttrGetTypeID(void);

/// A scaling factor of ratio "m" (multiplier) or "f" (fraction) and value 2,
/// 4, ..., 64.
MLIR_CAPI_EXPORTED MlirAttribute mlirVIRScalingFactorAttrGet(
    MlirContext ctx, MlirStringRef ratio, int64_t value);

MLIR_CAPI_EXPORTED MlirStringRef
mlirVIRScalingFactorAttrGetRatio(MlirAttribute attr);

MLIR_CAPI_EXPORTED int64_t mlirVIRScalingFactorAttrGetValue(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// DynamicVectorType
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAVIRDynamicVectorType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirVIRDynamicVectorTypeGetTypeID(void);

/// A dynamic vector of `shape` (static dims, then the dynamic one,
/// mlirShapedTypeGetDynamicSize()), `elementType` and an optional scaling
/// factor (a null attribute for none) in the context of `elementType`.
MLIR_CAPI_EXPORTED MlirType
mlirVIRDynamicVectorTypeGet(intptr_t rank, const int64_t *shape,
                            MlirType elementType, MlirAttribute scalingFactor);

MLIR_CAPI_EXPORTED intptr_t mlirVIRDynamicVectorTypeGetRank(MlirType type);

MLIR_CAPI_EXPORTED int64_t mlirVIRDynamicVectorTypeGetDimSize(MlirType type,
                                                              intptr_t dim);

MLIR_CAPI_EXPORTED MlirType
mlirVIRDynamicVectorTypeGetElementType(MlirType type);

/// The scaling factor, a null attribute for none.
MLIR_CAPI_EXPORTED MlirAttribute
mlirVIRDynamicVectorTypeGetScalingFactor(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // BUDDYMLIR_C_DIALECT_VIR_H
