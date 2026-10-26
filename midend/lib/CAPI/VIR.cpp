//===------------- VIR.cpp - C Interface for the VIR dialect --------------===//
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

#include "buddy-mlir-c/Dialect/VIR.h"

#include "Dialect/VIR/VIRAttrs.h"
#include "Dialect/VIR/VIRDialect.h"
#include "Dialect/VIR/VIRTypes.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace mlir;
using namespace buddy::vir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(VIR, vir, buddy::vir::VIRDialect)

//===----------------------------------------------------------------------===//
// ScalingFactorAttr
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAVIRScalingFactorAttr(MlirAttribute attr) {
  return isa<ScalingFactorAttr>(unwrap(attr));
}

MlirTypeID mlirVIRScalingFactorAttrGetTypeID(void) {
  return wrap(ScalingFactorAttr::getTypeID());
}

MlirAttribute mlirVIRScalingFactorAttrGet(MlirContext ctx, MlirStringRef ratio,
                                          int64_t value) {
  return wrap(ScalingFactorAttr::get(unwrap(ctx), unwrap(ratio), value));
}

MlirStringRef mlirVIRScalingFactorAttrGetRatio(MlirAttribute attr) {
  return wrap(cast<ScalingFactorAttr>(unwrap(attr)).getRatio());
}

int64_t mlirVIRScalingFactorAttrGetValue(MlirAttribute attr) {
  return cast<ScalingFactorAttr>(unwrap(attr)).getValue();
}

//===----------------------------------------------------------------------===//
// DynamicVectorType
//===----------------------------------------------------------------------===//

bool mlirTypeIsAVIRDynamicVectorType(MlirType type) {
  return isa<DynamicVectorType>(unwrap(type));
}

MlirTypeID mlirVIRDynamicVectorTypeGetTypeID(void) {
  return wrap(DynamicVectorType::getTypeID());
}

MlirType mlirVIRDynamicVectorTypeGet(intptr_t rank, const int64_t *shape,
                                     MlirType elementType,
                                     MlirAttribute scalingFactor) {
  return wrap(DynamicVectorType::get(
      llvm::ArrayRef(shape, static_cast<size_t>(rank)), unwrap(elementType),
      mlirAttributeIsNull(scalingFactor)
          ? ScalingFactorAttr()
          : cast<ScalingFactorAttr>(unwrap(scalingFactor))));
}

intptr_t mlirVIRDynamicVectorTypeGetRank(MlirType type) {
  return cast<DynamicVectorType>(unwrap(type)).getShape().size();
}

int64_t mlirVIRDynamicVectorTypeGetDimSize(MlirType type, intptr_t dim) {
  return cast<DynamicVectorType>(unwrap(type)).getShape()[dim];
}

MlirType mlirVIRDynamicVectorTypeGetElementType(MlirType type) {
  return wrap(cast<DynamicVectorType>(unwrap(type)).getElementType());
}

MlirAttribute mlirVIRDynamicVectorTypeGetScalingFactor(MlirType type) {
  return wrap(cast<DynamicVectorType>(unwrap(type)).getScalingFactor());
}
