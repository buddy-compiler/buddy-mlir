//===----------- XSAIAME.cpp - C Interface for the XSAIAME dialect --------===//
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

#include "buddy-mlir-c/Dialect/XSAIAME.h"

#include "Dialect/XSAIAME/XSAIAMEDialect.h"
#include "Dialect/XSAIAME/XSAIAMETypes.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace mlir;
using namespace buddy::xsaiame;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(XSAIAME, xsai_ame,
                                      buddy::xsaiame::XSAIAMEDialect)

//===----------------------------------------------------------------------===//
// TileType
//===----------------------------------------------------------------------===//

bool mlirTypeIsAXSAIAMETileType(MlirType type) {
  return isa<TileType>(unwrap(type));
}

MlirTypeID mlirXSAIAMETileTypeGetTypeID(void) {
  return wrap(TileType::getTypeID());
}

MlirType mlirXSAIAMETileTypeGetChecked(MlirLocation loc, MlirStringRef kind,
                                       intptr_t rank, const int64_t *shape,
                                       MlirStringRef element) {
  Location location = unwrap(loc);
  return wrap(TileType::getChecked(
      location, location.getContext(), unwrap(kind),
      llvm::ArrayRef(shape, static_cast<size_t>(rank)), unwrap(element)));
}

MlirStringRef mlirXSAIAMETileTypeGetKind(MlirType type) {
  return wrap(cast<TileType>(unwrap(type)).getKind());
}

intptr_t mlirXSAIAMETileTypeGetRank(MlirType type) {
  return cast<TileType>(unwrap(type)).getShape().size();
}

int64_t mlirXSAIAMETileTypeGetDimSize(MlirType type, intptr_t dim) {
  return cast<TileType>(unwrap(type)).getShape()[dim];
}

MlirStringRef mlirXSAIAMETileTypeGetElement(MlirType type) {
  return wrap(cast<TileType>(unwrap(type)).getElement());
}

//===----------------------------------------------------------------------===//
// SyncType
//===----------------------------------------------------------------------===//

bool mlirTypeIsAXSAIAMESyncType(MlirType type) {
  return isa<SyncType>(unwrap(type));
}

MlirTypeID mlirXSAIAMESyncTypeGetTypeID(void) {
  return wrap(SyncType::getTypeID());
}

MlirType mlirXSAIAMESyncTypeGet(MlirContext ctx) {
  return wrap(SyncType::get(unwrap(ctx)));
}
