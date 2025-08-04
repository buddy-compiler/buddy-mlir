//===------------ Dialects.cpp - C Interface for Dialects -----------------===//
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

#include "buddy-mlir-c/Dialects.h"

#include "Dialect/Bud/BudDialect.h"
#include "Dialect/Bud/BudOps.h"
#include "Dialect/DAP/DAPDialect.h"
#include "Dialect/DAP/DAPOps.h"
#include "Dialect/DIP/DIPDialect.h"
#include "Dialect/DIP/DIPOps.h"
#include "Dialect/Gemmini/GemminiDialect.h"
#include "Dialect/Gemmini/GemminiOps.h"
#include "Dialect/RVV/RVVDialect.h"
#include "Dialect/VectorExp/VectorExpDialect.h"
#include "Dialect/VectorExp/VectorExpOps.h"

#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Bud, bud, buddy::bud::BudDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(DAP, dap, buddy::dap::DAPDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(DIP, dip, buddy::dip::DIPDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Gemmini, gemmini,
                                      buddy::gemmini::GemminiDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(RVV, rvv, buddy::rvv::RVVDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(VectorExp, vector_exp,
                                      buddy::vector_exp::VectorExpDialect)
