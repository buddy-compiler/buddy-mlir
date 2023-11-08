//===- Dialects.cpp - C Interface for Dialects ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
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
#include "Dialect/Sche/ScheDialect.h"
#include "Dialect/Sche/ScheOps.h"
#include "Dialect/VectorExp/VectorExpDialect.h"
#include "Dialect/VectorExp/VectorExpOps.h"

#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Bud, bud, buddy::bud::BudDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(DAP, dap, buddy::dap::DAPDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(DIP, dip, buddy::dip::DIPDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Gemmini, gemmini, buddy::gemmini::GemminiDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(RVV, rvv, buddy::rvv::RVVDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Sche, sche, buddy::sche::ScheDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(VectorExp, vector_exp, buddy::vector_exp::VectorExpDialect)