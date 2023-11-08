/*===-- buddy-mlir-c/Dialects.h - Dialect functions  --------------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef BUDDYMLIR_C_DIALECTS_H
#define BUDDYMLIR_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Bud, bud);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(DAP, dap);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(dip, dip);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Gemmini, gemmini);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(RVV, rvv);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Sche, sche);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(VectorExp, vector_exp);

#ifdef __cplusplus
}
#endif

#endif  // BUDDYMLIR_C_DIALECTS_H