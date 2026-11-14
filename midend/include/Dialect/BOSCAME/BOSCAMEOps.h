//===- BOSCAMEOps.h - MLIR Dialect for RISC-V BOSC AME extension ----------===//
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
// C++ op class declarations for the `bosc_ame` dialect (generated from
// BOSCAME.td via TableGen). Include this header when constructing, matching,
// or rewriting BOSCAME ops. Dialect registration lives in BOSCAMEDialect.h.
//
// Attribute and type classes are included first because op declarations
// reference `#bosc_ame.spm` / DMA token types.
//
//===----------------------------------------------------------------------===//

#ifndef BOSCAME_BOSCAMEOPS_H
#define BOSCAME_BOSCAMEOPS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Op declarations reference SPM memory-space attrs and the DMA token type.
#include "BOSCAME/BOSCAMEAttrs.h"
#include "BOSCAME/BOSCAMETypes.h"

#define GET_OP_CLASSES
#include "BOSCAME/BOSCAME.h.inc"

#endif // BOSCAME_BOSCAMEOPS_H
