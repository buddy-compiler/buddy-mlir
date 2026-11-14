//===- BOSCAMEDialect.h - MLIR Dialect for RISC-V BOSC AME ------===//
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
// Dialect class for `bosc_ame`: SPM/DMA ops, AME tile transfers, and
// intrinsics used by the Buddy Kernel Contract -> plan -> emit pipeline.
// Include BOSCAMEOps.h for op classes; this header only pulls the dialect
// definition generated from BOSCAME.td.
//
//===----------------------------------------------------------------------===//

#ifndef BOSCAME_BOSCAMEDIALECT_H
#define BOSCAME_BOSCAMEDIALECT_H

#include "mlir/IR/Dialect.h"

#include "BOSCAME/BOSCAMEDialect.h.inc"

#endif // BOSCAME_BOSCAMEDIALECT_H
