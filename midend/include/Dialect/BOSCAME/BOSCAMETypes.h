//===- BOSCAMETypes.h - BOSCAME dialect type declarations -----------------===//
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
// This file declares the BOSCAME dialect types.  Today that is the DMA request
// handle `!bosc_ame.dma_token<channel>`.
//
//===----------------------------------------------------------------------===//

#ifndef BOSCAME_BOSCAMETYPES_H
#define BOSCAME_BOSCAMETYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "BOSCAME/BOSCAMETypes.h.inc"

#endif // BOSCAME_BOSCAMETYPES_H
