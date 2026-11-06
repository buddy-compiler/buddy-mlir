//===- BOSCAMEAttrs.h - BOSCAME dialect attribute declarations ------------===//
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
// This file declares the BOSCAME dialect attributes.  Today that is the SPM
// storage space attribute `#bosc_ame.spm<bank>` used as a memref memory space.
//
//===----------------------------------------------------------------------===//

#ifndef BOSCAME_BOSCAMEATTRS_H
#define BOSCAME_BOSCAMEATTRS_H

#include "mlir/IR/Attributes.h"

#define GET_ATTRDEF_CLASSES
#include "BOSCAME/BOSCAMEAttrs.h.inc"

#endif // BOSCAME_BOSCAMEATTRS_H
