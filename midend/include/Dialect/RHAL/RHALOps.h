//===- RHALOps.h - RHAL Dialect Operations Header -------------------------===//
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

#ifndef BUDDY_MIDEND_RHAL_RHALOPS_H
#define BUDDY_MIDEND_RHAL_RHALOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "RHAL/RHALDialect.h"

// Generated op declarations
#define GET_OP_CLASSES
#include "RHAL/RHALOps.h.inc"

#endif // BUDDY_MIDEND_RHAL_RHALOPS_H
