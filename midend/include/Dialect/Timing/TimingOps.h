//===- TimingOps.h - Timing Dialect Ops -------------------------------*- C++
//-*-===//
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
// This is the header file for operations in Timing dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TIMING_TIMINGOPS_H
#define TIMING_TIMINGOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Timing/TimingOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Timing/TimingOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "Timing/TimingOps.h.inc"

#endif // TIMING_TIMINGOPS_H
