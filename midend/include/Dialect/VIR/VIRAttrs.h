//===-- VIRAttrs.h - Dynamic Vector IR Attribute Declarations ---*- C++ -*-===//
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
// This file declares the attribute system for the Dynamic Vector IR (VIR)
// dialect, including scaling factor attributes.
//
//===----------------------------------------------------------------------===//

#ifndef VIR_VIRATTRS_H
#define VIR_VIRATTRS_H

#include "mlir/IR/Attributes.h"

#define GET_ATTRDEF_CLASSES
#include "VIR/VIRAttrs.h.inc"

#endif // VIR_VIRATTRS_H
