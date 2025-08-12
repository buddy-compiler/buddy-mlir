//===-- VIRTypes.h - Dynamic Vector IR Type Declarations --------*- C++ -*-===//
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
// This file declares the type system for the Dynamic Vector IR (VIR) dialect,
// including dynamic vector types and their associated interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef VIR_VIRTYPES_H
#define VIR_VIRTYPES_H

#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/ADTExtras.h"

#include "VIR/VIRAttrs.h"

#define GET_TYPEDEF_CLASSES
#include "VIR/VIRTypes.h.inc"

#endif // VIR_VIRTYPES_H
