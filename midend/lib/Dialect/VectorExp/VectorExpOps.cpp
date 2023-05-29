//===- VectorExpOps.cpp - Vector Experiment Dialect Ops -------------------===//
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
// This file defines operations in the vector experiment dialect.
//
//===----------------------------------------------------------------------===//

#include "VectorExp/VectorExpOps.h"
#include "VectorExp/VectorExpDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#define GET_OP_CLASSES
#include "VectorExp/VectorExpOps.cpp.inc"
