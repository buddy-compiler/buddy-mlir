//===-- MagiaDialect.cpp - MAGIA dialect definition -----------------------===//
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
// This file implements the MAGIA dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "Magia/MagiaDialect.h"
#include "Magia/MagiaOps.h"

using namespace mlir;
using namespace buddy::magia;

#include "Magia/MagiaDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Magia/Magia.cpp.inc"

void MagiaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Magia/Magia.cpp.inc"
      >();
}
