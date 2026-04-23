//====- XTAMEDialect.cpp - MLIR XTAME dialect implementation --------------===//
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

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "Dialect/XTAME/XTAMEDialect.h"
#include "Dialect/XTAME/XTAMEOps.h"

using namespace mlir;
using namespace buddy::xtame;

#include "XTAME/XTAMEDialect.cpp.inc"

#define GET_OP_CLASSES
#include "XTAME/XTAME.cpp.inc"

void XTAMEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "XTAME/XTAME.cpp.inc"
      >();
}
