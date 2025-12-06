//====- IMEDialect.cpp - IME Dialect Implementation -----===//
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

#include "Dialect/IME/IMEDialect.h"
#include "Dialect/IME/IMEOps.h"

using namespace mlir;
using namespace buddy::ime;

#include "IME/IMEDialect.cpp.inc"

#define GET_OP_CLASSES
#include "IME/IME.cpp.inc"

void IMEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "IME/IME.cpp.inc"
      >();
}
