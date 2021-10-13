//===- BudDialect.cpp - Bud Dialect Definition-------------------*- C++ -*-===//
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
// This file defines bud dialect.
//
//===----------------------------------------------------------------------===//

#include "Bud/BudDialect.h"
#include "Bud/BudOps.h"

using namespace mlir;
using namespace buddy::bud;

#include "Bud/BudOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Bud dialect.
//===----------------------------------------------------------------------===//

void BudDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Bud/BudOps.cpp.inc"
      >();
}
