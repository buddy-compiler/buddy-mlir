//===- RVProfOps.cpp - RVProf Dialect Operations ----------------*- C++ -*-===//
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
// This file defines the operations for the RVProf dialect.
//
//===----------------------------------------------------------------------===//

#include "RVProf/RVProfOps.h"
#include "RVProf/RVProfDialect.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace buddy::rvprof;

#define GET_OP_CLASSES
#include "RVProf/RVProfOps.cpp.inc"
