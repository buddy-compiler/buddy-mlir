//===- TileDialect.cpp - MLIR Tile dialect implementation ----------===//
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
// This file implements the Tile dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Tile/TileDialect.h"
#include "Tile/TileOps.h"

using namespace mlir;
using namespace buddy::tile;

#include "Tile/TileDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Tile/Tile.cpp.inc"

void TileDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Tile/Tile.cpp.inc"
      >();
}

