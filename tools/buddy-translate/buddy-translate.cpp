//===- buddy-translate.cpp --------------------------------------*- C++ -*-===//
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
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

#include "RVV/RVVDialect.h"

using namespace buddy;
using namespace mlir;

namespace buddy {
void registerBuddyToLLVMIRTranslation();
}

int main(int argc, char **argv) {
  mlir::registerAllTranslations();
  buddy::registerBuddyToLLVMIRTranslation();

  return failed(mlir::mlirTranslateMain(argc, argv, "Buddy Translation Tool"));
}
