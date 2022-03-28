//===- ConvertBuddyToLLVMIR.cpp -------------------------------------------===//
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
// This file implements a translation between the MLIR LLVM dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include "Target/LLVMIR/Dialect/RVV/RVVToLLVMIRTranslation.h"

using namespace buddy;
using namespace mlir;

namespace buddy {
void registerBuddyToLLVMIRTranslation() {
  TranslateFromMLIRRegistration registration(
      "buddy-to-llvmir",
      [](ModuleOp module, raw_ostream &output) {
        llvm::LLVMContext llvmContext;
        auto llvmModule = translateModuleToLLVMIR(module, llvmContext);
        if (!llvmModule)
          return failure();

        llvmModule->print(output, nullptr);
        return success();
      },
      [](DialectRegistry &registry) {
        // Register translation in upstream MLIR.
        registerArmNeonDialectTranslation(registry);
        registerAMXDialectTranslation(registry);
        registerArmSVEDialectTranslation(registry);
        registerLLVMDialectTranslation(registry);
        registerNVVMDialectTranslation(registry);
        registerOpenACCDialectTranslation(registry);
        registerOpenMPDialectTranslation(registry);
        registerROCDLDialectTranslation(registry);
        registerX86VectorDialectTranslation(registry);
        // Register translation in buddy project.
        registerRVVDialectTranslation(registry);
      });
}
} // namespace buddy
