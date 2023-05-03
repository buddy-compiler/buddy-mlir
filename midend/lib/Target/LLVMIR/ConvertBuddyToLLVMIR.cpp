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

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include "Target/LLVMIR/Dialect/Gemmini/GemminiToLLVMIRTranslation.h"
#include "Target/LLVMIR/Dialect/RVV/RVVToLLVMIRTranslation.h"

using namespace buddy;
using namespace mlir;

namespace buddy {
void registerBuddyToLLVMIRTranslation() {
  TranslateFromMLIRRegistration registration(
      "buddy-to-llvmir", "translate MLIR from buddy toolchain to LLVM IR",
      [](Operation *op, raw_ostream &output) {
        llvm::LLVMContext llvmContext;
        auto llvmModule = translateModuleToLLVMIR(op, llvmContext);
        if (!llvmModule)
          return failure();

        llvmModule->print(output, nullptr);
        return success();
      },
      [](DialectRegistry &registry) {
        // Register translation in upstream MLIR.
        registry.insert<DLTIDialect, func::FuncDialect>();
        registerAllToLLVMIRTranslations(registry);
        // Register translation in buddy project.
        registerRVVDialectTranslation(registry);
        registerGemminiDialectTranslation(registry);
      });
}
} // namespace buddy
