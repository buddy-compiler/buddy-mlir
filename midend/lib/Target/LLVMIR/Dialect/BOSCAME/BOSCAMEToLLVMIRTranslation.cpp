//======- BOSCAMEToLLVMIRTranslation.cpp - Translate BOSCAME to LLVM IR --====//
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
// This file implements a translation between the KXCAME dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "backend/include/llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/IRBuilder.h"

#include "BOSCAME/BOSCAMEDialect.h"
#include "BOSCAME/BOSCAMEOps.h"
#include "Target/LLVMIR/Dialect/BOSCAME/BOSCAMEToLLVMIRTranslation.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace buddy;

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the BOSCAME dialect to LLVM IR.
class BOSCAMEDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "BOSCAME/BOSCAMEConversions.inc"

    return failure();
  }
};
} // end namespace

void buddy::registerBOSCAMEDialectTranslation(DialectRegistry &registry) {
  registry.insert<boscame::BOSCAMEDialect>();
  registry.addExtension(
      +[](MLIRContext *ctx, boscame::BOSCAMEDialect *dialect) {
        dialect->addInterfaces<BOSCAMEDialectLLVMIRTranslationInterface>();
      });
}

void buddy::registerBOSCAMEDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerBOSCAMEDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
