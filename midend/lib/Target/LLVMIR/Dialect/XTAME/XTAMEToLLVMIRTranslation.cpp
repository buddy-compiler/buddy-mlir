//======- XTAMEToLLVMIRTranslation.cpp - Translate XTAME to LLVM IR ------====//
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
// This file implements a translation between the XTAME dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "backend/include/llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/IRBuilder.h"

#include "XTAME/XTAMEDialect.h"
#include "XTAME/XTAMEOps.h"
#include "Target/LLVMIR/Dialect/XTAME/XTAMEToLLVMIRTranslation.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace buddy;

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the XTAME dialect to LLVM IR.
class XTAMEDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "XTAME/XTAMEConversions.inc"

    return failure();
  }
};
} // end namespace

void buddy::registerXTAMEDialectTranslation(DialectRegistry &registry) {
  registry.insert<xtame::XTAMEDialect>();
  registry.addExtension(+[](MLIRContext *ctx, xtame::XTAMEDialect *dialect) {
    dialect->addInterfaces<XTAMEDialectLLVMIRTranslationInterface>();
  });
}

void buddy::registerXTAMEDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerXTAMEDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
