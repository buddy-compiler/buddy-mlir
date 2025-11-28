//====- LowerIMEPass.cpp - IME Dialect Lowering Pass  -----===//
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
// This file defines IME dialect lowering pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

#include "Dialect/IME/IMEDialect.h"
#include "Dialect/IME/IMEOps.h"
#include "Dialect/IME/Transform.h"

using namespace mlir;
using namespace buddy::ime;

namespace {
class LowerIMEToLLVMPass
    : public PassWrapper<LowerIMEToLLVMPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerIMEToLLVMPass)
  StringRef getArgument() const final { return "lower-ime"; }
  StringRef getDescription() const final {
    return "IME dialect lowering pass.";
  }
  LowerIMEToLLVMPass() = default;
  LowerIMEToLLVMPass(const LowerIMEToLLVMPass &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<IMEDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void LowerIMEToLLVMPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  LLVMTypeConverter converter(context);
  RewritePatternSet patterns(context);
  LLVMConversionTarget target(*context);

  configureIMELegalizeForExportTarget(target);
  populateIMELegalizeForLLVMExportPatterns(converter, patterns);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerIMEPass() { PassRegistration<LowerIMEToLLVMPass>(); }
} // namespace buddy
} // namespace mlir
