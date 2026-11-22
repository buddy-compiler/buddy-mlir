//====- LegalizeForLLVMExport.cpp - Prepare AME for LLVM translation ------===//
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

#include "Dialect/AME/AMEDialect.h"
#include "Dialect/AME/Transform.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
using namespace buddy::ame;

namespace {

/// Convert instruction operations through their intrinsic bridges, preserving
/// SSA types and the single LLVM struct result of zip/unzip operations.
class AMEOperationLowering : public ConvertToLLVMPattern {
public:
  AMEOperationLowering(const LLVMTypeConverter &converter, StringRef name)
      : ConvertToLLVMPattern(name, &converter.getContext(), converter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef mnemonic = op->getName().getStringRef();
    if (!mnemonic.consume_front("ame.") || mnemonic.starts_with("intr."))
      return failure();

    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                resultTypes)))
      return rewriter.notifyMatchFailure(op, "cannot convert AME result types");

    // Ownership mnemonics already include "ame." in the LLVM intrinsic name.
    std::string bridge = "ame.intr.";
    if (mnemonic == "acquire" || mnemonic == "release")
      bridge += "ame.";
    bridge += mnemonic.str();
    OperationState state(op->getLoc(), bridge);
    state.addOperands(operands);
    state.addTypes(resultTypes);
    state.addAttributes(op->getAttrs());
    Operation *intrinsic = rewriter.create(state);
    rewriter.replaceOp(op, intrinsic->getResults());
    return success();
  }
};

/// Intrinsic bridges can also be supplied directly by an LLVM-level frontend.
class AMEIntrinsicLowering : public ConvertToLLVMPattern {
public:
  AMEIntrinsicLowering(const LLVMTypeConverter &converter, StringRef name)
      : ConvertToLLVMPattern(name, &converter.getContext(), converter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef mnemonic = op->getName().getStringRef();
    if (!mnemonic.consume_front("ame.intr."))
      return failure();

    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                resultTypes)))
      return rewriter.notifyMatchFailure(op, "cannot convert AME result types");

    auto intrinsic = LLVM::CallIntrinsicOp::create(
        rewriter, op->getLoc(), resultTypes,
        rewriter.getStringAttr(("llvm.riscv.ztt." + mnemonic).str()), operands,
        LLVM::FastmathFlagsAttr{});
    rewriter.replaceOp(op, intrinsic->getResults());
    return success();
  }
};

struct LegalizeAMEForLLVMExport
    : public PassWrapper<LegalizeAMEForLLVMExport, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeAMEForLLVMExport)

  StringRef getArgument() const final { return "lower-ame"; }
  StringRef getDescription() const final {
    return "Lower AME ZTT v0.6 operations to LLVM intrinsic calls.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMEDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    LLVMConversionTarget target(getContext());
    configureAMELegalizeForExportTarget(target);
    // Other dialects are lowered by their own passes. Every AME operation
    // must be converted, including when this pass is used in a mixed module.
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    LLVMTypeConverter converter(&getContext());
    RewritePatternSet patterns(&getContext());
    populateAMELegalizeForLLVMExportPatterns(converter, patterns);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::populateAMELegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  context->getOrLoadDialect<AMEDialect>();
  for (RegisteredOperationName name : context->getRegisteredOperationsByDialect(
           AMEDialect::getDialectNamespace())) {
    if (name.getStringRef().starts_with("ame.intr."))
      patterns.add<AMEIntrinsicLowering>(converter, name.getStringRef());
    else
      patterns.add<AMEOperationLowering>(converter, name.getStringRef());
  }
}

void mlir::configureAMELegalizeForExportTarget(LLVMConversionTarget &target) {
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addIllegalDialect<AMEDialect>();
}

std::unique_ptr<Pass> buddy::ame::createLegalizeForLLVMExportPass() {
  return std::make_unique<LegalizeAMEForLLVMExport>();
}

namespace mlir::buddy {
void registerLowerAMEPass() { PassRegistration<LegalizeAMEForLLVMExport>(); }
} // namespace mlir::buddy
