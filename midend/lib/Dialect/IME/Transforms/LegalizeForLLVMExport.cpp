//====- LegalizeForLLVMExport.cpp - IME Lowering Pass -----===//
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

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

#include "Dialect/IME/IMEDialect.h"
#include "Dialect/IME/IMEOps.h"
#include "Dialect/IME/Transform.h"

using namespace mlir;
using namespace buddy::ime;

namespace {

struct IMEVmadotLowering : public ConvertOpToLLVMPattern<VmadotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(VmadotOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVmadotuLowering : public ConvertOpToLLVMPattern<VmadotuOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(VmadotuOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVmadotsuLowering : public ConvertOpToLLVMPattern<VmadotsuOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(VmadotsuOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVmadotusLowering : public ConvertOpToLLVMPattern<VmadotusOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(VmadotusOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVfmadotLowering : public ConvertOpToLLVMPattern<VfmadotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(VfmadotOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct LegalizeIMEForLLVMExport
    : public PassWrapper<LegalizeIMEForLLVMExport, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeIMEForLLVMExport)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext &context = getContext();

    LLVMConversionTarget target(context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<IMEDialect>();

    LLVMTypeConverter typeConverter(&context);

    RewritePatternSet patterns(&context);
    patterns.add<IMEVmadotLowering, IMEVmadotuLowering, IMEVmadotsuLowering,
                 IMEVmadotusLowering, IMEVfmadotLowering>(typeConverter);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::populateIMELegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<IMEVmadotLowering, IMEVmadotuLowering, IMEVmadotsuLowering,
               IMEVmadotusLowering, IMEVfmadotLowering>(converter);
}

void mlir::configureIMELegalizeForExportTarget(LLVMConversionTarget &target) {
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addIllegalDialect<IMEDialect>();
}

std::unique_ptr<Pass> buddy::ime::createLegalizeForLLVMExportPass() {
  return std::make_unique<LegalizeIMEForLLVMExport>();
}
