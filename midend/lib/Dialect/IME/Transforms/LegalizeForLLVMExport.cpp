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
    // ime.vmadot %c, %a, %b
    Value cStruct = adaptor.getOperands()[0];
    Value cPtr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), cStruct, ArrayRef<int64_t>{1}); // Index 1 is alignedPtr

    Value aStruct = adaptor.getOperands()[1];
    Value aPtr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), aStruct, ArrayRef<int64_t>{1});

    Value bStruct = adaptor.getOperands()[2];
    Value bPtr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), bStruct, ArrayRef<int64_t>{1});

    
    StringRef asmString = 
        "vsetvli t0, zero, e32, m2\n\t"
        "vxor.vv v28, v28, v28\n\t" 
        "vsetvli t0, zero, e8, m1\n\t"
        "vle8.v v0, ($1)\n\t" 
        "vle8.v v1, ($2)\n\t"
        "vmadot v28, v0, v1\n\t"
       "vsetvli t0, zero, e32, m2\n\t"
        "vse32.v v28, ($0)";

   StringRef constraints = "r,r,r,~{v0},~{v1},~{v28},~{v29},~{t0},~{memory}";

     rewriter.replaceOpWithNewOp<LLVM::InlineAsmOp>(
        op,
        TypeRange(),                            
        ValueRange{cPtr, aPtr, bPtr},           
        asmString,                              
        constraints,                            
        true,                                   
        false,                                  
        LLVM::AsmDialectAttr::get(getContext(), LLVM::AsmDialect::AD_ATT), 
        ArrayAttr()                             
    );

    return success();
  }
};

struct IMEVmadotuLowering : public ConvertOpToLLVMPattern<VmadotuOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(VmadotuOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value vdStruct = adaptor.getOperands()[0];
    Value vdPtr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), vdStruct, ArrayRef<int64_t>{1});

    Value vs1Struct = adaptor.getOperands()[1];
    Value vs1Ptr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), vs1Struct, ArrayRef<int64_t>{1});

    Value vs2Struct = adaptor.getOperands()[2];
    Value vs2Ptr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), vs2Struct, ArrayRef<int64_t>{1});

        StringRef asmString = 
        "vsetvli t0, zero, e32, m2\n\t"
        "vxor.vv v28, v28, v28\n\t"
        "vsetvli t0, zero, e8, m1\n\t"
        "vle8.v v0, ($1)\n\t"
        "vle8.v v1, ($2)\n\t"
        "vmadotu v28, v0, v1\n\t"
        "vsetvli t0, zero, e32, m2\n\t"
        "vse32.v v28, ($0)";

    StringRef constraints = "r,r,r,~{v0},~{v1},~{v28},~{v29},~{t0},~{memory}";

    rewriter.replaceOpWithNewOp<LLVM::InlineAsmOp>(
        op,
        TypeRange(),
        ValueRange{vdPtr, vs1Ptr, vs2Ptr},
        asmString,
        constraints,
        true,
        false,
        LLVM::AsmDialectAttr::get(getContext(), LLVM::AsmDialect::AD_ATT),
        ArrayAttr()
    );

    return success();
  }
};

struct IMEVmadotsuLowering : public ConvertOpToLLVMPattern<VmadotsuOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(VmadotsuOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value vdStruct = adaptor.getOperands()[0];
    Value vdPtr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), vdStruct, ArrayRef<int64_t>{1});

    Value vs1Struct = adaptor.getOperands()[1];
    Value vs1Ptr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), vs1Struct, ArrayRef<int64_t>{1});

    Value vs2Struct = adaptor.getOperands()[2];
    Value vs2Ptr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), vs2Struct, ArrayRef<int64_t>{1});

    StringRef asmString = 
        "vsetvli t0, zero, e32, m2\n\t"
        "vxor.vv v28, v28, v28\n\t"
        "vsetvli t0, zero, e8, m1\n\t"
        "vle8.v v0, ($1)\n\t"
        "vle8.v v1, ($2)\n\t"
        "vmadotsu v28, v0, v1\n\t"
        "vsetvli t0, zero, e32, m2\n\t"
        "vse32.v v28, ($0)";

    StringRef constraints = "r,r,r,~{v0},~{v1},~{v28},~{v29},~{t0},~{memory}";

    rewriter.replaceOpWithNewOp<LLVM::InlineAsmOp>(
        op,
        TypeRange(),
        ValueRange{vdPtr, vs1Ptr, vs2Ptr},
        asmString,
        constraints,
        true,
        false,
        LLVM::AsmDialectAttr::get(getContext(), LLVM::AsmDialect::AD_ATT),
        ArrayAttr()
    );

    return success();
  }
};

struct IMEVmadotusLowering : public ConvertOpToLLVMPattern<VmadotusOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(VmadotusOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value vdStruct = adaptor.getOperands()[0];
    Value vdPtr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), vdStruct, ArrayRef<int64_t>{1});

    Value vs1Struct = adaptor.getOperands()[1];
    Value vs1Ptr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), vs1Struct, ArrayRef<int64_t>{1});

    Value vs2Struct = adaptor.getOperands()[2];
    Value vs2Ptr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), vs2Struct, ArrayRef<int64_t>{1});

    StringRef asmString = 
        "vsetvli t0, zero, e32, m2\n\t"
        "vxor.vv v28, v28, v28\n\t"
        "vsetvli t0, zero, e8, m1\n\t"
        "vle8.v v0, ($1)\n\t"
        "vle8.v v1, ($2)\n\t"
        "vmadotus v28, v0, v1\n\t"
        "vsetvli t0, zero, e32, m2\n\t"
        "vse32.v v28, ($0)";

    StringRef constraints = "r,r,r,~{v0},~{v1},~{v28},~{v29},~{t0},~{memory}";

    rewriter.replaceOpWithNewOp<LLVM::InlineAsmOp>(
        op,
        TypeRange(),
        ValueRange{vdPtr, vs1Ptr, vs2Ptr},
        asmString,
        constraints,
        true,
        false,
        LLVM::AsmDialectAttr::get(getContext(), LLVM::AsmDialect::AD_ATT),
        ArrayAttr()
    );

    return success();
  }
};

struct IMEVfmadotLowering : public ConvertOpToLLVMPattern<VfmadotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(VfmadotOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value vdStruct = adaptor.getOperands()[0];
    Value vdPtr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), vdStruct, ArrayRef<int64_t>{1});

    Value vs1Struct = adaptor.getOperands()[1];
    Value vs1Ptr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), vs1Struct, ArrayRef<int64_t>{1});

    Value vs2Struct = adaptor.getOperands()[2];
    Value vs2Ptr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), vs2Struct, ArrayRef<int64_t>{1});

    StringRef asmString = 
        "vsetvli t0, zero, e16, m2\n\t"
        "vfmv.v.f v28, ft0\n\t"
        "vle16.v v0, ($1)\n\t"
        "vle16.v v1, ($2)\n\t"
        "vfmadot v28, v0, v1\n\t"
        "vse16.v v28, ($0)";

    StringRef constraints = "r,r,r,~{v0},~{v1},~{v28},~{v29},~{t0},~{memory}";

    rewriter.replaceOpWithNewOp<LLVM::InlineAsmOp>(
        op,
        TypeRange(),
        ValueRange{vdPtr, vs1Ptr, vs2Ptr},
        asmString,
        constraints,
        true,
        false,
        LLVM::AsmDialectAttr::get(getContext(), LLVM::AsmDialect::AD_ATT),
        ArrayAttr()
    );

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
