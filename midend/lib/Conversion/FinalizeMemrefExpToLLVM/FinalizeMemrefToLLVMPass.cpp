#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/AllocLikeConversion.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "MemrefExp/MemrefExpDialect.h"
#include "MemrefExp/MemrefExpOps.h"

using namespace mlir;
using namespace buddy;

namespace {
class FinalizeMemrefExpToLLVMPass
    : public PassWrapper<FinalizeMemrefExpToLLVMPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FinalizeMemrefExpToLLVMPass)
  StringRef getArgument() const final { return "finalize-memrefexp-to-llvm"; }
  StringRef getDescription() const final {
    return "convert memrefexp dialect to llvm dialect.";
  }
  FinalizeMemrefExpToLLVMPass() = default;
  
  // Override explicitly to allow conditional dialect dependence.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, memref::MemRefDialect, memref_exp::MemrefExpDialect>();
  }

  void runOnOperation() override;
};

struct NullOpLowering : public AllocLikeOpLLVMLowering {
  NullOpLowering(const LLVMTypeConverter &converter)
      : AllocLikeOpLLVMLowering(memref_exp::NullOp::getOperationName(),
                                converter) {}
  std::tuple<Value, Value> allocateBuffer(ConversionPatternRewriter &rewriter,
                                          Location loc, Value sizeBytes,
                                          Operation *op) const override {
    LLVM::LLVMPointerType llvmPointerType = LLVM::LLVMPointerType::get(op->getContext());
    auto ptr = rewriter.create<LLVM::ZeroOp>(loc, llvmPointerType);
    return {ptr, ptr};
  }
};
}

void FinalizeMemrefExpToLLVMPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  LLVMTypeConverter converter(context);
  RewritePatternSet patterns(context);
  LLVMConversionTarget target(*context);
  //target.addIllegalOp<memref_exp::NullOp, memref::ExtractAlignedPointerAsIndexOp>();
  patterns.add<NullOpLowering>(context);
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerFinalizeMemrefExpToLLVMPass() { PassRegistration<FinalizeMemrefExpToLLVMPass>(); }
} // namespace buddy
} // namespace mlir
