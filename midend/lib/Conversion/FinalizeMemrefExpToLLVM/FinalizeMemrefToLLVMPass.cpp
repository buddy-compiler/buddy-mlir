//====- FinalizeMemrefExpToLLVMPass.cpp -=====================================//
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
// This file defines memref experiment dialect lowering pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
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
    registry.insert<LLVM::LLVMDialect, memref::MemRefDialect,
                    memref_exp::MemrefExpDialect>();
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
    LLVM::LLVMPointerType llvmPointerType =
        LLVM::LLVMPointerType::get(op->getContext());
    auto ptr = rewriter.create<LLVM::ZeroOp>(loc, llvmPointerType);
    return {ptr, ptr};
  }
};
} // namespace

void FinalizeMemrefExpToLLVMPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  LLVMTypeConverter converter(context);
  RewritePatternSet patterns(context);
  LLVMConversionTarget target(*context);
  target.addIllegalOp<memref_exp::NullOp>();
  target.addLegalDialect<LLVM::LLVMDialect>();
  patterns.add<NullOpLowering>(context);
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerFinalizeMemrefExpToLLVMPass() {
  PassRegistration<FinalizeMemrefExpToLLVMPass>();
}
} // namespace buddy
} // namespace mlir
