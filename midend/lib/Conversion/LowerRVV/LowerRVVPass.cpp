//====- LowerRVVPass.cpp - RVV Dialect Lowering Pass  ---------------------===//
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
// This file defines RVV dialect lowering pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

#include "RVV/RVVDialect.h"
#include "RVV/Transforms.h"

using namespace mlir;
using namespace buddy;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
class LowerRVVToLLVMPass
    : public PassWrapper<LowerRVVToLLVMPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerRVVToLLVMPass)
  StringRef getArgument() const final { return "lower-rvv"; }
  StringRef getDescription() const final {
    return "RVV dialect lowering pass.";
  }
  LowerRVVToLLVMPass() = default;
  LowerRVVToLLVMPass(const LowerRVVToLLVMPass &) {}

  // Override explicitly to allow conditional dialect dependence.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<rvv::RVVDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void LowerRVVToLLVMPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  LLVMTypeConverter converter(context);
  RewritePatternSet patterns(context);
  LLVMConversionTarget target(*context);

  configureRVVLegalizeForExportTarget(target);
  populateRVVLegalizeForLLVMExportPatterns(converter, patterns);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerRVVPass() { PassRegistration<LowerRVVToLLVMPass>(); }
} // namespace buddy
} // namespace mlir
