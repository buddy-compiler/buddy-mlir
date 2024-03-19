//====- LowerGemminiToFunc.cpp - Gemmini Dialect Lowering Pass  -----------===//
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
// This file defines Gemmini dialect lowering pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Gemmini/GemminiDialect.h"
#include "Gemmini/GemminiOps.h"
#include "Gemmini/Transform.h"

using namespace mlir;
using namespace buddy;

namespace {
class LowerGemminiToFuncPass
    : public PassWrapper<LowerGemminiToFuncPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGemminiToFuncPass)
  StringRef getArgument() const final { return "lower-gemmini-to-func"; }
  StringRef getDescription() const final {
    return "gemmini dialect lowering to func pass.";
  }
  LowerGemminiToFuncPass() = default;
  LowerGemminiToFuncPass(const LowerGemminiToFuncPass &) {}

  Option<int64_t> dim{*this, "dim", llvm::cl::desc("Size of systolic array."),
                      llvm::cl::init(16)};
  Option<int64_t> addrLen{*this, "addr_len",
                          llvm::cl::desc("The length of address."),
                          llvm::cl::init(32)};
  Option<int64_t> accRows{*this, "acc_rows", llvm::cl::desc("The row of acc."),
                          llvm::cl::init(1024)};
  Option<int64_t> bankRows{*this, "bank_rows",
                           llvm::cl::desc("The row of the bank."),
                           llvm::cl::init(4096)};
  Option<std::string> elemType{*this, "elem_t",
                               llvm::cl::desc("The type of elem_t."),
                               llvm::cl::init("i8")};
  Option<std::string> accType{*this, "acc_t",
                              llvm::cl::desc("The type of acc_t."),
                              llvm::cl::init("i32")};

  // Override explicitly to allow conditional dialect dependence.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<func::FuncDialect>();
    // registry.insert<gemmini::GemminiDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void LowerGemminiToFuncPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  // The default elem_t is int8_t,
  // so the default size of elem_t is 1 type.
  size_t sizeOfElemT = sizeof(int8_t);
  if (elemType == "f32")
    sizeOfElemT = sizeof(float);
  // The default acc_t is int32_t,
  // so the default size of acc_t is 4 type.
  size_t sizeOfAccT = sizeof(int32_t);
  if (accType == "f32")
    sizeOfAccT = sizeof(float);
  ConversionTarget target(*context);
  target.addLegalDialect<memref::MemRefDialect, arith::ArithDialect, 
                         scf::SCFDialect, func::FuncDialect>();
  RewritePatternSet patterns(context);
  // func call op
  // configureGemminiLegalizeForFuncExportTarget(target);
  populateGemminiLegalizeForFuncExportPatterns(patterns, dim,
                                               addrLen, accRows, bankRows,
                                               sizeOfElemT, sizeOfAccT);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerGemminiToFuncPass() { PassRegistration<LowerGemminiToFuncPass>(); }
} // namespace buddy
} // namespace mlir
