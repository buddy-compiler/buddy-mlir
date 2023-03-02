//====- LowerVectorExpPass.cpp - Vector Experiment Dialect Lowering Pass  -===//
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
// This file defines vector experiment dialect lowering pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"

#include "VectorExp/VectorExpDialect.h"
#include "VectorExp/VectorExpOps.h"

using namespace mlir;
using namespace buddy;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
class VectorExpPredicationLowering
    : public ConvertOpToLLVMPattern<vector_exp::PredicationOp> {
public:
  using ConvertOpToLLVMPattern<
      vector_exp::PredicationOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector_exp::PredicationOp op,
                  vector_exp::PredicationOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the region and block from the predication operation.
    Location loc = op.getLoc();
    Region &configRegion = op.getRegion();
    mlir::Block &configBlock = configRegion.front();
    // Iterate region and get the operations inside.
    for (mlir::Operation &innerOp : configBlock.getOperations()) {
      //
      if (isa<arith::AddFOp>(innerOp)) {
        Type resultType = cast<arith::AddFOp>(innerOp).getResult().getType();
        Value result = rewriter.create<LLVM::VPFAddOp>(
            loc, resultType, cast<arith::AddFOp>(innerOp).getLhs(),
            cast<arith::AddFOp>(innerOp).getRhs(), op.getMask(), op.getVl());
        rewriter.replaceOp(op, result);
      } else if (isa<arith::MulFOp>(innerOp)) {
        Type resultType = cast<arith::MulFOp>(innerOp).getResult().getType();
        Value result = rewriter.create<LLVM::VPFMulOp>(
            loc, resultType, cast<arith::MulFOp>(innerOp).getLhs(),
            cast<arith::MulFOp>(innerOp).getRhs(), op.getMask(), op.getVl());
        rewriter.replaceOp(op, result);
      } else if (isa<vector::LoadOp>(innerOp)) {
        vector::LoadOp loadOp = cast<vector::LoadOp>(innerOp);
        // Prepare the MemRef descriptor for the `getStridedElementPtr`.
        // - Get the MemRef type of the load operation.
        // - Convert the MemRef type into LLVM struct type.
        // - Create UnrealizedConversionCastOp to provide the descriptor value.
        MemRefType memRefTy = loadOp.getMemRefType();
        Type structType = this->getTypeConverter()->convertType(memRefTy);
        Value memDesc = rewriter
                            .create<UnrealizedConversionCastOp>(
                                loc, structType, loadOp.getBase())
                            .getResult(0);
        // Prepare the integer indices for the `getStridedElementPtr`.
        // - Interate the indices of the load operation.
        // - Convert origin index type into integer type.
        // - Create UnrealizedConversionCastOp to provide the integer value.
        SmallVector<Value, 4> indices;
        for (Value idx : loadOp.getIndices()) {
          Type idxType = idx.getType();
          Type intType = this->getTypeConverter()->convertType(idxType);
          Value intIdx =
              rewriter.create<UnrealizedConversionCastOp>(loc, intType, idx)
                  .getResult(0);
          indices.push_back(intIdx);
        }
        // Prepare the data pointer for the VP load operation.
        // - Call the `getStridedElementPtr` with above descriptor and indices.
        Value dataPtr = this->getStridedElementPtr(loc, memRefTy, memDesc,
                                                   indices, rewriter);
        // Create VP load operation and replace the predication operation.
        // - Get the result type of the predication operation.
        // - Create VP load operation.
        // - Replace original predication operation.
        VectorType resultType = op.getResult().getType().cast<VectorType>();
        Value resultValue = rewriter.create<LLVM::VPLoadOp>(
            loc, resultType, dataPtr, op.getMask(), op.getVl());
        rewriter.replaceOp(op, resultValue);
      } else if (isa<vector::YieldOp>(innerOp)) {
        // Skip the YieldOp.
        continue;
      } else {
        // Unsupported inner operations.
        mlir::emitError(loc)
            << "unsupported inner operation " << innerOp.getName()
            << " of the predication operation.";
      }
    }
    return success();
  }
};
} // end anonymous namespace

void populateLowerVectorExpConversionPatterns(LLVMTypeConverter &converter,
                                              RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<VectorExpPredicationLowering>(converter);
  // clang-format on
}

//===----------------------------------------------------------------------===//
// LowerVectorExpPass
//===----------------------------------------------------------------------===//

namespace {
class LowerVectorExpPass
    : public PassWrapper<LowerVectorExpPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerVectorExpPass)
  LowerVectorExpPass() = default;
  LowerVectorExpPass(const LowerVectorExpPass &) {}

  StringRef getArgument() const final { return "lower-vector-exp"; }
  StringRef getDescription() const final {
    return "Lower Vector Experiment Dialect.";
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<
        arith::ArithDialect,
        buddy::vector_exp::VectorExpDialect,
        func::FuncDialect,
        memref::MemRefDialect,
        LLVM::LLVMDialect>();
    // clang-format on
  }
};
} // end anonymous namespace.

void LowerVectorExpPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  LLVMConversionTarget target(*context);
  // clang-format off
  target.addLegalDialect<
      arith::ArithDialect,
      func::FuncDialect,
      memref::MemRefDialect,
      LLVM::LLVMDialect
    >();
  // clang-format on
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  LLVMTypeConverter converter(context);
  RewritePatternSet patterns(context);
  populateLowerVectorExpConversionPatterns(converter, patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerVectorExpPass() { PassRegistration<LowerVectorExpPass>(); }
} // namespace buddy
} // namespace mlir
