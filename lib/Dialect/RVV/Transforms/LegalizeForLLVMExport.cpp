//===- LegalizeForLLVMExport.cpp - Prepare RVV for LLVM translation -------===//
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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

#include "RVV/RVVDialect.h"
#include "RVV/Transforms.h"

using namespace mlir;
using namespace buddy::rvv;

template <typename OpTy>
class ForwardOperands : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getOperands().getTypes() == op->getOperands().getTypes())
      return rewriter.notifyMatchFailure(op, "operand types already match");

    rewriter.updateRootInPlace(
        op, [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

class ReturnOpTypeConversion : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.updateRootInPlace(
        op, [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

struct RVVLoadOpLowering : public ConvertOpToLLVMPattern<RVVLoadOp> {
  using ConvertOpToLLVMPattern<RVVLoadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RVVLoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = loadOp.getMemRefType();
    if (!isConvertibleAndHasIdentityMaps(type))
      return failure();

    LLVMTypeConverter converter(loadOp.getContext());

    auto resultType = loadOp.getResult().getType();
    Value passthru =
        rewriter.create<LLVM::UndefOp>(loadOp.getLoc(), resultType);
    LLVM::LLVMPointerType llvmDataTypePtr =
        LLVM::LLVMPointerType::get(resultType);
    Value dataPtr = getStridedElementPtr(
        loadOp.getLoc(), type, adaptor.getBase(), adaptor.getIndex(), rewriter);
    Value bitCastedPtr = rewriter.create<LLVM::BitcastOp>(
        loadOp.getLoc(), llvmDataTypePtr, dataPtr);
    Value vl = loadOp.getOperand(2);
    Value vlCast = rewriter
                       .create<UnrealizedConversionCastOp>(
                           loadOp.getLoc(), rewriter.getI64Type(), vl)
                       .getResult(0);
    rewriter.replaceOpWithNewOp<RVVIntrLoadEleOp>(loadOp, resultType, passthru,
                                                  bitCastedPtr, vlCast);
    return success();
  }
};

struct RVVStoreOpLowering : public ConvertOpToLLVMPattern<RVVStoreOp> {
  using ConvertOpToLLVMPattern<RVVStoreOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RVVStoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = storeOp.getMemRefType();
    if (!isConvertibleAndHasIdentityMaps(type))
      return failure();

    LLVMTypeConverter converter(storeOp.getContext());

    auto resultType = storeOp.getValue().getType();
    LLVM::LLVMPointerType llvmDataTypePtr =
        LLVM::LLVMPointerType::get(resultType);
    ;
    Value dataPtr =
        getStridedElementPtr(storeOp.getLoc(), type, adaptor.getBase(),
                             adaptor.getIndex(), rewriter);
    Value bitCastedPtr = rewriter.create<LLVM::BitcastOp>(
        storeOp.getLoc(), llvmDataTypePtr, dataPtr);
    Value vl = storeOp.getOperand(3);
    Value vlCast = rewriter
                       .create<UnrealizedConversionCastOp>(
                           storeOp.getLoc(), rewriter.getI64Type(), vl)
                       .getResult(0);
    rewriter.replaceOpWithNewOp<RVVIntrStoreEleOp>(storeOp, adaptor.getValue(),
                                                   bitCastedPtr, vlCast);
    return success();
  }
};

using RVVSetVlOpLowering =
    OneToOneConvertToLLVMPattern<RVVSetVlOp, RVVIntrSetVlIOp>;

/// Populate the given list with patterns that convert from RVV to LLVM.
void mlir::populateRVVLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  // clang-format off
  patterns.add<ForwardOperands<func::CallOp>,
               ForwardOperands<func::CallIndirectOp>,
               ForwardOperands<func::ReturnOp>
               >(converter, &converter.getContext());
  patterns.add<RVVSetVlOpLowering>(converter);
  patterns.add<RVVLoadOpLowering,
               RVVStoreOpLowering>(converter);
  // clang-format on
}

void mlir::configureRVVLegalizeForExportTarget(LLVMConversionTarget &target) {
  // clang-format off
  target.addLegalOp<RVVIntrSetVlIOp,
                    RVVIntrLoadEleOp,
                    RVVIntrStoreEleOp>();
  target.addIllegalOp<RVVSetVlOp,
                      RVVLoadOp,
                      RVVStoreOp>();
  // clang-format on
}
