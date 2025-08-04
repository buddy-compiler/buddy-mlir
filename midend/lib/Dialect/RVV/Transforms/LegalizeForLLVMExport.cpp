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

// RVVTargetIndexBitwidth can be 64(by default) or 32(when rv32 option is set)
// to configure the size of operands' type in lowering.

static int64_t RVVTargetIndexBitwidth;

template <typename SourceOp, typename TargetOp>
class ConvertPassthruOperandOpToLLVMPattern
    : public ConvertOpToLLVMPattern<SourceOp> {
public:
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;

  /// This pattern creates an `undef` operation, inserts the `undef`
  /// operation to the beginning of the operand list, and creates the intrinsic
  /// operation.
  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    unsigned numResults = op->getNumResults();
    auto resultType = op->getResultTypes();
    Type packedType;

    Value src1 = op.getOperand(0);
    Value src2 = op.getOperand(1);
    Value vl = op.getOperand(2);
    Value vlCast = rewriter
                       .create<UnrealizedConversionCastOp>(
                           op.getLoc(),
                           rewriter.getIntegerType(RVVTargetIndexBitwidth), vl)
                       .getResult(0);
    SmallVector<Value, 6> operandsVector({src1, src2, vlCast});

    Value passthru = rewriter.create<LLVM::UndefOp>(loc, resultType[0]);
    operandsVector.insert(operandsVector.begin(), passthru);

    const LLVMTypeConverter *typeConverter = this->getTypeConverter();
    if (numResults != 0) {
      packedType = typeConverter->packFunctionResults(op->getResultTypes());
      if (!packedType)
        return failure();
    }

    // Create the intrinsic operation.
    OperationState state(loc, TargetOp::getOperationName());
    state.addTypes(packedType);
    state.addOperands(operandsVector);
    Operation *newOp = rewriter.create(state);
    return rewriter.replaceOp(op, newOp->getResult(0)), success();
  }
};

template <typename OpTy>
class ForwardOperands : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getOperands().getTypes() == op->getOperands().getTypes())
      return rewriter.notifyMatchFailure(op, "operand types already match");

    rewriter.modifyOpInPlace(op,
                             [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

class ReturnOpTypeConversion : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.modifyOpInPlace(op,
                             [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

struct RVVSetVlOpLowering : public ConvertOpToLLVMPattern<RVVSetVlOp> {
  using ConvertOpToLLVMPattern<RVVSetVlOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RVVSetVlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = rewriter.getIntegerType(RVVTargetIndexBitwidth);
    Value avl = op.getOperand(0);
    Value avlCast =
        rewriter
            .create<UnrealizedConversionCastOp>(
                op.getLoc(), rewriter.getIntegerType(RVVTargetIndexBitwidth),
                avl)
            .getResult(0);
    Value sew = op.getOperand(1);
    Value sewCast =
        rewriter
            .create<UnrealizedConversionCastOp>(
                op.getLoc(), rewriter.getIntegerType(RVVTargetIndexBitwidth),
                sew)
            .getResult(0);
    Value lmul = op.getOperand(2);
    Value lmulCast =
        rewriter
            .create<UnrealizedConversionCastOp>(
                op.getLoc(), rewriter.getIntegerType(RVVTargetIndexBitwidth),
                lmul)
            .getResult(0);
    rewriter.replaceOpWithNewOp<RVVIntrSetVlIOp>(op, resultType, avlCast,
                                                 sewCast, lmulCast);
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
    auto context = loadOp.getContext();
    Value passthru =
        rewriter.create<LLVM::UndefOp>(loadOp.getLoc(), resultType);
    LLVM::LLVMPointerType llvmDataTypePtr = LLVM::LLVMPointerType::get(context);
    Value dataPtr = getStridedElementPtr(
        loadOp.getLoc(), type, adaptor.getBase(), adaptor.getIndex(), rewriter);
    Value bitCastedPtr = rewriter.create<LLVM::BitcastOp>(
        loadOp.getLoc(), llvmDataTypePtr, dataPtr);
    Value vl = loadOp.getOperand(2);
    Value vlCast = rewriter
                       .create<UnrealizedConversionCastOp>(
                           loadOp.getLoc(),
                           rewriter.getIntegerType(RVVTargetIndexBitwidth), vl)
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
    auto context = storeOp.getContext();
    LLVM::LLVMPointerType llvmDataTypePtr = LLVM::LLVMPointerType::get(context);
    Value dataPtr =
        getStridedElementPtr(storeOp.getLoc(), type, adaptor.getBase(),
                             adaptor.getIndex(), rewriter);
    Value bitCastedPtr = rewriter.create<LLVM::BitcastOp>(
        storeOp.getLoc(), llvmDataTypePtr, dataPtr);
    Value vl = storeOp.getOperand(3);
    Value vlCast = rewriter
                       .create<UnrealizedConversionCastOp>(
                           storeOp.getLoc(),
                           rewriter.getIntegerType(RVVTargetIndexBitwidth), vl)
                       .getResult(0);
    rewriter.replaceOpWithNewOp<RVVIntrStoreEleOp>(storeOp, adaptor.getValue(),
                                                   bitCastedPtr, vlCast);
    return success();
  }
};

using RVVAddOpLowering =
    ConvertPassthruOperandOpToLLVMPattern<RVVAddOp, RVVIntrAddOp>;
using RVVMulOpLowering =
    ConvertPassthruOperandOpToLLVMPattern<RVVMulOp, RVVIntrMulOp>;

struct RsqrtOpLowering : public ConvertOpToLLVMPattern<RsqrtOp> {
  using ConvertOpToLLVMPattern<RsqrtOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RsqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto resultType = op.getResult().getType();
    Value passthru = rewriter.create<LLVM::UndefOp>(op.getLoc(), resultType);
    Value src = op.getOperand(0);
    Value vl = op.getOperand(1);
    Value vlCast = rewriter
                       .create<UnrealizedConversionCastOp>(
                           op.getLoc(),
                           rewriter.getIntegerType(RVVTargetIndexBitwidth), vl)
                       .getResult(0);
    rewriter.replaceOpWithNewOp<IntrFrsqrt7Op>(op, resultType, passthru, src,
                                               vlCast);
    return success();
  }
};

/// Populate the given list with patterns that convert from RVV to LLVM.
void mlir::populateRVVLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns,
    int64_t RVVIndexBitwidth) {

  RVVTargetIndexBitwidth = RVVIndexBitwidth;
  // clang-format off
  patterns.add<ForwardOperands<func::CallOp>,
               ForwardOperands<func::CallIndirectOp>,
               ForwardOperands<func::ReturnOp>
               >(converter, &converter.getContext());
  patterns.add<RVVSetVlOpLowering>(converter);
  patterns.add<RVVLoadOpLowering,
               RVVStoreOpLowering>(converter);
  patterns.add<RsqrtOpLowering>(converter);
  patterns.add<RVVAddOpLowering,
               RVVMulOpLowering>(converter);
  // clang-format on
}

void mlir::configureRVVLegalizeForExportTarget(LLVMConversionTarget &target) {
  // clang-format off
  target.addLegalOp<RVVIntrSetVlIOp,
                    RVVIntrLoadEleOp,
                    RVVIntrStoreEleOp,
                    IntrFrsqrt7Op,
                    RVVIntrAddOp,
                    RVVIntrMulOp>();
  target.addIllegalOp<RVVSetVlOp,
                      RVVLoadOp,
                      RVVStoreOp,
                      RsqrtOp,
                      RVVAddOp,
                      RVVMulOp>();
  // clang-format on
}
