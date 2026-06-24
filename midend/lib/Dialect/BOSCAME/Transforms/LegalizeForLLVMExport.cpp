//====- LegalizeForLLVMExport.cpp - Prepare BOSCAME for LLVM translation --===//
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

#include "Dialect/BOSCAME/BOSCAMEDialect.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/Transform.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace buddy::boscame;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static FlatSymbolRefAttr
getOrInsertIntrinsic(ConversionPatternRewriter &rewriter, ModuleOp module,
                     StringRef intrinsicName, LLVM::LLVMFunctionType funcType) {
  auto *ctx = rewriter.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(intrinsicName))
    return FlatSymbolRefAttr::get(ctx, intrinsicName);

  auto savedInsertionPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToEnd(module.getBody());
  LLVM::LLVMFuncOp::create(rewriter, module.getLoc(), intrinsicName, funcType,
                           LLVM::Linkage::External, false, LLVM::CConv::C);
  rewriter.restoreInsertionPoint(savedInsertionPoint);
  return FlatSymbolRefAttr::get(ctx, intrinsicName);
}

static Value extractPointerFromMemref(ConversionPatternRewriter &rewriter,
                                      Location loc, Value memref) {
  auto *ctx = rewriter.getContext();
  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  auto i64Type = IntegerType::get(ctx, 64);
  Value idx =
      memref::ExtractAlignedPointerAsIndexOp::create(rewriter, loc, memref);
  Value i64Val = arith::IndexCastOp::create(rewriter, loc, i64Type, idx);
  Value ptr = LLVM::IntToPtrOp::create(rewriter, loc, ptrType, i64Val);
  return ptr;
}

//===----------------------------------------------------------------------===//
// BOSCAME Lowering Patterns
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Configuration Operations Lowering
//===----------------------------------------------------------------------===//
// configuration imm operations
template <typename OpTy, uint64_t (OpTy::*AttrGetter)()>
struct BOSCAMEConfigImmLowering : public ConvertOpToLLVMPattern<OpTy> {
  StringRef intrinsicName;

  BOSCAMEConfigImmLowering(LLVMTypeConverter &typeConverter,
                           StringRef intrinsicName)
      : ConvertOpToLLVMPattern<OpTy>(typeConverter),
        intrinsicName(intrinsicName) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->template getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(i64Type, {i64Type});
    auto intrNameAttr =
        getOrInsertIntrinsic(rewriter, module, intrinsicName, funcType);

    uint64_t attrVal = (op.*AttrGetter)();
    Value immVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(attrVal));

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, TypeRange{i64Type}, intrNameAttr, ValueRange{immVal});
    rewriter.replaceOp(op, callOp.getResults());
    return success();
  }
};

// register configuration operations
template <typename OpTy>
struct BOSCAMEConfigRegLowering : public ConvertOpToLLVMPattern<OpTy> {
  StringRef intrinsicName;

  BOSCAMEConfigRegLowering(LLVMTypeConverter &typeConverter,
                           StringRef intrinsicName)
      : ConvertOpToLLVMPattern<OpTy>(typeConverter),
        intrinsicName(intrinsicName) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->template getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(i64Type, {i64Type});
    auto intrNameAttr =
        getOrInsertIntrinsic(rewriter, module, intrinsicName, funcType);

    Value rs1Val = adaptor.getRs1();

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, TypeRange{i64Type}, intrNameAttr, ValueRange{rs1Val});
    rewriter.replaceOp(op, callOp.getResults());
    return success();
  }
};

/// Lowering pattern for Elementwise Math Operations
template <typename OpTy>
struct BOSCAMEMath2Lowering : public ConvertOpToLLVMPattern<OpTy> {
  StringRef intrinsicName;

  BOSCAMEMath2Lowering(LLVMTypeConverter &typeConverter,
                       StringRef intrinsicName)
      : ConvertOpToLLVMPattern<OpTy>(typeConverter),
        intrinsicName(intrinsicName) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->template getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type});

    auto intrNameAttr =
        getOrInsertIntrinsic(rewriter, module, intrinsicName, funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrNameAttr,
                                  ValueRange{mdVal, ms1Val});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Load Operations Lowering
//===----------------------------------------------------------------------===//
template <typename OpTy>
struct BOSCAMELoadLowering : public ConvertOpToLLVMPattern<OpTy> {
  StringRef intrinsicName;

  BOSCAMELoadLowering(LLVMTypeConverter &typeConverter, StringRef intrinsicName)
      : ConvertOpToLLVMPattern<OpTy>(typeConverter),
        intrinsicName(intrinsicName) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->template getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, ptrType, i64Type});

    auto intrNameAttr =
        getOrInsertIntrinsic(rewriter, module, intrinsicName, funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrNameAttr,
        ValueRange{mdVal, basePtr, adaptor.getStride()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Store Operations Lowering
//===----------------------------------------------------------------------===//
template <typename OpTy>
struct BOSCAMEStoreLowering : public ConvertOpToLLVMPattern<OpTy> {
  StringRef intrinsicName;

  BOSCAMEStoreLowering(LLVMTypeConverter &typeConverter,
                       StringRef intrinsicName)
      : ConvertOpToLLVMPattern<OpTy>(typeConverter),
        intrinsicName(intrinsicName) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->template getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, ptrType, i64Type});

    auto intrNameAttr =
        getOrInsertIntrinsic(rewriter, module, intrinsicName, funcType);

    Value ms3Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs3()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrNameAttr,
        ValueRange{ms3Val, basePtr, adaptor.getStride()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Data Move / Broadcast / Transpose Lowering Templates
//===----------------------------------------------------------------------===//
/// 1. Pure Move
template <typename OpTy>
struct BOSCAMEPureMoveLowering : public ConvertOpToLLVMPattern<OpTy> {
  StringRef intrinsicName;
  BOSCAMEPureMoveLowering(LLVMTypeConverter &typeConverter,
                          StringRef intrinsicName)
      : ConvertOpToLLVMPattern<OpTy>(typeConverter),
        intrinsicName(intrinsicName) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    auto module = op->template getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type});
    auto intrNameAttr =
        getOrInsertIntrinsic(rewriter, module, intrinsicName, funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    rewriter.create<LLVM::CallOp>(op.getLoc(), TypeRange{}, intrNameAttr,
                                  ValueRange{mdVal, ms1Val});
    rewriter.eraseOp(op);
    return success();
  }
};

/// 2. Indexed Move
template <typename OpTy>
struct BOSCAMEIndexedMoveLowering : public ConvertOpToLLVMPattern<OpTy> {
  StringRef intrinsicName;
  BOSCAMEIndexedMoveLowering(LLVMTypeConverter &typeConverter,
                             StringRef intrinsicName)
      : ConvertOpToLLVMPattern<OpTy>(typeConverter),
        intrinsicName(intrinsicName) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    auto module = op->template getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, i64Type});
    auto intrNameAttr =
        getOrInsertIntrinsic(rewriter, module, intrinsicName, funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    rewriter.create<LLVM::CallOp>(op.getLoc(), TypeRange{}, intrNameAttr,
                                  ValueRange{mdVal, ms1Val, adaptor.getRs2()});
    rewriter.eraseOp(op);
    return success();
  }
};

/// 3. Immediate Indexed Move
template <typename OpTy, uint64_t (OpTy::*ImmGetter)()>
struct BOSCAMEImmIndexedMoveLowering : public ConvertOpToLLVMPattern<OpTy> {
  StringRef intrinsicName;
  BOSCAMEImmIndexedMoveLowering(LLVMTypeConverter &typeConverter,
                                StringRef intrinsicName)
      : ConvertOpToLLVMPattern<OpTy>(typeConverter),
        intrinsicName(intrinsicName) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    auto module = op->template getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, i64Type});
    auto intrNameAttr =
        getOrInsertIntrinsic(rewriter, module, intrinsicName, funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    uint64_t imm = (op.*ImmGetter)();
    Value immVal = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i64Type, rewriter.getI64IntegerAttr(imm));

    rewriter.create<LLVM::CallOp>(op.getLoc(), TypeRange{}, intrNameAttr,
                                  ValueRange{mdVal, ms1Val, immVal});
    rewriter.eraseOp(op);
    return success();
  }
};

/// 4. Matrix to Scalar
template <typename OpTy>
struct BOSCAMEMatrixToScalarLowering : public ConvertOpToLLVMPattern<OpTy> {
  StringRef intrinsicName;
  BOSCAMEMatrixToScalarLowering(LLVMTypeConverter &typeConverter,
                                StringRef intrinsicName)
      : ConvertOpToLLVMPattern<OpTy>(typeConverter),
        intrinsicName(intrinsicName) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    auto module = op->template getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(i64Type, {i64Type, i64Type});
    auto intrNameAttr =
        getOrInsertIntrinsic(rewriter, module, intrinsicName, funcType);

    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    auto callOp = rewriter.create<LLVM::CallOp>(
        op.getLoc(), TypeRange{i64Type}, intrNameAttr,
        ValueRange{ms1Val, adaptor.getRs2()});
    rewriter.replaceOp(op, callOp.getResults());
    return success();
  }
};

/// 5. Scalar to Matrix
template <typename OpTy>
struct BOSCAMEScalarToMatrixLowering : public ConvertOpToLLVMPattern<OpTy> {
  StringRef intrinsicName;
  BOSCAMEScalarToMatrixLowering(LLVMTypeConverter &typeConverter,
                                StringRef intrinsicName)
      : ConvertOpToLLVMPattern<OpTy>(typeConverter),
        intrinsicName(intrinsicName) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    auto module = op->template getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, i64Type});
    auto intrNameAttr =
        getOrInsertIntrinsic(rewriter, module, intrinsicName, funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i64Type, rewriter.getI64IntegerAttr(op.getMd()));

    rewriter.create<LLVM::CallOp>(
        op.getLoc(), TypeRange{}, intrNameAttr,
        ValueRange{mdVal, adaptor.getRs1(), adaptor.getRs2()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Tile Register Matrix Multiply Lowering
//===----------------------------------------------------------------------===//

/// Lowering pattern for tile register matrix multiply
template <typename OpTy>
struct BOSCAMEMath3Lowering : public ConvertOpToLLVMPattern<OpTy> {
  StringRef intrinsicName;

  BOSCAMEMath3Lowering(LLVMTypeConverter &typeConverter,
                       StringRef intrinsicName)
      : ConvertOpToLLVMPattern<OpTy>(typeConverter),
        intrinsicName(intrinsicName) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->template getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, i64Type});

    auto intrNameAttr =
        getOrInsertIntrinsic(rewriter, module, intrinsicName, funcType);

    Value mdVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms1Val = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));
    Value ms2Val = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrNameAttr,
                                  ValueRange{mdVal, ms1Val, ms2Val});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//
struct LegalizeBOSCAMEForLLVMExport
    : public PassWrapper<LegalizeBOSCAMEForLLVMExport,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeBOSCAMEForLLVMExport)

  StringRef getArgument() const final { return "lower-bosc-ame"; }
  StringRef getDescription() const final {
    return "BOSCAME dialect lowering pass.";
  }

  LegalizeBOSCAMEForLLVMExport() = default;
  LegalizeBOSCAMEForLLVMExport(const LegalizeBOSCAMEForLLVMExport &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<BOSCAMEDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext &context = getContext();

    LLVMConversionTarget target(context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<memref::MemRefDialect>();

    target.addIllegalDialect<buddy::boscame::BOSCAMEDialect>();

    LLVMTypeConverter typeConverter(&context);
    RewritePatternSet patterns(&context);

    populateBOSCAMELegalizeForLLVMExportPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::populateBOSCAMELegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // Configuration patterns
  patterns.add<BOSCAMEConfigImmLowering<MSettypeiOp, &MSettypeiOp::getTimm>>(
      converter, "llvm.riscv.bosc.msettypei");

  patterns.add<BOSCAMEConfigImmLowering<MSettypehiOp, &MSettypehiOp::getTimm>>(
      converter, "llvm.riscv.bosc.msettypehi");

  patterns.add<BOSCAMEConfigImmLowering<MSettilemiOp, &MSettilemiOp::getTilem>>(
      converter, "llvm.riscv.bosc.msettilemi");

  patterns.add<BOSCAMEConfigImmLowering<MSettileniOp, &MSettileniOp::getTilen>>(
      converter, "llvm.riscv.bosc.msettileni");

  patterns.add<BOSCAMEConfigImmLowering<MSettilekiOp, &MSettilekiOp::getTilek>>(
      converter, "llvm.riscv.bosc.msettileki");

  patterns.add<BOSCAMEConfigRegLowering<MSettypeOp>>(
      converter, "llvm.riscv.bosc.msettype");
  patterns.add<BOSCAMEConfigRegLowering<MSettilemOp>>(
      converter, "llvm.riscv.bosc.msettilem");
  patterns.add<BOSCAMEConfigRegLowering<MSettilenOp>>(
      converter, "llvm.riscv.bosc.msettilen");
  patterns.add<BOSCAMEConfigRegLowering<MSettilekOp>>(
      converter, "llvm.riscv.bosc.msettilek");

  // Element-wise matrix multiply-accumulate patterns
  // 1. ADD / SUB - No Widen
  patterns.add<BOSCAMEMath3Lowering<MadduMmOp>>(converter,
                                                "llvm.riscv.bosc.maddu.mm");
  patterns.add<BOSCAMEMath3Lowering<MadduHbMmOp>>(
      converter, "llvm.riscv.bosc.maddu.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MadduBMmOp>>(converter,
                                                 "llvm.riscv.bosc.maddu.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MadduHMmOp>>(converter,
                                                 "llvm.riscv.bosc.maddu.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MadduWMmOp>>(converter,
                                                 "llvm.riscv.bosc.maddu.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MadduDwMmOp>>(
      converter, "llvm.riscv.bosc.maddu.dw.mm");

  patterns.add<BOSCAMEMath3Lowering<MsadduMmOp>>(converter,
                                                 "llvm.riscv.bosc.msaddu.mm");
  patterns.add<BOSCAMEMath3Lowering<MsadduHbMmOp>>(
      converter, "llvm.riscv.bosc.msaddu.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MsadduBMmOp>>(
      converter, "llvm.riscv.bosc.msaddu.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MsadduHMmOp>>(
      converter, "llvm.riscv.bosc.msaddu.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MsadduWMmOp>>(
      converter, "llvm.riscv.bosc.msaddu.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MsadduDwMmOp>>(
      converter, "llvm.riscv.bosc.msaddu.dw.mm");

  patterns.add<BOSCAMEMath3Lowering<MaddMmOp>>(converter,
                                               "llvm.riscv.bosc.madd.mm");
  patterns.add<BOSCAMEMath3Lowering<MaddHbMmOp>>(converter,
                                                 "llvm.riscv.bosc.madd.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MaddBMmOp>>(converter,
                                                "llvm.riscv.bosc.madd.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MaddHMmOp>>(converter,
                                                "llvm.riscv.bosc.madd.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MaddWMmOp>>(converter,
                                                "llvm.riscv.bosc.madd.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MaddDwMmOp>>(converter,
                                                 "llvm.riscv.bosc.madd.dw.mm");

  patterns.add<BOSCAMEMath3Lowering<MsaddMmOp>>(converter,
                                                "llvm.riscv.bosc.msadd.mm");
  patterns.add<BOSCAMEMath3Lowering<MsaddHbMmOp>>(
      converter, "llvm.riscv.bosc.msadd.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MsaddBMmOp>>(converter,
                                                 "llvm.riscv.bosc.msadd.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MsaddHMmOp>>(converter,
                                                 "llvm.riscv.bosc.msadd.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MsaddWMmOp>>(converter,
                                                 "llvm.riscv.bosc.msadd.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MsaddDwMmOp>>(
      converter, "llvm.riscv.bosc.msadd.dw.mm");

  patterns.add<BOSCAMEMath3Lowering<MfaddMmOp>>(converter,
                                                "llvm.riscv.bosc.mfadd.mm");
  patterns.add<BOSCAMEMath3Lowering<MfaddCfMmOp>>(
      converter, "llvm.riscv.bosc.mfadd.cf.mm");
  patterns.add<BOSCAMEMath3Lowering<MfaddHfMmOp>>(
      converter, "llvm.riscv.bosc.mfadd.hf.mm");
  patterns.add<BOSCAMEMath3Lowering<MfaddFMmOp>>(converter,
                                                 "llvm.riscv.bosc.mfadd.f.mm");
  patterns.add<BOSCAMEMath3Lowering<MfaddDMmOp>>(converter,
                                                 "llvm.riscv.bosc.mfadd.d.mm");

  patterns.add<BOSCAMEMath3Lowering<MsubuMmOp>>(converter,
                                                "llvm.riscv.bosc.msubu.mm");
  patterns.add<BOSCAMEMath3Lowering<MsubuHbMmOp>>(
      converter, "llvm.riscv.bosc.msubu.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MsubuBMmOp>>(converter,
                                                 "llvm.riscv.bosc.msubu.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MsubuHMmOp>>(converter,
                                                 "llvm.riscv.bosc.msubu.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MsubuWMmOp>>(converter,
                                                 "llvm.riscv.bosc.msubu.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MsubuDwMmOp>>(
      converter, "llvm.riscv.bosc.msubu.dw.mm");

  patterns.add<BOSCAMEMath3Lowering<MssubuMmOp>>(converter,
                                                 "llvm.riscv.bosc.mssubu.mm");
  patterns.add<BOSCAMEMath3Lowering<MssubuHbMmOp>>(
      converter, "llvm.riscv.bosc.mssubu.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MssubuBMmOp>>(
      converter, "llvm.riscv.bosc.mssubu.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MssubuHMmOp>>(
      converter, "llvm.riscv.bosc.mssubu.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MssubuWMmOp>>(
      converter, "llvm.riscv.bosc.mssubu.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MssubuDwMmOp>>(
      converter, "llvm.riscv.bosc.mssubu.dw.mm");

  patterns.add<BOSCAMEMath3Lowering<MsubMmOp>>(converter,
                                               "llvm.riscv.bosc.msub.mm");
  patterns.add<BOSCAMEMath3Lowering<MsubHbMmOp>>(converter,
                                                 "llvm.riscv.bosc.msub.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MsubBMmOp>>(converter,
                                                "llvm.riscv.bosc.msub.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MsubHMmOp>>(converter,
                                                "llvm.riscv.bosc.msub.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MsubWMmOp>>(converter,
                                                "llvm.riscv.bosc.msub.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MsubDwMmOp>>(converter,
                                                 "llvm.riscv.bosc.msub.dw.mm");

  patterns.add<BOSCAMEMath3Lowering<MssubMmOp>>(converter,
                                                "llvm.riscv.bosc.mssub.mm");
  patterns.add<BOSCAMEMath3Lowering<MssubHbMmOp>>(
      converter, "llvm.riscv.bosc.mssub.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MssubBMmOp>>(converter,
                                                 "llvm.riscv.bosc.mssub.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MssubHMmOp>>(converter,
                                                 "llvm.riscv.bosc.mssub.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MssubWMmOp>>(converter,
                                                 "llvm.riscv.bosc.mssub.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MssubDwMmOp>>(
      converter, "llvm.riscv.bosc.mssub.dw.mm");

  patterns.add<BOSCAMEMath3Lowering<MfsubMmOp>>(converter,
                                                "llvm.riscv.bosc.mfsub.mm");
  patterns.add<BOSCAMEMath3Lowering<MfsubCfMmOp>>(
      converter, "llvm.riscv.bosc.mfsub.cf.mm");
  patterns.add<BOSCAMEMath3Lowering<MfsubHfMmOp>>(
      converter, "llvm.riscv.bosc.mfsub.hf.mm");
  patterns.add<BOSCAMEMath3Lowering<MfsubFMmOp>>(converter,
                                                 "llvm.riscv.bosc.mfsub.f.mm");
  patterns.add<BOSCAMEMath3Lowering<MfsubDMmOp>>(converter,
                                                 "llvm.riscv.bosc.mfsub.d.mm");

  // 2. ADD / SUB - Double Widen
  patterns.add<BOSCAMEMath3Lowering<MwadduMmOp>>(converter,
                                                 "llvm.riscv.bosc.mwaddu.mm");
  patterns.add<BOSCAMEMath3Lowering<MwadduHbMmOp>>(
      converter, "llvm.riscv.bosc.mwaddu.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MwadduBMmOp>>(
      converter, "llvm.riscv.bosc.mwaddu.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MwadduHMmOp>>(
      converter, "llvm.riscv.bosc.mwaddu.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MwadduWMmOp>>(
      converter, "llvm.riscv.bosc.mwaddu.w.mm");

  patterns.add<BOSCAMEMath3Lowering<MwaddMmOp>>(converter,
                                                "llvm.riscv.bosc.mwadd.mm");
  patterns.add<BOSCAMEMath3Lowering<MwaddHbMmOp>>(
      converter, "llvm.riscv.bosc.mwadd.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MwaddBMmOp>>(converter,
                                                 "llvm.riscv.bosc.mwadd.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MwaddHMmOp>>(converter,
                                                 "llvm.riscv.bosc.mwadd.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MwaddWMmOp>>(converter,
                                                 "llvm.riscv.bosc.mwadd.w.mm");

  patterns.add<BOSCAMEMath3Lowering<MfwaddMmOp>>(converter,
                                                 "llvm.riscv.bosc.mfwadd.mm");
  patterns.add<BOSCAMEMath3Lowering<MfwaddCfMmOp>>(
      converter, "llvm.riscv.bosc.mfwadd.cf.mm");
  patterns.add<BOSCAMEMath3Lowering<MfwaddHfMmOp>>(
      converter, "llvm.riscv.bosc.mfwadd.hf.mm");
  patterns.add<BOSCAMEMath3Lowering<MfwaddFMmOp>>(
      converter, "llvm.riscv.bosc.mfwadd.f.mm");

  patterns.add<BOSCAMEMath3Lowering<MwsubuMmOp>>(converter,
                                                 "llvm.riscv.bosc.mwsubu.mm");
  patterns.add<BOSCAMEMath3Lowering<MwsubuHbMmOp>>(
      converter, "llvm.riscv.bosc.mwsubu.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MwsubuBMmOp>>(
      converter, "llvm.riscv.bosc.mwsubu.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MwsubuHMmOp>>(
      converter, "llvm.riscv.bosc.mwsubu.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MwsubuWMmOp>>(
      converter, "llvm.riscv.bosc.mwsubu.w.mm");

  patterns.add<BOSCAMEMath3Lowering<MwsubMmOp>>(converter,
                                                "llvm.riscv.bosc.mwsub.mm");
  patterns.add<BOSCAMEMath3Lowering<MwsubHbMmOp>>(
      converter, "llvm.riscv.bosc.mwsub.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MwsubBMmOp>>(converter,
                                                 "llvm.riscv.bosc.mwsub.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MwsubHMmOp>>(converter,
                                                 "llvm.riscv.bosc.mwsub.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MwsubWMmOp>>(converter,
                                                 "llvm.riscv.bosc.mwsub.w.mm");

  patterns.add<BOSCAMEMath3Lowering<MfwsubMmOp>>(converter,
                                                 "llvm.riscv.bosc.mfwsub.mm");
  patterns.add<BOSCAMEMath3Lowering<MfwsubCfMmOp>>(
      converter, "llvm.riscv.bosc.mfwsub.cf.mm");
  patterns.add<BOSCAMEMath3Lowering<MfwsubHfMmOp>>(
      converter, "llvm.riscv.bosc.mfwsub.hf.mm");
  patterns.add<BOSCAMEMath3Lowering<MfwsubFMmOp>>(
      converter, "llvm.riscv.bosc.mfwsub.f.mm");

  // 3. MIN / MAX - No Widen
  patterns.add<BOSCAMEMath3Lowering<MminuMmOp>>(converter,
                                                "llvm.riscv.bosc.mminu.mm");
  patterns.add<BOSCAMEMath3Lowering<MminuHbMmOp>>(
      converter, "llvm.riscv.bosc.mminu.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MminuBMmOp>>(converter,
                                                 "llvm.riscv.bosc.mminu.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MminuHMmOp>>(converter,
                                                 "llvm.riscv.bosc.mminu.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MminuWMmOp>>(converter,
                                                 "llvm.riscv.bosc.mminu.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MminuDwMmOp>>(
      converter, "llvm.riscv.bosc.mminu.dw.mm");

  patterns.add<BOSCAMEMath3Lowering<MmaxuMmOp>>(converter,
                                                "llvm.riscv.bosc.mmaxu.mm");
  patterns.add<BOSCAMEMath3Lowering<MmaxuHbMmOp>>(
      converter, "llvm.riscv.bosc.mmaxu.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MmaxuBMmOp>>(converter,
                                                 "llvm.riscv.bosc.mmaxu.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MmaxuHMmOp>>(converter,
                                                 "llvm.riscv.bosc.mmaxu.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MmaxuWMmOp>>(converter,
                                                 "llvm.riscv.bosc.mmaxu.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MmaxuDwMmOp>>(
      converter, "llvm.riscv.bosc.mmaxu.dw.mm");

  patterns.add<BOSCAMEMath3Lowering<MminMmOp>>(converter,
                                               "llvm.riscv.bosc.mmin.mm");
  patterns.add<BOSCAMEMath3Lowering<MminHbMmOp>>(converter,
                                                 "llvm.riscv.bosc.mmin.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MminBMmOp>>(converter,
                                                "llvm.riscv.bosc.mmin.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MminHMmOp>>(converter,
                                                "llvm.riscv.bosc.mmin.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MminWMmOp>>(converter,
                                                "llvm.riscv.bosc.mmin.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MminDwMmOp>>(converter,
                                                 "llvm.riscv.bosc.mmin.dw.mm");

  patterns.add<BOSCAMEMath3Lowering<MmaxMmOp>>(converter,
                                               "llvm.riscv.bosc.mmax.mm");
  patterns.add<BOSCAMEMath3Lowering<MmaxHbMmOp>>(converter,
                                                 "llvm.riscv.bosc.mmax.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MmaxBMmOp>>(converter,
                                                "llvm.riscv.bosc.mmax.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MmaxHMmOp>>(converter,
                                                "llvm.riscv.bosc.mmax.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MmaxWMmOp>>(converter,
                                                "llvm.riscv.bosc.mmax.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MmaxDwMmOp>>(converter,
                                                 "llvm.riscv.bosc.mmax.dw.mm");

  patterns.add<BOSCAMEMath3Lowering<MfminMmOp>>(converter,
                                                "llvm.riscv.bosc.mfmin.mm");
  patterns.add<BOSCAMEMath3Lowering<MfminCfMmOp>>(
      converter, "llvm.riscv.bosc.mfmin.cf.mm");
  patterns.add<BOSCAMEMath3Lowering<MfminHfMmOp>>(
      converter, "llvm.riscv.bosc.mfmin.hf.mm");
  patterns.add<BOSCAMEMath3Lowering<MfminFMmOp>>(converter,
                                                 "llvm.riscv.bosc.mfmin.f.mm");
  patterns.add<BOSCAMEMath3Lowering<MfminDMmOp>>(converter,
                                                 "llvm.riscv.bosc.mfmin.d.mm");

  patterns.add<BOSCAMEMath3Lowering<MfmaxMmOp>>(converter,
                                                "llvm.riscv.bosc.mfmax.mm");
  patterns.add<BOSCAMEMath3Lowering<MfmaxCfMmOp>>(
      converter, "llvm.riscv.bosc.mfmax.cf.mm");
  patterns.add<BOSCAMEMath3Lowering<MfmaxHfMmOp>>(
      converter, "llvm.riscv.bosc.mfmax.hf.mm");
  patterns.add<BOSCAMEMath3Lowering<MfmaxFMmOp>>(converter,
                                                 "llvm.riscv.bosc.mfmax.f.mm");
  patterns.add<BOSCAMEMath3Lowering<MfmaxDMmOp>>(converter,
                                                 "llvm.riscv.bosc.mfmax.d.mm");

  // 4. MUL / MULH / DIV - No Widen
  patterns.add<BOSCAMEMath3Lowering<MsmuluMmOp>>(converter,
                                                 "llvm.riscv.bosc.msmulu.mm");
  patterns.add<BOSCAMEMath3Lowering<MsmuluHbMmOp>>(
      converter, "llvm.riscv.bosc.msmulu.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MsmuluBMmOp>>(
      converter, "llvm.riscv.bosc.msmulu.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MsmuluHMmOp>>(
      converter, "llvm.riscv.bosc.msmulu.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MsmuluWMmOp>>(
      converter, "llvm.riscv.bosc.msmulu.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MsmuluDwMmOp>>(
      converter, "llvm.riscv.bosc.msmulu.dw.mm");

  patterns.add<BOSCAMEMath3Lowering<MmulMmOp>>(converter,
                                               "llvm.riscv.bosc.mmul.mm");
  patterns.add<BOSCAMEMath3Lowering<MmulHbMmOp>>(converter,
                                                 "llvm.riscv.bosc.mmul.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MmulBMmOp>>(converter,
                                                "llvm.riscv.bosc.mmul.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MmulHMmOp>>(converter,
                                                "llvm.riscv.bosc.mmul.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MmulWMmOp>>(converter,
                                                "llvm.riscv.bosc.mmul.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MmulDwMmOp>>(converter,
                                                 "llvm.riscv.bosc.mmul.dw.mm");

  patterns.add<BOSCAMEMath3Lowering<MsmulMmOp>>(converter,
                                                "llvm.riscv.bosc.msmul.mm");
  patterns.add<BOSCAMEMath3Lowering<MsmulHbMmOp>>(
      converter, "llvm.riscv.bosc.msmul.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MsmulBMmOp>>(converter,
                                                 "llvm.riscv.bosc.msmul.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MsmulHMmOp>>(converter,
                                                 "llvm.riscv.bosc.msmul.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MsmulWMmOp>>(converter,
                                                 "llvm.riscv.bosc.msmul.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MsmulDwMmOp>>(
      converter, "llvm.riscv.bosc.msmul.dw.mm");

  patterns.add<BOSCAMEMath3Lowering<MfmulMmOp>>(converter,
                                                "llvm.riscv.bosc.mfmul.mm");
  patterns.add<BOSCAMEMath3Lowering<MfmulCfMmOp>>(
      converter, "llvm.riscv.bosc.mfmul.cf.mm");
  patterns.add<BOSCAMEMath3Lowering<MfmulHfMmOp>>(
      converter, "llvm.riscv.bosc.mfmul.hf.mm");
  patterns.add<BOSCAMEMath3Lowering<MfmulFMmOp>>(converter,
                                                 "llvm.riscv.bosc.mfmul.f.mm");
  patterns.add<BOSCAMEMath3Lowering<MfmulDMmOp>>(converter,
                                                 "llvm.riscv.bosc.mfmul.d.mm");

  patterns.add<BOSCAMEMath3Lowering<MmulhuMmOp>>(converter,
                                                 "llvm.riscv.bosc.mmulhu.mm");
  patterns.add<BOSCAMEMath3Lowering<MmulhuHbMmOp>>(
      converter, "llvm.riscv.bosc.mmulhu.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MmulhuBMmOp>>(
      converter, "llvm.riscv.bosc.mmulhu.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MmulhuHMmOp>>(
      converter, "llvm.riscv.bosc.mmulhu.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MmulhuWMmOp>>(
      converter, "llvm.riscv.bosc.mmulhu.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MmulhuDwMmOp>>(
      converter, "llvm.riscv.bosc.mmulhu.dw.mm");

  patterns.add<BOSCAMEMath3Lowering<MmulhMmOp>>(converter,
                                                "llvm.riscv.bosc.mmulh.mm");
  patterns.add<BOSCAMEMath3Lowering<MmulhHbMmOp>>(
      converter, "llvm.riscv.bosc.mmulh.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MmulhBMmOp>>(converter,
                                                 "llvm.riscv.bosc.mmulh.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MmulhHMmOp>>(converter,
                                                 "llvm.riscv.bosc.mmulh.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MmulhWMmOp>>(converter,
                                                 "llvm.riscv.bosc.mmulh.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MmulhDwMmOp>>(
      converter, "llvm.riscv.bosc.mmulh.dw.mm");

  patterns.add<BOSCAMEMath3Lowering<MmulhsuMmOp>>(converter,
                                                  "llvm.riscv.bosc.mmulhsu.mm");
  patterns.add<BOSCAMEMath3Lowering<MmulhsuHbMmOp>>(
      converter, "llvm.riscv.bosc.mmulhsu.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MmulhsuBMmOp>>(
      converter, "llvm.riscv.bosc.mmulhsu.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MmulhsuHMmOp>>(
      converter, "llvm.riscv.bosc.mmulhsu.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MmulhsuWMmOp>>(
      converter, "llvm.riscv.bosc.mmulhsu.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MmulhsuDwMmOp>>(
      converter, "llvm.riscv.bosc.mmulhsu.dw.mm");

  patterns.add<BOSCAMEMath3Lowering<MsmulsuMmOp>>(converter,
                                                  "llvm.riscv.bosc.msmulsu.mm");
  patterns.add<BOSCAMEMath3Lowering<MsmulsuHbMmOp>>(
      converter, "llvm.riscv.bosc.msmulsu.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MsmulsuBMmOp>>(
      converter, "llvm.riscv.bosc.msmulsu.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MsmulsuHMmOp>>(
      converter, "llvm.riscv.bosc.msmulsu.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MsmulsuWMmOp>>(
      converter, "llvm.riscv.bosc.msmulsu.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MsmulsuDwMmOp>>(
      converter, "llvm.riscv.bosc.msmulsu.dw.mm");

  patterns.add<BOSCAMEMath3Lowering<MfdivMmOp>>(converter,
                                                "llvm.riscv.bosc.mfdiv.mm");
  patterns.add<BOSCAMEMath3Lowering<MfdivCfMmOp>>(
      converter, "llvm.riscv.bosc.mfdiv.cf.mm");
  patterns.add<BOSCAMEMath3Lowering<MfdivHfMmOp>>(
      converter, "llvm.riscv.bosc.mfdiv.hf.mm");
  patterns.add<BOSCAMEMath3Lowering<MfdivFMmOp>>(converter,
                                                 "llvm.riscv.bosc.mfdiv.f.mm");
  patterns.add<BOSCAMEMath3Lowering<MfdivDMmOp>>(converter,
                                                 "llvm.riscv.bosc.mfdiv.d.mm");

  // 5. MUL - Double Widen
  patterns.add<BOSCAMEMath3Lowering<MwmuluMmOp>>(converter,
                                                 "llvm.riscv.bosc.mwmulu.mm");
  patterns.add<BOSCAMEMath3Lowering<MwmuluHbMmOp>>(
      converter, "llvm.riscv.bosc.mwmulu.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MwmuluBMmOp>>(
      converter, "llvm.riscv.bosc.mwmulu.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MwmuluHMmOp>>(
      converter, "llvm.riscv.bosc.mwmulu.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MwmuluWMmOp>>(
      converter, "llvm.riscv.bosc.mwmulu.w.mm");

  patterns.add<BOSCAMEMath3Lowering<MwmulMmOp>>(converter,
                                                "llvm.riscv.bosc.mwmul.mm");
  patterns.add<BOSCAMEMath3Lowering<MwmulHbMmOp>>(
      converter, "llvm.riscv.bosc.mwmul.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MwmulBMmOp>>(converter,
                                                 "llvm.riscv.bosc.mwmul.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MwmulHMmOp>>(converter,
                                                 "llvm.riscv.bosc.mwmul.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MwmulWMmOp>>(converter,
                                                 "llvm.riscv.bosc.mwmul.w.mm");

  patterns.add<BOSCAMEMath3Lowering<MwmulsuMmOp>>(converter,
                                                  "llvm.riscv.bosc.mwmulsu.mm");
  patterns.add<BOSCAMEMath3Lowering<MwmulsuHbMmOp>>(
      converter, "llvm.riscv.bosc.mwmulsu.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MwmulsuBMmOp>>(
      converter, "llvm.riscv.bosc.mwmulsu.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MwmulsuHMmOp>>(
      converter, "llvm.riscv.bosc.mwmulsu.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MwmulsuWMmOp>>(
      converter, "llvm.riscv.bosc.mwmulsu.w.mm");

  patterns.add<BOSCAMEMath3Lowering<MfwmulMmOp>>(converter,
                                                 "llvm.riscv.bosc.mfwmul.mm");
  patterns.add<BOSCAMEMath3Lowering<MfwmulCfMmOp>>(
      converter, "llvm.riscv.bosc.mfwmul.cf.mm");
  patterns.add<BOSCAMEMath3Lowering<MfwmulHfMmOp>>(
      converter, "llvm.riscv.bosc.mfwmul.hf.mm");
  patterns.add<BOSCAMEMath3Lowering<MfwmulFMmOp>>(
      converter, "llvm.riscv.bosc.mfwmul.f.mm");

  // 6. LOGIC & SHIFT - No Widen (Individual & Multiclass)
  patterns.add<BOSCAMEMath3Lowering<MandMmOp>>(converter,
                                               "llvm.riscv.bosc.mand.mm");
  patterns.add<BOSCAMEMath3Lowering<MorMmOp>>(converter,
                                              "llvm.riscv.bosc.mor.mm");
  patterns.add<BOSCAMEMath3Lowering<MxorMmOp>>(converter,
                                               "llvm.riscv.bosc.mxor.mm");

  patterns.add<BOSCAMEMath3Lowering<MsllMmOp>>(converter,
                                               "llvm.riscv.bosc.msll.mm");
  patterns.add<BOSCAMEMath3Lowering<MsllHbMmOp>>(converter,
                                                 "llvm.riscv.bosc.msll.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MsllBMmOp>>(converter,
                                                "llvm.riscv.bosc.msll.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MsllHMmOp>>(converter,
                                                "llvm.riscv.bosc.msll.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MsllWMmOp>>(converter,
                                                "llvm.riscv.bosc.msll.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MsllDwMmOp>>(converter,
                                                 "llvm.riscv.bosc.msll.dw.mm");

  patterns.add<BOSCAMEMath3Lowering<MsrlMmOp>>(converter,
                                               "llvm.riscv.bosc.msrl.mm");
  patterns.add<BOSCAMEMath3Lowering<MsrlHbMmOp>>(converter,
                                                 "llvm.riscv.bosc.msrl.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MsrlBMmOp>>(converter,
                                                "llvm.riscv.bosc.msrl.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MsrlHMmOp>>(converter,
                                                "llvm.riscv.bosc.msrl.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MsrlWMmOp>>(converter,
                                                "llvm.riscv.bosc.msrl.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MsrlDwMmOp>>(converter,
                                                 "llvm.riscv.bosc.msrl.dw.mm");

  patterns.add<BOSCAMEMath3Lowering<MsraMmOp>>(converter,
                                               "llvm.riscv.bosc.msra.mm");
  patterns.add<BOSCAMEMath3Lowering<MsraHbMmOp>>(converter,
                                                 "llvm.riscv.bosc.msra.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MsraBMmOp>>(converter,
                                                "llvm.riscv.bosc.msra.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MsraHMmOp>>(converter,
                                                "llvm.riscv.bosc.msra.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MsraWMmOp>>(converter,
                                                "llvm.riscv.bosc.msra.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MsraDwMmOp>>(converter,
                                                 "llvm.riscv.bosc.msra.dw.mm");

  // 7. SQRT - No Widen (Unary, 2-Operands)
  patterns.add<BOSCAMEMath2Lowering<MfsqrtMOp>>(converter,
                                                "llvm.riscv.bosc.mfsqrt.m");
  patterns.add<BOSCAMEMath2Lowering<MfsqrtCfMOp>>(
      converter, "llvm.riscv.bosc.mfsqrt.cf.m");
  patterns.add<BOSCAMEMath2Lowering<MfsqrtHfMOp>>(
      converter, "llvm.riscv.bosc.mfsqrt.hf.m");
  patterns.add<BOSCAMEMath2Lowering<MfsqrtFMOp>>(converter,
                                                 "llvm.riscv.bosc.mfsqrt.f.m");
  patterns.add<BOSCAMEMath2Lowering<MfsqrtDMOp>>(converter,
                                                 "llvm.riscv.bosc.mfsqrt.d.m");

  // Load/Store patterns
  // Matrix A
  patterns.add<BOSCAMELoadLowering<Mlae8mOp>>(converter,
                                              "llvm.riscv.bosc.mlae8.m");
  patterns.add<BOSCAMELoadLowering<Mlae16mOp>>(converter,
                                               "llvm.riscv.bosc.mlae16.m");
  patterns.add<BOSCAMELoadLowering<Mlae32mOp>>(converter,
                                               "llvm.riscv.bosc.mlae32.m");
  patterns.add<BOSCAMELoadLowering<Mlae64mOp>>(converter,
                                               "llvm.riscv.bosc.mlae64.m");

  patterns.add<BOSCAMELoadLowering<Mlate8mOp>>(converter,
                                               "llvm.riscv.bosc.mlate8.m");
  patterns.add<BOSCAMELoadLowering<Mlate16mOp>>(converter,
                                                "llvm.riscv.bosc.mlate16.m");
  patterns.add<BOSCAMELoadLowering<Mlate32mOp>>(converter,
                                                "llvm.riscv.bosc.mlate32.m");
  patterns.add<BOSCAMELoadLowering<Mlate64mOp>>(converter,
                                                "llvm.riscv.bosc.mlate64.m");

  patterns.add<BOSCAMELoadLowering<Mltre8mOp>>(converter,
                                               "llvm.riscv.bosc.mltre8.m");
  patterns.add<BOSCAMELoadLowering<Mltre16mOp>>(converter,
                                                "llvm.riscv.bosc.mltre16.m");
  patterns.add<BOSCAMELoadLowering<Mltre32mOp>>(converter,
                                                "llvm.riscv.bosc.mltre32.m");
  patterns.add<BOSCAMELoadLowering<Mltre64mOp>>(converter,
                                                "llvm.riscv.bosc.mltre64.m");

  // Matrix B
  patterns.add<BOSCAMELoadLowering<Mlbe8mOp>>(converter,
                                              "llvm.riscv.bosc.mlbe8.m");
  patterns.add<BOSCAMELoadLowering<Mlbe16mOp>>(converter,
                                               "llvm.riscv.bosc.mlbe16.m");
  patterns.add<BOSCAMELoadLowering<Mlbe32mOp>>(converter,
                                               "llvm.riscv.bosc.mlbe32.m");
  patterns.add<BOSCAMELoadLowering<Mlbe64mOp>>(converter,
                                               "llvm.riscv.bosc.mlbe64.m");

  patterns.add<BOSCAMELoadLowering<Mlbte8mOp>>(converter,
                                               "llvm.riscv.bosc.mlbte8.m");
  patterns.add<BOSCAMELoadLowering<Mlbte16mOp>>(converter,
                                                "llvm.riscv.bosc.mlbte16.m");
  patterns.add<BOSCAMELoadLowering<Mlbte32mOp>>(converter,
                                                "llvm.riscv.bosc.mlbte32.m");
  patterns.add<BOSCAMELoadLowering<Mlbte64mOp>>(converter,
                                                "llvm.riscv.bosc.mlbte64.m");

  // Matrix C
  patterns.add<BOSCAMELoadLowering<Mlce8mOp>>(converter,
                                              "llvm.riscv.bosc.mlce8.m");
  patterns.add<BOSCAMELoadLowering<Mlce16mOp>>(converter,
                                               "llvm.riscv.bosc.mlce16.m");
  patterns.add<BOSCAMELoadLowering<Mlce32mOp>>(converter,
                                               "llvm.riscv.bosc.mlce32.m");
  patterns.add<BOSCAMELoadLowering<Mlce64mOp>>(converter,
                                               "llvm.riscv.bosc.mlce64.m");

  patterns.add<BOSCAMELoadLowering<Mlcte8mOp>>(converter,
                                               "llvm.riscv.bosc.mlcte8.m");
  patterns.add<BOSCAMELoadLowering<Mlcte16mOp>>(converter,
                                                "llvm.riscv.bosc.mlcte16.m");
  patterns.add<BOSCAMELoadLowering<Mlcte32mOp>>(converter,
                                                "llvm.riscv.bosc.mlcte32.m");
  patterns.add<BOSCAMELoadLowering<Mlcte64mOp>>(converter,
                                                "llvm.riscv.bosc.mlcte64.m");

  patterns.add<BOSCAMELoadLowering<Mlacce8mOp>>(converter,
                                                "llvm.riscv.bosc.mlacce8.m");
  patterns.add<BOSCAMELoadLowering<Mlacce16mOp>>(converter,
                                                 "llvm.riscv.bosc.mlacce16.m");
  patterns.add<BOSCAMELoadLowering<Mlacce32mOp>>(converter,
                                                 "llvm.riscv.bosc.mlacce32.m");
  patterns.add<BOSCAMELoadLowering<Mlacce64mOp>>(converter,
                                                 "llvm.riscv.bosc.mlacce64.m");

  //===--------------------------------------------------------------------===//
  // Store Patterns
  //===--------------------------------------------------------------------===//

  // Matrix A
  patterns.add<BOSCAMEStoreLowering<Msae8mOp>>(converter,
                                               "llvm.riscv.bosc.msae8.m");
  patterns.add<BOSCAMEStoreLowering<Msae16mOp>>(converter,
                                                "llvm.riscv.bosc.msae16.m");
  patterns.add<BOSCAMEStoreLowering<Msae32mOp>>(converter,
                                                "llvm.riscv.bosc.msae32.m");
  patterns.add<BOSCAMEStoreLowering<Msae64mOp>>(converter,
                                                "llvm.riscv.bosc.msae64.m");

  patterns.add<BOSCAMEStoreLowering<Msate8mOp>>(converter,
                                                "llvm.riscv.bosc.msate8.m");
  patterns.add<BOSCAMEStoreLowering<Msate16mOp>>(converter,
                                                 "llvm.riscv.bosc.msate16.m");
  patterns.add<BOSCAMEStoreLowering<Msate32mOp>>(converter,
                                                 "llvm.riscv.bosc.msate32.m");
  patterns.add<BOSCAMEStoreLowering<Msate64mOp>>(converter,
                                                 "llvm.riscv.bosc.msate64.m");

  patterns.add<BOSCAMEStoreLowering<Mstre8mOp>>(converter,
                                                "llvm.riscv.bosc.mstre8.m");
  patterns.add<BOSCAMEStoreLowering<Mstre16mOp>>(converter,
                                                 "llvm.riscv.bosc.mstre16.m");
  patterns.add<BOSCAMEStoreLowering<Mstre32mOp>>(converter,
                                                 "llvm.riscv.bosc.mstre32.m");
  patterns.add<BOSCAMEStoreLowering<Mstre64mOp>>(converter,
                                                 "llvm.riscv.bosc.mstre64.m");

  // Matrix B
  patterns.add<BOSCAMEStoreLowering<Msbe8mOp>>(converter,
                                               "llvm.riscv.bosc.msbe8.m");
  patterns.add<BOSCAMEStoreLowering<Msbe16mOp>>(converter,
                                                "llvm.riscv.bosc.msbe16.m");
  patterns.add<BOSCAMEStoreLowering<Msbe32mOp>>(converter,
                                                "llvm.riscv.bosc.msbe32.m");
  patterns.add<BOSCAMEStoreLowering<Msbe64mOp>>(converter,
                                                "llvm.riscv.bosc.msbe64.m");

  patterns.add<BOSCAMEStoreLowering<Msbte8mOp>>(converter,
                                                "llvm.riscv.bosc.msbte8.m");
  patterns.add<BOSCAMEStoreLowering<Msbte16mOp>>(converter,
                                                 "llvm.riscv.bosc.msbte16.m");
  patterns.add<BOSCAMEStoreLowering<Msbte32mOp>>(converter,
                                                 "llvm.riscv.bosc.msbte32.m");
  patterns.add<BOSCAMEStoreLowering<Msbte64mOp>>(converter,
                                                 "llvm.riscv.bosc.msbte64.m");

  // Matrix C
  patterns.add<BOSCAMEStoreLowering<Msce8mOp>>(converter,
                                               "llvm.riscv.bosc.msce8.m");
  patterns.add<BOSCAMEStoreLowering<Msce16mOp>>(converter,
                                                "llvm.riscv.bosc.msce16.m");
  patterns.add<BOSCAMEStoreLowering<Msce32mOp>>(converter,
                                                "llvm.riscv.bosc.msce32.m");
  patterns.add<BOSCAMEStoreLowering<Msce64mOp>>(converter,
                                                "llvm.riscv.bosc.msce64.m");

  patterns.add<BOSCAMEStoreLowering<Mscte8mOp>>(converter,
                                                "llvm.riscv.bosc.mscte8.m");
  patterns.add<BOSCAMEStoreLowering<Mscte16mOp>>(converter,
                                                 "llvm.riscv.bosc.mscte16.m");
  patterns.add<BOSCAMEStoreLowering<Mscte32mOp>>(converter,
                                                 "llvm.riscv.bosc.mscte32.m");
  patterns.add<BOSCAMEStoreLowering<Mscte64mOp>>(converter,
                                                 "llvm.riscv.bosc.mscte64.m");

  patterns.add<BOSCAMEStoreLowering<Msacce8mOp>>(converter,
                                                 "llvm.riscv.bosc.msacce8.m");
  patterns.add<BOSCAMEStoreLowering<Msacce16mOp>>(converter,
                                                  "llvm.riscv.bosc.msacce16.m");
  patterns.add<BOSCAMEStoreLowering<Msacce32mOp>>(converter,
                                                  "llvm.riscv.bosc.msacce32.m");
  patterns.add<BOSCAMEStoreLowering<Msacce64mOp>>(converter,
                                                  "llvm.riscv.bosc.msacce64.m");

  // Data Move / Broadcast / Transpose Patterns
  // 1. Pure Move (Tile-Tile & Acc-Acc)
  patterns.add<BOSCAMEPureMoveLowering<MmveTt8Op>>(converter,
                                                   "llvm.riscv.bosc.mmve8.t.t");
  patterns.add<BOSCAMEPureMoveLowering<MmveTt16Op>>(
      converter, "llvm.riscv.bosc.mmve16.t.t");
  patterns.add<BOSCAMEPureMoveLowering<MmveTt32Op>>(
      converter, "llvm.riscv.bosc.mmve32.t.t");
  patterns.add<BOSCAMEPureMoveLowering<MmveTt64Op>>(
      converter, "llvm.riscv.bosc.mmve64.t.t");

  patterns.add<BOSCAMEPureMoveLowering<MmveAa8Op>>(converter,
                                                   "llvm.riscv.bosc.mmve8.a.a");
  patterns.add<BOSCAMEPureMoveLowering<MmveAa16Op>>(
      converter, "llvm.riscv.bosc.mmve16.a.a");
  patterns.add<BOSCAMEPureMoveLowering<MmveAa32Op>>(
      converter, "llvm.riscv.bosc.mmve32.a.a");
  patterns.add<BOSCAMEPureMoveLowering<MmveAa64Op>>(
      converter, "llvm.riscv.bosc.mmve64.a.a");

  // 2. GPR Indexed Move
  patterns.add<BOSCAMEIndexedMoveLowering<MmveAt8Op>>(
      converter, "llvm.riscv.bosc.mmve8.a.t");
  patterns.add<BOSCAMEIndexedMoveLowering<MmveAt16Op>>(
      converter, "llvm.riscv.bosc.mmve16.a.t");
  patterns.add<BOSCAMEIndexedMoveLowering<MmveAt32Op>>(
      converter, "llvm.riscv.bosc.mmve32.a.t");
  patterns.add<BOSCAMEIndexedMoveLowering<MmveAt64Op>>(
      converter, "llvm.riscv.bosc.mmve64.a.t");

  patterns.add<BOSCAMEIndexedMoveLowering<MmveTa8Op>>(
      converter, "llvm.riscv.bosc.mmve8.t.a");
  patterns.add<BOSCAMEIndexedMoveLowering<MmveTa16Op>>(
      converter, "llvm.riscv.bosc.mmve16.t.a");
  patterns.add<BOSCAMEIndexedMoveLowering<MmveTa32Op>>(
      converter, "llvm.riscv.bosc.mmve32.t.a");
  patterns.add<BOSCAMEIndexedMoveLowering<MmveTa64Op>>(
      converter, "llvm.riscv.bosc.mmve64.t.a");

  // 3. Immediate Indexed Move
  patterns.add<BOSCAMEImmIndexedMoveLowering<MmvieAt8Op, &MmvieAt8Op::getImm>>(
      converter, "llvm.riscv.bosc.mmvie8.a.t");
  patterns
      .add<BOSCAMEImmIndexedMoveLowering<MmvieAt16Op, &MmvieAt16Op::getImm>>(
          converter, "llvm.riscv.bosc.mmvie16.a.t");
  patterns
      .add<BOSCAMEImmIndexedMoveLowering<MmvieAt32Op, &MmvieAt32Op::getImm>>(
          converter, "llvm.riscv.bosc.mmvie32.a.t");
  patterns
      .add<BOSCAMEImmIndexedMoveLowering<MmvieAt64Op, &MmvieAt64Op::getImm>>(
          converter, "llvm.riscv.bosc.mmvie64.a.t");

  patterns.add<BOSCAMEImmIndexedMoveLowering<MmvieTa8Op, &MmvieTa8Op::getImm>>(
      converter, "llvm.riscv.bosc.mmvie8.t.a");
  patterns
      .add<BOSCAMEImmIndexedMoveLowering<MmvieTa16Op, &MmvieTa16Op::getImm>>(
          converter, "llvm.riscv.bosc.mmvie16.t.a");
  patterns
      .add<BOSCAMEImmIndexedMoveLowering<MmvieTa32Op, &MmvieTa32Op::getImm>>(
          converter, "llvm.riscv.bosc.mmvie32.t.a");
  patterns
      .add<BOSCAMEImmIndexedMoveLowering<MmvieTa64Op, &MmvieTa64Op::getImm>>(
          converter, "llvm.riscv.bosc.mmvie64.t.a");

  // 4. Matrix to Scalar Move
  patterns.add<BOSCAMEMatrixToScalarLowering<MmveXt8Op>>(
      converter, "llvm.riscv.bosc.mmve8.x.t");
  patterns.add<BOSCAMEMatrixToScalarLowering<MmveXt16Op>>(
      converter, "llvm.riscv.bosc.mmve16.x.t");
  patterns.add<BOSCAMEMatrixToScalarLowering<MmveXt32Op>>(
      converter, "llvm.riscv.bosc.mmve32.x.t");
  patterns.add<BOSCAMEMatrixToScalarLowering<MmveXt64Op>>(
      converter, "llvm.riscv.bosc.mmve64.x.t");

  patterns.add<BOSCAMEMatrixToScalarLowering<MmveXa8Op>>(
      converter, "llvm.riscv.bosc.mmve8.x.a");
  patterns.add<BOSCAMEMatrixToScalarLowering<MmveXa16Op>>(
      converter, "llvm.riscv.bosc.mmve16.x.a");
  patterns.add<BOSCAMEMatrixToScalarLowering<MmveXa32Op>>(
      converter, "llvm.riscv.bosc.mmve32.x.a");
  patterns.add<BOSCAMEMatrixToScalarLowering<MmveXa64Op>>(
      converter, "llvm.riscv.bosc.mmve64.x.a");

  patterns.add<BOSCAMEMatrixToScalarLowering<MfmveFt8Op>>(
      converter, "llvm.riscv.bosc.mfmve8.f.t");
  patterns.add<BOSCAMEMatrixToScalarLowering<MfmveFt16Op>>(
      converter, "llvm.riscv.bosc.mfmve16.f.t");
  patterns.add<BOSCAMEMatrixToScalarLowering<MfmveFt32Op>>(
      converter, "llvm.riscv.bosc.mfmve32.f.t");
  patterns.add<BOSCAMEMatrixToScalarLowering<MfmveFt64Op>>(
      converter, "llvm.riscv.bosc.mfmve64.f.t");

  patterns.add<BOSCAMEMatrixToScalarLowering<MfmveFa8Op>>(
      converter, "llvm.riscv.bosc.mfmve8.f.a");
  patterns.add<BOSCAMEMatrixToScalarLowering<MfmveFa16Op>>(
      converter, "llvm.riscv.bosc.mfmve16.f.a");
  patterns.add<BOSCAMEMatrixToScalarLowering<MfmveFa32Op>>(
      converter, "llvm.riscv.bosc.mfmve32.f.a");
  patterns.add<BOSCAMEMatrixToScalarLowering<MfmveFa64Op>>(
      converter, "llvm.riscv.bosc.mfmve64.f.a");

  // 5. Scalar GPR to Matrix Move
  patterns.add<BOSCAMEScalarToMatrixLowering<MmveTx8Op>>(
      converter, "llvm.riscv.bosc.mmve8.t.x");
  patterns.add<BOSCAMEScalarToMatrixLowering<MmveTx16Op>>(
      converter, "llvm.riscv.bosc.mmve16.t.x");
  patterns.add<BOSCAMEScalarToMatrixLowering<MmveTx32Op>>(
      converter, "llvm.riscv.bosc.mmve32.t.x");
  patterns.add<BOSCAMEScalarToMatrixLowering<MmveTx64Op>>(
      converter, "llvm.riscv.bosc.mmve64.t.x");

  patterns.add<BOSCAMEScalarToMatrixLowering<MmveAx8Op>>(
      converter, "llvm.riscv.bosc.mmve8.a.x");
  patterns.add<BOSCAMEScalarToMatrixLowering<MmveAx16Op>>(
      converter, "llvm.riscv.bosc.mmve16.a.x");
  patterns.add<BOSCAMEScalarToMatrixLowering<MmveAx32Op>>(
      converter, "llvm.riscv.bosc.mmve32.a.x");
  patterns.add<BOSCAMEScalarToMatrixLowering<MmveAx64Op>>(
      converter, "llvm.riscv.bosc.mmve64.a.x");

  patterns.add<BOSCAMEScalarToMatrixLowering<MfmveTf8Op>>(
      converter, "llvm.riscv.bosc.mfmve8.t.f");
  patterns.add<BOSCAMEScalarToMatrixLowering<MfmveTf16Op>>(
      converter, "llvm.riscv.bosc.mfmve16.t.f");
  patterns.add<BOSCAMEScalarToMatrixLowering<MfmveTf32Op>>(
      converter, "llvm.riscv.bosc.mfmve32.t.f");
  patterns.add<BOSCAMEScalarToMatrixLowering<MfmveTf64Op>>(
      converter, "llvm.riscv.bosc.mfmve64.t.f");

  patterns.add<BOSCAMEScalarToMatrixLowering<MfmveAf8Op>>(
      converter, "llvm.riscv.bosc.mfmve8.a.f");
  patterns.add<BOSCAMEScalarToMatrixLowering<MfmveAf16Op>>(
      converter, "llvm.riscv.bosc.mfmve16.a.f");
  patterns.add<BOSCAMEScalarToMatrixLowering<MfmveAf32Op>>(
      converter, "llvm.riscv.bosc.mfmve32.a.f");
  patterns.add<BOSCAMEScalarToMatrixLowering<MfmveAf64Op>>(
      converter, "llvm.riscv.bosc.mfmve64.a.f");

  // 6. Broadcast & Transpose Patterns
  // Broadcast
  patterns.add<BOSCAMEPureMoveLowering<MbcarMOp>>(converter,
                                                  "llvm.riscv.bosc.mbcar.m");
  patterns.add<BOSCAMEPureMoveLowering<MbcbrMOp>>(converter,
                                                  "llvm.riscv.bosc.mbcbr.m");
  patterns.add<BOSCAMEPureMoveLowering<MbccrMOp>>(converter,
                                                  "llvm.riscv.bosc.mbccr.m");

  // Broadcast Col & Elem (32-bit)
  patterns.add<BOSCAMEPureMoveLowering<Mbcace8Op>>(converter,
                                                   "llvm.riscv.bosc.mbcace8.m");
  patterns.add<BOSCAMEPureMoveLowering<Mbcace16Op>>(
      converter, "llvm.riscv.bosc.mbcace16.m");
  patterns.add<BOSCAMEPureMoveLowering<Mbcace32Op>>(
      converter, "llvm.riscv.bosc.mbcace32.m");
  patterns.add<BOSCAMEPureMoveLowering<Mbcace64Op>>(
      converter, "llvm.riscv.bosc.mbcace64.m");

  patterns.add<BOSCAMEPureMoveLowering<Mbcbce8Op>>(converter,
                                                   "llvm.riscv.bosc.mbcbce8.m");
  patterns.add<BOSCAMEPureMoveLowering<Mbcbce16Op>>(
      converter, "llvm.riscv.bosc.mbcbce16.m");
  patterns.add<BOSCAMEPureMoveLowering<Mbcbce32Op>>(
      converter, "llvm.riscv.bosc.mbcbce32.m");
  patterns.add<BOSCAMEPureMoveLowering<Mbcbce64Op>>(
      converter, "llvm.riscv.bosc.mbcbce64.m");

  patterns.add<BOSCAMEPureMoveLowering<Mbccce8Op>>(converter,
                                                   "llvm.riscv.bosc.mbccce8.m");
  patterns.add<BOSCAMEPureMoveLowering<Mbccce16Op>>(
      converter, "llvm.riscv.bosc.mbccce16.m");
  patterns.add<BOSCAMEPureMoveLowering<Mbccce32Op>>(
      converter, "llvm.riscv.bosc.mbccce32.m");
  patterns.add<BOSCAMEPureMoveLowering<Mbccce64Op>>(
      converter, "llvm.riscv.bosc.mbccce64.m");

  patterns.add<BOSCAMEPureMoveLowering<Mbcaee8Op>>(converter,
                                                   "llvm.riscv.bosc.mbcaee8.m");
  patterns.add<BOSCAMEPureMoveLowering<Mbcaee16Op>>(
      converter, "llvm.riscv.bosc.mbcaee16.m");
  patterns.add<BOSCAMEPureMoveLowering<Mbcaee32Op>>(
      converter, "llvm.riscv.bosc.mbcaee32.m");
  patterns.add<BOSCAMEPureMoveLowering<Mbcaee64Op>>(
      converter, "llvm.riscv.bosc.mbcaee64.m");

  patterns.add<BOSCAMEPureMoveLowering<Mbcbee8Op>>(converter,
                                                   "llvm.riscv.bosc.mbcbee8.m");
  patterns.add<BOSCAMEPureMoveLowering<Mbcbee16Op>>(
      converter, "llvm.riscv.bosc.mbcbee16.m");
  patterns.add<BOSCAMEPureMoveLowering<Mbcbee32Op>>(
      converter, "llvm.riscv.bosc.mbcbee32.m");
  patterns.add<BOSCAMEPureMoveLowering<Mbcbee64Op>>(
      converter, "llvm.riscv.bosc.mbcbee64.m");

  patterns.add<BOSCAMEPureMoveLowering<Mbccee8Op>>(converter,
                                                   "llvm.riscv.bosc.mbccee8.m");
  patterns.add<BOSCAMEPureMoveLowering<Mbccee16Op>>(
      converter, "llvm.riscv.bosc.mbccee16.m");
  patterns.add<BOSCAMEPureMoveLowering<Mbccee32Op>>(
      converter, "llvm.riscv.bosc.mbccee32.m");
  patterns.add<BOSCAMEPureMoveLowering<Mbccee64Op>>(
      converter, "llvm.riscv.bosc.mbccee64.m");

  // Transpose (32-bit)
  patterns.add<BOSCAMEPureMoveLowering<Mtae8Op>>(converter,
                                                 "llvm.riscv.bosc.mtae8.m");
  patterns.add<BOSCAMEPureMoveLowering<Mtae16Op>>(converter,
                                                  "llvm.riscv.bosc.mtae16.m");
  patterns.add<BOSCAMEPureMoveLowering<Mtae32Op>>(converter,
                                                  "llvm.riscv.bosc.mtae32.m");
  patterns.add<BOSCAMEPureMoveLowering<Mtae64Op>>(converter,
                                                  "llvm.riscv.bosc.mtae64.m");

  patterns.add<BOSCAMEPureMoveLowering<Mtbe8Op>>(converter,
                                                 "llvm.riscv.bosc.mtbe8.m");
  patterns.add<BOSCAMEPureMoveLowering<Mtbe16Op>>(converter,
                                                  "llvm.riscv.bosc.mtbe16.m");
  patterns.add<BOSCAMEPureMoveLowering<Mtbe32Op>>(converter,
                                                  "llvm.riscv.bosc.mtbe32.m");
  patterns.add<BOSCAMEPureMoveLowering<Mtbe64Op>>(converter,
                                                  "llvm.riscv.bosc.mtbe64.m");

  patterns.add<BOSCAMEPureMoveLowering<Mtce8Op>>(converter,
                                                 "llvm.riscv.bosc.mtce8.m");
  patterns.add<BOSCAMEPureMoveLowering<Mtce16Op>>(converter,
                                                  "llvm.riscv.bosc.mtce16.m");
  patterns.add<BOSCAMEPureMoveLowering<Mtce32Op>>(converter,
                                                  "llvm.riscv.bosc.mtce32.m");
  patterns.add<BOSCAMEPureMoveLowering<Mtce64Op>>(converter,
                                                  "llvm.riscv.bosc.mtce64.m");

  // Tile register matrix multiply patterns
  // 1. No-widen matrix multiply-accumulate
  patterns.add<BOSCAMEMath3Lowering<MmaMmOp>>(converter,
                                              "llvm.riscv.bosc.mma.mm");
  patterns.add<BOSCAMEMath3Lowering<MmaHmmOp>>(converter,
                                               "llvm.riscv.bosc.mma.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MmaWmmOp>>(converter,
                                               "llvm.riscv.bosc.mma.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MmaDwmmOp>>(converter,
                                                "llvm.riscv.bosc.mma.dw.mm");

  patterns.add<BOSCAMEMath3Lowering<MsmaMmOp>>(converter,
                                               "llvm.riscv.bosc.msma.mm");
  patterns.add<BOSCAMEMath3Lowering<MsmaHmmOp>>(converter,
                                                "llvm.riscv.bosc.msma.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MsmaWmmOp>>(converter,
                                                "llvm.riscv.bosc.msma.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MsmaDwmmOp>>(converter,
                                                 "llvm.riscv.bosc.msma.dw.mm");

  patterns.add<BOSCAMEMath3Lowering<MmauMmOp>>(converter,
                                               "llvm.riscv.bosc.mmau.mm");
  patterns.add<BOSCAMEMath3Lowering<MmauHmmOp>>(converter,
                                                "llvm.riscv.bosc.mmau.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MmauWmmOp>>(converter,
                                                "llvm.riscv.bosc.mmau.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MmauDwmmOp>>(converter,
                                                 "llvm.riscv.bosc.mmau.dw.mm");

  patterns.add<BOSCAMEMath3Lowering<MsmauMmOp>>(converter,
                                                "llvm.riscv.bosc.msmau.mm");
  patterns.add<BOSCAMEMath3Lowering<MsmauHmmOp>>(converter,
                                                 "llvm.riscv.bosc.msmau.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MsmauWmmOp>>(converter,
                                                 "llvm.riscv.bosc.msmau.w.mm");
  patterns.add<BOSCAMEMath3Lowering<MsmauDwmmOp>>(
      converter, "llvm.riscv.bosc.msmau.dw.mm");

  patterns.add<BOSCAMEMath3Lowering<MfmaMmOp>>(converter,
                                               "llvm.riscv.bosc.mfma.mm");
  patterns.add<BOSCAMEMath3Lowering<MfmaHfmmOp>>(converter,
                                                 "llvm.riscv.bosc.mfma.hf.mm");
  patterns.add<BOSCAMEMath3Lowering<MfmaFmmOp>>(converter,
                                                "llvm.riscv.bosc.mfma.f.mm");
  patterns.add<BOSCAMEMath3Lowering<MfmaDmmOp>>(converter,
                                                "llvm.riscv.bosc.mfma.d.mm");

  // 2. Double-widen matrix multiply-accumulate
  patterns.add<BOSCAMEMath3Lowering<MwmauMmOp>>(converter,
                                                "llvm.riscv.bosc.mwmau.mm");
  patterns.add<BOSCAMEMath3Lowering<MwmauHmmOp>>(converter,
                                                 "llvm.riscv.bosc.mwmau.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MwmauWmmOp>>(converter,
                                                 "llvm.riscv.bosc.mwmau.w.mm");

  patterns.add<BOSCAMEMath3Lowering<MswmauMmOp>>(converter,
                                                 "llvm.riscv.bosc.mswmau.mm");
  patterns.add<BOSCAMEMath3Lowering<MswmauHmmOp>>(
      converter, "llvm.riscv.bosc.mswmau.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MswmauWmmOp>>(
      converter, "llvm.riscv.bosc.mswmau.w.mm");

  patterns.add<BOSCAMEMath3Lowering<MwmaMmOp>>(converter,
                                               "llvm.riscv.bosc.mwma.mm");
  patterns.add<BOSCAMEMath3Lowering<MwmaHmmOp>>(converter,
                                                "llvm.riscv.bosc.mwma.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MwmaWmmOp>>(converter,
                                                "llvm.riscv.bosc.mwma.w.mm");

  patterns.add<BOSCAMEMath3Lowering<MswmaMmOp>>(converter,
                                                "llvm.riscv.bosc.mswma.mm");
  patterns.add<BOSCAMEMath3Lowering<MswmaHmmOp>>(converter,
                                                 "llvm.riscv.bosc.mswma.h.mm");
  patterns.add<BOSCAMEMath3Lowering<MswmaWmmOp>>(converter,
                                                 "llvm.riscv.bosc.mswma.w.mm");

  patterns.add<BOSCAMEMath3Lowering<MfwmaMmOp>>(converter,
                                                "llvm.riscv.bosc.mfwma.mm");
  patterns.add<BOSCAMEMath3Lowering<MfwmaCfmmOp>>(
      converter, "llvm.riscv.bosc.mfwma.cf.mm");
  patterns.add<BOSCAMEMath3Lowering<MfwmaHfmmOp>>(
      converter, "llvm.riscv.bosc.mfwma.hf.mm");
  patterns.add<BOSCAMEMath3Lowering<MfwmaFmmOp>>(converter,
                                                 "llvm.riscv.bosc.mfwma.f.mm");

  // 3. Quad-widen matrix multiply-accumulate
  patterns.add<BOSCAMEMath3Lowering<MqmauMmOp>>(converter,
                                                "llvm.riscv.bosc.mqmau.mm");
  patterns.add<BOSCAMEMath3Lowering<MqmauBmmOp>>(converter,
                                                 "llvm.riscv.bosc.mqmau.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MsqmauMmOp>>(converter,
                                                 "llvm.riscv.bosc.msqmau.mm");
  patterns.add<BOSCAMEMath3Lowering<MsqmauBmmOp>>(
      converter, "llvm.riscv.bosc.msqmau.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MqmaMmOp>>(converter,
                                               "llvm.riscv.bosc.mqma.mm");
  patterns.add<BOSCAMEMath3Lowering<MqmaBmmOp>>(converter,
                                                "llvm.riscv.bosc.mqma.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MsqmaMmOp>>(converter,
                                                "llvm.riscv.bosc.msqma.mm");
  patterns.add<BOSCAMEMath3Lowering<MsqmaBmmOp>>(converter,
                                                 "llvm.riscv.bosc.msqma.b.mm");
  patterns.add<BOSCAMEMath3Lowering<MfqmaMmOp>>(converter,
                                                "llvm.riscv.bosc.mfqma.mm");
  patterns.add<BOSCAMEMath3Lowering<MfqmaCfmmOp>>(
      converter, "llvm.riscv.bosc.mfqma.cf.mm");

  // 4. Oct-widen matrix multiply-accumulate
  patterns.add<BOSCAMEMath3Lowering<MomauMmOp>>(converter,
                                                "llvm.riscv.bosc.momau.mm");
  patterns.add<BOSCAMEMath3Lowering<MomauHbmmOp>>(
      converter, "llvm.riscv.bosc.momau.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MsomauMmOp>>(converter,
                                                 "llvm.riscv.bosc.msomau.mm");
  patterns.add<BOSCAMEMath3Lowering<MsomauHbmmOp>>(
      converter, "llvm.riscv.bosc.msomau.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MomaMmOp>>(converter,
                                               "llvm.riscv.bosc.moma.mm");
  patterns.add<BOSCAMEMath3Lowering<MomaHbmmOp>>(converter,
                                                 "llvm.riscv.bosc.moma.hb.mm");
  patterns.add<BOSCAMEMath3Lowering<MsomaMmOp>>(converter,
                                                "llvm.riscv.bosc.msoma.mm");
  patterns.add<BOSCAMEMath3Lowering<MsomaHbmmOp>>(
      converter, "llvm.riscv.bosc.msoma.hb.mm");
}

void mlir::configureBOSCAMELegalizeForExportTarget(
    LLVMConversionTarget &target) {
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<memref::MemRefDialect>();

  target.addIllegalDialect<buddy::boscame::BOSCAMEDialect>();
}

std::unique_ptr<Pass> buddy::boscame::createLegalizeForLLVMExportPass() {
  return std::make_unique<LegalizeBOSCAMEForLLVMExport>();
}

namespace mlir {
namespace buddy {
void registerLowerBOSCAMEPass() {
  PassRegistration<LegalizeBOSCAMEForLLVMExport>();
}
} // namespace buddy
} // namespace mlir
