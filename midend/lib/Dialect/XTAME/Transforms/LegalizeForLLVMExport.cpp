//====- LegalizeForLLVMExport.cpp - Prepare XTAME for LLVM translation ----===//
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

#include "Dialect/XTAME/Transform.h"
#include "Dialect/XTAME/XTAMEDialect.h"
#include "Dialect/XTAME/XTAMEOps.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace buddy::xtame;

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
// XTAME Lowering Patterns
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Configuration Operations Lowering
//===----------------------------------------------------------------------===//
template <typename OpTy>
struct XTAMEConfigLowering : public ConvertOpToLLVMPattern<OpTy> {
  StringRef intrinsicName;

  XTAMEConfigLowering(LLVMTypeConverter &typeConverter, StringRef intrinsicName)
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
    auto funcType =
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {i64Type});
    auto intrinsicNameSym =
        getOrInsertIntrinsic(rewriter, module, intrinsicName, funcType);

    Value configVal = adaptor.getOperands()[0];

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, intrinsicNameSym,
                         ValueRange{configVal});
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename OpTy, uint64_t (OpTy::*AttrGetter)()>
struct XTAMEConfigImmLowering : public ConvertOpToLLVMPattern<OpTy> {
  StringRef intrinsicName;
  XTAMEConfigImmLowering(LLVMTypeConverter &typeConverter,
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
    auto funcType =
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {i64Type});
    auto intrinsicNameSym =
        getOrInsertIntrinsic(rewriter, module, intrinsicName, funcType);

    uint64_t attrVal = (op.*AttrGetter)();
    Value val = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                         rewriter.getI64IntegerAttr(attrVal));

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, intrinsicNameSym,
                         ValueRange{val});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for mzero
template <typename OpTy>
struct XTAMEZeroLowering : public ConvertOpToLLVMPattern<OpTy> {
  StringRef intrinsicName;
  XTAMEZeroLowering(LLVMTypeConverter &typeConverter, StringRef intrinsicName)
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
    auto funcType =
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {i64Type});
    auto intrinsicNameSym =
        getOrInsertIntrinsic(rewriter, module, intrinsicName, funcType);

    Value mdVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, intrinsicNameSym,
                         ValueRange{mdVal});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Data Move Instructions between Matrix Registers
template <typename OpTy>
struct XTAMEDualAttrLowering : public ConvertOpToLLVMPattern<OpTy> {
  StringRef intrinsicName;
  XTAMEDualAttrLowering(LLVMTypeConverter &typeConverter,
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
    auto intrinsicNameSym =
        getOrInsertIntrinsic(rewriter, module, intrinsicName, funcType);

    Value mdVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms1Val = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, intrinsicNameSym,
                         ValueRange{mdVal, ms1Val});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Data Move Instructions between Integer and Matrix (Duplicate)
template <typename OpTy>
struct XTAMEDupLowering : public ConvertOpToLLVMPattern<OpTy> {
  StringRef intrinsicName;
  XTAMEDupLowering(LLVMTypeConverter &typeConverter, StringRef intrinsicName)
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
    auto intrinsicNameSym =
        getOrInsertIntrinsic(rewriter, module, intrinsicName, funcType);

    Value mdVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, intrinsicNameSym,
                         ValueRange{mdVal, adaptor.getRs2()});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Data Move Instructions between Integer and Matrix (Scalar to Matrix)
template <typename OpTy>
struct XTAMEMmovMXLowering : public ConvertOpToLLVMPattern<OpTy> {
  StringRef intrinsicName;
  XTAMEMmovMXLowering(LLVMTypeConverter &typeConverter, StringRef intrinsicName)
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
    auto intrinsicNameSym =
        getOrInsertIntrinsic(rewriter, module, intrinsicName, funcType);

    Value mdVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, intrinsicNameSym,
                         ValueRange{mdVal, adaptor.getRs2(), adaptor.getRs1()});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Data Move Instructions between Integer and Matrix (Matrix to Scalar)
template <typename OpTy>
struct XTAMEMmovXMLowering : public ConvertOpToLLVMPattern<OpTy> {
  StringRef intrinsicName;
  XTAMEMmovXMLowering(LLVMTypeConverter &typeConverter, StringRef intrinsicName)
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
    auto funcType = LLVM::LLVMFunctionType::get(i64Type, {i64Type, i64Type});
    auto intrinsicNameSym =
        getOrInsertIntrinsic(rewriter, module, intrinsicName, funcType);

    Value ms2Val = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));

    auto callOp = LLVM::CallOp::create(rewriter, loc, i64Type, intrinsicNameSym,
                                       ValueRange{ms2Val, adaptor.getRs1()});
    rewriter.replaceOp(op, callOp.getResults());
    return success();
  }
};

/// Data Broadcast Instructions
template <typename OpTy>
struct XTAMECmovMvILowering : public ConvertOpToLLVMPattern<OpTy> {
  StringRef intrinsicName;
  XTAMECmovMvILowering(LLVMTypeConverter &typeConverter,
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
    auto intrinsicNameSym =
        getOrInsertIntrinsic(rewriter, module, intrinsicName, funcType);

    Value mdVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms1Val = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));
    Value uimm3Val = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getUimm3()));

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, intrinsicNameSym,
                         ValueRange{mdVal, ms1Val, uimm3Val});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Load Operations Lowering
//===----------------------------------------------------------------------===//
template <typename OpTy>
struct XTAMELoadLowering : public ConvertOpToLLVMPattern<OpTy> {
  StringRef intrinsicName;

  XTAMELoadLowering(LLVMTypeConverter &typeConverter, StringRef intrinsicName)
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
                                                {i64Type, i64Type, ptrType});

    auto intrinsicNameSym =
        getOrInsertIntrinsic(rewriter, module, intrinsicName, funcType);

    Value mdVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, intrinsicNameSym,
                         ValueRange{mdVal, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

// Prefetch Instructions Lowering
template <typename OpTy>
struct XTAMEPrefetchLowering : public ConvertOpToLLVMPattern<OpTy> {
  StringRef intrinsicName;
  XTAMEPrefetchLowering(LLVMTypeConverter &typeConverter,
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
                                                {i64Type, ptrType});
    auto intrinsicNameSym =
        getOrInsertIntrinsic(rewriter, module, intrinsicName, funcType);

    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, intrinsicNameSym,
                         ValueRange{adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Store Operations Lowering
//===----------------------------------------------------------------------===//
template <typename OpTy>
struct XTAMEStoreLowering : public ConvertOpToLLVMPattern<OpTy> {
  StringRef intrinsicName;

  XTAMEStoreLowering(LLVMTypeConverter &typeConverter, StringRef intrinsicName)
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
                                                {i64Type, i64Type, ptrType});

    auto intrinsicNameSym =
        getOrInsertIntrinsic(rewriter, module, intrinsicName, funcType);

    Value ms3Val = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getMs3()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, intrinsicNameSym,
                         ValueRange{ms3Val, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

// ===----------------------------------------------------------------------===//
// Tile Register Matrix Multiply Lowering
// ===----------------------------------------------------------------------===//
template <typename OpTy>
struct XTAMETernaryOpLowering : public ConvertOpToLLVMPattern<OpTy> {
  StringRef intrinsicName;

  XTAMETernaryOpLowering(LLVMTypeConverter &typeConverter,
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

    auto intrinsicNameSym =
        getOrInsertIntrinsic(rewriter, module, intrinsicName, funcType);

    Value mdVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms2Val = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));
    Value ms1Val = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, intrinsicNameSym,
                         ValueRange{mdVal, ms2Val, ms1Val});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct LegalizeXTAMEForLLVMExport
    : public PassWrapper<LegalizeXTAMEForLLVMExport, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeXTAMEForLLVMExport)

  StringRef getArgument() const final { return "lower-xt-ame"; }
  StringRef getDescription() const final {
    return "XTAME dialect lowering pass.";
  }

  LegalizeXTAMEForLLVMExport() = default;
  LegalizeXTAMEForLLVMExport(const LegalizeXTAMEForLLVMExport &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<XTAMEDialect>();
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

    target.addIllegalDialect<buddy::xtame::XTAMEDialect>();

    LLVMTypeConverter typeConverter(&context);
    RewritePatternSet patterns(&context);

    // Configuration patterns
    patterns.add<XTAMEConfigLowering<ThMcfgOp>>(typeConverter,
                                                "llvm.riscv.th.mcfg");
    patterns.add<XTAMEConfigLowering<ThMcfgmOp>>(typeConverter,
                                                 "llvm.riscv.th.mcfgm");
    patterns.add<XTAMEConfigLowering<ThMcfgnOp>>(typeConverter,
                                                 "llvm.riscv.th.mcfgn");
    patterns.add<XTAMEConfigLowering<ThMcfgkOp>>(typeConverter,
                                                 "llvm.riscv.th.mcfgk");
    patterns.add<XTAMEConfigImmLowering<ThMcfgmiOp, &ThMcfgmiOp::getTilem>>(
        typeConverter, "llvm.riscv.th.mcfgmi");
    patterns.add<XTAMEConfigImmLowering<ThMcfgniOp, &ThMcfgniOp::getTilen>>(
        typeConverter, "llvm.riscv.th.mcfgni");
    patterns.add<XTAMEConfigImmLowering<ThMcfgkiOp, &ThMcfgkiOp::getTilek>>(
        typeConverter, "llvm.riscv.th.mcfgki");

    // MISC patterns
    patterns.add<XTAMEZeroLowering<ThMzeroOp>>(typeConverter,
                                               "llvm.riscv.th.mzero");
    patterns.add<XTAMEZeroLowering<ThMzero2rOp>>(typeConverter,
                                                 "llvm.riscv.th.mzero2r");
    patterns.add<XTAMEZeroLowering<ThMzero4rOp>>(typeConverter,
                                                 "llvm.riscv.th.mzero4r");
    patterns.add<XTAMEZeroLowering<ThMzero8rOp>>(typeConverter,
                                                 "llvm.riscv.th.mzero8r");
    patterns.add<XTAMEDualAttrLowering<ThMmovMmOp>>(typeConverter,
                                                    "llvm.riscv.th.mmov.mm");
    patterns.add<XTAMEDupLowering<ThMdupbMXOp>>(typeConverter,
                                                "llvm.riscv.th.mdupb.m.x");
    patterns.add<XTAMEDupLowering<ThMduphMXOp>>(typeConverter,
                                                "llvm.riscv.th.mduph.m.x");
    patterns.add<XTAMEDupLowering<ThMdupwMXOp>>(typeConverter,
                                                "llvm.riscv.th.mdupw.m.x");
    patterns.add<XTAMEDupLowering<ThMdupdMXOp>>(typeConverter,
                                                "llvm.riscv.th.mdupd.m.x");
    patterns.add<XTAMEMmovMXLowering<ThMmovbMXOp>>(typeConverter,
                                                   "llvm.riscv.th.mmovb.m.x");
    patterns.add<XTAMEMmovMXLowering<ThMmovhMXOp>>(typeConverter,
                                                   "llvm.riscv.th.mmovh.m.x");
    patterns.add<XTAMEMmovMXLowering<ThMmovwMXOp>>(typeConverter,
                                                   "llvm.riscv.th.mmovw.m.x");
    patterns.add<XTAMEMmovMXLowering<ThMmovdMXOp>>(typeConverter,
                                                   "llvm.riscv.th.mmovd.m.x");
    patterns.add<XTAMEMmovXMLowering<ThMmovbXMOp>>(typeConverter,
                                                   "llvm.riscv.th.mmovb.x.m");
    patterns.add<XTAMEMmovXMLowering<ThMmovhXMOp>>(typeConverter,
                                                   "llvm.riscv.th.mmovh.x.m");
    patterns.add<XTAMEMmovXMLowering<ThMmovwXMOp>>(typeConverter,
                                                   "llvm.riscv.th.mmovw.x.m");
    patterns.add<XTAMEMmovXMLowering<ThMmovdXMOp>>(typeConverter,
                                                   "llvm.riscv.th.mmovd.x.m");
    patterns.add<XTAMECmovMvILowering<ThMmovMvIOp>>(typeConverter,
                                                    "llvm.riscv.th.mmov.mv.i");
    patterns.add<XTAMECmovMvILowering<ThMcmovbMvIOp>>(
        typeConverter, "llvm.riscv.th.mcmovb.mv.i");
    patterns.add<XTAMECmovMvILowering<ThMcmovhMvIOp>>(
        typeConverter, "llvm.riscv.th.mcmovh.mv.i");
    patterns.add<XTAMECmovMvILowering<ThMcmovwMvIOp>>(
        typeConverter, "llvm.riscv.th.mcmovw.mv.i");
    patterns.add<XTAMECmovMvILowering<ThMcmovdMvIOp>>(
        typeConverter, "llvm.riscv.th.mcmovd.mv.i");
    patterns.add<XTAMETernaryOpLowering<ThMpackMmOp>>(typeConverter,
                                                      "llvm.riscv.th.mpack.mm");
    patterns.add<XTAMETernaryOpLowering<ThMpackhlMmOp>>(
        typeConverter, "llvm.riscv.th.mpackhl.mm");
    patterns.add<XTAMETernaryOpLowering<ThMpackhhMmOp>>(
        typeConverter, "llvm.riscv.th.mpackhh.mm");

    // Load/Store patterns
    patterns.add<XTAMELoadLowering<ThMlde8Op>>(typeConverter,
                                               "llvm.riscv.th.mlde8");
    patterns.add<XTAMELoadLowering<ThMlde16Op>>(typeConverter,
                                                "llvm.riscv.th.mlde16");
    patterns.add<XTAMELoadLowering<ThMlde32Op>>(typeConverter,
                                                "llvm.riscv.th.mlde32");
    patterns.add<XTAMELoadLowering<ThMlde64Op>>(typeConverter,
                                                "llvm.riscv.th.mlde64");
    patterns.add<XTAMELoadLowering<ThMldte8Op>>(typeConverter,
                                                "llvm.riscv.th.mldte8");
    patterns.add<XTAMELoadLowering<ThMldte16Op>>(typeConverter,
                                                 "llvm.riscv.th.mldte16");
    patterns.add<XTAMELoadLowering<ThMldte32Op>>(typeConverter,
                                                 "llvm.riscv.th.mldte32");
    patterns.add<XTAMELoadLowering<ThMldte64Op>>(typeConverter,
                                                 "llvm.riscv.th.mldte64");
    patterns.add<XTAMELoadLowering<ThMslde8Op>>(typeConverter,
                                                "llvm.riscv.th.mslde8");
    patterns.add<XTAMELoadLowering<ThMslde16Op>>(typeConverter,
                                                 "llvm.riscv.th.mslde16");
    patterns.add<XTAMELoadLowering<ThMslde32Op>>(typeConverter,
                                                 "llvm.riscv.th.mslde32");
    patterns.add<XTAMELoadLowering<ThMslde64Op>>(typeConverter,
                                                 "llvm.riscv.th.mslde64");
    patterns.add<XTAMELoadLowering<ThMsldte8Op>>(typeConverter,
                                                 "llvm.riscv.th.msldte8");
    patterns.add<XTAMELoadLowering<ThMsldte16Op>>(typeConverter,
                                                  "llvm.riscv.th.msldte16");
    patterns.add<XTAMELoadLowering<ThMsldte32Op>>(typeConverter,
                                                  "llvm.riscv.th.msldte32");
    patterns.add<XTAMELoadLowering<ThMsldte64Op>>(typeConverter,
                                                  "llvm.riscv.th.msldte64");

    patterns.add<XTAMEPrefetchLowering<ThMplde8Op>>(typeConverter,
                                                    "llvm.riscv.th.mplde8");
    patterns.add<XTAMEPrefetchLowering<ThMplde16Op>>(typeConverter,
                                                     "llvm.riscv.th.mplde16");
    patterns.add<XTAMEPrefetchLowering<ThMplde32Op>>(typeConverter,
                                                     "llvm.riscv.th.mplde32");
    patterns.add<XTAMEPrefetchLowering<ThMplde64Op>>(typeConverter,
                                                     "llvm.riscv.th.mplde64");
    patterns.add<XTAMEPrefetchLowering<ThMpldte8Op>>(typeConverter,
                                                     "llvm.riscv.th.mpldte8");
    patterns.add<XTAMEPrefetchLowering<ThMpldte16Op>>(typeConverter,
                                                      "llvm.riscv.th.mpldte16");
    patterns.add<XTAMEPrefetchLowering<ThMpldte32Op>>(typeConverter,
                                                      "llvm.riscv.th.mpldte32");
    patterns.add<XTAMEPrefetchLowering<ThMpldte64Op>>(typeConverter,
                                                      "llvm.riscv.th.mpldte64");

    patterns.add<XTAMEStoreLowering<ThMste8Op>>(typeConverter,
                                                "llvm.riscv.th.mste8");
    patterns.add<XTAMEStoreLowering<ThMste16Op>>(typeConverter,
                                                 "llvm.riscv.th.mste16");
    patterns.add<XTAMEStoreLowering<ThMste32Op>>(typeConverter,
                                                 "llvm.riscv.th.mste32");
    patterns.add<XTAMEStoreLowering<ThMste64Op>>(typeConverter,
                                                 "llvm.riscv.th.mste64");
    patterns.add<XTAMEStoreLowering<ThMstte8Op>>(typeConverter,
                                                 "llvm.riscv.th.mstte8");
    patterns.add<XTAMEStoreLowering<ThMstte16Op>>(typeConverter,
                                                  "llvm.riscv.th.mstte16");
    patterns.add<XTAMEStoreLowering<ThMstte32Op>>(typeConverter,
                                                  "llvm.riscv.th.mstte32");
    patterns.add<XTAMEStoreLowering<ThMstte64Op>>(typeConverter,
                                                  "llvm.riscv.th.mstte64");
    patterns.add<XTAMEStoreLowering<ThMsste8Op>>(typeConverter,
                                                 "llvm.riscv.th.msste8");
    patterns.add<XTAMEStoreLowering<ThMsste16Op>>(typeConverter,
                                                  "llvm.riscv.th.msste16");
    patterns.add<XTAMEStoreLowering<ThMsste32Op>>(typeConverter,
                                                  "llvm.riscv.th.msste32");
    patterns.add<XTAMEStoreLowering<ThMsste64Op>>(typeConverter,
                                                  "llvm.riscv.th.msste64");
    patterns.add<XTAMEStoreLowering<ThMsstte8Op>>(typeConverter,
                                                  "llvm.riscv.th.msstte8");
    patterns.add<XTAMEStoreLowering<ThMsstte16Op>>(typeConverter,
                                                   "llvm.riscv.th.msstte16");
    patterns.add<XTAMEStoreLowering<ThMsstte32Op>>(typeConverter,
                                                   "llvm.riscv.th.msstte32");
    patterns.add<XTAMEStoreLowering<ThMsstte64Op>>(typeConverter,
                                                   "llvm.riscv.th.msstte64");

    // Tile register matrix multiply patterns
    patterns.add<XTAMETernaryOpLowering<ThMmaccWBOp>>(
        typeConverter, "llvm.riscv.th.mmacc.w.b");
    patterns.add<XTAMETernaryOpLowering<ThMmaccuWBOp>>(
        typeConverter, "llvm.riscv.th.mmaccu.w.b");
    patterns.add<XTAMETernaryOpLowering<ThMmaccusWBOp>>(
        typeConverter, "llvm.riscv.th.mmaccus.w.b");
    patterns.add<XTAMETernaryOpLowering<ThMmaccsuWBOp>>(
        typeConverter, "llvm.riscv.th.mmaccsu.w.b");

    patterns.add<XTAMETernaryOpLowering<ThMfmaccHOp>>(typeConverter,
                                                      "llvm.riscv.th.mfmacc.h");
    patterns.add<XTAMETernaryOpLowering<ThMfmaccBf16Op>>(
        typeConverter, "llvm.riscv.th.mfmacc.bf16");
    patterns.add<XTAMETernaryOpLowering<ThMfmaccSOp>>(typeConverter,
                                                      "llvm.riscv.th.mfmacc.s");
    patterns.add<XTAMETernaryOpLowering<ThMfmaccDOp>>(typeConverter,
                                                      "llvm.riscv.th.mfmacc.d");
    patterns.add<XTAMETernaryOpLowering<ThMfmaccHE4m3Op>>(
        typeConverter, "llvm.riscv.th.mfmacc.h.e4m3");
    patterns.add<XTAMETernaryOpLowering<ThMfmaccHE5m2Op>>(
        typeConverter, "llvm.riscv.th.mfmacc.h.e5m2");
    patterns.add<XTAMETernaryOpLowering<ThMfmaccBf16E4m3Op>>(
        typeConverter, "llvm.riscv.th.mfmacc.bf16.e4m3");
    patterns.add<XTAMETernaryOpLowering<ThMfmaccBf16E5m2Op>>(
        typeConverter, "llvm.riscv.th.mfmacc.bf16.e5m2");
    patterns.add<XTAMETernaryOpLowering<ThMfmaccSHOp>>(
        typeConverter, "llvm.riscv.th.mfmacc.s.h");
    patterns.add<XTAMETernaryOpLowering<ThMfmaccSBf16Op>>(
        typeConverter, "llvm.riscv.th.mfmacc.s.bf16");
    patterns.add<XTAMETernaryOpLowering<ThMfmaccDSOp>>(
        typeConverter, "llvm.riscv.th.mfmacc.d.s");
    patterns.add<XTAMETernaryOpLowering<ThMfmaccSE4m3Op>>(
        typeConverter, "llvm.riscv.th.mfmacc.s.e4m3");
    patterns.add<XTAMETernaryOpLowering<ThMfmaccSE5m2Op>>(
        typeConverter, "llvm.riscv.th.mfmacc.s.e5m2");

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::populateXTAMELegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // Configuration patterns
  patterns.add<XTAMEConfigLowering<ThMcfgOp>>(converter, "llvm.riscv.th.mcfg");
  patterns.add<XTAMEConfigLowering<ThMcfgmOp>>(converter,
                                               "llvm.riscv.th.mcfgm");
  patterns.add<XTAMEConfigLowering<ThMcfgnOp>>(converter,
                                               "llvm.riscv.th.mcfgn");
  patterns.add<XTAMEConfigLowering<ThMcfgkOp>>(converter,
                                               "llvm.riscv.th.mcfgk");
  patterns.add<XTAMEConfigImmLowering<ThMcfgmiOp, &ThMcfgmiOp::getTilem>>(
      converter, "llvm.riscv.th.mcfgmi");
  patterns.add<XTAMEConfigImmLowering<ThMcfgniOp, &ThMcfgniOp::getTilen>>(
      converter, "llvm.riscv.th.mcfgni");
  patterns.add<XTAMEConfigImmLowering<ThMcfgkiOp, &ThMcfgkiOp::getTilek>>(
      converter, "llvm.riscv.th.mcfgki");

  // MISC patterns
  patterns.add<XTAMEZeroLowering<ThMzeroOp>>(converter, "llvm.riscv.th.mzero");
  patterns.add<XTAMEZeroLowering<ThMzero2rOp>>(converter,
                                               "llvm.riscv.th.mzero2r");
  patterns.add<XTAMEZeroLowering<ThMzero4rOp>>(converter,
                                               "llvm.riscv.th.mzero4r");
  patterns.add<XTAMEZeroLowering<ThMzero8rOp>>(converter,
                                               "llvm.riscv.th.mzero8r");
  patterns.add<XTAMEDualAttrLowering<ThMmovMmOp>>(converter,
                                                  "llvm.riscv.th.mmov.mm");
  patterns.add<XTAMEDupLowering<ThMdupbMXOp>>(converter,
                                              "llvm.riscv.th.mdupb.m.x");
  patterns.add<XTAMEDupLowering<ThMduphMXOp>>(converter,
                                              "llvm.riscv.th.mduph.m.x");
  patterns.add<XTAMEDupLowering<ThMdupwMXOp>>(converter,
                                              "llvm.riscv.th.mdupw.m.x");
  patterns.add<XTAMEDupLowering<ThMdupdMXOp>>(converter,
                                              "llvm.riscv.th.mdupd.m.x");
  patterns.add<XTAMEMmovMXLowering<ThMmovbMXOp>>(converter,
                                                 "llvm.riscv.th.mmovb.m.x");
  patterns.add<XTAMEMmovMXLowering<ThMmovhMXOp>>(converter,
                                                 "llvm.riscv.th.mmovh.m.x");
  patterns.add<XTAMEMmovMXLowering<ThMmovwMXOp>>(converter,
                                                 "llvm.riscv.th.mmovw.m.x");
  patterns.add<XTAMEMmovMXLowering<ThMmovdMXOp>>(converter,
                                                 "llvm.riscv.th.mmovd.m.x");
  patterns.add<XTAMEMmovXMLowering<ThMmovbXMOp>>(converter,
                                                 "llvm.riscv.th.mmovb.x.m");
  patterns.add<XTAMEMmovXMLowering<ThMmovhXMOp>>(converter,
                                                 "llvm.riscv.th.mmovh.x.m");
  patterns.add<XTAMEMmovXMLowering<ThMmovwXMOp>>(converter,
                                                 "llvm.riscv.th.mmovw.x.m");
  patterns.add<XTAMEMmovXMLowering<ThMmovdXMOp>>(converter,
                                                 "llvm.riscv.th.mmovd.x.m");
  patterns.add<XTAMECmovMvILowering<ThMmovMvIOp>>(converter,
                                                  "llvm.riscv.th.mmov.mv.i");
  patterns.add<XTAMECmovMvILowering<ThMcmovbMvIOp>>(
      converter, "llvm.riscv.th.mcmovb.mv.i");
  patterns.add<XTAMECmovMvILowering<ThMcmovhMvIOp>>(
      converter, "llvm.riscv.th.mcmovh.mv.i");
  patterns.add<XTAMECmovMvILowering<ThMcmovwMvIOp>>(
      converter, "llvm.riscv.th.mcmovw.mv.i");
  patterns.add<XTAMECmovMvILowering<ThMcmovdMvIOp>>(
      converter, "llvm.riscv.th.mcmovd.mv.i");
  patterns.add<XTAMETernaryOpLowering<ThMpackMmOp>>(converter,
                                                    "llvm.riscv.th.mpack.mm");
  patterns.add<XTAMETernaryOpLowering<ThMpackhlMmOp>>(
      converter, "llvm.riscv.th.mpackhl.mm");
  patterns.add<XTAMETernaryOpLowering<ThMpackhhMmOp>>(
      converter, "llvm.riscv.th.mpackhh.mm");

  // Load/Store patterns
  patterns.add<XTAMELoadLowering<ThMlde8Op>>(converter, "llvm.riscv.th.mlde8");
  patterns.add<XTAMELoadLowering<ThMlde16Op>>(converter,
                                              "llvm.riscv.th.mlde16");
  patterns.add<XTAMELoadLowering<ThMlde32Op>>(converter,
                                              "llvm.riscv.th.mlde32");
  patterns.add<XTAMELoadLowering<ThMlde64Op>>(converter,
                                              "llvm.riscv.th.mlde64");
  patterns.add<XTAMELoadLowering<ThMldte8Op>>(converter,
                                              "llvm.riscv.th.mldte8");
  patterns.add<XTAMELoadLowering<ThMldte16Op>>(converter,
                                               "llvm.riscv.th.mldte16");
  patterns.add<XTAMELoadLowering<ThMldte32Op>>(converter,
                                               "llvm.riscv.th.mldte32");
  patterns.add<XTAMELoadLowering<ThMldte64Op>>(converter,
                                               "llvm.riscv.th.mldte64");
  patterns.add<XTAMELoadLowering<ThMslde8Op>>(converter,
                                              "llvm.riscv.th.mslde8");
  patterns.add<XTAMELoadLowering<ThMslde16Op>>(converter,
                                               "llvm.riscv.th.mslde16");
  patterns.add<XTAMELoadLowering<ThMslde32Op>>(converter,
                                               "llvm.riscv.th.mslde32");
  patterns.add<XTAMELoadLowering<ThMslde64Op>>(converter,
                                               "llvm.riscv.th.mslde64");
  patterns.add<XTAMELoadLowering<ThMsldte8Op>>(converter,
                                               "llvm.riscv.th.msldte8");
  patterns.add<XTAMELoadLowering<ThMsldte16Op>>(converter,
                                                "llvm.riscv.th.msldte16");
  patterns.add<XTAMELoadLowering<ThMsldte32Op>>(converter,
                                                "llvm.riscv.th.msldte32");
  patterns.add<XTAMELoadLowering<ThMsldte64Op>>(converter,
                                                "llvm.riscv.th.msldte64");

  patterns.add<XTAMEPrefetchLowering<ThMplde8Op>>(converter,
                                                  "llvm.riscv.th.mplde8");
  patterns.add<XTAMEPrefetchLowering<ThMplde16Op>>(converter,
                                                   "llvm.riscv.th.mplde16");
  patterns.add<XTAMEPrefetchLowering<ThMplde32Op>>(converter,
                                                   "llvm.riscv.th.mplde32");
  patterns.add<XTAMEPrefetchLowering<ThMplde64Op>>(converter,
                                                   "llvm.riscv.th.mplde64");
  patterns.add<XTAMEPrefetchLowering<ThMpldte8Op>>(converter,
                                                   "llvm.riscv.th.mpldte8");
  patterns.add<XTAMEPrefetchLowering<ThMpldte16Op>>(converter,
                                                    "llvm.riscv.th.mpldte16");
  patterns.add<XTAMEPrefetchLowering<ThMpldte32Op>>(converter,
                                                    "llvm.riscv.th.mpldte32");
  patterns.add<XTAMEPrefetchLowering<ThMpldte64Op>>(converter,
                                                    "llvm.riscv.th.mpldte64");

  patterns.add<XTAMEStoreLowering<ThMste8Op>>(converter, "llvm.riscv.th.mste8");
  patterns.add<XTAMEStoreLowering<ThMste16Op>>(converter,
                                               "llvm.riscv.th.mste16");
  patterns.add<XTAMEStoreLowering<ThMste32Op>>(converter,
                                               "llvm.riscv.th.mste32");
  patterns.add<XTAMEStoreLowering<ThMste64Op>>(converter,
                                               "llvm.riscv.th.mste64");
  patterns.add<XTAMEStoreLowering<ThMstte8Op>>(converter,
                                               "llvm.riscv.th.mstte8");
  patterns.add<XTAMEStoreLowering<ThMstte16Op>>(converter,
                                                "llvm.riscv.th.mstte16");
  patterns.add<XTAMEStoreLowering<ThMstte32Op>>(converter,
                                                "llvm.riscv.th.mstte32");
  patterns.add<XTAMEStoreLowering<ThMstte64Op>>(converter,
                                                "llvm.riscv.th.mstte64");
  patterns.add<XTAMEStoreLowering<ThMsste8Op>>(converter,
                                               "llvm.riscv.th.msste8");
  patterns.add<XTAMEStoreLowering<ThMsste16Op>>(converter,
                                                "llvm.riscv.th.msste16");
  patterns.add<XTAMEStoreLowering<ThMsste32Op>>(converter,
                                                "llvm.riscv.th.msste32");
  patterns.add<XTAMEStoreLowering<ThMsste64Op>>(converter,
                                                "llvm.riscv.th.msste64");
  patterns.add<XTAMEStoreLowering<ThMsstte8Op>>(converter,
                                                "llvm.riscv.th.msstte8");
  patterns.add<XTAMEStoreLowering<ThMsstte16Op>>(converter,
                                                 "llvm.riscv.th.msstte16");
  patterns.add<XTAMEStoreLowering<ThMsstte32Op>>(converter,
                                                 "llvm.riscv.th.msstte32");
  patterns.add<XTAMEStoreLowering<ThMsstte64Op>>(converter,
                                                 "llvm.riscv.th.msstte64");

  patterns.add<XTAMETernaryOpLowering<ThMmaccWBOp>>(converter,
                                                    "llvm.riscv.th.mmacc.w.b");
  patterns.add<XTAMETernaryOpLowering<ThMmaccuWBOp>>(
      converter, "llvm.riscv.th.mmaccu.w.b");
  patterns.add<XTAMETernaryOpLowering<ThMmaccusWBOp>>(
      converter, "llvm.riscv.th.mmaccus.w.b");
  patterns.add<XTAMETernaryOpLowering<ThMmaccsuWBOp>>(
      converter, "llvm.riscv.th.mmaccsu.w.b");

  // Tile register matrix multiply patterns (float-point types)
  patterns.add<XTAMETernaryOpLowering<ThMfmaccHOp>>(converter,
                                                    "llvm.riscv.th.mfmacc.h");
  patterns.add<XTAMETernaryOpLowering<ThMfmaccBf16Op>>(
      converter, "llvm.riscv.th.mfmacc.bf16");
  patterns.add<XTAMETernaryOpLowering<ThMfmaccSOp>>(converter,
                                                    "llvm.riscv.th.mfmacc.s");
  patterns.add<XTAMETernaryOpLowering<ThMfmaccDOp>>(converter,
                                                    "llvm.riscv.th.mfmacc.d");
  patterns.add<XTAMETernaryOpLowering<ThMfmaccHE4m3Op>>(
      converter, "llvm.riscv.th.mfmacc.h.e4m3");
  patterns.add<XTAMETernaryOpLowering<ThMfmaccHE5m2Op>>(
      converter, "llvm.riscv.th.mfmacc.h.e5m2");
  patterns.add<XTAMETernaryOpLowering<ThMfmaccBf16E4m3Op>>(
      converter, "llvm.riscv.th.mfmacc.bf16.e4m3");
  patterns.add<XTAMETernaryOpLowering<ThMfmaccBf16E5m2Op>>(
      converter, "llvm.riscv.th.mfmacc.bf16.e5m2");
  patterns.add<XTAMETernaryOpLowering<ThMfmaccSHOp>>(
      converter, "llvm.riscv.th.mfmacc.s.h");
  patterns.add<XTAMETernaryOpLowering<ThMfmaccSBf16Op>>(
      converter, "llvm.riscv.th.mfmacc.s.bf16");
  patterns.add<XTAMETernaryOpLowering<ThMfmaccDSOp>>(
      converter, "llvm.riscv.th.mfmacc.d.s");
  patterns.add<XTAMETernaryOpLowering<ThMfmaccSE4m3Op>>(
      converter, "llvm.riscv.th.mfmacc.s.e4m3");
  patterns.add<XTAMETernaryOpLowering<ThMfmaccSE5m2Op>>(
      converter, "llvm.riscv.th.mfmacc.s.e5m2");
}

void mlir::configureXTAMELegalizeForExportTarget(LLVMConversionTarget &target) {
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<memref::MemRefDialect>();

  target.addIllegalDialect<buddy::xtame::XTAMEDialect>();
}

std::unique_ptr<Pass> buddy::xtame::createLegalizeForLLVMExportPass() {
  return std::make_unique<LegalizeXTAMEForLLVMExport>();
}

namespace mlir {
namespace buddy {
void registerLowerXTAMEPass() {
  PassRegistration<LegalizeXTAMEForLLVMExport>();
}
} // namespace buddy
} // namespace mlir
