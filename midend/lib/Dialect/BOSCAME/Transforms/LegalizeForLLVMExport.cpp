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
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), intrinsicName, funcType,
                                    LLVM::Linkage::External, false,
                                    LLVM::CConv::C);
  rewriter.restoreInsertionPoint(savedInsertionPoint);
  return FlatSymbolRefAttr::get(ctx, intrinsicName);
}

static Value extractPointerFromMemref(ConversionPatternRewriter &rewriter,
                                      Location loc, Value memref) {
  auto *ctx = rewriter.getContext();
  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  auto i64Type = IntegerType::get(ctx, 64);
  Value idx =
      rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, memref);
  Value i64Val = rewriter.create<arith::IndexCastOp>(loc, i64Type, idx);
  Value ptr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, i64Val);
  return ptr;
}

//===----------------------------------------------------------------------===//
// BOSCAME Lowering Patterns
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Configuration Operations Lowering
//===----------------------------------------------------------------------===//

/// Lowering pattern for msettilemi
struct BOSCAMEMSettilemiLowering : public ConvertOpToLLVMPattern<MSettilemiOp> {
  using ConvertOpToLLVMPattern<MSettilemiOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MSettilemiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(i64Type, {i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.bosc.msettilemi", funcType);

    Value tilemVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getTilem()));

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, TypeRange{i64Type}, intrinsicName, ValueRange{tilemVal});
    rewriter.replaceOp(op, callOp.getResults());
    return success();
  }
};

/// Lowering pattern for msettileni
struct BOSCAMEMSettileniLowering : public ConvertOpToLLVMPattern<MSettileniOp> {
  using ConvertOpToLLVMPattern<MSettileniOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MSettileniOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(i64Type, {i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.bosc.msettileni", funcType);

    Value tilenVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getTilen()));

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, TypeRange{i64Type}, intrinsicName, ValueRange{tilenVal});
    rewriter.replaceOp(op, callOp.getResults());
    return success();
  }
};

/// Lowering pattern for msettileki
struct BOSCAMEMSettilekiLowering : public ConvertOpToLLVMPattern<MSettilekiOp> {
  using ConvertOpToLLVMPattern<MSettilekiOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MSettilekiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(i64Type, {i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.bosc.msettileki", funcType);

    Value tilekVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getTilek()));

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, TypeRange{i64Type}, intrinsicName, ValueRange{tilekVal});
    rewriter.replaceOp(op, callOp.getResults());
    return success();
  }
};

/// Lowering pattern for mzero
struct BOSCAMEMzeroLowering : public ConvertOpToLLVMPattern<MzeroOp> {
  using ConvertOpToLLVMPattern<MzeroOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MzeroOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType =
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.bosc.mzero", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for mlae32.m (load left matrix A)
struct BOSCAMEMlae32mLowering : public ConvertOpToLLVMPattern<Mlae32mOp> {
  using ConvertOpToLLVMPattern<Mlae32mOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Mlae32mOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, ptrType, i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.bosc.mlae32.m", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{mdVal, basePtr, adaptor.getStride()});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for mlbe32.m (load right matrix B)
struct BOSCAMEMlbe32mLowering : public ConvertOpToLLVMPattern<Mlbe32mOp> {
  using ConvertOpToLLVMPattern<Mlbe32mOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Mlbe32mOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, ptrType, i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.bosc.mlbe32.m", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{mdVal, basePtr, adaptor.getStride()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Store Operations Lowering
//===----------------------------------------------------------------------===//

/// Lowering pattern for msce32.m (store output matrix C)
struct BOSCAMEMsce32mLowering : public ConvertOpToLLVMPattern<Msce32mOp> {
  using ConvertOpToLLVMPattern<Msce32mOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Msce32mOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, ptrType, i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.bosc.msce32.m", funcType);

    Value ms3Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs3()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{ms3Val, basePtr, adaptor.getStride()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Tile Register Matrix Multiply Lowering
//===----------------------------------------------------------------------===//

/// Lowering pattern for mma.w.mm (tile register matrix multiply)
struct BOSCAMEMmaWmmLowering : public ConvertOpToLLVMPattern<MmaWmmOp> {
  using ConvertOpToLLVMPattern<MmaWmmOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MmaWmmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.bosc.mma.w.mm", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));
    Value ms2Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
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

    // Configuration operations
    target.addIllegalOp<MSettilemiOp>();
    target.addIllegalOp<MSettileniOp>();
    target.addIllegalOp<MSettilekiOp>();
    target.addIllegalOp<MzeroOp>();

    // Load/Store operations
    target.addIllegalOp<Mlae32mOp>();
    target.addIllegalOp<Mlbe32mOp>();
    target.addIllegalOp<Msce32mOp>();

    // Tile register matrix multiply
    target.addIllegalOp<MmaWmmOp>();

    LLVMTypeConverter typeConverter(&context);

    RewritePatternSet patterns(&context);

    // Configuration patterns
    patterns.add<BOSCAMEMSettilemiLowering>(typeConverter);
    patterns.add<BOSCAMEMSettileniLowering>(typeConverter);
    patterns.add<BOSCAMEMSettilekiLowering>(typeConverter);
    patterns.add<BOSCAMEMzeroLowering>(typeConverter);

    // Load/Store patterns
    patterns.add<BOSCAMEMlae32mLowering>(typeConverter);
    patterns.add<BOSCAMEMlbe32mLowering>(typeConverter);
    patterns.add<BOSCAMEMsce32mLowering>(typeConverter);

    // Tile register matrix multiply patterns
    patterns.add<BOSCAMEMmaWmmLowering>(typeConverter);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::populateBOSCAMELegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // Configuration patterns
  patterns.add<BOSCAMEMSettilemiLowering>(converter);
  patterns.add<BOSCAMEMSettileniLowering>(converter);
  patterns.add<BOSCAMEMSettilekiLowering>(converter);
  patterns.add<BOSCAMEMzeroLowering>(converter);

  // Load/Store patterns
  patterns.add<BOSCAMEMlae32mLowering>(converter);
  patterns.add<BOSCAMEMlbe32mLowering>(converter);
  patterns.add<BOSCAMEMsce32mLowering>(converter);

  // Tile register matrix multiply patterns
  patterns.add<BOSCAMEMmaWmmLowering>(converter);
}

void mlir::configureBOSCAMELegalizeForExportTarget(
    LLVMConversionTarget &target) {
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<memref::MemRefDialect>();

  // Configuration operations
  target.addIllegalOp<MSettilemiOp>();
  target.addIllegalOp<MSettileniOp>();
  target.addIllegalOp<MSettilekiOp>();
  target.addIllegalOp<MzeroOp>();

  // Load/Store operations
  target.addIllegalOp<Mlae32mOp>();
  target.addIllegalOp<Mlbe32mOp>();
  target.addIllegalOp<Msce32mOp>();

  // Tile register matrix multiply
  target.addIllegalOp<MmaWmmOp>();
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
