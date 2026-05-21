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
// XTAME Lowering Patterns
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Configuration Operations Lowering
//===----------------------------------------------------------------------===//

/// Lowering pattern for th.mcfg
struct XTAMEThMcfgLowering : public ConvertOpToLLVMPattern<ThMcfgOp> {
  using ConvertOpToLLVMPattern<ThMcfgOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMcfgOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mcfg", funcType);

    Value mtypeVal = adaptor.getMtype();

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mtypeVal});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mcfgm
struct XTAMEThMcfgmLowering : public ConvertOpToLLVMPattern<ThMcfgmOp> {
  using ConvertOpToLLVMPattern<ThMcfgmOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMcfgmOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mcfgm", funcType);

    Value tilemVal = adaptor.getTilem();

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{tilemVal});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mcfgn
struct XTAMEThMcfgnLowering : public ConvertOpToLLVMPattern<ThMcfgnOp> {
  using ConvertOpToLLVMPattern<ThMcfgnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMcfgnOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mcfgn", funcType);

    Value tilenVal = adaptor.getTilen();

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{tilenVal});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mcfgk
struct XTAMEThMcfgkLowering : public ConvertOpToLLVMPattern<ThMcfgkOp> {
  using ConvertOpToLLVMPattern<ThMcfgkOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMcfgkOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mcfgk", funcType);

    Value tilekVal = adaptor.getTilek();

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{tilekVal});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mcfgmi (set tile M dimension with immediate)
struct XTAMEThMcfgmiLowering : public ConvertOpToLLVMPattern<ThMcfgmiOp> {
  using ConvertOpToLLVMPattern<ThMcfgmiOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMcfgmiOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mcfgmi", funcType);

    Value tilemVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getTilem()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{tilemVal});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mcfgni
struct XTAMEThMcfgniLowering : public ConvertOpToLLVMPattern<ThMcfgniOp> {
  using ConvertOpToLLVMPattern<ThMcfgniOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMcfgniOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mcfgni", funcType);

    Value tilenVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getTilen()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{tilenVal});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mcfgki
struct XTAMEThMcfgkiLowering : public ConvertOpToLLVMPattern<ThMcfgkiOp> {
  using ConvertOpToLLVMPattern<ThMcfgkiOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMcfgkiOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mcfgki", funcType);

    Value tilekVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getTilek()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{tilekVal});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for mzero
struct XTAMEThMzeroLowering : public ConvertOpToLLVMPattern<ThMzeroOp> {
  using ConvertOpToLLVMPattern<ThMzeroOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMzeroOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mzero", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMzero2rLowering : public ConvertOpToLLVMPattern<ThMzero2rOp> {
  using ConvertOpToLLVMPattern<ThMzero2rOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMzero2rOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mzero2r", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMzero4rLowering : public ConvertOpToLLVMPattern<ThMzero4rOp> {
  using ConvertOpToLLVMPattern<ThMzero4rOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMzero4rOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mzero4r", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMzero8rLowering : public ConvertOpToLLVMPattern<ThMzero8rOp> {
  using ConvertOpToLLVMPattern<ThMzero8rOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMzero8rOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mzero8r", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Data Move Instructions between Matrix Registers
struct XTAMEThMmovMmLowering : public ConvertOpToLLVMPattern<ThMmovMmOp> {
  using ConvertOpToLLVMPattern<ThMmovMmOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMmovMmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mmov.mm", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms1Val});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Data Move Instructions between Integer and Matrix (Duplicate)
struct XTAMEThMdupbMXLowering : public ConvertOpToLLVMPattern<ThMdupbMXOp> {
  using ConvertOpToLLVMPattern<ThMdupbMXOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMdupbMXOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mdupb.m.x", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, adaptor.getRs2()});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMduphMXLowering : public ConvertOpToLLVMPattern<ThMduphMXOp> {
  using ConvertOpToLLVMPattern<ThMduphMXOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMduphMXOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mduph.m.x", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, adaptor.getRs2()});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMdupwMXLowering : public ConvertOpToLLVMPattern<ThMdupwMXOp> {
  using ConvertOpToLLVMPattern<ThMdupwMXOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMdupwMXOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mdupw.m.x", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, adaptor.getRs2()});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMdupdMXLowering : public ConvertOpToLLVMPattern<ThMdupdMXOp> {
  using ConvertOpToLLVMPattern<ThMdupdMXOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMdupdMXOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mdupd.m.x", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, adaptor.getRs2()});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Data Move Instructions between Integer and Matrix (Scalar to Matrix)
struct XTAMEThMmovbMXLowering : public ConvertOpToLLVMPattern<ThMmovbMXOp> {
  using ConvertOpToLLVMPattern<ThMmovbMXOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMmovbMXOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mmovb.m.x", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{mdVal, adaptor.getRs2(), adaptor.getRs1()});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMmovhMXLowering : public ConvertOpToLLVMPattern<ThMmovhMXOp> {
  using ConvertOpToLLVMPattern<ThMmovhMXOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMmovhMXOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mmovh.m.x", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{mdVal, adaptor.getRs2(), adaptor.getRs1()});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMmovwMXLowering : public ConvertOpToLLVMPattern<ThMmovwMXOp> {
  using ConvertOpToLLVMPattern<ThMmovwMXOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMmovwMXOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mmovw.m.x", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{mdVal, adaptor.getRs2(), adaptor.getRs1()});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMmovdMXLowering : public ConvertOpToLLVMPattern<ThMmovdMXOp> {
  using ConvertOpToLLVMPattern<ThMmovdMXOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMmovdMXOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mmovd.m.x", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{mdVal, adaptor.getRs2(), adaptor.getRs1()});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Data Move Instructions between Integer and Matrix (Matrix to Scalar)
struct XTAMEThMmovbXMLowering : public ConvertOpToLLVMPattern<ThMmovbXMOp> {
  using ConvertOpToLLVMPattern<ThMmovbXMOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMmovbXMOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(i64Type, {i64Type, i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mmovb.x.m", funcType);

    Value ms2Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, i64Type, intrinsicName, ValueRange{ms2Val, adaptor.getRs1()});
    rewriter.replaceOp(op, callOp.getResults());
    return success();
  }
};

struct XTAMEThMmovhXMLowering : public ConvertOpToLLVMPattern<ThMmovhXMOp> {
  using ConvertOpToLLVMPattern<ThMmovhXMOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMmovhXMOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(i64Type, {i64Type, i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mmovh.x.m", funcType);

    Value ms2Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, i64Type, intrinsicName, ValueRange{ms2Val, adaptor.getRs1()});
    rewriter.replaceOp(op, callOp.getResults());
    return success();
  }
};

struct XTAMEThMmovwXMLowering : public ConvertOpToLLVMPattern<ThMmovwXMOp> {
  using ConvertOpToLLVMPattern<ThMmovwXMOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMmovwXMOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(i64Type, {i64Type, i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mmovw.x.m", funcType);

    Value ms2Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, i64Type, intrinsicName, ValueRange{ms2Val, adaptor.getRs1()});
    rewriter.replaceOp(op, callOp.getResults());
    return success();
  }
};

struct XTAMEThMmovdXMLowering : public ConvertOpToLLVMPattern<ThMmovdXMOp> {
  using ConvertOpToLLVMPattern<ThMmovdXMOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMmovdXMOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(i64Type, {i64Type, i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mmovd.x.m", funcType);

    Value ms2Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, i64Type, intrinsicName, ValueRange{ms2Val, adaptor.getRs1()});
    rewriter.replaceOp(op, callOp.getResults());
    return success();
  }
};

/// Data Broadcast Instructions
struct XTAMEThMmovMvILowering : public ConvertOpToLLVMPattern<ThMmovMvIOp> {
  using ConvertOpToLLVMPattern<ThMmovMvIOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMmovMvIOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mmov.mv.i", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));
    Value uimm3Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getUimm3()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms1Val, uimm3Val});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMcmovbMvILowering : public ConvertOpToLLVMPattern<ThMcmovbMvIOp> {
  using ConvertOpToLLVMPattern<ThMcmovbMvIOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMcmovbMvIOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mcmovb.mv.i", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));
    Value uimm3Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getUimm3()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms1Val, uimm3Val});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMcmovhMvILowering : public ConvertOpToLLVMPattern<ThMcmovhMvIOp> {
  using ConvertOpToLLVMPattern<ThMcmovhMvIOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMcmovhMvIOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mcmovh.mv.i", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));
    Value uimm3Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getUimm3()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms1Val, uimm3Val});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMcmovwMvILowering : public ConvertOpToLLVMPattern<ThMcmovwMvIOp> {
  using ConvertOpToLLVMPattern<ThMcmovwMvIOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMcmovwMvIOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mcmovw.mv.i", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));
    Value uimm3Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getUimm3()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms1Val, uimm3Val});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMcmovdMvILowering : public ConvertOpToLLVMPattern<ThMcmovdMvIOp> {
  using ConvertOpToLLVMPattern<ThMcmovdMvIOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMcmovdMvIOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mcmovd.mv.i", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));
    Value uimm3Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getUimm3()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms1Val, uimm3Val});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Matrix Pack Instructions
struct XTAMEThMpackMmLowering : public ConvertOpToLLVMPattern<ThMpackMmOp> {
  using ConvertOpToLLVMPattern<ThMpackMmOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMpackMmOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mpack.mm", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms2Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms2Val, ms1Val});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMpackhlMmLowering : public ConvertOpToLLVMPattern<ThMpackhlMmOp> {
  using ConvertOpToLLVMPattern<ThMpackhlMmOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMpackhlMmOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mpackhl.mm", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms2Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms2Val, ms1Val});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMpackhhMmLowering : public ConvertOpToLLVMPattern<ThMpackhhMmOp> {
  using ConvertOpToLLVMPattern<ThMpackhhMmOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMpackhhMmOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mpackhh.mm", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms2Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms2Val, ms1Val});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Load Operations Lowering
//===----------------------------------------------------------------------===//

/// Lowering pattern for th.mlde8 (load tile register with 8-bit elements)
struct XTAMEThMlde8Lowering : public ConvertOpToLLVMPattern<ThMlde8Op> {
  using ConvertOpToLLVMPattern<ThMlde8Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMlde8Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    // type void (i64, i64, ptr)
    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mlde8", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{mdVal, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mlde16
struct XTAMEThMlde16Lowering : public ConvertOpToLLVMPattern<ThMlde16Op> {
  using ConvertOpToLLVMPattern<ThMlde16Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMlde16Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mlde16", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{mdVal, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mlde32
struct XTAMEThMlde32Lowering : public ConvertOpToLLVMPattern<ThMlde32Op> {
  using ConvertOpToLLVMPattern<ThMlde32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMlde32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mlde32", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{mdVal, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mlde64
struct XTAMEThMlde64Lowering : public ConvertOpToLLVMPattern<ThMlde64Op> {
  using ConvertOpToLLVMPattern<ThMlde64Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMlde64Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mlde64", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{mdVal, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mldte8 (matrix transposed load with 8-bit elements)
struct XTAMEThMldte8Lowering : public ConvertOpToLLVMPattern<ThMldte8Op> {
  using ConvertOpToLLVMPattern<ThMldte8Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMldte8Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mldte8", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{mdVal, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mldte16 (matrix transposed load with 16-bit
/// elements)
struct XTAMEThMldte16Lowering : public ConvertOpToLLVMPattern<ThMldte16Op> {
  using ConvertOpToLLVMPattern<ThMldte16Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMldte16Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mldte16", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{mdVal, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mldte32 (matrix transposed load with 32-bit
/// elements)
struct XTAMEThMldte32Lowering : public ConvertOpToLLVMPattern<ThMldte32Op> {
  using ConvertOpToLLVMPattern<ThMldte32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMldte32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mldte32", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{mdVal, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mldte64 (matrix transposed load with 64-bit
/// elements)
struct XTAMEThMldte64Lowering : public ConvertOpToLLVMPattern<ThMldte64Op> {
  using ConvertOpToLLVMPattern<ThMldte64Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMldte64Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mldte64", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{mdVal, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mslde8 (Stream load 8-bit matrix tile)
struct XTAMEThMslde8Lowering : public ConvertOpToLLVMPattern<ThMslde8Op> {
  using ConvertOpToLLVMPattern<ThMslde8Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMslde8Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mslde8", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{mdVal, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMslde16Lowering : public ConvertOpToLLVMPattern<ThMslde16Op> {
  using ConvertOpToLLVMPattern<ThMslde16Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMslde16Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mslde16", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{mdVal, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMslde32Lowering : public ConvertOpToLLVMPattern<ThMslde32Op> {
  using ConvertOpToLLVMPattern<ThMslde32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMslde32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mslde32", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{mdVal, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMslde64Lowering : public ConvertOpToLLVMPattern<ThMslde64Op> {
  using ConvertOpToLLVMPattern<ThMslde64Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMslde64Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mslde64", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{mdVal, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.msldte8 (Transposed stream load 8-bit matrix tile)
struct XTAMEThMsldte8Lowering : public ConvertOpToLLVMPattern<ThMsldte8Op> {
  using ConvertOpToLLVMPattern<ThMsldte8Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMsldte8Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.msldte8", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{mdVal, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMsldte16Lowering : public ConvertOpToLLVMPattern<ThMsldte16Op> {
  using ConvertOpToLLVMPattern<ThMsldte16Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMsldte16Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.msldte16", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{mdVal, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMsldte32Lowering : public ConvertOpToLLVMPattern<ThMsldte32Op> {
  using ConvertOpToLLVMPattern<ThMsldte32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMsldte32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.msldte32", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{mdVal, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMsldte64Lowering : public ConvertOpToLLVMPattern<ThMsldte64Op> {
  using ConvertOpToLLVMPattern<ThMsldte64Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMsldte64Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.msldte64", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{mdVal, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mplde8 (Prefetch 8-bit matrix tile)
struct XTAMEThMplde8Lowering : public ConvertOpToLLVMPattern<ThMplde8Op> {
  using ConvertOpToLLVMPattern<ThMplde8Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMplde8Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mplde8", funcType);

    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMplde16Lowering : public ConvertOpToLLVMPattern<ThMplde16Op> {
  using ConvertOpToLLVMPattern<ThMplde16Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMplde16Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mplde16", funcType);

    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMplde32Lowering : public ConvertOpToLLVMPattern<ThMplde32Op> {
  using ConvertOpToLLVMPattern<ThMplde32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMplde32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mplde32", funcType);

    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMplde64Lowering : public ConvertOpToLLVMPattern<ThMplde64Op> {
  using ConvertOpToLLVMPattern<ThMplde64Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMplde64Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mplde64", funcType);

    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mpldte8 (Transposed prefetch 8-bit matrix tile)
struct XTAMEThMpldte8Lowering : public ConvertOpToLLVMPattern<ThMpldte8Op> {
  using ConvertOpToLLVMPattern<ThMpldte8Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMpldte8Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mpldte8", funcType);

    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMpldte16Lowering : public ConvertOpToLLVMPattern<ThMpldte16Op> {
  using ConvertOpToLLVMPattern<ThMpldte16Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMpldte16Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mpldte16", funcType);

    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMpldte32Lowering : public ConvertOpToLLVMPattern<ThMpldte32Op> {
  using ConvertOpToLLVMPattern<ThMpldte32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMpldte32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mpldte32", funcType);

    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMpldte64Lowering : public ConvertOpToLLVMPattern<ThMpldte64Op> {
  using ConvertOpToLLVMPattern<ThMpldte64Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMpldte64Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mpldte64", funcType);

    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Store Operations Lowering
//===----------------------------------------------------------------------===//

/// Lowering pattern for th.mste8 (store output matrix with 8-bit elements)
struct XTAMEThMste8Lowering : public ConvertOpToLLVMPattern<ThMste8Op> {
  using ConvertOpToLLVMPattern<ThMste8Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMste8Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mste8", funcType);

    Value ms3Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs3()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{ms3Val, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mste16 (store output matrix with 16-bit elements)
struct XTAMEThMste16Lowering : public ConvertOpToLLVMPattern<ThMste16Op> {
  using ConvertOpToLLVMPattern<ThMste16Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMste16Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mste16", funcType);

    Value ms3Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs3()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{ms3Val, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mste32 (store output matrix with 32-bit elements)
struct XTAMEThMste32Lowering : public ConvertOpToLLVMPattern<ThMste32Op> {
  using ConvertOpToLLVMPattern<ThMste32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMste32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mste32", funcType);

    Value ms3Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs3()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{ms3Val, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mste64 (store output matrix with 64-bit elements)
struct XTAMEThMste64Lowering : public ConvertOpToLLVMPattern<ThMste64Op> {
  using ConvertOpToLLVMPattern<ThMste64Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMste64Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mste64", funcType);

    Value ms3Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs3()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{ms3Val, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mstte8 (store transposed output matrix with 8-bit
/// elements)
struct XTAMEThMstte8Lowering : public ConvertOpToLLVMPattern<ThMstte8Op> {
  using ConvertOpToLLVMPattern<ThMstte8Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMstte8Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mstte8", funcType);

    Value ms3Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs3()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{ms3Val, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMstte16Lowering : public ConvertOpToLLVMPattern<ThMstte16Op> {
  using ConvertOpToLLVMPattern<ThMstte16Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMstte16Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mstte16", funcType);

    Value ms3Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs3()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{ms3Val, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMstte32Lowering : public ConvertOpToLLVMPattern<ThMstte32Op> {
  using ConvertOpToLLVMPattern<ThMstte32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMstte32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mstte32", funcType);

    Value ms3Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs3()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{ms3Val, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMstte64Lowering : public ConvertOpToLLVMPattern<ThMstte64Op> {
  using ConvertOpToLLVMPattern<ThMstte64Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMstte64Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.mstte64", funcType);

    Value ms3Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs3()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{ms3Val, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.msste8 (store stream output matrix with 8-bit
/// elements)
struct XTAMEThMsste8Lowering : public ConvertOpToLLVMPattern<ThMsste8Op> {
  using ConvertOpToLLVMPattern<ThMsste8Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMsste8Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.msste8", funcType);

    Value ms3Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs3()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{ms3Val, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMsste16Lowering : public ConvertOpToLLVMPattern<ThMsste16Op> {
  using ConvertOpToLLVMPattern<ThMsste16Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMsste16Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.msste16", funcType);

    Value ms3Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs3()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{ms3Val, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMsste32Lowering : public ConvertOpToLLVMPattern<ThMsste32Op> {
  using ConvertOpToLLVMPattern<ThMsste32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMsste32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.msste32", funcType);

    Value ms3Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs3()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{ms3Val, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMsste64Lowering : public ConvertOpToLLVMPattern<ThMsste64Op> {
  using ConvertOpToLLVMPattern<ThMsste64Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMsste64Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.msste64", funcType);

    Value ms3Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs3()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{ms3Val, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.msstte8 (store transposed stream output matrix with
/// 8-bit elements)
struct XTAMEThMsstte8Lowering : public ConvertOpToLLVMPattern<ThMsstte8Op> {
  using ConvertOpToLLVMPattern<ThMsstte8Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMsstte8Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.msstte8", funcType);

    Value ms3Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs3()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{ms3Val, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMsstte16Lowering : public ConvertOpToLLVMPattern<ThMsstte16Op> {
  using ConvertOpToLLVMPattern<ThMsstte16Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMsstte16Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.msstte16", funcType);

    Value ms3Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs3()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{ms3Val, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMsstte32Lowering : public ConvertOpToLLVMPattern<ThMsstte32Op> {
  using ConvertOpToLLVMPattern<ThMsstte32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMsstte32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.msstte32", funcType);

    Value ms3Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs3()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{ms3Val, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

struct XTAMEThMsstte64Lowering : public ConvertOpToLLVMPattern<ThMsstte64Op> {
  using ConvertOpToLLVMPattern<ThMsstte64Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMsstte64Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                {i64Type, i64Type, ptrType});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.th.msstte64", funcType);

    Value ms3Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs3()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, intrinsicName,
        ValueRange{ms3Val, adaptor.getStride(), basePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Tile Register Matrix Multiply Lowering
//===----------------------------------------------------------------------===//

/// Lowering pattern for th.mmacc.w.b (tsigned 8bit, output quad-widened 16bit)
struct XTAMEThMmaccWBLowering : public ConvertOpToLLVMPattern<ThMmaccWBOp> {
  using ConvertOpToLLVMPattern<ThMmaccWBOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMmaccWBOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mmacc.w.b", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms2Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms2Val, ms1Val});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mmaccu.w.b (unsigned 8bit, output quad-widened
/// 16bit)
struct XTAMEThMmaccuWBLowering : public ConvertOpToLLVMPattern<ThMmaccuWBOp> {
  using ConvertOpToLLVMPattern<ThMmaccuWBOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMmaccuWBOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mmaccu.w.b", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms2Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms2Val, ms1Val});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mmaccus.w.b (unsigned-signed 8bit, output
/// quad-widened 16bit)
struct XTAMEThMmaccusWBLowering : public ConvertOpToLLVMPattern<ThMmaccusWBOp> {
  using ConvertOpToLLVMPattern<ThMmaccusWBOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMmaccusWBOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mmaccus.w.b", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms2Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms2Val, ms1Val});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mmaccsu.w.b (signed-unsigned 8bit, output
/// quad-widened 16bit)
struct XTAMEThMmaccsuWBLowering : public ConvertOpToLLVMPattern<ThMmaccsuWBOp> {
  using ConvertOpToLLVMPattern<ThMmaccsuWBOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMmaccsuWBOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mmaccsu.w.b", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms2Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms2Val, ms1Val});
    rewriter.eraseOp(op);
    return success();
  }
};

// ===----------------------------------------------------------------------===//
// Tile Register Matrix Multiply Lowering (float-point types)
// ===----------------------------------------------------------------------===//

/// Lowering pattern for th.mfmacc.h (16-bit float point(fp16), output fp16)
struct XTAMEThMfmaccHLowering : public ConvertOpToLLVMPattern<ThMfmaccHOp> {
  using ConvertOpToLLVMPattern<ThMfmaccHOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMfmaccHOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mfmacc.h", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms2Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms2Val, ms1Val});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mfmacc.bf16 (16-bit float point(bf16), output fp16)
struct XTAMEThMfmaccBf16Lowering
    : public ConvertOpToLLVMPattern<ThMfmaccBf16Op> {
  using ConvertOpToLLVMPattern<ThMfmaccBf16Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMfmaccBf16Op op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mfmacc.bf16", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms2Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms2Val, ms1Val});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mfmacc.s (32-bit float point(fp32), output fp32)
struct XTAMEThMfmaccSLowering : public ConvertOpToLLVMPattern<ThMfmaccSOp> {
  using ConvertOpToLLVMPattern<ThMfmaccSOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMfmaccSOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mfmacc.s", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms2Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms2Val, ms1Val});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mfmacc.d (64-bit float point (fp64), output fp64)
struct XTAMEThMfmaccDLowering : public ConvertOpToLLVMPattern<ThMfmaccDOp> {
  using ConvertOpToLLVMPattern<ThMfmaccDOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMfmaccDOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mfmacc.d", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms2Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms2Val, ms1Val});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mfmacc.h.e4m3
///(8-bit float point, output double-widened 16-bit float point)
struct XTAMEThMfmaccHE4m3Lowering
    : public ConvertOpToLLVMPattern<ThMfmaccHE4m3Op> {
  using ConvertOpToLLVMPattern<ThMfmaccHE4m3Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMfmaccHE4m3Op op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mfmacc.h.e4m3", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms2Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms2Val, ms1Val});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mfmacc.h.e5m2
///(8-bit float point, output double-widened 16-bit float point)
struct XTAMEThMfmaccHE5m2Lowering
    : public ConvertOpToLLVMPattern<ThMfmaccHE5m2Op> {
  using ConvertOpToLLVMPattern<ThMfmaccHE5m2Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMfmaccHE5m2Op op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mfmacc.h.e5m2", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms2Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms2Val, ms1Val});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mfmacc.bf16.e4m3
///(8-bit float point, output double-widen ed 16-bit float point)
struct XTAMEThMfmaccBf16E4m3Lowering
    : public ConvertOpToLLVMPattern<ThMfmaccBf16E4m3Op> {
  using ConvertOpToLLVMPattern<ThMfmaccBf16E4m3Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMfmaccBf16E4m3Op op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mfmacc.bf16.e4m3", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms2Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms2Val, ms1Val});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mfmacc.bf16.e5m2
///(8-bit float point, output double-widen ed 16-bit float point)
struct XTAMEThMfmaccBf16E5m2Lowering
    : public ConvertOpToLLVMPattern<ThMfmaccBf16E5m2Op> {
  using ConvertOpToLLVMPattern<ThMfmaccBf16E5m2Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMfmaccBf16E5m2Op op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mfmacc.bf16.e5m2", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms2Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms2Val, ms1Val});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mfmacc.s.h
///(16-bit float point(fp16), output 32-bit float point(fp32))
struct XTAMEThMfmaccSHLowering : public ConvertOpToLLVMPattern<ThMfmaccSHOp> {
  using ConvertOpToLLVMPattern<ThMfmaccSHOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMfmaccSHOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mfmacc.s.h", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms2Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms2Val, ms1Val});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mfmacc.s.bf16
///(16-bit float point(bf16), output 32-bit float point(fp32))
struct XTAMEThMfmaccSBf16Lowering
    : public ConvertOpToLLVMPattern<ThMfmaccSBf16Op> {
  using ConvertOpToLLVMPattern<ThMfmaccSBf16Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMfmaccSBf16Op op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mfmacc.s.bf16", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms2Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms2Val, ms1Val});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mfmacc.d.s
///(32-bit float point(fp32), output 64-bit float point(fp64))
struct XTAMEThMfmaccDSLowering : public ConvertOpToLLVMPattern<ThMfmaccDSOp> {
  using ConvertOpToLLVMPattern<ThMfmaccDSOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMfmaccDSOp op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mfmacc.d.s", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms2Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms2Val, ms1Val});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mfmacc.s.e4m3
///(8-bit float point, output 32-bit float point)
struct XTAMEThMfmaccSE4m3Lowering
    : public ConvertOpToLLVMPattern<ThMfmaccSE4m3Op> {
  using ConvertOpToLLVMPattern<ThMfmaccSE4m3Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMfmaccSE4m3Op op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mfmacc.s.e4m3", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms2Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms2Val, ms1Val});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for th.mfmacc.s.e5m2
///(8-bit float point, output 32-bit float point)
struct XTAMEThMfmaccSE5m2Lowering
    : public ConvertOpToLLVMPattern<ThMfmaccSE5m2Op> {
  using ConvertOpToLLVMPattern<ThMfmaccSE5m2Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThMfmaccSE5m2Op op, OpAdaptor adaptor,
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
        rewriter, module, "llvm.riscv.buddy.th.mfmacc.s.e5m2", funcType);

    Value mdVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms2Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));
    Value ms1Val = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, intrinsicName,
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

    // Configuration operations
    target.addIllegalOp<ThMcfgOp>();
    target.addIllegalOp<ThMcfgmOp>();
    target.addIllegalOp<ThMcfgnOp>();
    target.addIllegalOp<ThMcfgkOp>();
    target.addIllegalOp<ThMcfgmiOp>();
    target.addIllegalOp<ThMcfgniOp>();
    target.addIllegalOp<ThMcfgkiOp>();

    // MISC operations
    target.addIllegalOp<ThMzeroOp>();
    target.addIllegalOp<ThMzero2rOp>();
    target.addIllegalOp<ThMzero4rOp>();
    target.addIllegalOp<ThMzero8rOp>();
    target.addIllegalOp<ThMmovMmOp>();
    target.addIllegalOp<ThMdupbMXOp>();
    target.addIllegalOp<ThMduphMXOp>();
    target.addIllegalOp<ThMdupwMXOp>();
    target.addIllegalOp<ThMdupdMXOp>();
    target.addIllegalOp<ThMmovbMXOp>();
    target.addIllegalOp<ThMmovhMXOp>();
    target.addIllegalOp<ThMmovwMXOp>();
    target.addIllegalOp<ThMmovdMXOp>();
    target.addIllegalOp<ThMmovbXMOp>();
    target.addIllegalOp<ThMmovhXMOp>();
    target.addIllegalOp<ThMmovwXMOp>();
    target.addIllegalOp<ThMmovdXMOp>();
    target.addIllegalOp<ThMmovMvIOp>();
    target.addIllegalOp<ThMcmovbMvIOp>();
    target.addIllegalOp<ThMcmovhMvIOp>();
    target.addIllegalOp<ThMcmovwMvIOp>();
    target.addIllegalOp<ThMcmovdMvIOp>();
    target.addIllegalOp<ThMpackMmOp>();
    target.addIllegalOp<ThMpackhlMmOp>();
    target.addIllegalOp<ThMpackhhMmOp>();

    // Load/Store operations
    target.addIllegalOp<ThMlde8Op>();
    target.addIllegalOp<ThMlde16Op>();
    target.addIllegalOp<ThMlde32Op>();
    target.addIllegalOp<ThMlde64Op>();
    target.addIllegalOp<ThMldte8Op>();
    target.addIllegalOp<ThMldte16Op>();
    target.addIllegalOp<ThMldte32Op>();
    target.addIllegalOp<ThMldte64Op>();
    target.addIllegalOp<ThMslde8Op>();
    target.addIllegalOp<ThMslde16Op>();
    target.addIllegalOp<ThMslde32Op>();
    target.addIllegalOp<ThMslde64Op>();
    target.addIllegalOp<ThMsldte8Op>();
    target.addIllegalOp<ThMsldte16Op>();
    target.addIllegalOp<ThMsldte32Op>();
    target.addIllegalOp<ThMsldte64Op>();
    target.addIllegalOp<ThMste8Op>();
    target.addIllegalOp<ThMste16Op>();
    target.addIllegalOp<ThMste32Op>();
    target.addIllegalOp<ThMste64Op>();
    target.addIllegalOp<ThMstte8Op>();
    target.addIllegalOp<ThMstte16Op>();
    target.addIllegalOp<ThMstte32Op>();
    target.addIllegalOp<ThMstte64Op>();
    target.addIllegalOp<ThMsste8Op>();
    target.addIllegalOp<ThMsste16Op>();
    target.addIllegalOp<ThMsste32Op>();
    target.addIllegalOp<ThMsste64Op>();
    target.addIllegalOp<ThMsstte8Op>();
    target.addIllegalOp<ThMsstte16Op>();
    target.addIllegalOp<ThMsstte32Op>();
    target.addIllegalOp<ThMsstte64Op>();
    target.addIllegalOp<ThMplde8Op>();
    target.addIllegalOp<ThMplde16Op>();
    target.addIllegalOp<ThMplde32Op>();
    target.addIllegalOp<ThMplde64Op>();
    target.addIllegalOp<ThMpldte8Op>();
    target.addIllegalOp<ThMpldte16Op>();
    target.addIllegalOp<ThMpldte32Op>();
    target.addIllegalOp<ThMpldte64Op>();

    // Tile register matrix multiply
    target.addIllegalOp<ThMmaccWBOp>();
    target.addIllegalOp<ThMmaccuWBOp>();
    target.addIllegalOp<ThMmaccusWBOp>();
    target.addIllegalOp<ThMmaccsuWBOp>();

    // Tile register matrix multiply (float-point types)
    target.addIllegalOp<ThMfmaccHOp>();
    target.addIllegalOp<ThMfmaccBf16Op>();
    target.addIllegalOp<ThMfmaccSOp>();
    target.addIllegalOp<ThMfmaccDOp>();
    target.addIllegalOp<ThMfmaccHE4m3Op>();
    target.addIllegalOp<ThMfmaccHE5m2Op>();
    target.addIllegalOp<ThMfmaccBf16E4m3Op>();
    target.addIllegalOp<ThMfmaccBf16E5m2Op>();
    target.addIllegalOp<ThMfmaccSHOp>();
    target.addIllegalOp<ThMfmaccSBf16Op>();
    target.addIllegalOp<ThMfmaccDSOp>();
    target.addIllegalOp<ThMfmaccSE4m3Op>();
    target.addIllegalOp<ThMfmaccSE5m2Op>();

    LLVMTypeConverter typeConverter(&context);

    RewritePatternSet patterns(&context);

    // Configuration patterns
    patterns.add<XTAMEThMcfgLowering>(typeConverter);
    patterns.add<XTAMEThMcfgmLowering>(typeConverter);
    patterns.add<XTAMEThMcfgnLowering>(typeConverter);
    patterns.add<XTAMEThMcfgkLowering>(typeConverter);
    patterns.add<XTAMEThMcfgmiLowering>(typeConverter);
    patterns.add<XTAMEThMcfgniLowering>(typeConverter);
    patterns.add<XTAMEThMcfgkiLowering>(typeConverter);

    // MISC patterns
    patterns.add<XTAMEThMzeroLowering>(typeConverter);
    patterns.add<XTAMEThMzero2rLowering>(typeConverter);
    patterns.add<XTAMEThMzero4rLowering>(typeConverter);
    patterns.add<XTAMEThMzero8rLowering>(typeConverter);
    patterns.add<XTAMEThMmovMmLowering>(typeConverter);
    patterns.add<XTAMEThMdupbMXLowering>(typeConverter);
    patterns.add<XTAMEThMduphMXLowering>(typeConverter);
    patterns.add<XTAMEThMdupwMXLowering>(typeConverter);
    patterns.add<XTAMEThMdupdMXLowering>(typeConverter);
    patterns.add<XTAMEThMmovbMXLowering>(typeConverter);
    patterns.add<XTAMEThMmovhMXLowering>(typeConverter);
    patterns.add<XTAMEThMmovwMXLowering>(typeConverter);
    patterns.add<XTAMEThMmovdMXLowering>(typeConverter);
    patterns.add<XTAMEThMmovbXMLowering>(typeConverter);
    patterns.add<XTAMEThMmovhXMLowering>(typeConverter);
    patterns.add<XTAMEThMmovwXMLowering>(typeConverter);
    patterns.add<XTAMEThMmovdXMLowering>(typeConverter);
    patterns.add<XTAMEThMmovMvILowering>(typeConverter);
    patterns.add<XTAMEThMcmovbMvILowering>(typeConverter);
    patterns.add<XTAMEThMcmovhMvILowering>(typeConverter);
    patterns.add<XTAMEThMcmovwMvILowering>(typeConverter);
    patterns.add<XTAMEThMcmovdMvILowering>(typeConverter);
    patterns.add<XTAMEThMpackMmLowering>(typeConverter);
    patterns.add<XTAMEThMpackhlMmLowering>(typeConverter);
    patterns.add<XTAMEThMpackhhMmLowering>(typeConverter);

    // Load/Store patterns
    patterns.add<XTAMEThMlde8Lowering>(typeConverter);
    patterns.add<XTAMEThMlde16Lowering>(typeConverter);
    patterns.add<XTAMEThMlde32Lowering>(typeConverter);
    patterns.add<XTAMEThMlde64Lowering>(typeConverter);
    patterns.add<XTAMEThMldte8Lowering>(typeConverter);
    patterns.add<XTAMEThMldte16Lowering>(typeConverter);
    patterns.add<XTAMEThMldte32Lowering>(typeConverter);
    patterns.add<XTAMEThMldte64Lowering>(typeConverter);
    patterns.add<XTAMEThMslde8Lowering>(typeConverter);
    patterns.add<XTAMEThMslde16Lowering>(typeConverter);
    patterns.add<XTAMEThMslde32Lowering>(typeConverter);
    patterns.add<XTAMEThMslde64Lowering>(typeConverter);
    patterns.add<XTAMEThMsldte8Lowering>(typeConverter);
    patterns.add<XTAMEThMsldte16Lowering>(typeConverter);
    patterns.add<XTAMEThMsldte32Lowering>(typeConverter);
    patterns.add<XTAMEThMsldte64Lowering>(typeConverter);

    patterns.add<XTAMEThMplde8Lowering>(typeConverter);
    patterns.add<XTAMEThMplde16Lowering>(typeConverter);
    patterns.add<XTAMEThMplde32Lowering>(typeConverter);
    patterns.add<XTAMEThMplde64Lowering>(typeConverter);
    patterns.add<XTAMEThMpldte8Lowering>(typeConverter);
    patterns.add<XTAMEThMpldte16Lowering>(typeConverter);
    patterns.add<XTAMEThMpldte32Lowering>(typeConverter);
    patterns.add<XTAMEThMpldte64Lowering>(typeConverter);

    patterns.add<XTAMEThMste8Lowering>(typeConverter);
    patterns.add<XTAMEThMste16Lowering>(typeConverter);
    patterns.add<XTAMEThMste32Lowering>(typeConverter);
    patterns.add<XTAMEThMste64Lowering>(typeConverter);
    patterns.add<XTAMEThMstte8Lowering>(typeConverter);
    patterns.add<XTAMEThMstte16Lowering>(typeConverter);
    patterns.add<XTAMEThMstte32Lowering>(typeConverter);
    patterns.add<XTAMEThMstte64Lowering>(typeConverter);
    patterns.add<XTAMEThMsste8Lowering>(typeConverter);
    patterns.add<XTAMEThMsste16Lowering>(typeConverter);
    patterns.add<XTAMEThMsste32Lowering>(typeConverter);
    patterns.add<XTAMEThMsste64Lowering>(typeConverter);
    patterns.add<XTAMEThMsstte8Lowering>(typeConverter);
    patterns.add<XTAMEThMsstte16Lowering>(typeConverter);
    patterns.add<XTAMEThMsstte32Lowering>(typeConverter);
    patterns.add<XTAMEThMsstte64Lowering>(typeConverter);

    // Tile register matrix multiply patterns
    patterns.add<XTAMEThMmaccWBLowering>(typeConverter);
    patterns.add<XTAMEThMmaccuWBLowering>(typeConverter);
    patterns.add<XTAMEThMmaccusWBLowering>(typeConverter);
    patterns.add<XTAMEThMmaccsuWBLowering>(typeConverter);

    // Tile register matrix multiply patterns (float-point types)
    patterns.add<XTAMEThMfmaccHLowering>(typeConverter);
    patterns.add<XTAMEThMfmaccBf16Lowering>(typeConverter);
    patterns.add<XTAMEThMfmaccSLowering>(typeConverter);
    patterns.add<XTAMEThMfmaccDLowering>(typeConverter);
    patterns.add<XTAMEThMfmaccHE4m3Lowering>(typeConverter);
    patterns.add<XTAMEThMfmaccHE5m2Lowering>(typeConverter);
    patterns.add<XTAMEThMfmaccBf16E4m3Lowering>(typeConverter);
    patterns.add<XTAMEThMfmaccBf16E5m2Lowering>(typeConverter);
    patterns.add<XTAMEThMfmaccSHLowering>(typeConverter);
    patterns.add<XTAMEThMfmaccSBf16Lowering>(typeConverter);
    patterns.add<XTAMEThMfmaccDSLowering>(typeConverter);
    patterns.add<XTAMEThMfmaccSE4m3Lowering>(typeConverter);
    patterns.add<XTAMEThMfmaccSE5m2Lowering>(typeConverter);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::populateXTAMELegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // Configuration patterns
  patterns.add<XTAMEThMcfgLowering>(converter);
  patterns.add<XTAMEThMcfgmLowering>(converter);
  patterns.add<XTAMEThMcfgnLowering>(converter);
  patterns.add<XTAMEThMcfgkLowering>(converter);
  patterns.add<XTAMEThMcfgmiLowering>(converter);
  patterns.add<XTAMEThMcfgniLowering>(converter);
  patterns.add<XTAMEThMcfgkiLowering>(converter);

  // MISC patterns
  patterns.add<XTAMEThMzeroLowering>(converter);
  patterns.add<XTAMEThMzero2rLowering>(converter);
  patterns.add<XTAMEThMzero4rLowering>(converter);
  patterns.add<XTAMEThMzero8rLowering>(converter);
  patterns.add<XTAMEThMmovMmLowering>(converter);
  patterns.add<XTAMEThMdupbMXLowering>(converter);
  patterns.add<XTAMEThMduphMXLowering>(converter);
  patterns.add<XTAMEThMdupwMXLowering>(converter);
  patterns.add<XTAMEThMdupdMXLowering>(converter);
  patterns.add<XTAMEThMmovbMXLowering>(converter);
  patterns.add<XTAMEThMmovhMXLowering>(converter);
  patterns.add<XTAMEThMmovwMXLowering>(converter);
  patterns.add<XTAMEThMmovdMXLowering>(converter);
  patterns.add<XTAMEThMmovbXMLowering>(converter);
  patterns.add<XTAMEThMmovhXMLowering>(converter);
  patterns.add<XTAMEThMmovwXMLowering>(converter);
  patterns.add<XTAMEThMmovdXMLowering>(converter);
  patterns.add<XTAMEThMmovMvILowering>(converter);
  patterns.add<XTAMEThMcmovbMvILowering>(converter);
  patterns.add<XTAMEThMcmovhMvILowering>(converter);
  patterns.add<XTAMEThMcmovwMvILowering>(converter);
  patterns.add<XTAMEThMcmovdMvILowering>(converter);
  patterns.add<XTAMEThMpackMmLowering>(converter);
  patterns.add<XTAMEThMpackhlMmLowering>(converter);
  patterns.add<XTAMEThMpackhhMmLowering>(converter);

  // Load/Store patterns
  patterns.add<XTAMEThMlde8Lowering>(converter);
  patterns.add<XTAMEThMlde16Lowering>(converter);
  patterns.add<XTAMEThMlde32Lowering>(converter);
  patterns.add<XTAMEThMlde64Lowering>(converter);
  patterns.add<XTAMEThMldte8Lowering>(converter);
  patterns.add<XTAMEThMldte16Lowering>(converter);
  patterns.add<XTAMEThMldte32Lowering>(converter);
  patterns.add<XTAMEThMldte64Lowering>(converter);
  patterns.add<XTAMEThMslde8Lowering>(converter);
  patterns.add<XTAMEThMslde16Lowering>(converter);
  patterns.add<XTAMEThMslde32Lowering>(converter);
  patterns.add<XTAMEThMslde64Lowering>(converter);
  patterns.add<XTAMEThMsldte8Lowering>(converter);
  patterns.add<XTAMEThMsldte16Lowering>(converter);
  patterns.add<XTAMEThMsldte32Lowering>(converter);
  patterns.add<XTAMEThMsldte64Lowering>(converter);

  patterns.add<XTAMEThMplde8Lowering>(converter);
  patterns.add<XTAMEThMplde16Lowering>(converter);
  patterns.add<XTAMEThMplde32Lowering>(converter);
  patterns.add<XTAMEThMplde64Lowering>(converter);
  patterns.add<XTAMEThMpldte8Lowering>(converter);
  patterns.add<XTAMEThMpldte16Lowering>(converter);
  patterns.add<XTAMEThMpldte32Lowering>(converter);
  patterns.add<XTAMEThMpldte64Lowering>(converter);

  patterns.add<XTAMEThMste8Lowering>(converter);
  patterns.add<XTAMEThMste16Lowering>(converter);
  patterns.add<XTAMEThMste32Lowering>(converter);
  patterns.add<XTAMEThMste64Lowering>(converter);
  patterns.add<XTAMEThMstte8Lowering>(converter);
  patterns.add<XTAMEThMstte16Lowering>(converter);
  patterns.add<XTAMEThMstte32Lowering>(converter);
  patterns.add<XTAMEThMstte64Lowering>(converter);
  patterns.add<XTAMEThMsste8Lowering>(converter);
  patterns.add<XTAMEThMsste16Lowering>(converter);
  patterns.add<XTAMEThMsste32Lowering>(converter);
  patterns.add<XTAMEThMsste64Lowering>(converter);
  patterns.add<XTAMEThMsstte8Lowering>(converter);
  patterns.add<XTAMEThMsstte16Lowering>(converter);
  patterns.add<XTAMEThMsstte32Lowering>(converter);
  patterns.add<XTAMEThMsstte64Lowering>(converter);

  // Tile register matrix multiply patterns
  patterns.add<XTAMEThMmaccWBLowering>(converter);
  patterns.add<XTAMEThMmaccuWBLowering>(converter);
  patterns.add<XTAMEThMmaccusWBLowering>(converter);
  patterns.add<XTAMEThMmaccsuWBLowering>(converter);

  // Tile register matrix multiply patterns (float-point types)
  patterns.add<XTAMEThMfmaccHLowering>(converter);
  patterns.add<XTAMEThMfmaccBf16Lowering>(converter);
  patterns.add<XTAMEThMfmaccSLowering>(converter);
  patterns.add<XTAMEThMfmaccDLowering>(converter);
  patterns.add<XTAMEThMfmaccHE4m3Lowering>(converter);
  patterns.add<XTAMEThMfmaccHE5m2Lowering>(converter);
  patterns.add<XTAMEThMfmaccBf16E4m3Lowering>(converter);
  patterns.add<XTAMEThMfmaccBf16E5m2Lowering>(converter);
  patterns.add<XTAMEThMfmaccSHLowering>(converter);
  patterns.add<XTAMEThMfmaccSBf16Lowering>(converter);
  patterns.add<XTAMEThMfmaccDSLowering>(converter);
  patterns.add<XTAMEThMfmaccSE4m3Lowering>(converter);
  patterns.add<XTAMEThMfmaccSE5m2Lowering>(converter);
}

void mlir::configureXTAMELegalizeForExportTarget(LLVMConversionTarget &target) {
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<memref::MemRefDialect>();

  // Configuration operations
  target.addIllegalOp<ThMcfgOp>();
  target.addIllegalOp<ThMcfgmOp>();
  target.addIllegalOp<ThMcfgnOp>();
  target.addIllegalOp<ThMcfgkOp>();
  target.addIllegalOp<ThMcfgmiOp>();
  target.addIllegalOp<ThMcfgniOp>();
  target.addIllegalOp<ThMcfgkiOp>();

  // MISC operations
  target.addIllegalOp<ThMzeroOp>();
  target.addIllegalOp<ThMzero2rOp>();
  target.addIllegalOp<ThMzero4rOp>();
  target.addIllegalOp<ThMzero8rOp>();
  target.addIllegalOp<ThMmovMmOp>();
  target.addIllegalOp<ThMdupbMXOp>();
  target.addIllegalOp<ThMduphMXOp>();
  target.addIllegalOp<ThMdupwMXOp>();
  target.addIllegalOp<ThMdupdMXOp>();
  target.addIllegalOp<ThMmovbMXOp>();
  target.addIllegalOp<ThMmovhMXOp>();
  target.addIllegalOp<ThMmovwMXOp>();
  target.addIllegalOp<ThMmovdMXOp>();
  target.addIllegalOp<ThMmovbXMOp>();
  target.addIllegalOp<ThMmovhXMOp>();
  target.addIllegalOp<ThMmovwXMOp>();
  target.addIllegalOp<ThMmovdXMOp>();
  target.addIllegalOp<ThMmovMvIOp>();
  target.addIllegalOp<ThMcmovbMvIOp>();
  target.addIllegalOp<ThMcmovhMvIOp>();
  target.addIllegalOp<ThMcmovwMvIOp>();
  target.addIllegalOp<ThMcmovdMvIOp>();
  target.addIllegalOp<ThMpackMmOp>();
  target.addIllegalOp<ThMpackhlMmOp>();
  target.addIllegalOp<ThMpackhhMmOp>();

  // Load/Store operations
  target.addIllegalOp<ThMlde8Op>();
  target.addIllegalOp<ThMlde16Op>();
  target.addIllegalOp<ThMlde32Op>();
  target.addIllegalOp<ThMlde64Op>();
  target.addIllegalOp<ThMldte8Op>();
  target.addIllegalOp<ThMldte16Op>();
  target.addIllegalOp<ThMldte32Op>();
  target.addIllegalOp<ThMldte64Op>();
  target.addIllegalOp<ThMslde8Op>();
  target.addIllegalOp<ThMslde16Op>();
  target.addIllegalOp<ThMslde32Op>();
  target.addIllegalOp<ThMslde64Op>();
  target.addIllegalOp<ThMsldte8Op>();
  target.addIllegalOp<ThMsldte16Op>();
  target.addIllegalOp<ThMsldte32Op>();
  target.addIllegalOp<ThMsldte64Op>();
  target.addIllegalOp<ThMplde8Op>();
  target.addIllegalOp<ThMplde16Op>();
  target.addIllegalOp<ThMplde32Op>();
  target.addIllegalOp<ThMplde64Op>();
  target.addIllegalOp<ThMpldte8Op>();
  target.addIllegalOp<ThMpldte16Op>();
  target.addIllegalOp<ThMpldte32Op>();
  target.addIllegalOp<ThMpldte64Op>();

  target.addIllegalOp<ThMste8Op>();
  target.addIllegalOp<ThMste16Op>();
  target.addIllegalOp<ThMste32Op>();
  target.addIllegalOp<ThMste64Op>();
  target.addIllegalOp<ThMstte8Op>();
  target.addIllegalOp<ThMstte16Op>();
  target.addIllegalOp<ThMstte32Op>();
  target.addIllegalOp<ThMstte64Op>();
  target.addIllegalOp<ThMsste8Op>();
  target.addIllegalOp<ThMsste16Op>();
  target.addIllegalOp<ThMsste32Op>();
  target.addIllegalOp<ThMsste64Op>();
  target.addIllegalOp<ThMsstte8Op>();
  target.addIllegalOp<ThMsstte16Op>();
  target.addIllegalOp<ThMsstte32Op>();
  target.addIllegalOp<ThMsstte64Op>();

  // Tile register matrix multiply
  target.addIllegalOp<ThMmaccWBOp>();
  target.addIllegalOp<ThMmaccuWBOp>();
  target.addIllegalOp<ThMmaccusWBOp>();
  target.addIllegalOp<ThMmaccsuWBOp>();

  // Tile register matrix multiply (float-point types)
  target.addIllegalOp<ThMfmaccHOp>();
  target.addIllegalOp<ThMfmaccBf16Op>();
  target.addIllegalOp<ThMfmaccSOp>();
  target.addIllegalOp<ThMfmaccDOp>();
  target.addIllegalOp<ThMfmaccHE4m3Op>();
  target.addIllegalOp<ThMfmaccHE5m2Op>();
  target.addIllegalOp<ThMfmaccBf16E4m3Op>();
  target.addIllegalOp<ThMfmaccBf16E5m2Op>();
  target.addIllegalOp<ThMfmaccSHOp>();
  target.addIllegalOp<ThMfmaccSBf16Op>();
  target.addIllegalOp<ThMfmaccDSOp>();
  target.addIllegalOp<ThMfmaccSE4m3Op>();
  target.addIllegalOp<ThMfmaccSE5m2Op>();
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
