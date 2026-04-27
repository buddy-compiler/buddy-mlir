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
    target.addIllegalOp<ThMcfgmiOp>();
    target.addIllegalOp<ThMcfgniOp>();
    target.addIllegalOp<ThMcfgkiOp>();
    target.addIllegalOp<ThMzeroOp>();

    // Load/Store operations
    target.addIllegalOp<ThMlde8Op>();
    target.addIllegalOp<ThMlde16Op>();
    target.addIllegalOp<ThMlde32Op>();
    target.addIllegalOp<ThMlde64Op>();
    target.addIllegalOp<ThMldte8Op>();
    target.addIllegalOp<ThMldte16Op>();
    target.addIllegalOp<ThMldte32Op>();
    target.addIllegalOp<ThMldte64Op>();
    target.addIllegalOp<ThMste8Op>();
    target.addIllegalOp<ThMste16Op>();
    target.addIllegalOp<ThMste32Op>();
    target.addIllegalOp<ThMste64Op>();

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
    patterns.add<XTAMEThMcfgmiLowering>(typeConverter);
    patterns.add<XTAMEThMcfgniLowering>(typeConverter);
    patterns.add<XTAMEThMcfgkiLowering>(typeConverter);
    patterns.add<XTAMEThMzeroLowering>(typeConverter);

    // Load/Store patterns
    patterns.add<XTAMEThMlde8Lowering>(typeConverter);
    patterns.add<XTAMEThMlde16Lowering>(typeConverter);
    patterns.add<XTAMEThMlde32Lowering>(typeConverter);
    patterns.add<XTAMEThMlde64Lowering>(typeConverter);
    patterns.add<XTAMEThMldte8Lowering>(typeConverter);
    patterns.add<XTAMEThMldte16Lowering>(typeConverter);
    patterns.add<XTAMEThMldte32Lowering>(typeConverter);
    patterns.add<XTAMEThMldte64Lowering>(typeConverter);
    patterns.add<XTAMEThMste8Lowering>(typeConverter);
    patterns.add<XTAMEThMste16Lowering>(typeConverter);
    patterns.add<XTAMEThMste32Lowering>(typeConverter);
    patterns.add<XTAMEThMste64Lowering>(typeConverter);

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
  patterns.add<XTAMEThMcfgmiLowering>(converter);
  patterns.add<XTAMEThMcfgniLowering>(converter);
  patterns.add<XTAMEThMcfgkiLowering>(converter);
  patterns.add<XTAMEThMzeroLowering>(converter);

  // Load/Store patterns
  patterns.add<XTAMEThMlde8Lowering>(converter);
  patterns.add<XTAMEThMlde16Lowering>(converter);
  patterns.add<XTAMEThMlde32Lowering>(converter);
  patterns.add<XTAMEThMlde64Lowering>(converter);
  patterns.add<XTAMEThMldte8Lowering>(converter);
  patterns.add<XTAMEThMldte16Lowering>(converter);
  patterns.add<XTAMEThMldte32Lowering>(converter);
  patterns.add<XTAMEThMldte64Lowering>(converter);
  patterns.add<XTAMEThMste8Lowering>(converter);
  patterns.add<XTAMEThMste16Lowering>(converter);
  patterns.add<XTAMEThMste32Lowering>(converter);
  patterns.add<XTAMEThMste64Lowering>(converter);

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
  target.addIllegalOp<ThMcfgmiOp>();
  target.addIllegalOp<ThMcfgniOp>();
  target.addIllegalOp<ThMcfgkiOp>();
  target.addIllegalOp<ThMzeroOp>();

  // Load/Store operations
  target.addIllegalOp<ThMlde8Op>();
  target.addIllegalOp<ThMlde16Op>();
  target.addIllegalOp<ThMlde32Op>();
  target.addIllegalOp<ThMlde64Op>();
  target.addIllegalOp<ThMldte8Op>();
  target.addIllegalOp<ThMldte16Op>();
  target.addIllegalOp<ThMldte32Op>();
  target.addIllegalOp<ThMldte64Op>();
  target.addIllegalOp<ThMste8Op>();
  target.addIllegalOp<ThMste16Op>();
  target.addIllegalOp<ThMste32Op>();
  target.addIllegalOp<ThMste64Op>();

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
