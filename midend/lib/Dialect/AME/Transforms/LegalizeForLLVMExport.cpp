//====- LegalizeForLLVMExport.cpp - Prepare AME for LLVM translation ------===//
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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "Dialect/AME/AMEDialect.h"
#include "Dialect/AME/AMEOps.h"
#include "Dialect/AME/Transform.h"

using namespace mlir;
using namespace buddy::ame;

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
      memref::ExtractAlignedPointerAsIndexOp::create(rewriter, loc, memref);
  Value i64Val = arith::IndexCastOp::create(rewriter, loc, i64Type, idx);
  Value ptr = LLVM::IntToPtrOp::create(rewriter, loc, ptrType, i64Val);
  return ptr;
}

//===----------------------------------------------------------------------===//
// AME Lowering Patterns
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Configuration Operations Lowering
//===----------------------------------------------------------------------===//

/// Lowering pattern for msettilemi (set tile M dimension with immediate)
struct AMEMSettilemiLowering : public ConvertOpToLLVMPattern<MSettilemiOp> {
  using ConvertOpToLLVMPattern<MSettilemiOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MSettilemiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module) return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), {i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.msettilemi", funcType);

    Value tilemVal = LLVM::ConstantOp::create(rewriter, 
        loc, i64Type, rewriter.getI64IntegerAttr(op.getTilem()));

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, intrinsicName,
                                  ValueRange{tilemVal});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for msettileni
struct AMEMSettileniLowering : public ConvertOpToLLVMPattern<MSettileniOp> {
  using ConvertOpToLLVMPattern<MSettileniOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MSettileniOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module) return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), {i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.msettileni", funcType);

    Value tilenVal = LLVM::ConstantOp::create(rewriter, 
        loc, i64Type, rewriter.getI64IntegerAttr(op.getTilen()));

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, intrinsicName,
                                  ValueRange{tilenVal});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for msettileki
struct AMEMSettilekiLowering : public ConvertOpToLLVMPattern<MSettilekiOp> {
  using ConvertOpToLLVMPattern<MSettilekiOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MSettilekiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module) return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), {i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.msettileki", funcType);

    Value tilekVal = LLVM::ConstantOp::create(rewriter, 
        loc, i64Type, rewriter.getI64IntegerAttr(op.getTilek()));

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, intrinsicName,
                                  ValueRange{tilekVal});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for mzero
struct AMEMzeroLowering : public ConvertOpToLLVMPattern<MzeroOp> {
  using ConvertOpToLLVMPattern<MzeroOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MzeroOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module) return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), {i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.mzero", funcType);

    Value mdVal = LLVM::ConstantOp::create(rewriter, 
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Load Operations Lowering
//===----------------------------------------------------------------------===//

/// Lowering pattern for mlae32.m (load left matrix A)
struct AMEMlae32mLowering : public ConvertOpToLLVMPattern<Mlae32mOp> {
  using ConvertOpToLLVMPattern<Mlae32mOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Mlae32mOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module) return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), {i64Type, ptrType, i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.mlae32.m", funcType);

    Value mdVal = LLVM::ConstantOp::create(rewriter, 
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, basePtr, adaptor.getStride()});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for mlbe32.m (load right matrix B)
struct AMEMlbe32mLowering : public ConvertOpToLLVMPattern<Mlbe32mOp> {
  using ConvertOpToLLVMPattern<Mlbe32mOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Mlbe32mOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module) return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), {i64Type, ptrType, i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.mlbe32.m", funcType);

    Value mdVal = LLVM::ConstantOp::create(rewriter, 
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, basePtr, adaptor.getStride()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Store Operations Lowering
//===----------------------------------------------------------------------===//

/// Lowering pattern for msce32.m (store output matrix C)
struct AMEMsce32mLowering : public ConvertOpToLLVMPattern<Msce32mOp> {
  using ConvertOpToLLVMPattern<Msce32mOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Msce32mOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module) return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto funcType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), {i64Type, ptrType, i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.msce32.m", funcType);

    Value ms3Val = LLVM::ConstantOp::create(rewriter, 
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs3()));
    Value basePtr = extractPointerFromMemref(rewriter, loc, op.getBase());

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, intrinsicName,
                                  ValueRange{ms3Val, basePtr, adaptor.getStride()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Tile Register Matrix Multiply Lowering
//===----------------------------------------------------------------------===//

/// Lowering pattern for mma.w.mm.tile (tile register matrix multiply)
struct AMEMmaWmmTileLowering : public ConvertOpToLLVMPattern<MmaWmmTileOp> {
  using ConvertOpToLLVMPattern<MmaWmmTileOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MmaWmmTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module) return failure();

    auto i64Type = IntegerType::get(ctx, 64);
    auto funcType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), {i64Type, i64Type, i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.mma.w.mm.tile", funcType);

    Value mdVal = LLVM::ConstantOp::create(rewriter, 
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMd()));
    Value ms1Val = LLVM::ConstantOp::create(rewriter, 
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs1()));
    Value ms2Val = LLVM::ConstantOp::create(rewriter, 
        loc, i64Type, rewriter.getI64IntegerAttr(op.getMs2()));

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdVal, ms1Val, ms2Val});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// High-level Matrix Multiply Lowering (MemRef version)
//===----------------------------------------------------------------------===//

/// Lowering pattern for mqma.b.mm (int8 quad-widen matrix multiply)
struct AMEMqmaBmmLowering : public ConvertOpToLLVMPattern<MqmaBmmOp> {
  using ConvertOpToLLVMPattern<MqmaBmmOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MqmaBmmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    // Get memref types and compute dimensions
    auto mdType = cast<MemRefType>(op.getMd().getType());
    auto ms1Type = cast<MemRefType>(op.getMs1().getType());

    auto mdShape = mdType.getShape();
    auto ms1Shape = ms1Type.getShape();

    int64_t M = mdShape[0];
    int64_t N = mdShape[1];
    int64_t K = ms1Shape[1]; // K dimension from ms1

    // Define LLVM types
    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    // Get base pointers from memrefs
    Value mdBase = extractPointerFromMemref(rewriter, loc, op.getMd());
    Value ms1Base = extractPointerFromMemref(rewriter, loc, op.getMs1());
    Value ms2Base = extractPointerFromMemref(rewriter, loc, op.getMs2());

    // Create constants for dimensions
    Value mVal = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                                    rewriter.getI64IntegerAttr(M));
    Value nVal = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                                    rewriter.getI64IntegerAttr(N));
    Value kVal = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                                    rewriter.getI64IntegerAttr(K));

    // Create intrinsic function type for mqma.b.mm
    // void @llvm.riscv.buddy.mqma.b.mm(ptr md, ptr ms1, ptr ms2, i64 M, i64 N, i64 K)
    auto funcType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx),
        {ptrType, ptrType, ptrType, i64Type, i64Type, i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.mqma.b.mm", funcType);

    // Call the intrinsic
    LLVM::CallOp::create(rewriter, loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdBase, ms1Base, ms2Base,
                                            mVal, nVal, kVal});

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for mma.w.mm (int32 matrix multiply)
struct AMEMmaWmmLowering : public ConvertOpToLLVMPattern<MmaWmmOp> {
  using ConvertOpToLLVMPattern<MmaWmmOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MmaWmmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    // Get memref types and compute dimensions
    auto mdType = cast<MemRefType>(op.getMd().getType());
    auto ms1Type = cast<MemRefType>(op.getMs1().getType());

    auto mdShape = mdType.getShape();
    auto ms1Shape = ms1Type.getShape();

    int64_t M = mdShape[0];
    int64_t N = mdShape[1];
    int64_t K = ms1Shape[1];

    // Define LLVM types
    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    // Get base pointers from memrefs
    Value mdBase = extractPointerFromMemref(rewriter, loc, op.getMd());
    Value ms1Base = extractPointerFromMemref(rewriter, loc, op.getMs1());
    Value ms2Base = extractPointerFromMemref(rewriter, loc, op.getMs2());

    // Create constants for dimensions
    Value mVal = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                                    rewriter.getI64IntegerAttr(M));
    Value nVal = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                                    rewriter.getI64IntegerAttr(N));
    Value kVal = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                                    rewriter.getI64IntegerAttr(K));

    // Create intrinsic function type for mma.w.mm
    auto funcType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx),
        {ptrType, ptrType, ptrType, i64Type, i64Type, i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.mma.w.mm", funcType);

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdBase, ms1Base, ms2Base,
                                            mVal, nVal, kVal});

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for mma.dw.mm (int64 matrix multiply)
struct AMEMmaDwmmLowering : public ConvertOpToLLVMPattern<MmaDwmmOp> {
  using ConvertOpToLLVMPattern<MmaDwmmOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MmaDwmmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    // Get memref types and compute dimensions
    auto mdType = cast<MemRefType>(op.getMd().getType());
    auto ms1Type = cast<MemRefType>(op.getMs1().getType());

    auto mdShape = mdType.getShape();
    auto ms1Shape = ms1Type.getShape();

    int64_t M = mdShape[0];
    int64_t N = mdShape[1];
    int64_t K = ms1Shape[1];

    // Define LLVM types
    auto i64Type = IntegerType::get(ctx, 64);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    // Get base pointers from memrefs
    Value mdBase = extractPointerFromMemref(rewriter, loc, op.getMd());
    Value ms1Base = extractPointerFromMemref(rewriter, loc, op.getMs1());
    Value ms2Base = extractPointerFromMemref(rewriter, loc, op.getMs2());

    // Create constants for dimensions
    Value mVal = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                                    rewriter.getI64IntegerAttr(M));
    Value nVal = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                                    rewriter.getI64IntegerAttr(N));
    Value kVal = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                                    rewriter.getI64IntegerAttr(K));

    // Create intrinsic function type for mma.dw.mm
    auto funcType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx),
        {ptrType, ptrType, ptrType, i64Type, i64Type, i64Type});

    auto intrinsicName = getOrInsertIntrinsic(
        rewriter, module, "llvm.riscv.buddy.mma.dw.mm", funcType);

    LLVM::CallOp::create(rewriter, loc, TypeRange{}, intrinsicName,
                                  ValueRange{mdBase, ms1Base, ms2Base,
                                            mVal, nVal, kVal});

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct LegalizeAMEForLLVMExport
    : public PassWrapper<LegalizeAMEForLLVMExport, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeAMEForLLVMExport)

  StringRef getArgument() const final { return "lower-ame"; }
  StringRef getDescription() const final {
    return "AME dialect lowering pass.";
  }

  LegalizeAMEForLLVMExport() = default;
  LegalizeAMEForLLVMExport(const LegalizeAMEForLLVMExport &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<AMEDialect>();
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
    target.addIllegalOp<MmaWmmTileOp>();

    // High-level matrix multiply (MemRef)
    target.addIllegalOp<MqmaBmmOp>();
    target.addIllegalOp<MmaWmmOp>();
    target.addIllegalOp<MmaDwmmOp>();

    LLVMTypeConverter typeConverter(&context);

    RewritePatternSet patterns(&context);

    // Configuration patterns
    patterns.add<AMEMSettilemiLowering>(typeConverter);
    patterns.add<AMEMSettileniLowering>(typeConverter);
    patterns.add<AMEMSettilekiLowering>(typeConverter);
    patterns.add<AMEMzeroLowering>(typeConverter);

    // Load/Store patterns
    patterns.add<AMEMlae32mLowering>(typeConverter);
    patterns.add<AMEMlbe32mLowering>(typeConverter);
    patterns.add<AMEMsce32mLowering>(typeConverter);

    // Tile register matrix multiply patterns
    patterns.add<AMEMmaWmmTileLowering>(typeConverter);

    // High-level matrix multiply patterns
    patterns.add<AMEMqmaBmmLowering>(typeConverter);
    patterns.add<AMEMmaWmmLowering>(typeConverter);
    patterns.add<AMEMmaDwmmLowering>(typeConverter);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::populateAMELegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // Configuration patterns
  patterns.add<AMEMSettilemiLowering>(converter);
  patterns.add<AMEMSettileniLowering>(converter);
  patterns.add<AMEMSettilekiLowering>(converter);
  patterns.add<AMEMzeroLowering>(converter);

  // Load/Store patterns
  patterns.add<AMEMlae32mLowering>(converter);
  patterns.add<AMEMlbe32mLowering>(converter);
  patterns.add<AMEMsce32mLowering>(converter);

  // Tile register matrix multiply patterns
  patterns.add<AMEMmaWmmTileLowering>(converter);

  // High-level matrix multiply patterns
  patterns.add<AMEMqmaBmmLowering>(converter);
  patterns.add<AMEMmaWmmLowering>(converter);
  patterns.add<AMEMmaDwmmLowering>(converter);
}

void mlir::configureAMELegalizeForExportTarget(LLVMConversionTarget &target) {
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
  target.addIllegalOp<MmaWmmTileOp>();

  // High-level matrix multiply (MemRef)
  target.addIllegalOp<MqmaBmmOp>();
  target.addIllegalOp<MmaWmmOp>();
  target.addIllegalOp<MmaDwmmOp>();
}

std::unique_ptr<Pass> buddy::ame::createLegalizeForLLVMExportPass() {
  return std::make_unique<LegalizeAMEForLLVMExport>();
}

namespace mlir {
namespace buddy {
void registerLowerAMEPass() { PassRegistration<LegalizeAMEForLLVMExport>(); }
} // namespace buddy
} // namespace mlir
