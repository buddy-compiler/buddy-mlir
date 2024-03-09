//====------------ LegalizeForFuncExport.cpp - Gemmini To Func ------------===//
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
// This file defines Gemmini dialect lowering to 3rd party Library.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Gemmini/GemminiDialect.h"
#include "Gemmini/GemminiOps.h"
#include "Gemmini/Transform.h"

using namespace mlir;
using namespace buddy::gemmini;

namespace {

int64_t getNumberFromValue(Value &value) {
  return dyn_cast<IntegerAttr>(value.getDefiningOp()->getAttr("value")).getInt();
}

acc_scale_t_bits acc_scale_t_to_acc_scale_t_bits(acc_scale_t x) {
  union {
    acc_scale_t_bits b;
    acc_scale_t f;
  } un;

  un.f = x;
  return un.b;
}

scale_t_bits scale_t_to_scale_t_bits(scale_t x) {
  union {
    scale_t_bits b;
    scale_t f;
  } un;

  un.f = x;
  return un.b;
}

static MemRefType makeStridedLayoutDynamic(MemRefType type) {
  // Dynamic layout for the type
  auto dynamicLayout = StridedLayoutAttr::get(
      type.getContext(), ShapedType::kDynamic,
      SmallVector<int64_t>(type.getRank(), ShapedType::kDynamic));

  // Dynamic shape for the type
  auto dynamicShape = MemRefType::get(
    SmallVector<int64_t>(type.getRank(), ShapedType::kDynamic),
    type.getElementType(), dynamicLayout);

  return MemRefType::Builder(dynamicShape);
}

/// Helper function to extract the operand types that are passed to the
/// generated CallOp. MemRefTypes have their layout canonicalized since the
/// information is not used in signature generation.
/// Note that static size information is not modified.
static SmallVector<Type, 4> extractOperandTypes(Operation *op) {
  SmallVector<Type, 4> result;
  result.reserve(op->getNumOperands());
  for (auto type : op->getOperandTypes()) {
    // The underlying descriptor type (e.g. LLVM) does not have layout
    // information. Canonicalizing the type at the level of std when going into
    // a library call avoids needing to introduce DialectCastOp.
    if (auto memrefType = type.dyn_cast<MemRefType>())
      result.push_back(makeStridedLayoutDynamic(memrefType));
    else
      result.push_back(type);
  }
  return result;
}

// Get a SymbolRefAttr containing the library function name for 
// the Gemmini oeparation. If the library function does not exist, 
// insert a declaration.
static FailureOr<FlatSymbolRefAttr> 
getLibraryCallSymbolRef(Operation *op, std::string fnName, 
                        PatternRewriter &rewriter) {
  if (fnName.empty()) {
    return rewriter.notifyMatchFailure(op, "No library call defined for: ");
  }

  // fnName is a dynamic std::String, unique it via a SymbolRefAttr.
  FlatSymbolRefAttr fnNameAttr =
      SymbolRefAttr::get(rewriter.getContext(), fnName);
  auto module = op->getParentOfType<ModuleOp>();
  if (module.lookupSymbol(fnName)) {
    return fnNameAttr;
  }

  SmallVector<Type, 4> inputTypes({IntegerType::get(rewriter.getContext(), 64), 
                                  IntegerType::get(rewriter.getContext(), 64)});
  if (op->getNumResults() != 0) {
    return rewriter.notifyMatchFailure(
        op,
        "Library call can be generated only for ops that "
        "have void return types");
  }
  auto libFnType = rewriter.getFunctionType(inputTypes, op->getResultTypes());

  OpBuilder::InsertionGuard guard(rewriter);
  // Insert before module terminator.
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));
  func::FuncOp funcOp = rewriter.create<func::FuncOp>(op->getLoc(), 
                                                      fnNameAttr.getValue(),
                                                      libFnType);
  // Insert a function attribute that will trigger the emission of the
  // corresponding `_mlir_ciface_xxx` interface so that external libraries see
  // a normalized ABI. This interface is added during std to llvm conversion.
  funcOp->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                  UnitAttr::get(op->getContext()));
  funcOp.setPrivate();

//   module.dump();
  return fnNameAttr;
}

void gemminiMvinOffset(Operation *op, const Value &mem, const size_t offset,
                       const uint32_t SpAddr, const size_t cols,
                       const size_t rows, int64_t addrLen,
                       PatternRewriter &rewriter) {
  Location loc = mem.getLoc();
  Value offsetOp = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(offset));
  IntegerType i64Type = rewriter.getI64Type();
  Value configPtr = rewriter.create<arith::AddIOp>(loc, i64Type, mem, offsetOp);
  uint64_t spadAddrInt = (uint64_t)rows << (addrLen + 16) |
                         (uint64_t)cols << addrLen | (uint64_t)SpAddr;
  Value spad = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(spadAddrInt));
  auto libraryCallName = getLibraryCallSymbolRef(op, "mvin", rewriter);
  rewriter.create<func::CallOp>(
    loc, libraryCallName->getValue(), TypeRange(), 
    ValueRange({configPtr, spad}));
}

void gemminiMvin2Offset(Operation *op, const Value &mem, const size_t offset,
                       const uint32_t SpAddr, const size_t cols,
                       const size_t rows, int64_t addrLen,
                       PatternRewriter &rewriter) {
  Location loc = mem.getLoc();
  Value offsetOp = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(offset));
  IntegerType i64Type = rewriter.getI64Type();
  Value configPtr = rewriter.create<arith::AddIOp>(loc, i64Type, mem, offsetOp);
  uint64_t spadAddrInt = (uint64_t)rows << (addrLen + 16) |
                         (uint64_t)cols << addrLen | (uint64_t)SpAddr;
  Value spad = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(spadAddrInt));
  auto libraryCallName = getLibraryCallSymbolRef(op, "mvin2", rewriter);
  rewriter.create<func::CallOp>(
    loc, libraryCallName->getValue(), TypeRange(), 
    ValueRange({configPtr, spad}));
}

void gemminiMvin3Offset(Operation *op, const Value &mem, const size_t offset,
                       const uint32_t SpAddr, const size_t cols,
                       const size_t rows, int64_t addrLen,
                       PatternRewriter &rewriter) {
  Location loc = mem.getLoc();
  Value offsetOp = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(offset));
  IntegerType i64Type = rewriter.getI64Type();
  Value configPtr = rewriter.create<arith::AddIOp>(loc, i64Type, mem, offsetOp);
  uint64_t spadAddrInt = (uint64_t)rows << (addrLen + 16) |
                         (uint64_t)cols << addrLen | (uint64_t)SpAddr;
  Value spad = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(spadAddrInt));
  auto libraryCallName = getLibraryCallSymbolRef(op, "mvin3", rewriter);
  rewriter.create<func::CallOp>(
    loc, libraryCallName->getValue(), TypeRange(), 
    ValueRange({configPtr, spad}));
}

void gemminiMvoutOffset(Operation *op, const Value &mem, const size_t offset,
                        const uint32_t SpAddr, const size_t cols,
                        const size_t rows, int64_t addrLen,
                        PatternRewriter &rewriter) {
  Location loc = mem.getLoc();
  Value offsetOp = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(offset));
  IntegerType i64Type = rewriter.getI64Type();
  Value configPtr = rewriter.create<arith::AddIOp>(loc, i64Type, mem, offsetOp);
  uint64_t spadAddrInt = (uint64_t)rows << (addrLen + 16) |
                         (uint64_t)cols << addrLen | (uint64_t)SpAddr;
  Value spad = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(spadAddrInt));
  auto libraryCallName = getLibraryCallSymbolRef(op, "mvout", rewriter);
  rewriter.create<func::CallOp>(
    loc, libraryCallName->getValue(), TypeRange(), 
    ValueRange({configPtr, spad}));
}

} // namespace

struct GemminiFlushLowering : public OpRewritePattern<FlushOp> {
  using OpRewritePattern<FlushOp>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(FlushOp flushOp, PatternRewriter &rewriter) const override {
    Location loc = flushOp.getLoc();
    Value skip = flushOp.getSkip();
    IntegerAttr rs2Attr = rewriter.getI64IntegerAttr(0);
    Value rs2 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(), rs2Attr);

    auto libraryCallName = getLibraryCallSymbolRef(flushOp, "flush", rewriter);
    if (failed(libraryCallName))
      return failure();

    rewriter.replaceOpWithNewOp<func::CallOp>(
      flushOp, libraryCallName->getValue(), TypeRange(), 
      ValueRange({skip, rs2}));

    return success();
  }
};

struct GemminiConfigStLowering : public OpRewritePattern<ConfigStOp> {
  using OpRewritePattern<ConfigStOp>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(ConfigStOp configStOp, 
                  PatternRewriter &rewriter) const override {
    Value strideValue = configStOp.getStride();
    int stride = getNumberFromValue(strideValue);
    float scale = configStOp.getScale().convertToFloat();
    Location loc = configStOp.getLoc();
    uint64_t rs1 = ((uint64_t)configStOp.getActivation() << 2) | CONFIG_ST;
    uint64_t arg = (uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)scale)
                       << 32 |
                   (uint32_t)stride;
    Value value1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value value2 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(arg));

    auto libraryCallName = getLibraryCallSymbolRef(configStOp, 
                                                  "config_st", 
                                                  rewriter);
    if (failed(libraryCallName))
      return failure();

    rewriter.replaceOpWithNewOp<func::CallOp>(
      configStOp, libraryCallName->getValue(), TypeRange(), 
      ValueRange({value1, value2}));
    return success();
  }
};

struct GemminiConfigLdLowering : public OpRewritePattern<ConfigLdOp> {
  using OpRewritePattern<ConfigLdOp>::OpRewritePattern;
  explicit GemminiConfigLdLowering(MLIRContext *context, int64_t dim)
      : OpRewritePattern(context), dim(dim) {}
  LogicalResult
  matchAndRewrite(ConfigLdOp configLdOp, 
                  PatternRewriter &rewriter) const override {
    Value rs2Value = configLdOp.getStride();
    float scale = configLdOp.getScale().convertToFloat();
    uint64_t blockMvinStride = configLdOp.getBlockMvinStride();
    if (blockMvinStride == (uint64_t)-1)
      blockMvinStride = dim;
    uint64_t pixelRepeats = configLdOp.getPixelRepeats();
    uint64_t rs1 = (uint64_t)scale_t_to_scale_t_bits(scale) << 32 |
                   (blockMvinStride << 16) | pixelRepeats << 8 |
                   configLdOp.getId() << 3 | configLdOp.getShrunk() << 2 |
                   CONFIG_LD;
    Location loc = configLdOp.getLoc();
    Value rs1value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));

    auto libraryCallName = getLibraryCallSymbolRef(configLdOp, 
                                                  "config_ld", 
                                                  rewriter);
    if (failed(libraryCallName))
      return failure();

    rewriter.replaceOpWithNewOp<func::CallOp>(
      configLdOp, libraryCallName->getValue(), TypeRange(), 
      ValueRange({rs1value, rs2Value}));
    return success();
  }

private:
  int64_t dim;
};

struct GemminiMvinLowering : public OpRewritePattern<MvinOp> {
  using OpRewritePattern<MvinOp>::OpRewritePattern;
  explicit GemminiMvinLowering(MLIRContext *context, int64_t addrLen)
      : OpRewritePattern(context), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(MvinOp mvinOp, PatternRewriter &rewriter) const override {
    Value input = mvinOp.getInput();
    Location loc = input.getLoc();
    MemRefType memRefType = 
        dyn_cast<MemRefType>(mvinOp.getOperandTypes().front());
    llvm::ArrayRef<int64_t> memRefShape = memRefType.getShape();
    TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
    Value extractOp = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, resultType, input);
    IntegerType i64Type = rewriter.getI64Type();
    Value indexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOp);
    Value spadAddrValue = mvinOp.getAddr();
    uint64_t number = getNumberFromValue(spadAddrValue);
    uint64_t spadAddrInt = (uint64_t)memRefShape[0] << (addrLen + 16) |
                           (uint64_t)memRefShape[1] << addrLen | number;
    Value spad = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(spadAddrInt));

    auto libraryCallName = getLibraryCallSymbolRef(mvinOp, "mvin", rewriter);
    if (failed(libraryCallName))
      return failure();

    rewriter.replaceOpWithNewOp<func::CallOp>(
      mvinOp, libraryCallName->getValue(), TypeRange(), 
      ValueRange({indexCastOp, spad}));
    return success();
  }

private:
  int64_t addrLen;
};

struct GemminiMvin2Lowering : public OpRewritePattern<Mvin2Op> {
  using OpRewritePattern<Mvin2Op>::OpRewritePattern;
  explicit GemminiMvin2Lowering(MLIRContext *context, int64_t addrLen)
      : OpRewritePattern(context), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(Mvin2Op mvin2Op, PatternRewriter &rewriter) const override {
    Value input = mvin2Op.getInput();
    Location loc = input.getLoc();
    MemRefType memRefType = 
        dyn_cast<MemRefType>(mvin2Op.getOperandTypes().front());
    llvm::ArrayRef<int64_t> memRefShape = memRefType.getShape();
    TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
    Value extractOp = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, resultType, input);
    IntegerType i64Type = rewriter.getI64Type();
    Value indexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOp);
    Value spadAddrValue = mvin2Op.getAddr();
    uint64_t number = getNumberFromValue(spadAddrValue);
    uint64_t spadAddrInt = (uint64_t)memRefShape[0] << (addrLen + 16) |
                           (uint64_t)memRefShape[1] << addrLen | number;
    Value spad = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(spadAddrInt));

    auto libraryCallName = getLibraryCallSymbolRef(mvin2Op, "mvin2", rewriter);
    if (failed(libraryCallName))
      return failure();

    rewriter.replaceOpWithNewOp<func::CallOp>(
      mvin2Op, libraryCallName->getValue(), TypeRange(), 
      ValueRange({indexCastOp, spad}));

    return success();
  }

private:
  int64_t addrLen;
};

struct GemminiMvin3Lowering : public OpRewritePattern<Mvin3Op> {
  using OpRewritePattern<Mvin3Op>::OpRewritePattern;
  explicit GemminiMvin3Lowering(MLIRContext *context, int64_t addrLen)
      : OpRewritePattern(context), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(Mvin3Op mvin3Op, PatternRewriter &rewriter) const override {
    Value input = mvin3Op.getInput();
    Location loc = input.getLoc();
    MemRefType memRefType = 
        dyn_cast<MemRefType>(mvin3Op.getOperandTypes().front());
    llvm::ArrayRef<int64_t> memRefShape = memRefType.getShape();
    TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
    Value extractOp = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, resultType, input);
    IntegerType i64Type = rewriter.getI64Type();
    Value indexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOp);
    Value spadAddrValue = mvin3Op.getAddr();
    uint64_t number = getNumberFromValue(spadAddrValue);
    uint64_t spadAddrInt = (uint64_t)memRefShape[0] << (addrLen + 16) |
                           (uint64_t)memRefShape[1] << addrLen | number;
    Value spad = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(spadAddrInt));

    auto libraryCallName = getLibraryCallSymbolRef(mvin3Op, "mvin3", rewriter);
    if (failed(libraryCallName))
      return failure();

    rewriter.replaceOpWithNewOp<func::CallOp>(
      mvin3Op, libraryCallName->getValue(), TypeRange(), 
      ValueRange({indexCastOp, spad}));

    return success();
  }

private:
  int64_t addrLen;
};

struct GemminiMvoutLowering : public OpRewritePattern<MvoutOp> {
  using OpRewritePattern<MvoutOp>::OpRewritePattern;
  explicit GemminiMvoutLowering(MLIRContext *context, int64_t addrLen)
      : OpRewritePattern(context), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(MvoutOp mvoutOp, PatternRewriter &rewriter) const override {
    Value output = mvoutOp.getOutput();
    TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
    Location loc = mvoutOp.getLoc();
    Value extractOp = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, resultType, output);
    IntegerType i64Type = rewriter.getI64Type();
    Value indexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOp);
    Value spadAddr = mvoutOp.getAddr();
    uint64_t number = getNumberFromValue(spadAddr);
    MemRefType memRefType =
        dyn_cast<MemRefType>(mvoutOp.getOperandTypes().front());
    llvm::ArrayRef<int64_t> memRefShape = memRefType.getShape();
    uint64_t spadAddrInt = (uint64_t)memRefShape[0] << (addrLen + 16) |
                           (uint64_t)memRefShape[1] << addrLen | number;
    Value newSpad = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(spadAddrInt));

    auto libraryCallName = getLibraryCallSymbolRef(mvoutOp, "mvout", rewriter);
    if (failed(libraryCallName))
      return failure();

    rewriter.replaceOpWithNewOp<func::CallOp>(
      mvoutOp, libraryCallName->getValue(), TypeRange(), 
      ValueRange({indexCastOp, newSpad}));
    return success();
  }

private:
  int64_t addrLen;
};

struct GemminiConfigExLowering : public OpRewritePattern<ConfigExOp> {
  using OpRewritePattern<ConfigExOp>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(ConfigExOp configExOp, 
                  PatternRewriter &rewriter) const override {
    IntegerType i64Type = rewriter.getI64Type();
    Location loc = configExOp.getLoc();
    float scale = configExOp.getSysAccScale().convertToFloat();
    uint64_t rs1 =
        (uint64_t)acc_scale_t_to_acc_scale_t_bits(scale) << 32 |
        configExOp.getAStride() << 16 | configExOp.getBTranspose() << 9 |
        configExOp.getATranspose() << 8 | configExOp.getSetOnlyStrides() << 7 |
        configExOp.getSysAct() << 3 | configExOp.getDataflow() << 2 | CONFIG_EX;

    uint64_t rs2 = configExOp.getCStride() << 48 | configExOp.getSysShift();
    IntegerAttr rs1Attr = rewriter.getI64IntegerAttr(rs1);
    IntegerAttr rs2Attr = rewriter.getI64IntegerAttr(rs2);
    Value rs1Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs1Attr);
    Value rs2Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs2Attr);

    auto libraryCallName = getLibraryCallSymbolRef(configExOp, 
                                                  "config_ex", 
                                                  rewriter);
    if (failed(libraryCallName))
      return failure();

    rewriter.replaceOpWithNewOp<func::CallOp>(
      configExOp, libraryCallName->getValue(), TypeRange(), 
      ValueRange({rs1Value, rs2Value}));

    return success();
  }
};

struct GemminiConfigNormLowering : public OpRewritePattern<ConfigNormOp> {
  using OpRewritePattern<ConfigNormOp>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(ConfigNormOp configNormOp, 
                  PatternRewriter &rewriter) const override {
    Location loc = configNormOp.getLoc();
    uint64_t rs1 = (((uint64_t)((uint32_t)configNormOp.getQConst())) << 32) |
                   (configNormOp.getQConstType() & 1) << 18 |
                   (configNormOp.getSetStatsIdOnly() & 1) << 17 |
                   (configNormOp.getActMsb() & 1) << 16 |
                   configNormOp.getStatsId() << 8 | CONFIG_BERT;
    uint64_t rs2 = (((uint64_t)((uint32_t)configNormOp.getIgeluQc())) << 32) |
                   ((uint64_t)((uint32_t)configNormOp.getIgeluQb()));
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
    
    auto libraryCallName = getLibraryCallSymbolRef(configNormOp, 
                                                  "config_norm", 
                                                  rewriter);
    if (failed(libraryCallName))
      return failure();

    rewriter.replaceOpWithNewOp<func::CallOp>(
      configNormOp, libraryCallName->getValue(), TypeRange(), 
      ValueRange({rs1Value, rs2Value}));

    return success();
  }
};

struct GemminiPreloadZerosLowering
    : public OpRewritePattern<PreloadZerosOp> {
  using OpRewritePattern<PreloadZerosOp>::OpRewritePattern;
  explicit GemminiPreloadZerosLowering(MLIRContext *context,
                                       int64_t dim, int64_t addrLen)
      : OpRewritePattern(context), dim(dim), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(PreloadZerosOp preloadZerosOp, 
                  PatternRewriter &rewriter) const override {
    Value addr = preloadZerosOp.getAddr();
    Value cRows = preloadZerosOp.getCRows();
    Value cCols = preloadZerosOp.getCCols();
    Location loc = preloadZerosOp.getLoc();
    uint64_t addrInt = getNumberFromValue(addr);
    uint64_t cRowsInt = getNumberFromValue(cRows);
    uint64_t cColsInt = getNumberFromValue(cCols);
    uint64_t rs1 = (uint64_t)dim << (addrLen + 16) | (uint64_t)dim << addrLen |
                   (uint64_t)-1;
    uint64_t rs2 = cRowsInt << (addrLen + 16) | cColsInt << (addrLen) | addrInt;
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
    
    auto libraryCallName = getLibraryCallSymbolRef(preloadZerosOp, 
                                                  "preload", 
                                                  rewriter);
    if (failed(libraryCallName))
      return failure();

    rewriter.replaceOpWithNewOp<func::CallOp>(
      preloadZerosOp, libraryCallName->getValue(), TypeRange(), 
      ValueRange({rs1Value, rs2Value}));

    return success();
  }

private:
  int64_t dim;
  int64_t addrLen;
};

struct GemminiPreloadLowering : public OpRewritePattern<PreloadOp> {
  using OpRewritePattern<PreloadOp>::OpRewritePattern;
  explicit GemminiPreloadLowering(MLIRContext *context, int64_t addrLen)
      : OpRewritePattern(context), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(PreloadOp preloadOp, 
                  PatternRewriter &rewriter) const override {
    Value bdAddr = preloadOp.getBdAddr();
    Value cAddr = preloadOp.getCAddr();
    Value bdCols = preloadOp.getBdCols();
    Value bdRows = preloadOp.getBdRows();
    Value cCols = preloadOp.getCCols();
    Value cRows = preloadOp.getCRows();
    Location loc = preloadOp.getLoc();
    uint64_t bdAddrInt = getNumberFromValue(bdAddr);
    uint64_t cAddrInt = getNumberFromValue(cAddr);
    uint64_t bdColsInt = getNumberFromValue(bdCols);
    uint64_t bdRowsInt = getNumberFromValue(bdRows);
    uint64_t cColsInt = getNumberFromValue(cCols);
    uint64_t cRowsInt = getNumberFromValue(cRows);
    uint64_t rs1 = bdRowsInt << (addrLen + 16) | bdColsInt << addrLen |
                   (uint64_t)bdAddrInt;
    uint64_t rs2 =
        cRowsInt << (addrLen + 16) | cColsInt << addrLen | (uint64_t)cAddrInt;
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
    
    auto libraryCallName = getLibraryCallSymbolRef(preloadOp, 
                                                  "preload", 
                                                  rewriter);
    if (failed(libraryCallName))
      return failure();

    rewriter.replaceOpWithNewOp<func::CallOp>(
      preloadOp, libraryCallName->getValue(), TypeRange(), 
      ValueRange({rs1Value, rs2Value}));

    return success();
  }

private:
  int64_t addrLen;
};

struct GemminiComputePreloadedLowering
    : public OpRewritePattern<ComputePreloadedOp> {
  using OpRewritePattern<ComputePreloadedOp>::OpRewritePattern;
  explicit GemminiComputePreloadedLowering(MLIRContext *context,
                                           int64_t addrLen)
      : OpRewritePattern(context), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(ComputePreloadedOp computePreloadedOp, 
                  PatternRewriter &rewriter) const override {
    Value aAddr = computePreloadedOp.getAAddr();
    Value bdAddr = computePreloadedOp.getBdAddr();
    Value aRows = computePreloadedOp.getARows();
    Value aCols = computePreloadedOp.getACols();
    Value bdRows = computePreloadedOp.getBdRows();
    Value bdCols = computePreloadedOp.getBdCols();
    Location loc = computePreloadedOp.getLoc();
    uint64_t aAddrInt = getNumberFromValue(aAddr);
    uint64_t bdAddrInt = getNumberFromValue(bdAddr);
    uint64_t aRowsInt = getNumberFromValue(aRows);
    uint64_t aColsInt = getNumberFromValue(aCols);
    uint64_t bdRowsInt = getNumberFromValue(bdRows);
    uint64_t bdColsInt = getNumberFromValue(bdCols);
    uint64_t rs1 = aRowsInt << (addrLen + 16) | aColsInt << addrLen | aAddrInt;
    uint64_t rs2 =
        bdRowsInt << (addrLen + 16) | bdColsInt << addrLen | bdAddrInt;
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
    
    auto libraryCallName = getLibraryCallSymbolRef(computePreloadedOp, 
                                                  "compute_preloaded", 
                                                  rewriter);
    if (failed(libraryCallName))
      return failure();

    rewriter.replaceOpWithNewOp<func::CallOp>(
      computePreloadedOp, libraryCallName->getValue(), TypeRange(), 
      ValueRange({rs1Value, rs2Value}));

    return success();
  }

private:
  int64_t addrLen;
};

struct GemminiComputeAccumulatedLowering
    : public OpRewritePattern<ComputeAccumulatedOp> {
  using OpRewritePattern<ComputeAccumulatedOp>::OpRewritePattern;
  explicit GemminiComputeAccumulatedLowering(MLIRContext *context,
                                             int64_t addrLen)
      : OpRewritePattern(context), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(ComputeAccumulatedOp computeAccumulatedOp, 
                  PatternRewriter &rewriter) const override {
    Value aAddr = computeAccumulatedOp.getAAddr();
    Value bdAddr = computeAccumulatedOp.getBdAddr();
    Value aRows = computeAccumulatedOp.getARows();
    Value aCols = computeAccumulatedOp.getACols();
    Value bdRows = computeAccumulatedOp.getBdRows();
    Value bdCols = computeAccumulatedOp.getBdCols();
    Location loc = computeAccumulatedOp.getLoc();
    uint64_t aAddrInt = getNumberFromValue(aAddr);
    uint64_t bdAddrInt = getNumberFromValue(bdAddr);
    uint64_t aRowsInt = getNumberFromValue(aRows);
    uint64_t aColsInt = getNumberFromValue(aCols);
    uint64_t bdRowsInt = getNumberFromValue(bdRows);
    uint64_t bdColsInt = getNumberFromValue(bdCols);
    uint64_t rs1 = aRowsInt << (addrLen + 16) | aColsInt << addrLen | aAddrInt;
    uint64_t rs2 =
        bdRowsInt << (addrLen + 16) | bdColsInt << addrLen | bdAddrInt;
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
    
    auto libraryCallName = getLibraryCallSymbolRef(computeAccumulatedOp, 
                                                  "compute_accumulated", 
                                                  rewriter);
    if (failed(libraryCallName))
      return failure();

    rewriter.replaceOpWithNewOp<func::CallOp>(
      computeAccumulatedOp, libraryCallName->getValue(), TypeRange(), 
      ValueRange({rs1Value, rs2Value}));

    return success();
  }

private:
  int64_t addrLen;
};

class GemminiTileMatMulLowering : public OpRewritePattern<TileMatMulOp> {
  void gemminiLoopWs(size_t i, size_t j, size_t k, size_t padI, size_t padJ,
                     size_t padK, Value &a, Value &b, Value &d, Value &c,
                     size_t aRowStride, size_t bRowStride, size_t dRowStride,
                     size_t cRowStride, bool aTranspose, bool bTranspose,
                     bool fullC, bool lowD, bool exAccumulate, int act,
                     TileMatMulOp &tileMatMulOp,
                     PatternRewriter &rewriter) const {
    // loopWsConfigBounds instruction.
    uint64_t rs1 = (uint64_t)padK << 32 | (uint64_t)padJ << 16 | (uint64_t)padI;
    uint64_t rs2 = (uint64_t)k << 32 | (uint64_t)j << 16 | (uint64_t)i;
    IntegerType i64Type = rewriter.getI64Type();
    Location loc = a.getLoc();
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
    auto libraryCallName = getLibraryCallSymbolRef(tileMatMulOp, 
                                                  "loop_ws_config_bounds",
                                                   rewriter);
    rewriter.create<func::CallOp>(
      loc, libraryCallName->getValue(), TypeRange(), 
      ValueRange({rs1Value, rs2Value}));
    // loopWsConfigAddrsAB instruction.
    libraryCallName = getLibraryCallSymbolRef(tileMatMulOp, 
                                              "loop_ws_config_addrs_ab",
                                              rewriter);
    rewriter.create<func::CallOp>(
      loc, libraryCallName->getValue(), TypeRange(), 
      ValueRange({a, b}));
    // loopWsConfigAddrsDC instruction
    libraryCallName = getLibraryCallSymbolRef(tileMatMulOp, 
                                              "loop_ws_config_addrs_dc",
                                              rewriter);
    rewriter.create<func::CallOp>(
      loc, libraryCallName->getValue(), TypeRange(), 
      ValueRange({d, c}));
    // loopWsConfigStridesAB instruction
    rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(aRowStride));
    rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(bRowStride));
    libraryCallName = getLibraryCallSymbolRef(tileMatMulOp, 
                                              "loop_ws_config_strides_ab",
                                              rewriter);
    rewriter.create<func::CallOp>(
      loc, libraryCallName->getValue(), TypeRange(), 
      ValueRange({rs1Value, rs2Value}));
    // loopWsConfigStrideDC instruction
    rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(dRowStride));
    rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(cRowStride));
    libraryCallName = getLibraryCallSymbolRef(tileMatMulOp, 
                                            "ciface_loop_ws_config_strides_dc",
                                            rewriter);
    rewriter.create<func::CallOp>(
      loc, libraryCallName->getValue(), TypeRange(), 
      ValueRange({rs1Value, rs2Value}));
    // LoopWs instruction
    rs1 = (uint64_t)act << 8 | lowD << 2 | (fullC) << 1 | exAccumulate;
    rs2 = bTranspose << 1 | aTranspose;
    rs1Value = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(rs1));
    rs2Value = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(rs2));
    libraryCallName = getLibraryCallSymbolRef(tileMatMulOp, 
                                              "loop_ws",
                                              rewriter);
    rewriter.create<func::CallOp>(
      loc, libraryCallName->getValue(), TypeRange(), 
      ValueRange({rs1Value, rs2Value}));
  }

  void spTiledMatmulWs(Value &a, Value &b, Value &d, Value &c,
                       scale_t aScaleFactor, scale_t bScaleFactor,
                       scale_acc_t dScaleFactor, size_t i, size_t j, size_t k,
                       size_t padI, size_t padJ, size_t padK, size_t strideA,
                       size_t strideB, size_t strideD, size_t strideC,
                       bool aTranspose, bool bTranspose, bool fullC, bool lowD,
                       bool noBias, bool repeatingBias, int act,
                       TileMatMulOp &tileMatMulOp,
                       PatternRewriter &rewriter) const {

    gemminiLoopWs(i, j, k, padI, padJ, padK, a, b, d, c, strideA, strideB,
                  repeatingBias ? 0 : strideD, strideC, aTranspose, bTranspose,
                  fullC, lowD, !noBias, act, tileMatMulOp, rewriter);
  }

  // Tiling functions
  void spTiledMatmulOs(Value &a, Value &b, Value &d, Value &c,
                       scale_t aScaleFactor, scale_t bScaleFactor,
                       scale_acc_t dScaleFactor, size_t i, size_t j, size_t k,
                       size_t padI, size_t padJ, size_t padK, size_t strideA,
                       size_t strideB, size_t strideD, size_t strideC,
                       bool aTranspose, bool bTranspose, bool fullC, bool lowD,
                       bool noBias, bool repeatingBias, int act,
                       TileMatMulOp &tileMatMulOp,
                       PatternRewriter &rewriter) const {
    const uint32_t aSpAddrStart = 0;
    const uint32_t bSpAddrStart = BANK_NUM * bankRows - k * j * dim;
    const uint32_t dSpAddrStart = 1 << (addrLen - 1);
    const uint32_t cSpAddrStart =
        (3 << (addrLen - 2)) | (fullC << (addrLen - 3));

    const size_t maxBlockLen = MAX_BYTES / (dim * 1);
    const size_t maxBlockLenAcc = MAX_BYTES / (dim * 4);

    const int aBlocks = k <= maxBlockLen ? k : maxBlockLen;
    const int bBlocks = j <= maxBlockLen ? j : maxBlockLen;
    const int dBlocks = j <= maxBlockLenAcc ? j : maxBlockLenAcc;

    Location loc = a.getLoc();
    bool dAddrNull = llvm::dyn_cast<arith::ConstantOp>(d.getDefiningOp()) &&
                     getNumberFromValue(d) == 0;
    bool cAddrNull = llvm::dyn_cast<arith::ConstantOp>(c.getDefiningOp()) &&
                     getNumberFromValue(c) == 0;

    // Move-in D
    if (!dAddrNull && !noBias) {
      const size_t dStride = repeatingBias ? 0 : strideD * sizeOfAccT;
      Value strideValue = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI64IntegerAttr(dStride));
      rewriter.create<ConfigLdOp>(loc, strideValue,
                                  llvm::APFloat((float)dScaleFactor));

      for (size_t i0 = 0; i0 < i; i0++) {
        for (size_t j0 = 0; j0 < j; j0 += dBlocks) {
          const size_t biasRow = repeatingBias ? 0 : i0;
          const size_t offset = (biasRow * strideD + j0) * dim * sizeOfAccT;
          const uint32_t dSpAddrAcc = dSpAddrStart + (i0 * j + j0) * dim;
          const size_t blocks = j0 + dBlocks <= j ? dBlocks : j - j0;
          const size_t cols = blocks * dim - (j0 + blocks >= j ? padJ : 0);
          const size_t rows = dim - (i0 == i - 1 ? padI : 0);
          gemminiMvinOffset(tileMatMulOp, d, offset, dSpAddrAcc, cols, rows, 
                            addrLen, rewriter);
        }
      }
    }

    // Move-in B
    Value strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(strideB));
    rewriter.create<ConfigLdOp>(loc, strideValue,
                                llvm::APFloat((float)bScaleFactor));
    for (size_t j0 = 0; j0 < j; j0 += bBlocks) {
      for (size_t k0 = 0; k0 < k; k0++) {
        const size_t offset = (k0 * strideB + j0) * dim * sizeOfElemT;
        const uint32_t bSpAddr = bSpAddrStart + (k0 * j + j0) * dim;
        const size_t blocks = j0 + bBlocks <= j ? bBlocks : j - j0;
        const size_t cols = blocks * dim - (j0 + blocks >= j ? padJ : 0);
        const size_t rows = dim - (k0 == k - 1 ? padK : 0);
        gemminiMvinOffset(tileMatMulOp, b, offset, bSpAddr, cols, rows, 
                          addrLen, rewriter);
      }
    }

    // Move-in A
    strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(strideA));
    rewriter.create<ConfigLdOp>(loc, strideValue,
                                llvm::APFloat((float)aScaleFactor));

    for (size_t i0 = 0; i0 < i; i0++) {
      for (size_t k0 = 0; k0 < k; k0 += aBlocks) {
        const size_t offset = (i0 * strideA + k0) * dim * sizeOfElemT;
        const uint32_t aSpAddr = aSpAddrStart + (i0 * k + k0) * dim;
        const size_t blocks = k0 + aBlocks <= k ? aBlocks : k - k0;
        const size_t cols = blocks * dim - (k0 + blocks >= k ? padK : 0);
        const size_t rows = dim - (i0 == i - 1 ? padI : 0);
        gemminiMvinOffset(tileMatMulOp, a, offset, aSpAddr, cols, rows, 
                          addrLen, rewriter);
      }
    }

    for (size_t i0 = 0; i0 < i; i0++) {
      for (size_t j0 = 0; j0 < j; j0++) {
        const uint32_t cSpAddr = cSpAddrStart + (i0 * j + j0) * dim;
        for (size_t k0 = 0; k0 < k; k0++) {

          const uint32_t aSpAddr = aSpAddrStart + (i0 * k + k0) * dim;
          const uint32_t bSpAddr = bSpAddrStart + (k0 * j + j0) * dim;

          uint32_t outSpAddr = k0 == k - 1 ? cSpAddr : GARBAGE_ADDR;

          // If we're not using a bias, then we want to overwrite what's in the
          // accumulator, rather than writing over it

          int noBiasNewMatrix = noBias && !dAddrNull && k0 == k - 1;
          if (noBiasNewMatrix) {
            outSpAddr &= ~(1 << (addrLen - 2));
          }

          const size_t aCols = dim - (k0 == k - 1 ? padK : 0);
          const size_t aRows = dim - (i0 == i - 1 ? padI : 0);
          const size_t bCols = dim - (j0 == j - 1 ? padJ : 0);
          const size_t bRows = dim - (k0 == k - 1 ? padK : 0);
          const size_t cCols = dim - (j0 == j - 1 ? padJ : 0);
          const size_t cRows = dim - (i0 == i - 1 ? padI : 0);

          Value aColsOp = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getI64IntegerAttr(aCols));
          Value aRowsOp = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getI64IntegerAttr(aRows));
          Value bColsOp = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getI64IntegerAttr(bCols));
          Value bRowsOp = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getI64IntegerAttr(bRows));
          Value cColsOp = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getI64IntegerAttr(cCols));
          Value cRowsOp = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getI64IntegerAttr(cRows));

          Value aSpAddrOp = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getI64IntegerAttr(aSpAddr));
          Value bSpAddrOp = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getI64IntegerAttr(bSpAddr));
          Value outSpAddrOp = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getI64IntegerAttr(outSpAddr));

          Value garbageAddrOp = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getI64IntegerAttr(GARBAGE_ADDR));
          Value dimOp = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getI64IntegerAttr(dim));

          rewriter.create<PreloadOp>(loc, garbageAddrOp, outSpAddrOp, dimOp,
                                     dimOp, cRowsOp, cColsOp);

          if (k0 == 0) { // First iteration
            rewriter.create<ComputePreloadedOp>(
                loc, aSpAddrOp, bSpAddrOp, aRowsOp, aColsOp, bRowsOp, bColsOp);

          } else { // All other iterations
            rewriter.create<ComputeAccumulatedOp>(
                loc, aSpAddrOp, bSpAddrOp, aRowsOp, aColsOp, bRowsOp, bColsOp);
          }
        }
      }
    }
    // Move-out C
    if (!cAddrNull) {
      const size_t sizeof_C = fullC ? sizeOfAccT : sizeOfElemT;

      for (size_t i0 = 0; i0 < i; i0++) {
        for (size_t j0 = 0; j0 < j; j0++) {
          const size_t offset = (i0 * strideC + j0) * dim * sizeof_C;
          const uint32_t cSpAddr = cSpAddrStart + (i0 * j + j0) * dim;

          const size_t cCols = dim - (j0 == j - 1 ? padJ : 0);
          const size_t cRows = dim - (i0 == j - 1 ? padI : 0);

          gemminiMvoutOffset(tileMatMulOp, c, offset, cSpAddr, cCols, cRows, addrLen,
                             rewriter);
        }
      }
    }
  }

  void tiledMatmulOuter(
      size_t dimI, size_t dimJ, size_t dimK, Value &A, Value &B, Value &D,
      Value &C, size_t strideA, size_t strideB, size_t strideD, size_t strideC,
      scale_t aScaleFactor, scale_t bScaleFactor, scale_acc_t dScaleFactor,
      size_t tileI, size_t tileJ, size_t tileK, int act, acc_scale_t scale,
      acc_scale_t bertScale, bool repeatingBias, bool aTranspose,
      bool bTranspose, bool fullC, bool lowD, uint8_t weightA, int dataflow,
      TileMatMulOp &tileMatMulOp, PatternRewriter &rewriter) const {
    const size_t dimIPadded = (dimI / dim + (dimI % dim != 0)) * dim;
    const size_t dimJPadded = (dimJ / dim + (dimJ % dim != 0)) * dim;
    const size_t dimKPadded = (dimK / dim + (dimK % dim != 0)) * dim;
    const size_t I0 =
        dimIPadded / (tileI * dim) + (dimIPadded % (tileI * dim) != 0);
    const size_t J0 =
        dimJPadded / (tileJ * dim) + (dimJPadded % (tileJ * dim) != 0);
    const size_t K0 =
        dimKPadded / (tileK * dim) + (dimKPadded % (tileK * dim) != 0);
    const size_t lastI =
        dimIPadded % (tileI * dim) == 0 ? tileI : (dimIPadded / dim) % tileI;
    const size_t lastJ =
        dimJPadded % (tileJ * dim) == 0 ? tileJ : (dimJPadded / dim) % tileJ;
    const size_t lastK =
        dimKPadded % (tileK * dim) == 0 ? tileK : (dimKPadded / dim) % tileK;
    const size_t paddingI = dimIPadded - dimI;
    const size_t paddingJ = dimJPadded - dimJ;
    const size_t paddingK = dimKPadded - dimK;
    const bool noBias = false;
    const size_t sizeofD = lowD ? sizeOfElemT : sizeOfAccT;
    const size_t sizeofC = fullC ? sizeOfAccT : sizeOfElemT;
    Location loc = tileMatMulOp.getLoc();
    llvm::APFloat accScaleIdentity((float)ACC_SCALE_IDENTITY);
    rewriter.create<ConfigExOp>(loc, /*dataflow = */ dataflow,
                                /*sysAct = */ act & 3,
                                /* sysShift = */ 0, accScaleIdentity);
    Value strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(strideC * sizeofC));
    rewriter.create<ConfigStOp>(loc, strideValue, act & 3,
                                llvm::APFloat(scale));
    strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(strideA * sizeOfElemT));
    rewriter.create<ConfigLdOp>(loc, strideValue, llvm::APFloat(aScaleFactor),
                                false, 0);
    strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(strideB * sizeOfElemT));
    rewriter.create<ConfigLdOp>(loc, strideValue, llvm::APFloat(bScaleFactor),
                                false, 1);
    strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(strideD * sizeofD));
    rewriter.create<ConfigLdOp>(loc, strideValue,
                                llvm::APFloat((float)dScaleFactor), lowD, 2);

    /*
      Add config norm op
    */
    if (act == IGELU) {
      const float sqrt_2 = 1.41421356237;
      const float S = bertScale;
      const float S_erf = (-0.2888 * ((S * S) / 2));

      const uint32_t qb = -1.769 / (S / sqrt_2);
      const uint32_t qc = 1.0 / S_erf;
      rewriter.create<ConfigNormOp>(loc, 0, 0, 0, 0, 0, qb, qc);
    }

    if (act == SOFTMAX) {
      const float a = 0.3585;
      const float b = 1.353;
      const float c = 0.344;

      const uint32_t qln2 = (int)(0.693147 / bertScale);
      const uint32_t qln2_inv = 65536 / qln2;
      const uint32_t qb = b / bertScale;
      const uint32_t qc = c / (a * bertScale * bertScale);
      rewriter.create<ConfigNormOp>(loc, qln2, 0, 0, 1, 0, qb, qc);
      rewriter.create<ConfigNormOp>(loc, qln2_inv, 1, 0, 1, 0, qb, qc);
    }

    for (size_t i0 = 0; i0 < I0; i0++)
      for (size_t j0 = 0; j0 < J0; j0++)
        for (size_t k0 = 0; k0 < K0; k0++) {
          Value pre;
          Location loc = A.getLoc();
          if (k0 != 0) {
            IntegerAttr preAttr = rewriter.getI64IntegerAttr(0);
            pre = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(),
                                                     preAttr);
          } else {
            size_t biasRow = repeatingBias ? 0 : i0 * tileI * dim;
            size_t offset = (biasRow * strideD + j0 * tileJ * dim) * sizeofD;
            IntegerAttr offsetAttr = rewriter.getI64IntegerAttr(offset);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64Type(), offsetAttr);
            pre = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), D,
                                                 offsetValue);
          }

          Value out;
          if (k0 == K0 - 1) {
            size_t offset =
                (i0 * tileI * dim * strideC + j0 * tileJ * dim) * sizeofC;
            IntegerAttr offsetAttr = rewriter.getI64IntegerAttr(offset);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64Type(), offsetAttr);
            out = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), C,
                                                 offsetValue);
          } else {
            IntegerAttr outAttr = rewriter.getI64IntegerAttr(0);
            out = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(),
                                                     outAttr);
          }
          const size_t i = i0 < I0 - 1 ? tileI : lastI;
          const size_t j = j0 < J0 - 1 ? tileJ : lastJ;
          const size_t k = k0 < K0 - 1 ? tileK : lastK;
          const size_t padI = i0 == I0 - 1 ? paddingI : 0;
          const size_t padJ = j0 == J0 - 1 ? paddingJ : 0;
          const size_t padK = k0 == K0 - 1 ? paddingK : 0;
          Value a;
          if (aTranspose) {
            size_t offset =
                (k0 * tileK * dim * strideA + i0 * tileI * dim) * sizeOfElemT;
            IntegerAttr offsetAttr = rewriter.getI64IntegerAttr(offset);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64Type(), offsetAttr);
            a = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), A,
                                               offsetValue);
          } else {
            size_t offset =
                (i0 * tileI * dim * strideA + k0 * tileK * dim) * sizeOfElemT;
            IntegerAttr offsetAttr = rewriter.getI64IntegerAttr(offset);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64Type(), offsetAttr);
            a = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), A,
                                               offsetValue);
          }
          Value b;
          if (bTranspose) {
            size_t offset =
                (j0 * tileJ * dim * strideB + k0 * tileK * dim) * sizeOfElemT;
            IntegerAttr offsetAttr = rewriter.getI64IntegerAttr(offset);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64Type(), offsetAttr);
            b = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), B,
                                               offsetValue);
          } else {
            size_t offset =
                (k0 * tileK * dim * strideB + j0 * tileJ * dim) * sizeOfElemT;
            IntegerAttr offsetAttr = rewriter.getI64IntegerAttr(offset);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64Type(), offsetAttr);
            b = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), B,
                                               offsetValue);
          }
          if (dataflow == OUTPUT_STATIONARY) {
            spTiledMatmulOs(a, b, pre, out, aScaleFactor, bScaleFactor,
                            dScaleFactor, i, j, k, padI, padJ, padK, strideA,
                            strideB, strideD, strideC, aTranspose, bTranspose,
                            fullC, lowD, noBias, repeatingBias, act,
                            tileMatMulOp, rewriter);
          } else { // WS
            spTiledMatmulWs(a, b, pre, out, aScaleFactor, bScaleFactor,
                            dScaleFactor, i, j, k, padI, padJ, padK, strideA,
                            strideB, strideD, strideC, aTranspose, bTranspose,
                            fullC, lowD, noBias, repeatingBias, act,
                            tileMatMulOp, rewriter);
          }
        }
    IntegerAttr flushAttr = rewriter.getI64IntegerAttr(0);
    Value flushValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(), flushAttr);
    auto libraryCallName = getLibraryCallSymbolRef(tileMatMulOp, 
                                                  "flush",
                                                  rewriter);
    rewriter.create<func::CallOp>(
      loc, libraryCallName->getValue(), TypeRange(), 
      ValueRange({flushValue, flushValue}));
    return;
  }

  size_t tiledMatmulTotalSpadRows(size_t I, size_t J, size_t K) const {
    return (I * K + K * J) * dim;
  }

  size_t tiledMatmulTotalAccRows(size_t I, size_t J) const {
    return (I * J) * dim;
  }

public:
  using OpRewritePattern<TileMatMulOp>::OpRewritePattern;
  explicit GemminiTileMatMulLowering(MLIRContext *context,
                                     int64_t dim, int64_t addrLen,
                                     int64_t accRows, int64_t bankRows,
                                     size_t sizeOfElemT, size_t sizeOfAccT)
      : OpRewritePattern(context), dim(dim), addrLen(addrLen),
        accRows(accRows), bankRows(bankRows), sizeOfElemT(sizeOfElemT),
        sizeOfAccT(sizeOfAccT) {}
  LogicalResult
  matchAndRewrite(TileMatMulOp tileMatMulOp,
                  PatternRewriter &rewriter) const override {
    size_t dbPartitionRows = ((BANK_NUM * bankRows / 2) / 2);
    size_t dbMatsInPartition = (dbPartitionRows / dim);
    size_t dbMatsInAcc((accRows / 2) / dim);
    size_t dbMaxTileIJ((size_t)sqrt(dbMatsInAcc));
    size_t dbMaxTileK(dbMatsInPartition / dbMaxTileIJ);

    Value aArray = tileMatMulOp.getAArray();
    Value bArray = tileMatMulOp.getBArray();
    Value cArray = tileMatMulOp.getCArray();
    Value dArray = tileMatMulOp.getDArray();
    MemRefType aArrayType = dyn_cast<MemRefType>(aArray.getType());
    MemRefType bArrayType = dyn_cast<MemRefType>(bArray.getType());
    MemRefType cArrayType = dyn_cast<MemRefType>(cArray.getType());
    MemRefType dArrayType = dyn_cast<MemRefType>(dArray.getType());
    StridedLayoutAttr aArrayLayout = dyn_cast<StridedLayoutAttr>(aArrayType.getLayout());
    StridedLayoutAttr bArrayLayout = dyn_cast<StridedLayoutAttr>(bArrayType.getLayout());
    StridedLayoutAttr cArrayLayout = dyn_cast<StridedLayoutAttr>(cArrayType.getLayout());
    SmallVector<Type> resultType = {rewriter.getIndexType()};
    TypeRange typeRange(resultType);
    Location loc = tileMatMulOp.getLoc();
    IntegerType i64Type = rewriter.getI64Type();
    Value aArrayExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, typeRange,
                                                                aArray);
    if (aArrayLayout) {
      Value offset = rewriter.create<arith::ConstantIndexOp>(
          loc, aArrayLayout.getOffset() * sizeOfElemT);
      aArrayExtractOp =
          rewriter.create<arith::AddIOp>(loc, aArrayExtractOp, offset);
    }
    Value aArrayindexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, aArrayExtractOp);
    Value bArrayExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, typeRange,
                                                                bArray);
    if (bArrayLayout) {
      Value offset = rewriter.create<arith::ConstantIndexOp>(
          loc, bArrayLayout.getOffset() * sizeOfElemT);
      bArrayExtractOp =
          rewriter.create<arith::AddIOp>(loc, bArrayExtractOp, offset);
    }
    Value bArrayindexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, bArrayExtractOp);
    Value cArrayExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, typeRange,
                                                                cArray);
    if (cArrayLayout) {
      Value offset = rewriter.create<arith::ConstantIndexOp>(
          loc, cArrayLayout.getOffset() * sizeOfElemT);
      cArrayExtractOp =
          rewriter.create<arith::AddIOp>(loc, cArrayExtractOp, offset);
    }
    Value cArrayindexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, cArrayExtractOp);
    Value dArrayExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, typeRange,
                                                                dArray);
    Value dArrayindexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, dArrayExtractOp);
    llvm::ArrayRef<int64_t> aArrayShape = aArrayType.getShape();
    llvm::ArrayRef<int64_t> bArrayShape = bArrayType.getShape();
    llvm::ArrayRef<int64_t> cArrayShape = cArrayType.getShape();
    llvm::ArrayRef<int64_t> dArrayShape = dArrayType.getShape();
    size_t dimI = aArrayShape[0];
    size_t dimK = aArrayShape[1];
    size_t dimJ = bArrayShape[1];
    size_t strideA = aArrayShape[1];
    size_t strideB = bArrayShape[1];
    size_t strideC = cArrayShape[1];
    size_t strideD = dArrayShape[1];
    scale_t aScaleFactor = tileMatMulOp.getAScaleFactor().convertToFloat();
    scale_t bScaleFactor = tileMatMulOp.getBScaleFactor().convertToFloat();
    scale_acc_t dScaleFactor = tileMatMulOp.getDScaleFactor().convertToFloat();
    int act = tileMatMulOp.getAct();
    acc_scale_t scale = tileMatMulOp.getAccScale().convertToFloat();
    acc_scale_t bertScale = tileMatMulOp.getBertScale().convertToFloat();
    bool repeatingBias = tileMatMulOp.getRepeatingBias();
    bool aTranspose = tileMatMulOp.getATranspose();
    bool bTranspose = tileMatMulOp.getBTranspose();
    bool fullC = tileMatMulOp.getFullC();
    bool lowD = tileMatMulOp.getLowD();
    uint8_t weightA = tileMatMulOp.getWeightA();
    size_t dimIPaded = (dimI / dim + (dimI % dim != 0)) * dim;
    size_t dimJPaded = (dimJ / dim + (dimJ % dim != 0)) * dim;
    size_t dimKPaded = (dimK / dim + (dimK % dim != 0)) * dim;
    size_t maxSpadRows = BANK_NUM * bankRows / 2;
    size_t maxAccRows = accRows / 2;
    size_t tileI, tileJ, tileK;
    if (act == LAYERNORM || act == SOFTMAX) {
      tileI = 1;
      tileJ = dimJPaded / dim;
      tileK = 1;
    } else {
      tileI = dimIPaded / dim < dbMaxTileIJ ? dimIPaded / dim : dbMaxTileIJ;
      tileJ = dimJPaded / dim < dbMaxTileIJ ? dimJPaded / dim : dbMaxTileIJ;
      tileK = dimKPaded / dim < dbMaxTileK ? dimKPaded / dim : dbMaxTileK;
    }
    while (true) {
      bool increased = false;

      if (tiledMatmulTotalSpadRows(tileI, tileJ + 1, tileK) <= maxSpadRows &&
          tiledMatmulTotalAccRows(tileI, tileJ + 1) <= maxAccRows &&
          (tileJ + 1) * dim <= dimJPaded) {
        tileJ++;
        increased = true;
      }

      if (tiledMatmulTotalSpadRows(tileI + 1, tileJ, tileK) <= maxSpadRows &&
          tiledMatmulTotalAccRows(tileI + 1, tileJ) <= maxAccRows &&
          (tileI + 1) * dim <= dimIPaded) {
        tileI++;
        increased = true;
      }

      if (tiledMatmulTotalSpadRows(tileI, tileJ, tileK + 1) <= maxSpadRows &&
          (tileK + 1) * dim <= dimKPaded) {
        tileK++;
        increased = true;
      }
      if (!increased)
        break;
    }
    int dataflow = tileMatMulOp.getDataflow();

    tiledMatmulOuter(dimI, dimJ, dimK, aArrayindexCastOp, bArrayindexCastOp,
                     dArrayindexCastOp, cArrayindexCastOp, strideA, strideB,
                     strideD, strideC, aScaleFactor, bScaleFactor, dScaleFactor,
                     tileI, tileJ, tileK, act, scale, bertScale, repeatingBias,
                     aTranspose, bTranspose, fullC, lowD, weightA, dataflow,
                     tileMatMulOp, rewriter);
    return success();
  };

private:
  int64_t dim;
  int64_t addrLen;
  int64_t accRows;
  int64_t bankRows;
  size_t sizeOfElemT;
  size_t sizeOfAccT;
};

void mlir::populateGemminiLegalizeForFuncExportPatterns(
    RewritePatternSet &patterns, int64_t dim, int64_t addrLen, 
    int64_t accRows, int64_t bankRows, size_t sizeOfElemT,
    size_t sizeOfAccT) {
//   patterns
//       .add<ForwardOperands<func::CallOp>, ForwardOperands<func::CallIndirectOp>,
//            ForwardOperands<func::ReturnOp>>(converter, &converter.getContext());
  patterns.add<GemminiFlushLowering>(patterns.getContext());
  patterns.add<GemminiConfigStLowering>(patterns.getContext());
  patterns.add<GemminiConfigLdLowering>(patterns.getContext(), dim);
  patterns.add<GemminiMvinLowering>(patterns.getContext(), addrLen);
  patterns.add<GemminiMvin2Lowering>(patterns.getContext(), addrLen);
  patterns.add<GemminiMvin3Lowering>(patterns.getContext(), addrLen);
  patterns.add<GemminiMvoutLowering>(patterns.getContext(), addrLen);
  patterns.add<GemminiConfigExLowering>(patterns.getContext());
  patterns.add<GemminiConfigNormLowering>(patterns.getContext());
  patterns.add<GemminiPreloadZerosLowering>(patterns.getContext(), dim, addrLen);
  patterns.add<GemminiPreloadLowering>(patterns.getContext(), addrLen);
  patterns.add<GemminiComputePreloadedLowering>(patterns.getContext(), addrLen);
  patterns.add<GemminiComputeAccumulatedLowering>(patterns.getContext(), addrLen);
  patterns.add<GemminiTileMatMulLowering>(patterns.getContext(), dim, addrLen, accRows,
                                          bankRows, sizeOfElemT, sizeOfAccT);
//   patterns.add<GemminiTileConvLowering>(patterns.getContext(), dim, addrLen, accRows,
//                                         bankRows, sizeOfElemT, sizeOfAccT);
}