//===- LegalizeForLLVMExport.cpp - Prepare Gemmini for LLVM translation --===//
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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "Gemmini/GemminiDialect.h"
#include "Gemmini/GemminiOps.h"
#include "Gemmini/Transform.h"

using namespace mlir;
using namespace buddy::gemmini;

namespace {

int64_t getNumberFromValue(Value &value) {
  return value.getDefiningOp()
      ->getAttr("value")
      .dyn_cast<IntegerAttr>()
      .getInt();
}

static acc_scale_t_bits acc_scale_t_to_acc_scale_t_bits(acc_scale_t x) {
  union {
    acc_scale_t_bits b;
    acc_scale_t f;
  } un;

  un.f = x;
  return un.b;
}

static scale_t_bits scale_t_to_scale_t_bits(scale_t x) {
  union {
    scale_t_bits b;
    scale_t f;
  } un;

  un.f = x;
  return un.b;
}

static size_t tiled_matmul_total_spad_rows(size_t I, size_t J, size_t K) {
  return (I * K + K * J) * DIM;
}

static size_t tiled_matmul_total_acc_rows(size_t I, size_t J) {
  return (I * J) * DIM;
}

}; // namespace

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

struct GemminiFlushOpLowering : public ConvertOpToLLVMPattern<FlushOp> {
  using ConvertOpToLLVMPattern<FlushOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(FlushOp flushOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = flushOp.getLoc();
    Value skip = flushOp.getSkip();
    IntegerAttr rs2Attr = rewriter.getI64IntegerAttr(0);
    Value rs2 =
        rewriter.create<arith::ConstantOp>(loc, rs2Attr, rewriter.getI64Type());
    rewriter.replaceOpWithNewOp<Flush_IntrOp>(flushOp, skip, rs2);
    return success();
  }
};

struct GemminiConfigStOpLowering : public ConvertOpToLLVMPattern<ConfigStOp> {
  using ConvertOpToLLVMPattern<ConfigStOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ConfigStOp configStOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value strideValue = configStOp.getStride();
    int stride = getNumberFromValue(strideValue);
    float scale = configStOp.getScale().convertToFloat();
    Type i64Type = rewriter.getI64Type();
    Attribute input0 = rewriter.getI64IntegerAttr(CONFIG_ST);
    Location loc = configStOp.getLoc();
    uint64_t arg = (uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)scale)
                       << 32 |
                   (uint32_t)stride;
    Attribute input1 = rewriter.getI64IntegerAttr(arg);
    Value value1 = rewriter.create<arith::ConstantOp>(loc, input0, i64Type);
    Value value2 = rewriter.create<arith::ConstantOp>(loc, input1, i64Type);
    rewriter.replaceOpWithNewOp<ConfigSt_IntrOp>(configStOp, value1, value2);
    return success();
  }
};

struct GemminiConfigLdOpLowering : public ConvertOpToLLVMPattern<ConfigLdOp> {
  using ConvertOpToLLVMPattern<ConfigLdOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ConfigLdOp configLdOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value rs2Value = configLdOp.getStride();
    float scale = configLdOp.getScale().convertToFloat();
    uint64_t rs1 = (uint64_t)scale_t_to_scale_t_bits(scale) << 32 |
                   ((uint64_t)16 << 16) | (uint64_t)1 << 8 |
                   configLdOp.getId() << 3 | configLdOp.getShrunk() << 2 |
                   CONFIG_LD;
    Type i64Type = rewriter.getI64Type();
    Attribute rs1Attr = rewriter.getI64IntegerAttr(rs1);
    Location loc = configLdOp.getLoc();
    Value rs1value = rewriter.create<arith::ConstantOp>(loc, rs1Attr, i64Type);
    rewriter.replaceOpWithNewOp<ConifgLd_IntrOp>(configLdOp, rs1value,
                                                 rs2Value);
    return success();
  }
};

struct GemminiConfigExOpLowering : public ConvertOpToLLVMPattern<ConfigExOp> {
  using ConvertOpToLLVMPattern<ConfigExOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ConfigExOp configExOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    IntegerType i64Type = rewriter.getI64Type();
    Location loc = configExOp.getLoc();
    float scale = configExOp.getSysAccScale().convertToFloat();
    uint64_t rs1 =
        (uint64_t)acc_scale_t_to_acc_scale_t_bits(scale) << 32 |
        configExOp.getAStride() << 16 | configExOp.getBTranspose() << 9 |
        configExOp.getATranspose() << 8 | configExOp.getSysAct() << 3 |
        configExOp.getDataflow() << 2 | CONFIG_EX;

    uint64_t rs2 = (uint64_t)1 << 48 | configExOp.getSysShift();
    IntegerAttr rs1Attr = rewriter.getI64IntegerAttr(rs1);
    IntegerAttr rs2Attr = rewriter.getI64IntegerAttr(rs2);
    Value rs1Value = rewriter.create<arith::ConstantOp>(loc, rs1Attr, i64Type);
    Value rs2Value = rewriter.create<arith::ConstantOp>(loc, rs2Attr, i64Type);
    rewriter.replaceOpWithNewOp<ConfigEX_IntrOp>(configExOp, rs1Value,
                                                 rs2Value);
    return success();
  }
};

struct GemminiMvinOpLowering : public ConvertOpToLLVMPattern<MvinOp> {
  using ConvertOpToLLVMPattern<MvinOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MvinOp mvinOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = mvinOp.getInput();
    Location loc = input.getLoc();
    MemRefType memRefType =
        mvinOp.getOperandTypes().front().dyn_cast<MemRefType>();
    llvm::ArrayRef<int64_t> memRefShape = memRefType.getShape();
    TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
    Value extractOp = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, resultType, input);
    IntegerType i64Type = rewriter.getI64Type();
    Value indexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOp);
    Value spadAddrValue = mvinOp.getAddr();
    uint64_t number = getNumberFromValue(spadAddrValue);
    uint64_t spadAddrInt = (uint64_t)memRefShape[0] << (ADDR_LEN + 16) |
                           (uint64_t)memRefShape[1] << ADDR_LEN | number;
    Attribute newSpadAddr = rewriter.getI64IntegerAttr(spadAddrInt);
    Value spad = rewriter.create<arith::ConstantOp>(loc, newSpadAddr, i64Type);
    rewriter.replaceOpWithNewOp<Mvin_IntrOp>(mvinOp, indexCastOp, spad);
    return success();
  }
};

struct GemminiMvoutLowering : public ConvertOpToLLVMPattern<MvoutOp> {
  using ConvertOpToLLVMPattern<MvoutOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MvoutOp mvoutOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
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
        mvoutOp.getOperandTypes().front().dyn_cast<MemRefType>();
    llvm::ArrayRef<int64_t> memRefShape = memRefType.getShape();
    uint64_t spadAddrInt = (uint64_t)memRefShape[0] << (ADDR_LEN + 16) |
                           (uint64_t)memRefShape[1] << ADDR_LEN | number;
    Attribute newSpadAddr = rewriter.getI64IntegerAttr(spadAddrInt);
    Value newSpad =
        rewriter.create<arith::ConstantOp>(loc, newSpadAddr, i64Type);
    rewriter.replaceOpWithNewOp<Mvout_IntrOp>(mvoutOp, indexCastOp, newSpad);
    return success();
  }
};

struct GemminiPreloadZerosLowering
    : public ConvertOpToLLVMPattern<PreloadZerosOp> {
  using ConvertOpToLLVMPattern<PreloadZerosOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(PreloadZerosOp preloadZerosOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value addr = preloadZerosOp.getAddr();
    Value cRows = preloadZerosOp.getCRows();
    Value cCols = preloadZerosOp.getCCols();
    Location loc = preloadZerosOp.getLoc();
    IntegerType i64Type = rewriter.getI64Type();
    uint64_t addrInt = getNumberFromValue(addr);
    uint64_t cRowsInt = getNumberFromValue(cRows);
    uint64_t cColsInt = getNumberFromValue(cCols);
    uint64_t rs1 = (uint64_t)16 << (ADDR_LEN + 16) | (uint64_t)16 << ADDR_LEN |
                   (uint64_t)-1;
    uint64_t rs2 =
        cRowsInt << (ADDR_LEN + 16) | cColsInt << (ADDR_LEN) | addrInt;
    Attribute rs1Attr = rewriter.getI64IntegerAttr(rs1);
    Attribute rs2Attr = rewriter.getI64IntegerAttr(rs2);
    Value rs1Value = rewriter.create<arith::ConstantOp>(loc, rs1Attr, i64Type);
    Value rs2Value = rewriter.create<arith::ConstantOp>(loc, rs2Attr, i64Type);
    rewriter.replaceOpWithNewOp<Preload_IntrOp>(preloadZerosOp, rs1Value,
                                                rs2Value);
    return success();
  }
};

struct GemminiPreloadLowering : public ConvertOpToLLVMPattern<PreloadOp> {
  using ConvertOpToLLVMPattern<PreloadOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(PreloadOp preloadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value bdAddr = preloadOp.getBdAddr();
    Value cAddr = preloadOp.getCAddr();
    Value bdCols = preloadOp.getBdCols();
    Value bdRows = preloadOp.getBdRows();
    Value cCols = preloadOp.getCCols();
    Value cRows = preloadOp.getBdRows();
    Location loc = preloadOp.getLoc();
    IntegerType i64Type = rewriter.getI64Type();
    uint64_t bdAddrInt = getNumberFromValue(bdAddr);
    uint64_t cAddrInt = getNumberFromValue(cAddr);
    uint64_t bdColsInt = getNumberFromValue(bdCols);
    uint64_t bdRowsInt = getNumberFromValue(bdRows);
    uint64_t cColsInt = getNumberFromValue(cCols);
    uint64_t cRowsInt = getNumberFromValue(cRows);
    uint64_t rs1 = bdRowsInt << (ADDR_LEN + 16) | bdColsInt << ADDR_LEN |
                   (uint64_t)bdAddrInt;
    uint64_t rs2 =
        cRowsInt << (ADDR_LEN + 16) | cColsInt << ADDR_LEN | (uint64_t)cAddrInt;
    Attribute rs1Attr = rewriter.getI64IntegerAttr(rs1);
    Attribute rs2Attr = rewriter.getI64IntegerAttr(rs2);
    Value rs1Value = rewriter.create<arith::ConstantOp>(loc, rs1Attr, i64Type);
    Value rs2Value = rewriter.create<arith::ConstantOp>(loc, rs2Attr, i64Type);
    rewriter.replaceOpWithNewOp<Preload_IntrOp>(preloadOp, rs1Value, rs2Value);
    return success();
  }
};

struct GemminiComputePreloadedLowering
    : public ConvertOpToLLVMPattern<ComputePreloadedOp> {
  using ConvertOpToLLVMPattern<ComputePreloadedOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ComputePreloadedOp computePreloadedOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value aAddr = computePreloadedOp.getAAddr();
    Value bdAddr = computePreloadedOp.getBdAddr();
    Value aRows = computePreloadedOp.getARows();
    Value aCols = computePreloadedOp.getACols();
    Value bdRows = computePreloadedOp.getBdRows();
    Value bdCols = computePreloadedOp.getBdCols();
    Location loc = computePreloadedOp.getLoc();
    IntegerType i64Type = rewriter.getI64Type();
    uint64_t aAddrInt = getNumberFromValue(aAddr);
    uint64_t bdAddrInt = getNumberFromValue(bdAddr);
    uint64_t aRowsInt = getNumberFromValue(aRows);
    uint64_t aColsInt = getNumberFromValue(aCols);
    uint64_t bdRowsInt = getNumberFromValue(bdRows);
    uint64_t bdColsInt = getNumberFromValue(bdCols);
    uint64_t rs1 =
        aRowsInt << (ADDR_LEN + 16) | aColsInt << ADDR_LEN | aAddrInt;
    uint64_t rs2 =
        bdRowsInt << (ADDR_LEN + 16) | bdColsInt << ADDR_LEN | bdAddrInt;
    Attribute rs1Attr = rewriter.getI64IntegerAttr(rs1);
    Attribute rs2Attr = rewriter.getI64IntegerAttr(rs2);
    Value rs1Value = rewriter.create<arith::ConstantOp>(loc, rs1Attr, i64Type);
    Value rs2Value = rewriter.create<arith::ConstantOp>(loc, rs2Attr, i64Type);
    rewriter.replaceOpWithNewOp<ComputePreloaded_IntrOp>(computePreloadedOp,
                                                         rs1Value, rs2Value);
    return success();
  }
};

struct GemminiComputeAccumulatedLowering
    : public ConvertOpToLLVMPattern<ComputeAccumulatedOp> {
  using ConvertOpToLLVMPattern<ComputeAccumulatedOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ComputeAccumulatedOp computeAccumulatedOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value aAddr = computeAccumulatedOp.getAAddr();
    Value bdAddr = computeAccumulatedOp.getBdAddr();
    Value aRows = computeAccumulatedOp.getARows();
    Value aCols = computeAccumulatedOp.getACols();
    Value bdRows = computeAccumulatedOp.getBdRows();
    Value bdCols = computeAccumulatedOp.getBdCols();
    Location loc = computeAccumulatedOp.getLoc();
    IntegerType i64Type = rewriter.getI64Type();
    uint64_t aAddrInt = getNumberFromValue(aAddr);
    uint64_t bdAddrInt = getNumberFromValue(bdAddr);
    uint64_t aRowsInt = getNumberFromValue(aRows);
    uint64_t aColsInt = getNumberFromValue(aCols);
    uint64_t bdRowsInt = getNumberFromValue(bdRows);
    uint64_t bdColsInt = getNumberFromValue(bdCols);
    uint64_t rs1 =
        aRowsInt << (ADDR_LEN + 16) | aColsInt << ADDR_LEN | aAddrInt;
    uint64_t rs2 =
        bdRowsInt << (ADDR_LEN + 16) | bdColsInt << ADDR_LEN | bdAddrInt;
    Attribute rs1Attr = rewriter.getI64IntegerAttr(rs1);
    Attribute rs2Attr = rewriter.getI64IntegerAttr(rs2);
    Value rs1Value = rewriter.create<arith::ConstantOp>(loc, rs1Attr, i64Type);
    Value rs2Value = rewriter.create<arith::ConstantOp>(loc, rs2Attr, i64Type);
    rewriter.replaceOpWithNewOp<ComputeAccumulated_IntrOp>(computeAccumulatedOp,
                                                           rs1Value, rs2Value);

    return success();
  }
};

class GemminiTileMatMulLowering : public ConvertOpToLLVMPattern<TileMatMulOp> {
  void gemmini_loop_ws(size_t i, size_t j, size_t k, size_t padI, size_t padJ,
                       size_t padK, Value &a, Value &b, Value &d, Value &c,
                       size_t aRowStride, size_t bRowStride, size_t dRowStride,
                       size_t cRowStride, bool aTranspose, bool bTranspose,
                       bool fullC, bool lowD, bool exAccumulate, int act,
                       TileMatMulOp &tileMatMulOp,
                       ConversionPatternRewriter &rewriter) const {
    // loopWsConfigBounds instruction.
    uint64_t rs1 = (uint64_t)padK << 32 | (uint64_t)padJ << 16 | (uint64_t)padI;
    uint64_t rs2 = (uint64_t)k << 32 | (uint64_t)j << 16 | (uint64_t)i;
    IntegerType i64Type = rewriter.getI64Type();
    Location loc = a.getLoc();
    IntegerAttr rs1Attr = rewriter.getI64IntegerAttr(rs1);
    IntegerAttr rs2Attr = rewriter.getI64IntegerAttr(rs2);
    Value rs1Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs1Attr);
    Value rs2Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs2Attr);
    rewriter.create<LoopWsConfigBounds_IntrOp>(loc, rs1Value, rs2Value);
    // loopWsConfigAddrsAB instruction.
    rewriter.create<LoopWsConfigAddrsAB_IntrOp>(loc, a, b);
    // loopWsConfigAddrsDC instruction
    rewriter.create<LoopWsConfigAddrsDC_IntrOp>(loc, d, c);
    // loopWsConfigStridesAB instruction
    rs1Attr = rewriter.getI64IntegerAttr(aRowStride);
    rs2Attr = rewriter.getI64IntegerAttr(bRowStride);
    rs1Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs1Attr);
    rs2Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs2Attr);
    rewriter.create<LoopWsConfigStridesAB_IntrOp>(loc, rs1Value, rs2Value);
    // loopWsConfigStrideDC instruction
    rs1Attr = rewriter.getI64IntegerAttr(dRowStride);
    rs2Attr = rewriter.getI64IntegerAttr(cRowStride);
    rs1Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs1Attr);
    rs2Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs2Attr);
    rewriter.create<LoopWsConfigStridesDC_IntrOp>(loc, rs1Value, rs2Value);
    rs1 = (uint64_t)act << 8 | lowD << 2 | (fullC) << 1 | exAccumulate;
    rs2 = bTranspose << 1 | aTranspose;
    rs1Attr = rewriter.getI64IntegerAttr(rs1);
    rs2Attr = rewriter.getI64IntegerAttr(rs2);
    rs1Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs1Attr);
    rs2Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs2Attr);
    rewriter.create<LoopWs_IntrOp>(loc, rs1Value, rs2Value);
  }

  void inner(Value &a, Value &b, Value &pre, Value &out, scale_t aScaleFactor,
             scale_t bScaleFactor, scale_acc_t dScaleFactor, size_t i, size_t j,
             size_t k, size_t padI, size_t padJ, size_t padK, size_t strideA,
             size_t strideB, size_t strideD, size_t strideC, bool aTranspose,
             bool bTranspose, bool fullC, bool lowD, bool noBias,
             bool repeatingBias, int act, TileMatMulOp &tileMatMulOp,
             ConversionPatternRewriter &rewriter) const {

    gemmini_loop_ws(i, j, k, padI, padJ, padK, a, b, pre, out, strideA, strideB,
                    repeatingBias ? 0 : strideD, strideC, aTranspose,
                    bTranspose, fullC, lowD, !noBias, act, tileMatMulOp,
                    rewriter);
  }

  void tiledMatmulOuter(size_t dimI, size_t dimJ, size_t dimK, Value &A,
                        Value &B, Value &D, Value &C, size_t strideA,
                        size_t strideB, size_t strideD, size_t strideC,
                        scale_t aScaleFactor, scale_t bScaleFactor,
                        scale_acc_t dScaleFactor, size_t tileI, size_t tileJ,
                        size_t tileK, int act, acc_scale_t scale,
                        acc_scale_t bertScale, bool repeatingBias,
                        bool aTranspose, bool bTranspose, bool fullC, bool lowD,
                        uint8_t weightA, TileMatMulOp &tileMatMulOp,
                        ConversionPatternRewriter &rewriter) const {
    const size_t dimIPadded = (dimI / DIM + (dimI % DIM != 0)) * DIM;
    const size_t dimJPadded = (dimJ / DIM + (dimJ % DIM != 0)) * DIM;
    const size_t dimKPadded = (dimK / DIM + (dimK % DIM != 0)) * DIM;
    const size_t I0 =
        dimIPadded / (tileI * DIM) + (dimIPadded % (tileI * DIM) != 0);
    const size_t J0 =
        dimJPadded / (tileJ * DIM) + (dimJPadded % (tileJ * DIM) != 0);
    const size_t K0 =
        dimKPadded / (tileK * DIM) + (dimKPadded % (tileK * DIM) != 0);
    const size_t lastI =
        dimIPadded % (tileI * DIM) == 0 ? tileI : (dimIPadded / DIM) % tileI;
    const size_t lastJ =
        dimJPadded % (tileJ * DIM) == 0 ? tileJ : (dimJPadded / DIM) % tileJ;
    const size_t lastK =
        dimKPadded % (tileK * DIM) == 0 ? tileK : (dimKPadded / DIM) % tileK;
    const size_t paddingI = dimIPadded - dimI;
    const size_t paddingJ = dimJPadded - dimJ;
    const size_t paddingK = dimKPadded - dimK;
    const bool noBias = false;
    const size_t sizeofD = lowD ? sizeof(elem_t) : sizeof(acc_t);
    const size_t sizeofC = fullC ? sizeof(acc_t) : sizeof(elem_t);
    Location loc = tileMatMulOp.getLoc();
    llvm::APFloat accScaleIdentity((float)ACC_SCALE_IDENTITY);
    rewriter.create<ConfigExOp>(loc, /*dataflow = */ 1, /*sysAct = */ act & 3,
                                /* sysShift = */ 0, accScaleIdentity);
    Attribute strideAttr = rewriter.getI64IntegerAttr(strideC * sizeofC);
    Value strideValue = rewriter.create<arith::ConstantOp>(
        loc, strideAttr, rewriter.getI64Type());
    rewriter.create<ConfigStOp>(loc, strideValue, act & 3,
                                llvm::APFloat(scale));
    strideAttr = rewriter.getI64IntegerAttr(strideA * sizeof(elem_t));
    strideValue = rewriter.create<arith::ConstantOp>(loc, strideAttr,
                                                     rewriter.getI64Type());
    rewriter.create<ConfigLdOp>(loc, strideValue, llvm::APFloat(aScaleFactor),
                                false, 0);
    strideAttr = rewriter.getI64IntegerAttr(strideB * sizeof(elem_t));
    strideValue = rewriter.create<arith::ConstantOp>(loc, strideAttr,
                                                     rewriter.getI64Type());
    rewriter.create<ConfigLdOp>(loc, strideValue, llvm::APFloat(bScaleFactor),
                                false, 1);
    strideAttr = rewriter.getI16IntegerAttr(strideD * sizeofD);
    strideValue = rewriter.create<arith::ConstantOp>(loc, strideAttr,
                                                     rewriter.getI64Type());
    rewriter.create<ConfigLdOp>(loc, strideValue,
                                llvm::APFloat((float)dScaleFactor), lowD, 2);
    for (size_t i0 = 0; i0 < I0; i0++)
      for (size_t j0 = 0; j0 < J0; j0++)
        for (size_t k0 = 0; k0 < K0; k0++) {
          Value pre;
          Location loc = A.getLoc();
          if (k0 != 0) {
            IntegerAttr preAttr = rewriter.getI64IntegerAttr(0);
            pre = rewriter.create<arith::ConstantOp>(loc, preAttr,
                                                     rewriter.getI64Type());
          } else {
            size_t biasRow = repeatingBias ? 0 : i0 * tileI * DIM;
            size_t offset = (biasRow * strideD + j0 * tileJ * DIM) * sizeofD *
                            sizeof(elem_t);
            IntegerAttr offsetAttr = rewriter.getI64IntegerAttr(offset);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, offsetAttr, rewriter.getI64Type());
            pre = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), D,
                                                 offsetValue);
          }

          Value out;
          if (k0 == K0 - 1) {
            size_t offset = (i0 * tileI * DIM * strideC + j0 * tileJ * DIM) *
                            sizeofC * sizeof(elem_t);
            IntegerAttr offsetAttr = rewriter.getI64IntegerAttr(offset);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, offsetAttr, rewriter.getI64Type());
            out = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), C,
                                                 offsetValue);
          } else {
            IntegerAttr outAttr = rewriter.getI64IntegerAttr(0);
            out = rewriter.create<arith::ConstantOp>(loc, outAttr,
                                                     rewriter.getI64Type());
          }
          const size_t i = i0 < I0 - 1 ? tileI : lastI;
          const size_t j = j0 < J0 - 1 ? tileJ : lastJ;
          const size_t k = k0 < K0 - 1 ? tileK : lastK;
          const size_t padI = i0 == I0 - 1 ? paddingI : 0;
          const size_t padJ = j0 == J0 - 1 ? paddingJ : 0;
          const size_t padK = k0 == K0 - 1 ? paddingK : 0;
          Value a;
          if (aTranspose) {
            size_t offset = (k0 * tileK * DIM * strideA + i0 * tileI * DIM) *
                            sizeof(elem_t);
            IntegerAttr offsetAttr = rewriter.getI64IntegerAttr(offset);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, offsetAttr, rewriter.getI64Type());
            a = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), A,
                                               offsetValue);
          } else {
            size_t offset = (i0 * tileI * DIM * strideA + k0 * tileK * DIM) *
                            sizeof(elem_t);
            IntegerAttr offsetAttr = rewriter.getI64IntegerAttr(offset);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, offsetAttr, rewriter.getI64Type());
            a = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), A,
                                               offsetValue);
          }
          Value b;
          if (bTranspose) {
            size_t offset = (j0 * tileJ * DIM * strideB + k0 * tileK * DIM) *
                            sizeof(elem_t);
            IntegerAttr offsetAttr = rewriter.getI64IntegerAttr(offset);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, offsetAttr, rewriter.getI64Type());
            b = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), B,
                                               offsetValue);
          } else {
            size_t offset = (k0 * tileK * DIM * strideB + j0 * tileJ * DIM) *
                            sizeof(elem_t);
            IntegerAttr offsetAttr = rewriter.getI64IntegerAttr(offset);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, offsetAttr, rewriter.getI64Type());
            b = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), B,
                                               offsetValue);
          }
          inner(a, b, pre, out, aScaleFactor, bScaleFactor, dScaleFactor, i, j,
                k, padI, padJ, padK, strideA, strideB, strideD, strideC,
                aTranspose, bTranspose, fullC, lowD, noBias, repeatingBias, act,
                tileMatMulOp, rewriter);
        }
    IntegerAttr flushAttr = rewriter.getI64IntegerAttr(0);
    Value flushValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(), flushAttr);
    rewriter.replaceOpWithNewOp<Flush_IntrOp>(tileMatMulOp, flushValue,
                                              flushValue);
    return;
  }

public:
  using ConvertOpToLLVMPattern<TileMatMulOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(TileMatMulOp tileMatMulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
#define partitionRows (BANK_NUM * BANK_ROWS / 2)
#define matsInPartition (partition_rows / DIM)
#define matsInAcc (ACC_ROWS / DIM)
#define maxTileIJ ((size_t)sqrt(mats_in_acc))
#define maxTileK (matsInPartition / maxTileIJ)

#define dbPartitionRows ((BANK_NUM * BANK_ROWS / 2) / 2)
#define dbMatsInPartition (dbPartitionRows / DIM)
#define dbMatsInAcc ((ACC_ROWS / 2) / DIM)
#define dbMaxTileIJ ((size_t)sqrt(dbMatsInAcc))
#define dbMaxTileK (dbMatsInPartition / dbMaxTileIJ)

    Value aArray = tileMatMulOp.getAArray();
    Value bArray = tileMatMulOp.getBArray();
    Value cArray = tileMatMulOp.getCArray();
    Value dArray = tileMatMulOp.getDArray();
    TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
    Location loc = tileMatMulOp.getLoc();
    IntegerType i64Type = rewriter.getI64Type();
    Value aArrayExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, resultType,
                                                                aArray);
    Value aArrayindexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, aArrayExtractOp);
    Value bArrayExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, resultType,
                                                                bArray);
    Value bArrayindexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, bArrayExtractOp);
    Value cArrayExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, resultType,
                                                                cArray);
    Value cArrayindexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, cArrayExtractOp);
    Value dArrayExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, resultType,
                                                                dArray);
    Value dArrayindexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, dArrayExtractOp);
    MemRefType aArrayType = aArray.getType().dyn_cast<MemRefType>();
    MemRefType bArrayType = bArray.getType().dyn_cast<MemRefType>();
    MemRefType cArrayType = cArray.getType().dyn_cast<MemRefType>();
    MemRefType dArrayType = dArray.getType().dyn_cast<MemRefType>();
    llvm::ArrayRef<long> aArrayShape = aArrayType.getShape();
    llvm::ArrayRef<long> bArrayShape = bArrayType.getShape();
    llvm::ArrayRef<long> cArrayShape = cArrayType.getShape();
    llvm::ArrayRef<long> dArrayShape = dArrayType.getShape();
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
    size_t dimIPaded = (dimI / DIM + (dimI % DIM != 0)) * DIM;
    size_t dimJPaded = (dimJ / DIM + (dimJ % DIM != 0)) * DIM;
    size_t dimKPaded = (dimK / DIM + (dimK % DIM != 0)) * DIM;
    size_t maxSpadRows = BANK_NUM * BANK_ROWS / 2;
    size_t maxAccRows = ACC_ROWS / 2;
    size_t tileI, tileJ, tileK;
    if (act == LAYERNORM || act == SOFTMAX) {
      tileI = 1;
      tileJ = dimJPaded | DIM;
      tileK = 1;
    } else {
      tileI = dimIPaded / DIM < dbMaxTileIJ ? dimIPaded / DIM : dbMaxTileIJ;
      tileJ = dimJPaded / DIM < dbMaxTileIJ ? dimJPaded / DIM : dbMaxTileIJ;
      tileK = dimKPaded / DIM < dbMaxTileK ? dimKPaded / DIM : dbMaxTileK;
    }
    while (true) {
      bool increased = false;

      if (tiled_matmul_total_spad_rows(tileI, tileJ + 1, tileK) <=
              maxSpadRows &&
          tiled_matmul_total_acc_rows(tileI, tileJ + 1) <= maxAccRows &&
          (tileJ + 1) * DIM <= dimJPaded) {
        tileJ++;
        increased = true;
      }

      if (tiled_matmul_total_spad_rows(tileI + 1, tileJ, tileK) <=
              maxSpadRows &&
          tiled_matmul_total_acc_rows(tileI + 1, tileJ) <= maxAccRows &&
          (tileI + 1) * DIM <= dimIPaded) {
        tileI++;
        increased = true;
      }

      if (tiled_matmul_total_spad_rows(tileI, tileJ, tileK + 1) <=
              maxSpadRows &&
          (tileK + 1) * DIM <= dimKPaded) {
        tileK++;
        increased = true;
      }
      if (!increased)
        break;
    }

#undef partitionRows
#undef matsInPartition
#undef matsInAcc
#undef maxTileIJ
#undef maxTileK

#undef dbPartitionRows
#undef dbMatsInPartition
#undef dbMatsInAcc
#undef dbMaxTileIJ
#undef dbMaxTileK

    tiledMatmulOuter(dimI, dimJ, dimK, aArrayindexCastOp, bArrayindexCastOp,
                     dArrayindexCastOp, cArrayindexCastOp, strideA, strideB,
                     strideD, strideC, aScaleFactor, bScaleFactor, dScaleFactor,
                     tileI, tileJ, tileK, act, scale, bertScale, repeatingBias,
                     aTranspose, bTranspose, fullC, lowD, weightA, tileMatMulOp,
                     rewriter);
    return success();
  };
};

void mlir::populateGemminiLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns
      .add<ForwardOperands<func::CallOp>, ForwardOperands<func::CallIndirectOp>,
           ForwardOperands<func::ReturnOp>>(converter, &converter.getContext());
  patterns.add<GemminiFlushOpLowering>(converter);
  patterns.add<GemminiConfigStOpLowering>(converter);
  patterns.add<GemminiConfigLdOpLowering>(converter);
  patterns.add<GemminiMvinOpLowering>(converter);
  patterns.add<GemminiMvoutLowering>(converter);
  patterns.add<GemminiConfigExOpLowering>(converter);
  patterns.add<GemminiPreloadZerosLowering>(converter);
  patterns.add<GemminiPreloadLowering>(converter);
  patterns.add<GemminiComputePreloadedLowering>(converter);
  patterns.add<GemminiComputeAccumulatedLowering>(converter);
  patterns.add<GemminiTileMatMulLowering>(converter);
}

void mlir::configureGemminiegalizeForExportTarget(
    LLVMConversionTarget &target) {
  target.addLegalOp<Flush_IntrOp, ConfigSt_IntrOp, ConifgLd_IntrOp,
                    ConfigEX_IntrOp, Mvin_IntrOp, Mvout_IntrOp, Preload_IntrOp,
                    ComputePreloaded_IntrOp, ComputeAccumulated_IntrOp,
                    LoopWsConfigBounds_IntrOp, LoopWsConfigAddrsAB_IntrOp,
                    LoopWsConfigAddrsDC_IntrOp, LoopWsConfigStridesAB_IntrOp,
                    LoopWsConfigStridesDC_IntrOp, LoopWs_IntrOp>();
  target.addIllegalOp<FlushOp, ConfigStOp, ConfigLdOp, ConfigExOp, MvinOp,
                      MvoutOp, PrintOp, PreloadZerosOp, PreloadOp,
                      ComputePreloadedOp, ComputeAccumulatedOp, TileMatMulOp>();
}
