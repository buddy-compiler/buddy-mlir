//===- LegalizeForLLVMExport.cpp - Prepare Gemmini for LLVM translation ---===//
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
#include <stdio.h>

using namespace mlir;
using namespace buddy::gemmini;

namespace {

int64_t getNumberFromValue(Value &value) {
  return value.getDefiningOp()
      ->getAttr("value")
      .dyn_cast<IntegerAttr>()
      .getInt();
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
    Value rs2 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
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
    Location loc = configStOp.getLoc();
    uint64_t arg = (uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)scale)
                       << 32 |
                   (uint32_t)stride;
    Value value1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(CONFIG_ST));
    Value value2 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(arg));
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
    Location loc = configLdOp.getLoc();
    Value rs1value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
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
    Location loc = configExOp.getLoc();
    float scale = configExOp.getSysAccScale().convertToFloat();
    uint64_t rs1 =
        (uint64_t)acc_scale_t_to_acc_scale_t_bits(scale) << 32 |
        configExOp.getAStride() << 16 | configExOp.getBTranspose() << 9 |
        configExOp.getATranspose() << 8 | configExOp.getSetOnlyStrides() << 7 |
        configExOp.getSysAct() << 3 | configExOp.getDataflow() << 2 | CONFIG_EX;

    uint64_t rs2 = configExOp.getCStride() << 48 | configExOp.getSysShift();
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
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
    Value spad = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(spadAddrInt));
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
    Value newSpad = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(spadAddrInt));
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
    uint64_t addrInt = getNumberFromValue(addr);
    uint64_t cRowsInt = getNumberFromValue(cRows);
    uint64_t cColsInt = getNumberFromValue(cCols);
    uint64_t rs1 = (uint64_t)16 << (ADDR_LEN + 16) | (uint64_t)16 << ADDR_LEN |
                   (uint64_t)-1;
    uint64_t rs2 =
        cRowsInt << (ADDR_LEN + 16) | cColsInt << (ADDR_LEN) | addrInt;
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
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
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
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
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
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
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
    rewriter.replaceOpWithNewOp<ComputeAccumulated_IntrOp>(computeAccumulatedOp,
                                                           rs1Value, rs2Value);

    return success();
  }
};

class GemminiTileMatMulLowering : public ConvertOpToLLVMPattern<TileMatMulOp> {
  void gemminiLoopWs(size_t i, size_t j, size_t k, size_t padI, size_t padJ,
                     size_t padK, Value &a, Value &b, Value &d, Value &c,
                     size_t aRowStride, size_t bRowStride, size_t dRowStride,
                     size_t cRowStride, bool aTranspose, bool bTranspose,
                     bool fullC, bool lowD, bool exAccumulate, int act,
                     TileMatMulOp &tileMatMulOp,
                     ConversionPatternRewriter &rewriter) const {
    // loopWsConfigBounds instruction.
    uint64_t rs1 = (uint64_t)padK << 32 | (uint64_t)padJ << 16 | (uint64_t)padI;
    uint64_t rs2 = (uint64_t)k << 32 | (uint64_t)j << 16 | (uint64_t)i;
    Location loc = a.getLoc();
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
    rewriter.create<LoopWsConfigBounds_IntrOp>(loc, rs1Value, rs2Value);
    // loopWsConfigAddrsAB instruction.
    rewriter.create<LoopWsConfigAddrsAB_IntrOp>(loc, a, b);
    // loopWsConfigAddrsDC instruction
    rewriter.create<LoopWsConfigAddrsDC_IntrOp>(loc, d, c);
    // loopWsConfigStridesAB instruction
    rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(aRowStride));
    rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(bRowStride));
    rewriter.create<LoopWsConfigStridesAB_IntrOp>(loc, rs1Value, rs2Value);
    // loopWsConfigStrideDC instruction
    rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(dRowStride));
    rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(cRowStride));
    rewriter.create<LoopWsConfigStridesDC_IntrOp>(loc, rs1Value, rs2Value);
    rs1 = (uint64_t)act << 8 | lowD << 2 | (fullC) << 1 | exAccumulate;
    rs2 = bTranspose << 1 | aTranspose;
    rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
    rewriter.create<LoopWs_IntrOp>(loc, rs1Value, rs2Value);
  }

  void inner(Value &a, Value &b, Value &pre, Value &out, scale_t aScaleFactor,
             scale_t bScaleFactor, scale_acc_t dScaleFactor, size_t i, size_t j,
             size_t k, size_t padI, size_t padJ, size_t padK, size_t strideA,
             size_t strideB, size_t strideD, size_t strideC, bool aTranspose,
             bool bTranspose, bool fullC, bool lowD, bool noBias,
             bool repeatingBias, int act, TileMatMulOp &tileMatMulOp,
             ConversionPatternRewriter &rewriter) const {

    gemminiLoopWs(i, j, k, padI, padJ, padK, a, b, pre, out, strideA, strideB,
                  repeatingBias ? 0 : strideD, strideC, aTranspose, bTranspose,
                  fullC, lowD, !noBias, act, tileMatMulOp, rewriter);
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
    Value strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(strideC * sizeofC));
    rewriter.create<ConfigStOp>(loc, strideValue, act & 3,
                                llvm::APFloat(scale));
    strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(strideA * sizeof(elem_t)));
    rewriter.create<ConfigLdOp>(loc, strideValue, llvm::APFloat(aScaleFactor),
                                false, 0);
    strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(strideB * sizeof(elem_t)));
    rewriter.create<ConfigLdOp>(loc, strideValue, llvm::APFloat(bScaleFactor),
                                false, 1);
    strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(strideD * sizeofD));
    rewriter.create<ConfigLdOp>(loc, strideValue,
                                llvm::APFloat((float)dScaleFactor), lowD, 2);
    for (size_t i0 = 0; i0 < I0; i0++)
      for (size_t j0 = 0; j0 < J0; j0++)
        for (size_t k0 = 0; k0 < K0; k0++) {
          Value pre;
          Location loc = A.getLoc();
          if (k0 != 0) {
            pre = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64IntegerAttr(0));
          } else {
            size_t biasRow = repeatingBias ? 0 : i0 * tileI * DIM;
            size_t offset = (biasRow * strideD + j0 * tileJ * DIM) * sizeofD *
                            sizeof(elem_t);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64IntegerAttr(offset));
            pre = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), D,
                                                 offsetValue);
          }

          Value out;
          if (k0 == K0 - 1) {
            size_t offset = (i0 * tileI * DIM * strideC + j0 * tileJ * DIM) *
                            sizeofC * sizeof(elem_t);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64IntegerAttr(offset));
            out = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), C,
                                                 offsetValue);
          } else {
            out = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64IntegerAttr(0));
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
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64IntegerAttr(offset));
            a = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), A,
                                               offsetValue);
          } else {
            size_t offset = (i0 * tileI * DIM * strideA + k0 * tileK * DIM) *
                            sizeof(elem_t);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64IntegerAttr(offset));
            a = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), A,
                                               offsetValue);
          }
          Value b;
          if (bTranspose) {
            size_t offset = (j0 * tileJ * DIM * strideB + k0 * tileK * DIM) *
                            sizeof(elem_t);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64IntegerAttr(offset));
            b = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), B,
                                               offsetValue);
          } else {
            size_t offset = (k0 * tileK * DIM * strideB + j0 * tileJ * DIM) *
                            sizeof(elem_t);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64IntegerAttr(offset));
            b = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), B,
                                               offsetValue);
          }
          inner(a, b, pre, out, aScaleFactor, bScaleFactor, dScaleFactor, i, j,
                k, padI, padJ, padK, strideA, strideB, strideD, strideC,
                aTranspose, bTranspose, fullC, lowD, noBias, repeatingBias, act,
                tileMatMulOp, rewriter);
        }
    Value flushValue =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
    rewriter.replaceOpWithNewOp<Flush_IntrOp>(tileMatMulOp, flushValue,
                                              flushValue);
    return;
  }

  size_t tiledMatmulTotalSpadRows(size_t I, size_t J, size_t K) const {
    return (I * K + K * J) * DIM;
  }

  size_t tiledMatmulTotalAccRows(size_t I, size_t J) const {
    return (I * J) * DIM;
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

      if (tiledMatmulTotalSpadRows(tileI, tileJ + 1, tileK) <= maxSpadRows &&
          tiledMatmulTotalAccRows(tileI, tileJ + 1) <= maxAccRows &&
          (tileJ + 1) * DIM <= dimJPaded) {
        tileJ++;
        increased = true;
      }

      if (tiledMatmulTotalSpadRows(tileI + 1, tileJ, tileK) <= maxSpadRows &&
          tiledMatmulTotalAccRows(tileI + 1, tileJ) <= maxAccRows &&
          (tileI + 1) * DIM <= dimIPaded) {
        tileI++;
        increased = true;
      }

      if (tiledMatmulTotalSpadRows(tileI, tileJ, tileK + 1) <= maxSpadRows &&
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

class GemminiTileConvOpLowering : public ConvertOpToLLVMPattern<TileConvOp> {

  void gemminiLoopConvWs(
      int batchSize, int inDim, int inChannels, int outChannels, int outDim,
      int poolOutDim, int stride, int padding, int kernelDim,
      int kernelDilation, int poolSize, int poolStride, int poolPadding,
      int batches, int porows, int pocols, int pochs, int krows, int kcols,
      int kchs, int lpad, int rpad, int upad, int dpad, int plpad, int prpad,
      int pupad, int pdpad, int orows, int ocols, Value &weights, Value &output,
      Value &bias, Value &input, bool noBias, bool noPool, bool downsample,
      bool writ180, bool inputDilated, int act, bool transOutput1203,
      bool transWeight1203, bool transWeight0132, bool transInput3120,
      int maxPixelsPerRow, bool dw, TileConvOp &tileConvOp,
      ConversionPatternRewriter &rewriter) const {
    Location loc = tileConvOp.getLoc();
    IntegerType i64Type = rewriter.getI64Type();
    // loopConvWsConfig1
    uint64_t rs1 = (uint64_t)outChannels << 48 | (uint64_t)inChannels << 32 |
                   (uint64_t)inDim << 16 | (uint64_t)batchSize;
    uint64_t rs2 = (uint64_t)padding << 48 | (uint64_t)stride << 32 |
                   (uint64_t)poolOutDim << 16 | (uint64_t)outDim;
    IntegerAttr rs1Attr = rewriter.getI64IntegerAttr(rs1);
    IntegerAttr rs2Attr = rewriter.getI64IntegerAttr(rs2);
    Value rs1Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs1Attr);
    Value rs2Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs2Attr);
    rewriter.create<LoopConvWsConfig1_IntrOp>(loc, rs1Value, rs2Value);
    // loopConvWsConfig2
    rs1 = (uint64_t)kernelDim << 48 | (uint64_t)poolSize << 32 |
          (uint64_t)poolStride << 16 | (uint64_t)poolPadding;
    rs2 = (uint64_t)batches << 48 | (uint64_t)porows << 32 |
          (uint64_t)pocols << 16 | (uint64_t)pochs;
    rs1Attr = rewriter.getI64IntegerAttr(rs1);
    rs2Attr = rewriter.getI64IntegerAttr(rs2);
    rs1Value = rewriter.create<arith::ConstantOp>(loc,i64Type, rs1Attr);
    rs2Value = rewriter.create<arith::ConstantOp>(loc,i64Type, rs2Attr);
    rewriter.create<LoopConvWsConfig2_IntrOp>(loc, rs1Value, rs2Value);
    // loopConvWsConfig3
    rs1 = (uint64_t)krows << 48 | (uint64_t)kcols << 32 | (uint64_t)kchs << 16 |
          (uint64_t)lpad;
    rs2 = (uint64_t)rpad << 48 | (uint64_t)upad << 32 | (uint64_t)dpad << 16 |
          (uint64_t)plpad;
    rs1Attr = rewriter.getI64IntegerAttr(rs1);
    rs2Attr = rewriter.getI64IntegerAttr(rs2);
    rs1Value = rewriter.create<arith::ConstantOp>(loc,i64Type, rs1Attr);
    rs2Value = rewriter.create<arith::ConstantOp>(loc,i64Type, rs2Attr);
    rewriter.create<LoopConvWsConfig3_IntrOp>(loc, rs1Value, rs2Value);
    // loopConvWsConfig4
    rs1 = (uint64_t)orows << 48 | (uint64_t)prpad << 32 |
          (uint64_t)pupad << 16 | (uint64_t)pdpad;
    rs2 = (uint64_t)kernelDilation << 16 | (uint64_t)ocols;
    rs1Attr = rewriter.getI64IntegerAttr(rs1);
    rs2Attr = rewriter.getI64IntegerAttr(rs2);
    rs1Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs1Attr);
    rs2Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs2Attr);
    rewriter.create<LoopConvWsConfig4_IntrOp>(loc, rs1Value, rs2Value);
    // loopConvWsconfig5
    rewriter.create<LoopConvWsConfig5_IntrOp>(loc, weights, output);
    // loopConvWsconfig6
    rewriter.create<LoopConvWsConfig6_IntrOp>(loc, bias, input);
    // loopConvWs
    rs1 = (uint64_t)maxPixelsPerRow << 8 | dw << 6 | transInput3120 << 5 |
          transWeight0132 << 4 | transWeight1203 << 3 | transOutput1203 << 2 |
          writ180 << 1 | noBias;
    rs2 = act << 3 | inputDilated << 2 | downsample << 1 | noPool;
    rs1Attr = rewriter.getI64IntegerAttr(rs1);
    rs2Attr = rewriter.getI64IntegerAttr(rs2);
    rs1Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs1Attr);
    rs2Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs2Attr);
    rewriter.create<LoopConvWs_IntrOp>(loc, rs1Value, rs2Value);
  }

  void spTiledConv(int batchSize, int inDim, int inChannels, int outChannels,
                   int outDim, int poolOutDim, int stride, int padding,
                   int kernelDim, int kernelDilation, int poolSize,
                   int poolStride, int poolPadding, int batches, int porows,
                   int pocols, int pochs, int krows, int kcols, int kchs,
                   int lpad, int rpad, int upad, int dpad, int plpad, int prpad,
                   int pupad, int pdpad, Value &input, Value &weights,
                   Value &output, Value &bias, int act, acc_scale_t scale,
                   bool wrot180, bool transOutput1203, bool transInput3120,
                   bool transWeight1203, bool transWeight0132, bool noBias,
                   bool noPool, bool downsample, bool inputDilated, bool dw,
                   TileConvOp &tileConvOp,
                   ConversionPatternRewriter &rewriter) const {
    if (dw) {
      kchs = 1;
      pochs = 1;
    }

    const int orows = porows * poolStride + poolSize - 1 - pupad - pdpad;
    const int ocols = pocols * poolStride + poolSize - 1 - plpad - prpad;
    const int ichs = kchs;

#ifdef HAS_FIRST_LAYER_OPTIMIZATIONS
    const bool transposed =
        transOutput1203 || transInput3120 || transWeight1203 || transWeight0132;
    int maxPixelsPerRow = transposed || wrot180 || downsample || inputDilated ||
                                  kernelDilation > 1 || ichs > DIM
                              ? 1
                              : DIM / ichs;
    if (maxPixelsPerRow > kcols)
      maxPixelsPerRow = kcols;
#else
    const int maxPixelsPerRow = 1;
#endif
    gemminiLoopConvWs(
        batchSize, inDim, inChannels, outChannels, outDim, poolOutDim, stride,
        padding, kernelDim, kernelDilation, poolSize, poolStride, poolPadding,
        batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad,
        dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias,
        input, noBias, noPool, downsample, wrot180, inputDilated, act,
        transOutput1203, transWeight1203, transWeight0132, transInput3120,
        maxPixelsPerRow, dw, tileConvOp, rewriter);
  }

  void tiledConv(int batchSize, int inDim, int inChannels, int outChannels,
                 int outDim, int stride, int inputDilation, int kernelDilation,
                 int padding, int kernelDim, bool wrot180, bool transOutput1203,
                 bool transInput3120, bool transWeight1203,
                 bool transWeight0132, int batches, int porows, int pocols,
                 int pochs, int krows, int kcols, int kchs, const Value &input,
                 const Value &weights, const Value &bias, Value &output,
                 int act, acc_scale_t scale, int poolSize, int poolStride,
                 int poolPadding, TileConvOp &tileConvOp,
                 ConversionPatternRewriter &rewriter) const {
    bool noBias = false;
    bool noPool = poolStride == 0;
    if (noPool) {
      poolSize = 1;
      poolStride = 1;
      poolPadding = 0;
    }
    const bool downsample = stride == 2 && kernelDim == 1 && inDim % 2 == 0 &&
                            padding == 0 && noPool && inputDilation == 1 &&
                            !transInput3120;
    const int inputDilated = inputDilation == 2;
    int64_t stDramStride = transOutput1203
                               ? batchSize * outChannels * sizeof(elem_t)
                               : outChannels * sizeof(elem_t);
    IntegerAttr strideAttr = rewriter.getI64IntegerAttr(stDramStride);
    Location loc = tileConvOp.getLoc();
    Value strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(), strideAttr);
    rewriter.create<ConfigStOp>(loc, strideValue, act, llvm::APFloat(scale));
    rewriter.create<ConfigExOp>(
        loc, /*dataflow = */ 1, /*act = */ 0, /*shift = */ 0,
        /*scale = */ llvm::APFloat((float)0), /*cStride = */ inputDilation,
        /*aStride = */ stride >> downsample,
        /*aTranspose = */ transInput3120, /*bTranspose*/ transWeight0132,
        /*setOnlyStrides = */ false);
    const int poolOutDim =
        (outDim + 2 * poolPadding - poolSize) / poolStride + 1;
    const int dilatedInDim = inDim + (inputDilation - 1) * (inDim - 1);
    for (int b = 0; b < batchSize; b += batches) {
      for (int porow = 0; porow < poolOutDim; porow += porows) {
        const int orow = porow * poolStride - poolPadding;
        for (int pocol = 0; pocol < poolOutDim; pocol += pocols) {
          const int ocol = pocol * poolStride - poolPadding;
          for (int poch = 0; poch < outChannels; poch += pochs) {
            for (int krow = 0; krow < kernelDim; krow += krows) {
              const int orow_floored = orow < 0 ? 0 : orow;

              int irow =
                  orow_floored * stride + krow * kernelDilation - padding;
              for (int kcol = 0; kcol < kernelDim; kcol += kcols) {
                const int ocol_floored = ocol < 0 ? 0 : ocol;
                int icol =
                    ocol_floored * stride + kcol * kernelDilation - padding;

                for (int kch = 0; kch < inChannels; kch += kchs) {
                  IntegerAttr offsetAttr =
                      rewriter.getI64IntegerAttr(((b * poolOutDim * poolOutDim +
                                                   porow * poolOutDim + pocol) *
                                                      outChannels +
                                                  poch) *
                                                 sizeof(elem_t));
                  Value offsetValue = rewriter.create<arith::ConstantOp>(
                      loc, rewriter.getI64Type(), offsetAttr);
                  Value out = rewriter.create<arith::AddIOp>(
                      tileConvOp.getLoc(), rewriter.getI64Type(), output,
                      offsetValue);
                  if (transOutput1203) {
                    offsetAttr = rewriter.getI64IntegerAttr(
                        ((porow * poolOutDim * batchSize + pocol * batchSize +
                          b) *
                             outChannels +
                         poch) *
                        sizeof(elem_t));
                    offsetValue = rewriter.create<arith::ConstantOp>(
                        loc,rewriter.getI64Type(), offsetAttr);
                    out = rewriter.create<arith::AddIOp>(tileConvOp.getLoc(),
                                                         rewriter.getI64Type(),
                                                         output, offsetValue);
                  }

                  if (krow + krows < kernelDim || kcol + kcols < kernelDim ||
                      kch + kchs < inChannels) {
                    IntegerAttr attr = rewriter.getI16IntegerAttr(0);
                    out = rewriter.create<arith::ConstantOp>(
                        tileConvOp.getLoc(),rewriter.getI64Type(), attr);
                  }
                  IntegerAttr pochAttr =
                      rewriter.getI64IntegerAttr(poch * sizeof(acc_t));
                  Value pochValue = rewriter.create<arith::ConstantOp>(
                      tileConvOp.getLoc(), rewriter.getI64Type(), pochAttr);
                  Value bias_ = rewriter.create<arith::AddIOp>(
                      tileConvOp.getLoc(), rewriter.getI64Type(), bias,
                      pochValue);
                  if (krow > 0 || kcol > 0 || kch > 0) {
                    IntegerAttr attr = rewriter.getI64IntegerAttr(0);
                    bias_ = rewriter.create<arith::ConstantOp>(
                        tileConvOp.getLoc(),rewriter.getI64Type(), attr);
                  }

                  const int batches_ =
                      batchSize - b > batches ? batches : batchSize - b;
                  const int porows_ =
                      poolOutDim - porow > porows ? porows : poolOutDim - porow;
                  const int pocols_ =
                      poolOutDim - pocol > pocols ? pocols : poolOutDim - pocol;
                  const int pochs_ =
                      outChannels - poch > pochs ? pochs : outChannels - poch;
                  const int krows_ =
                      kernelDim - krow > krows ? krows : kernelDim - krow;
                  const int kcols_ =
                      kernelDim - kcol > kcols ? kcols : kernelDim - kcol;
                  const int kchs_ =
                      inChannels - kch > kchs ? kchs : inChannels - kch;

                  const int ocols_ = pocols_ * poolStride + poolSize - 1;
                  const int orows_ = porows_ * poolStride + poolSize - 1;

                  const int plpad = ocol < 0 ? -ocol : 0;
                  const int prpad =
                      ocol + ocols_ > outDim ? ocol + ocols_ - outDim : 0;
                  const int pupad = orow < 0 ? -orow : 0;
                  const int pdpad =
                      orow + orows_ > outDim ? orow + orows_ - outDim : 0;

                  const int dilatedKrows_ =
                      krows_ + (kernelDilation - 1) * (krows_ - 1);
                  const int dilatedKcols_ =
                      kcols_ + (kernelDilation - 1) * (kcols_ - 1);

                  const int icols_ =
                      (ocols_ - plpad - prpad) * stride + dilatedKcols_ - 1;
                  const int irows_ =
                      (orows_ - pupad - pdpad) * stride + dilatedKrows_ - 1;

                  int lpad = icol < 0 ? -icol : 0;
                  int rpad = icol + icols_ > dilatedInDim
                                 ? icol + icols_ - dilatedInDim
                                 : 0;
                  int upad = irow < 0 ? -irow : 0;
                  int dpad = irow + irows_ > dilatedInDim
                                 ? irow + irows_ - dilatedInDim
                                 : 0;

                  if (inputDilated) {
                    lpad += lpad == 0 && icol % 2 != 0;
                    rpad += rpad == 0 && (icol + icols_) % 2 != 1;
                    upad += upad == 0 && irow % 2 != 0;
                    dpad += dpad == 0 && (irow + irows_) % 2 != 1;
                  }

                  int krow_ = krow;
                  int kcol_ = kcol;
                  if (wrot180) {
                    krow_ = kernelDim - krow - krows_;
                    kcol_ = kernelDim - kcol - kcols_;
                  }
                  offsetAttr = rewriter.getI64IntegerAttr(
                      ((krow_ * kernelDim * inChannels + kcol_ * inChannels +
                        kch) *
                           outChannels +
                       poch) *
                      sizeof(elem_t));
                  offsetValue = rewriter.create<arith::ConstantOp>(
                      tileConvOp.getLoc(),rewriter.getI64Type(), offsetAttr);
                  Value weightsSlice = rewriter.create<arith::AddIOp>(
                      tileConvOp.getLoc(), rewriter.getI64Type(), weights,
                      offsetValue);
                  if (transWeight1203) {
                    offsetAttr = rewriter.getI64IntegerAttr(
                        ((kch * kernelDim * kernelDim + krow_ * kernelDim +
                          kcol_) *
                             outChannels +
                         poch) *
                        sizeof(elem_t));
                    offsetValue = rewriter.create<arith::ConstantOp>(
                        tileConvOp.getLoc(),rewriter.getI64Type(), offsetAttr);
                    weightsSlice = rewriter.create<arith::AddIOp>(
                        tileConvOp.getLoc(), rewriter.getI64Type(), weights,
                        offsetValue);
                  } else if (transWeight0132) {
                    offsetAttr = rewriter.getI64IntegerAttr(
                        ((krow_ * kernelDim * outChannels +
                          kcol_ * outChannels + poch) *
                             inChannels +
                         kch) *
                        sizeof(elem_t));
                    offsetValue = rewriter.create<arith::ConstantOp>(
                        tileConvOp.getLoc(),rewriter.getI64Type(), offsetAttr);
                    weightsSlice = rewriter.create<arith::AddIOp>(
                        tileConvOp.getLoc(), rewriter.getI64Type(), weights,
                        offsetValue);
                  }
                  offsetAttr = rewriter.getI64IntegerAttr(
                      ((b * inDim * inDim +
                        ((irow + upad) >> inputDilated) * inDim +
                        ((icol + lpad) >> inputDilated)) *
                           inChannels +
                       kch) *
                      sizeof(elem_t));
                  offsetValue = rewriter.create<arith::ConstantOp>(
                      tileConvOp.getLoc(),rewriter.getI64Type(), offsetAttr);
                  Value in = rewriter.create<arith::AddIOp>(
                      tileConvOp.getLoc(), rewriter.getI64Type(), input,
                      offsetValue);
                  if (transInput3120) {
                    offsetAttr = rewriter.getI64IntegerAttr(
                        ((kch * inDim * inDim +
                          ((irow + upad) >> inputDilated) * inDim +
                          ((icol + lpad) >> inputDilated)) *
                             batchSize +
                         b) *
                        sizeof(elem_t));
                    in = rewriter.create<arith::AddIOp>(tileConvOp.getLoc(),
                                                        rewriter.getI64Type(),
                                                        input, offsetValue);
                  }

                  spTiledConv(batchSize, inDim, inChannels, outChannels, outDim,
                              poolOutDim, stride, padding, kernelDim,
                              kernelDilation, poolSize, poolStride, poolPadding,
                              batches_, porows_, pocols_, pochs_, krows_,
                              kcols_, kchs_, lpad, rpad, upad, dpad, plpad,
                              prpad, pupad, pdpad, in, weightsSlice, out, bias_,
                              act, scale, wrot180, transOutput1203,
                              transInput3120, transWeight1203, transWeight0132,
                              noBias, noPool, downsample, inputDilated, false,
                              tileConvOp, rewriter);
                }
              }
            }
          }
        }
      }
    }
    IntegerAttr flushAttr = rewriter.getI64IntegerAttr(0);
    Value flushValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(), flushAttr);
    rewriter.replaceOpWithNewOp<Flush_IntrOp>(tileConvOp, flushValue,
                                              flushValue);
  }

  int tiledConvTotalSpadRows(bool acc, int stride, int inputDilation,
                             int kernelDilation, bool downsample,
                             bool transWeight0132, bool transInput3120,
                             int batches, int porows, int pocols, int ochs,
                             int krows, int kcols, int kchs, int poolSize,
                             int poolStride) const {

    const int orows = porows * poolStride + poolSize - 1;
    const int ocols = pocols * poolStride + poolSize - 1;

    const int krowsDilated = krows + (kernelDilation - 1) * (krows - 1);
    const int kcolsDilated = kcols + (kernelDilation - 1) * (kcols - 1);

    int irows = orows * stride + krowsDilated - 1;
    int icols = ocols * stride + kcolsDilated - 1;
    const int ichs = kchs;

    irows = irows / inputDilation + (irows % inputDilation != 0);
    icols = icols / inputDilation + (icols % inputDilation != 0);

    const int inChannelsPerBank = ichs / DIM + (ichs % DIM != 0);
    const int outChannelsPerBank = ochs / DIM + (ochs % DIM != 0);
    const int batchesPerBank = batches / DIM + (batches % DIM != 0);

    const int aRows = transInput3120
                          ? (batchesPerBank * ichs * (irows >> downsample) *
                             (icols >> downsample))
                          : (inChannelsPerBank * batches *
                             (irows >> downsample) * (icols >> downsample));

    const int bRows = transWeight0132
                          ? inChannelsPerBank * kcols * krows * ochs
                          : outChannelsPerBank * kcols * krows * kchs;

    const int cRows = outChannelsPerBank * batches * orows * ocols;

    return acc ? cRows : aRows + bRows;
  }

public:
  using ConvertOpToLLVMPattern<TileConvOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(TileConvOp tileConvOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = tileConvOp.getInput();
    Value output = tileConvOp.getOutput();
    Value weights = tileConvOp.getWeights();
    Value bias = tileConvOp.getBias();
    MemRefType inputType = input.getType().dyn_cast<MemRefType>();
    MemRefType outputType = output.getType().dyn_cast<MemRefType>();
    MemRefType weightsType = weights.getType().dyn_cast<MemRefType>();
    MemRefType biasType = bias.getType().dyn_cast<MemRefType>();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    ArrayRef<int64_t> weightsShape = weightsType.getShape();
    ArrayRef<int64_t> biasShape = biasType.getShape();
    // inDim
    if (inputShape[1] != inputShape[2]) {
      llvm::outs() << "inDim error.\n";
      return failure();
    }
    // outChannels
    if (biasShape[0] != outputShape[1] || biasShape[0] != weightsShape[1]) {
      llvm::outs() << "outChannels error.\n";
      return failure();
    }
    Value outDimValue = tileConvOp.getOutDim();
    int outDim = getNumberFromValue(outDimValue);
    Value kernelDimValue = tileConvOp.getKernelDim();
    int kernelDim = getNumberFromValue(kernelDimValue);
    int batchSize = inputShape[0];
    int inDim = inputShape[1];
    int inChannels = inputShape[3];
    int outChannels = biasShape[0];
    int stride = tileConvOp.getStride();
    int inputDilation = tileConvOp.getInputDilation();
    int kernelDilation = tileConvOp.getKernelDilation();
    int padding = tileConvOp.getPadding();
    int act = tileConvOp.getAct();
    float scale = tileConvOp.getScale().convertToFloat();
    int poolSize = tileConvOp.getPoolSize();
    int poolStride = tileConvOp.getPoolStride();
    int poolPadding = tileConvOp.getPoolPadding();
    bool wrot180 = tileConvOp.getWrot180();
    bool transOutput1203 = tileConvOp.getTransOutput1203();
    bool transInput3120 = tileConvOp.getTransInput3120();
    bool transWeight1203 = tileConvOp.getTransWeight1203();
    bool transWeight0132 = tileConvOp.getTransWeight0132();
    Location loc = tileConvOp.getLoc();
    IntegerType i64Type = rewriter.getI64Type();
    Value inputExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, input);
    Value inputIndexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, inputExtractOp);
    Value outputExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, output);
    Value outputIndexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, outputExtractOp);
    Value biasExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, bias);
    Value biasIndexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, biasExtractOp);
    Value weightsExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, weights);
    Value weightsIndexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, weightsExtractOp);
    const bool noPool = poolSize == 0;
    if (noPool) {
      poolSize = 1;
      poolStride = 1;
      poolPadding = 0;
    }
    const int poolOutDim =
        (outDim + 2 * poolPadding - poolSize) / poolStride + 1;
    const bool downsample = stride == 2 && kernelDim == 1 && padding == 0 &&
                            noPool && inDim % 2 == 0;
    int args[] = {batchSize, poolOutDim, poolOutDim, outChannels,
                  kernelDim, kernelDim,  inChannels};
    const int maxArgs[] = {batchSize, poolOutDim, poolOutDim, outChannels,
                           kernelDim, kernelDim,  inChannels};
    const int orowsIdx = 1;
    const int ocolsIdx = 2;
    const int outChannelsIdx = 3;
    const int inChannelsIdx = 6;
    const int maxSpadRows = (BANK_NUM * BANK_ROWS / 2);
    const int maxAccRows = (ACC_ROWS / 2);
    int spadRows = tiledConvTotalSpadRows(
        false, stride, inputDilation, kernelDilation, downsample,
        transWeight0132, transInput3120, args[0], args[1], args[2], args[3],
        args[4], args[5], args[6], poolSize, poolStride);
    int accRows = tiledConvTotalSpadRows(
        true, stride, inputDilation, kernelDilation, downsample,
        transWeight0132, transInput3120, args[0], args[1], args[2], args[3],
        args[4], args[5], args[6], poolSize, poolStride);
    while (spadRows > maxSpadRows || accRows > maxAccRows) {
      int maxVal = -1;
      int maxIdx = -1;
      for (size_t i = 0; i < sizeof(args) / sizeof(args[0]); i++) {
        if (!(i == ocolsIdx && args[i] <= DIM && args[orowsIdx] > 1) &&
            args[i] > maxVal) {
          maxVal = args[i];
          maxIdx = i;
        }
      }

      if (maxIdx == outChannelsIdx || maxIdx == inChannelsIdx) {
        if (args[maxIdx] % DIM != 0) {
          args[maxIdx] = (args[maxIdx] / DIM) * DIM;
        } else {
          args[maxIdx] -= DIM;
        }
        args[maxIdx] = args[maxIdx] == 0 ? 1 : args[maxIdx];
      } else {
        args[maxIdx]--;
      }
      spadRows = tiledConvTotalSpadRows(
          false, stride, inputDilation, kernelDilation, downsample,
          transWeight0132, transInput3120, args[0], args[1], args[2], args[3],
          args[4], args[5], args[6], poolSize, poolStride);
      accRows = tiledConvTotalSpadRows(
          true, stride, inputDilation, kernelDilation, downsample,
          transWeight0132, transInput3120, args[0], args[1], args[2], args[3],
          args[4], args[5], args[6], poolSize, poolStride);
    }
    bool notIncreased = false;
    while (!notIncreased) {
      notIncreased = true;

      int argsCandidate[] = {args[0], args[1], args[2], args[3],
                             args[4], args[5], args[6]};
      argsCandidate[ocolsIdx]++;

      if (argsCandidate[ocolsIdx] > maxArgs[ocolsIdx])
        continue;

      spadRows = tiledConvTotalSpadRows(
          false, stride, inputDilation, kernelDilation, downsample,
          transWeight0132, transInput3120, argsCandidate[0], argsCandidate[1],
          argsCandidate[2], argsCandidate[3], argsCandidate[4],
          argsCandidate[5], argsCandidate[6], poolSize, poolStride);
      accRows = tiledConvTotalSpadRows(
          true, stride, inputDilation, kernelDilation, downsample,
          transWeight0132, transInput3120, argsCandidate[0], argsCandidate[1],
          argsCandidate[2], argsCandidate[3], argsCandidate[4],
          argsCandidate[5], argsCandidate[6], poolSize, poolStride);

      if (spadRows <= maxSpadRows && accRows <= maxAccRows) {
        args[ocolsIdx] = argsCandidate[ocolsIdx];
        notIncreased = false;
      }
    }

    bool nothingIncreased = false;
    while (!nothingIncreased) {
      nothingIncreased = true;
      for (size_t i = 0; i < sizeof(args) / sizeof(args[0]); i++) {
        int argsCandidate[] = {args[0], args[1], args[2], args[3],
                               args[4], args[5], args[6]};
        argsCandidate[i]++;

        if (argsCandidate[i] > maxArgs[i])
          continue;
        spadRows = tiledConvTotalSpadRows(
            false, stride, inputDilation, kernelDilation, downsample,
            transWeight0132, transInput3120, argsCandidate[0], argsCandidate[1],
            argsCandidate[2], argsCandidate[3], argsCandidate[4],
            argsCandidate[5], argsCandidate[6], poolSize, poolStride);
        accRows = tiledConvTotalSpadRows(
            true, stride, inputDilation, kernelDilation, downsample,
            transWeight0132, transInput3120, argsCandidate[0], argsCandidate[1],
            argsCandidate[2], argsCandidate[3], argsCandidate[4],
            argsCandidate[5], argsCandidate[6], poolSize, poolStride);

        if (spadRows <= maxSpadRows && accRows <= maxAccRows) {
          args[i] = argsCandidate[i];
          nothingIncreased = false;
        }
      }
    }
    const int batches = args[0];
    const int orows = args[1];
    const int ocols = args[2];
    const int ochs = args[3];
    const int krows = args[4];
    const int kcols = args[5];
    const int kchs = args[6];
    tiledConv(batchSize, inDim, inChannels, outChannels, outDim, stride,
              inputDilation, kernelDilation, padding, kernelDim, wrot180,
              transOutput1203, transInput3120, transWeight1203, transWeight0132,
              batches, orows, ocols, ochs, krows, kcols, kchs, inputIndexCastOp,
              weightsIndexCastOp, biasIndexCastOp, outputIndexCastOp, act,
              scale, poolSize, noPool ? 0 : poolStride, poolPadding, tileConvOp,
              rewriter);
    return success();
  }
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
  patterns.add<GemminiTileConvOpLowering>(converter);
}

void mlir::configureGemminiegalizeForExportTarget(
    LLVMConversionTarget &target) {
  target.addLegalOp<
      Flush_IntrOp, ConfigSt_IntrOp, ConifgLd_IntrOp, ConfigEX_IntrOp,
      Mvin_IntrOp, Mvout_IntrOp, Preload_IntrOp, ComputePreloaded_IntrOp,
      ComputeAccumulated_IntrOp, LoopWsConfigBounds_IntrOp,
      LoopWsConfigAddrsAB_IntrOp, LoopWsConfigAddrsDC_IntrOp,
      LoopWsConfigStridesAB_IntrOp, LoopWsConfigStridesDC_IntrOp, LoopWs_IntrOp,
      LoopConvWsConfig1_IntrOp, LoopConvWsConfig2_IntrOp,
      LoopConvWsConfig3_IntrOp, LoopConvWsConfig4_IntrOp,
      LoopConvWsConfig5_IntrOp, LoopConvWsConfig6_IntrOp, LoopConvWs_IntrOp>();
  target.addIllegalOp<FlushOp, ConfigStOp, ConfigLdOp, ConfigExOp, MvinOp,
                      MvoutOp, PrintOp, PreloadZerosOp, PreloadOp,
                      ComputePreloadedOp, ComputeAccumulatedOp, TileMatMulOp,
                      TileConvOp>();
}
