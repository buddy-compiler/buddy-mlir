//===- LegalizeForLLVMExport.cpp - Prepare Buckyball for LLVM translation ---===//
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
// Lowers Buckyball dialect ops to intrinsic ops using bank-based ISA encoding.
// Encoding follows bb-tests/workloads/lib/bbhw/isa/*.c macros.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"
#include "Buckyball/Transform.h"

using namespace mlir;
using namespace buddy::buckyball;

//===----------------------------------------------------------------------===//
// ISA encoding helpers — mirrors bb-tests/workloads/lib/bbhw/isa/isa.h
//===----------------------------------------------------------------------===//

static uint64_t field(uint64_t val, int startBit, int endBit) {
  uint64_t mask = (1ULL << (endBit - startBit + 1)) - 1;
  return (val & mask) << startBit;
}

static constexpr uint64_t BB_RD0 = 1ULL << 24;
static constexpr uint64_t BB_RD1 = 1ULL << 25;
static constexpr uint64_t BB_WR  = 1ULL << 26;

static uint64_t bbBank0(uint64_t id) { return field(id, 0, 7); }
static uint64_t bbBank1(uint64_t id) { return field(id, 8, 15); }
static uint64_t bbBank2(uint64_t id) { return field(id, 16, 23); }

static Value cst(ConversionPatternRewriter &rewriter, Location loc,
                 uint64_t v) {
  return rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(v));
}

static Value extractPtr(ConversionPatternRewriter &rewriter, Location loc,
                        Value memref) {
  Value idx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
      loc, rewriter.getIndexType(), memref);
  return rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), idx);
}

static Value dimAsI64(ConversionPatternRewriter &rewriter, Location loc,
                      Value memref, unsigned dim) {
  Value d = rewriter.create<memref::DimOp>(loc, memref, dim);
  return rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), d);
}

//===----------------------------------------------------------------------===//
// ForwardOperands / ReturnOpTypeConversion
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Fence
//===----------------------------------------------------------------------===//

struct BuckyballFenceLowering : public ConvertOpToLLVMPattern<FenceOp> {
  using ConvertOpToLLVMPattern<FenceOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(FenceOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<Fence_IntrOp>(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Mvin — bb_mvin(mem_addr, bank_id, depth, stride)
//   rs1 = BB_BANK0(bank_id) | BB_WR | FIELD(mem_addr, 27, 58)
//   rs2 = FIELD(depth, 0, 9) | FIELD(stride, 10, 28)
//===----------------------------------------------------------------------===//

struct BuckyballMvinLowering : public ConvertOpToLLVMPattern<MvinOp> {
  using ConvertOpToLLVMPattern<MvinOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MvinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value addr = adaptor.getAddr();
    Value stride = adaptor.getStride();

    Value memAddr = extractPtr(rewriter, loc, input);
    Value depth = dimAsI64(rewriter, loc, input, 0);

    // rs1 = BB_BANK0(addr) | BB_WR | FIELD(memAddr, 27, 58)
    Value rs1 = rewriter.create<arith::ShLIOp>(loc, memAddr, cst(rewriter, loc, 27));
    rs1 = rewriter.create<arith::OrIOp>(loc, rs1, cst(rewriter, loc, BB_WR));
    rs1 = rewriter.create<arith::OrIOp>(loc, rs1, addr);

    // rs2 = FIELD(depth, 0, 9) | FIELD(stride, 10, 28)
    Value strideShl = rewriter.create<arith::ShLIOp>(loc, stride, cst(rewriter, loc, 10));
    Value rs2 = rewriter.create<arith::OrIOp>(loc, depth, strideShl);

    rewriter.replaceOpWithNewOp<Mvin_IntrOp>(op, rs1, rs2);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Mvout — bb_mvout(mem_addr, bank_id, depth, stride)
//   rs1 = BB_BANK0(bank_id) | BB_RD0 | FIELD(mem_addr, 27, 58)
//   rs2 = FIELD(depth, 0, 9) | FIELD(stride, 10, 28)
//===----------------------------------------------------------------------===//

struct BuckyballMvoutLowering : public ConvertOpToLLVMPattern<MvoutOp> {
  using ConvertOpToLLVMPattern<MvoutOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MvoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value output = op.getOutput();
    Value addr = adaptor.getAddr();

    Value memAddr = extractPtr(rewriter, loc, output);
    Value depth = dimAsI64(rewriter, loc, output, 0);

    // rs1 = BB_BANK0(addr) | BB_RD0 | FIELD(memAddr, 27, 58)
    Value rs1 = rewriter.create<arith::ShLIOp>(loc, memAddr, cst(rewriter, loc, 27));
    rs1 = rewriter.create<arith::OrIOp>(loc, rs1, cst(rewriter, loc, BB_RD0));
    rs1 = rewriter.create<arith::OrIOp>(loc, rs1, addr);

    // rs2 = FIELD(depth, 0, 9) | FIELD(stride=1, 10, 28)
    Value stride1 = cst(rewriter, loc, 1ULL << 10);
    Value rs2 = rewriter.create<arith::OrIOp>(loc, depth, stride1);

    rewriter.replaceOpWithNewOp<Mvout_IntrOp>(op, rs1, rs2);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MatMul — mvin_A + mvin_B + mul_warp16 + mvout_C
//===----------------------------------------------------------------------===//

struct BuckyballMatMulLowering : public ConvertOpToLLVMPattern<MatMulOp> {
  using ConvertOpToLLVMPattern<MatMulOp>::ConvertOpToLLVMPattern;
  explicit BuckyballMatMulLowering(LLVMTypeConverter &tc,
                                   int64_t lane, int64_t warp, int64_t bankDepth)
      : ConvertOpToLLVMPattern(tc), lane(lane), warp(warp), bankDepth(bankDepth) {}

  LogicalResult
  matchAndRewrite(MatMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value aMemArray = op.getAMemArray();
    Value bMemArray = op.getBMemArray();
    Value cMemArray = op.getCMemArray();

    auto aType = cast<MemRefType>(aMemArray.getType());
    uint64_t M = aType.getShape()[0];
    uint64_t K = aType.getShape()[1];

    const uint64_t aBankId = 0, bBankId = 1, cBankId = 2;

    // --- Mvin A ---
    Value aPtr = extractPtr(rewriter, loc, aMemArray);
    uint64_t aRs1Const = bbBank0(aBankId) | BB_WR;
    Value rs1A = rewriter.create<arith::OrIOp>(loc,
        rewriter.create<arith::ShLIOp>(loc, aPtr, cst(rewriter, loc, 27)),
        cst(rewriter, loc, aRs1Const));
    uint64_t aRs2 = field(M, 0, 9) | field(1, 10, 28);
    rewriter.create<Mvin_IntrOp>(loc, rs1A, cst(rewriter, loc, aRs2));

    // --- Mvin B ---
    Value bPtr = extractPtr(rewriter, loc, bMemArray);
    uint64_t bRs1Const = bbBank0(bBankId) | BB_WR;
    Value rs1B = rewriter.create<arith::OrIOp>(loc,
        rewriter.create<arith::ShLIOp>(loc, bPtr, cst(rewriter, loc, 27)),
        cst(rewriter, loc, bRs1Const));
    uint64_t bRs2 = field(K, 0, 9) | field(1, 10, 28);
    rewriter.create<Mvin_IntrOp>(loc, rs1B, cst(rewriter, loc, bRs2));

    // --- Compute: mul_warp16 ---
    uint64_t iter = (M + lane - 1) / lane;
    uint64_t mulRs1 = bbBank0(aBankId) | bbBank1(bBankId) | bbBank2(cBankId)
                    | BB_RD0 | BB_RD1 | BB_WR;
    uint64_t mulRs2 = field(iter, 0, 9);
    rewriter.create<Mul_Warp16_IntrOp>(loc, cst(rewriter, loc, mulRs1),
                                       cst(rewriter, loc, mulRs2));

    // --- Mvout C ---
    Value cPtr = extractPtr(rewriter, loc, cMemArray);
    uint64_t cRs1Const = bbBank0(cBankId) | BB_RD0;
    Value rs1C = rewriter.create<arith::OrIOp>(loc,
        rewriter.create<arith::ShLIOp>(loc, cPtr, cst(rewriter, loc, 27)),
        cst(rewriter, loc, cRs1Const));
    uint64_t cRs2 = field(M, 0, 9) | field(1, 10, 28);
    rewriter.create<Mvout_IntrOp>(loc, rs1C, cst(rewriter, loc, cRs2));

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t lane, warp, bankDepth;
};

//===----------------------------------------------------------------------===//
// Transpose — mvin + transpose_intr + mvout
//===----------------------------------------------------------------------===//

struct BuckyballTransposeLowering : public ConvertOpToLLVMPattern<TransposeOp> {
  using ConvertOpToLLVMPattern<TransposeOp>::ConvertOpToLLVMPattern;
  explicit BuckyballTransposeLowering(LLVMTypeConverter &tc,
                                     int64_t lane, int64_t bankDepth)
      : ConvertOpToLLVMPattern(tc), lane(lane), bankDepth(bankDepth) {}

  LogicalResult
  matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();

    auto inType = cast<MemRefType>(input.getType());
    uint64_t rows = inType.getShape()[0];
    uint64_t cols = inType.getShape()[1];

    const uint64_t inBankId = 0, outBankId = 1;

    // --- Mvin input ---
    Value inPtr = extractPtr(rewriter, loc, input);
    uint64_t inRs1Const = bbBank0(inBankId) | BB_WR;
    Value rs1In = rewriter.create<arith::OrIOp>(loc,
        rewriter.create<arith::ShLIOp>(loc, inPtr, cst(rewriter, loc, 27)),
        cst(rewriter, loc, inRs1Const));
    uint64_t inRs2 = field(rows, 0, 9) | field(1, 10, 28);
    rewriter.create<Mvin_IntrOp>(loc, rs1In, cst(rewriter, loc, inRs2));

    // --- Transpose ---
    uint64_t iter = (rows + lane - 1) / lane;
    uint64_t tRs1 = bbBank0(inBankId) | bbBank2(outBankId) | BB_RD0 | BB_WR;
    uint64_t tRs2 = field(iter, 0, 9);
    rewriter.create<Transpose_IntrOp>(loc, cst(rewriter, loc, tRs1),
                                      cst(rewriter, loc, tRs2));

    // --- Mvout output ---
    Value outPtr = extractPtr(rewriter, loc, output);
    uint64_t outRs1Const = bbBank0(outBankId) | BB_RD0;
    Value rs1Out = rewriter.create<arith::OrIOp>(loc,
        rewriter.create<arith::ShLIOp>(loc, outPtr, cst(rewriter, loc, 27)),
        cst(rewriter, loc, outRs1Const));
    uint64_t outRs2 = field(cols, 0, 9) | field(1, 10, 28);
    rewriter.create<Mvout_IntrOp>(loc, rs1Out, cst(rewriter, loc, outRs2));

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t lane, bankDepth;
};

//===----------------------------------------------------------------------===//
// Im2col — mvin + im2col_intr + mvout
//===----------------------------------------------------------------------===//

struct BuckyballIm2colLowering : public ConvertOpToLLVMPattern<Im2colOp> {
  using ConvertOpToLLVMPattern<Im2colOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(Im2colOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();

    auto inType = cast<MemRefType>(input.getType());
    uint64_t inRows = inType.getShape()[0];

    const uint64_t inBankId = 0, outBankId = 1;

    // --- Mvin input ---
    Value inPtr = extractPtr(rewriter, loc, input);
    uint64_t inRs1Const = bbBank0(inBankId) | BB_WR;
    Value rs1In = rewriter.create<arith::OrIOp>(loc,
        rewriter.create<arith::ShLIOp>(loc, inPtr, cst(rewriter, loc, 27)),
        cst(rewriter, loc, inRs1Const));
    uint64_t inRs2 = field(inRows, 0, 9) | field(1, 10, 28);
    rewriter.create<Mvin_IntrOp>(loc, rs1In, cst(rewriter, loc, inRs2));

    // --- Im2col intrinsic ---
    uint64_t im2colRs1 = bbBank0(inBankId) | bbBank2(outBankId) | BB_RD0 | BB_WR;

    // Build rs2 from parameters
    Value kCol = adaptor.getKCol();
    Value kRow = adaptor.getKRow();
    Value inCol = adaptor.getInCol();
    Value inRow = adaptor.getInRow();
    Value startCol = adaptor.getStartCol();
    Value startRow = adaptor.getStartRow();

    // rs2 = FIELD(kcol,0,3) | FIELD(krow,4,7) | FIELD(incol,8,12)
    //      | FIELD(inrow,13,22) | FIELD(startcol,23,27) | FIELD(startrow,28,37)
    Value rs2 = kCol;
    rs2 = rewriter.create<arith::OrIOp>(loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, kRow, cst(rewriter, loc, 4)));
    rs2 = rewriter.create<arith::OrIOp>(loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, inCol, cst(rewriter, loc, 8)));
    rs2 = rewriter.create<arith::OrIOp>(loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, inRow, cst(rewriter, loc, 13)));
    rs2 = rewriter.create<arith::OrIOp>(loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, startCol, cst(rewriter, loc, 23)));
    rs2 = rewriter.create<arith::OrIOp>(loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, startRow, cst(rewriter, loc, 28)));

    rewriter.create<Im2col_IntrOp>(loc, cst(rewriter, loc, im2colRs1), rs2);

    // --- Mvout output ---
    Value outPtr = extractPtr(rewriter, loc, output);
    auto outType = cast<MemRefType>(output.getType());
    uint64_t outRows = outType.getShape()[0];
    uint64_t outRs1Const = bbBank0(outBankId) | BB_RD0;
    Value rs1Out = rewriter.create<arith::OrIOp>(loc,
        rewriter.create<arith::ShLIOp>(loc, outPtr, cst(rewriter, loc, 27)),
        cst(rewriter, loc, outRs1Const));
    uint64_t outRs2 = field(outRows, 0, 9) | field(1, 10, 28);
    rewriter.create<Mvout_IntrOp>(loc, rs1Out, cst(rewriter, loc, outRs2));

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Quant — mvin + quant_intr + mvout
//===----------------------------------------------------------------------===//

struct BuckyballQuantLowering : public ConvertOpToLLVMPattern<QuantOp> {
  using ConvertOpToLLVMPattern<QuantOp>::ConvertOpToLLVMPattern;
  explicit BuckyballQuantLowering(LLVMTypeConverter &tc, int64_t lane)
      : ConvertOpToLLVMPattern(tc), lane(lane) {}

  LogicalResult
  matchAndRewrite(QuantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();
    IntegerType i64 = rewriter.getI64Type();

    auto inType = cast<MemRefType>(input.getType());
    uint64_t rows = inType.getShape()[0];

    const uint64_t inBankId = 0, outBankId = 1;

    // --- Mvin input ---
    Value inPtr = extractPtr(rewriter, loc, input);
    uint64_t inRs1Const = bbBank0(inBankId) | BB_WR;
    Value rs1In = rewriter.create<arith::OrIOp>(loc,
        rewriter.create<arith::ShLIOp>(loc, inPtr, cst(rewriter, loc, 27)),
        cst(rewriter, loc, inRs1Const));
    uint64_t inRs2 = field(rows, 0, 9) | field(1, 10, 28);
    rewriter.create<Mvin_IntrOp>(loc, rs1In, cst(rewriter, loc, inRs2));

    // --- Quant intrinsic ---
    uint64_t qRs1 = bbBank0(inBankId) | bbBank2(outBankId) | BB_RD0 | BB_WR;
    uint64_t iter = (rows + lane - 1) / lane;

    Value scale = adaptor.getScale();
    Value scaleBits = rewriter.create<arith::BitcastOp>(
        loc, rewriter.getI32Type(), scale);
    Value scaleBits64 = rewriter.create<arith::ExtUIOp>(loc, i64, scaleBits);
    Value scaleShl = rewriter.create<arith::ShLIOp>(loc, scaleBits64,
                                                    cst(rewriter, loc, 10));
    Value rs2 = rewriter.create<arith::OrIOp>(loc,
        cst(rewriter, loc, field(iter, 0, 9)), scaleShl);

    rewriter.create<Quant_IntrOp>(loc, cst(rewriter, loc, qRs1), rs2);

    // --- Mvout output ---
    Value outPtr = extractPtr(rewriter, loc, output);
    auto outType = cast<MemRefType>(output.getType());
    uint64_t outRows = outType.getShape()[0];
    uint64_t outRs1Const = bbBank0(outBankId) | BB_RD0;
    Value rs1Out = rewriter.create<arith::OrIOp>(loc,
        rewriter.create<arith::ShLIOp>(loc, outPtr, cst(rewriter, loc, 27)),
        cst(rewriter, loc, outRs1Const));
    uint64_t outRs2 = field(outRows, 0, 9) | field(1, 10, 28);
    rewriter.create<Mvout_IntrOp>(loc, rs1Out, cst(rewriter, loc, outRs2));

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t lane;
};

//===----------------------------------------------------------------------===//
// Dequant — same encoding as Quant but different funct7
//===----------------------------------------------------------------------===//

struct BuckyballDequantLowering : public ConvertOpToLLVMPattern<DequantOp> {
  using ConvertOpToLLVMPattern<DequantOp>::ConvertOpToLLVMPattern;
  explicit BuckyballDequantLowering(LLVMTypeConverter &tc, int64_t lane)
      : ConvertOpToLLVMPattern(tc), lane(lane) {}

  LogicalResult
  matchAndRewrite(DequantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();
    IntegerType i64 = rewriter.getI64Type();

    auto inType = cast<MemRefType>(input.getType());
    uint64_t rows = inType.getShape()[0];

    const uint64_t inBankId = 0, outBankId = 1;

    // --- Mvin input ---
    Value inPtr = extractPtr(rewriter, loc, input);
    uint64_t inRs1Const = bbBank0(inBankId) | BB_WR;
    Value rs1In = rewriter.create<arith::OrIOp>(loc,
        rewriter.create<arith::ShLIOp>(loc, inPtr, cst(rewriter, loc, 27)),
        cst(rewriter, loc, inRs1Const));
    uint64_t inRs2 = field(rows, 0, 9) | field(1, 10, 28);
    rewriter.create<Mvin_IntrOp>(loc, rs1In, cst(rewriter, loc, inRs2));

    // --- Dequant intrinsic ---
    uint64_t dRs1 = bbBank0(inBankId) | bbBank2(outBankId) | BB_RD0 | BB_WR;
    uint64_t iter = (rows + lane - 1) / lane;

    Value scale = adaptor.getScale();
    Value scaleBits = rewriter.create<arith::BitcastOp>(
        loc, rewriter.getI32Type(), scale);
    Value scaleBits64 = rewriter.create<arith::ExtUIOp>(loc, i64, scaleBits);
    Value scaleShl = rewriter.create<arith::ShLIOp>(loc, scaleBits64,
                                                    cst(rewriter, loc, 10));
    Value rs2 = rewriter.create<arith::OrIOp>(loc,
        cst(rewriter, loc, field(iter, 0, 9)), scaleShl);

    rewriter.create<Dequant_IntrOp>(loc, cst(rewriter, loc, dRs1), rs2);

    // --- Mvout output ---
    Value outPtr = extractPtr(rewriter, loc, output);
    auto outType = cast<MemRefType>(output.getType());
    uint64_t outRows = outType.getShape()[0];
    uint64_t outRs1Const = bbBank0(outBankId) | BB_RD0;
    Value rs1Out = rewriter.create<arith::OrIOp>(loc,
        rewriter.create<arith::ShLIOp>(loc, outPtr, cst(rewriter, loc, 27)),
        cst(rewriter, loc, outRs1Const));
    uint64_t outRs2 = field(outRows, 0, 9) | field(1, 10, 28);
    rewriter.create<Mvout_IntrOp>(loc, rs1Out, cst(rewriter, loc, outRs2));

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t lane;
};

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void mlir::populateBuckyballLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    int64_t lane, int64_t warp, int64_t bankDepth, int64_t bankNum) {
  patterns
      .add<ForwardOperands<func::CallOp>, ForwardOperands<func::CallIndirectOp>,
           ForwardOperands<func::ReturnOp>>(converter, &converter.getContext());
  patterns.add<BuckyballFenceLowering>(converter);
  patterns.add<BuckyballMvinLowering>(converter);
  patterns.add<BuckyballMvoutLowering>(converter);
  patterns.add<BuckyballMatMulLowering>(converter, lane, warp, bankDepth);
  patterns.add<BuckyballTransposeLowering>(converter, lane, bankDepth);
  patterns.add<BuckyballIm2colLowering>(converter);
  patterns.add<BuckyballQuantLowering>(converter, lane);
  patterns.add<BuckyballDequantLowering>(converter, lane);
}

void mlir::configureBuckyballLegalizeForExportTarget(
    LLVMConversionTarget &target) {
  target.addLegalOp<Fence_IntrOp, Mvin_IntrOp, Mvout_IntrOp,
                    Mul_Warp16_IntrOp, Transpose_IntrOp, Im2col_IntrOp,
                    Quant_IntrOp, Dequant_IntrOp, Relu_IntrOp>();
  target.addIllegalOp<FenceOp, MvinOp, MvoutOp, MatMulOp, TransposeOp,
                      Im2colOp, QuantOp, DequantOp>();
  target.addLegalDialect<memref::MemRefDialect>();
  target.addLegalDialect<arith::ArithDialect>();
}
