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
// Lowers Buckyball dialect ops to RISC-V custom intrinsics. Bit layouts match
// bb-tests/workloads/lib/bbhw/isa/isa.h and the per-instruction *.c wrappers
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/ErrorHandling.h"

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"
#include "Buckyball/Transform.h"

using namespace mlir;
using namespace buddy::buckyball;

namespace {

//===----------------------------------------------------------------------===//
// isa.h FIELD(val, start_bit, end_bit)
//===----------------------------------------------------------------------===//

static uint64_t fieldBits(uint64_t val, int startBit, int endBit) {
  uint64_t width = endBit - startBit + 1;
  uint64_t mask = (1ULL << width) - 1;
  return (val & mask) << startBit;
}

//===----------------------------------------------------------------------===//
// MemRef / arith helpers
//===----------------------------------------------------------------------===//

static Value cstI64(OpBuilder &b, Location loc, uint64_t v) {
  return b.create<arith::ConstantOp>(loc, b.getI64Type(),
                                     b.getI64IntegerAttr(v));
}

static int64_t elemByteSize(Type el) {
  if (auto it = dyn_cast<IntegerType>(el))
    return it.getWidth() / 8;
  if (auto ft = dyn_cast<FloatType>(el))
    return ft.getWidth() / 8;
  return -1;
}

/// Base aligned pointer + linear byte offset for strided subviews (ExtractAlignedPointer
/// alone misses StridedLayoutAttr offset).
static Value extractPtr(OpBuilder &b, Location loc, Value memref) {
  auto ty = cast<MemRefType>(memref.getType());
  Value idx = b.create<memref::ExtractAlignedPointerAsIndexOp>(
      loc, b.getIndexType(), memref);
  SmallVector<int64_t, 4> strides;
  int64_t offElems = 0;
  if (failed(ty.getStridesAndOffset(strides, offElems)))
    return b.create<arith::IndexCastOp>(loc, b.getI64Type(), idx);
  if (ShapedType::isDynamic(offElems))
    llvm_unreachable(
        "bb memref intrinsic: dynamic linear offset not supported yet");
  if (offElems == 0)
    return b.create<arith::IndexCastOp>(loc, b.getI64Type(), idx);
  int64_t eb = elemByteSize(ty.getElementType());
  if (eb <= 0)
    llvm_unreachable("bb memref intrinsic: unsupported element type for ptr offset");
  int64_t offBytes = offElems * eb;
  Value add = b.create<arith::ConstantIndexOp>(loc, offBytes);
  idx = b.create<arith::AddIOp>(loc, idx, add);
  return b.create<arith::IndexCastOp>(loc, b.getI64Type(), idx);
}

/// rs1 = BB_BANK0 | BB_BANK1 | BB_BANK2 | BB_ITER — isa.h
/// BB_BANK0/BB_BANK1: read bank ids (rbank); BB_BANK2: write bank id (wbank).
/// Each id masked to 10 bits; iter to 34 bits (bits [63:30]).
static Value packRs1BanksIter(OpBuilder &b, Location loc, Value rBank0,
                              Value rBank1, Value wBank, Value iter) {
  Value rBank0Field =
      b.create<arith::AndIOp>(loc, rBank0, cstI64(b, loc, 0x3FF));
  Value rBank1Field = b.create<arith::ShLIOp>(
      loc,
      b.create<arith::AndIOp>(loc, rBank1, cstI64(b, loc, 0x3FF)),
      cstI64(b, loc, 10));
  Value wBankField = b.create<arith::ShLIOp>(
      loc,
      b.create<arith::AndIOp>(loc, wBank, cstI64(b, loc, 0x3FF)),
      cstI64(b, loc, 20));
  Value iterField = b.create<arith::ShLIOp>(
      loc,
      b.create<arith::AndIOp>(loc, iter, cstI64(b, loc, (1ULL << 34) - 1)),
      cstI64(b, loc, 30));
  Value rs1Part01 =
      b.create<arith::OrIOp>(loc, rBank0Field, rBank1Field);
  Value rs1Part012 =
      b.create<arith::OrIOp>(loc, rs1Part01, wBankField);
  return b.create<arith::OrIOp>(loc, rs1Part012, iterField);
}

/// mvin/mvout: only BB_BANK0 (rbank); BB_BANK1/BB_BANK2 unset (`33_mvin.c`, `16_mvout.c`).
static Value packRs1BankIter(OpBuilder &b, Location loc, Value bankId,
                             Value depth) {
  Value z = cstI64(b, loc, 0);
  return packRs1BanksIter(b, loc, bankId, z, z, depth);
}

/// rs2 = FIELD(mem, 0, 38) | FIELD(stride, 39, 57)
static Value packRs2MemStride(OpBuilder &b, Location loc, Value memAddr,
                              Value stride) {
  Value mem = b.create<arith::AndIOp>(loc, memAddr, cstI64(b, loc, (1ULL << 39) - 1));
  Value s = b.create<arith::AndIOp>(loc, stride, cstI64(b, loc, (1ULL << 19) - 1));
  Value sHi = b.create<arith::ShLIOp>(loc, s, cstI64(b, loc, 39));
  return b.create<arith::OrIOp>(loc, mem, sHi);
}

/// bb_mset(bank_id, alloc, row, col) — rs1 = BB_BANK0 only (32_mset.c)
static void emitMset(OpBuilder &b, Location loc, uint64_t bankId, uint64_t row,
                     uint64_t col, uint64_t alloc) {
  uint64_t rs1 = fieldBits(bankId, 0, 9);
  uint64_t rs2 =
      fieldBits(row, 0, 4) | fieldBits(col, 5, 9) | fieldBits(alloc, 10, 10);
  b.create<Mset_IntrOp>(loc, cstI64(b, loc, rs1), cstI64(b, loc, rs2));
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
// Fence — lowers to int_riscv_bb_fence / BB_FENCE when +buddyext-bb.
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
// Mset
//===----------------------------------------------------------------------===//

struct BuckyballMsetLowering : public ConvertOpToLLVMPattern<MsetOp> {
  using ConvertOpToLLVMPattern<MsetOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value bankId = adaptor.getBankId();
    Value rs1 = rewriter.create<arith::AndIOp>(loc, bankId, cstI64(rewriter, loc, 0x3FF));
    uint64_t allocBit = op.getAlloc() ? 1u : 0u;
    uint64_t rowVal =
        op.getAlloc() ? static_cast<uint64_t>(op.getRow()) : 0u;
    uint64_t colVal =
        op.getAlloc() ? static_cast<uint64_t>(op.getCol()) : 0u;
    uint64_t rs2Val = fieldBits(rowVal, 0, 4) | fieldBits(colVal, 5, 9) |
                      fieldBits(allocBit, 10, 10);
    rewriter.replaceOpWithNewOp<Mset_IntrOp>(
        op, rs1, cstI64(rewriter, loc, rs2Val));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Mvin / Mvout — 33_mvin.c, 16_mvout.c
//===----------------------------------------------------------------------===//

struct BuckyballMvinLowering : public ConvertOpToLLVMPattern<MvinOp> {
  using ConvertOpToLLVMPattern<MvinOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MvinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value bankId = adaptor.getAddr();
    Value stride = adaptor.getStride();

    Value memAddr = extractPtr(rewriter, loc, input);
    Value depth = adaptor.getDepth();

    Value rs1 = packRs1BankIter(rewriter, loc, bankId, depth);
    Value rs2 = packRs2MemStride(rewriter, loc, memAddr, stride);

    rewriter.replaceOpWithNewOp<Mvin_IntrOp>(op, rs1, rs2);
    return success();
  }
};

struct BuckyballMvoutLowering : public ConvertOpToLLVMPattern<MvoutOp> {
  using ConvertOpToLLVMPattern<MvoutOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MvoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value output = op.getOutput();
    Value bankId = adaptor.getAddr();

    Value memAddr = extractPtr(rewriter, loc, output);
    Value depth = adaptor.getDepth();
    Value stride = adaptor.getStride();

    Value rs1 = packRs1BankIter(rewriter, loc, bankId, depth);
    Value rs2 = packRs2MemStride(rewriter, loc, memAddr, stride);

    rewriter.replaceOpWithNewOp<Mvout_IntrOp>(op, rs1, rs2);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MatMul — mvin A, mvin B, mul_warp16 (iter = K), mvout C
// Shapes A[M,K], B[K,N], C[M,N]; K and N must be multiples of 16 (i8 cols=1 line size).
//===----------------------------------------------------------------------===//

struct BuckyballMatMulLowering : public ConvertOpToLLVMPattern<MatMulOp> {
  using ConvertOpToLLVMPattern<MatMulOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MatMulOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value aMem = op.getAMemArray();
    Value bMem = op.getBMemArray();
    Value cMem = op.getCMemArray();

    auto aTy = cast<MemRefType>(aMem.getType());
    auto bTy = cast<MemRefType>(bMem.getType());
    auto cTy = cast<MemRefType>(cMem.getType());

    if (!aTy.hasStaticShape() || !bTy.hasStaticShape() || !cTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "bb_matmul requires static memref shapes");

    uint64_t M = aTy.getShape()[0];
    uint64_t K = aTy.getShape()[1];
    uint64_t Kb = bTy.getShape()[0];
    uint64_t N = bTy.getShape()[1];
    if (K != Kb)
      return rewriter.notifyMatchFailure(op, "inner dimensions must match");

    if (K % 16 != 0 || N % 16 != 0)
      return rewriter.notifyMatchFailure(
          op, "K and N must be multiples of 16 for this lowering");

    const uint64_t aBank = 0, bBank = 1, cBank = 2;
    uint64_t depthA = M * (K / 16);
    uint64_t depthB = K * (N / 16);
    uint64_t depthC = M * (N / 16);

    emitMset(rewriter, loc, aBank, 1, 1, 1);
    emitMset(rewriter, loc, bBank, 1, 1, 1);
    emitMset(rewriter, loc, cBank, 1, 4, 1);

    Value aPtr = extractPtr(rewriter, loc, aMem);
    Value bPtr = extractPtr(rewriter, loc, bMem);
    Value cPtr = extractPtr(rewriter, loc, cMem);
    // Row-major tile A[M,K] is dense: 16-byte mvin lines scan contiguous storage (stride=1).
    Value strideA = cstI64(rewriter, loc, 1);
    // B/C may be subviews: row pitch in memory uses parent leading dim, not tile width.
    // bebop mvin/mvout: byte step = 16*rs2_stride*line_blocks -> rs2_stride = rowElem/16.
    SmallVector<int64_t, 4> bStrides, cStrides;
    int64_t bOff = 0, cOff = 0;
    if (failed(bTy.getStridesAndOffset(bStrides, bOff)) || bStrides.size() < 2)
      return rewriter.notifyMatchFailure(
          op, "bb_matmul B memref needs static strides (use subview with strided layout)");
    if (failed(cTy.getStridesAndOffset(cStrides, cOff)) || cStrides.size() < 2)
      return rewriter.notifyMatchFailure(
          op, "bb_matmul C memref needs static strides (use subview with strided layout)");
    if (ShapedType::isDynamic(bStrides[0]) ||
        ShapedType::isDynamic(cStrides[0]))
      return rewriter.notifyMatchFailure(op, "bb_matmul: static row stride required");
    if (bStrides[0] % 16 != 0 || cStrides[0] % 16 != 0)
      return rewriter.notifyMatchFailure(
          op, "bb_matmul: row stride (elements) must be divisible by 16");

    Value strideBN =
        cstI64(rewriter, loc, (uint64_t)bStrides[0] / 16);
    Value strideCN =
        cstI64(rewriter, loc, (uint64_t)cStrides[0] / 16);

    Value rs1A = packRs1BankIter(rewriter, loc, cstI64(rewriter, loc, aBank),
                                 cstI64(rewriter, loc, depthA));
    Value rs2A = packRs2MemStride(rewriter, loc, aPtr, strideA);
    rewriter.create<Mvin_IntrOp>(loc, rs1A, rs2A);

    Value rs1B = packRs1BankIter(rewriter, loc, cstI64(rewriter, loc, bBank),
                                 cstI64(rewriter, loc, depthB));
    Value rs2B = packRs2MemStride(rewriter, loc, bPtr, strideBN);
    rewriter.create<Mvin_IntrOp>(loc, rs1B, rs2B);

    uint64_t rs1Mul = fieldBits(aBank, 0, 9) | fieldBits(bBank, 10, 19) |
                      fieldBits(cBank, 20, 29) | fieldBits(K, 30, 63);
    uint64_t rs2Mul = fieldBits(0, 0, 63);
    rewriter.create<Mul_Warp16_IntrOp>(loc, cstI64(rewriter, loc, rs1Mul),
                                       cstI64(rewriter, loc, rs2Mul));

    Value rs1C = packRs1BankIter(rewriter, loc, cstI64(rewriter, loc, cBank),
                                 cstI64(rewriter, loc, depthC));
    Value rs2C = packRs2MemStride(rewriter, loc, cPtr, strideCN);
    rewriter.create<Mvout_IntrOp>(loc, rs1C, rs2C);

    emitMset(rewriter, loc, aBank, 0, 0, 0);
    emitMset(rewriter, loc, bBank, 0, 0, 0);
    emitMset(rewriter, loc, cBank, 0, 0, 0);

    rewriter.eraseOp(op);
    return success();
  }
};

struct BuckyballMulWarp16Lowering
    : public ConvertOpToLLVMPattern<MulWarp16Op> {
  using ConvertOpToLLVMPattern<MulWarp16Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MulWarp16Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getOp1Bank(),
                                 adaptor.getOp2Bank(), adaptor.getWrBank(),
                                 adaptor.getIter());
    Value rs2 = adaptor.getMode();
    rewriter.replaceOpWithNewOp<Mul_Warp16_IntrOp>(op, rs1, rs2);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Transpose — 49_transpose.c (rs2 = mode)
//===----------------------------------------------------------------------===//

struct BuckyballTransposeLowering : public ConvertOpToLLVMPattern<TransposeOp> {
  using ConvertOpToLLVMPattern<TransposeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TransposeOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();

    auto inTy = cast<MemRefType>(input.getType());
    if (!inTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "transpose needs static shapes");

    uint64_t rows = inTy.getShape()[0];
    uint64_t cols = inTy.getShape()[1];

    if (rows != 16)
      return rewriter.notifyMatchFailure(
          op, "transpose lowering expects 16 rows for the i8 16xK path");

    const uint64_t inBank = 0, outBank = 1;

    emitMset(rewriter, loc, inBank, 1, 1, 1);
    emitMset(rewriter, loc, outBank, 1, 1, 1);

    Value inPtr = extractPtr(rewriter, loc, input);
    Value stride1 = cstI64(rewriter, loc, 1);
    uint64_t depthIn = rows * (cols / 16);
    if (cols % 16 != 0)
      return rewriter.notifyMatchFailure(op, "cols must be a multiple of 16");

    Value rs1In = packRs1BankIter(rewriter, loc, cstI64(rewriter, loc, inBank),
                                  cstI64(rewriter, loc, depthIn));
    Value rs2In = packRs2MemStride(rewriter, loc, inPtr, stride1);
    rewriter.create<Mvin_IntrOp>(loc, rs1In, rs2In);

    uint64_t rs1T = fieldBits(inBank, 0, 9) | fieldBits(outBank, 20, 29) |
                    fieldBits(cols, 30, 63);
    uint64_t rs2T = fieldBits(0, 0, 63);
    rewriter.create<Transpose_IntrOp>(loc, cstI64(rewriter, loc, rs1T),
                                      cstI64(rewriter, loc, rs2T));

    Value outPtr = extractPtr(rewriter, loc, output);
    uint64_t depthOut = cols * (rows / 16);
    if (rows % 16 != 0)
      return rewriter.notifyMatchFailure(op, "rows must be a multiple of 16");

    Value rs1Out = packRs1BankIter(rewriter, loc, cstI64(rewriter, loc, outBank),
                                   cstI64(rewriter, loc, depthOut));
    Value rs2Out = packRs2MemStride(rewriter, loc, outPtr, stride1);
    rewriter.create<Mvout_IntrOp>(loc, rs1Out, rs2Out);

    emitMset(rewriter, loc, inBank, 0, 0, 0);
    emitMset(rewriter, loc, outBank, 0, 0, 0);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Im2col — 48_im2col.c
//===----------------------------------------------------------------------===//

struct BuckyballIm2colLowering : public ConvertOpToLLVMPattern<Im2colOp> {
  using ConvertOpToLLVMPattern<Im2colOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Im2colOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();

    auto inTy = cast<MemRefType>(input.getType());
    if (!inTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "im2col needs static input shape");

    uint64_t inRows = inTy.getShape()[0];
    const uint64_t inBank = 0, outBank = 1;

    emitMset(rewriter, loc, inBank, 1, 1, 1);
    emitMset(rewriter, loc, outBank, 1, 1, 1);

    Value inPtr = extractPtr(rewriter, loc, input);
    Value stride1 = cstI64(rewriter, loc, 1);
    Value rs1In = packRs1BankIter(rewriter, loc, cstI64(rewriter, loc, inBank),
                                  cstI64(rewriter, loc, inRows));
    Value rs2In = packRs2MemStride(rewriter, loc, inPtr, stride1);
    rewriter.create<Mvin_IntrOp>(loc, rs1In, rs2In);

    uint64_t im2colRs1 = fieldBits(inBank, 0, 9) | fieldBits(outBank, 20, 29);

    Value kCol = adaptor.getKCol();
    Value kRow = adaptor.getKRow();
    Value inCol = adaptor.getInCol();
    Value inRow = adaptor.getInRow();
    Value startCol = adaptor.getStartCol();
    Value startRow = adaptor.getStartRow();

    Value rs2 = kCol;
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2, rewriter.create<arith::ShLIOp>(loc, kRow, cstI64(rewriter, loc, 4)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2, rewriter.create<arith::ShLIOp>(loc, inCol, cstI64(rewriter, loc, 8)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2, rewriter.create<arith::ShLIOp>(loc, inRow, cstI64(rewriter, loc, 13)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, startCol, cstI64(rewriter, loc, 23)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, startRow, cstI64(rewriter, loc, 28)));

    rewriter.create<Im2col_IntrOp>(loc, cstI64(rewriter, loc, im2colRs1), rs2);

    Value outPtr = extractPtr(rewriter, loc, output);
    auto outTy = cast<MemRefType>(output.getType());
    if (!outTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "im2col needs static output shape");
    uint64_t outRows = outTy.getShape()[0];

    Value rs1Out = packRs1BankIter(rewriter, loc, cstI64(rewriter, loc, outBank),
                                    cstI64(rewriter, loc, outRows));
    Value rs2Out = packRs2MemStride(rewriter, loc, outPtr, stride1);
    rewriter.create<Mvout_IntrOp>(loc, rs1Out, rs2Out);

    emitMset(rewriter, loc, inBank, 0, 0, 0);
    emitMset(rewriter, loc, outBank, 0, 0, 0);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Quant / Dequant — 51_quant.c, 52_dequant.c (rs2 = fp32 bits, bits [31:0])
//===----------------------------------------------------------------------===//

struct BuckyballQuantLowering : public ConvertOpToLLVMPattern<QuantOp> {
  using ConvertOpToLLVMPattern<QuantOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(QuantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();
    IntegerType i64 = rewriter.getI64Type();

    auto inTy = cast<MemRefType>(input.getType());
    if (!inTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "quant needs static shapes");

    uint64_t rows = inTy.getShape()[0];
    const uint64_t inBank = 0, outBank = 1;

    emitMset(rewriter, loc, inBank, 1, 4, 1);
    emitMset(rewriter, loc, outBank, 1, 1, 1);

    Value inPtr = extractPtr(rewriter, loc, input);
    Value stride1 = cstI64(rewriter, loc, 1);
    Value rs1In = packRs1BankIter(rewriter, loc, cstI64(rewriter, loc, inBank),
                                  cstI64(rewriter, loc, rows));
    Value rs2In = packRs2MemStride(rewriter, loc, inPtr, stride1);
    rewriter.create<Mvin_IntrOp>(loc, rs1In, rs2In);

    uint64_t qRs1 =
        fieldBits(inBank, 0, 9) | fieldBits(outBank, 20, 29) | fieldBits(rows, 30, 63);

    Value scale = adaptor.getScale();
    Value scaleBits = rewriter.create<arith::BitcastOp>(
        loc, rewriter.getI32Type(), scale);
    Value scale64 = rewriter.create<arith::ExtUIOp>(loc, i64, scaleBits);
    Value rs2q = rewriter.create<arith::AndIOp>(loc, scale64, cstI64(rewriter, loc, 0xFFFFFFFFULL));

    rewriter.create<Quant_IntrOp>(loc, cstI64(rewriter, loc, qRs1), rs2q);

    Value outPtr = extractPtr(rewriter, loc, output);
    auto outTy = cast<MemRefType>(output.getType());
    uint64_t outRows = outTy.getShape()[0];

    Value rs1Out = packRs1BankIter(rewriter, loc, cstI64(rewriter, loc, outBank),
                                     cstI64(rewriter, loc, outRows));
    Value rs2Out = packRs2MemStride(rewriter, loc, outPtr, stride1);
    rewriter.create<Mvout_IntrOp>(loc, rs1Out, rs2Out);

    emitMset(rewriter, loc, inBank, 0, 0, 0);
    emitMset(rewriter, loc, outBank, 0, 0, 0);

    rewriter.eraseOp(op);
    return success();
  }
};

struct BuckyballDequantLowering : public ConvertOpToLLVMPattern<DequantOp> {
  using ConvertOpToLLVMPattern<DequantOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(DequantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();
    IntegerType i64 = rewriter.getI64Type();

    auto inTy = cast<MemRefType>(input.getType());
    if (!inTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dequant needs static shapes");

    uint64_t rows = inTy.getShape()[0];
    const uint64_t inBank = 0, outBank = 1;

    emitMset(rewriter, loc, inBank, 1, 1, 1);
    emitMset(rewriter, loc, outBank, 1, 4, 1);

    Value inPtr = extractPtr(rewriter, loc, input);
    Value stride1 = cstI64(rewriter, loc, 1);
    Value rs1In = packRs1BankIter(rewriter, loc, cstI64(rewriter, loc, inBank),
                                  cstI64(rewriter, loc, rows));
    Value rs2In = packRs2MemStride(rewriter, loc, inPtr, stride1);
    rewriter.create<Mvin_IntrOp>(loc, rs1In, rs2In);

    uint64_t dRs1 =
        fieldBits(inBank, 0, 9) | fieldBits(outBank, 20, 29) | fieldBits(rows, 30, 63);

    Value scale = adaptor.getScale();
    Value scaleBits = rewriter.create<arith::BitcastOp>(
        loc, rewriter.getI32Type(), scale);
    Value scale64 = rewriter.create<arith::ExtUIOp>(loc, i64, scaleBits);
    Value rs2d = rewriter.create<arith::AndIOp>(loc, scale64, cstI64(rewriter, loc, 0xFFFFFFFFULL));

    rewriter.create<Dequant_IntrOp>(loc, cstI64(rewriter, loc, dRs1), rs2d);

    Value outPtr = extractPtr(rewriter, loc, output);
    auto outTy = cast<MemRefType>(output.getType());
    uint64_t outRows = outTy.getShape()[0];

    Value rs1Out = packRs1BankIter(rewriter, loc, cstI64(rewriter, loc, outBank),
                                    cstI64(rewriter, loc, outRows));
    Value rs2Out = packRs2MemStride(rewriter, loc, outPtr, stride1);
    rewriter.create<Mvout_IntrOp>(loc, rs1Out, rs2Out);

    emitMset(rewriter, loc, inBank, 0, 0, 0);
    emitMset(rewriter, loc, outBank, 0, 0, 0);

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void mlir::populateBuckyballLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    int64_t lane, int64_t warp, int64_t bankDepth, int64_t bankNum) {
  (void)lane;
  (void)warp;
  (void)bankDepth;
  (void)bankNum;

  patterns
      .add<ForwardOperands<func::CallOp>, ForwardOperands<func::CallIndirectOp>,
           ForwardOperands<func::ReturnOp>>(converter, &converter.getContext());
  patterns.add<BuckyballFenceLowering>(converter);
  patterns.add<BuckyballMsetLowering>(converter);
  patterns.add<BuckyballMvinLowering>(converter);
  patterns.add<BuckyballMvoutLowering>(converter);
  patterns.add<BuckyballMatMulLowering>(converter);
  patterns.add<BuckyballMulWarp16Lowering>(converter);
  patterns.add<BuckyballTransposeLowering>(converter);
  patterns.add<BuckyballIm2colLowering>(converter);
  patterns.add<BuckyballQuantLowering>(converter);
  patterns.add<BuckyballDequantLowering>(converter);
}

void mlir::configureBuckyballLegalizeForExportTarget(
    LLVMConversionTarget &target) {
  target.addLegalOp<Fence_IntrOp, Mvin_IntrOp, Mvout_IntrOp, Mul_Warp16_IntrOp,
                    Transpose_IntrOp, Im2col_IntrOp, Quant_IntrOp,
                    Dequant_IntrOp, Relu_IntrOp, Mset_IntrOp>();
  target.addIllegalOp<FenceOp, MsetOp, MvinOp, MvoutOp, MatMulOp, MulWarp16Op,
                      TransposeOp, Im2colOp, QuantOp, DequantOp>();
  target.addLegalDialect<memref::MemRefDialect>();
  target.addLegalDialect<arith::ArithDialect>();
}
