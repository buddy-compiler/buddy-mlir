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

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"
#include "Buckyball/Transform.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace buddy::buckyball;

namespace {
int64_t getNumberFromValue(Value &value) {
  return dyn_cast<IntegerAttr>(value.getDefiningOp()->getAttr("value")).getInt();
} 
}// namespace


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
// Instruction-level Ops: Transform Xop to X_IntrOp
//===----------------------------------------------------------------------===//

struct BuckyballFenceLowering : public ConvertOpToLLVMPattern<FenceOp> {
  using ConvertOpToLLVMPattern<FenceOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(FenceOp fenceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<Fence_IntrOp>(fenceOp);
    return success();
  }
};

struct BuckyballMvinLowering : public ConvertOpToLLVMPattern<MvinOp> {
  using ConvertOpToLLVMPattern<MvinOp>::ConvertOpToLLVMPattern;
  explicit BuckyballMvinLowering(LLVMTypeConverter &typeConverter,
                               int64_t spAddrLen)
      : ConvertOpToLLVMPattern(typeConverter), spAddrLen(spAddrLen) {}
  LogicalResult
  matchAndRewrite(MvinOp mvinOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = mvinOp.getLoc();
    // Use original for memref operations (needs MemRefType)
    Value input = mvinOp.getInput();
    // Use adaptor for already-converted scalar values
    Value addr = adaptor.getAddr();
    Value stride = adaptor.getStride();

    TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
    Value extractOp = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, resultType, input);
    IntegerType i64Type = rewriter.getI64Type();
    Value indexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOp);
    
    Value row = rewriter.create<memref::DimOp>(loc, input, 0);
    row = rewriter.create<arith::IndexCastOp>(loc, i64Type, row);
    Value shift1 = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(spAddrLen));
    row = rewriter.create<arith::ShLIOp>(loc, row, shift1);
    Value shift2 = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(10));
    Value strideShifted = rewriter.create<arith::ShLIOp>(loc, stride, shift2);

    // rs1 = indexCastOp
    // rs2 = (stride << spAddrLen+10) | (row << spAddrLen) | spadAddrValue
    Value rs1 = indexCastOp;
    Value rs2 = rewriter.create<arith::OrIOp>(loc, row, addr);
    rs2 = rewriter.create<arith::OrIOp>(loc, rs2, strideShifted);

    rewriter.replaceOpWithNewOp<Mvin_IntrOp>(mvinOp, rs1, rs2);
    return success();
  }

private:
  int64_t spAddrLen;
};

struct BuckyballMvoutLowering : public ConvertOpToLLVMPattern<MvoutOp> {
  using ConvertOpToLLVMPattern<MvoutOp>::ConvertOpToLLVMPattern;
  explicit BuckyballMvoutLowering(LLVMTypeConverter &typeConverter,
                                int64_t spAddrLen)
      : ConvertOpToLLVMPattern(typeConverter), spAddrLen(spAddrLen) {}
  LogicalResult
  matchAndRewrite(MvoutOp mvoutOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = mvoutOp.getLoc();
    // Use original for memref operations (needs MemRefType)
    Value output = mvoutOp.getOutput();
    // Use adaptor for already-converted scalar values
    Value addr = adaptor.getAddr();

    TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
    Value extractOp = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, resultType, output);
    IntegerType i64Type = rewriter.getI64Type();
    Value indexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOp);

    Value row = rewriter.create<memref::DimOp>(loc, output, 0);
    row = rewriter.create<arith::IndexCastOp>(loc, i64Type, row);
    Value shift1 = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(spAddrLen));
    row = rewriter.create<arith::ShLIOp>(loc, row, shift1);
    
    // rs1 = indexCastOp
    // rs2 = row << spAddrLen | spadAddrValue
    Value rs1 = indexCastOp;
    Value rs2 = rewriter.create<arith::OrIOp>(loc, row, addr);
    rewriter.replaceOpWithNewOp<Mvout_IntrOp>(mvoutOp, rs1, rs2);
    return success();
  }

private:
  int64_t spAddrLen;
};


struct BuckyballTransposeLowering : public ConvertOpToLLVMPattern<TransposeOp> {
  using ConvertOpToLLVMPattern<TransposeOp>::ConvertOpToLLVMPattern;
    explicit BuckyballTransposeLowering(LLVMTypeConverter &typeConverter,
                                   int64_t dim, int64_t spAddrLen, 
                                   int64_t spadRows, int64_t warp, int64_t lane)
      : ConvertOpToLLVMPattern(typeConverter), dim(dim), spAddrLen(spAddrLen),
        spadRows(spadRows), warp(warp), lane(lane) {}
  LogicalResult
  matchAndRewrite(TransposeOp transposeOp, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter) const override {
  Location loc = transposeOp.getLoc();

  // Get original memref types (before conversion)
  Value input = transposeOp.getInput();
  Value output = transposeOp.getOutput();

  MemRefType inType = dyn_cast<MemRefType>(transposeOp.getOperandTypes()[0]);
  MemRefType outType = dyn_cast<MemRefType>(transposeOp.getOperandTypes()[1]);

  llvm::ArrayRef<int64_t> inShape = inType.getShape();
  llvm::ArrayRef<int64_t> outShape = outType.getShape();

  // Expect 2-D transpose
  if (inShape.size() != 2 || outShape.size() != 2)
    return rewriter.notifyMatchFailure(transposeOp, "only 2-D transpose supported");

  uint64_t rows = inShape[0];
  uint64_t cols = inShape[1];

  IntegerType i64Type = rewriter.getI64Type();

  // --- scratchpad address layout (follow MatMul convention) ---
  // Input matrix A placed at sp address 0
  const uint64_t aSpAddrStart = 0;
  // Output placed in high region so it doesn't overlap input (use high bit like MatMul's cSpAddrStart)
  const uint64_t outSpAddrStart = spadRows / 2 ;

  // --- Load input matrix to scratchpad (Mvin) ---
  Value extractIn = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, input);
  Value indexCastIn = rewriter.create<arith::IndexCastOp>(loc, i64Type, extractIn);

  // Encode load: follow MatMul style: (rows << (spAddrLen + 16)) | (cols << spAddrLen) | spadStart | (1ULL << (spAddrLen + 26))
  uint64_t aAddrEncoded = (rows << (spAddrLen + 16)) | (cols << spAddrLen) | aSpAddrStart | (1ULL << (spAddrLen + 26));
  Value rs1A = indexCastIn;
  Value rs2A = rewriter.create<arith::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(aAddrEncoded));
  rewriter.create<Mvin_IntrOp>(loc, rs1A, rs2A);
  
  // --- Generate warp transpose operations ---
  // We'll tile the matrix into warp-sized tiles. Using defaults consistent with MatMul examples:
  uint64_t metaRowNum = (rows + lane - 1) / lane;
  uint64_t metaColNum = (cols + lane - 1) / lane;

  // For each meta-tile, compute the scratchpad addresses for input tile and output tile,
  // then issue a warp transpose intrinsic. The exact encoding for the transpose intrinsic
  // is assumed here; adjust if platform uses different bitfields.
  for (uint64_t r0 = 0; r0 < metaRowNum; ++r0) {
    for (uint64_t c0 = 0; c0 < metaColNum; ++c0) {
      // Compute starting scratchpad addresses for the tile.
      // Layout assumption: input stored in row-major contiguous blocks of 'lane' rows each.
      uint64_t inTileSpAddr = aSpAddrStart + r0 * lane + c0 * (rows); // similar to MatMul aWarpSpAddr pattern
      // For output (transposed), place tile at outSpAddrStart + c0*lane + r0*(cols)
      uint64_t outTileSpAddr = outSpAddrStart + c0 * lane + r0 * (cols);

      // nRows/nCols for this tile (may be smaller at edges)
      uint64_t tileRows = std::min<uint64_t>(lane, rows - r0 * lane);
      uint64_t tileCols = std::min<uint64_t>(lane, cols - c0 * lane);

      uint iter = std::max<uint64_t>(tileRows, tileCols);
      // Encode rs1 and rs2 for transpose intrinsic.
      // We'll follow a similar packing convention:
      // rs1 = (inTileSpAddr) | (outTileSpAddr << spAddrLen)  (or reversed) --
      // choose consistent packing: rs1 = (out << spAddrLen) | in
      uint64_t rs1 = (outTileSpAddr << spAddrLen) | inTileSpAddr;
      // rs2 encodes tile dimensions: (tileRows << spAddrLen) | tileCols
      uint64_t rs2 = (iter << spAddrLen);

      Value rs1Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(rs1));
      Value rs2Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(rs2));

      // Issue the warp transpose intrinsic.
      rewriter.create<Transpose_IntrOp>(loc, rs1Value, rs2Value);
    }
  }

  // --- Mvout - Write back output matrix ---
  Value extractOut = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, output);
  Value indexCastOut = rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOut);

  // Encode mvout address: (rows << spAddrLen) | spadAddrStart  (here rows = out rows)
  uint64_t mvoutAddrEncoded = (outShape[0] << spAddrLen) | outSpAddrStart;
  Value rs1Mvout = indexCastOut;
  Value rs2Mvout = rewriter.create<arith::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(mvoutAddrEncoded));
  rewriter.create<Mvout_IntrOp>(loc, rs1Mvout, rs2Mvout);

  rewriter.eraseOp(transposeOp);
  return success();
}

private:
  int64_t dim;
  int64_t spAddrLen;
  int64_t spadRows;
  int64_t warp;
  int64_t lane;
};

//===----------------------------------------------------------------------===//
// Hardware-level MatMul Op (combines Meta-tile and Warp-level compute)
//===----------------------------------------------------------------------===//
struct BuckyballMatMulLowering : public ConvertOpToLLVMPattern<MatMulOp> {
  using ConvertOpToLLVMPattern<MatMulOp>::ConvertOpToLLVMPattern;
  explicit BuckyballMatMulLowering(LLVMTypeConverter &typeConverter,
                                   int64_t dim, int64_t spAddrLen, 
                                   int64_t spadRows, int64_t warp, int64_t lane)
      : ConvertOpToLLVMPattern(typeConverter), dim(dim), spAddrLen(spAddrLen),
        spadRows(spadRows), warp(warp), lane(lane) {}
  
  LogicalResult
  matchAndRewrite(MatMulOp matMulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = matMulOp.getLoc();
    
    // Get original memref types (before conversion)
    Value aMemArray = matMulOp.getAMemArray();
    Value bMemArray = matMulOp.getBMemArray();
    Value cMemArray = matMulOp.getCMemArray();
    
    MemRefType aMemArrayType = dyn_cast<MemRefType>(matMulOp.getOperandTypes()[0]);
    MemRefType bMemArrayType = dyn_cast<MemRefType>(matMulOp.getOperandTypes()[1]);
    MemRefType cMemArrayType = dyn_cast<MemRefType>(matMulOp.getOperandTypes()[2]);
    
    llvm::ArrayRef<int64_t> aMemArrayShape = aMemArrayType.getShape();
    llvm::ArrayRef<int64_t> bMemArrayShape = bMemArrayType.getShape();
    llvm::ArrayRef<int64_t> cMemArrayShape = cMemArrayType.getShape();
    
    uint64_t M = aMemArrayShape[0];
    uint64_t K = aMemArrayShape[1];
    uint64_t N = bMemArrayShape[1];
    
    IntegerType i64Type = rewriter.getI64Type();
    
    // Compute scratchpad addresses
    const uint64_t aSpAddrStart = 0;
    const uint64_t bSpAddrStart = spadRows - (K * N) / dim;
    const uint64_t cSpAddrStart = 1ULL << (spAddrLen - 1);
    
    // Load A matrix to scratchpad
    Value extractA = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, aMemArray);
    Value indexCastA = rewriter.create<arith::IndexCastOp>(loc, i64Type, extractA);
    uint64_t aAddrEncoded = (M << (spAddrLen + 16)) | (K << spAddrLen) | aSpAddrStart | (1ULL << (spAddrLen + 26));
    Value rs1A = indexCastA;
    Value rs2A = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(aAddrEncoded));
    rewriter.create<Mvin_IntrOp>(loc, rs1A, rs2A);
    
    // Load B matrix to scratchpad  
    Value extractB = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, bMemArray);
    Value indexCastB = rewriter.create<arith::IndexCastOp>(loc, i64Type, extractB);
    uint64_t bAddrEncoded = (K << (spAddrLen + 16)) | (N << spAddrLen) | bSpAddrStart | (1ULL << (spAddrLen + 26));
    Value rs1B = indexCastB;
    Value rs2B = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(bAddrEncoded));
    rewriter.create<Mvin_IntrOp>(loc, rs1B, rs2B);
    
    // Compute: Generate warp multiply operations
    uint64_t metaMNum = (M + lane - 1) / lane;
    uint64_t metaKNum = (K + warp - 1) / warp;
    
    for (size_t k0 = 0; k0 < metaKNum; k0++) {
      for (size_t m0 = 0; m0 < metaMNum; m0++) {
        // Compute addresses for warp multiply
        uint64_t aWarpSpAddr = aSpAddrStart + m0 * lane + k0 * M;
        uint64_t bWarpSpAddr = bSpAddrStart + k0 * N;
        uint64_t cWarpSpAddr = cSpAddrStart + m0 * lane;
        uint64_t nLen = N;
        
        // Encode rs1 and rs2 for warp multiply intrinsic
        // rs1 = (bSpAddr << spAddrLen) | aSpAddr 
        // rs2 = (nLen << spAddrLen) | cSpAddr 
        uint64_t rs1 = (bWarpSpAddr << spAddrLen) | aWarpSpAddr;
        uint64_t rs2 = (nLen << spAddrLen) | cWarpSpAddr;
        
        Value rs1Value = rewriter.create<arith::ConstantOp>(
            loc, i64Type, rewriter.getI64IntegerAttr(rs1));
        Value rs2Value = rewriter.create<arith::ConstantOp>(
            loc, i64Type, rewriter.getI64IntegerAttr(rs2));
        
        // Execute warp multiply intrinsic
        rewriter.create<Mul_Warp16_IntrOp>(loc, rs1Value, rs2Value);
      }
    }
    
    // Mvout - Write back C matrix  
    Value extractC = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, cMemArray);
    Value indexCastC = rewriter.create<arith::IndexCastOp>(loc, i64Type, extractC);
    
    // Encode mvout address: (rows << spAddrLen) | spadAddr
    uint64_t mvoutAddrEncoded = (cMemArrayShape[0] << spAddrLen) | cSpAddrStart;
    Value rs1Mvout = indexCastC;
    Value rs2Mvout = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(mvoutAddrEncoded));
    rewriter.create<Mvout_IntrOp>(loc, rs1Mvout, rs2Mvout);
    
    rewriter.eraseOp(matMulOp);
    return success();
  }
  
private:
  int64_t dim;
  int64_t spAddrLen;
  int64_t spadRows;
  int64_t warp;
  int64_t lane;
};

void mlir::populateBuckyballLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, int64_t dim,
    int64_t memAddrLen, int64_t spAddrLen, int64_t accRows, int64_t spadRows, size_t sizeOfElemT,
    size_t sizeOfAccT, int64_t warp , int64_t lane, int64_t hartId) {
  patterns
      .add<ForwardOperands<func::CallOp>, ForwardOperands<func::CallIndirectOp>,
           ForwardOperands<func::ReturnOp>>(converter, &converter.getContext());
  patterns.add<BuckyballFenceLowering>(converter);
  patterns.add<BuckyballMvinLowering>(converter, spAddrLen);
  patterns.add<BuckyballMvoutLowering>(converter, spAddrLen);
  patterns.add<BuckyballMatMulLowering>(converter, dim, spAddrLen, spadRows, warp, lane);
  patterns.add<BuckyballTransposeLowering>(converter, dim, spAddrLen, spadRows, warp, lane);
}

void mlir::configureBuckyballLegalizeForExportTarget(
    LLVMConversionTarget &target) {
  target.addLegalOp<Fence_IntrOp, Mvin_IntrOp, 
                    Mvout_IntrOp, Mul_Warp16_IntrOp, Transpose_IntrOp>();
  target.addIllegalOp<FenceOp, MvinOp, MvoutOp, MatMulOp, TransposeOp>();
  // Allow memref operations during conversion
  target.addLegalDialect<memref::MemRefDialect>();
  target.addLegalDialect<arith::ArithDialect>();
}
