//===- MatMulGemmini.cpp --------------------------------------------===//
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
// This file implements the risc version gemmini matmul.
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/Affine/LoopUtils.h>
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

#include "Utils/Utils.h"
#include "Gemmini/GemminiDialect.h"
#include "Gemmini/GemminiOps.h"
#include "Gemmini/Transform.h"
#include <iostream>

using namespace mlir;
using namespace buddy;
using namespace buddy::gemmini;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class MatMulGemminiPattern : public ConversionPattern {
public:
  explicit MatMulGemminiPattern(MLIRContext *context)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), 1, context) {
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    
    // Needed for i64 var runtime printf debug.
    // ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    // auto printfRef = getOrInsertPrintf(rewriter, parentModule);
    // Value formatSpecifierCst = getOrCreateGlobalString(
    //     loc, rewriter, "print_index_format", StringRef("%ld\n\0", 5), parentModule);

    // Create D matrix.
    auto i8Type = rewriter.getIntegerType(8);
    auto memrefType = MemRefType::get({32, 32}, i8Type);
    Value D = rewriter.create<memref::AllocOp>(loc, memrefType);
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI8IntegerAttr(2));
    rewriter.create<linalg::FillOp>(
        loc, /*inputs=*/ValueRange{zero}, /*outputs=*/ValueRange{D});
      
    // Get input A, B, C.
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);
    // Get shape of input and output
    ShapedType ATy = A.getType().cast<ShapedType>();
    Type eleTy = ATy.getElementType();
    ShapedType BTy = B.getType().cast<ShapedType>();
    // ShapedType CTy = C.getType().cast<ShapedType>();

    auto ctx = op->getContext();
    // Some constants.
    const Value c0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value c1 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    // Create DimOp.
    const Value aRow = rewriter.create<memref::DimOp>(loc, A, c0);
    const Value cstARow = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(ATy.getDimSize(0)));
    // This algorithm does not use the column A index.
    // const Value aCol = rewriter.create<memref::DimOp>(loc, A, c1);
    const Value bRow = rewriter.create<memref::DimOp>(loc, B, c0);
    const Value bCol = rewriter.create<memref::DimOp>(loc, B, c1);
    Type i64Type = rewriter.getIntegerType(64);

    // Create forI, forJ, forK loops.
    auto forI = rewriter.create<affine::AffineForOp> (
        loc, 0, rewriter.getIndexAttr(ATy.getDimSize(0)).getInt(), 1, std::nullopt);
    // auto forI = rewriter.create<affine::AffineForOp>(
    //     loc, ValueRange{c0}, rewriter.getDimIdentityMap(),
    //     ValueRange{aRow}, rewriter.getDimIdentityMap(), /*Step=*/1, std::nullopt);
    Value ivI = forI.getInductionVar();
    rewriter.setInsertionPointToStart(forI.getBody());

    auto forJ = rewriter.create<affine::AffineForOp> (
        loc, 0, rewriter.getIndexAttr(BTy.getDimSize(1)).getInt(), 1, std::nullopt);
    // auto forJ = rewriter.create<affine::AffineForOp>(
    //     loc, ValueRange{c0}, rewriter.getDimIdentityMap(),
    //     ValueRange{bCol}, rewriter.getDimIdentityMap(), /*Step=*/1, std::nullopt);
    Value ivJ = forJ.getInductionVar();
    rewriter.setInsertionPointToStart(forJ.getBody());

    Value aEle;
    Value bEle;
    Value cEle;
    Value result;
    auto forK = rewriter.create<affine::AffineForOp>(
        loc, 0, rewriter.getIndexAttr(ATy.getDimSize(1)).getInt(), 1, std::nullopt,
        [&](OpBuilder &builder, Location loc, Value ivK, ValueRange itrArgs) {
          aEle = builder.create<affine::AffineLoadOp>(
              loc, A, ValueRange{ivI, ivK});
          bEle = builder.create<affine::AffineLoadOp>(
              loc, B, ValueRange{ivK, ivJ});
          cEle = builder.create<affine::AffineLoadOp>(
              loc, C, ValueRange{ivI, ivJ});
          result = builder.create<arith::MulFOp>(loc, aEle, bEle);
          result = builder.create<arith::AddFOp>(loc, cEle, result);
          builder.create<affine::AffineStoreOp>(
              loc, result, C, ValueRange{ivI, ivJ});
          builder.create<affine::AffineYieldOp>(loc);
        });
    // auto forK = rewriter.create<affine::AffineForOp>(
    //     loc, ValueRange{c0}, rewriter.getDimIdentityMap(),
    //     ValueRange{bRow}, rewriter.getDimIdentityMap(), /*Step=*/1, std::nullopt,
    //     [&](OpBuilder &builder, Location loc, Value ivK, ValueRange itrArgs) {
    //       Value aEle = builder.create<affine::AffineLoadOp>(
    //           loc, A, ValueRange{ivI, ivK});
    //       Value bEle = builder.create<affine::AffineLoadOp>(
    //           loc, B, ValueRange{ivK, ivJ});
    //       Value cEle = builder.create<affine::AffineLoadOp>(
    //           loc, C, ValueRange{ivI, ivJ});
    //       Value result = builder.create<arith::MulFOp>(loc, aEle, bEle);
    //       result = builder.create<arith::AddFOp>(loc, cEle, result);
    //       builder.create<affine::AffineStoreOp>(
    //           loc, result, C, ValueRange{ivI, ivJ});
    //       builder.create<affine::AffineYieldOp>(loc);
    //     });

    // Tile forI, forJ, forK loops to create a band of loops.
    // The tile size is 16x16x16.
    SmallVector<affine::AffineForOp, 6> band;
    band.push_back(forI);
    band.push_back(forJ);
    band.push_back(forK);
    SmallVector<unsigned, 6> tileSizes;
    tileSizes.push_back(16);
    tileSizes.push_back(16);
    tileSizes.push_back(16);
    SmallVector<affine::AffineForOp, 6> tiledNest;
    tilePerfectlyNested(band, tileSizes, &tiledNest);
    
    // Some constants used for gemmini intrinsics.
    rewriter.setInsertionPointToStart(tiledNest[2].getBody());
    Value cAddrLen = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(32));
    Value c16 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(16));
    Value cRows = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(16));
    Value cCols = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(16));
    Value shiftAmt = rewriter.create<arith::AddIOp>(loc, cAddrLen, c16);
    Value shiftedRows = rewriter.create<arith::ShLIOp>(loc, cRows, shiftAmt);
    Value shiftedCols = rewriter.create<arith::ShLIOp>(loc, cCols, cAddrLen);

    Value ivIo = tiledNest[0].getInductionVar();
    Value ivJo = tiledNest[1].getInductionVar();
    Value ivKo = tiledNest[2].getInductionVar();

    // Move-in A
    Value aStride = rewriter.create<arith::IndexCastOp>(
      loc, i64Type, bRow);
    rewriter.create<ConfigLdOp>(loc, aStride,
                                llvm::APFloat((float)1.0));
    
    Value aSpAddr = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(0));
    Value aTmp = rewriter.create<arith::OrIOp>(loc, shiftedRows, shiftedCols);
    Value aSpad = rewriter.create<arith::OrIOp>(loc, aTmp, aSpAddr);
    Value aBase = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
      loc, rewriter.getIndexType(), A);
    auto aMeta = rewriter.create<memref::ExtractStridedMetadataOp>(loc, A);
    Value aStride0 = aMeta.getStrides()[0]; // index
    Value aStride1 = aMeta.getStrides()[1]; // index
    Value ivIoMulaStride0 = rewriter.create<arith::MulIOp>(loc, ivIo, aStride0);
    Value ivKoMulaStride1 = rewriter.create<arith::MulIOp>(loc, ivKo, aStride1);
    Value aOffset = rewriter.create<arith::AddIOp>(loc, ivIoMulaStride0, ivKoMulaStride1);
    Value aOffsetI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, aOffset);
    Value aDramAddr = rewriter.create<arith::AddIOp>(loc, aBase, aOffset);
    Value aDramAddrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, aDramAddr);
    rewriter.create<Mvin_IntrOp>(loc, aDramAddrI64, aSpad);

    // Move-in B
    Value bStride = rewriter.create<arith::IndexCastOp>(
      loc, i64Type, bCol);
    rewriter.create<ConfigLdOp>(loc, bStride,
                                llvm::APFloat((float)1.0));

    Value bSpAddr = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(256));
    Value bTmp = rewriter.create<arith::OrIOp>(loc, shiftedRows, shiftedCols);
    Value bSpad = rewriter.create<arith::OrIOp>(loc, bTmp, bSpAddr);
    Value bBase = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
      loc, rewriter.getIndexType(), B);
    auto bMeta = rewriter.create<memref::ExtractStridedMetadataOp>(loc, B);
    Value bStride0 = bMeta.getStrides()[0]; // index
    Value bStride1 = bMeta.getStrides()[1]; // index
    Value ivJoMulbStride0 = rewriter.create<arith::MulIOp>(loc, ivJo, bStride0);
    Value ivKoMulbStride1 = rewriter.create<arith::MulIOp>(loc, ivKo, bStride1);
    Value bOffset = rewriter.create<arith::AddIOp>(loc, ivJoMulbStride0, ivKoMulbStride1);
    Value bDramAddr = rewriter.create<arith::AddIOp>(loc, bBase, bOffset);
    Value bDramAddrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, bDramAddr);
    rewriter.create<Mvin_IntrOp>(loc, bDramAddrI64, bSpad);

    // Compute
    rewriter.create<ConfigExOp>(loc, /*dataflow = */0, /*sysAct = */0,
                                /*sysShift = */0, llvm::APFloat((float)1.0));

    Value cSpAddr = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(512));
    Value dSpAddr = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(768));
    Value aRowI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, aRow);
    Value bRowI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, bRow);
    Value bColI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, bCol);
    rewriter.create<PreloadOp>(
      loc, dSpAddr, cSpAddr, c16, c16, c16, c16);
    
    Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value isZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, ivKo, zeroIndex);
    rewriter.create<scf::IfOp>(loc, isZero,
        [&](OpBuilder &builder, Location loc) {
          // if-true block: ivKo == 0
          builder.create<ComputePreloadedOp>(
              loc, aSpAddr, bSpAddr, c16, c16, c16, c16);
          builder.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &builder, Location loc) {
          // if-false block: ivKo != 0
          builder.create<ComputeAccumulatedOp>(
              loc, aSpAddr, bSpAddr, c16, c16, c16, c16);
          builder.create<scf::YieldOp>(loc);
        });

    // Move-out C
    Value cStride = rewriter.create<arith::IndexCastOp>(
      loc, i64Type, bCol);
    rewriter.create<ConfigStOp>(
      loc, cStride, 0, llvm::APFloat((float)1.0));

    Value cTmp = rewriter.create<arith::OrIOp>(loc, shiftedRows, shiftedCols);
    Value cSpad = rewriter.create<arith::OrIOp>(loc, cTmp, cSpAddr);
    Value cBase = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
      loc, rewriter.getIndexType(), C);
    auto cMeta = rewriter.create<memref::ExtractStridedMetadataOp>(loc, C);
    Value cStride0 = cMeta.getStrides()[0]; // index
    Value cStride1 = cMeta.getStrides()[1]; // index
    Value ivIoMulcStride0 = rewriter.create<arith::MulIOp>(loc, ivIo, cStride0);
    Value ivJoMulcStride1 = rewriter.create<arith::MulIOp>(loc, ivJo, cStride1);
    Value cOffset = rewriter.create<arith::AddIOp>(loc, ivIoMulcStride0, ivJoMulcStride1);
    Value cDramAddr = rewriter.create<arith::AddIOp>(loc, cBase, cOffset);
    Value cDramAddrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, cDramAddr);
    rewriter.create<Mvout_IntrOp>(loc, cDramAddrI64, cSpad);

    rewriter.eraseOp(tiledNest[3]);
    rewriter.eraseOp(op);
    return success();
  }
private:
  // Helper functions to create global string and printf function.
  static LLVM::LLVMFunctionType getPrintfType(MLIRContext *context) {
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmPtr = LLVM::LLVMPointerType::get(context);
    return LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtr, true);
  }

  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                            ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
      return SymbolRefAttr::get(context, "printf");

    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf",
                                      getPrintfType(context));
    return SymbolRefAttr::get(context, "printf");
  }

  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                      StringRef name, StringRef value,
                                      ModuleOp module) {
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMArrayType::get(
          IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value), 0);
    }

    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                  builder.getIndexAttr(0));
    return builder.create<LLVM::GEPOp>(
        loc, LLVM::LLVMPointerType::get(builder.getContext()), global.getType(),
        globalPtr, ArrayRef<Value>({cst0, cst0}));
  }

};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// MatMulGemminiPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg matmul operations to mixture of
/// Affine + Gemmini intrinsic operations.
namespace {
class MatMulGemminiPass
    : public PassWrapper<MatMulGemminiPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulGemminiPass)
  StringRef getArgument() const final { return "matmul-gemmini"; }
  StringRef getDescription() const final { return "MatMul Gemmini."; }
  MatMulGemminiPass() = default;
  MatMulGemminiPass(const MatMulGemminiPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, gemmini::GemminiDialect,
                    LLVM::LLVMDialect>();
  }

};
} // end anonymous namespace.

void MatMulGemminiPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                       scf::SCFDialect, memref::MemRefDialect,
                       gemmini::GemminiDialect, LLVM::LLVMDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<MatMulGemminiPattern>(context);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerMatMulGemminiPass() {
  PassRegistration<MatMulGemminiPass>();
}
} // namespace buddy
} // namespace mlir
