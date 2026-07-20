//=== LowerLinalgToBOSCAME.cpp - Linalg to BOSCAME Dialect Lowering Pass --===//
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
// This file defines Linalg dialect lowering pass to BOSCAME dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Dialect/BOSCAME/BOSCAMEDialect.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"

using namespace mlir;
using namespace buddy::boscame;

namespace {

class MatmulToBOSCAMELowering : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    if (!op.hasPureBufferSemantics())
      return failure();

    Value A = op.getDpsInputOperand(0)->get();
    Value B = op.getDpsInputOperand(1)->get();
    Value C = op.getDpsInitOperand(0)->get();
    Value originalC = C;
    memref::CopyOp copyToFinalOutput;
    SmallVector<linalg::FillOp> deadInitFills;

    auto AType = dyn_cast<MemRefType>(A.getType());
    auto BType = dyn_cast<MemRefType>(B.getType());
    auto CType = dyn_cast<MemRefType>(C.getType());

    if (!AType || !BType || !CType)
      return failure();

    auto moveDefsBeforeMatmul = [&](auto &moveDefsBeforeMatmul,
                                    Value value) -> LogicalResult {
      Operation *def = value.getDefiningOp();
      if (!def || def->getBlock() != op->getBlock())
        return success();
      if (!op->isBeforeInBlock(def))
        return success();
      if (!isMemoryEffectFree(def))
        return failure();

      for (Value operand : def->getOperands()) {
        if (operand.getDefiningOp() == op)
          return failure();
        if (failed(moveDefsBeforeMatmul(moveDefsBeforeMatmul, operand)))
          return failure();
      }

      def->moveBefore(op);
      return success();
    };

    for (Operation *user : llvm::make_early_inc_range(C.getUsers())) {
      auto copyOp = dyn_cast<memref::CopyOp>(user);
      if (!copyOp || copyOp.getSource() != C)
        continue;

      auto targetType = dyn_cast<MemRefType>(copyOp.getTarget().getType());
      if (!targetType || targetType.getElementType() != CType.getElementType())
        continue;
      if (targetType.getRank() != CType.getRank() ||
          targetType.getShape() != CType.getShape())
        continue;
      if (failed(moveDefsBeforeMatmul(moveDefsBeforeMatmul,
                                      copyOp.getTarget())))
        continue;

      copyToFinalOutput = copyOp;
      for (Operation *user : originalC.getUsers()) {
        auto fillOp = dyn_cast<linalg::FillOp>(user);
        if (fillOp && fillOp.getDpsInitOperand(0)->get() == originalC)
          deadInitFills.push_back(fillOp);
      }
      C = copyOp.getTarget();
      CType = targetType;
      break;
    }

    Type elemTypeA = AType.getElementType();
    Type elemTypeB = BType.getElementType();
    Type elemTypeC = CType.getElementType();

    if (elemTypeA != elemTypeB) {
      return rewriter.notifyMatchFailure(
          op, "Operand A and B must have the same type.");
    }

    int64_t tileM = 4, tileN = 4, tileK = 4;
    int64_t mmaMtypeImm = 0; // mtype for A/B elements — used during MMA
    int64_t accMtypeImm = 0; // mtype for C   element  — used for mlce/msce

    // Helper: build mtype CSR value for Qwen3 FPGA AME.
    //
    // The mtype CSR encodes element type + MMA enable in a single 64-bit
    // register.  Qwen3's FPGA RTL expects a bitfield-encoded value, NOT a
    // raw element-width (see kernel/src/backends/ame/core/ame_core.c).
    //
    // mtype CSR layout (RISC-V Matrix Extension v0.5 / Qwen3 RTL):
    //   bit 16: mma   (matrix multiply-accumulate enable — MUST be 1)
    //   bit 12: mf64,  bit 11: mf32,  bit 10: mbf16, bit 9: mf16
    //   bit  8: mint4
    //   bit  7: mint64, bit  6: mint32, bit  5: mint16, bit 4: mint8
    //   bits 1:0: msew (element width: 0=e8, 1=e16, 2=e32, 3=e64)
    //
    // NOTE: float / bf16 / int4 / int16 / int64 encodings are inferred from
    // the same field layout and have NOT been validated against FPGA RTL.
    // Qwen3's validated int8 path uses AME_MTYPE_INT8 for mqma and
    // AME_MTYPE_INT32 for mlce32/msce32, while the C buffer is float.
    auto mtypeVal = [](unsigned msew, unsigned typeBit) -> int64_t {
      return (1LL << 16) | (1LL << typeBit) | msew;
    };

    // Compute the accumulator (C) mtype — the hardware requires switching
    // mtype to INT32 before mlce32 / msce32, even when the MMA step used
    // INT8 (see ame_matmul_*_i8_i8_f32 in ame_core.c).
    bool isQwenI8F32Matmul = elemTypeA.isInteger(8) && elemTypeC.isF32();

    if (elemTypeC.isInteger(32) || isQwenI8F32Matmul)
      accMtypeImm = mtypeVal(2, 6);  // 0x10042: mma=1, mint32=1, msew=2
    else if (elemTypeC.isF32())
      accMtypeImm = mtypeVal(2, 11); // 0x10802: mma=1, mf32=1, msew=2
    else if (elemTypeC.isInteger(16))
      accMtypeImm = mtypeVal(1, 5);  // mma=1, mint16=1, msew=1
    else if (elemTypeC.isInteger(64))
      accMtypeImm = mtypeVal(3, 7);  // mma=1, mint64=1, msew=3
    else if (elemTypeC.isF64())
      accMtypeImm = mtypeVal(3, 12); // mma=1, mf64=1, msew=3
    else
      accMtypeImm = mtypeVal(2, 6);  // fallback: INT32

    // [1] Qwen3 FPGA AME int8 path: mqma accumulates i8*i8 and msce32
    // writes fp32 bits to C.  Reject i32 C for this path because the hardware
    // behavior observed on FPGA and in Qwen3 kernels is i8*i8->f32.
    if (isQwenI8F32Matmul) {
      tileK = 16;
      mmaMtypeImm = mtypeVal(0, 4); // 0x10010: mma=1, mint8=1, msew=0
    }
    // [2] (f16/bf16 * f16/bf16 -> f32)
    else if ((elemTypeA.isF16() || elemTypeA.isBF16()) && elemTypeC.isF32()) {
      tileK = 8;
      if (elemTypeA.isF16())
        mmaMtypeImm = mtypeVal(1, 9);  // mma=1, mf16=1,  msew=1
      else
        mmaMtypeImm = mtypeVal(1, 10); // mma=1, mbf16=1, msew=1
    }
    // [3] (i16 * i16 -> i32)
    else if (elemTypeA.isInteger(16) && elemTypeC.isInteger(32)) {
      tileK = 8;
      mmaMtypeImm = mtypeVal(1, 5); // mma=1, mint16=1, msew=1
    }
    // [4] (f32 * f32 -> f32)
    else if (elemTypeA.isF32() && elemTypeC.isF32()) {
      tileK = 4;
      mmaMtypeImm = mtypeVal(2, 11); // mma=1, mf32=1, msew=2
    }
    // [5] (i32 * i32 -> i32) — confirmed against Qwen3 AME_MTYPE_INT32
    else if (elemTypeA.isInteger(32) && elemTypeC.isInteger(32)) {
      tileK = 4;
      mmaMtypeImm = mtypeVal(2, 6); // 0x10042: mma=1, mint32=1, msew=2
    }
    // [6] (f64 * f64 -> f64)
    else if (elemTypeA.isF64() && elemTypeC.isF64()) {
      tileK = 2;
      mmaMtypeImm = mtypeVal(3, 12); // mma=1, mf64=1, msew=3
    }
    // [7] (i4 * i4 -> i32)
    else if (elemTypeA.isInteger(4) && elemTypeC.isInteger(32)) {
      tileK = 32;
      mmaMtypeImm = mtypeVal(0, 8); // mma=1, mint4=1, msew=0
    } else if (elemTypeA.isInteger(8) && elemTypeC.isInteger(32)) {
      return rewriter.notifyMatchFailure(
          op, "Qwen3 FPGA AME i8 matmul stores fp32 bits; use f32 C.");
    } else {
      return rewriter.notifyMatchFailure(
          op, "Unsupported mixed-precision combination.");
    }

    Value dimM = memref::DimOp::create(rewriter, loc, A, 0);
    Value dimK = memref::DimOp::create(rewriter, loc, A, 1);
    Value dimN = memref::DimOp::create(rewriter, loc, B, 1);

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value stepM = arith::ConstantIndexOp::create(rewriter, loc, tileM);
    Value stepK = arith::ConstantIndexOp::create(rewriter, loc, tileK);
    Value stepN = arith::ConstantIndexOp::create(rewriter, loc, tileN);

    auto loopM = scf::ForOp::create(rewriter, loc, c0, dimM, stepM);
    rewriter.setInsertionPointToStart(loopM.getBody());
    Value ivM = loopM.getInductionVar();

    auto loopN = scf::ForOp::create(rewriter, loc, c0, dimN, stepN);
    rewriter.setInsertionPointToStart(loopN.getBody());
    Value ivN = loopN.getInductionVar();

    auto calcCurrentSize = [&](Value bound, Value iv, int64_t step) {
      Value remain = arith::SubIOp::create(rewriter, loc, bound, iv);
      Value stepVal = arith::ConstantIndexOp::create(rewriter, loc, step);
      Value cmp = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::slt, remain, stepVal);
      return arith::SelectOp::create(rewriter, loc, cmp, remain, stepVal);
    };

    Value currM = calcCurrentSize(dimM, ivM, tileM);
    Value currN = calcCurrentSize(dimN, ivN, tileN);

    Value currMI64 =
        arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), currM);
    Value currNI64 =
        arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), currN);

    SmallVector<OpFoldResult> stridesAttr = {rewriter.getIndexAttr(1),
                                             rewriter.getIndexAttr(1)};
    Value subC = memref::SubViewOp::create(
        rewriter, loc, C, ArrayRef<OpFoldResult>{ivM, ivN},
        ArrayRef<OpFoldResult>{currM, currN}, stridesAttr);

    auto getRowStride = [&](Value subview) -> Value {
      auto meta =
          memref::ExtractStridedMetadataOp::create(rewriter, loc, subview);
      Value strideElem = meta.getResult(4);

      auto memrefType = cast<MemRefType>(subview.getType());
      unsigned bytesPerElem = memrefType.getElementTypeBitWidth() / 8;
      Value bytesVal =
          arith::ConstantIndexOp::create(rewriter, loc, bytesPerElem);

      Value strideBytes =
          arith::MulIOp::create(rewriter, loc, strideElem, bytesVal);

      return arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(),
                                        strideBytes);
    };
    Value strideC = getRowStride(subC);

    //===----------------------------------------------------------------===//
    // The accumulator must remain resident across all K tiles:
    //
    //   1. msettilem/n + msettype(accMtype) + mlce32  (load C once)
    //   2. for each K tile: msettilek + msettype(mmaMtype) + MMA
    //   3. msettype(accMtype) + msce32                (store C once)
    //
    // On Qwen3 i8->f32 hardware, msce32 converts the integer accumulator to
    // fp32. Reloading that fp32 value with mlce32 between K tiles instead
    // treats its IEEE-754 bits as an integer accumulator value.
    //===----------------------------------------------------------------===//

    // --- Step 1: accumulator load (mtype = C element type) ---
    MSettilemOp::create(rewriter, loc, rewriter.getI64Type(), currMI64);
    MSettilenOp::create(rewriter, loc, rewriter.getI64Type(), currNI64);
    Value accMtypeVal = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(accMtypeImm));
    MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), accMtypeVal);

    // FPGA RTL only supports mlce32 for accumulator init (not msub.w.mm).
    if (isQwenI8F32Matmul) {
      Mlce32mOp::create(rewriter, loc, 0, subC, strideC);
    } else if (elemTypeC.isInteger(32)) {
      MsubWMmOp::create(rewriter, loc, 0, 0, 0);
    } else if (elemTypeC.isInteger(16)) {
      MsubHMmOp::create(rewriter, loc, 0, 0, 0);
    } else if (elemTypeC.isInteger(64)) {
      MsubDwMmOp::create(rewriter, loc, 0, 0, 0);
    }

    auto loopK = scf::ForOp::create(rewriter, loc, c0, dimK, stepK);
    rewriter.setInsertionPointToStart(loopK.getBody());
    Value ivK = loopK.getInductionVar();
    Value currK = calcCurrentSize(dimK, ivK, tileK);
    Value currKI64 =
        arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), currK);

    Value subA = memref::SubViewOp::create(
        rewriter, loc, A, ArrayRef<OpFoldResult>{ivM, ivK},
        ArrayRef<OpFoldResult>{currM, currK}, stridesAttr);
    Value subB = memref::SubViewOp::create(
        rewriter, loc, B, ArrayRef<OpFoldResult>{ivK, ivN},
        ArrayRef<OpFoldResult>{currK, currN}, stridesAttr);
    Value strideA = getRowStride(subA);
    Value strideB = getRowStride(subB);

    // --- Step 2: MMA for one K tile (mtype = A/B element type) ---
    MSettilekOp::create(rewriter, loc, rewriter.getI64Type(), currKI64);
    Value mmaMtypeVal = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(mmaMtypeImm));
    MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), mmaMtypeVal);

    if (elemTypeA.isInteger(8)) {
      Mlae8mOp::create(rewriter, loc, 0, subA, strideA);
      Mlbte8mOp::create(rewriter, loc, 1, subB, strideB);
    } else if (elemTypeA.isF16() || elemTypeA.isBF16() ||
               elemTypeA.isInteger(16)) {
      Mlae16mOp::create(rewriter, loc, 0, subA, strideA);
      Mlbe16mOp::create(rewriter, loc, 1, subB, strideB);
    } else if (elemTypeA.isInteger(32) || elemTypeA.isF32()) {
      Mlae32mOp::create(rewriter, loc, 0, subA, strideA);
      Mlbe32mOp::create(rewriter, loc, 1, subB, strideB);
    } else if (elemTypeA.isInteger(64) || elemTypeA.isF64()) {
      Mlae64mOp::create(rewriter, loc, 0, subA, strideA);
      Mlbe64mOp::create(rewriter, loc, 1, subB, strideB);
    }

    if (isQwenI8F32Matmul) {
      MqmaBmmOp::create(rewriter, loc, 0, 0, 1);
    } else if (elemTypeC.isInteger(32) && elemTypeA.isInteger(32)) {
      MmaWmmOp::create(rewriter, loc, 0, 0, 1);
    } else if (elemTypeC.isInteger(16) && elemTypeA.isInteger(16)) {
      MmaHmmOp::create(rewriter, loc, 0, 0, 1);
    } else if (elemTypeC.isInteger(64) && elemTypeA.isInteger(64)) {
      MmaDwmmOp::create(rewriter, loc, 0, 0, 1);
    }

    rewriter.setInsertionPointAfter(loopK);

    // --- Step 3: accumulator store after all K tiles (mtype = C type) ---
    Value accMtypeVal2 = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(accMtypeImm));
    MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), accMtypeVal2);

    if (elemTypeC.isInteger(32) || elemTypeC.isF32()) {
      Msce32mOp::create(rewriter, loc, 0, subC, strideC);
    } else if (elemTypeC.isInteger(64) || elemTypeC.isF64()) {
      Msce64mOp::create(rewriter, loc, 0, subC, strideC);
    } else if (elemTypeC.isInteger(16) || elemTypeC.isF16()) {
      Msce16mOp::create(rewriter, loc, 0, subC, strideC);
    }

    rewriter.setInsertionPointAfter(loopM);
    rewriter.eraseOp(op);
    if (copyToFinalOutput) {
      rewriter.eraseOp(copyToFinalOutput);
      for (linalg::FillOp fillOp : deadInitFills)
        rewriter.eraseOp(fillOp);
      if (Operation *def = originalC.getDefiningOp(); def && def->use_empty())
        rewriter.eraseOp(def);
    }

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
class LowerLinalgToBOSCAMEPass
    : public PassWrapper<LowerLinalgToBOSCAMEPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLinalgToBOSCAMEPass)

  StringRef getArgument() const final { return "lower-linalg-to-boscame"; }
  StringRef getDescription() const final {
    return "Lower linalg dialect operations to BOSCAME dialect operations.";
  }

  LowerLinalgToBOSCAMEPass() = default;
  LowerLinalgToBOSCAMEPass(const LowerLinalgToBOSCAMEPass &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<BOSCAMEDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void LowerLinalgToBOSCAMEPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  RewritePatternSet patterns(context);
  patterns.add<MatmulToBOSCAMELowering>(context);

  ConversionTarget target(*context);
  target.addLegalDialect<BOSCAMEDialect, arith::ArithDialect,
                         memref::MemRefDialect, scf::SCFDialect>();
  target.addIllegalOp<linalg::MatmulOp>();

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }

  SmallVector<memref::AllocOp> deadAllocs;
  module.walk([&](memref::AllocOp allocOp) {
    if (allocOp->use_empty())
      deadAllocs.push_back(allocOp);
  });
  for (memref::AllocOp allocOp : deadAllocs)
    allocOp.erase();
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace buddy {
void registerLowerLinalgToBOSCAMEPass() {
  PassRegistration<LowerLinalgToBOSCAMEPass>();
}
} // namespace buddy
} // namespace mlir
