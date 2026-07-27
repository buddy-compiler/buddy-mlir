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
#include "mlir/IR/Matchers.h"
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

static bool hasSupported2DLayout(MemRefType type) {
  SmallVector<int64_t> strides;
  int64_t offset;
  return succeeded(type.getStridesAndOffset(strides, offset)) &&
         strides.size() == 2 && strides.front() > 0 && strides.back() == 1;
}

struct QwenDirectCMatch {
  memref::AllocOp temporaryAlloc;
  linalg::FillOp zeroFill;
  memref::CopyOp copyToFinal;
  memref::DeallocOp temporaryDealloc;
  Value finalOutput;
};

// Match only the bufferization shape emitted for the Qwen3 Triton matmul:
//
//   %tmp = memref.alloc
//   linalg.fill 0 -> %tmp
//   linalg.matmul A, B -> %tmp
//   <pure address calculation for %final>
//   memref.copy %tmp, %final
//   memref.dealloc %tmp  // optional
//
// The strict use and ordering checks are what make it safe to compute directly
// into %final. Any less constrained matmul remains legal and is handled by the
// later linalg-to-VIR pipeline.
static FailureOr<QwenDirectCMatch> matchQwenDirectCMatmul(linalg::MatmulOp op) {
  if (!op.hasPureBufferSemantics() || op.getNumDpsInputs() != 2 ||
      op.getNumDpsInits() != 1)
    return failure();

  Value A = op.getDpsInputOperand(0)->get();
  Value B = op.getDpsInputOperand(1)->get();
  Value temporaryC = op.getDpsInitOperand(0)->get();

  auto AType = dyn_cast<MemRefType>(A.getType());
  auto BType = dyn_cast<MemRefType>(B.getType());
  auto temporaryCType = dyn_cast<MemRefType>(temporaryC.getType());
  if (!AType || !BType || !temporaryCType)
    return failure();

  if (AType.getRank() != 2 || BType.getRank() != 2 ||
      temporaryCType.getRank() != 2)
    return failure();

  if (!AType.hasStaticShape() || !BType.hasStaticShape() ||
      !temporaryCType.hasStaticShape())
    return failure();

  int64_t dimM = AType.getDimSize(0);
  int64_t dimK = AType.getDimSize(1);
  int64_t dimN = BType.getDimSize(1);
  if (dimM <= 0 || dimN <= 0 || dimK <= 0 || dimK % 16 != 0 ||
      BType.getDimSize(0) != dimK || temporaryCType.getDimSize(0) != dimM ||
      temporaryCType.getDimSize(1) != dimN)
    return failure();

  if (!AType.getElementType().isSignlessInteger(8) ||
      !BType.getElementType().isSignlessInteger(8) ||
      !temporaryCType.getElementType().isF32())
    return failure();

  if (op.getCast() != linalg::TypeFn::cast_signed || op.hasUserDefinedMaps())
    return failure();

  if (!hasSupported2DLayout(AType) || !hasSupported2DLayout(BType) ||
      !hasSupported2DLayout(temporaryCType))
    return failure();

  auto temporaryAlloc = temporaryC.getDefiningOp<memref::AllocOp>();
  if (!temporaryAlloc || temporaryAlloc->getBlock() != op->getBlock() ||
      !temporaryAlloc->isBeforeInBlock(op))
    return failure();

  linalg::FillOp zeroFill;
  memref::CopyOp copyToFinal;
  memref::DeallocOp temporaryDealloc;
  bool sawMatmul = false;

  for (Operation *user : temporaryC.getUsers()) {
    if (user == op.getOperation()) {
      if (sawMatmul)
        return failure();
      sawMatmul = true;
      continue;
    }

    if (auto fillOp = dyn_cast<linalg::FillOp>(user)) {
      if (zeroFill || fillOp.getDpsInitOperand(0)->get() != temporaryC ||
          !matchPattern(fillOp.getDpsInputOperand(0)->get(), m_PosZeroFloat()))
        return failure();
      zeroFill = fillOp;
      continue;
    }

    if (auto copyOp = dyn_cast<memref::CopyOp>(user)) {
      if (copyToFinal || copyOp.getSource() != temporaryC)
        return failure();
      copyToFinal = copyOp;
      continue;
    }

    if (auto deallocOp = dyn_cast<memref::DeallocOp>(user)) {
      if (temporaryDealloc || deallocOp.getMemref() != temporaryC)
        return failure();
      temporaryDealloc = deallocOp;
      continue;
    }

    return failure();
  }

  if (!sawMatmul || !zeroFill || !copyToFinal)
    return failure();
  if (zeroFill->getBlock() != op->getBlock() ||
      copyToFinal->getBlock() != op->getBlock() ||
      zeroFill->getNextNode() != op.getOperation() ||
      !op->isBeforeInBlock(copyToFinal))
    return failure();

  if (temporaryDealloc && (temporaryDealloc->getBlock() != op->getBlock() ||
                           !copyToFinal->isBeforeInBlock(temporaryDealloc)))
    return failure();

  Value finalOutput = copyToFinal.getTarget();
  auto finalType = dyn_cast<MemRefType>(finalOutput.getType());
  if (!finalType || finalOutput == temporaryC || !finalOutput.hasOneUse() ||
      *finalOutput.getUsers().begin() != copyToFinal.getOperation())
    return failure();

  if (finalType.getRank() != temporaryCType.getRank() ||
      finalType.getShape() != temporaryCType.getShape() ||
      finalType.getElementType() != temporaryCType.getElementType() ||
      finalType.getMemorySpace() != temporaryCType.getMemorySpace() ||
      !hasSupported2DLayout(finalType))
    return failure();

  // Recompute at the original copy point instead of moving %final's defining
  // operations earlier. This preserves the time at which final C becomes
  // visible. Requiring the intervening operations to be pure guarantees that
  // A, B, and all memory are unchanged between the old matmul and copy.
  for (Operation *between = op->getNextNode(); between != copyToFinal;
       between = between->getNextNode()) {
    if (!between || between->getNumRegions() != 0 || !isPure(between))
      return failure();
  }

  return QwenDirectCMatch{temporaryAlloc, zeroFill, copyToFinal,
                          temporaryDealloc, finalOutput};
}

class MatmulToBOSCAMELowering : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    FailureOr<QwenDirectCMatch> directC = matchQwenDirectCMatmul(op);
    if (failed(directC))
      return rewriter.notifyMatchFailure(
          op, "matmul is not a safe Qwen3 FPGA AME direct-C operation");

    Value A = op.getDpsInputOperand(0)->get();
    Value B = op.getDpsInputOperand(1)->get();
    Value C = directC->finalOutput;

    // Insert the replacement at the old copy point. The final output and its
    // address calculation already dominate this location, and final C is not
    // made visible earlier than in the original program.
    rewriter.setInsertionPoint(directC->copyToFinal);
    Value zero = directC->zeroFill.getDpsInputOperand(0)->get();
    linalg::FillOp::create(rewriter, directC->zeroFill.getLoc(), zero, C);

    constexpr int64_t tileM = 4;
    constexpr int64_t tileN = 4;
    constexpr int64_t tileK = 16;

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
    auto mtypeVal = [](unsigned msew, unsigned typeBit) -> int64_t {
      return (1LL << 16) | (1LL << typeBit) | msew;
    };

    const int64_t mmaMtypeImm =
        mtypeVal(0, 4); // 0x10010: mma=1, mint8=1, msew=0
    const int64_t accMtypeImm =
        mtypeVal(2, 6); // 0x10042: mma=1, mint32=1, msew=2

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
    Value accMtypeVal =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                  rewriter.getI64IntegerAttr(accMtypeImm));
    MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), accMtypeVal);

    // FPGA RTL only supports mlce32 for accumulator init (not msub.w.mm).
    Mlce32mOp::create(rewriter, loc, 0, subC, strideC);

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
    Value mmaMtypeVal =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                  rewriter.getI64IntegerAttr(mmaMtypeImm));
    MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), mmaMtypeVal);

    Mlae8mOp::create(rewriter, loc, 0, subA, strideA);
    Mlbte8mOp::create(rewriter, loc, 1, subB, strideB);
    MqmaBmmOp::create(rewriter, loc, 0, 0, 1);

    rewriter.setInsertionPointAfter(loopK);

    // --- Step 3: accumulator store after all K tiles (mtype = C type) ---
    Value accMtypeVal2 =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                  rewriter.getI64IntegerAttr(accMtypeImm));
    MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), accMtypeVal2);

    Msce32mOp::create(rewriter, loc, 0, subC, strideC);

    rewriter.setInsertionPointAfter(loopM);
    if (directC->temporaryDealloc)
      rewriter.eraseOp(directC->temporaryDealloc);
    rewriter.eraseOp(directC->copyToFinal);
    rewriter.eraseOp(op);
    rewriter.eraseOp(directC->zeroFill);
    rewriter.eraseOp(directC->temporaryAlloc);

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
                         linalg::LinalgDialect, memref::MemRefDialect,
                         scf::SCFDialect>();
  target.addDynamicallyLegalOp<linalg::MatmulOp>(
      [](linalg::MatmulOp op) { return failed(matchQwenDirectCMatmul(op)); });

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
