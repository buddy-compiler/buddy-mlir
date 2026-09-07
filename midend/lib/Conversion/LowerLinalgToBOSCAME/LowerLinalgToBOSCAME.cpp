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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

static bool hasSupportedRowMajor2DLayout(MemRefType type) {
  SmallVector<int64_t> strides;
  int64_t offset;
  return succeeded(type.getStridesAndOffset(strides, offset)) &&
         strides.size() == 2 &&
         (ShapedType::isDynamic(strides.front()) || strides.front() > 0) &&
         (ShapedType::isDynamic(strides.back()) || strides.back() == 1);
}

// Triton materializes a logical [K, N] tile loaded from a row-major [N, full-K]
// weight as a zero-copy memref with strides [1, full-K]. The AME mlbe path can
// consume that physical [N, full-K] layout directly.
static bool hasSupportedTransposed2DLayout(MemRefType type) {
  SmallVector<int64_t> strides;
  int64_t offset;
  return succeeded(type.getStridesAndOffset(strides, offset)) &&
         strides.size() == 2 && strides.front() == 1 && strides.back() > 0;
}

static bool hasSupportedBLayout(MemRefType type) {
  return hasSupportedRowMajor2DLayout(type) ||
         hasSupportedTransposed2DLayout(type);
}

static constexpr StringLiteral kTritonConsumerFenceAttr =
    "bosc_ame.triton_consumer_fence";

struct QwenDirectCMatch {
  memref::AllocOp temporaryAlloc;
  linalg::FillOp zeroFill;
  memref::CopyOp copyToFinal;
  memref::DeallocOp temporaryDealloc;
  Value finalOutput;
};

struct TritonI8DotCastMatch {
  memref::AllocOp integerAlloc;
  Operation *integerInitialization;
  linalg::GenericOp castToF32;
  memref::AllocOp floatAlloc;
  memref::AllocOp zeroTemplateAlloc;
  linalg::FillOp zeroTemplateFill;
};

// Triton defines an integer dot as i8 x i8 -> i32 even when its result is
// immediately converted to fp32.  The FPGA AME instead keeps the exact i32
// accumulator internally and converts it to fp32 at msce32.m.  For K <= 1024
// the worst-case signed-i8 dot does not exceed 2^24, so this fusion is exact.
static FailureOr<TritonI8DotCastMatch>
matchTritonI8DotCast(linalg::MatmulOp op) {
  if (!op.hasPureBufferSemantics() || op.getNumDpsInputs() != 2 ||
      op.getNumDpsInits() != 1 || op.getCast() != linalg::TypeFn::cast_signed ||
      op.hasUserDefinedMaps())
    return failure();

  Value A = op.getDpsInputOperand(0)->get();
  Value B = op.getDpsInputOperand(1)->get();
  Value integerC = op.getDpsInitOperand(0)->get();
  auto AType = dyn_cast<MemRefType>(A.getType());
  auto BType = dyn_cast<MemRefType>(B.getType());
  auto integerCType = dyn_cast<MemRefType>(integerC.getType());
  if (!AType || !BType || !integerCType || AType.getRank() != 2 ||
      BType.getRank() != 2 || integerCType.getRank() != 2 ||
      !AType.hasStaticShape() || !BType.hasStaticShape() ||
      !integerCType.hasStaticShape())
    return failure();

  int64_t dimM = AType.getDimSize(0);
  int64_t dimK = AType.getDimSize(1);
  int64_t dimN = BType.getDimSize(1);
  if (dimM <= 0 || dimN <= 0 || dimK <= 0 || dimK % 16 != 0 || dimK > 1024 ||
      BType.getDimSize(0) != dimK ||
      integerCType.getShape() != ArrayRef<int64_t>({dimM, dimN}) ||
      !AType.getElementType().isSignlessInteger(8) ||
      !BType.getElementType().isSignlessInteger(8) ||
      !integerCType.getElementType().isSignlessInteger(32) ||
      !hasSupportedRowMajor2DLayout(AType) || !hasSupportedBLayout(BType))
    return failure();

  auto integerAlloc = integerC.getDefiningOp<memref::AllocOp>();
  if (!integerAlloc || integerAlloc->getBlock() != op->getBlock() ||
      !integerAlloc->isBeforeInBlock(op))
    return failure();

  Operation *integerInitialization = nullptr;
  linalg::GenericOp castToF32;
  memref::AllocOp zeroTemplateAlloc;
  linalg::FillOp zeroTemplateFill;
  for (Operation *user : integerC.getUsers()) {
    if (user == op.getOperation())
      continue;
    if (auto fill = dyn_cast<linalg::FillOp>(user)) {
      if (integerInitialization ||
          fill.getDpsInitOperand(0)->get() != integerC ||
          !matchPattern(fill.getDpsInputOperand(0)->get(), m_Zero()))
        return failure();
      integerInitialization = fill;
      continue;
    }
    if (auto copy = dyn_cast<memref::CopyOp>(user)) {
      // A direct zero buffer may also seed earlier group-dot destinations.
      // Those are reads and do not change the value seen by this matmul.
      if (copy.getSource() == integerC) {
        if (copy->getBlock() != op->getBlock() || !copy->isBeforeInBlock(op))
          return failure();
        continue;
      }
      if (integerInitialization || copy.getTarget() != integerC)
        return failure();

      Value zeroTemplate = copy.getSource();
      auto zeroTemplateType = dyn_cast<MemRefType>(zeroTemplate.getType());
      zeroTemplateAlloc = zeroTemplate.getDefiningOp<memref::AllocOp>();
      if (!zeroTemplateType || zeroTemplateType != integerCType ||
          !zeroTemplateAlloc || zeroTemplateAlloc->getBlock() != op->getBlock())
        return failure();

      // Multiple static Triton group dots reuse one zero tensor. Bufferization
      // copies it into fresh destinations and may use the template itself for
      // the last dot. Only users before this copy matter: they must initialize
      // the template or read it through another copy, never modify it.
      for (Operation *templateUser : zeroTemplate.getUsers()) {
        if (auto templateFill = dyn_cast<linalg::FillOp>(templateUser)) {
          if (zeroTemplateFill ||
              templateFill.getDpsInitOperand(0)->get() != zeroTemplate ||
              !matchPattern(templateFill.getDpsInputOperand(0)->get(),
                            m_Zero()))
            return failure();
          zeroTemplateFill = templateFill;
          continue;
        }
        if (templateUser == copy.getOperation())
          continue;
        if (templateUser->getBlock() != copy->getBlock())
          return failure();
        if (copy->isBeforeInBlock(templateUser))
          continue;
        auto earlierCopy = dyn_cast<memref::CopyOp>(templateUser);
        if (!earlierCopy || earlierCopy.getSource() != zeroTemplate)
          return failure();
      }
      if (!zeroTemplateFill ||
          !zeroTemplateAlloc->isBeforeInBlock(zeroTemplateFill) ||
          !zeroTemplateFill->isBeforeInBlock(copy))
        return failure();
      integerInitialization = copy;
      continue;
    }
    if (auto generic = dyn_cast<linalg::GenericOp>(user)) {
      if (castToF32 || generic.getNumDpsInputs() != 1 ||
          generic.getNumDpsInits() != 1 ||
          generic.getDpsInputOperand(0)->get() != integerC)
        return failure();
      castToF32 = generic;
      continue;
    }
    return failure();
  }

  if (!integerInitialization || !castToF32 ||
      castToF32->getBlock() != op->getBlock() ||
      !op->isBeforeInBlock(castToF32))
    return failure();
  if (isa<memref::CopyOp>(integerInitialization)) {
    if (integerAlloc->getNextNode() != integerInitialization ||
        integerInitialization->getNextNode() != op.getOperation())
      return failure();
  } else if (!integerAlloc->isBeforeInBlock(integerInitialization) ||
             !integerInitialization->isBeforeInBlock(op)) {
    return failure();
  }

  Value floatC = castToF32.getDpsInitOperand(0)->get();
  auto floatCType = dyn_cast<MemRefType>(floatC.getType());
  auto floatAlloc = floatC.getDefiningOp<memref::AllocOp>();
  if (!floatCType || !floatAlloc || floatCType.getRank() != 2 ||
      floatCType.getShape() != integerCType.getShape() ||
      !floatCType.getElementType().isF32() ||
      !hasSupportedRowMajor2DLayout(floatCType) ||
      floatAlloc->getBlock() != op->getBlock() ||
      !floatAlloc->isBeforeInBlock(castToF32))
    return failure();

  for (Operation *user : floatC.getUsers()) {
    if (user == castToF32.getOperation())
      continue;
    if (user->getBlock() != op->getBlock() || !castToF32->isBeforeInBlock(user))
      return failure();
  }

  if (castToF32.getNumLoops() != 2 || castToF32.getNumParallelLoops() != 2)
    return failure();
  for (AffineMap map : castToF32.getIndexingMapsArray())
    if (!map.isIdentity())
      return failure();

  Block &body = castToF32.getRegion().front();
  if (body.getNumArguments() != 2)
    return failure();
  auto cast = dyn_cast<arith::SIToFPOp>(body.front());
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!cast || !yield || cast->getNextNode() != yield.getOperation() ||
      cast.getIn() != body.getArgument(0) || yield.getNumOperands() != 1 ||
      yield.getValues().front() != cast.getOut())
    return failure();

  return TritonI8DotCastMatch{integerAlloc,      integerInitialization,
                              castToF32,         floatAlloc,
                              zeroTemplateAlloc, zeroTemplateFill};
}

// Match the bufferization shapes emitted for Qwen3 matmuls:
//
//   %tmp = memref.alloc
//   linalg.fill 0 -> %tmp
//   linalg.matmul A, B -> %tmp
//   <pure address calculation for %final>
//   memref.copy %tmp, %final  // optional; Buddy Frontend uses %tmp directly
//   memref.dealloc %tmp  // optional
//
// The strict use and ordering checks are what make it safe to compute directly
// into either %final or the original destination. Any less constrained matmul
// remains legal and is handled by the later linalg-to-VIR pipeline.
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

  // Dynamic strides occur on Buddy Frontend function arguments.  The Qwen3
  // generated wrapper/bare-metal runner provides tight row-major buffers;
  // getRowStride below still reads the runtime leading stride.
  if (!hasSupportedRowMajor2DLayout(AType) || !hasSupportedBLayout(BType) ||
      !hasSupportedRowMajor2DLayout(temporaryCType))
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
      // Triton accumulates all quantization groups in the first dot buffer
      // and copies that fully scaled result to the ABI output much later.
      // That copy is a downstream read, not the direct-C copy shape below.
      if (op->hasAttr(kTritonConsumerFenceAttr)) {
        if (copyOp.getSource() != temporaryC ||
            copyOp->getBlock() != op->getBlock() ||
            !op->isBeforeInBlock(copyOp))
          return failure();
        continue;
      }
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

    // Buddy Frontend keeps the matmul destination as the SSA buffer consumed
    // by following elementwise operations.  Such uses are safe when they are
    // in the same block and occur after the matmul.
    if (user->getBlock() != op->getBlock() || !op->isBeforeInBlock(user))
      return failure();
  }

  if (!sawMatmul || !zeroFill)
    return failure();
  if (zeroFill->getBlock() != op->getBlock() ||
      zeroFill->getNextNode() != op.getOperation())
    return failure();

  if (temporaryDealloc && (temporaryDealloc->getBlock() != op->getBlock() ||
                           !op->isBeforeInBlock(temporaryDealloc)))
    return failure();

  if (!copyToFinal)
    return QwenDirectCMatch{temporaryAlloc, zeroFill, copyToFinal,
                            temporaryDealloc, temporaryC};

  if (copyToFinal->getBlock() != op->getBlock() ||
      !op->isBeforeInBlock(copyToFinal))
    return failure();

  if (temporaryDealloc && !copyToFinal->isBeforeInBlock(temporaryDealloc))
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
      !hasSupportedRowMajor2DLayout(finalType))
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

class TritonI8DotCastToF32Matmul : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<TritonI8DotCastMatch> match = matchTritonI8DotCast(op);
    if (failed(match))
      return failure();

    Value A = op.getDpsInputOperand(0)->get();
    Value B = op.getDpsInputOperand(1)->get();
    Value oldFloatC = match->castToF32.getDpsInitOperand(0)->get();
    auto floatType = cast<MemRefType>(oldFloatC.getType());
    Location loc = op.getLoc();

    // Preserve the original dot's execution point and redirect all consumers
    // of the later sitofp buffer to this exact-fp32 AME destination.
    rewriter.setInsertionPoint(op);
    Value floatC = memref::AllocOp::create(rewriter, loc, floatType);
    Value zero = arith::ConstantFloatOp::create(
        rewriter, loc, rewriter.getF32Type(), APFloat(0.0f));
    linalg::FillOp::create(rewriter, loc, zero, floatC);
    auto floatMatmul = linalg::MatmulOp::create(rewriter, loc, ValueRange{A, B},
                                                ValueRange{floatC});
    floatMatmul.setCast(linalg::TypeFn::cast_signed);
    floatMatmul->setAttr(kTritonConsumerFenceAttr, rewriter.getUnitAttr());

    oldFloatC.replaceAllUsesWith(floatC);
    rewriter.eraseOp(match->castToF32);
    rewriter.eraseOp(match->floatAlloc);
    rewriter.eraseOp(op);

    auto eraseZeroBufferIfDead = [&](memref::AllocOp alloc,
                                     linalg::FillOp fill) {
      if (alloc && fill && alloc.getResult().hasOneUse() &&
          *alloc.getResult().getUsers().begin() == fill.getOperation()) {
        rewriter.eraseOp(fill);
        rewriter.eraseOp(alloc);
      }
    };
    if (isa<memref::CopyOp>(match->integerInitialization)) {
      rewriter.eraseOp(match->integerInitialization);
      rewriter.eraseOp(match->integerAlloc);
      eraseZeroBufferIfDead(match->zeroTemplateAlloc, match->zeroTemplateFill);
    } else {
      eraseZeroBufferIfDead(match->integerAlloc,
                            cast<linalg::FillOp>(match->integerInitialization));
    }
    return success();
  }
};

class MatmulToBOSCAMELowering : public OpRewritePattern<linalg::MatmulOp> {
public:
  MatmulToBOSCAMELowering(MLIRContext *context, bool tritonW8A8FastPath)
      : OpRewritePattern<linalg::MatmulOp>(context),
        tritonW8A8FastPath(tritonW8A8FastPath) {}

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
    auto AType = cast<MemRefType>(A.getType());
    auto BType = cast<MemRefType>(B.getType());
    const bool transposedB = !hasSupportedRowMajor2DLayout(BType) &&
                             hasSupportedTransposed2DLayout(BType);

    const bool copiedDestination = static_cast<bool>(directC->copyToFinal);
    if (copiedDestination) {
      // Insert the replacement at the old copy point. The final output and its
      // address calculation already dominate this location, and final C is not
      // made visible earlier than in the original program.
      rewriter.setInsertionPoint(directC->copyToFinal);
      Value zero = directC->zeroFill.getDpsInputOperand(0)->get();
      linalg::FillOp::create(rewriter, directC->zeroFill.getLoc(), zero, C);
    } else {
      // Buddy Frontend bufferization already exposes the destination buffer to
      // downstream operations, so replace the matmul in place and retain the
      // original allocation/fill/lifetime.
      rewriter.setInsertionPoint(op);
    }

    constexpr int64_t tileM = 16;
    constexpr int64_t tileN = 16;
    constexpr int64_t tileK = 64;

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

    auto calcCurrentSize = [&](Value bound, Value iv, int64_t step) {
      Value remain = arith::SubIOp::create(rewriter, loc, bound, iv);
      Value stepVal = arith::ConstantIndexOp::create(rewriter, loc, step);
      Value cmp = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::slt, remain, stepVal);
      return arith::SelectOp::create(rewriter, loc, cmp, remain, stepVal);
    };

    SmallVector<OpFoldResult> stridesAttr = {rewriter.getIndexAttr(1),
                                             rewriter.getIndexAttr(1)};
    auto getStrideBytes = [&](Value buffer, unsigned dimension) -> Value {
      auto meta =
          memref::ExtractStridedMetadataOp::create(rewriter, loc, buffer);
      Value strideElem = meta.getResult(4 + dimension);

      auto memrefType = cast<MemRefType>(buffer.getType());
      unsigned bytesPerElem = memrefType.getElementTypeBitWidth() / 8;
      Value bytesVal =
          arith::ConstantIndexOp::create(rewriter, loc, bytesPerElem);

      Value strideBytes =
          arith::MulIOp::create(rewriter, loc, strideElem, bytesVal);

      return arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(),
                                        strideBytes);
    };

    auto makeSubview = [&](Value source, ArrayRef<OpFoldResult> offsets,
                           ArrayRef<OpFoldResult> sizes) -> Value {
      return memref::SubViewOp::create(rewriter, loc, source, offsets, sizes,
                                       stridesAttr);
    };

    auto finishReplacement = [&]() {
      rewriter.eraseOp(op);
      if (copiedDestination) {
        if (directC->temporaryDealloc)
          rewriter.eraseOp(directC->temporaryDealloc);
        rewriter.eraseOp(directC->copyToFinal);
        rewriter.eraseOp(directC->zeroFill);
        rewriter.eraseOp(directC->temporaryAlloc);
      }
    };

    const int64_t staticM = AType.getDimSize(0);
    const int64_t staticK = AType.getDimSize(1);
    const int64_t staticN = BType.getDimSize(1);
    const bool use2A4B = tritonW8A8FastPath && transposedB && staticM == 32 &&
                         staticN % 64 == 0 && staticK % 64 == 0;
    const bool useDecodeWide = tritonW8A8FastPath && transposedB &&
                               staticM > 0 && staticM <= 16 &&
                               staticN % 64 == 0 && staticK % 64 == 0;

    if (use2A4B || useDecodeWide) {
      Value strideA = getStrideBytes(A, 0);
      // Logical B is [K, N], but the Triton view is physically [N, K].
      Value strideB = getStrideBytes(B, 1);
      Value strideC = getStrideBytes(C, 0);
      Value tileMValue = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI64Type(),
          rewriter.getI64IntegerAttr(use2A4B ? 16 : staticM));
      Value tileNValue = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(16));
      Value tileKValue = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(64));
      Value accMtype =
          arith::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                    rewriter.getI64IntegerAttr(accMtypeImm));
      Value mmaMtype =
          arith::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                    rewriter.getI64IntegerAttr(mmaMtypeImm));
      MSettilemOp::create(rewriter, loc, rewriter.getI64Type(), tileMValue);
      MSettilenOp::create(rewriter, loc, rewriter.getI64Type(), tileNValue);
      MSettilekOp::create(rewriter, loc, rewriter.getI64Type(), tileKValue);

      auto cTile = [&](int64_t row, int64_t column, int64_t rows) {
        return makeSubview(
            C,
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(row),
                                   rewriter.getIndexAttr(column)},
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(rows),
                                   rewriter.getIndexAttr(16)});
      };
      auto aTile = [&](int64_t row, Value k, int64_t rows) {
        return makeSubview(
            A, ArrayRef<OpFoldResult>{rewriter.getIndexAttr(row), k},
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(rows),
                                   rewriter.getIndexAttr(64)});
      };
      auto bTile = [&](Value k, int64_t column) {
        return makeSubview(
            B, ArrayRef<OpFoldResult>{k, rewriter.getIndexAttr(column)},
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(64),
                                   rewriter.getIndexAttr(16)});
      };

      auto emitFourWeights = [&](Value k, int64_t column,
                                 int64_t accumulatorBase,
                                 int64_t activationRegister) {
        Value weight0 = bTile(k, column);
        Value weight1 = bTile(k, column + 16);
        Value weight2 = bTile(k, column + 32);
        Value weight3 = bTile(k, column + 48);
        Mlbe8mOp::create(rewriter, loc, 4, weight0, strideB);
        Mlbe8mOp::create(rewriter, loc, 5, weight1, strideB);
        MqmaBmmOp::create(rewriter, loc, accumulatorBase, activationRegister,
                          4);
        Mlbe8mOp::create(rewriter, loc, 6, weight2, strideB);
        MqmaBmmOp::create(rewriter, loc, accumulatorBase + 1,
                          activationRegister, 5);
        Mlbe8mOp::create(rewriter, loc, 7, weight3, strideB);
        MqmaBmmOp::create(rewriter, loc, accumulatorBase + 2,
                          activationRegister, 6);
        MqmaBmmOp::create(rewriter, loc, accumulatorBase + 3,
                          activationRegister, 7);
      };

      if (use2A4B) {
        for (int64_t column = 0; column < staticN; column += 64) {
          MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), accMtype);
          for (int64_t lane = 0; lane < 4; ++lane) {
            Mlce32mOp::create(rewriter, loc, lane,
                              cTile(0, column + lane * 16, 16), strideC);
            Mlce32mOp::create(rewriter, loc, lane + 4,
                              cTile(16, column + lane * 16, 16), strideC);
          }

          MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), mmaMtype);
          auto kLoop = scf::ForOp::create(rewriter, loc, c0, dimK, stepK);
          rewriter.setInsertionPointToStart(kLoop.getBody());
          Value k = kLoop.getInductionVar();
          Mlae8mOp::create(rewriter, loc, 0, aTile(0, k, 16), strideA);
          Mlae8mOp::create(rewriter, loc, 2, aTile(16, k, 16), strideA);
          Value weight0 = bTile(k, column);
          Value weight1 = bTile(k, column + 16);
          Value weight2 = bTile(k, column + 32);
          Value weight3 = bTile(k, column + 48);
          Mlbe8mOp::create(rewriter, loc, 4, weight0, strideB);
          Mlbe8mOp::create(rewriter, loc, 5, weight1, strideB);
          MqmaBmmOp::create(rewriter, loc, 0, 0, 4);
          Mlbe8mOp::create(rewriter, loc, 6, weight2, strideB);
          MqmaBmmOp::create(rewriter, loc, 4, 2, 4);
          Mlbe8mOp::create(rewriter, loc, 7, weight3, strideB);
          MqmaBmmOp::create(rewriter, loc, 1, 0, 5);
          MqmaBmmOp::create(rewriter, loc, 5, 2, 5);
          MqmaBmmOp::create(rewriter, loc, 2, 0, 6);
          MqmaBmmOp::create(rewriter, loc, 6, 2, 6);
          MqmaBmmOp::create(rewriter, loc, 3, 0, 7);
          MqmaBmmOp::create(rewriter, loc, 7, 2, 7);
          rewriter.setInsertionPointAfter(kLoop);

          MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), accMtype);
          for (int64_t lane = 0; lane < 4; ++lane) {
            Msce32mOp::create(rewriter, loc, lane,
                              cTile(0, column + lane * 16, 16), strideC);
            Msce32mOp::create(rewriter, loc, lane + 4,
                              cTile(16, column + lane * 16, 16), strideC);
          }
        }
      } else {
        // Decode-style M<=16 path.  Consume N128 with acc0..acc7 and retain
        // N64 as the exact fallback for the last half-block.
        for (int64_t column = 0; column < staticN;) {
          int64_t lanes = staticN - column >= 128 ? 8 : 4;
          MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), accMtype);
          for (int64_t lane = 0; lane < lanes; ++lane)
            Mlce32mOp::create(rewriter, loc, lane,
                              cTile(0, column + lane * 16, staticM), strideC);

          MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), mmaMtype);
          auto kLoop = scf::ForOp::create(rewriter, loc, c0, dimK, stepK);
          rewriter.setInsertionPointToStart(kLoop.getBody());
          Value k = kLoop.getInductionVar();
          Mlae8mOp::create(rewriter, loc, 0, aTile(0, k, staticM), strideA);
          emitFourWeights(k, column, 0, 0);
          if (lanes == 8)
            emitFourWeights(k, column + 64, 4, 0);
          rewriter.setInsertionPointAfter(kLoop);

          MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), accMtype);
          for (int64_t lane = 0; lane < lanes; ++lane)
            Msce32mOp::create(rewriter, loc, lane,
                              cTile(0, column + lane * 16, staticM), strideC);
          column += lanes * 16;
        }
      }

      if (op->hasAttr(kTritonConsumerFenceAttr))
        LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::seq_cst);
      finishReplacement();
      return success();
    }

    auto loopM = scf::ForOp::create(rewriter, loc, c0, dimM, stepM);
    rewriter.setInsertionPointToStart(loopM.getBody());
    Value ivM = loopM.getInductionVar();

    auto loopN = scf::ForOp::create(rewriter, loc, c0, dimN, stepN);
    rewriter.setInsertionPointToStart(loopN.getBody());
    Value ivN = loopN.getInductionVar();

    Value currM = calcCurrentSize(dimM, ivM, tileM);
    Value currN = calcCurrentSize(dimN, ivN, tileN);

    Value currMI64 =
        arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), currM);
    Value currNI64 =
        arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), currN);

    Value subC = makeSubview(C, ArrayRef<OpFoldResult>{ivM, ivN},
                             ArrayRef<OpFoldResult>{currM, currN});
    Value strideC = getStrideBytes(subC, 0);

    //===----------------------------------------------------------------===//
    // The accumulator must remain resident across all K tiles:
    //
    //   1. msettilem/n + msettype(accMtype) + mlce32  (load C once)
    //   2. msettype(mmaMtype), then for each K tile: msettilek + MMA
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

    Value mmaMtypeVal =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                  rewriter.getI64IntegerAttr(mmaMtypeImm));
    MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), mmaMtypeVal);

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
    Value strideA = getStrideBytes(subA, 0);
    Value strideB = getStrideBytes(subB, transposedB ? 1 : 0);

    // --- Step 2: MMA for one K tile (mtype = A/B element type) ---
    MSettilekOp::create(rewriter, loc, rewriter.getI64Type(), currKI64);

    Mlae8mOp::create(rewriter, loc, 0, subA, strideA);
    if (transposedB)
      Mlbe8mOp::create(rewriter, loc, 1, subB, strideB);
    else
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
    if (op->hasAttr(kTritonConsumerFenceAttr))
      LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::seq_cst);
    finishReplacement();

    return success();
  }

private:
  bool tritonW8A8FastPath;
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
  LowerLinalgToBOSCAMEPass(const LowerLinalgToBOSCAMEPass &pass)
      : PassWrapper(pass) {}

  Option<bool> tritonW8A8FastPath{
      *this, "triton-w8a8-fast-path",
      llvm::cl::desc("Fuse exact Triton i8 dot + sitofp and use the Qwen "
                     "2A4B/1A8B AME schedules for transposed weights"),
      llvm::cl::init(false)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<BOSCAMEDialect, arith::ArithDialect, linalg::LinalgDialect,
                LLVM::LLVMDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void LowerLinalgToBOSCAMEPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  // Materialize every eligible Triton i32-dot/sitofp fusion before dialect
  // conversion. Keeping this as a distinct greedy phase guarantees that all
  // newly created f32 matmuls are visible to the AME conversion, including
  // the first of several static quantization groups.
  if (tritonW8A8FastPath) {
    RewritePatternSet fusionPatterns(context);
    fusionPatterns.add<TritonI8DotCastToF32Matmul>(context);
    if (failed(applyPatternsGreedily(module, std::move(fusionPatterns)))) {
      signalPassFailure();
      return;
    }
  }

  RewritePatternSet patterns(context);
  patterns.add<MatmulToBOSCAMELowering>(context, tritonW8A8FastPath);

  ConversionTarget target(*context);
  target.addLegalDialect<BOSCAMEDialect, arith::ArithDialect,
                         linalg::LinalgDialect, LLVM::LLVMDialect,
                         memref::MemRefDialect, scf::SCFDialect>();
  target.addDynamicallyLegalOp<linalg::MatmulOp>([&](linalg::MatmulOp op) {
    if (op->hasAttr(kTritonConsumerFenceAttr))
      return false;
    if (tritonW8A8FastPath && succeeded(matchTritonI8DotCast(op)))
      return false;
    return failed(matchQwenDirectCMatmul(op));
  });

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
