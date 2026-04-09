//===- MatMulVectorizationBLIS.cpp ----------------------------------------===//
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
// This file implements the BLIS-style matmul vectorization.
//
//===----------------------------------------------------------------------===//

#include <optional>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

//===----------------------------------------------------------------------===//
// Helpers: detect dequant chain and (optional) int4 unpack chain.
//===----------------------------------------------------------------------===//

struct DequantChain {
  Value i8Weight;   // i8 weight after unpack (for int4) or original i8 weight
  Value scale;      // scale memref
  Value dequantBuf; // B operand of matmul (to be eliminated)
  Value castBuf;    // intermediate cast buffer (to be eliminated)
  linalg::GenericOp castOp;
  linalg::GenericOp mulOp;
  int scaleAxis; // first non-1 dimension in scale type
};

static bool isSingleSitofpGeneric(linalg::GenericOp op) {
  if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
    return false;
  auto &body = op.getRegion().front();
  if (body.getNumArguments() != 2)
    return false;
  auto yield = cast<linalg::YieldOp>(body.getTerminator());
  auto sitofpOp =
      dyn_cast_or_null<arith::SIToFPOp>(yield.getOperand(0).getDefiningOp());
  if (!sitofpOp)
    return false;
  if (sitofpOp.getOperand() != body.getArgument(0))
    return false;
  auto inType = dyn_cast<MemRefType>(op.getInputs()[0].getType());
  return inType && inType.getElementType().isInteger(8);
}

static bool isSingleMulfGeneric(linalg::GenericOp op) {
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
    return false;
  auto &body = op.getRegion().front();
  if (body.getNumArguments() != 3)
    return false;
  auto yield = cast<linalg::YieldOp>(body.getTerminator());
  auto mulfOp =
      dyn_cast_or_null<arith::MulFOp>(yield.getOperand(0).getDefiningOp());
  if (!mulfOp)
    return false;
  auto lhs = mulfOp.getLhs();
  auto rhs = mulfOp.getRhs();
  return (lhs == body.getArgument(0) && rhs == body.getArgument(1)) ||
         (lhs == body.getArgument(1) && rhs == body.getArgument(0));
}

static std::optional<DequantChain> findDequantChain(Value matmulB) {
  // matmulB is the B operand of linalg.matmul (a memref). We find:
  //   cast_generic(i8 -> f32/f16) -> castBuf
  //   mul_generic(castBuf * scale) -> dequantBuf (= matmulB)
  linalg::GenericOp mulGeneric = nullptr;
  for (auto *user : matmulB.getUsers()) {
    auto genOp = dyn_cast<linalg::GenericOp>(user);
    if (!genOp)
      continue;
    for (auto output : genOp.getDpsInits()) {
      if (output == matmulB && isSingleMulfGeneric(genOp)) {
        mulGeneric = genOp;
        break;
      }
    }
    if (mulGeneric)
      break;
  }
  if (!mulGeneric)
    return std::nullopt;

  Value castResult = mulGeneric.getInputs()[0];
  Value scaleInput = mulGeneric.getInputs()[1];

  linalg::GenericOp castGeneric = nullptr;
  for (auto *user : castResult.getUsers()) {
    auto genOp = dyn_cast<linalg::GenericOp>(user);
    if (!genOp)
      continue;
    for (auto output : genOp.getDpsInits()) {
      if (output == castResult && isSingleSitofpGeneric(genOp)) {
        castGeneric = genOp;
        break;
      }
    }
    if (castGeneric)
      break;
  }
  if (!castGeneric) {
    std::swap(castResult, scaleInput);
    for (auto *user : castResult.getUsers()) {
      auto genOp = dyn_cast<linalg::GenericOp>(user);
      if (!genOp)
        continue;
      for (auto output : genOp.getDpsInits()) {
        if (output == castResult && isSingleSitofpGeneric(genOp)) {
          castGeneric = genOp;
          break;
        }
      }
      if (castGeneric)
        break;
    }
    if (!castGeneric)
      return std::nullopt;
  }

  Value i8Weight = castGeneric.getInputs()[0];

  auto scaleType = dyn_cast<MemRefType>(scaleInput.getType());
  int scaleAxis = -1;
  if (scaleType) {
    for (int i = 0; i < (int)scaleType.getRank(); i++) {
      if (scaleType.getDimSize(i) != 1) {
        scaleAxis = i;
        break;
      }
    }
  }

  DequantChain chain;
  chain.i8Weight = i8Weight;
  chain.scale = scaleInput;
  chain.dequantBuf = matmulB;
  chain.castBuf = castResult;
  chain.castOp = castGeneric;
  chain.mulOp = mulGeneric;
  chain.scaleAxis = scaleAxis;
  return chain;
}

struct Int4UnpackInfo {
  Value packedWeight; // Original packed i8 buffer [K, N/2]
  SmallVector<Operation *> allOps;
  SmallVector<Value> allBuffers;
};

static linalg::GenericOp findProducerGeneric(Value buf) {
  for (auto *user : buf.getUsers()) {
    auto gen = dyn_cast<linalg::GenericOp>(user);
    if (!gen)
      continue;
    for (auto out : gen.getDpsInits()) {
      if (out == buf)
        return gen;
    }
  }
  return nullptr;
}

static bool bodyHasOnlyOp(linalg::GenericOp gen,
                          llvm::function_ref<bool(Operation *)> pred) {
  auto &body = gen.getRegion().front();
  auto yieldOp = cast<linalg::YieldOp>(body.getTerminator());
  auto *defOp = yieldOp.getOperand(0).getDefiningOp();
  return defOp && pred(defOp);
}

static std::optional<Int4UnpackInfo> findInt4UnpackChain(Value i8Weight) {
  // i8Weight should be defined by a reinterpret_cast from a [K, N/2, 2] concat
  // buffer that interleaves low/high nibbles.
  auto rcOp = i8Weight.getDefiningOp<memref::ReinterpretCastOp>();
  if (!rcOp)
    return std::nullopt;

  Value concatBuf = rcOp.getSource();
  auto concatType = dyn_cast<MemRefType>(concatBuf.getType());
  if (!concatType || concatType.getRank() != 3 || concatType.getDimSize(2) != 2)
    return std::nullopt;

  Int4UnpackInfo info;
  info.allBuffers.push_back(concatBuf);

  // Find the two memref.copy operations into slices of concatBuf.
  SmallVector<memref::CopyOp> copies;
  for (auto *user : concatBuf.getUsers()) {
    auto sliceRC = dyn_cast<memref::ReinterpretCastOp>(user);
    if (!sliceRC || sliceRC == rcOp)
      continue;
    info.allOps.push_back(sliceRC);
    for (auto *sliceUser : sliceRC.getResult().getUsers()) {
      auto cp = dyn_cast<memref::CopyOp>(sliceUser);
      if (cp && cp.getTarget() == sliceRC.getResult()) {
        copies.push_back(cp);
        info.allOps.push_back(cp);
      }
    }
  }
  if (copies.size() != 2)
    return std::nullopt;

  Value packedBuf = nullptr;

  for (auto cp : copies) {
    Value src = cp.getSource();
    auto srcRC = src.getDefiningOp<memref::ReinterpretCastOp>();
    if (!srcRC)
      return std::nullopt;
    info.allOps.push_back(srcRC);

    Value nibbleBuf = srcRC.getSource();
    info.allBuffers.push_back(nibbleBuf);

    auto producer = findProducerGeneric(nibbleBuf);
    if (!producer)
      return std::nullopt;

    // Producer must be shrsi generic.
    if (!bodyHasOnlyOp(producer,
                       [](Operation *op) { return isa<arith::ShRSIOp>(op); }))
      return std::nullopt;
    info.allOps.push_back(producer);

    Value firstInput = producer.getInputs()[0];

    // Low path: shrsi <- shli <- andi <- packed
    auto shliProducer = findProducerGeneric(firstInput);
    if (shliProducer && bodyHasOnlyOp(shliProducer, [](Operation *op) {
          return isa<arith::ShLIOp>(op);
        })) {
      info.allOps.push_back(shliProducer);
      info.allBuffers.push_back(firstInput);

      Value andiInput = shliProducer.getInputs()[0];
      auto andiProducer = findProducerGeneric(andiInput);
      if (!andiProducer || !bodyHasOnlyOp(andiProducer, [](Operation *op) {
            return isa<arith::AndIOp>(op);
          }))
        return std::nullopt;
      info.allOps.push_back(andiProducer);
      info.allBuffers.push_back(andiInput);

      Value candidate = andiProducer.getInputs()[0];
      if (packedBuf && packedBuf != candidate)
        return std::nullopt;
      packedBuf = candidate;
    } else {
      // High path: shrsi(packed, shift)
      if (packedBuf && packedBuf != firstInput)
        return std::nullopt;
      packedBuf = firstInput;
    }
  }

  if (!packedBuf)
    return std::nullopt;
  auto packedType = dyn_cast<MemRefType>(packedBuf.getType());
  if (!packedType || packedType.getRank() != 2 ||
      !packedType.getElementType().isInteger(8))
    return std::nullopt;

  info.packedWeight = packedBuf;
  return info;
}

class MatMulVectorizationBLISPattern : public ConversionPattern {
public:
  explicit MatMulVectorizationBLISPattern(MLIRContext *context)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // Create constant indices
    const Value c0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value c1 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    const Value c2 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(2));
    const Value c3 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(3));
    const Value c4 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(4));
    const Value c5 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(5));
    const Value c6 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(6));
    const Value c7 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(7));

    // Fixed BLIS blocking parameters from txt file
    const Value nc = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(256)); // nc = 256
    const Value kc = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(128)); // kc = 128
    const Value mc = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(64)); // mc = 64
    const Value mr = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(8)); // mr = 8
    const Value nr = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(32)); // nr = 32

    // Get input A, B, C
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);

    // Detect dequant chain for B (w8a32/w8a16/w4a16 prefill).
    auto dequantChain = findDequantChain(B);
    std::optional<Int4UnpackInfo> int4Info;
    if (dequantChain)
      int4Info = findInt4UnpackChain(dequantChain->i8Weight);

    // Get dimensions
    const Value m = rewriter.create<memref::DimOp>(loc, A, c0);
    const Value n = rewriter.create<memref::DimOp>(loc, C, c1);
    const Value k = rewriter.create<memref::DimOp>(loc, A, c1);

    // Get element types: input type from A, accumulation type from C
    ShapedType ATy = cast<ShapedType>(A.getType());
    Type eleTy = ATy.getElementType();
    ShapedType CTy = cast<ShapedType>(C.getType());
    Type accEleTy = CTy.getElementType();
    bool isInteger = isa<IntegerType>(eleTy);
    VectorType vectorTy = VectorType::get({32}, eleTy);
    VectorType accVecTy =
        isInteger ? VectorType::get({32}, accEleTy) : vectorTy;

    // Pre-create constants for int4 unpack (scalar) and scale indexing.
    Value i8Mask15 = nullptr;
    Value i8Shift4 = nullptr;
    if (dequantChain && int4Info) {
      i8Mask15 = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(rewriter.getI8Type(), 15));
      i8Shift4 = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(rewriter.getI8Type(), 4));
    }

    // BLIS 5-loop structure
    // Loop 1: jc - column blocking
    rewriter.create<scf::ParallelOp>(
        loc, c0, n, nc, [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          Value jc = ivs[0];
          // Compute actual nc for this block
          auto jcEnd = builder.create<arith::AddIOp>(loc, jc, nc);
          auto jcBound = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::slt, jcEnd, n);
          auto jcActualEnd =
              builder.create<arith::SelectOp>(loc, jcBound, jcEnd, n);
          auto ncActual = builder.create<arith::SubIOp>(loc, jcActualEnd, jc);

          // Loop 2: pc - k blocking
          builder.create<scf::ForOp>(
              loc, c0, k, kc, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value pc, ValueRange) {
                // Compute actual kc for this block
                auto pcEnd = builder.create<arith::AddIOp>(loc, pc, kc);
                auto pcBound = builder.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::slt, pcEnd, k);
                auto pcActualEnd =
                    builder.create<arith::SelectOp>(loc, pcBound, pcEnd, k);
                auto kcActual =
                    builder.create<arith::SubIOp>(loc, pcActualEnd, pc);

                // Check if we should allocate B_packed (avoid
                // allocation for empty blocks)
                auto kcActualPositive = builder.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::sgt, kcActual, c0);
                auto ncActualPositive = builder.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::sgt, ncActual, c0);
                auto shouldAllocB = builder.create<arith::AndIOp>(
                    loc, kcActualPositive, ncActualPositive);

                builder.create<scf::IfOp>(
                    loc, shouldAllocB,
                    [&](OpBuilder &builder, Location loc) {
                      // Allocate and pack B block
                      auto B_packedType = MemRefType::get(
                          {ShapedType::kDynamic, ShapedType::kDynamic}, eleTy,
                          AffineMap(), nullptr);
                      Value B_packed = builder.create<memref::AllocOp>(
                          loc, B_packedType, ValueRange{kcActual, ncActual});

                      // Pack B block
                      // clang-format off
                    builder.create<scf::ForOp>(
                        loc, c0, kcActual, c1, ValueRange{},
                        [&](OpBuilder &builder, Location loc, Value kp, ValueRange) {
                          builder.create<scf::ForOp>(
                              loc, c0, ncActual, c1, ValueRange{},
                              [&](OpBuilder &builder, Location loc, Value j, ValueRange) {
                                auto bRowIdx = builder.create<arith::AddIOp>(loc, pc, kp);
                                auto bColIdx = builder.create<arith::AddIOp>(loc, jc, j);

                                // If B is produced by a dequant chain, fuse dequant into packing.
                                if (dequantChain) {
                                  // Load per-channel scale (broadcasted over K).
                                  Value scaleScalar = nullptr;
                                  if (auto scaleType = dyn_cast<MemRefType>(dequantChain->scale.getType())) {
                                    SmallVector<Value> scaleIdx;
                                    int axis = dequantChain->scaleAxis;
                                    if (axis < 0)
                                      axis = (int)scaleType.getRank() - 1;
                                    for (int si = 0; si < (int)scaleType.getRank(); si++) {
                                      if (si == axis)
                                        scaleIdx.push_back(bColIdx);
                                      else
                                        scaleIdx.push_back(c0);
                                    }
                                    scaleScalar = builder.create<memref::LoadOp>(
                                        loc, dequantChain->scale, scaleIdx);
                                  }

                                  Value weightI8 = nullptr;
                                  if (int4Info) {
                                    // packed index = bColIdx / 2
                                    Value packedColIdx = builder.create<arith::DivUIOp>(loc, bColIdx, c2);
                                    Value packedByte = builder.create<memref::LoadOp>(
                                        loc, int4Info->packedWeight, ValueRange{bRowIdx, packedColIdx});
                                    // low = ((packed & 0x0F) << 4) >> 4
                                    Value lowMasked = builder.create<arith::AndIOp>(loc, packedByte, i8Mask15);
                                    Value lowShifted = builder.create<arith::ShLIOp>(loc, lowMasked, i8Shift4);
                                    Value low = builder.create<arith::ShRSIOp>(loc, lowShifted, i8Shift4);
                                    // high = packed >> 4
                                    Value high = builder.create<arith::ShRSIOp>(loc, packedByte, i8Shift4);
                                    // select based on parity
                                    Value rem = builder.create<arith::RemUIOp>(loc, bColIdx, c2);
                                    Value isLow = builder.create<arith::CmpIOp>(
                                        loc, arith::CmpIPredicate::eq, rem, c0);
                                    weightI8 = builder.create<arith::SelectOp>(loc, isLow, low, high);
                                  } else {
                                    weightI8 = builder.create<memref::LoadOp>(
                                        loc, dequantChain->i8Weight, ValueRange{bRowIdx, bColIdx});
                                  }

                                  // i8 -> eleTy, then * scale
                                  Value wFloat = builder.create<arith::SIToFPOp>(loc, eleTy, weightI8);
                                  Value wDequant = scaleScalar
                                                       ? builder.create<arith::MulFOp>(loc, wFloat, scaleScalar)
                                                       : wFloat;
                                  builder.create<memref::StoreOp>(loc, wDequant, B_packed,
                                                                  ValueRange{kp, j});
                                } else {
                                  // Default path: just pack from B.
                                  auto bVal = builder.create<memref::LoadOp>(loc, B,
                                                                           ValueRange{bRowIdx, bColIdx});
                                  builder.create<memref::StoreOp>(loc, bVal, B_packed,
                                                                ValueRange{kp, j});
                                }
                                builder.create<scf::YieldOp>(loc);
                              });
                          builder.create<scf::YieldOp>(loc);
                        });
                    // Loop 3: ic - row blocking
                    builder.create<scf::ParallelOp>(
                            loc, c0, m, mc,
                            [&](OpBuilder &builder, Location loc, ValueRange ivs) {
                            Value ic = ivs[0];
                          // Compute actual mc for this block
                          auto icEnd = builder.create<arith::AddIOp>(loc, ic, mc);
                          auto icBound = builder.create<arith::CmpIOp>(loc,
                                                        arith::CmpIPredicate::slt, icEnd, m);
                          auto icActualEnd = builder.create<arith::SelectOp>(loc, icBound, icEnd, m);
                          auto mcActual = builder.create<arith::SubIOp>(loc, icActualEnd, ic);

                          // Check if we should allocate A_packed
                          auto mcActualPositive = builder.create<arith::CmpIOp>(
                              loc, arith::CmpIPredicate::sgt, mcActual, c0);
                          auto shouldAllocA = builder.create<arith::AndIOp>(
                              loc, mcActualPositive, kcActualPositive);
                          builder.create<scf::IfOp>(loc, shouldAllocA,
                            [&](OpBuilder &builder, Location loc) {
                              // Allocate and pack A block
                              auto A_packedType = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                                                eleTy,AffineMap(),  nullptr);
                              Value A_packed = builder.create<memref::AllocOp>(loc, A_packedType,
                                                                             ValueRange{mcActual, kcActual});

                              // Pack A block
                              builder.create<scf::ForOp>(
                                  loc, c0, mcActual, c1, ValueRange{},
                                  [&](OpBuilder &builder, Location loc, Value i, ValueRange) {
                                    builder.create<scf::ForOp>(
                                        loc, c0, kcActual, c1, ValueRange{},
                                        [&](OpBuilder &builder, Location loc, Value kp, ValueRange) {
                                          auto aRowIdx = builder.create<arith::AddIOp>(loc, ic, i);
                                          auto aColIdx = builder.create<arith::AddIOp>(loc, pc, kp);
                                          auto aVal = builder.create<memref::LoadOp>(loc, A,
                                                                                   ValueRange{aRowIdx, aColIdx});
                                          builder.create<memref::StoreOp>(loc, aVal, A_packed,
                                                                        ValueRange{i, kp});
                                          builder.create<scf::YieldOp>(loc);
                                        });
                                    builder.create<scf::YieldOp>(loc);
                                  });

                              // Loop 4: jr - micro column blocking
                              builder.create<scf::ForOp>(
                                  loc, c0, ncActual, nr, ValueRange{},
                                  [&](OpBuilder &builder, Location loc, Value jr, ValueRange) {
                                    auto jrEnd = builder.create<arith::AddIOp>(loc, jr, nr);
                                    auto jrBound = builder.create<arith::CmpIOp>(
                                    loc, arith::CmpIPredicate::slt, jrEnd, ncActual);
                                    auto jrActualEnd = builder.create<arith::SelectOp>(
                                    loc, jrBound, jrEnd, ncActual);
                                    auto nrActual = builder.create<arith::SubIOp>(
                                    loc, jrActualEnd, jr);
                                    // Process micro blocks
                                    builder.create<scf::ForOp>(
                                        loc, c0, nrActual, nr, ValueRange{},
                                        [&](OpBuilder &builder, Location loc, Value nIdx, ValueRange) {
                                          auto nIdxEnd = builder.create<arith::AddIOp>(loc, nIdx, nr);
                                          auto nIdxBound = builder.create<arith::CmpIOp>(
                                          loc, arith::CmpIPredicate::slt, nIdxEnd, nrActual);
                                          auto nIdxActualEnd = builder.create<arith::SelectOp>(
                                          loc, nIdxBound, nIdxEnd, nrActual);
                                          auto colsToProcess = builder.create<arith::SubIOp>(
                                          loc, nIdxActualEnd, nIdx);
                                          // Check if we can vectorize (at least 32 columns)
                                          auto canVectorize = builder.create<arith::CmpIOp>(
                                              loc, arith::CmpIPredicate::sge, colsToProcess, nr);

                                          builder.create<scf::IfOp>(
                                              loc, canVectorize,
                                              [&](OpBuilder &builder, Location loc) {
                                                // Vectorized path 32
                                                builder.create<scf::ForOp>(
                                                    loc, c0, mcActual, mr, ValueRange{},
                                                    [&](OpBuilder &builder, Location loc, Value ir, ValueRange) {
                                                      auto irEnd = builder.create<arith::AddIOp>(loc, ir, mr);
                                                      auto irBound = builder.create<arith::CmpIOp>(
                                                      loc, arith::CmpIPredicate::slt, irEnd, mcActual);
                                                      auto irActualEnd = builder.create<arith::SelectOp>(
                                                      loc, irBound, irEnd, mcActual);
                                                      auto mrActual = builder.create<arith::SubIOp>(loc, irActualEnd, ir);
                                                      // Check if we have full 8 rows - FIXED: use mr instead of c8
                                                      auto hasFullRows = builder.create<arith::CmpIOp>(
                                                          loc, arith::CmpIPredicate::sge, mrActual, mr);
                                                      builder.create<scf::IfOp>(
                                                          loc, hasFullRows,
                                                          [&](OpBuilder &builder, Location loc) {
                                                            // Full rows vectorized processing
                                                            auto ir0 = builder.create<arith::AddIOp>(loc, ir, c0);
                                                            auto ir1 = builder.create<arith::AddIOp>(loc, ir, c1);
                                                            auto ir2 = builder.create<arith::AddIOp>(loc, ir, c2);
                                                            auto ir3 = builder.create<arith::AddIOp>(loc, ir, c3);
                                                            auto ir4 = builder.create<arith::AddIOp>(loc, ir, c4);
                                                            auto ir5 = builder.create<arith::AddIOp>(loc, ir, c5);
                                                            auto ir6 = builder.create<arith::AddIOp>(loc, ir, c6);
                                                            auto ir7 = builder.create<arith::AddIOp>(loc, ir, c7);

                                                            auto sumInit = builder.create<arith::ConstantOp>(
                                                                loc, accVecTy, builder.getZeroAttr(accVecTy));

                                                            auto sumIterVecs = builder.create<scf::ForOp>(
                                                                loc, c0, kcActual, c1,
                                                                ValueRange{sumInit, sumInit, sumInit, sumInit,
                                                                         sumInit, sumInit, sumInit, sumInit},
                                                                [&](OpBuilder &builder, Location loc, Value kInner,
                                                                    ValueRange iterArgs) {
                                                                  // Load A values for 8 rows
                                                                  Value aVal0 = builder.create<memref::LoadOp>(
                                                                      loc, A_packed, ValueRange{ir0, kInner});
                                                                  Value aVal1 = builder.create<memref::LoadOp>(
                                                                      loc, A_packed, ValueRange{ir1, kInner});
                                                                  Value aVal2 = builder.create<memref::LoadOp>(
                                                                      loc, A_packed, ValueRange{ir2, kInner});
                                                                  Value aVal3 = builder.create<memref::LoadOp>(
                                                                      loc, A_packed, ValueRange{ir3, kInner});
                                                                  Value aVal4 = builder.create<memref::LoadOp>(
                                                                      loc, A_packed, ValueRange{ir4, kInner});
                                                                  Value aVal5 = builder.create<memref::LoadOp>(
                                                                      loc, A_packed, ValueRange{ir5, kInner});
                                                                  Value aVal6 = builder.create<memref::LoadOp>(
                                                                      loc, A_packed, ValueRange{ir6, kInner});
                                                                  Value aVal7 = builder.create<memref::LoadOp>(
                                                                      loc, A_packed, ValueRange{ir7, kInner});

                                                                  if (isInteger) {
                                                                    aVal0 = builder.create<arith::ExtSIOp>(loc, accEleTy, aVal0);
                                                                    aVal1 = builder.create<arith::ExtSIOp>(loc, accEleTy, aVal1);
                                                                    aVal2 = builder.create<arith::ExtSIOp>(loc, accEleTy, aVal2);
                                                                    aVal3 = builder.create<arith::ExtSIOp>(loc, accEleTy, aVal3);
                                                                    aVal4 = builder.create<arith::ExtSIOp>(loc, accEleTy, aVal4);
                                                                    aVal5 = builder.create<arith::ExtSIOp>(loc, accEleTy, aVal5);
                                                                    aVal6 = builder.create<arith::ExtSIOp>(loc, accEleTy, aVal6);
                                                                    aVal7 = builder.create<arith::ExtSIOp>(loc, accEleTy, aVal7);
                                                                  }

                                                                  // Broadcast A values to accumulation vectors
                                                                  auto aVec0 = builder.create<vector::BroadcastOp>(
                                                                      loc, accVecTy, aVal0);
                                                                  auto aVec1 = builder.create<vector::BroadcastOp>(
                                                                      loc, accVecTy, aVal1);
                                                                  auto aVec2 = builder.create<vector::BroadcastOp>(
                                                                      loc, accVecTy, aVal2);
                                                                  auto aVec3 = builder.create<vector::BroadcastOp>(
                                                                      loc, accVecTy, aVal3);
                                                                  auto aVec4 = builder.create<vector::BroadcastOp>(
                                                                      loc, accVecTy, aVal4);
                                                                  auto aVec5 = builder.create<vector::BroadcastOp>(
                                                                      loc, accVecTy, aVal5);
                                                                  auto aVec6 = builder.create<vector::BroadcastOp>(
                                                                      loc, accVecTy, aVal6);
                                                                  auto aVec7 = builder.create<vector::BroadcastOp>(
                                                                      loc, accVecTy, aVal7);

                                                                  // Load B vector and sign-extend for integer
                                                                  auto bColIdx = builder.create<arith::AddIOp>(loc, jr, nIdx);
                                                                  Value bVec = builder.create<vector::LoadOp>(
                                                                      loc, vectorTy, B_packed,
                                                                      ValueRange{kInner, bColIdx});
                                                                  if (isInteger)
                                                                    bVec = builder.create<arith::ExtSIOp>(loc, accVecTy, bVec);

                                                                  // Multiply-accumulate
                                                                  Value resSumVec0, resSumVec1, resSumVec2, resSumVec3,
                                                                        resSumVec4, resSumVec5, resSumVec6, resSumVec7;
                                                                  if (isInteger) {
                                                                    resSumVec0 = builder.create<arith::AddIOp>(loc, iterArgs[0],
                                                                        builder.create<arith::MulIOp>(loc, aVec0, bVec));
                                                                    resSumVec1 = builder.create<arith::AddIOp>(loc, iterArgs[1],
                                                                        builder.create<arith::MulIOp>(loc, aVec1, bVec));
                                                                    resSumVec2 = builder.create<arith::AddIOp>(loc, iterArgs[2],
                                                                        builder.create<arith::MulIOp>(loc, aVec2, bVec));
                                                                    resSumVec3 = builder.create<arith::AddIOp>(loc, iterArgs[3],
                                                                        builder.create<arith::MulIOp>(loc, aVec3, bVec));
                                                                    resSumVec4 = builder.create<arith::AddIOp>(loc, iterArgs[4],
                                                                        builder.create<arith::MulIOp>(loc, aVec4, bVec));
                                                                    resSumVec5 = builder.create<arith::AddIOp>(loc, iterArgs[5],
                                                                        builder.create<arith::MulIOp>(loc, aVec5, bVec));
                                                                    resSumVec6 = builder.create<arith::AddIOp>(loc, iterArgs[6],
                                                                        builder.create<arith::MulIOp>(loc, aVec6, bVec));
                                                                    resSumVec7 = builder.create<arith::AddIOp>(loc, iterArgs[7],
                                                                        builder.create<arith::MulIOp>(loc, aVec7, bVec));
                                                                  } else {
                                                                    resSumVec0 = builder.create<vector::FMAOp>(loc, aVec0, bVec, iterArgs[0]);
                                                                    resSumVec1 = builder.create<vector::FMAOp>(loc, aVec1, bVec, iterArgs[1]);
                                                                    resSumVec2 = builder.create<vector::FMAOp>(loc, aVec2, bVec, iterArgs[2]);
                                                                    resSumVec3 = builder.create<vector::FMAOp>(loc, aVec3, bVec, iterArgs[3]);
                                                                    resSumVec4 = builder.create<vector::FMAOp>(loc, aVec4, bVec, iterArgs[4]);
                                                                    resSumVec5 = builder.create<vector::FMAOp>(loc, aVec5, bVec, iterArgs[5]);
                                                                    resSumVec6 = builder.create<vector::FMAOp>(loc, aVec6, bVec, iterArgs[6]);
                                                                    resSumVec7 = builder.create<vector::FMAOp>(loc, aVec7, bVec, iterArgs[7]);
                                                                  }

                                                                  builder.create<scf::YieldOp>(
                                                                      loc, ValueRange{resSumVec0, resSumVec1, resSumVec2,
                                                                                      resSumVec3, resSumVec4, resSumVec5,
                                                                                      resSumVec6, resSumVec7});
                                                                });

                                                            // Store results with accumulation
                                                            auto cRow0 = builder.create<arith::AddIOp>(loc, ic, ir0);
                                                            auto cRow1 = builder.create<arith::AddIOp>(loc, ic, ir1);
                                                            auto cRow2 = builder.create<arith::AddIOp>(loc, ic, ir2);
                                                            auto cRow3 = builder.create<arith::AddIOp>(loc, ic, ir3);
                                                            auto cRow4 = builder.create<arith::AddIOp>(loc, ic, ir4);
                                                            auto cRow5 = builder.create<arith::AddIOp>(loc, ic, ir5);
                                                            auto cRow6 = builder.create<arith::AddIOp>(loc, ic, ir6);
                                                            auto cRow7 = builder.create<arith::AddIOp>(loc, ic, ir7);
                                                            auto cColBase = builder.create<arith::AddIOp>(loc, jc, jr);
                                                            auto cColIdx = builder.create<arith::AddIOp>(loc, cColBase, nIdx);

                                                            // Load current C values (accumulation type)
                                                            auto cVec0 = builder.create<vector::LoadOp>(
                                                                loc, accVecTy, C, ValueRange{cRow0, cColIdx});
                                                            auto cVec1 = builder.create<vector::LoadOp>(
                                                                loc, accVecTy, C, ValueRange{cRow1, cColIdx});
                                                            auto cVec2 = builder.create<vector::LoadOp>(
                                                                loc, accVecTy, C, ValueRange{cRow2, cColIdx});
                                                            auto cVec3 = builder.create<vector::LoadOp>(
                                                                loc, accVecTy, C, ValueRange{cRow3, cColIdx});
                                                            auto cVec4 = builder.create<vector::LoadOp>(
                                                                loc, accVecTy, C, ValueRange{cRow4, cColIdx});
                                                            auto cVec5 = builder.create<vector::LoadOp>(
                                                                loc, accVecTy, C, ValueRange{cRow5, cColIdx});
                                                            auto cVec6 = builder.create<vector::LoadOp>(
                                                                loc, accVecTy, C, ValueRange{cRow6, cColIdx});
                                                            auto cVec7 = builder.create<vector::LoadOp>(
                                                                loc, accVecTy, C, ValueRange{cRow7, cColIdx});

                                                            // Accumulate: C = C + A*B
                                                            Value finalVec0, finalVec1, finalVec2, finalVec3,
                                                                  finalVec4, finalVec5, finalVec6, finalVec7;
                                                            if (isInteger) {
                                                              finalVec0 = builder.create<arith::AddIOp>(loc, cVec0, sumIterVecs.getResult(0));
                                                              finalVec1 = builder.create<arith::AddIOp>(loc, cVec1, sumIterVecs.getResult(1));
                                                              finalVec2 = builder.create<arith::AddIOp>(loc, cVec2, sumIterVecs.getResult(2));
                                                              finalVec3 = builder.create<arith::AddIOp>(loc, cVec3, sumIterVecs.getResult(3));
                                                              finalVec4 = builder.create<arith::AddIOp>(loc, cVec4, sumIterVecs.getResult(4));
                                                              finalVec5 = builder.create<arith::AddIOp>(loc, cVec5, sumIterVecs.getResult(5));
                                                              finalVec6 = builder.create<arith::AddIOp>(loc, cVec6, sumIterVecs.getResult(6));
                                                              finalVec7 = builder.create<arith::AddIOp>(loc, cVec7, sumIterVecs.getResult(7));
                                                            } else {
                                                              finalVec0 = builder.create<arith::AddFOp>(loc, cVec0, sumIterVecs.getResult(0));
                                                              finalVec1 = builder.create<arith::AddFOp>(loc, cVec1, sumIterVecs.getResult(1));
                                                              finalVec2 = builder.create<arith::AddFOp>(loc, cVec2, sumIterVecs.getResult(2));
                                                              finalVec3 = builder.create<arith::AddFOp>(loc, cVec3, sumIterVecs.getResult(3));
                                                              finalVec4 = builder.create<arith::AddFOp>(loc, cVec4, sumIterVecs.getResult(4));
                                                              finalVec5 = builder.create<arith::AddFOp>(loc, cVec5, sumIterVecs.getResult(5));
                                                              finalVec6 = builder.create<arith::AddFOp>(loc, cVec6, sumIterVecs.getResult(6));
                                                              finalVec7 = builder.create<arith::AddFOp>(loc, cVec7, sumIterVecs.getResult(7));
                                                            }

                                                            // Store back to C
                                                            builder.create<vector::StoreOp>(
                                                                loc, finalVec0, C, ValueRange{cRow0, cColIdx});
                                                            builder.create<vector::StoreOp>(
                                                                loc, finalVec1, C, ValueRange{cRow1, cColIdx});
                                                            builder.create<vector::StoreOp>(
                                                                loc, finalVec2, C, ValueRange{cRow2, cColIdx});
                                                            builder.create<vector::StoreOp>(
                                                                loc, finalVec3, C, ValueRange{cRow3, cColIdx});
                                                            builder.create<vector::StoreOp>(
                                                                loc, finalVec4, C, ValueRange{cRow4, cColIdx});
                                                            builder.create<vector::StoreOp>(
                                                                loc, finalVec5, C, ValueRange{cRow5, cColIdx});
                                                            builder.create<vector::StoreOp>(
                                                                loc, finalVec6, C, ValueRange{cRow6, cColIdx});
                                                            builder.create<vector::StoreOp>(
                                                                loc, finalVec7, C, ValueRange{cRow7, cColIdx});

                                                            builder.create<scf::YieldOp>(loc);
                                                          },
                                                          [&](OpBuilder &builder, Location loc) {
                                                            // Scalar path for incomplete rows
                                                            builder.create<scf::ForOp>(
                                                                loc, c0, nr, c1, ValueRange{},
                                                                [&](OpBuilder &builder, Location loc, Value jj, ValueRange) {
                                                                  builder.create<scf::ForOp>(
                                                                      loc, ir, irActualEnd, c1, ValueRange{},
                                                                      [&](OpBuilder &builder, Location loc, Value ii, ValueRange) {
                                                                        Value sumInit;
                                                                        if (isInteger)
                                                                          sumInit = builder.create<arith::ConstantOp>(
                                                                              loc, accEleTy, builder.getIntegerAttr(accEleTy, 0));
                                                                        else
                                                                          sumInit = builder.create<arith::ConstantOp>(
                                                                              loc, accEleTy, builder.getFloatAttr(accEleTy, 0.0));
                                                                        auto sumIter = builder.create<scf::ForOp>(
                                                                            loc, c0, kcActual, c1,
                                                                            ValueRange{sumInit},
                                                                            [&](OpBuilder &builder, Location loc, Value kInner,
                                                                                ValueRange iterArgs) {
                                                                              Value aVal = builder.create<memref::LoadOp>(
                                                                                  loc, A_packed, ValueRange{ii, kInner});
                                                                              auto bColBase = builder.create<arith::AddIOp>(loc, jr, nIdx);
                                                                              auto bColIdx = builder.create<arith::AddIOp>(loc, bColBase, jj);
                                                                              Value bVal = builder.create<memref::LoadOp>(
                                                                                  loc, B_packed, ValueRange{kInner, bColIdx});
                                                                              Value prod, newSum;
                                                                              if (isInteger) {
                                                                                aVal = builder.create<arith::ExtSIOp>(loc, accEleTy, aVal);
                                                                                bVal = builder.create<arith::ExtSIOp>(loc, accEleTy, bVal);
                                                                                prod = builder.create<arith::MulIOp>(loc, aVal, bVal);
                                                                                newSum = builder.create<arith::AddIOp>(loc, iterArgs[0], prod);
                                                                              } else {
                                                                                prod = builder.create<arith::MulFOp>(loc, aVal, bVal);
                                                                                newSum = builder.create<arith::AddFOp>(loc, iterArgs[0], prod);
                                                                              }
                                                                              builder.create<scf::YieldOp>(loc, ValueRange{newSum});
                                                                            });
                                                                        auto cRowIdx = builder.create<arith::AddIOp>(loc, ic, ii);
                                                                        auto cColBase = builder.create<arith::AddIOp>(loc, jc, jr);
                                                                        auto cColBase2 = builder.create<arith::AddIOp>(loc, cColBase, nIdx);
                                                                        auto cColIdx = builder.create<arith::AddIOp>(loc, cColBase2, jj);
                                                                        auto currentVal = builder.create<memref::LoadOp>(
                                                                            loc, C, ValueRange{cRowIdx, cColIdx});
                                                                        Value finalSum;
                                                                        if (isInteger)
                                                                          finalSum = builder.create<arith::AddIOp>(loc, currentVal, sumIter.getResult(0));
                                                                        else
                                                                          finalSum = builder.create<arith::AddFOp>(loc, currentVal, sumIter.getResult(0));
                                                                        builder.create<memref::StoreOp>(
                                                                            loc, finalSum, C, ValueRange{cRowIdx, cColIdx});
                                                                        builder.create<scf::YieldOp>(loc);
                                                                      });
                                                                  builder.create<scf::YieldOp>(loc);
                                                                });
                                                            builder.create<scf::YieldOp>(loc);
                                                          });

                                                      builder.create<scf::YieldOp>(loc);
                                                    });
                                                builder.create<scf::YieldOp>(loc);
                                              },
                                              [&](OpBuilder &builder, Location loc) {
                                                // Scalar path for incomplete columns
                                                builder.create<scf::ForOp>(
                                                    loc, nIdx, nIdxActualEnd, c1, ValueRange{},
                                                    [&](OpBuilder &builder, Location loc, Value nIdxTail, ValueRange) {
                                                      builder.create<scf::ForOp>(
                                                          loc, c0, mcActual, mr, ValueRange{},
                                                          [&](OpBuilder &builder, Location loc, Value ir, ValueRange) {
                                                            auto irEnd = builder.create<arith::AddIOp>(loc, ir, mr);
                                                            auto irBound = builder.create<arith::CmpIOp>(
                                                            loc, arith::CmpIPredicate::slt, irEnd, mcActual);
                                                            auto irActualEnd = builder.create<arith::SelectOp>(
                                                            loc, irBound, irEnd, mcActual);

                                                            builder.create<scf::ForOp>(
                                                                loc, ir, irActualEnd, c1, ValueRange{},
                                                                [&](OpBuilder &builder, Location loc, Value ii, ValueRange) {
                                                                  Value sumInit;
                                                                  if (isInteger)
                                                                    sumInit = builder.create<arith::ConstantOp>(
                                                                        loc, accEleTy, builder.getIntegerAttr(accEleTy, 0));
                                                                  else
                                                                    sumInit = builder.create<arith::ConstantOp>(
                                                                        loc, accEleTy, builder.getFloatAttr(accEleTy, 0.0));
                                                                  auto sumIter = builder.create<scf::ForOp>(
                                                                      loc, c0, kcActual, c1,
                                                                      ValueRange{sumInit},
                                                                      [&](OpBuilder &builder, Location loc, Value kInner,
                                                                          ValueRange iterArgs) {
                                                                        Value aVal = builder.create<memref::LoadOp>(
                                                                            loc, A_packed, ValueRange{ii, kInner});
                                                                        auto bColBase = builder.create<arith::AddIOp>(loc, jr, nIdxTail);
                                                                        Value bVal = builder.create<memref::LoadOp>(
                                                                            loc, B_packed, ValueRange{kInner, bColBase});
                                                                        Value prod, newSum;
                                                                        if (isInteger) {
                                                                          aVal = builder.create<arith::ExtSIOp>(loc, accEleTy, aVal);
                                                                          bVal = builder.create<arith::ExtSIOp>(loc, accEleTy, bVal);
                                                                          prod = builder.create<arith::MulIOp>(loc, aVal, bVal);
                                                                          newSum = builder.create<arith::AddIOp>(loc, iterArgs[0], prod);
                                                                        } else {
                                                                          prod = builder.create<arith::MulFOp>(loc, aVal, bVal);
                                                                          newSum = builder.create<arith::AddFOp>(loc, iterArgs[0], prod);
                                                                        }
                                                                        builder.create<scf::YieldOp>(loc, ValueRange{newSum});
                                                                      });
                                                                  auto cRowIdx = builder.create<arith::AddIOp>(loc, ic, ii);
                                                                  auto cColBase = builder.create<arith::AddIOp>(loc, jc, jr);
                                                                  auto cColIdx = builder.create<arith::AddIOp>(loc, cColBase, nIdxTail);
                                                                  auto currentVal = builder.create<memref::LoadOp>(
                                                                      loc, C, ValueRange{cRowIdx, cColIdx});
                                                                  Value finalSum;
                                                                  if (isInteger)
                                                                    finalSum = builder.create<arith::AddIOp>(loc, currentVal, sumIter.getResult(0));
                                                                  else
                                                                    finalSum = builder.create<arith::AddFOp>(loc, currentVal, sumIter.getResult(0));
                                                                  builder.create<memref::StoreOp>(
                                                                      loc, finalSum, C, ValueRange{cRowIdx, cColIdx});
                                                                  builder.create<scf::YieldOp>(loc);
                                                                });
                                                            builder.create<scf::YieldOp>(loc);
                                                          });
                                                      builder.create<scf::YieldOp>(loc);
                                                    });
                                                builder.create<scf::YieldOp>(loc);
                                              });

                                          builder.create<scf::YieldOp>(loc);
                                        });
                                    builder.create<scf::YieldOp>(loc);
                                  });
                                  // clang-format on
                                  // Deallocate A_packed
                                  builder.create<memref::DeallocOp>(loc,
                                                                    A_packed);
                                  builder.create<scf::YieldOp>(loc);
                                },
                                [&](OpBuilder &builder, Location loc) {
                                  // Skip A_packed allocation for empty blocks
                                  builder.create<scf::YieldOp>(loc);
                                });
                          });

                      // Deallocate B_packed
                      builder.create<memref::DeallocOp>(loc, B_packed);
                      builder.create<scf::YieldOp>(loc);
                    },
                    [&](OpBuilder &builder, Location loc) {
                      // Skip B_packed allocation for empty blocks
                      builder.create<scf::YieldOp>(loc);
                    });

                builder.create<scf::YieldOp>(loc);
              });
        });

    rewriter.eraseOp(op);

    // If we fused dequant into packing, clean up the now-dead dequant/unpack
    // IR.
    if (dequantChain) {
      SmallPtrSet<Operation *, 32> erasedOps;
      erasedOps.insert(op);

      auto canEraseBuf = [&](Value buf, Operation *producer) -> bool {
        for (auto *user : buf.getUsers()) {
          if (user == producer || erasedOps.contains(user))
            continue;
          if (isa<memref::DeallocOp>(user))
            continue;
          return false;
        }
        return true;
      };

      auto eraseBufferChain = [&](Value buf, Operation *producer) {
        SmallVector<Operation *> toErase;
        for (auto *user : buf.getUsers()) {
          if (isa<memref::DeallocOp>(user))
            toErase.push_back(user);
        }
        for (auto *deadOp : toErase) {
          rewriter.eraseOp(deadOp);
          erasedOps.insert(deadOp);
        }
        rewriter.eraseOp(producer);
        erasedOps.insert(producer);
        if (auto allocOp = buf.getDefiningOp()) {
          rewriter.eraseOp(allocOp);
          erasedOps.insert(allocOp);
        }
      };

      if (canEraseBuf(dequantChain->dequantBuf, dequantChain->mulOp))
        eraseBufferChain(dequantChain->dequantBuf, dequantChain->mulOp);

      if (canEraseBuf(dequantChain->castBuf, dequantChain->castOp))
        eraseBufferChain(dequantChain->castBuf, dequantChain->castOp);

      if (int4Info) {
        // Best-effort cleanup of unpack intermediates if they became dead.
        for (auto *deadOp : llvm::reverse(int4Info->allOps)) {
          if (!erasedOps.contains(deadOp)) {
            rewriter.eraseOp(deadOp);
            erasedOps.insert(deadOp);
          }
        }
        for (auto buf : llvm::reverse(int4Info->allBuffers)) {
          for (auto *user : buf.getUsers()) {
            if (isa<memref::DeallocOp>(user) && !erasedOps.contains(user)) {
              rewriter.eraseOp(user);
              erasedOps.insert(user);
            }
          }
          if (auto *defOp = buf.getDefiningOp()) {
            if (!erasedOps.contains(defOp)) {
              bool allUsersDead = true;
              for (auto *user : buf.getUsers()) {
                if (!erasedOps.contains(user)) {
                  allUsersDead = false;
                  break;
                }
              }
              if (allUsersDead) {
                rewriter.eraseOp(defOp);
                erasedOps.insert(defOp);
              }
            }
          }
        }
      }
    }

    return success();
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// MatMulVectorizationBLISPass
//===----------------------------------------------------------------------===//

namespace {
class MatMulVectorizationBLISPass
    : public PassWrapper<MatMulVectorizationBLISPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulVectorizationBLISPass)
  StringRef getArgument() const final { return "matmul-vectorization-blis"; }
  StringRef getDescription() const final {
    return "BLIS-style MatMul Vectorization with fixed blocking parameters.";
  }
  MatMulVectorizationBLISPass() = default;
  MatMulVectorizationBLISPass(const MatMulVectorizationBLISPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, scf::SCFDialect, affine::AffineDialect,
                VectorDialect, memref::MemRefDialect, arith::ArithDialect>();
  }
};
} // end anonymous namespace.

void MatMulVectorizationBLISPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                       scf::SCFDialect, memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<MatMulVectorizationBLISPattern>(context);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerMatMulVectorizationBLISPass() {
  PassRegistration<MatMulVectorizationBLISPass>();
}
} // namespace buddy
} // namespace mlir
