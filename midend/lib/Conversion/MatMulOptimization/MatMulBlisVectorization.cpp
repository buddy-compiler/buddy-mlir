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
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0));
    const Value c1 =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(1));
    const Value c2 =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(2));
    const Value c3 =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(3));
    const Value c4 =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(4));
    const Value c5 =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(5));
    const Value c6 =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(6));
    const Value c7 =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(7));

    // Fixed BLIS blocking parameters from txt file
    const Value nc = arith::ConstantOp::create(rewriter, 
        loc, rewriter.getIndexAttr(256)); // nc = 256
    const Value kc = arith::ConstantOp::create(rewriter, 
        loc, rewriter.getIndexAttr(128)); // kc = 128
    const Value mc = arith::ConstantOp::create(rewriter, 
        loc, rewriter.getIndexAttr(64)); // mc = 64
    const Value mr = arith::ConstantOp::create(rewriter, 
        loc, rewriter.getIndexAttr(8)); // mr = 8
    const Value nr = arith::ConstantOp::create(rewriter, 
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
    const Value m = memref::DimOp::create(rewriter, loc, A, c0);
    const Value n = memref::DimOp::create(rewriter, loc, C, c1);
    const Value k = memref::DimOp::create(rewriter, loc, A, c1);

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
      i8Mask15 = arith::ConstantOp::create(rewriter, 
          loc, rewriter.getIntegerAttr(rewriter.getI8Type(), 15));
      i8Shift4 = arith::ConstantOp::create(rewriter, 
          loc, rewriter.getIntegerAttr(rewriter.getI8Type(), 4));
    }

    // BLIS 5-loop structure
    // Loop 1: jc - column blocking
    scf::ParallelOp::create(rewriter, 
        loc, c0, n, nc, [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          Value jc = ivs[0];
          // Compute actual nc for this block
          auto jcEnd = arith::AddIOp::create(builder, loc, jc, nc);
          auto jcBound = arith::CmpIOp::create(builder, 
              loc, arith::CmpIPredicate::slt, jcEnd, n);
          auto jcActualEnd =
              arith::SelectOp::create(builder, loc, jcBound, jcEnd, n);
          auto ncActual = arith::SubIOp::create(builder, loc, jcActualEnd, jc);

          // Loop 2: pc - k blocking
          scf::ForOp::create(builder, 
              loc, c0, k, kc, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value pc, ValueRange) {
                // Compute actual kc for this block
                auto pcEnd = arith::AddIOp::create(builder, loc, pc, kc);
                auto pcBound = arith::CmpIOp::create(builder, 
                    loc, arith::CmpIPredicate::slt, pcEnd, k);
                auto pcActualEnd =
                    arith::SelectOp::create(builder, loc, pcBound, pcEnd, k);
                auto kcActual =
                    arith::SubIOp::create(builder, loc, pcActualEnd, pc);

                // Check if we should allocate B_packed (avoid
                // allocation for empty blocks)
                auto kcActualPositive = arith::CmpIOp::create(builder, 
                    loc, arith::CmpIPredicate::sgt, kcActual, c0);
                auto ncActualPositive = arith::CmpIOp::create(builder, 
                    loc, arith::CmpIPredicate::sgt, ncActual, c0);
                auto shouldAllocB = arith::AndIOp::create(builder, 
                    loc, kcActualPositive, ncActualPositive);

                scf::IfOp::create(builder, 
                    loc, shouldAllocB,
                    [&](OpBuilder &builder, Location loc) {
                      // Allocate and pack B block
                      auto B_packedType = MemRefType::get(
                          {ShapedType::kDynamic, ShapedType::kDynamic}, eleTy,
                          AffineMap(), nullptr);
                      Value B_packed = memref::AllocOp::create(builder, 
                          loc, B_packedType, ValueRange{kcActual, ncActual});

                      // Pack B block
                      // clang-format off
                    scf::ForOp::create(builder, 
                        loc, c0, kcActual, c1, ValueRange{},
                        [&](OpBuilder &builder, Location loc, Value kp, ValueRange) {
                          scf::ForOp::create(builder, 
                              loc, c0, ncActual, c1, ValueRange{},
                              [&](OpBuilder &builder, Location loc, Value j, ValueRange) {
                                auto bRowIdx = arith::AddIOp::create(builder, loc, pc, kp);
                                auto bColIdx = arith::AddIOp::create(builder, loc, jc, j);

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
                                    scaleScalar = memref::LoadOp::create(builder, 
                                        loc, dequantChain->scale, scaleIdx);
                                  }

                                  Value weightI8 = nullptr;
                                  if (int4Info) {
                                    // packed index = bColIdx / 2
                                    Value packedColIdx = arith::DivUIOp::create(builder, loc, bColIdx, c2);
                                    Value packedByte = memref::LoadOp::create(builder, 
                                        loc, int4Info->packedWeight, ValueRange{bRowIdx, packedColIdx});
                                    // low = ((packed & 0x0F) << 4) >> 4
                                    Value lowMasked = arith::AndIOp::create(builder, loc, packedByte, i8Mask15);
                                    Value lowShifted = arith::ShLIOp::create(builder, loc, lowMasked, i8Shift4);
                                    Value low = arith::ShRSIOp::create(builder, loc, lowShifted, i8Shift4);
                                    // high = packed >> 4
                                    Value high = arith::ShRSIOp::create(builder, loc, packedByte, i8Shift4);
                                    // select based on parity
                                    Value rem = arith::RemUIOp::create(builder, loc, bColIdx, c2);
                                    Value isLow = arith::CmpIOp::create(builder, 
                                        loc, arith::CmpIPredicate::eq, rem, c0);
                                    weightI8 = arith::SelectOp::create(builder, loc, isLow, low, high);
                                  } else {
                                    weightI8 = memref::LoadOp::create(builder, 
                                        loc, dequantChain->i8Weight, ValueRange{bRowIdx, bColIdx});
                                  }

                                  // i8 -> eleTy, then * scale
                                  Value wFloat = arith::SIToFPOp::create(builder, loc, eleTy, weightI8);
                                  Value wDequant = scaleScalar
                                                       ? arith::MulFOp::create(builder, loc, wFloat, scaleScalar)
                                                       : wFloat;
                                  memref::StoreOp::create(builder, loc, wDequant, B_packed,
                                                                  ValueRange{kp, j});
                                } else {
                                  // Default path: just pack from B.
                                  auto bVal = memref::LoadOp::create(builder, loc, B,
                                                                           ValueRange{bRowIdx, bColIdx});
                                  memref::StoreOp::create(builder, loc, bVal, B_packed,
                                                                ValueRange{kp, j});
                                }
                                scf::YieldOp::create(builder, loc);
                              });
                          scf::YieldOp::create(builder, loc);
                        });
                    // Loop 3: ic - row blocking
                    scf::ParallelOp::create(builder, 
                            loc, c0, m, mc,
                            [&](OpBuilder &builder, Location loc, ValueRange ivs) {
                            Value ic = ivs[0];
                          // Compute actual mc for this block
                          auto icEnd = arith::AddIOp::create(builder, loc, ic, mc);
                          auto icBound = arith::CmpIOp::create(builder, loc,
                                                        arith::CmpIPredicate::slt, icEnd, m);
                          auto icActualEnd = arith::SelectOp::create(builder, loc, icBound, icEnd, m);
                          auto mcActual = arith::SubIOp::create(builder, loc, icActualEnd, ic);

                          // Check if we should allocate A_packed
                          auto mcActualPositive = arith::CmpIOp::create(builder, 
                              loc, arith::CmpIPredicate::sgt, mcActual, c0);
                          auto shouldAllocA = arith::AndIOp::create(builder, 
                              loc, mcActualPositive, kcActualPositive);
                          scf::IfOp::create(builder, loc, shouldAllocA,
                            [&](OpBuilder &builder, Location loc) {
                              // Allocate and pack A block
                              auto A_packedType = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                                                eleTy,AffineMap(),  nullptr);
                              Value A_packed = memref::AllocOp::create(builder, loc, A_packedType,
                                                                             ValueRange{mcActual, kcActual});

                              // Pack A block
                              scf::ForOp::create(builder, 
                                  loc, c0, mcActual, c1, ValueRange{},
                                  [&](OpBuilder &builder, Location loc, Value i, ValueRange) {
                                    scf::ForOp::create(builder, 
                                        loc, c0, kcActual, c1, ValueRange{},
                                        [&](OpBuilder &builder, Location loc, Value kp, ValueRange) {
                                          auto aRowIdx = arith::AddIOp::create(builder, loc, ic, i);
                                          auto aColIdx = arith::AddIOp::create(builder, loc, pc, kp);
                                          auto aVal = memref::LoadOp::create(builder, loc, A,
                                                                                   ValueRange{aRowIdx, aColIdx});
                                          memref::StoreOp::create(builder, loc, aVal, A_packed,
                                                                        ValueRange{i, kp});
                                          scf::YieldOp::create(builder, loc);
                                        });
                                    scf::YieldOp::create(builder, loc);
                                  });

                              // Loop 4: jr - micro column blocking
                              scf::ForOp::create(builder, 
                                  loc, c0, ncActual, nr, ValueRange{},
                                  [&](OpBuilder &builder, Location loc, Value jr, ValueRange) {
                                    auto jrEnd = arith::AddIOp::create(builder, loc, jr, nr);
                                    auto jrBound = arith::CmpIOp::create(builder, 
                                    loc, arith::CmpIPredicate::slt, jrEnd, ncActual);
                                    auto jrActualEnd = arith::SelectOp::create(builder, 
                                    loc, jrBound, jrEnd, ncActual);
                                    auto nrActual = arith::SubIOp::create(builder, 
                                    loc, jrActualEnd, jr);
                                    // Process micro blocks
                                    scf::ForOp::create(builder, 
                                        loc, c0, nrActual, nr, ValueRange{},
                                        [&](OpBuilder &builder, Location loc, Value nIdx, ValueRange) {
                                          auto nIdxEnd = arith::AddIOp::create(builder, loc, nIdx, nr);
                                          auto nIdxBound = arith::CmpIOp::create(builder, 
                                          loc, arith::CmpIPredicate::slt, nIdxEnd, nrActual);
                                          auto nIdxActualEnd = arith::SelectOp::create(builder, 
                                          loc, nIdxBound, nIdxEnd, nrActual);
                                          auto colsToProcess = arith::SubIOp::create(builder, 
                                          loc, nIdxActualEnd, nIdx);
                                          // Check if we can vectorize (at least 32 columns)
                                          auto canVectorize = arith::CmpIOp::create(builder, 
                                              loc, arith::CmpIPredicate::sge, colsToProcess, nr);

                                          scf::IfOp::create(builder, 
                                              loc, canVectorize,
                                              [&](OpBuilder &builder, Location loc) {
                                                // Vectorized path 32
                                                scf::ForOp::create(builder, 
                                                    loc, c0, mcActual, mr, ValueRange{},
                                                    [&](OpBuilder &builder, Location loc, Value ir, ValueRange) {
                                                      auto irEnd = arith::AddIOp::create(builder, loc, ir, mr);
                                                      auto irBound = arith::CmpIOp::create(builder, 
                                                      loc, arith::CmpIPredicate::slt, irEnd, mcActual);
                                                      auto irActualEnd = arith::SelectOp::create(builder, 
                                                      loc, irBound, irEnd, mcActual);
                                                      auto mrActual = arith::SubIOp::create(builder, loc, irActualEnd, ir);
                                                      // Check if we have full 8 rows - FIXED: use mr instead of c8
                                                      auto hasFullRows = arith::CmpIOp::create(builder, 
                                                          loc, arith::CmpIPredicate::sge, mrActual, mr);
                                                      scf::IfOp::create(builder, 
                                                          loc, hasFullRows,
                                                          [&](OpBuilder &builder, Location loc) {
                                                            // Full rows vectorized processing
                                                            auto ir0 = arith::AddIOp::create(builder, loc, ir, c0);
                                                            auto ir1 = arith::AddIOp::create(builder, loc, ir, c1);
                                                            auto ir2 = arith::AddIOp::create(builder, loc, ir, c2);
                                                            auto ir3 = arith::AddIOp::create(builder, loc, ir, c3);
                                                            auto ir4 = arith::AddIOp::create(builder, loc, ir, c4);
                                                            auto ir5 = arith::AddIOp::create(builder, loc, ir, c5);
                                                            auto ir6 = arith::AddIOp::create(builder, loc, ir, c6);
                                                            auto ir7 = arith::AddIOp::create(builder, loc, ir, c7);

                                                            auto sumInit = arith::ConstantOp::create(builder, 
                                                                loc, accVecTy, builder.getZeroAttr(accVecTy));

                                                            auto sumIterVecs = scf::ForOp::create(builder, 
                                                                loc, c0, kcActual, c1,
                                                                ValueRange{sumInit, sumInit, sumInit, sumInit,
                                                                         sumInit, sumInit, sumInit, sumInit},
                                                                [&](OpBuilder &builder, Location loc, Value kInner,
                                                                    ValueRange iterArgs) {
                                                                  // Load A values for 8 rows
                                                                  Value aVal0 = memref::LoadOp::create(builder, 
                                                                      loc, A_packed, ValueRange{ir0, kInner});
                                                                  Value aVal1 = memref::LoadOp::create(builder, 
                                                                      loc, A_packed, ValueRange{ir1, kInner});
                                                                  Value aVal2 = memref::LoadOp::create(builder, 
                                                                      loc, A_packed, ValueRange{ir2, kInner});
                                                                  Value aVal3 = memref::LoadOp::create(builder, 
                                                                      loc, A_packed, ValueRange{ir3, kInner});
                                                                  Value aVal4 = memref::LoadOp::create(builder, 
                                                                      loc, A_packed, ValueRange{ir4, kInner});
                                                                  Value aVal5 = memref::LoadOp::create(builder, 
                                                                      loc, A_packed, ValueRange{ir5, kInner});
                                                                  Value aVal6 = memref::LoadOp::create(builder, 
                                                                      loc, A_packed, ValueRange{ir6, kInner});
                                                                  Value aVal7 = memref::LoadOp::create(builder, 
                                                                      loc, A_packed, ValueRange{ir7, kInner});

                                                                  if (isInteger) {
                                                                    aVal0 = arith::ExtSIOp::create(builder, loc, accEleTy, aVal0);
                                                                    aVal1 = arith::ExtSIOp::create(builder, loc, accEleTy, aVal1);
                                                                    aVal2 = arith::ExtSIOp::create(builder, loc, accEleTy, aVal2);
                                                                    aVal3 = arith::ExtSIOp::create(builder, loc, accEleTy, aVal3);
                                                                    aVal4 = arith::ExtSIOp::create(builder, loc, accEleTy, aVal4);
                                                                    aVal5 = arith::ExtSIOp::create(builder, loc, accEleTy, aVal5);
                                                                    aVal6 = arith::ExtSIOp::create(builder, loc, accEleTy, aVal6);
                                                                    aVal7 = arith::ExtSIOp::create(builder, loc, accEleTy, aVal7);
                                                                  }

                                                                  // Broadcast A values to accumulation vectors
                                                                  auto aVec0 = vector::BroadcastOp::create(builder, 
                                                                      loc, accVecTy, aVal0);
                                                                  auto aVec1 = vector::BroadcastOp::create(builder, 
                                                                      loc, accVecTy, aVal1);
                                                                  auto aVec2 = vector::BroadcastOp::create(builder, 
                                                                      loc, accVecTy, aVal2);
                                                                  auto aVec3 = vector::BroadcastOp::create(builder, 
                                                                      loc, accVecTy, aVal3);
                                                                  auto aVec4 = vector::BroadcastOp::create(builder, 
                                                                      loc, accVecTy, aVal4);
                                                                  auto aVec5 = vector::BroadcastOp::create(builder, 
                                                                      loc, accVecTy, aVal5);
                                                                  auto aVec6 = vector::BroadcastOp::create(builder, 
                                                                      loc, accVecTy, aVal6);
                                                                  auto aVec7 = vector::BroadcastOp::create(builder, 
                                                                      loc, accVecTy, aVal7);

                                                                  // Load B vector and sign-extend for integer
                                                                  auto bColIdx = arith::AddIOp::create(builder, loc, jr, nIdx);
                                                                  Value bVec = vector::LoadOp::create(builder, 
                                                                      loc, vectorTy, B_packed,
                                                                      ValueRange{kInner, bColIdx});
                                                                  if (isInteger)
                                                                    bVec = arith::ExtSIOp::create(builder, loc, accVecTy, bVec);

                                                                  // Multiply-accumulate
                                                                  Value resSumVec0, resSumVec1, resSumVec2, resSumVec3,
                                                                        resSumVec4, resSumVec5, resSumVec6, resSumVec7;
                                                                  if (isInteger) {
                                                                    resSumVec0 = arith::AddIOp::create(builder, loc, iterArgs[0],
                                                                        arith::MulIOp::create(builder, loc, aVec0, bVec));
                                                                    resSumVec1 = arith::AddIOp::create(builder, loc, iterArgs[1],
                                                                        arith::MulIOp::create(builder, loc, aVec1, bVec));
                                                                    resSumVec2 = arith::AddIOp::create(builder, loc, iterArgs[2],
                                                                        arith::MulIOp::create(builder, loc, aVec2, bVec));
                                                                    resSumVec3 = arith::AddIOp::create(builder, loc, iterArgs[3],
                                                                        arith::MulIOp::create(builder, loc, aVec3, bVec));
                                                                    resSumVec4 = arith::AddIOp::create(builder, loc, iterArgs[4],
                                                                        arith::MulIOp::create(builder, loc, aVec4, bVec));
                                                                    resSumVec5 = arith::AddIOp::create(builder, loc, iterArgs[5],
                                                                        arith::MulIOp::create(builder, loc, aVec5, bVec));
                                                                    resSumVec6 = arith::AddIOp::create(builder, loc, iterArgs[6],
                                                                        arith::MulIOp::create(builder, loc, aVec6, bVec));
                                                                    resSumVec7 = arith::AddIOp::create(builder, loc, iterArgs[7],
                                                                        arith::MulIOp::create(builder, loc, aVec7, bVec));
                                                                  } else {
                                                                    resSumVec0 = vector::FMAOp::create(builder, loc, aVec0, bVec, iterArgs[0]);
                                                                    resSumVec1 = vector::FMAOp::create(builder, loc, aVec1, bVec, iterArgs[1]);
                                                                    resSumVec2 = vector::FMAOp::create(builder, loc, aVec2, bVec, iterArgs[2]);
                                                                    resSumVec3 = vector::FMAOp::create(builder, loc, aVec3, bVec, iterArgs[3]);
                                                                    resSumVec4 = vector::FMAOp::create(builder, loc, aVec4, bVec, iterArgs[4]);
                                                                    resSumVec5 = vector::FMAOp::create(builder, loc, aVec5, bVec, iterArgs[5]);
                                                                    resSumVec6 = vector::FMAOp::create(builder, loc, aVec6, bVec, iterArgs[6]);
                                                                    resSumVec7 = vector::FMAOp::create(builder, loc, aVec7, bVec, iterArgs[7]);
                                                                  }

                                                                  scf::YieldOp::create(builder, 
                                                                      loc, ValueRange{resSumVec0, resSumVec1, resSumVec2,
                                                                                      resSumVec3, resSumVec4, resSumVec5,
                                                                                      resSumVec6, resSumVec7});
                                                                });

                                                            // Store results with accumulation
                                                            auto cRow0 = arith::AddIOp::create(builder, loc, ic, ir0);
                                                            auto cRow1 = arith::AddIOp::create(builder, loc, ic, ir1);
                                                            auto cRow2 = arith::AddIOp::create(builder, loc, ic, ir2);
                                                            auto cRow3 = arith::AddIOp::create(builder, loc, ic, ir3);
                                                            auto cRow4 = arith::AddIOp::create(builder, loc, ic, ir4);
                                                            auto cRow5 = arith::AddIOp::create(builder, loc, ic, ir5);
                                                            auto cRow6 = arith::AddIOp::create(builder, loc, ic, ir6);
                                                            auto cRow7 = arith::AddIOp::create(builder, loc, ic, ir7);
                                                            auto cColBase = arith::AddIOp::create(builder, loc, jc, jr);
                                                            auto cColIdx = arith::AddIOp::create(builder, loc, cColBase, nIdx);

                                                            // Load current C values (accumulation type)
                                                            auto cVec0 = vector::LoadOp::create(builder, 
                                                                loc, accVecTy, C, ValueRange{cRow0, cColIdx});
                                                            auto cVec1 = vector::LoadOp::create(builder, 
                                                                loc, accVecTy, C, ValueRange{cRow1, cColIdx});
                                                            auto cVec2 = vector::LoadOp::create(builder, 
                                                                loc, accVecTy, C, ValueRange{cRow2, cColIdx});
                                                            auto cVec3 = vector::LoadOp::create(builder, 
                                                                loc, accVecTy, C, ValueRange{cRow3, cColIdx});
                                                            auto cVec4 = vector::LoadOp::create(builder, 
                                                                loc, accVecTy, C, ValueRange{cRow4, cColIdx});
                                                            auto cVec5 = vector::LoadOp::create(builder, 
                                                                loc, accVecTy, C, ValueRange{cRow5, cColIdx});
                                                            auto cVec6 = vector::LoadOp::create(builder, 
                                                                loc, accVecTy, C, ValueRange{cRow6, cColIdx});
                                                            auto cVec7 = vector::LoadOp::create(builder, 
                                                                loc, accVecTy, C, ValueRange{cRow7, cColIdx});

                                                            // Accumulate: C = C + A*B
                                                            Value finalVec0, finalVec1, finalVec2, finalVec3,
                                                                  finalVec4, finalVec5, finalVec6, finalVec7;
                                                            if (isInteger) {
                                                              finalVec0 = arith::AddIOp::create(builder, loc, cVec0, sumIterVecs.getResult(0));
                                                              finalVec1 = arith::AddIOp::create(builder, loc, cVec1, sumIterVecs.getResult(1));
                                                              finalVec2 = arith::AddIOp::create(builder, loc, cVec2, sumIterVecs.getResult(2));
                                                              finalVec3 = arith::AddIOp::create(builder, loc, cVec3, sumIterVecs.getResult(3));
                                                              finalVec4 = arith::AddIOp::create(builder, loc, cVec4, sumIterVecs.getResult(4));
                                                              finalVec5 = arith::AddIOp::create(builder, loc, cVec5, sumIterVecs.getResult(5));
                                                              finalVec6 = arith::AddIOp::create(builder, loc, cVec6, sumIterVecs.getResult(6));
                                                              finalVec7 = arith::AddIOp::create(builder, loc, cVec7, sumIterVecs.getResult(7));
                                                            } else {
                                                              finalVec0 = arith::AddFOp::create(builder, loc, cVec0, sumIterVecs.getResult(0));
                                                              finalVec1 = arith::AddFOp::create(builder, loc, cVec1, sumIterVecs.getResult(1));
                                                              finalVec2 = arith::AddFOp::create(builder, loc, cVec2, sumIterVecs.getResult(2));
                                                              finalVec3 = arith::AddFOp::create(builder, loc, cVec3, sumIterVecs.getResult(3));
                                                              finalVec4 = arith::AddFOp::create(builder, loc, cVec4, sumIterVecs.getResult(4));
                                                              finalVec5 = arith::AddFOp::create(builder, loc, cVec5, sumIterVecs.getResult(5));
                                                              finalVec6 = arith::AddFOp::create(builder, loc, cVec6, sumIterVecs.getResult(6));
                                                              finalVec7 = arith::AddFOp::create(builder, loc, cVec7, sumIterVecs.getResult(7));
                                                            }

                                                            // Store back to C
                                                            vector::StoreOp::create(builder, 
                                                                loc, finalVec0, C, ValueRange{cRow0, cColIdx});
                                                            vector::StoreOp::create(builder, 
                                                                loc, finalVec1, C, ValueRange{cRow1, cColIdx});
                                                            vector::StoreOp::create(builder, 
                                                                loc, finalVec2, C, ValueRange{cRow2, cColIdx});
                                                            vector::StoreOp::create(builder, 
                                                                loc, finalVec3, C, ValueRange{cRow3, cColIdx});
                                                            vector::StoreOp::create(builder, 
                                                                loc, finalVec4, C, ValueRange{cRow4, cColIdx});
                                                            vector::StoreOp::create(builder, 
                                                                loc, finalVec5, C, ValueRange{cRow5, cColIdx});
                                                            vector::StoreOp::create(builder, 
                                                                loc, finalVec6, C, ValueRange{cRow6, cColIdx});
                                                            vector::StoreOp::create(builder, 
                                                                loc, finalVec7, C, ValueRange{cRow7, cColIdx});

                                                            scf::YieldOp::create(builder, loc);
                                                          },
                                                          [&](OpBuilder &builder, Location loc) {
                                                            // Scalar path for incomplete rows
                                                            scf::ForOp::create(builder, 
                                                                loc, c0, nr, c1, ValueRange{},
                                                                [&](OpBuilder &builder, Location loc, Value jj, ValueRange) {
                                                                  scf::ForOp::create(builder, 
                                                                      loc, ir, irActualEnd, c1, ValueRange{},
                                                                      [&](OpBuilder &builder, Location loc, Value ii, ValueRange) {
                                                                        Value sumInit;
                                                                        if (isInteger)
                                                                          sumInit = arith::ConstantOp::create(builder, 
                                                                              loc, accEleTy, builder.getIntegerAttr(accEleTy, 0));
                                                                        else
                                                                          sumInit = arith::ConstantOp::create(builder, 
                                                                              loc, accEleTy, builder.getFloatAttr(accEleTy, 0.0));
                                                                        auto sumIter = scf::ForOp::create(builder, 
                                                                            loc, c0, kcActual, c1,
                                                                            ValueRange{sumInit},
                                                                            [&](OpBuilder &builder, Location loc, Value kInner,
                                                                                ValueRange iterArgs) {
                                                                              Value aVal = memref::LoadOp::create(builder, 
                                                                                  loc, A_packed, ValueRange{ii, kInner});
                                                                              auto bColBase = arith::AddIOp::create(builder, loc, jr, nIdx);
                                                                              auto bColIdx = arith::AddIOp::create(builder, loc, bColBase, jj);
                                                                              Value bVal = memref::LoadOp::create(builder, 
                                                                                  loc, B_packed, ValueRange{kInner, bColIdx});
                                                                              Value prod, newSum;
                                                                              if (isInteger) {
                                                                                aVal = arith::ExtSIOp::create(builder, loc, accEleTy, aVal);
                                                                                bVal = arith::ExtSIOp::create(builder, loc, accEleTy, bVal);
                                                                                prod = arith::MulIOp::create(builder, loc, aVal, bVal);
                                                                                newSum = arith::AddIOp::create(builder, loc, iterArgs[0], prod);
                                                                              } else {
                                                                                prod = arith::MulFOp::create(builder, loc, aVal, bVal);
                                                                                newSum = arith::AddFOp::create(builder, loc, iterArgs[0], prod);
                                                                              }
                                                                              scf::YieldOp::create(builder, loc, ValueRange{newSum});
                                                                            });
                                                                        auto cRowIdx = arith::AddIOp::create(builder, loc, ic, ii);
                                                                        auto cColBase = arith::AddIOp::create(builder, loc, jc, jr);
                                                                        auto cColBase2 = arith::AddIOp::create(builder, loc, cColBase, nIdx);
                                                                        auto cColIdx = arith::AddIOp::create(builder, loc, cColBase2, jj);
                                                                        auto currentVal = memref::LoadOp::create(builder, 
                                                                            loc, C, ValueRange{cRowIdx, cColIdx});
                                                                        Value finalSum;
                                                                        if (isInteger)
                                                                          finalSum = arith::AddIOp::create(builder, loc, currentVal, sumIter.getResult(0));
                                                                        else
                                                                          finalSum = arith::AddFOp::create(builder, loc, currentVal, sumIter.getResult(0));
                                                                        memref::StoreOp::create(builder, 
                                                                            loc, finalSum, C, ValueRange{cRowIdx, cColIdx});
                                                                        scf::YieldOp::create(builder, loc);
                                                                      });
                                                                  scf::YieldOp::create(builder, loc);
                                                                });
                                                            scf::YieldOp::create(builder, loc);
                                                          });

                                                      scf::YieldOp::create(builder, loc);
                                                    });
                                                scf::YieldOp::create(builder, loc);
                                              },
                                              [&](OpBuilder &builder, Location loc) {
                                                // Scalar path for incomplete columns
                                                scf::ForOp::create(builder, 
                                                    loc, nIdx, nIdxActualEnd, c1, ValueRange{},
                                                    [&](OpBuilder &builder, Location loc, Value nIdxTail, ValueRange) {
                                                      scf::ForOp::create(builder, 
                                                          loc, c0, mcActual, mr, ValueRange{},
                                                          [&](OpBuilder &builder, Location loc, Value ir, ValueRange) {
                                                            auto irEnd = arith::AddIOp::create(builder, loc, ir, mr);
                                                            auto irBound = arith::CmpIOp::create(builder, 
                                                            loc, arith::CmpIPredicate::slt, irEnd, mcActual);
                                                            auto irActualEnd = arith::SelectOp::create(builder, 
                                                            loc, irBound, irEnd, mcActual);

                                                            scf::ForOp::create(builder, 
                                                                loc, ir, irActualEnd, c1, ValueRange{},
                                                                [&](OpBuilder &builder, Location loc, Value ii, ValueRange) {
                                                                  Value sumInit;
                                                                  if (isInteger)
                                                                    sumInit = arith::ConstantOp::create(builder, 
                                                                        loc, accEleTy, builder.getIntegerAttr(accEleTy, 0));
                                                                  else
                                                                    sumInit = arith::ConstantOp::create(builder, 
                                                                        loc, accEleTy, builder.getFloatAttr(accEleTy, 0.0));
                                                                  auto sumIter = scf::ForOp::create(builder, 
                                                                      loc, c0, kcActual, c1,
                                                                      ValueRange{sumInit},
                                                                      [&](OpBuilder &builder, Location loc, Value kInner,
                                                                          ValueRange iterArgs) {
                                                                        Value aVal = memref::LoadOp::create(builder, 
                                                                            loc, A_packed, ValueRange{ii, kInner});
                                                                        auto bColBase = arith::AddIOp::create(builder, loc, jr, nIdxTail);
                                                                        Value bVal = memref::LoadOp::create(builder, 
                                                                            loc, B_packed, ValueRange{kInner, bColBase});
                                                                        Value prod, newSum;
                                                                        if (isInteger) {
                                                                          aVal = arith::ExtSIOp::create(builder, loc, accEleTy, aVal);
                                                                          bVal = arith::ExtSIOp::create(builder, loc, accEleTy, bVal);
                                                                          prod = arith::MulIOp::create(builder, loc, aVal, bVal);
                                                                          newSum = arith::AddIOp::create(builder, loc, iterArgs[0], prod);
                                                                        } else {
                                                                          prod = arith::MulFOp::create(builder, loc, aVal, bVal);
                                                                          newSum = arith::AddFOp::create(builder, loc, iterArgs[0], prod);
                                                                        }
                                                                        scf::YieldOp::create(builder, loc, ValueRange{newSum});
                                                                      });
                                                                  auto cRowIdx = arith::AddIOp::create(builder, loc, ic, ii);
                                                                  auto cColBase = arith::AddIOp::create(builder, loc, jc, jr);
                                                                  auto cColIdx = arith::AddIOp::create(builder, loc, cColBase, nIdxTail);
                                                                  auto currentVal = memref::LoadOp::create(builder, 
                                                                      loc, C, ValueRange{cRowIdx, cColIdx});
                                                                  Value finalSum;
                                                                  if (isInteger)
                                                                    finalSum = arith::AddIOp::create(builder, loc, currentVal, sumIter.getResult(0));
                                                                  else
                                                                    finalSum = arith::AddFOp::create(builder, loc, currentVal, sumIter.getResult(0));
                                                                  memref::StoreOp::create(builder, 
                                                                      loc, finalSum, C, ValueRange{cRowIdx, cColIdx});
                                                                  scf::YieldOp::create(builder, loc);
                                                                });
                                                            scf::YieldOp::create(builder, loc);
                                                          });
                                                      scf::YieldOp::create(builder, loc);
                                                    });
                                                scf::YieldOp::create(builder, loc);
                                              });

                                          scf::YieldOp::create(builder, loc);
                                        });
                                    scf::YieldOp::create(builder, loc);
                                  });
                                  // clang-format on
                                  // Deallocate A_packed
                                  memref::DeallocOp::create(builder, loc,
                                                                    A_packed);
                                  scf::YieldOp::create(builder, loc);
                                },
                                [&](OpBuilder &builder, Location loc) {
                                  // Skip A_packed allocation for empty blocks
                                  scf::YieldOp::create(builder, loc);
                                });
                          });

                      // Deallocate B_packed
                      memref::DeallocOp::create(builder, loc, B_packed);
                      scf::YieldOp::create(builder, loc);
                    },
                    [&](OpBuilder &builder, Location loc) {
                      // Skip B_packed allocation for empty blocks
                      scf::YieldOp::create(builder, loc);
                    });

                scf::YieldOp::create(builder, loc);
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
