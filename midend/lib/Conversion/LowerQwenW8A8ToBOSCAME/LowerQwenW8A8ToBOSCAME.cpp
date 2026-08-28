//===- LowerQwenW8A8ToBOSCAME.cpp - Qwen3 W8A8 to AME ---------*- C++ -*-===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialect/BOSCAME/BOSCAMEDialect.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"

using namespace mlir;
using namespace buddy::boscame;

namespace {

constexpr StringLiteral kZeroGlobal = "__buddy_qwen_w8a8_zero_f32";
constexpr StringLiteral kScratchGlobal = "__buddy_qwen_w8a8_scratch_f32";
constexpr StringLiteral kTailActivationGlobal =
    "__buddy_qwen_w8a8_tail_activation_i8";
constexpr int64_t kOutBlock = 64;
constexpr int64_t kHardwareM = 16;
constexpr int64_t kPrefillRows = 2 * kHardwareM;
constexpr int64_t kHardwareN = 16;
constexpr int64_t kHardwareK = 64;
constexpr int64_t kMTypeI8 = (1LL << 16) | (1LL << 4);
constexpr int64_t kMTypeI32 = (1LL << 16) | (1LL << 6) | 2;
constexpr StringLiteral kProfileStart = "buddyTraceCycleStartPath";
constexpr StringLiteral kProfileEnd = "buddyTraceCycleEndPath";
constexpr int64_t kProfileLinearTotal = 251;
constexpr int64_t kProfileActivationQuantize = 252;
constexpr int64_t kProfileOutputZero = 253;
constexpr int64_t kProfileAMEKernel = 254;
constexpr int64_t kProfileRVVAccumulation = 255;
constexpr StringLiteral kRvvAccumulateN64 =
    "buddy_w8a8_rvv_accumulate_n64";
constexpr int64_t kProfileQuantizeAmax = 249;
constexpr int64_t kProfileQuantizeWrite = 250;

static Value indexConstant(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(builder, loc, value);
}

static Value i64Constant(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantOp::create(builder, loc, builder.getI64Type(),
                                   builder.getI64IntegerAttr(value));
}

static Value f32Constant(OpBuilder &builder, Location loc, float value) {
  return arith::ConstantOp::create(builder, loc, builder.getF32Type(),
                                   builder.getF32FloatAttr(value));
}

static Value makeSubview(PatternRewriter &rewriter, Location loc, Value source,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<int64_t> sizes) {
  SmallVector<OpFoldResult> sizeValues;
  SmallVector<OpFoldResult> strides;
  for (int64_t size : sizes) {
    sizeValues.push_back(rewriter.getIndexAttr(size));
    strides.push_back(rewriter.getIndexAttr(1));
  }
  return memref::SubViewOp::create(rewriter, loc, source, offsets, sizeValues,
                                   strides);
}

static void emitProfileCall(PatternRewriter &rewriter, Location loc,
                            StringRef callee, int64_t id) {
  SmallVector<Value> arguments;
  arguments.push_back(i64Constant(rewriter, loc, id));
  arguments.push_back(i64Constant(rewriter, loc, 1));
  arguments.push_back(i64Constant(rewriter, loc, id));
  for (int64_t unused = 0; unused < 3; ++unused)
    arguments.push_back(i64Constant(rewriter, loc, -1));
  func::CallOp::create(rewriter, loc, callee, TypeRange{}, arguments);
}

class QuantizePerGroupLowering : public OpRewritePattern<QuantizePerGroupOp> {
public:
  QuantizePerGroupLowering(MLIRContext *context, bool profilePhases)
      : OpRewritePattern<QuantizePerGroupOp>(context),
        profilePhases(profilePhases) {}

  LogicalResult matchAndRewrite(QuantizePerGroupOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<MemRefType>(op.getInput().getType());
    auto quantizedType = dyn_cast<MemRefType>(op.getQuantized().getType());
    auto scalesType = dyn_cast<MemRefType>(op.getScales().getType());
    if (!inputType || !quantizedType || !scalesType)
      return op.emitOpError(
          "must be bufferized before --lower-qwen-w8a8-to-boscame");

    Location loc = op.getLoc();
    int64_t tokens = inputType.getDimSize(0);
    int64_t width = inputType.getDimSize(1);
    int64_t groupSize = op.getGroupSize();
    int64_t groups = width / groupSize;
    Value c0 = indexConstant(rewriter, loc, 0);
    Value c1 = indexConstant(rewriter, loc, 1);
    Value tokenBound = indexConstant(rewriter, loc, tokens);
    Value groupBound = indexConstant(rewriter, loc, groups);
    Value elementBound = indexConstant(rewriter, loc, groupSize);
    Value groupSizeIndex = indexConstant(rewriter, loc, groupSize);
    Value zeroF = f32Constant(rewriter, loc, 0.0f);
    Value oneF = f32Constant(rewriter, loc, 1.0f);
    Value halfF = f32Constant(rewriter, loc, 0.5f);
    Value minusHalfF = f32Constant(rewriter, loc, -0.5f);
    Value c127F = f32Constant(rewriter, loc, 127.0f);
    Value minus127I = arith::ConstantIntOp::create(rewriter, loc, -127, 32);
    Value c127I = arith::ConstantIntOp::create(rewriter, loc, 127, 32);
    if (profilePhases)
      emitProfileCall(rewriter, loc, kProfileStart, kProfileActivationQuantize);

    auto tokenLoop = scf::ForOp::create(rewriter, loc, c0, tokenBound, c1);
    rewriter.setInsertionPointToStart(tokenLoop.getBody());
    Value token = tokenLoop.getInductionVar();
    auto groupLoop = scf::ForOp::create(rewriter, loc, c0, groupBound, c1);
    rewriter.setInsertionPointToStart(groupLoop.getBody());
    Value group = groupLoop.getInductionVar();
    Value groupBase =
        arith::MulIOp::create(rewriter, loc, group, groupSizeIndex);

    if (profilePhases)
      emitProfileCall(rewriter, loc, kProfileStart, kProfileQuantizeAmax);
    auto maxLoop = scf::ForOp::create(rewriter, loc, c0, elementBound, c1,
                                      ValueRange{zeroF});
    Block *maxBody = maxLoop.getBody();
    rewriter.setInsertionPointToStart(maxBody);
    Value element = maxLoop.getInductionVar();
    Value inputColumn =
        arith::AddIOp::create(rewriter, loc, groupBase, element);
    Value inputValue = memref::LoadOp::create(rewriter, loc, op.getInput(),
                                              ValueRange{token, inputColumn});
    Value absolute = math::AbsFOp::create(rewriter, loc, inputValue);
    Value larger = arith::CmpFOp::create(
        rewriter, loc, arith::CmpFPredicate::OGT, absolute,
        maxLoop.getRegionIterArgs().front());
    Value maximum = arith::SelectOp::create(
        rewriter, loc, larger, absolute, maxLoop.getRegionIterArgs().front());
    scf::YieldOp::create(rewriter, loc, maximum);

    rewriter.setInsertionPointAfter(maxLoop);
    if (profilePhases)
      emitProfileCall(rewriter, loc, kProfileEnd, kProfileQuantizeAmax);
    Value amax = maxLoop.getResult(0);
    Value nonzero = arith::CmpFOp::create(
        rewriter, loc, arith::CmpFPredicate::OGT, amax, zeroF);
    Value rawScale = arith::DivFOp::create(rewriter, loc, amax, c127F);
    Value scale =
        arith::SelectOp::create(rewriter, loc, nonzero, rawScale, oneF);
    memref::StoreOp::create(rewriter, loc, scale, op.getScales(),
                            ValueRange{token, group});

    if (profilePhases)
      emitProfileCall(rewriter, loc, kProfileStart, kProfileQuantizeWrite);
    auto quantLoop = scf::ForOp::create(rewriter, loc, c0, elementBound, c1);
    rewriter.setInsertionPointToStart(quantLoop.getBody());
    Value quantElement = quantLoop.getInductionVar();
    Value quantColumn =
        arith::AddIOp::create(rewriter, loc, groupBase, quantElement);
    Value value = memref::LoadOp::create(rewriter, loc, op.getInput(),
                                         ValueRange{token, quantColumn});
    Value scaled = arith::DivFOp::create(rewriter, loc, value, scale);
    Value positive = arith::CmpFOp::create(
        rewriter, loc, arith::CmpFPredicate::OGE, scaled, zeroF);
    Value adjustment =
        arith::SelectOp::create(rewriter, loc, positive, halfF, minusHalfF);
    Value adjusted = arith::AddFOp::create(rewriter, loc, scaled, adjustment);
    Value rounded =
        arith::FPToSIOp::create(rewriter, loc, rewriter.getI32Type(), adjusted);
    Value clampedLow =
        arith::MaxSIOp::create(rewriter, loc, rounded, minus127I);
    Value clamped = arith::MinSIOp::create(rewriter, loc, clampedLow, c127I);
    Value quantized = arith::TruncIOp::create(
        rewriter, loc, rewriter.getIntegerType(8), clamped);
    memref::StoreOp::create(rewriter, loc, quantized, op.getQuantized(),
                            ValueRange{token, quantColumn});

    rewriter.setInsertionPointAfter(quantLoop);
    if (profilePhases)
      emitProfileCall(rewriter, loc, kProfileEnd, kProfileQuantizeWrite);

    rewriter.setInsertionPointAfter(tokenLoop);
    if (profilePhases)
      emitProfileCall(rewriter, loc, kProfileEnd, kProfileActivationQuantize);
    rewriter.eraseOp(op);
    return success();
  }

private:
  bool profilePhases;
};

// Kept as an opt-in scalar/N64 fallback for FPGA A/B diagnosis.  The optimized
// lowering below is registered by default.
class W8A8LinearScalarFallback : public OpRewritePattern<W8A8LinearOp> {
public:
  using OpRewritePattern<W8A8LinearOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(W8A8LinearOp op,
                                PatternRewriter &rewriter) const override {
    auto xqType = dyn_cast<MemRefType>(op.getXq().getType());
    auto xsType = dyn_cast<MemRefType>(op.getXs().getType());
    auto wqType = dyn_cast<MemRefType>(op.getWq().getType());
    auto wsType = dyn_cast<MemRefType>(op.getWs().getType());
    auto outputType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!xqType || !xsType || !wqType || !wsType || !outputType)
      return op.emitOpError(
          "must be bufferized before --lower-qwen-w8a8-to-boscame");
    Location loc = op.getLoc();
    int64_t tokens = xqType.getDimSize(0);
    int64_t width = xqType.getDimSize(1);
    int64_t groups = xsType.getDimSize(1);
    int64_t outputWidth = outputType.getDimSize(1);
    int64_t outputBlocks = outputWidth / kOutBlock;
    int64_t groupSize = op.getGroupSize();
    auto scratchType =
        MemRefType::get({kPrefillRows, kOutBlock}, rewriter.getF32Type());
    auto tailActivationType =
        MemRefType::get({kHardwareM, kHardwareK}, rewriter.getI8Type());
    Value zero =
        memref::GetGlobalOp::create(rewriter, loc, scratchType, kZeroGlobal);
    Value scratch =
        memref::GetGlobalOp::create(rewriter, loc, scratchType, kScratchGlobal);
    Value tailActivation = memref::GetGlobalOp::create(
        rewriter, loc, tailActivationType, kTailActivationGlobal);

    Value c0 = indexConstant(rewriter, loc, 0);
    Value c1 = indexConstant(rewriter, loc, 1);
    Value c16 = indexConstant(rewriter, loc, kHardwareM);
    Value c64 = indexConstant(rewriter, loc, kHardwareK);
    Value tokenBound = indexConstant(rewriter, loc, tokens);
    Value outputBound = indexConstant(rewriter, loc, outputWidth);
    Value blockBound = indexConstant(rewriter, loc, outputBlocks);
    Value groupBound = indexConstant(rewriter, loc, groups);
    Value groupSizeIndex = indexConstant(rewriter, loc, groupSize);
    Value zeroF = f32Constant(rewriter, loc, 0.0f);
    Value zeroI8 = arith::ConstantIntOp::create(rewriter, loc, 0, 8);
    Value strideA = i64Constant(rewriter, loc, width);
    Value strideB = i64Constant(rewriter, loc, groupSize);
    Value strideC = i64Constant(rewriter, loc, kOutBlock * sizeof(float));
    Value strideOneI8 = i64Constant(rewriter, loc, sizeof(int8_t));
    Value strideOneF32 = i64Constant(rewriter, loc, sizeof(float));
    Value oneI64 = i64Constant(rewriter, loc, 1);
    Value nSixteen = i64Constant(rewriter, loc, kHardwareN);
    Value kSixtyFour = i64Constant(rewriter, loc, kHardwareK);
    Value typeI8 = i64Constant(rewriter, loc, kMTypeI8);
    Value typeI32 = i64Constant(rewriter, loc, kMTypeI32);

    // The FPGA bare-metal CRT does not clear .bss.  Materialize the persistent
    // accumulator seed before the first AME load instead of relying on the
    // LLVM zeroinitializer surviving placement in a NOBITS section.
    Value mSeedBound = indexConstant(rewriter, loc, kHardwareM);
    auto zeroSeedRows = scf::ForOp::create(rewriter, loc, c0, mSeedBound, c1);
    rewriter.setInsertionPointToStart(zeroSeedRows.getBody());
    auto zeroSeedColumns = scf::ForOp::create(rewriter, loc, c0, c64, c1);
    rewriter.setInsertionPointToStart(zeroSeedColumns.getBody());
    memref::StoreOp::create(rewriter, loc, zeroF, zero,
                            ValueRange{zeroSeedRows.getInductionVar(),
                                       zeroSeedColumns.getInductionVar()});
    rewriter.setInsertionPointAfter(zeroSeedRows);

    // Match ame_i8_resync_state() in Qwen3's validated AME backend.  A CPU
    // reset does not necessarily reset the accelerator's internal mstart and
    // tile state, so every complete W8A8 operation begins with one minimal
    // legal loadA/loadB/MMA/loadC/storeC round trip.  This is an operation
    // boundary synchronization, not a per-group fence.
    memref::StoreOp::create(rewriter, loc, zeroI8, tailActivation,
                            ValueRange{c0, c0});
    MSettilemOp::create(rewriter, loc, rewriter.getI64Type(), oneI64);
    MSettilenOp::create(rewriter, loc, rewriter.getI64Type(), oneI64);
    MSettilekOp::create(rewriter, loc, rewriter.getI64Type(), oneI64);
    MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), typeI8);
    Value syncI8 = makeSubview(rewriter, loc, tailActivation,
                               ArrayRef<OpFoldResult>{rewriter.getIndexAttr(0),
                                                      rewriter.getIndexAttr(0)},
                               ArrayRef<int64_t>{1, 1});
    Mlae8mOp::create(rewriter, loc, 0, syncI8, strideOneI8);
    Mlbte8mOp::create(rewriter, loc, 1, syncI8, strideOneI8);
    MqmaBmmOp::create(rewriter, loc, 0, 0, 1);
    MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), typeI32);
    Value syncZero =
        makeSubview(rewriter, loc, zero,
                    ArrayRef<OpFoldResult>{rewriter.getIndexAttr(0),
                                           rewriter.getIndexAttr(0)},
                    ArrayRef<int64_t>{1, 1});
    Value syncScratch =
        makeSubview(rewriter, loc, scratch,
                    ArrayRef<OpFoldResult>{rewriter.getIndexAttr(0),
                                           rewriter.getIndexAttr(0)},
                    ArrayRef<int64_t>{1, 1});
    Mlce32mOp::create(rewriter, loc, 0, syncZero, strideOneF32);
    Msce32mOp::create(rewriter, loc, 0, syncScratch, strideOneF32);
    LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::seq_cst);

    auto zeroRows = scf::ForOp::create(rewriter, loc, c0, tokenBound, c1);
    rewriter.setInsertionPointToStart(zeroRows.getBody());
    auto zeroColumns = scf::ForOp::create(rewriter, loc, c0, outputBound, c1);
    rewriter.setInsertionPointToStart(zeroColumns.getBody());
    memref::StoreOp::create(
        rewriter, loc, zeroF, op.getOutput(),
        ValueRange{zeroRows.getInductionVar(), zeroColumns.getInductionVar()});
    rewriter.setInsertionPointAfter(zeroRows);

    // N and K stay fixed for the full operation.  Match the validated Qwen3
    // backend by programming M to the exact number of rows in each static
    // token tile, including decode M=1 and short prefill tails.
    MSettilenOp::create(rewriter, loc, rewriter.getI64Type(), nSixteen);
    MSettilekOp::create(rewriter, loc, rewriter.getI64Type(), kSixtyFour);

    auto emitTokenTile = [&](Value tokenBase, int64_t tileRows) {
      Value tileM = i64Constant(rewriter, loc, tileRows);
      Value tileRowBound = indexConstant(rewriter, loc, tileRows);
      MSettilemOp::create(rewriter, loc, rewriter.getI64Type(), tileM);

      auto blockLoop = scf::ForOp::create(rewriter, loc, c0, blockBound, c1);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      Value outputBlock = blockLoop.getInductionVar();
      Value outputBase = arith::MulIOp::create(rewriter, loc, outputBlock, c64);
      auto groupLoop = scf::ForOp::create(rewriter, loc, c0, groupBound, c1);
      rewriter.setInsertionPointToStart(groupLoop.getBody());
      Value group = groupLoop.getInductionVar();

      MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), typeI32);
      for (int64_t n = 0; n < 4; ++n) {
        Value zeroTile = makeSubview(
            rewriter, loc, zero,
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(0),
                                   rewriter.getIndexAttr(n * kHardwareN)},
            ArrayRef<int64_t>{tileRows, kHardwareN});
        Mlce32mOp::create(rewriter, loc, n, zeroTile, strideC);
      }

      Value groupBase =
          arith::MulIOp::create(rewriter, loc, group, groupSizeIndex);
      auto kLoop = scf::ForOp::create(rewriter, loc, c0, groupSizeIndex, c64);
      rewriter.setInsertionPointToStart(kLoop.getBody());
      Value kOffset = kLoop.getInductionVar();
      Value activationOffset =
          arith::AddIOp::create(rewriter, loc, groupBase, kOffset);
      Value activationTile =
          makeSubview(rewriter, loc, op.getXq(),
                      ArrayRef<OpFoldResult>{tokenBase, activationOffset},
                      ArrayRef<int64_t>{tileRows, kHardwareK});
      MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), typeI8);
      Mlae8mOp::create(rewriter, loc, 0, activationTile, strideA);
      SmallVector<Value, 4> weightTiles;
      for (int64_t n = 0; n < 4; ++n) {
        weightTiles.push_back(makeSubview(
            rewriter, loc, op.getWq(),
            ArrayRef<OpFoldResult>{outputBlock, group,
                                   rewriter.getIndexAttr(n * kHardwareN),
                                   kOffset},
            ArrayRef<int64_t>{1, 1, kHardwareN, kHardwareK}));
      }

      // Match Qwen3's validated 1A x 4B schedule.  Every mqma produces one
      // tileRows x N16 tile.
      Mlbe8mOp::create(rewriter, loc, 4, weightTiles[0], strideB);
      Mlbe8mOp::create(rewriter, loc, 5, weightTiles[1], strideB);
      MqmaBmmOp::create(rewriter, loc, 0, 0, 4);
      Mlbe8mOp::create(rewriter, loc, 6, weightTiles[2], strideB);
      MqmaBmmOp::create(rewriter, loc, 1, 0, 5);
      Mlbe8mOp::create(rewriter, loc, 7, weightTiles[3], strideB);
      MqmaBmmOp::create(rewriter, loc, 2, 0, 6);
      MqmaBmmOp::create(rewriter, loc, 3, 0, 7);

      rewriter.setInsertionPointAfter(kLoop);
      MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), typeI32);
      for (int64_t n = 0; n < 4; ++n) {
        Value scratchTile = makeSubview(
            rewriter, loc, scratch,
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(0),
                                   rewriter.getIndexAttr(n * kHardwareN)},
            ArrayRef<int64_t>{tileRows, kHardwareN});
        Msce32mOp::create(rewriter, loc, n, scratchTile, strideC);
      }

      // The AME stores above are asynchronous with respect to scalar CPU
      // loads.  Qwen3's validated backend places this fence after the four
      // msce32.m instructions, before consuming the fp32 dot-product tile for
      // scale accumulation.  Keep the synchronization at that exact producer /
      // consumer boundary rather than fencing individual AME instructions.
      LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::seq_cst);

      auto accumulateRows =
          scf::ForOp::create(rewriter, loc, c0, tileRowBound, c1);
      rewriter.setInsertionPointToStart(accumulateRows.getBody());
      Value rowInTile = accumulateRows.getInductionVar();
      Value token = arith::AddIOp::create(rewriter, loc, tokenBase, rowInTile);
      auto accumulateColumns = scf::ForOp::create(rewriter, loc, c0, c64, c1);
      rewriter.setInsertionPointToStart(accumulateColumns.getBody());
      Value columnInBlock = accumulateColumns.getInductionVar();
      Value outputColumn =
          arith::AddIOp::create(rewriter, loc, outputBase, columnInBlock);
      Value dot = memref::LoadOp::create(rewriter, loc, scratch,
                                         ValueRange{rowInTile, columnInBlock});
      Value activationScale = memref::LoadOp::create(rewriter, loc, op.getXs(),
                                                     ValueRange{token, group});
      Value weightScale = memref::LoadOp::create(
          rewriter, loc, op.getWs(), ValueRange{group, outputColumn});
      Value oldOutput = memref::LoadOp::create(rewriter, loc, op.getOutput(),
                                               ValueRange{token, outputColumn});
      Value scaledDot =
          arith::MulFOp::create(rewriter, loc, dot, activationScale);
      scaledDot = arith::MulFOp::create(rewriter, loc, scaledDot, weightScale);
      Value accumulated =
          arith::AddFOp::create(rewriter, loc, oldOutput, scaledDot);
      memref::StoreOp::create(rewriter, loc, accumulated, op.getOutput(),
                              ValueRange{token, outputColumn});

      rewriter.setInsertionPointAfter(blockLoop);
    };

    int64_t fullTiles = tokens / kHardwareM;
    int64_t tailRows = tokens % kHardwareM;
    if (fullTiles != 0) {
      Value fullTileBound = indexConstant(rewriter, loc, fullTiles);
      auto tokenTileLoop =
          scf::ForOp::create(rewriter, loc, c0, fullTileBound, c1);
      rewriter.setInsertionPointToStart(tokenTileLoop.getBody());
      Value tokenBase = arith::MulIOp::create(
          rewriter, loc, tokenTileLoop.getInductionVar(), c16);
      emitTokenTile(tokenBase, kHardwareM);
      rewriter.setInsertionPointAfter(tokenTileLoop);
    }
    if (tailRows != 0) {
      Value tailBase = indexConstant(rewriter, loc, fullTiles * kHardwareM);
      emitTokenTile(tailBase, tailRows);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

class W8A8LinearLowering : public OpRewritePattern<W8A8LinearOp> {
public:
  W8A8LinearLowering(MLIRContext *context, bool profilePhases)
      : OpRewritePattern<W8A8LinearOp>(context), profilePhases(profilePhases) {}

  LogicalResult matchAndRewrite(W8A8LinearOp op,
                                PatternRewriter &rewriter) const override {
    auto xqType = dyn_cast<MemRefType>(op.getXq().getType());
    auto xsType = dyn_cast<MemRefType>(op.getXs().getType());
    auto wqType = dyn_cast<MemRefType>(op.getWq().getType());
    auto wsType = dyn_cast<MemRefType>(op.getWs().getType());
    auto outputType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!xqType || !xsType || !wqType || !wsType || !outputType)
      return op.emitOpError(
          "must be bufferized before --lower-qwen-w8a8-to-boscame");

    Location loc = op.getLoc();
    int64_t tokens = xqType.getDimSize(0);
    int64_t width = xqType.getDimSize(1);
    int64_t groups = xsType.getDimSize(1);
    int64_t outputWidth = outputType.getDimSize(1);
    int64_t outputBlocks = outputWidth / kOutBlock;
    int64_t groupSize = op.getGroupSize();

    auto scratchType =
        MemRefType::get({kPrefillRows, kOutBlock}, rewriter.getF32Type());
    auto tailActivationType =
        MemRefType::get({kHardwareM, kHardwareK}, rewriter.getI8Type());
    Value zero =
        memref::GetGlobalOp::create(rewriter, loc, scratchType, kZeroGlobal);
    Value scratch =
        memref::GetGlobalOp::create(rewriter, loc, scratchType, kScratchGlobal);
    Value tailActivation = memref::GetGlobalOp::create(
        rewriter, loc, tailActivationType, kTailActivationGlobal);

    // One-shot bufferization exposes function arguments with fully dynamic
    // strided layouts even though the Qwen3 ABI guarantees dense row-major
    // buffers.  Refine that existing descriptor (no copy and no ABI change)
    // so vector.load/store can prove a unit innermost stride.
    auto asContiguous = [&](Value value, MemRefType type) -> Value {
      if (type.getLayout().isIdentity())
        return value;

      SmallVector<int64_t> strides(type.getRank());
      SmallVector<OpFoldResult> sizes;
      SmallVector<OpFoldResult> strideValues(type.getRank());
      sizes.reserve(type.getRank());
      int64_t runningStride = 1;
      for (int64_t dimension = type.getRank() - 1; dimension >= 0;
           --dimension) {
        strides[dimension] = runningStride;
        strideValues[dimension] = rewriter.getIndexAttr(runningStride);
        runningStride *= type.getDimSize(dimension);
      }
      for (int64_t size : type.getShape())
        sizes.push_back(rewriter.getIndexAttr(size));

      auto metadata =
          memref::ExtractStridedMetadataOp::create(rewriter, loc, value);
      auto contiguousLayout = StridedLayoutAttr::get(
          rewriter.getContext(), ShapedType::kDynamic, strides);
      auto contiguousType =
          MemRefType::get(type.getShape(), type.getElementType(),
                          contiguousLayout, type.getMemorySpace());
      return memref::ReinterpretCastOp::create(
          rewriter, loc, contiguousType, metadata.getBaseBuffer(),
          metadata.getOffset(), sizes, strideValues);
    };
    Value contiguousOutput = asContiguous(op.getOutput(), outputType);

    Value c0 = indexConstant(rewriter, loc, 0);
    Value c1 = indexConstant(rewriter, loc, 1);
    Value c2 = indexConstant(rewriter, loc, 2);
    Value c16 = indexConstant(rewriter, loc, kHardwareM);
    Value c32 = indexConstant(rewriter, loc, kPrefillRows);
    Value c64 = indexConstant(rewriter, loc, kHardwareK);
    Value tokenBound = indexConstant(rewriter, loc, tokens);
    Value outputBound = indexConstant(rewriter, loc, outputWidth);
    Value groupBound = indexConstant(rewriter, loc, groups);
    Value groupSizeIndex = indexConstant(rewriter, loc, groupSize);
    Value zeroF = f32Constant(rewriter, loc, 0.0f);
    Value zeroI8 = arith::ConstantIntOp::create(rewriter, loc, 0, 8);
    Value strideA = i64Constant(rewriter, loc, width);
    Value strideB = i64Constant(rewriter, loc, groupSize);
    Value strideC = i64Constant(rewriter, loc, kOutBlock * sizeof(float));
    Value strideOneI8 = i64Constant(rewriter, loc, sizeof(int8_t));
    Value strideOneF32 = i64Constant(rewriter, loc, sizeof(float));
    Value oneI64 = i64Constant(rewriter, loc, 1);
    Value mSixteen = i64Constant(rewriter, loc, kHardwareM);
    Value nSixteen = i64Constant(rewriter, loc, kHardwareN);
    Value kSixtyFour = i64Constant(rewriter, loc, kHardwareK);
    Value typeI8 = i64Constant(rewriter, loc, kMTypeI8);
    Value typeI32 = i64Constant(rewriter, loc, kMTypeI32);
    auto zeroVectorType =
        VectorType::get({kHardwareN}, rewriter.getF32Type());
    if (profilePhases)
      emitProfileCall(rewriter, loc, kProfileStart, kProfileLinearTotal);

    // The bare-metal CRT does not clear .bss.  Seed all rows needed by the
    // 2A x 4B path, using N16 vector stores instead of a scalar N64 loop.
    Value zeroVector =
        vector::BroadcastOp::create(rewriter, loc, zeroVectorType, zeroF);
    auto zeroSeedRows = scf::ForOp::create(rewriter, loc, c0, c32, c1);
    rewriter.setInsertionPointToStart(zeroSeedRows.getBody());
    for (int64_t n = 0; n < kOutBlock; n += kHardwareN)
      vector::StoreOp::create(rewriter, loc, zeroVector, zero,
                              ValueRange{zeroSeedRows.getInductionVar(),
                                         indexConstant(rewriter, loc, n)});
    rewriter.setInsertionPointAfter(zeroSeedRows);

    // Keep first-group accumulation semantics while replacing [T,D] scalar
    // clearing with contiguous N16 vector stores.
    if (profilePhases)
      emitProfileCall(rewriter, loc, kProfileStart, kProfileOutputZero);
    auto zeroRows = scf::ForOp::create(rewriter, loc, c0, tokenBound, c1);
    rewriter.setInsertionPointToStart(zeroRows.getBody());
    auto zeroColumns = scf::ForOp::create(rewriter, loc, c0, outputBound, c16);
    rewriter.setInsertionPointToStart(zeroColumns.getBody());
    vector::StoreOp::create(
        rewriter, loc, zeroVector, contiguousOutput,
        ValueRange{zeroRows.getInductionVar(), zeroColumns.getInductionVar()});
    rewriter.setInsertionPointAfter(zeroRows);
    if (profilePhases)
      emitProfileCall(rewriter, loc, kProfileEnd, kProfileOutputZero);

    // Preserve the validated operation-boundary AME resynchronization.  Keep
    // it after every RVV initialization store: vector instructions update
    // state shared with AME on the FPGA, so running them after resync can make
    // the first B load of every K64 step ineffective.
    memref::StoreOp::create(rewriter, loc, zeroI8, tailActivation,
                            ValueRange{c0, c0});
    MSettilemOp::create(rewriter, loc, rewriter.getI64Type(), oneI64);
    MSettilenOp::create(rewriter, loc, rewriter.getI64Type(), oneI64);
    MSettilekOp::create(rewriter, loc, rewriter.getI64Type(), oneI64);
    MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), typeI8);
    Value syncI8 = makeSubview(rewriter, loc, tailActivation,
                               ArrayRef<OpFoldResult>{rewriter.getIndexAttr(0),
                                                      rewriter.getIndexAttr(0)},
                               ArrayRef<int64_t>{1, 1});
    Mlae8mOp::create(rewriter, loc, 0, syncI8, strideOneI8);
    Mlbte8mOp::create(rewriter, loc, 1, syncI8, strideOneI8);
    MqmaBmmOp::create(rewriter, loc, 0, 0, 1);
    MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), typeI32);
    Value syncZero =
        makeSubview(rewriter, loc, zero,
                    ArrayRef<OpFoldResult>{rewriter.getIndexAttr(0),
                                           rewriter.getIndexAttr(0)},
                    ArrayRef<int64_t>{1, 1});
    Value syncScratch =
        makeSubview(rewriter, loc, scratch,
                    ArrayRef<OpFoldResult>{rewriter.getIndexAttr(0),
                                           rewriter.getIndexAttr(0)},
                    ArrayRef<int64_t>{1, 1});
    Mlce32mOp::create(rewriter, loc, 0, syncZero, strideOneF32);
    Msce32mOp::create(rewriter, loc, 0, syncScratch, strideOneF32);
    LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::seq_cst);

    // N=16 and K=64 are invariant for every supported Qwen3 geometry.
    MSettilenOp::create(rewriter, loc, rewriter.getI64Type(), nSixteen);
    MSettilekOp::create(rewriter, loc, rewriter.getI64Type(), kSixtyFour);

    auto emitAccumulatorZero = [&](int64_t accumulatorBase, int64_t zeroRowBase,
                                   int64_t count) {
      for (int64_t n = 0; n < count; ++n) {
        Value zeroTile = makeSubview(
            rewriter, loc, zero,
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(zeroRowBase),
                                   rewriter.getIndexAttr(n * kHardwareN)},
            ArrayRef<int64_t>{kHardwareM, kHardwareN});
        Mlce32mOp::create(rewriter, loc, accumulatorBase + n, zeroTile,
                          strideC);
      }
    };

    auto emitAccumulateRows = [&](Value scratchRowBase, Value tokenBase,
                                  int64_t rows, Value outputBase, Value group) {
      Value rowBound = indexConstant(rewriter, loc, rows);
      auto rowLoop = scf::ForOp::create(rewriter, loc, c0, rowBound, c1);
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      Value row = rowLoop.getInductionVar();
      Value scratchRow =
          arith::AddIOp::create(rewriter, loc, scratchRowBase, row);
      Value token = arith::AddIOp::create(rewriter, loc, tokenBase, row);

      // The generic vector lowering was observed to insert a vlenb CSR read
      // while spilling vectors around profiling calls.  That CSR is not
      // implemented by the current FPGA core.  Call a small RVV leaf helper
      // whose exact e32,m1 instruction sequence matches the validated Qwen3
      // kernel instead of allowing register allocation to rewrite it.
      Value activationScale = memref::LoadOp::create(rewriter, loc, op.getXs(),
                                                     ValueRange{token, group});
      Value activationScaleBits = arith::BitcastOp::create(
          rewriter, loc, rewriter.getI32Type(), activationScale);

      auto elementAddress = [&](Value buffer, Value linearElement) {
        Value aligned = memref::ExtractAlignedPointerAsIndexOp::create(
            rewriter, loc, buffer);
        auto metadata =
            memref::ExtractStridedMetadataOp::create(rewriter, loc, buffer);
        Value elementWithOffset = arith::AddIOp::create(
            rewriter, loc, linearElement, metadata.getOffset());
        Value byteOffset = arith::MulIOp::create(
            rewriter, loc, elementWithOffset,
            indexConstant(rewriter, loc, sizeof(float)));
        Value address =
            arith::AddIOp::create(rewriter, loc, aligned, byteOffset);
        return arith::IndexCastOp::create(
                   rewriter, loc, rewriter.getI64Type(), address)
            .getResult();
      };

      Value scratchLinear = arith::AddIOp::create(
          rewriter, loc,
          arith::MulIOp::create(
              rewriter, loc, scratchRow,
              indexConstant(rewriter, loc, kOutBlock)),
          c0);
      Value weightScaleLinear = arith::AddIOp::create(
          rewriter, loc,
          arith::MulIOp::create(
              rewriter, loc, group,
              indexConstant(rewriter, loc, outputWidth)),
          outputBase);
      Value outputLinear = arith::AddIOp::create(
          rewriter, loc,
          arith::MulIOp::create(
              rewriter, loc, token,
              indexConstant(rewriter, loc, outputWidth)),
          outputBase);
      func::CallOp::create(
          rewriter, loc, kRvvAccumulateN64, TypeRange{},
          ValueRange{elementAddress(contiguousOutput, outputLinear),
                     elementAddress(scratch, scratchLinear),
                     elementAddress(op.getWs(), weightScaleLinear),
                     activationScaleBits});
    };

    auto makeWeightTile = [&](Value outputBlock, Value group, int64_t n,
                              Value kOffset) {
      return makeSubview(rewriter, loc, op.getWq(),
                         ArrayRef<OpFoldResult>{
                             outputBlock, group,
                             rewriter.getIndexAttr(n * kHardwareN), kOffset},
                         ArrayRef<int64_t>{1, 1, kHardwareN, kHardwareK});
    };

    auto emit1A4B = [&](Value tokenBase, int64_t tileRows, Value outputBlock,
                        Value group) {
      if (profilePhases)
        emitProfileCall(rewriter, loc, kProfileStart, kProfileAMEKernel);
      MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), typeI32);
      emitAccumulatorZero(/*accumulatorBase=*/0, /*zeroRowBase=*/0,
                          /*count=*/4);
      Value groupBase =
          arith::MulIOp::create(rewriter, loc, group, groupSizeIndex);
      MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), typeI8);
      auto kLoop = scf::ForOp::create(rewriter, loc, c0, groupSizeIndex, c64);
      rewriter.setInsertionPointToStart(kLoop.getBody());
      Value kOffset = kLoop.getInductionVar();
      Value activationOffset =
          arith::AddIOp::create(rewriter, loc, groupBase, kOffset);
      Value activationTile =
          makeSubview(rewriter, loc, op.getXq(),
                      ArrayRef<OpFoldResult>{tokenBase, activationOffset},
                      ArrayRef<int64_t>{tileRows, kHardwareK});
      Mlae8mOp::create(rewriter, loc, 0, activationTile, strideA);
      Value weight0 = makeWeightTile(outputBlock, group, 0, kOffset);
      Value weight1 = makeWeightTile(outputBlock, group, 1, kOffset);
      Value weight2 = makeWeightTile(outputBlock, group, 2, kOffset);
      Value weight3 = makeWeightTile(outputBlock, group, 3, kOffset);
      Mlbe8mOp::create(rewriter, loc, 4, weight0, strideB);
      Mlbe8mOp::create(rewriter, loc, 5, weight1, strideB);
      MqmaBmmOp::create(rewriter, loc, 0, 0, 4);
      Mlbe8mOp::create(rewriter, loc, 6, weight2, strideB);
      MqmaBmmOp::create(rewriter, loc, 1, 0, 5);
      Mlbe8mOp::create(rewriter, loc, 7, weight3, strideB);
      MqmaBmmOp::create(rewriter, loc, 2, 0, 6);
      MqmaBmmOp::create(rewriter, loc, 3, 0, 7);
      rewriter.setInsertionPointAfter(kLoop);

      MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), typeI32);
      for (int64_t n = 0; n < 4; ++n) {
        Value scratchTile = makeSubview(
            rewriter, loc, scratch,
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(0),
                                   rewriter.getIndexAttr(n * kHardwareN)},
            ArrayRef<int64_t>{tileRows, kHardwareN});
        Msce32mOp::create(rewriter, loc, n, scratchTile, strideC);
      }
      LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::seq_cst);
      if (profilePhases)
        emitProfileCall(rewriter, loc, kProfileEnd, kProfileAMEKernel);
      Value outputBase = arith::MulIOp::create(rewriter, loc, outputBlock, c64);
      if (profilePhases)
        emitProfileCall(rewriter, loc, kProfileStart, kProfileRVVAccumulation);
      emitAccumulateRows(c0, tokenBase, tileRows, outputBase, group);
      if (profilePhases)
        emitProfileCall(rewriter, loc, kProfileEnd, kProfileRVVAccumulation);
    };

    auto emit2A4B = [&](Value tokenBase, Value outputBlock, Value group) {
      if (profilePhases)
        emitProfileCall(rewriter, loc, kProfileStart, kProfileAMEKernel);
      MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), typeI32);
      emitAccumulatorZero(/*accumulatorBase=*/0, /*zeroRowBase=*/0,
                          /*count=*/4);
      emitAccumulatorZero(/*accumulatorBase=*/4,
                          /*zeroRowBase=*/kHardwareM, /*count=*/4);
      Value groupBase =
          arith::MulIOp::create(rewriter, loc, group, groupSizeIndex);
      Value secondTokenBase =
          arith::AddIOp::create(rewriter, loc, tokenBase, c16);
      MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), typeI8);
      auto kLoop = scf::ForOp::create(rewriter, loc, c0, groupSizeIndex, c64);
      rewriter.setInsertionPointToStart(kLoop.getBody());
      Value kOffset = kLoop.getInductionVar();
      Value activationOffset =
          arith::AddIOp::create(rewriter, loc, groupBase, kOffset);
      Value activation0 =
          makeSubview(rewriter, loc, op.getXq(),
                      ArrayRef<OpFoldResult>{tokenBase, activationOffset},
                      ArrayRef<int64_t>{kHardwareM, kHardwareK});
      Value activation1 =
          makeSubview(rewriter, loc, op.getXq(),
                      ArrayRef<OpFoldResult>{secondTokenBase, activationOffset},
                      ArrayRef<int64_t>{kHardwareM, kHardwareK});
      Mlae8mOp::create(rewriter, loc, 0, activation0, strideA);
      Mlae8mOp::create(rewriter, loc, 2, activation1, strideA);
      Value weight0 = makeWeightTile(outputBlock, group, 0, kOffset);
      Value weight1 = makeWeightTile(outputBlock, group, 1, kOffset);
      Value weight2 = makeWeightTile(outputBlock, group, 2, kOffset);
      Value weight3 = makeWeightTile(outputBlock, group, 3, kOffset);
      // Keep the FPGA-validated 2A x 4B issue order.
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

      MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), typeI32);
      for (int64_t n = 0; n < 4; ++n) {
        Value scratch0 = makeSubview(
            rewriter, loc, scratch,
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(0),
                                   rewriter.getIndexAttr(n * kHardwareN)},
            ArrayRef<int64_t>{kHardwareM, kHardwareN});
        Value scratch1 = makeSubview(
            rewriter, loc, scratch,
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(kHardwareM),
                                   rewriter.getIndexAttr(n * kHardwareN)},
            ArrayRef<int64_t>{kHardwareM, kHardwareN});
        Msce32mOp::create(rewriter, loc, n, scratch0, strideC);
        Msce32mOp::create(rewriter, loc, 4 + n, scratch1, strideC);
      }
      LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::seq_cst);
      if (profilePhases)
        emitProfileCall(rewriter, loc, kProfileEnd, kProfileAMEKernel);
      Value outputBase = arith::MulIOp::create(rewriter, loc, outputBlock, c64);
      if (profilePhases)
        emitProfileCall(rewriter, loc, kProfileStart, kProfileRVVAccumulation);
      emitAccumulateRows(c0, tokenBase, kHardwareM, outputBase, group);
      emitAccumulateRows(c16, secondTokenBase, kHardwareM, outputBase, group);
      if (profilePhases)
        emitProfileCall(rewriter, loc, kProfileEnd, kProfileRVVAccumulation);
    };

    auto emitDecodePair = [&](Value outputBlock0, Value group) {
      if (profilePhases)
        emitProfileCall(rewriter, loc, kProfileStart, kProfileAMEKernel);
      MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), typeI32);
      emitAccumulatorZero(/*accumulatorBase=*/0, /*zeroRowBase=*/0,
                          /*count=*/4);
      emitAccumulatorZero(/*accumulatorBase=*/4, /*zeroRowBase=*/0,
                          /*count=*/4);
      Value outputBlock1 =
          arith::AddIOp::create(rewriter, loc, outputBlock0, c1);
      Value groupBase =
          arith::MulIOp::create(rewriter, loc, group, groupSizeIndex);
      MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), typeI8);
      auto kLoop = scf::ForOp::create(rewriter, loc, c0, groupSizeIndex, c64);
      rewriter.setInsertionPointToStart(kLoop.getBody());
      Value kOffset = kLoop.getInductionVar();
      Value activationOffset =
          arith::AddIOp::create(rewriter, loc, groupBase, kOffset);
      Value activation =
          makeSubview(rewriter, loc, op.getXq(),
                      ArrayRef<OpFoldResult>{c0, activationOffset},
                      ArrayRef<int64_t>{1, kHardwareK});
      Mlae8mOp::create(rewriter, loc, 0, activation, strideA);

      auto emitFourWeights = [&](Value outputBlock, int64_t accumulatorBase) {
        Value weight0 = makeWeightTile(outputBlock, group, 0, kOffset);
        Value weight1 = makeWeightTile(outputBlock, group, 1, kOffset);
        Value weight2 = makeWeightTile(outputBlock, group, 2, kOffset);
        Value weight3 = makeWeightTile(outputBlock, group, 3, kOffset);
        Mlbe8mOp::create(rewriter, loc, 4, weight0, strideB);
        Mlbe8mOp::create(rewriter, loc, 5, weight1, strideB);
        MqmaBmmOp::create(rewriter, loc, accumulatorBase + 0, 0, 4);
        Mlbe8mOp::create(rewriter, loc, 6, weight2, strideB);
        MqmaBmmOp::create(rewriter, loc, accumulatorBase + 1, 0, 5);
        Mlbe8mOp::create(rewriter, loc, 7, weight3, strideB);
        MqmaBmmOp::create(rewriter, loc, accumulatorBase + 2, 0, 6);
        MqmaBmmOp::create(rewriter, loc, accumulatorBase + 3, 0, 7);
      };
      emitFourWeights(outputBlock0, 0);
      emitFourWeights(outputBlock1, 4);
      rewriter.setInsertionPointAfter(kLoop);

      MSettypeOp::create(rewriter, loc, rewriter.getI64Type(), typeI32);
      for (int64_t n = 0; n < 4; ++n) {
        Value scratch0 = makeSubview(
            rewriter, loc, scratch,
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(0),
                                   rewriter.getIndexAttr(n * kHardwareN)},
            ArrayRef<int64_t>{1, kHardwareN});
        Value scratch1 = makeSubview(
            rewriter, loc, scratch,
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1),
                                   rewriter.getIndexAttr(n * kHardwareN)},
            ArrayRef<int64_t>{1, kHardwareN});
        Msce32mOp::create(rewriter, loc, n, scratch0, strideC);
        Msce32mOp::create(rewriter, loc, 4 + n, scratch1, strideC);
      }
      LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::seq_cst);
      if (profilePhases)
        emitProfileCall(rewriter, loc, kProfileEnd, kProfileAMEKernel);
      Value outputBase0 =
          arith::MulIOp::create(rewriter, loc, outputBlock0, c64);
      Value outputBase1 =
          arith::MulIOp::create(rewriter, loc, outputBlock1, c64);
      if (profilePhases)
        emitProfileCall(rewriter, loc, kProfileStart, kProfileRVVAccumulation);
      emitAccumulateRows(c0, c0, 1, outputBase0, group);
      emitAccumulateRows(c1, c0, 1, outputBase1, group);
      if (profilePhases)
        emitProfileCall(rewriter, loc, kProfileEnd, kProfileRVVAccumulation);
    };

    if (tokens == 1) {
      // GS=512/1024 both take this fixed-K-step path: K is configured once,
      // and each iteration advances by the statically legal K64 tile.
      MSettilemOp::create(rewriter, loc, rewriter.getI64Type(), oneI64);
      auto groupLoop = scf::ForOp::create(rewriter, loc, c0, groupBound, c1);
      rewriter.setInsertionPointToStart(groupLoop.getBody());
      Value group = groupLoop.getInductionVar();
      int64_t pairedBlocks = outputBlocks / 2;
      if (pairedBlocks != 0) {
        Value pairBound = indexConstant(rewriter, loc, pairedBlocks);
        auto pairLoop = scf::ForOp::create(rewriter, loc, c0, pairBound, c1);
        rewriter.setInsertionPointToStart(pairLoop.getBody());
        Value outputBlock0 = arith::MulIOp::create(
            rewriter, loc, pairLoop.getInductionVar(), c2);
        emitDecodePair(outputBlock0, group);
        rewriter.setInsertionPointAfter(pairLoop);
      }
      if (outputBlocks % 2 != 0) {
        Value lastBlock = indexConstant(rewriter, loc, outputBlocks - 1);
        emit1A4B(c0, 1, lastBlock, group);
      }
      rewriter.setInsertionPointAfter(groupLoop);
    } else {
      int64_t fullPairs = tokens / kPrefillRows;
      int64_t remainingRows = tokens % kPrefillRows;
      bool invariantM = tokens <= kHardwareM || remainingRows == 0;
      if (tokens <= kHardwareM) {
        Value exactM = i64Constant(rewriter, loc, tokens);
        MSettilemOp::create(rewriter, loc, rewriter.getI64Type(), exactM);
      } else if (remainingRows == 0) {
        MSettilemOp::create(rewriter, loc, rewriter.getI64Type(), mSixteen);
      }

      auto groupLoop = scf::ForOp::create(rewriter, loc, c0, groupBound, c1);
      rewriter.setInsertionPointToStart(groupLoop.getBody());
      Value group = groupLoop.getInductionVar();
      Value blockBound = indexConstant(rewriter, loc, outputBlocks);
      auto blockLoop = scf::ForOp::create(rewriter, loc, c0, blockBound, c1);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      Value outputBlock = blockLoop.getInductionVar();

      if (fullPairs != 0) {
        if (remainingRows != 0)
          MSettilemOp::create(rewriter, loc, rewriter.getI64Type(), mSixteen);
        Value pairBound = indexConstant(rewriter, loc, fullPairs);
        auto pairLoop = scf::ForOp::create(rewriter, loc, c0, pairBound, c1);
        rewriter.setInsertionPointToStart(pairLoop.getBody());
        Value tokenBase = arith::MulIOp::create(
            rewriter, loc, pairLoop.getInductionVar(), c32);
        emit2A4B(tokenBase, outputBlock, group);
        rewriter.setInsertionPointAfter(pairLoop);
      }

      int64_t staticTokenBase = fullPairs * kPrefillRows;
      if (remainingRows >= kHardwareM) {
        if (!invariantM && fullPairs == 0)
          MSettilemOp::create(rewriter, loc, rewriter.getI64Type(), mSixteen);
        Value tokenBase = indexConstant(rewriter, loc, staticTokenBase);
        emit1A4B(tokenBase, kHardwareM, outputBlock, group);
        staticTokenBase += kHardwareM;
        remainingRows -= kHardwareM;
      }
      if (remainingRows != 0) {
        if (!invariantM) {
          Value tailM = i64Constant(rewriter, loc, remainingRows);
          MSettilemOp::create(rewriter, loc, rewriter.getI64Type(), tailM);
        }
        Value tokenBase = indexConstant(rewriter, loc, staticTokenBase);
        emit1A4B(tokenBase, remainingRows, outputBlock, group);
      }
      rewriter.setInsertionPointAfter(blockLoop);
      rewriter.setInsertionPointAfter(groupLoop);
    }

    if (profilePhases)
      emitProfileCall(rewriter, loc, kProfileEnd, kProfileLinearTotal);
    rewriter.eraseOp(op);
    return success();
  }

private:
  bool profilePhases;
};

class LowerQwenW8A8ToBOSCAMEPass
    : public PassWrapper<LowerQwenW8A8ToBOSCAMEPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerQwenW8A8ToBOSCAMEPass)

  LowerQwenW8A8ToBOSCAMEPass() = default;
  LowerQwenW8A8ToBOSCAMEPass(const LowerQwenW8A8ToBOSCAMEPass &pass)
      : PassWrapper(pass) {}

  StringRef getArgument() const final { return "lower-qwen-w8a8-to-boscame"; }
  StringRef getDescription() const final {
    return "Lower native Qwen3 W8A8 semantic ops to OUTBLK64 BOSCAME";
  }

  Option<bool> scalarFallback{
      *this, "scalar-fallback",
      llvm::cl::desc("Use the original scalar accumulation and N64 AME "
                     "schedule for FPGA A/B diagnosis"),
      llvm::cl::init(false)};

  Option<bool> profilePhases{
      *this, "profile-phases",
      llvm::cl::desc("Instrument W8A8 quantize, output zero, AME, RVV "
                     "accumulation and total phases with cycle trace calls"),
      llvm::cl::init(false)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<BOSCAMEDialect, arith::ArithDialect, func::FuncDialect,
                    math::MathDialect, memref::MemRefDialect, scf::SCFDialect,
                    vector::VectorDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (profilePhases) {
      OpBuilder builder(module.getBodyRegion());
      Type i64 = builder.getI64Type();
      auto functionType =
          builder.getFunctionType({i64, i64, i64, i64, i64, i64}, {});
      for (StringRef name : {kProfileStart, kProfileEnd}) {
        if (module.lookupSymbol(name))
          continue;
        auto function =
            func::FuncOp::create(builder, module.getLoc(), name, functionType);
        function.setPrivate();
        function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
      }
    }
    bool hasLinear = false;
    module.walk([&](W8A8LinearOp) { hasLinear = true; });
    if (!hasLinear) {
      lower(module);
      return;
    }

    OpBuilder globalBuilder(module.getBodyRegion());
    globalBuilder.setInsertionPointToStart(module.getBody());
    if (!scalarFallback && !module.lookupSymbol(kRvvAccumulateN64)) {
      Type i64 = globalBuilder.getI64Type();
      Type i32 = globalBuilder.getI32Type();
      auto functionType =
          globalBuilder.getFunctionType({i64, i64, i64, i32}, {});
      auto function = func::FuncOp::create(globalBuilder, module.getLoc(),
                                            kRvvAccumulateN64, functionType);
      function.setPrivate();
    }
    auto scratchType =
        MemRefType::get({kPrefillRows, kOutBlock}, globalBuilder.getF32Type());
    auto tailActivationType =
        MemRefType::get({kHardwareM, kHardwareK}, globalBuilder.getI8Type());
    auto makeGlobal = [&](StringRef name, bool initializeToZero) {
      if (module.lookupSymbol(name))
        return;
      Attribute initialValue = UnitAttr::get(&getContext());
      if (initializeToZero) {
        auto tensorType = RankedTensorType::get({kPrefillRows, kOutBlock},
                                                globalBuilder.getF32Type());
        initialValue = DenseElementsAttr::get(
            tensorType, globalBuilder.getF32FloatAttr(0.0f));
      }
      memref::GlobalOp::create(
          globalBuilder, module.getLoc(), name,
          globalBuilder.getStringAttr("private"), scratchType, initialValue,
          /*constant=*/false, globalBuilder.getI64IntegerAttr(64));
    };
    makeGlobal(kZeroGlobal, true);
    makeGlobal(kScratchGlobal, false);
    if (!module.lookupSymbol(kTailActivationGlobal))
      memref::GlobalOp::create(
          globalBuilder, module.getLoc(), kTailActivationGlobal,
          globalBuilder.getStringAttr("private"), tailActivationType,
          UnitAttr::get(&getContext()), /*constant=*/false,
          globalBuilder.getI64IntegerAttr(64));
    lower(module);
  }

private:
  void lower(ModuleOp module) {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<QuantizePerGroupLowering>(context, profilePhases);
    if (scalarFallback)
      patterns.add<W8A8LinearScalarFallback>(context);
    else
      patterns.add<W8A8LinearLowering>(context, profilePhases);
    ConversionTarget target(*context);
    target.addLegalDialect<BOSCAMEDialect, arith::ArithDialect,
                           func::FuncDialect, math::MathDialect,
                           memref::MemRefDialect, scf::SCFDialect,
                           vector::VectorDialect, LLVM::LLVMDialect>();
    target.addLegalOp<ModuleOp>();
    target.addIllegalOp<QuantizePerGroupOp, W8A8LinearOp>();
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::buddy {
void registerLowerQwenW8A8ToBOSCAMEPass() {
  PassRegistration<LowerQwenW8A8ToBOSCAMEPass>();
}
} // namespace mlir::buddy
