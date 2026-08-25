//===- LowerQwenW8A8ToBOSCAME.cpp - Qwen3 W8A8 to AME ---------*- C++ -*-===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
constexpr int64_t kOutBlock = 64;
constexpr int64_t kHardwareM = 16;
constexpr int64_t kHardwareN = 16;
constexpr int64_t kHardwareK = 64;
constexpr int64_t kMTypeI8 = (1LL << 16) | (1LL << 4);
constexpr int64_t kMTypeI32 = (1LL << 16) | (1LL << 6) | 2;

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

class QuantizePerGroupLowering : public OpRewritePattern<QuantizePerGroupOp> {
public:
  using OpRewritePattern<QuantizePerGroupOp>::OpRewritePattern;

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

    auto tokenLoop = scf::ForOp::create(rewriter, loc, c0, tokenBound, c1);
    rewriter.setInsertionPointToStart(tokenLoop.getBody());
    Value token = tokenLoop.getInductionVar();
    auto groupLoop = scf::ForOp::create(rewriter, loc, c0, groupBound, c1);
    rewriter.setInsertionPointToStart(groupLoop.getBody());
    Value group = groupLoop.getInductionVar();
    Value groupBase =
        arith::MulIOp::create(rewriter, loc, group, groupSizeIndex);

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
    Value maximum = arith::MaximumFOp::create(
        rewriter, loc, maxLoop.getRegionIterArgs().front(), absolute);
    scf::YieldOp::create(rewriter, loc, maximum);

    rewriter.setInsertionPointAfter(maxLoop);
    Value amax = maxLoop.getResult(0);
    Value nonzero = arith::CmpFOp::create(
        rewriter, loc, arith::CmpFPredicate::OGT, amax, zeroF);
    Value rawScale = arith::DivFOp::create(rewriter, loc, amax, c127F);
    Value scale =
        arith::SelectOp::create(rewriter, loc, nonzero, rawScale, oneF);
    memref::StoreOp::create(rewriter, loc, scale, op.getScales(),
                            ValueRange{token, group});

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

    rewriter.setInsertionPointAfter(tokenLoop);
    rewriter.eraseOp(op);
    return success();
  }
};

class W8A8LinearLowering : public OpRewritePattern<W8A8LinearOp> {
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
        MemRefType::get({kHardwareM, kOutBlock}, rewriter.getF32Type());
    Value zero =
        memref::GetGlobalOp::create(rewriter, loc, scratchType, kZeroGlobal);
    Value scratch =
        memref::GetGlobalOp::create(rewriter, loc, scratchType, kScratchGlobal);

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
    Value strideA = i64Constant(rewriter, loc, width);
    Value strideB = i64Constant(rewriter, loc, groupSize);
    Value strideC = i64Constant(rewriter, loc, kOutBlock * sizeof(float));
    Value nSixteen = i64Constant(rewriter, loc, kHardwareN);
    Value kSixtyFour = i64Constant(rewriter, loc, kHardwareK);
    Value typeI8 = i64Constant(rewriter, loc, kMTypeI8);
    Value typeI32 = i64Constant(rewriter, loc, kMTypeI32);

    // The FPGA bare-metal CRT does not clear .bss.  Materialize the persistent
    // accumulator seed before the first AME load instead of relying on the
    // LLVM zeroinitializer surviving placement in a NOBITS section.
    Value mSeedBound = indexConstant(rewriter, loc, kHardwareM);
    auto zeroSeedRows =
        scf::ForOp::create(rewriter, loc, c0, mSeedBound, c1);
    rewriter.setInsertionPointToStart(zeroSeedRows.getBody());
    auto zeroSeedColumns =
        scf::ForOp::create(rewriter, loc, c0, c64, c1);
    rewriter.setInsertionPointToStart(zeroSeedColumns.getBody());
    memref::StoreOp::create(
        rewriter, loc, zeroF, zero,
        ValueRange{zeroSeedRows.getInductionVar(),
                   zeroSeedColumns.getInductionVar()});
    rewriter.setInsertionPointAfter(zeroSeedRows);

    auto zeroRows =
        scf::ForOp::create(rewriter, loc, c0, tokenBound, c1);
    rewriter.setInsertionPointToStart(zeroRows.getBody());
    auto zeroColumns =
        scf::ForOp::create(rewriter, loc, c0, outputBound, c1);
    rewriter.setInsertionPointToStart(zeroColumns.getBody());
    memref::StoreOp::create(
        rewriter, loc, zeroF, op.getOutput(),
        ValueRange{zeroRows.getInductionVar(), zeroColumns.getInductionVar()});
    rewriter.setInsertionPointAfter(zeroRows);

    // N and K stay fixed for the full operation. M is programmed once per
    // token tile below, using M=16 for prefill and the exact static tail size.
    MSettilenOp::create(rewriter, loc, rewriter.getI64Type(), nSixteen);
    MSettilekOp::create(rewriter, loc, rewriter.getI64Type(), kSixtyFour);

    auto emitTokenTile = [&](Value tokenBase, int64_t tileRows) {
      Value tileM = i64Constant(rewriter, loc, tileRows);
      Value tileRowBound = indexConstant(rewriter, loc, tileRows);
      MSettilemOp::create(rewriter, loc, rewriter.getI64Type(), tileM);

      auto blockLoop =
          scf::ForOp::create(rewriter, loc, c0, blockBound, c1);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      Value outputBlock = blockLoop.getInductionVar();
      Value outputBase =
          arith::MulIOp::create(rewriter, loc, outputBlock, c64);
      auto groupLoop =
          scf::ForOp::create(rewriter, loc, c0, groupBound, c1);
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
      auto kLoop =
          scf::ForOp::create(rewriter, loc, c0, groupSizeIndex, c64);
      rewriter.setInsertionPointToStart(kLoop.getBody());
      Value kOffset = kLoop.getInductionVar();
      Value activationOffset =
          arith::AddIOp::create(rewriter, loc, groupBase, kOffset);
      Value activationTile = makeSubview(
          rewriter, loc, op.getXq(),
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

      // Match Qwen3's validated 1A x 4B schedule.  With M programmed to 16,
      // each mqma produces a full M16 x N16 tile; smaller static tails use the
      // same instruction sequence with their exact M value.
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

      auto accumulateRows =
          scf::ForOp::create(rewriter, loc, c0, tileRowBound, c1);
      rewriter.setInsertionPointToStart(accumulateRows.getBody());
      Value rowInTile = accumulateRows.getInductionVar();
      Value token =
          arith::AddIOp::create(rewriter, loc, tokenBase, rowInTile);
      auto accumulateColumns =
          scf::ForOp::create(rewriter, loc, c0, c64, c1);
      rewriter.setInsertionPointToStart(accumulateColumns.getBody());
      Value columnInBlock = accumulateColumns.getInductionVar();
      Value outputColumn =
          arith::AddIOp::create(rewriter, loc, outputBase, columnInBlock);
      Value dot = memref::LoadOp::create(
          rewriter, loc, scratch, ValueRange{rowInTile, columnInBlock});
      Value activationScale = memref::LoadOp::create(
          rewriter, loc, op.getXs(), ValueRange{token, group});
      Value weightScale = memref::LoadOp::create(
          rewriter, loc, op.getWs(), ValueRange{group, outputColumn});
      Value oldOutput = memref::LoadOp::create(
          rewriter, loc, op.getOutput(), ValueRange{token, outputColumn});
      Value scaledDot =
          arith::MulFOp::create(rewriter, loc, dot, activationScale);
      scaledDot =
          arith::MulFOp::create(rewriter, loc, scaledDot, weightScale);
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

class LowerQwenW8A8ToBOSCAMEPass
    : public PassWrapper<LowerQwenW8A8ToBOSCAMEPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerQwenW8A8ToBOSCAMEPass)

  StringRef getArgument() const final { return "lower-qwen-w8a8-to-boscame"; }
  StringRef getDescription() const final {
    return "Lower native Qwen3 W8A8 semantic ops to OUTBLK64 BOSCAME";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<BOSCAMEDialect, arith::ArithDialect, math::MathDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool hasLinear = false;
    module.walk([&](W8A8LinearOp) { hasLinear = true; });
    if (!hasLinear) {
      lower(module);
      return;
    }

    OpBuilder globalBuilder(module.getBodyRegion());
    globalBuilder.setInsertionPointToStart(module.getBody());
    auto scratchType = MemRefType::get({kHardwareM, kOutBlock},
                                       globalBuilder.getF32Type());
    auto makeGlobal = [&](StringRef name, bool initializeToZero) {
      if (module.lookupSymbol(name))
        return;
      Attribute initialValue = UnitAttr::get(&getContext());
      if (initializeToZero) {
        auto tensorType = RankedTensorType::get(
            {kHardwareM, kOutBlock}, globalBuilder.getF32Type());
        initialValue = DenseElementsAttr::get(
            tensorType, globalBuilder.getF32FloatAttr(0.0));
      }
      memref::GlobalOp::create(
          globalBuilder, module.getLoc(), name,
          globalBuilder.getStringAttr("private"), scratchType, initialValue,
          /*constant=*/false, globalBuilder.getI64IntegerAttr(64));
    };
    makeGlobal(kZeroGlobal, true);
    makeGlobal(kScratchGlobal, false);
    lower(module);
  }

private:
  void lower(ModuleOp module) {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<QuantizePerGroupLowering, W8A8LinearLowering>(context);
    ConversionTarget target(*context);
    target
        .addLegalDialect<BOSCAMEDialect, arith::ArithDialect, math::MathDialect,
                         memref::MemRefDialect, scf::SCFDialect>();
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
