//===- LegalizeForLLVMExport.cpp - Native XSAIAME lowering ------*- C++ -*-===//
//
// Part of the Buddy Compiler project, under the Apache License v2.0.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "Dialect/XSAIAME/Transform.h"
#include "Dialect/XSAIAME/XSAIAMEDialect.h"
#include "Dialect/XSAIAME/XSAIAMEOps.h"
#include "Dialect/XSAIAME/XSAIAMETileRules.h"
#include "Dialect/XSAIAME/XSAIAMETypes.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"

#include <array>
#include <utility>

using namespace mlir;
using namespace ::buddy::xsaiame;

namespace {

static StringRef getTargetExtName(StringRef element) {
  return llvm::StringSwitch<StringRef>(element)
      .Case("i8", "riscv.xsai.zames.tile.int8")
      .Case("i32", "riscv.xsai.zames.tile.int32")
      .Case("fp8e4m3", "riscv.xsai.zames.tile.fp8e4")
      .Case("fp8e5m2", "riscv.xsai.zames.tile.fp8e5")
      .Case("fp16", "riscv.xsai.zames.tile.fp16")
      .Case("bf16", "riscv.xsai.zames.tile.bf16")
      .Case("fp32", "riscv.xsai.zames.tile.fp32")
      .Default("");
}

static Type convertTileType(TileType tile) {
  StringRef name = getTargetExtName(tile.getElement());
  if (name.empty())
    return {};
  return LLVM::LLVMTargetExtType::get(tile.getContext(), name, ArrayRef<Type>{},
                                      ArrayRef<unsigned>{});
}

static Type convertSyncType(SyncType sync) {
  return LLVM::LLVMTargetExtType::get(sync.getContext(),
                                      "riscv.xsai.zmasync.sync",
                                      ArrayRef<Type>{}, ArrayRef<unsigned>{});
}

static LogicalResult checkTile(Location loc, TileType tile) {
  if (convertTileType(tile))
    return success();
  return emitError(loc)
         << "lower-xsai-ame does not support XSAIAME element token "
         << tile.getElement();
}

// The matrix unit computes only complete tiles (XSAIAMETileRules.h), and a
// partial transfer would leave stale register data under the next complete
// mma. Every lowered tile is therefore complete, so each msettile is constant.
static LogicalResult checkCompleteTile(Location loc, TileType tile) {
  ArrayRef<int64_t> shape = tile.getShape();
  int64_t expected[2] = {tile.isB() ? kTileN : kTileM,
                         tile.isC() ? kTileN : getTileK(tile.getElement())};
  if (shape[0] == expected[0] && shape[1] == expected[1])
    return success();
  return emitError(loc) << "lower-xsai-ame requires a complete "
                        << tile.getKind() << " tile " << expected[0] << "x"
                        << expected[1] << " for element " << tile.getElement()
                        << ", got " << tile;
}

static bool isInputElement(StringRef element) {
  return element == "i8" || element == "fp8e4m3" || element == "fp8e5m2" ||
         element == "fp16" || element == "bf16";
}

static LogicalResult checkLoadStore(Location loc, TileType tile) {
  if (failed(checkTile(loc, tile)))
    return failure();
  if (((tile.isA() || tile.isB()) && isInputElement(tile.getElement())) ||
      (tile.isC() &&
       (tile.getElement() == "i32" || tile.getElement() == "fp32")))
    return checkCompleteTile(loc, tile);
  return emitError(loc)
         << "lower-xsai-ame does not support load/store for tile kind "
         << tile.getKind() << " with element " << tile.getElement();
}

static LogicalResult checkMma(Location loc, TileType a, TileType b,
                              TileType c) {
  if (failed(checkTile(loc, a)) || failed(checkTile(loc, b)) ||
      failed(checkTile(loc, c)))
    return failure();
  if (!a.isA() || !b.isB() || !c.isC())
    return emitError(loc)
           << "lower-xsai-ame expects mma operands to be a, b, c";
  if (a.getElement() == b.getElement() &&
      ((a.getElement() == "i8" && c.getElement() == "i32") ||
       (a.getElement() != "i8" && isInputElement(a.getElement()) &&
        c.getElement() == "fp32")))
    return success(succeeded(checkCompleteTile(loc, a)) &&
                   succeeded(checkCompleteTile(loc, b)) &&
                   succeeded(checkCompleteTile(loc, c)));
  return emitError(loc)
         << "lower-xsai-ame does not support mma element combination "
         << a.getElement() << " x " << b.getElement() << " -> "
         << c.getElement();
}

/// Let LLVM infer overloaded target-extension signatures. In particular do not
/// manufacture mangled declarations or return values for void configuration
/// ops.
static LLVM::CallIntrinsicOp intrinsic(ConversionPatternRewriter &rewriter,
                                       Location loc, StringRef name,
                                       TypeRange results, ValueRange operands) {
  return rewriter.create<LLVM::CallIntrinsicOp>(
      loc, results, rewriter.getStringAttr(name), operands,
      LLVM::FastmathFlagsAttr::get(rewriter.getContext(),
                                   LLVM::FastmathFlags::none));
}

static Value constantI64(ConversionPatternRewriter &rewriter, Location loc,
                         int64_t value) {
  return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(),
                                           rewriter.getI64IntegerAttr(value));
}

static Value castToI64(ConversionPatternRewriter &rewriter, Location loc,
                       Value value) {
  if (value.getType().isIndex())
    return rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(),
                                               value);
  auto type = dyn_cast<IntegerType>(value.getType());
  if (!type)
    return {};
  if (type.getWidth() < 64)
    return rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), value);
  if (type.getWidth() > 64)
    return rewriter.create<arith::TruncIOp>(loc, rewriter.getI64Type(), value);
  return value;
}

/// Aligned pointer alone is insufficient for subviews. Apply the metadata
/// offset in bytes and independently scale the dynamic major stride by element
/// width.
static FailureOr<std::pair<Value, Value>>
viewAddress(ConversionPatternRewriter &rewriter, Location loc, Value view) {
  auto type = dyn_cast<MemRefType>(view.getType());
  SmallVector<int64_t, 2> strides;
  int64_t offset;
  if (!type || type.getRank() != 2 ||
      failed(type.getStridesAndOffset(strides, offset)) || strides[1] != 1) {
    emitError(loc)
        << "lower-xsai-ame requires a rank-2 memref view with static "
           "unit minor stride";
    return failure();
  }
  Attribute memorySpace = type.getMemorySpace();
  auto integerSpace = dyn_cast_or_null<IntegerAttr>(memorySpace);
  if (memorySpace && (!integerSpace || !integerSpace.getValue().isZero())) {
    emitError(loc)
        << "lower-xsai-ame requires the default memory address space";
    return failure();
  }
  Type element = type.getElementType();
  unsigned bits = element.getIntOrFloatBitWidth();
  if (bits != 8 && bits != 16 && bits != 32) {
    emitError(loc)
        << "lower-xsai-ame requires an 8-, 16-, or 32-bit memory element";
    return failure();
  }
  auto metadata = rewriter.create<memref::ExtractStridedMetadataOp>(loc, view);
  Value base = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
      loc, metadata.getBaseBuffer());
  base = castToI64(rewriter, loc, base);
  Value byteWidth = constantI64(rewriter, loc, bits / 8);
  Value byteOffset = rewriter.create<arith::MulIOp>(
      loc, castToI64(rewriter, loc, metadata.getOffset()), byteWidth);
  Value address = rewriter.create<arith::AddIOp>(loc, base, byteOffset);
  Value pointer = rewriter.create<LLVM::IntToPtrOp>(
      loc, LLVM::LLVMPointerType::get(rewriter.getContext()), address);
  Value rowStride = rewriter.create<arith::MulIOp>(
      loc, castToI64(rewriter, loc, metadata.getStrides()[0]), byteWidth);
  return std::make_pair(pointer, rowStride);
}

static void setDimension(ConversionPatternRewriter &rewriter, Location loc,
                         unsigned axis, int64_t extent) {
  static constexpr StringLiteral immediate[] = {
      "llvm.riscv.xsai.zames.msettilemi", "llvm.riscv.xsai.zames.msettileni",
      "llvm.riscv.xsai.zames.msettileki"};
  intrinsic(rewriter, loc, immediate[axis], {},
            ValueRange{constantI64(rewriter, loc, extent)});
}

static void setLoadStoreShape(ConversionPatternRewriter &rewriter, Location loc,
                              TileType tile) {
  ArrayRef<int64_t> shape = tile.getShape();
  setDimension(rewriter, loc, tile.isB() ? 1 : 0, shape[0]);
  setDimension(rewriter, loc, tile.isC() ? 1 : 2, shape[1]);
}

static StringRef loadName(TileType tile, bool transpose) {
  if (tile.isA())
    return transpose ? "llvm.riscv.xsai.zames.mlat"
                     : "llvm.riscv.xsai.zames.mla";
  if (tile.isB())
    return transpose ? "llvm.riscv.xsai.zames.mlbt"
                     : "llvm.riscv.xsai.zames.mlb";
  return transpose ? "llvm.riscv.xsai.zames.mlct" : "llvm.riscv.xsai.zames.mlc";
}

static StringRef storeName(TileType tile, bool transpose) {
  if (tile.isA())
    return transpose ? "llvm.riscv.xsai.zames.msat"
                     : "llvm.riscv.xsai.zames.msa";
  if (tile.isB())
    return transpose ? "llvm.riscv.xsai.zames.msbt"
                     : "llvm.riscv.xsai.zames.msb";
  return transpose ? "llvm.riscv.xsai.zames.msct" : "llvm.riscv.xsai.zames.msc";
}

struct LoadLowering : ConvertOpToLLVMPattern<LoadOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(LoadOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TileType tile = op.getResult().getType();
    if (failed(checkLoadStore(op.getLoc(), tile)))
      return failure();
    auto address = viewAddress(rewriter, op.getLoc(), op.getView());
    if (failed(address))
      return failure();
    setLoadStoreShape(rewriter, op.getLoc(), tile);
    auto call =
        intrinsic(rewriter, op.getLoc(), loadName(tile, op.getTranspose()),
                  TypeRange{convertTileType(tile)},
                  ValueRange{address->first, address->second});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct StoreLowering : ConvertOpToLLVMPattern<StoreOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TileType tile = op.getTile().getType();
    if (failed(checkLoadStore(op.getLoc(), tile)))
      return failure();
    auto address = viewAddress(rewriter, op.getLoc(), op.getView());
    if (failed(address))
      return failure();
    setLoadStoreShape(rewriter, op.getLoc(), tile);
    intrinsic(rewriter, op.getLoc(), storeName(tile, op.getTranspose()), {},
              ValueRange{adaptor.getTile(), address->first, address->second});
    rewriter.eraseOp(op);
    return success();
  }
};

struct MmaLowering : ConvertOpToLLVMPattern<MmaOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MmaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TileType a = op.getA().getType(), b = op.getB().getType();
    if (failed(checkMma(op.getLoc(), a, b, op.getC().getType())))
      return failure();
    int64_t extents[] = {a.getShape()[0], b.getShape()[0], a.getShape()[1]};
    for (unsigned axis = 0; axis < 3; ++axis)
      setDimension(rewriter, op.getLoc(), axis, extents[axis]);
    auto call =
        intrinsic(rewriter, op.getLoc(), "llvm.riscv.xsai.zames.mmacc",
                  TypeRange{convertTileType(op.getResult().getType())},
                  ValueRange{adaptor.getA(), adaptor.getB(), adaptor.getC()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct MzeroLowering : ConvertOpToLLVMPattern<MzeroOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MzeroOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TileType tile = op.getResult().getType();
    if (failed(checkTile(op.getLoc(), tile)))
      return failure();
    // The instruction clears the whole physical register, regardless of shape.
    auto call = intrinsic(rewriter, op.getLoc(), "llvm.riscv.xsai.zames.mzero",
                          TypeRange{convertTileType(tile)}, {});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct ResetLowering : ConvertOpToLLVMPattern<MsyncRegResetOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MsyncRegResetOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto call = intrinsic(
        rewriter, op.getLoc(), "llvm.riscv.xsai.zmasync.msyncregreset",
        TypeRange{convertSyncType(op.getResult().getType())}, {});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct ReleaseLowering : ConvertOpToLLVMPattern<MreleaseOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MreleaseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto call = intrinsic(
        rewriter, op.getLoc(), "llvm.riscv.xsai.zmasync.mrelease",
        TypeRange{adaptor.getSync().getType()}, ValueRange{adaptor.getSync()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct AcquireLowering : ConvertOpToLLVMPattern<MacquireOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MacquireOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value target = castToI64(rewriter, op.getLoc(), adaptor.getTarget());
    if (!target)
      return rewriter.notifyMatchFailure(
          op, "acquire target must be index/integer");
    intrinsic(rewriter, op.getLoc(), "llvm.riscv.xsai.zmasync.macquire", {},
              ValueRange{adaptor.getSync(), target});
    rewriter.eraseOp(op);
    return success();
  }
};

struct FenceLowering : ConvertOpToLLVMPattern<MfenceOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MfenceOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    intrinsic(rewriter, op.getLoc(), "llvm.riscv.xsai.zmasync.mfence", {}, {});
    rewriter.eraseOp(op);
    return success();
  }
};

/// The generic function-signature pattern converts only the entry block in
/// LLVM24. Non-entry CFG arguments must also change, even when the public
/// signature itself is unchanged (for example a local sync-token diamond).
struct FunctionLowering : OpConversionPattern<func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FunctionType type = op.getFunctionType();
    TypeConverter::SignatureConversion signature(type.getNumInputs());
    SmallVector<Type> results;
    const TypeConverter &converter = *getTypeConverter();
    if (failed(converter.convertSignatureArgs(type.getInputs(), signature)) ||
        failed(converter.convertTypes(type.getResults(), results)))
      return failure();
    if (failed(
            rewriter.convertRegionTypes(&op.getBody(), converter, &signature)))
      return failure();
    FunctionType converted = FunctionType::get(
        op.getContext(), signature.getConvertedTypes(), results);
    rewriter.modifyOpInPlace(op, [&] { op.setType(converted); });
    return success();
  }
};

struct FunctionConstantLowering : OpConversionPattern<func::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::ConstantOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = getTypeConverter()->convertType(op.getResult().getType());
    if (!isa_and_nonnull<FunctionType>(converted))
      return failure();
    auto constant = rewriter.create<func::ConstantOp>(op.getLoc(), converted,
                                                      op.getValueAttr());
    constant->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    rewriter.replaceOp(op, constant.getResult());
    return success();
  }
};

struct IndirectCallLowering : OpConversionPattern<func::CallIndirectOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::CallIndirectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<FunctionType>(adaptor.getCallee().getType()))
      return failure();
    auto call = rewriter.create<func::CallIndirectOp>(
        op.getLoc(), adaptor.getCallee(), adaptor.getCalleeOperands());
    call.setArgAttrsAttr(op.getArgAttrsAttr());
    call.setResAttrsAttr(op.getResAttrsAttr());
    call->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

/// Conversion visitation is not execution order. Every pattern emits safe
/// setters; only this subsequent ordered, block-local scan removes redundancy.
static void deduplicateConfiguration(Region &region) {
  for (Block &block : region) {
    std::array<Value, 3> known{};
    auto invalidate = [&]() { known.fill(Value{}); };
    for (Operation &op : llvm::make_early_inc_range(block)) {
      if (op.getNumRegions()) {
        invalidate();
        for (Region &nested : op.getRegions())
          deduplicateConfiguration(nested);
        continue;
      }
      auto call = dyn_cast<LLVM::CallIntrinsicOp>(op);
      if (call) {
        StringRef name = call.getIntrin();
        int axis = llvm::StringSwitch<int>(name)
                       .Case("llvm.riscv.xsai.zames.msettilemi", 0)
                       .Case("llvm.riscv.xsai.zames.msettileni", 1)
                       .Case("llvm.riscv.xsai.zames.msettileki", 2)
                       .Default(-1);
        if (axis >= 0) {
          Value value = call.getArgs().front();
          llvm::APInt oldConstant, newConstant;
          bool same =
              known[axis] &&
              (known[axis] == value ||
               (matchPattern(known[axis], m_ConstantInt(&oldConstant)) &&
                matchPattern(value, m_ConstantInt(&newConstant)) &&
                oldConstant == newConstant));
          if (same)
            call.erase();
          else
            known[axis] = value;
          continue;
        }
        // These operations observe, but never overwrite, M/N/K. All other
        // intrinsics (including minit/msetcfg) are conservative barriers.
        if (name == "llvm.riscv.xsai.zames.mzero" ||
            name == "llvm.riscv.xsai.zames.mmacc" ||
            name == "llvm.riscv.xsai.zames.mla" ||
            name == "llvm.riscv.xsai.zames.mlat" ||
            name == "llvm.riscv.xsai.zames.mlb" ||
            name == "llvm.riscv.xsai.zames.mlbt" ||
            name == "llvm.riscv.xsai.zames.mlc" ||
            name == "llvm.riscv.xsai.zames.mlct" ||
            name == "llvm.riscv.xsai.zames.msa" ||
            name == "llvm.riscv.xsai.zames.msat" ||
            name == "llvm.riscv.xsai.zames.msb" ||
            name == "llvm.riscv.xsai.zames.msbt" ||
            name == "llvm.riscv.xsai.zames.msc" ||
            name == "llvm.riscv.xsai.zames.msct" ||
            name == "llvm.riscv.xsai.zmasync.msyncregreset" ||
            name == "llvm.riscv.xsai.zmasync.mrelease" ||
            name == "llvm.riscv.xsai.zmasync.macquire" ||
            name == "llvm.riscv.xsai.zmasync.mfence")
          continue;
        invalidate();
      } else if (isa<CallOpInterface, LLVM::InlineAsmOp>(op)) {
        invalidate();
      }
    }
  }
}

// The native sync intrinsics are readnone. Keep scalar accesses and matrix
// transfers on the correct side of the completion boundary even under LLVM
// optimization; an empty memory-clobber asm emits no machine instruction.
static void compilerMemoryBarrier(OpBuilder &builder, Location loc) {
  LLVM::InlineAsmOp::create(
      builder, loc, TypeRange{}, ValueRange{}, "", "~{memory}",
      /*has_side_effects=*/true, /*is_align_stack=*/false,
      LLVM::tailcallkind::TailCallKind::None, /*convergent=*/false,
      LLVM::AsmDialectAttr{}, ArrayAttr{});
}

static void serializeMatrixMemory(Region &region) {
  for (Block &block : region) {
    bool matrixReads = false;
    for (Operation &op : llvm::make_early_inc_range(block)) {
      for (Region &nested : op.getRegions())
        serializeMatrixMemory(nested);
      OpBuilder builder(&op);
      Location loc = op.getLoc();
      if (isa<LoadOp>(op)) {
        if (!matrixReads) {
          compilerMemoryBarrier(builder, loc);
          MfenceOp::create(builder, loc);
          compilerMemoryBarrier(builder, loc);
        }
        matrixReads = true;
        continue;
      }
      if (isa<StoreOp>(op)) {
        auto syncType = SyncType::get(builder.getContext());
        Value sync = MsyncRegResetOp::create(builder, loc, syncType);
        builder.setInsertionPointAfter(&op);
        compilerMemoryBarrier(builder, loc);
        Value released = MreleaseOp::create(builder, loc, syncType, sync);
        Value one = arith::ConstantIndexOp::create(builder, loc, 1);
        MacquireOp::create(builder, loc, released, one);
        compilerMemoryBarrier(builder, loc);
        matrixReads = false;
        continue;
      }
      if (!isa<MmaOp, MzeroOp>(op) &&
          (op.getNumRegions() || !isMemoryEffectFree(&op)))
        matrixReads = false;
    }
  }
}

class LegalizeXSAIAMEForLLVMExport
    : public PassWrapper<LegalizeXSAIAMEForLLVMExport,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeXSAIAMEForLLVMExport)
  LegalizeXSAIAMEForLLVMExport() = default;
  LegalizeXSAIAMEForLLVMExport(const LegalizeXSAIAMEForLLVMExport &other)
      : PassWrapper(other) {}

  Option<bool> synchronous{
      *this, "synchronous",
      llvm::cl::desc("Wait for every matrix store before scalar consumers"),
      llvm::cl::init(false)};

  StringRef getArgument() const final { return "lower-xsai-ame"; }
  StringRef getDescription() const final {
    return "Lower XSAIAME to native LLVM intrinsics with structural SSA "
           "conversion";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<XSAIAMEDialect, LLVM::LLVMDialect, arith::ArithDialect,
                    memref::MemRefDialect, func::FuncDialect,
                    cf::ControlFlowDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    MLIRContext &context = getContext();
    if (synchronous)
      serializeMatrixMemory(getOperation().getBodyRegion());
    LLVMTypeConverter converter(&context);
    // This pass does not lower memref descriptors or scalar/structured
    // dialects. Preserve their types so standalone func/CF/SCF conversion
    // changes only tile/sync carriers. Export clients can instead supply a full
    // converter.
    converter.addConversion([](Type type) { return type; });
    RewritePatternSet patterns(&context);
    populateXSAIAMELegalizeForLLVMExportPatterns(converter, patterns);
    LLVMConversionTarget target(context);
    configureXSAIAMELegalizeForExportTarget(target, converter);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    deduplicateConfiguration(getOperation().getBodyRegion());
  }
};

} // namespace

void mlir::populateXSAIAMELegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  converter.addConversion(
      [](TileType tile) -> Type { return convertTileType(tile); });
  converter.addConversion(
      [](SyncType sync) -> Type { return convertSyncType(sync); });
  converter.addConversion([&converter](FunctionType type) -> Type {
    SmallVector<Type> inputs, results;
    if (failed(converter.convertTypes(type.getInputs(), inputs)) ||
        failed(converter.convertTypes(type.getResults(), results)))
      return {};
    return FunctionType::get(type.getContext(), inputs, results);
  });
  patterns
      .add<FunctionLowering, FunctionConstantLowering, IndirectCallLowering>(
          converter, patterns.getContext());
  patterns.add<LoadLowering, StoreLowering, MmaLowering, MzeroLowering,
               ResetLowering, ReleaseLowering, AcquireLowering, FenceLowering>(
      converter);
  populateCallOpTypeConversionPattern(patterns, converter);
  populateReturnOpTypeConversionPattern(patterns, converter);
  populateBranchOpInterfaceTypeConversionPattern(patterns, converter);
  scf::populateSCFStructuralTypeConversions(converter, patterns);
}

void mlir::configureXSAIAMELegalizeForExportTarget(
    LLVMConversionTarget &target, const LLVMTypeConverter &converter) {
  target.addLegalDialect<LLVM::LLVMDialect, arith::ArithDialect,
                         memref::MemRefDialect>();
  target.addDynamicallyLegalOp<func::FuncOp>([&converter](func::FuncOp op) {
    return converter.isSignatureLegal(op.getFunctionType()) &&
           converter.isLegal(&op.getBody());
  });
  target.addDynamicallyLegalOp<func::CallOp, func::CallIndirectOp,
                               func::ConstantOp, func::ReturnOp>(
      [&converter](Operation *op) { return converter.isLegal(op); });
  target.addDynamicallyLegalDialect<cf::ControlFlowDialect>([&converter](
                                                                Operation *op) {
    if (isa<BranchOpInterface>(op))
      return isLegalForBranchOpInterfaceTypeConversionPattern(op, converter);
    return converter.isLegal(op);
  });
  scf::populateSCFStructuralTypeConversionTarget(converter, target);
  target.addDynamicallyLegalOp<scf::ForOp, scf::IfOp, scf::IndexSwitchOp,
                               scf::ExecuteRegionOp, scf::WhileOp>(
      [&converter](Operation *op) {
        return converter.isLegal(op) &&
               llvm::all_of(op->getRegions(), [&](Region &region) {
                 return converter.isLegal(&region);
               });
      });
  target.addIllegalDialect<XSAIAMEDialect>();
}

namespace buddy::xsaiame {
std::unique_ptr<mlir::Pass> createLegalizeForLLVMExportPass() {
  return std::make_unique<LegalizeXSAIAMEForLLVMExport>();
}
} // namespace buddy::xsaiame

void mlir::buddy::registerLowerXSAIAMEPass() {
  PassRegistration<LegalizeXSAIAMEForLLVMExport>();
}
