//===- LegalizeForLLVMExport.cpp - Unified BOSCAME/SPM/DMA LLVM lowering -===//
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
// `-lower-bosc-ame` is the single BOSCAME lowering entry point.  It owns the
// whole AME / SPM / DMA surface and finishes the module with the standard
// MemRef/Func/Arith/CF/SCF/Vector/Math lowering so that its output is ready
// for `buddy-translate --buddy-to-llvmir`.
//
// The pass runs in four stages:
//
//   1. Checks.  The cross-operation SPM/DMA contract (token lifetime, channel
//      occupancy, aliasing) and the SPM-typed-value rules are verified while
//      the `#bosc_ame.spm<bank>` type annotation is still present.  A module
//      that cannot be lowered is rejected before any type information is lost.
//
//   2. Preparation.  Converter-agnostic rewrites that the standard lowering
//      expects to have run first: linalg to loops, strided-metadata
//      expansion, memref expansion and vector-to-SCF.  They only rewrite
//      operations this pass sees through, never BOSCAME operations.
//
//   3. SCF structure.  Loop-carried BOSCAME matrix values (`vector<4x4xT>`)
//      are converted to the scalable register type with the same type
//      converter used by the final conversion.  Without this an `scf.for`
//      accumulator chain would survive with a fixed vector type and a
//      permanent `unrealized_conversion_cast` bridge.
//
//   4. Conversion.  A single LLVM conversion with one `LLVMTypeConverter`
//      lowers, in one go:
//
//        * `bosc_ame.spm.view` to an LLVM memref descriptor whose base is the
//          fixed bank base address plus the view offset.  The bank is
//          compiled into the address and never has to be recovered, so no
//          workspace root argument is needed and the kernel ABI stays clean.
//          `#bosc_ame.spm<bank>` maps to LLVM address space 0: the two banks
//          are two windows in one address space, not two address spaces.
//        * `bosc_ame.dma.start` / `bosc_ame.dma.wait` to calls into the
//          runtime ABI.  The DMA token becomes the `i64` returned by the
//          start call (the effective source address), which the matching
//          wait hands back to the runtime as its completion barrier.  The
//          wait call returns void and the status check lives in the runtime,
//          so the operation is simply erased.
//        * AME matrix load/store to `bosc_ame.intr.*` intrinsics whose base
//          pointer is decoded from the converted memref descriptor
//          (`alignedPtr + offset * elementBytes`), so nested subviews and
//          strided layouts are folded in exactly once.
//        * the remaining BOSCAME operations to their `bosc_ame.intr.*`
//          counterparts, keeping the existing scalable-vector mapping.
//        * MemRef, Func, Arith, CF, SCF, Vector, Math, Index and UB to the
//          LLVM dialect.
//
//      Leftover `unrealized_conversion_cast` operations are reconciled at the
//      end, mirroring the `-reconcile-unrealized-casts` stage of the shipped
//      pipeline.
//
//===----------------------------------------------------------------------===//

#include "Conversion/BOSCAMEHelpers/BOSCAMEHelpers.h"
#include "Dialect/BOSCAME/BOSCAMEAttrs.h"
#include "Dialect/BOSCAME/BOSCAMEDialect.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/BOSCAMETypes.h"
#include "Dialect/BOSCAME/SpmDma.h"
#include "Dialect/BOSCAME/TileOps.h"
#include "Dialect/BOSCAME/Transform.h"
#include "Dialect/BOSCAME/Transforms/BOSCAMETileLowering.h"

#include "BOSCAMEConversionUtils.h"

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"

#include <memory>
#include <optional>
#include <string>

using namespace mlir;
using namespace buddy::boscame;
using namespace buddy::boscame::spmdma;

namespace {

//===----------------------------------------------------------------------===//
// BOSCAME matrix register types
//===----------------------------------------------------------------------===//

// The scalable register type a matrix value is lowered to, and the descriptor
// decoding helpers, are shared with the tile lowering; see
// `BOSCAMEConversionUtils.h`.
using buddy::boscame::detail::createBOSCAMEIntrinsicOp;
using buddy::boscame::detail::extractPointerFromDescriptor;
using buddy::boscame::detail::getBOSCAMERegisterType;

//===----------------------------------------------------------------------===//
// SPM storage space
//===----------------------------------------------------------------------===//

// True when `type` is a ranked memref in an SPM memory space.
bool hasSpmSpace(Type type) {
  auto memref = dyn_cast_or_null<MemRefType>(type);
  return memref && isa_and_nonnull<SPMAttr>(memref.getMemorySpace());
}

// The fixed hardware base address of `bank`, as a byte address.
uint64_t getBankBaseAddress(int64_t bank) {
  assert(bank >= 0 && bank < kSpmBankCount && "bank out of range");
  return kSpmBankBaseAddress[bank];
}

// Operations whose result may legitimately keep an SPM memory space: they
// only re-describe a buffer `bosc_ame.spm.view` already anchored in a bank
// and never change the base address.  Anything else would need an SPM
// allocation the hardware does not hand out and is rejected.
bool preservesSpmSpace(Operation *op) {
  return isa<SpmViewOp, memref::ViewOp, memref::SubViewOp, memref::CastOp,
             memref::CollapseShapeOp, memref::ExpandShapeOp,
             memref::TransposeOp, memref::ReinterpretCastOp,
             memref::MemorySpaceCastOp, memref::AssumeAlignmentOp,
             memref::ExtractStridedMetadataOp, arith::SelectOp, scf::IfOp,
             scf::ForOp, scf::WhileOp, scf::IndexSwitchOp>(op);
}

// Reject SPM-typed values this pass cannot give an address to: an SPM-typed
// function argument has no statically known position inside its bank, and an
// SPM-typed value produced by an operation other than a view of an existing
// SPM buffer has no address at all.  SPM memory-space casts that change the
// bank are rejected too.
LogicalResult validateSpmSurface(ModuleOp module) {
  bool ok = true;
  module.walk([&](Operation *op) {
    for (Value result : op->getResults()) {
      if (!hasSpmSpace(result.getType()) || preservesSpmSpace(op))
        continue;
      op->emitOpError()
          << "produces an SPM-typed value, but only `bosc_ame.spm.view` and "
             "memref view-like operations may carry `"
          << SPMAttr::getMnemonic()
          << "`; this value has no SPM address and cannot be lowered";
      ok = false;
    }
    if (auto cast = dyn_cast<memref::MemorySpaceCastOp>(op)) {
      std::optional<int64_t> sourceBank =
          getSpmBank(cast.getSource().getType());
      std::optional<int64_t> destBank = getSpmBank(cast.getDest().getType());
      if (sourceBank && destBank && *sourceBank != *destBank) {
        cast.emitOpError() << "cannot reinterpret an SPM buffer in bank "
                           << *sourceBank << " as bank " << *destBank;
        ok = false;
      }
    }
    // SPM is a fixed hardware workspace: it is addressed through
    // `bosc_ame.spm.view`, which compiles the bank base into the address.  A
    // kernel that receives an SPM memref argument would need a runtime base
    // address the ABI no longer carries.
    if (auto function = dyn_cast<FunctionOpInterface>(op)) {
      for (BlockArgument argument : function.getArguments()) {
        if (!hasSpmSpace(argument.getType()))
          continue;
        function->emitOpError()
            << "takes an SPM memref argument " << argument.getType()
            << "; SPM is addressed through `bosc_ame.spm.view` at the fixed "
               "bank base address and is not an ABI parameter";
        ok = false;
      }
    }
  });
  return success(ok);
}

//===----------------------------------------------------------------------===//
// DMA runtime declarations
//===----------------------------------------------------------------------===//

// Declare the two runtime entry points the DMA lowering calls, but only when
// the module actually contains a request.  A module without DMA must not gain
// two external declarations.
void declareDmaRuntimeHelpers(ModuleOp module) {
  bool needed = false;
  module.walk([&](Operation *op) { needed |= isa<DmaStartOp, DmaWaitOp>(op); });
  if (!needed)
    return;

  MLIRContext *context = module.getContext();
  Type i32Type = IntegerType::get(context, 32);
  Type i64Type = IntegerType::get(context, 64);
  Type voidType = LLVM::LLVMVoidType::get(context);
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());
  if (!module.lookupSymbol<LLVM::LLVMFuncOp>(kDmaStartHelperName))
    LLVM::LLVMFuncOp::create(builder, module.getLoc(), kDmaStartHelperName,
                             LLVM::LLVMFunctionType::get(
                                 i64Type, {i32Type, i64Type, i64Type, i64Type}),
                             LLVM::Linkage::External, false, LLVM::CConv::C);
  // The wait helper checks the hardware status itself: a normal return means
  // "completed and succeeded", a failure takes the runtime error path.
  if (!module.lookupSymbol<LLVM::LLVMFuncOp>(kDmaWaitHelperName))
    LLVM::LLVMFuncOp::create(
        builder, module.getLoc(), kDmaWaitHelperName,
        LLVM::LLVMFunctionType::get(voidType, {i32Type, i64Type}),
        LLVM::Linkage::External, false, LLVM::CConv::C);
}

LogicalResult declareFrameRuntimeHelpers(ModuleOp module) {
  bool needed = false;
  module.walk(
      [&](Operation *op) { needed |= isa<DDRFrameOp, DDRReleaseOp>(op); });
  if (!needed)
    return success();
  OpBuilder b(module.getContext());
  b.setInsertionPointToStart(module.getBody());
  Type i64 = b.getI64Type();
  Type voidType = LLVM::LLVMVoidType::get(b.getContext());
  for (auto [name, type] :
       {std::pair<StringRef, LLVM::LLVMFunctionType>{
            "bosc_ame_private_ddr_acquire",
            LLVM::LLVMFunctionType::get(i64, {i64})},
        {"bosc_ame_private_ddr_release",
         LLVM::LLVMFunctionType::get(voidType, {i64, i64})}}) {
    if (Operation *existing = SymbolTable::lookupSymbolIn(module, name)) {
      auto function = dyn_cast<LLVM::LLVMFuncOp>(existing);
      if (!function || function.getFunctionType() != type)
        return existing->emitError(
            "incompatible private DDR runtime declaration");
    } else
      LLVM::LLVMFuncOp::create(b, module.getLoc(), name, type);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Type converter
//===----------------------------------------------------------------------===//

static void addBOSCAMETypeConversions(LLVMTypeConverter &converter) {
  // LLVMTypeConverter normally lowers a fixed 2-D vector to nested LLVM
  // arrays.  BOSCAME matrix registers are represented by scalable vectors in
  // the backend, so keep those values in the register form throughout this
  // conversion instead of materializing unrealized array casts.
  converter.addConversion([](VectorType type) -> std::optional<Type> {
    if (type.getRank() != 2)
      return std::nullopt;
    return getBOSCAMERegisterType(type);
  });

  // A DMA token is the effective source address of the request, i.e. an i64.
  converter.addConversion([](DMATokenType type) -> Type {
    return IntegerType::get(type.getContext(), 64);
  });

  // Both SPM banks are windows into the same address space, so the memory
  // space annotation lowers to address space 0.  Mapping bank 0/1 to LLVM
  // address spaces 0/1 would split one physical space into two.
  converter.addTypeAttributeConversion(
      [](MemRefType type,
         SPMAttr attr) -> TypeConverter::AttributeConversionResult {
        return IntegerAttr::get(IntegerType::get(attr.getContext(), 64), 0);
      });
}

//===----------------------------------------------------------------------===//
// Conversion helpers
//===----------------------------------------------------------------------===//

static bool isMemRefValue(Value value) {
  return isa<MemRefType, UnrankedMemRefType>(value.getType());
}

// Materialize the effective byte address of a converted memref descriptor as
// an `i64`, which is what the DMA runtime ABI takes.
static Value extractAddressFromDescriptor(OpBuilder &builder, Location loc,
                                          Value descriptor,
                                          Type originalMemRefType,
                                          const LLVMTypeConverter &converter) {
  // DMA lengths are bytes, while memref descriptor offsets are ELEMENTS.
  // bufferPtr performs the typed GEP (including dynamic offsets) exactly once.
  auto type = dyn_cast<MemRefType>(originalMemRefType);
  if (!type || !getSpmElementByteWidth(type.getElementType()))
    return Value();
  Value pointer = extractPointerFromDescriptor(builder, loc, descriptor,
                                               originalMemRefType, converter);
  if (!pointer)
    return Value();
  return LLVM::PtrToIntOp::create(
      builder, loc, IntegerType::get(builder.getContext(), 64), pointer);
}

// Immediate BOSCAME operations carry an I64Attr. LLVM_IntrOpBase needs the
// same immediate represented as an SSA i64 constant on the bridge op.
static Value materializeBOSCAMEImmediate(Operation *op,
                                         ConversionPatternRewriter &rewriter) {
  static const char *const attributeNames[] = {"timm", "tilem", "tilen",
                                               "tilek", "imm"};
  for (const char *name : attributeNames) {
    IntegerAttr attribute = op->getAttrOfType<IntegerAttr>(name);
    if (!attribute)
      continue;
    Type i64Type = IntegerType::get(rewriter.getContext(), 64);
    return LLVM::ConstantOp::create(
        rewriter, op->getLoc(), i64Type,
        rewriter.getI64IntegerAttr(attribute.getInt()));
  }
  return Value();
}

//===----------------------------------------------------------------------===//
// BOSCAME -> bosc_ame.intr.*
//===----------------------------------------------------------------------===//

// The old pass had one typed template and one patterns.add<> call per
// instruction. Their bodies were identical except for operation name and
// register-number attributes. The SSA ODS no longer has those attributes, so
// this generic conversion is both smaller and semantically correct.
class BOSCAMEToIntrinsicLowering : public ConversionPattern {
public:
  BOSCAMEToIntrinsicLowering(const TypeConverter &converter,
                             MLIRContext *context)
      : ConversionPattern(converter, Pattern::MatchAnyOpTypeTag(),
                          PatternBenefit(1), context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef name = op->getName().getStringRef();
    if (!name.starts_with("bosc_ame.") || name.starts_with("bosc_ame.intr."))
      return failure();

    // The SPM view and the DMA requests have dedicated conversion patterns
    // with a higher benefit; never fabricate an intrinsic for them.
    if (isa<SpmViewOp, DDRFrameOp, DDRReleaseOp, DmaStartOp, DmaWaitOp,
            MatrixCopyOp>(op))
      return failure();

    // The tile operations are handled by `BOSCAMETileLowering.cpp`, which
    // derives the hardware configuration and the physical stride from the
    // operand types.  The generic pattern must not turn them into a
    // `bosc_ame.intr.tile.*` operation: those do not exist, and even if they
    // did, the naming convention of this pattern cannot express the slot, the
    // stride or the configuration sequence.
    if (isa<TileLoadAOp, TileLoadBOp, TileLoadAccOp, TileZeroAccOp, TileMmaOp,
            TileStoreAccOp>(op))
      return failure();

    const auto &converter =
        *static_cast<const LLVMTypeConverter *>(typeConverter);
    Location loc = op->getLoc();

    // Matrix load: memref, i64 stride -> scalable LLVM vector.
    if (op->getNumResults() == 1 && op->getNumOperands() == 2 &&
        isMemRefValue(op->getOperand(0))) {
      Type resultType = getBOSCAMERegisterType(op->getResult(0).getType());
      if (!resultType)
        return failure();
      Value base = extractPointerFromDescriptor(
          rewriter, loc, operands[0], op->getOperand(0).getType(), converter);
      if (!base)
        return failure();
      SmallVector<Value> loadOperands{base, operands[1]};
      Operation *intrinsic = createBOSCAMEIntrinsicOp(
          rewriter, loc, name, TypeRange{resultType}, loadOperands);
      rewriter.replaceOp(op, intrinsic->getResults());
      return success();
    }

    // Matrix store: scalable LLVM vector, memref, i64 stride -> no result.
    if (op->getNumResults() == 0 && op->getNumOperands() == 3 &&
        isMemRefValue(op->getOperand(1))) {
      Value base = extractPointerFromDescriptor(
          rewriter, loc, operands[1], op->getOperand(1).getType(), converter);
      if (!base)
        return failure();
      createBOSCAMEIntrinsicOp(rewriter, loc, name, TypeRange(),
                               ValueRange{operands[0], base, operands[2]});
      rewriter.eraseOp(op);
      return success();
    }

    SmallVector<Value, 4> intrinsicOperands(operands.begin(), operands.end());
    if (Value immediate = materializeBOSCAMEImmediate(op, rewriter))
      intrinsicOperands.push_back(immediate);

    SmallVector<Type, 1> resultTypes;
    for (Value result : op->getResults()) {
      Type resultType = result.getType();
      if (isa<VectorType>(resultType))
        resultType = getBOSCAMERegisterType(resultType);
      if (!resultType)
        return failure();
      resultTypes.push_back(resultType);
    }

    Operation *intrinsic = createBOSCAMEIntrinsicOp(
        rewriter, loc, name, TypeRange(resultTypes), intrinsicOperands);
    rewriter.replaceOp(op, intrinsic->getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// bosc_ame.spm.view -> LLVM memref descriptor at the fixed bank base
//===----------------------------------------------------------------------===//

// `bosc_ame.spm.view {bank, offset}` becomes an LLVM memref descriptor whose
// base is `bankBase[bank] + offset`.  The bank is a compile-time hardware
// constant, so no root argument, no `memref.view` and no runtime-provided
// base pointer is involved.  The descriptor offset is zero (the verifier
// guarantees a contiguous, zero-offset view), so later address computations
// add the descriptor offset exactly once.
class SpmViewOpConversion : public ConvertToLLVMPattern {
public:
  SpmViewOpConversion(const LLVMTypeConverter &converter, MLIRContext *context)
      : ConvertToLLVMPattern(SpmViewOp::getOperationName(), context, converter,
                             PatternBenefit(2)) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto view = cast<SpmViewOp>(op);
    auto memrefType = dyn_cast<MemRefType>(view.getView().getType());
    if (!memrefType)
      return failure();
    int64_t bank = static_cast<int64_t>(view.getBank());
    if (bank < 0 || bank >= kSpmBankCount)
      return failure();

    Location loc = view.getLoc();
    uint64_t address = getBankBaseAddress(bank) + view.getOffset();
    Type i64Type = IntegerType::get(rewriter.getContext(), 64);
    Value addressValue = LLVM::ConstantOp::create(
        rewriter, loc, i64Type,
        rewriter.getI64IntegerAttr(static_cast<int64_t>(address)));
    Type pointerType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
    Value base =
        LLVM::IntToPtrOp::create(rewriter, loc, pointerType, addressValue);
    Value descriptor = MemRefDescriptor::fromStaticShape(
        rewriter, loc, *getTypeConverter(), memrefType, base);
    rewriter.replaceOp(op, descriptor);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// bosc_ame.dma.start / bosc_ame.dma.wait -> runtime calls
//===----------------------------------------------------------------------===//

// `bosc_ame.dma.start` becomes a call to `bosc_ame_dma_start(channel, src,
// dst, len)`.  The runtime returns the effective source address, which *is*
// the request token, so the call result replaces the token value directly.
class DDRFrameConversion : public ConvertToLLVMPattern {
public:
  DDRFrameConversion(const LLVMTypeConverter &converter, MLIRContext *context)
      : ConvertToLLVMPattern(DDRFrameOp::getOperationName(), context, converter,
                             PatternBenefit(3)) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value>,
                                ConversionPatternRewriter &b) const override {
    auto type = cast<MemRefType>(op->getResult(0).getType());
    Value bytes =
        LLVM::ConstantOp::create(b, op->getLoc(), b.getI64Type(),
                                 b.getI64IntegerAttr(type.getDimSize(0)));
    auto call = LLVM::CallOp::create(
        b, op->getLoc(), TypeRange{b.getI64Type()},
        FlatSymbolRefAttr::get(b.getContext(), "bosc_ame_private_ddr_acquire"),
        ValueRange{bytes});
    Value pointer = LLVM::IntToPtrOp::create(
        b, op->getLoc(), LLVM::LLVMPointerType::get(b.getContext()),
        call.getResult());
    Value descriptor = MemRefDescriptor::fromStaticShape(
        b, op->getLoc(), *getTypeConverter(), type, pointer);
    b.replaceOp(op, descriptor);
    return success();
  }
};
class DDRReleaseConversion : public ConvertToLLVMPattern {
public:
  DDRReleaseConversion(const LLVMTypeConverter &converter, MLIRContext *context)
      : ConvertToLLVMPattern(DDRReleaseOp::getOperationName(), context,
                             converter, PatternBenefit(3)) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &b) const override {
    auto type = cast<MemRefType>(op->getOperand(0).getType());
    Value bytes =
        LLVM::ConstantOp::create(b, op->getLoc(), b.getI64Type(),
                                 b.getI64IntegerAttr(type.getDimSize(0)));
    Value address = extractAddressFromDescriptor(b, op->getLoc(), operands[0],
                                                 type, *getTypeConverter());
    LLVM::CallOp::create(
        b, op->getLoc(), TypeRange{},
        FlatSymbolRefAttr::get(b.getContext(), "bosc_ame_private_ddr_release"),
        ValueRange{address, bytes});
    b.eraseOp(op);
    return success();
  }
};

class DmaStartOpConversion : public ConvertToLLVMPattern {
public:
  DmaStartOpConversion(const LLVMTypeConverter &converter, MLIRContext *context)
      : ConvertToLLVMPattern(DmaStartOp::getOperationName(), context, converter,
                             PatternBenefit(2)) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto start = cast<DmaStartOp>(op);
    if (!getDmaEndpoint(dyn_cast<MemRefType>(start.getSrc().getType())) ||
        !getDmaEndpoint(dyn_cast<MemRefType>(start.getDst().getType())))
      return start.emitOpError()
             << "DMA endpoints must be 1-D i8 memrefs with a static length";

    Location loc = start.getLoc();
    const auto &converter =
        *static_cast<const LLVMTypeConverter *>(getTypeConverter());
    Value channel = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(static_cast<int32_t>(start.getChannel())));
    Value src = extractAddressFromDescriptor(
        rewriter, loc, operands[0], start.getSrc().getType(), converter);
    Value dst = extractAddressFromDescriptor(
        rewriter, loc, operands[1], start.getDst().getType(), converter);
    if (!src || !dst)
      return failure();
    // `bytes` is an index operand; the converter has already turned it into the
    // LLVM index type (i64), which is the runtime's length type as well.
    Value length = operands[2];

    FlatSymbolRefAttr callee =
        FlatSymbolRefAttr::get(rewriter.getContext(), kDmaStartHelperName);
    Type i64Type = IntegerType::get(rewriter.getContext(), 64);
    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{i64Type}, callee,
                                     ValueRange{channel, src, dst, length});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// `bosc_ame.dma.wait` becomes a call to `bosc_ame_dma_wait(channel, token)`.
// The channel comes from the *original* token type (the token has already
// been converted to the start call's `i64` result), and the converted token
// is the effective source address the runtime uses as the completion
// barrier.  The helper returns void and owns the status check, so the wait
// operation itself is erased.  No start/wait bookkeeping map is needed.
class DmaWaitOpConversion : public ConvertToLLVMPattern {
public:
  DmaWaitOpConversion(const LLVMTypeConverter &converter, MLIRContext *context)
      : ConvertToLLVMPattern(DmaWaitOp::getOperationName(), context, converter,
                             PatternBenefit(2)) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto wait = cast<DmaWaitOp>(op);
    auto tokenType = dyn_cast<DMATokenType>(wait.getToken().getType());
    if (!tokenType)
      return wait.emitOpError() << "operand must be a `"
                                << DMATokenType::getMnemonic() << "` value";

    Location loc = wait.getLoc();
    Value channel = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(
            static_cast<int32_t>(tokenType.getChannel())));
    FlatSymbolRefAttr callee =
        FlatSymbolRefAttr::get(rewriter.getContext(), kDmaWaitHelperName);
    LLVM::CallOp::create(rewriter, loc, TypeRange{}, callee,
                         ValueRange{channel, operands[0]});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Preparation stages
//===----------------------------------------------------------------------===//

// Private provenance for the snapshot operations introduced below. Dedicated
// conversion patterns consume it and make only these accesses volatile, so
// LLVM cannot forward a vector store into scalar extracts/reductions again.
constexpr StringLiteral kScalarOrderedSnapshot =
    "bosc_ame.scalar_ordered_snapshot";

class ScalarOrderedSnapshotLoadLowering
    : public ConvertOpToLLVMPattern<memref::LoadOp> {
public:
  explicit ScalarOrderedSnapshotLoadLowering(const LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<memref::LoadOp>(converter, PatternBenefit(20)) {}

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op->hasAttr(kScalarOrderedSnapshot))
      return failure();
    Value address =
        getStridedElementPtr(rewriter, op.getLoc(), op.getMemRefType(),
                             adaptor.getMemref(), adaptor.getIndices());
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, rewriter.getF32Type(),
                                              address, 4, true);
    return success();
  }
};

class ScalarOrderedSnapshotStoreLowering
    : public ConvertOpToLLVMPattern<vector::StoreOp> {
public:
  explicit ScalarOrderedSnapshotStoreLowering(
      const LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<vector::StoreOp>(converter, PatternBenefit(20)) {
  }

  LogicalResult
  matchAndRewrite(vector::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op->hasAttr(kScalarOrderedSnapshot))
      return failure();
    Value address =
        getStridedElementPtr(rewriter, op.getLoc(), op.getMemRefType(),
                             adaptor.getBase(), adaptor.getIndices());
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getValueToStore(),
                                               address, 4, true);
    return success();
  }
};

// Keep the reduction accumulator in scalar SSA from the start. Reconstructing
// it from machine vector seed/destination registers after register allocation
// is unsafe on the NR target. Only the ordered f32 ADD surface is selected;
// other kinds/types and nonordered reductions retain their existing lowering.
// Ordered f32 ADD forms outside the checked surface fail closed when enabled.
// Compiler-generated vector operations retain the usual
// vstart=0 contract; no physical vector CSR or VLEN assumption is introduced.
LogicalResult scalarizeF32OrderedReductions(ModuleOp module, bool rewrite) {
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (function.isExternal())
      continue;
    SmallVector<vector::ReductionOp> reductions;
    int64_t maxLanes = 0;
    bool unsupported = false;
    function.walk([&](vector::ReductionOp reduction) {
      VectorType type = reduction.getVector().getType();
      if (!type.getElementType().isF32() ||
          reduction.getKind() != vector::CombiningKind::ADD ||
          reduction.getFastmath() != arith::FastMathFlags::none)
        return;
      auto reject = [&](StringRef reason) {
        reduction.emitOpError()
            << "scalar-f32-ordered-reductions requires " << reason;
        unsupported = true;
      };
      if (type.getRank() != 1 || type.isScalable() || !reduction.getAcc() ||
          reduction.isMasked()) {
        reject("an unmasked fixed rank-one vector and an explicit scalar "
               "accumulator");
        return;
      }
      // A single entry snapshot must not be shared by parallel invocations of
      // the reduction. Unknown region semantics are deliberately rejected.
      for (Operation *parent = reduction->getParentOp();
           parent != function.getOperation(); parent = parent->getParentOp()) {
        if (!isa<scf::ForOp, scf::WhileOp, scf::IfOp, scf::ExecuteRegionOp,
                 scf::IndexSwitchOp>(parent)) {
          reject("sequential func/SCF regions (parallel, async and unknown "
                 "region semantics cannot share an entry snapshot)");
          return;
        }
      }
      reductions.push_back(reduction);
      maxLanes = std::max(maxLanes, type.getDimSize(0));
    });
    if (unsupported)
      return failure();
    if (!rewrite || reductions.empty())
      continue;
    // One statically bounded entry allocation is shared by sequential
    // reductions, including reductions nested inside loops. Never allocate
    // another snapshot on each dynamic reduction invocation.
    OpBuilder entryBuilder = OpBuilder::atBlockBegin(&function.front());
    Value snapshot = memref::AllocaOp::create(
        entryBuilder, function.getLoc(),
        MemRefType::get({maxLanes}, entryBuilder.getF32Type()),
        entryBuilder.getI64IntegerAttr(16));
    for (vector::ReductionOp reduction : reductions) {
      OpBuilder builder(reduction);
      Location loc = reduction.getLoc();
      Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
      Value lanes = arith::ConstantIndexOp::create(
          builder, loc, reduction.getVector().getType().getDimSize(0));
      Value one = arith::ConstantIndexOp::create(builder, loc, 1);
      auto store = vector::StoreOp::create(builder, loc, reduction.getVector(),
                                           snapshot, ValueRange{zero});
      store->setAttr(kScalarOrderedSnapshot, builder.getUnitAttr());
      auto loop = scf::ForOp::create(builder, loc, zero, lanes, one,
                                     ValueRange{reduction.getAcc()});
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(loop.getBody());
        auto lane = memref::LoadOp::create(builder, loc, snapshot,
                                           ValueRange{loop.getInductionVar()});
        lane->setAttr(kScalarOrderedSnapshot, builder.getUnitAttr());
        Value next = arith::AddFOp::create(
            builder, loc, loop.getRegionIterArgs()[0], lane.getResult());
        scf::YieldOp::create(builder, loc, next);
      }
      reduction.replaceAllUsesWith(loop.getResult(0));
      reduction.erase();
    }
  }
  return success();
}

// Snapshot the vector but keep max's seed, accumulator and result in scalar
// SSA. Use the matching arithmetic operation: maximum propagates NaNs and
// orders signed zeros, whereas maxnum ignores a single NaN and permits either
// sign for mixed zeros. No invented -infinity seed is used for an unseeded
// reduction: that would incorrectly turn an all-NaN maxnum into -infinity.
// The source vector already contains any materialized padding, so every lane
// participates; masked/scalable forms fail closed before mask lowering.
LogicalResult scalarizeF32MaxReductions(ModuleOp module, bool rewrite) {
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (function.isExternal())
      continue;
    SmallVector<vector::ReductionOp> reductions;
    int64_t maxLanes = 0;
    bool unsupported = false;
    function.walk([&](vector::ReductionOp reduction) {
      VectorType type = reduction.getVector().getType();
      auto kind = reduction.getKind();
      if (!type.getElementType().isF32() ||
          (kind != vector::CombiningKind::MAXNUMF &&
           kind != vector::CombiningKind::MAXIMUMF &&
           kind != vector::CombiningKind::MAXIMUMNUMF))
        return;
      auto reject = [&](StringRef reason) {
        reduction.emitOpError()
            << "scalar-f32-max-reductions requires " << reason;
        unsupported = true;
      };
      if (type.getRank() != 1 || type.isScalable() || reduction.isMasked()) {
        reject("an unmasked fixed rank-one vector");
        return;
      }
      for (Operation *parent = reduction->getParentOp();
           parent != function.getOperation(); parent = parent->getParentOp()) {
        if (!isa<scf::ForOp, scf::WhileOp, scf::IfOp, scf::ExecuteRegionOp,
                 scf::IndexSwitchOp>(parent)) {
          reject("sequential func/SCF regions (parallel, async and unknown "
                 "region semantics cannot share an entry snapshot)");
          return;
        }
      }
      reductions.push_back(reduction);
      maxLanes = std::max(maxLanes, type.getDimSize(0));
    });
    if (unsupported)
      return failure();
    if (!rewrite || reductions.empty())
      continue;
    OpBuilder entryBuilder = OpBuilder::atBlockBegin(&function.front());
    Value snapshot = memref::AllocaOp::create(
        entryBuilder, function.getLoc(),
        MemRefType::get({maxLanes}, entryBuilder.getF32Type()),
        entryBuilder.getI64IntegerAttr(16));
    for (vector::ReductionOp reduction : reductions) {
      OpBuilder builder(reduction);
      Location loc = reduction.getLoc();
      Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
      Value lanes = arith::ConstantIndexOp::create(
          builder, loc, reduction.getVector().getType().getDimSize(0));
      Value one = arith::ConstantIndexOp::create(builder, loc, 1);
      auto store = vector::StoreOp::create(builder, loc, reduction.getVector(),
                                           snapshot, ValueRange{zero});
      store->setAttr(kScalarOrderedSnapshot, builder.getUnitAttr());
      Value initial = reduction.getAcc();
      Value begin = zero;
      if (!initial) {
        auto first =
            memref::LoadOp::create(builder, loc, snapshot, ValueRange{zero});
        first->setAttr(kScalarOrderedSnapshot, builder.getUnitAttr());
        initial = first;
        begin = one;
      }
      auto loop = scf::ForOp::create(builder, loc, begin, lanes, one,
                                     ValueRange{initial});
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(loop.getBody());
        auto lane = memref::LoadOp::create(builder, loc, snapshot,
                                           ValueRange{loop.getInductionVar()});
        lane->setAttr(kScalarOrderedSnapshot, builder.getUnitAttr());
        Value acc = loop.getRegionIterArgs()[0];
        Value next;
        switch (reduction.getKind()) {
        case vector::CombiningKind::MAXNUMF:
          next = arith::MaxNumFOp::create(builder, loc, acc, lane.getResult(),
                                          reduction.getFastmath());
          break;
        case vector::CombiningKind::MAXIMUMF:
          next = arith::MaximumFOp::create(builder, loc, acc, lane.getResult(),
                                           reduction.getFastmath());
          break;
        case vector::CombiningKind::MAXIMUMNUMF:
          next = arith::MaximumNumFOp::create(
              builder, loc, acc, lane.getResult(), reduction.getFastmath());
          break;
        default:
          llvm_unreachable("only selected floating-point max reductions");
        }
        scf::YieldOp::create(builder, loc, next);
      }
      reduction.replaceAllUsesWith(loop.getResult(0));
      reduction.erase();
    }
  }
  return success();
}

// Batch only compiler-generated pure fills/copies. One vector holds at most
// 64 bytes; CPU DDR read/write transitions still bracket every real access.
// Matching the exact loop body excludes arithmetic, reductions and aliases.
void vectorizeGeneratedMemoryTasks(ModuleOp module, MLIRContext &context) {
  auto target = module->getAttrOfType<StringAttr>("bosc_ame.target");
  if (!target || target.getValue() != "nr-fpga" ||
      !module->hasAttr(buddy::boscame::lowering::kCpuDdrPolicy))
    return;
  SmallVector<scf::ExecuteRegionOp> regions;
  module.walk([&](scf::ExecuteRegionOp op) {
    auto task = op->getAttrOfType<DictionaryAttr>("contract.task");
    auto kind = task ? task.getAs<StringAttr>("kind") : StringAttr();
    if (kind &&
        (kind.getValue() == "fill" || kind.getValue() == "memory-load" ||
         kind.getValue() == "memory-store"))
      regions.push_back(op);
  });
  for (scf::ExecuteRegionOp region : regions) {
    if (!region.getRegion().hasOneBlock())
      continue;
    // Emission keeps loop-bound constants beside each loop. Accept only
    // those constants and a single loop; never cross other computations.
    auto onlyLoopAndConstants = [](Block &block) -> scf::ForOp {
      scf::ForOp loop;
      for (Operation &op : block.without_terminator()) {
        if (isa<arith::ConstantOp>(op))
          continue;
        auto candidate = dyn_cast<scf::ForOp>(op);
        if (!candidate || loop)
          return {};
        loop = candidate;
      }
      return loop;
    };
    Block &body = region.getRegion().front();
    auto outer = onlyLoopAndConstants(body);
    if (!outer || outer.getNumResults())
      continue;
    auto inner = onlyLoopAndConstants(*outer.getBody());
    if (!inner || inner.getNumResults())
      continue;
    auto bound = [](Value value) -> std::optional<int64_t> {
      APInt integer;
      if (matchPattern(value, m_ConstantInt(&integer)))
        return integer.getSExtValue();
      return std::nullopt;
    };
    auto lo = bound(outer.getLowerBound()), hi = bound(outer.getUpperBound());
    auto il = bound(inner.getLowerBound()), ih = bound(inner.getUpperBound());
    auto os = bound(outer.getStep()), is = bound(inner.getStep());
    if (!lo || !hi || !il || !ih || !os || !is || *lo != 0 || *il != 0 ||
        *os != 1 || *is != 1 || *hi <= 0 || *ih <= 0)
      continue;
    SmallVector<Operation *> ops;
    for (Operation &op : inner.getBody()->without_terminator())
      ops.push_back(&op);
    memref::LoadOp load;
    memref::StoreOp store;
    if (ops.size() == 1)
      store = dyn_cast<memref::StoreOp>(ops[0]);
    else if (ops.size() == 2) {
      load = dyn_cast<memref::LoadOp>(ops[0]);
      store = dyn_cast<memref::StoreOp>(ops[1]);
    }
    if (!store || (ops.size() == 2 && !load) ||
        (load && store.getValue() != load.getResult()) ||
        (!load && !store.getValue().getDefiningOp<arith::ConstantOp>()))
      continue;
    auto matches = [&](ValueRange indices) {
      return indices.size() == 2 && indices[0] == outer.getInductionVar() &&
             indices[1] == inner.getInductionVar();
    };
    if (!matches(store.getIndices()) || (load && !matches(load.getIndices())))
      continue;
    auto dest = store.getMemRefType();
    auto source = load ? load.getMemRefType() : dest;
    auto element = dest.getElementType();
    if (source.getElementType() != element ||
        !(element.isInteger(8) || element.isInteger(32) || element.isF32()))
      continue;
    SmallVector<int64_t> ds, ss;
    int64_t offset;
    if (failed(dest.getStridesAndOffset(ds, offset)) ||
        failed(source.getStridesAndOffset(ss, offset)))
      continue;
    unsigned axis = ds[1] == 1 && ss[1] == 1 ? 1 : 0;
    if (ds[axis] != 1 || ss[axis] != 1 ||
        (load && store.getMemref() == load.getMemref()))
      continue;
    if (!dest.hasStaticShape() || !source.hasStaticShape() ||
        *hi > dest.getDimSize(0) || *ih > dest.getDimSize(1) ||
        *hi > source.getDimSize(0) || *ih > source.getDimSize(1))
      continue;
    // Different SSA values alone do not prove non-aliasing. Require the
    // public disjoint ABI and one private frame plus one external argument.
    auto root = [](Value value) {
      while (auto view =
                 dyn_cast_or_null<ViewLikeOpInterface>(value.getDefiningOp()))
        value = view.getViewSource();
      return value;
    };
    Value destRoot = root(store.getMemref());
    bool destPrivate = isa_and_nonnull<DDRFrameOp>(destRoot.getDefiningOp());
    if (load) {
      Value sourceRoot = root(load.getMemref());
      bool sourcePrivate =
          isa_and_nonnull<DDRFrameOp>(sourceRoot.getDefiningOp());
      auto abi = module->getAttrOfType<DictionaryAttr>("contract.abi");
      auto disjoint = abi ? abi.getAs<BoolAttr>("args_disjoint") : BoolAttr();
      auto externalArgument = [](Value value) {
        auto arg = dyn_cast<BlockArgument>(value);
        return arg && isa<func::FuncOp>(arg.getOwner()->getParentOp());
      };
      if (!disjoint || !disjoint.getValue() || sourceRoot == destRoot ||
          !((sourcePrivate && externalArgument(destRoot)) ||
            (destPrivate && externalArgument(sourceRoot))))
        continue;
    } else if (!destPrivate) {
      continue;
    }
    // Loop-local descriptors need a more general dependence analysis.
    if (auto def = store.getMemref().getDefiningOp();
        def && outer->isProperAncestor(def))
      continue;
    if (load) {
      if (auto def = load.getMemref().getDefiningOp();
          def && outer->isProperAncestor(def))
        continue;
    }
    int64_t contiguous = axis == 1 ? *ih : *hi;
    int64_t rows = axis == 1 ? *hi : *ih;
    if (rows > 1 && (ds[1 - axis] < contiguous || ss[1 - axis] < contiguous))
      continue;
    unsigned bytes = element.getIntOrFloatBitWidth() / 8;
    int64_t lanes = std::min<int64_t>(64 / bytes, contiguous);
    while (lanes > 1 && contiguous % lanes)
      --lanes;
    if (lanes < 2)
      continue;
    OpBuilder b(outer);
    Location loc = outer.getLoc();
    Value zero = arith::ConstantIndexOp::create(b, loc, 0);
    Value rowEnd = arith::ConstantIndexOp::create(b, loc, rows);
    Value colEnd = arith::ConstantIndexOp::create(b, loc, contiguous);
    Value one = arith::ConstantIndexOp::create(b, loc, 1);
    Value step = arith::ConstantIndexOp::create(b, loc, lanes);
    auto row = scf::ForOp::create(b, loc, zero, rowEnd, one);
    b.setInsertionPointToStart(row.getBody());
    auto col = scf::ForOp::create(b, loc, zero, colEnd, step);
    b.setInsertionPointToStart(col.getBody());
    SmallVector<Value> indices =
        axis == 1
            ? SmallVector<Value>{row.getInductionVar(), col.getInductionVar()}
            : SmallVector<Value>{col.getInductionVar(), row.getInductionVar()};
    auto vt = VectorType::get({lanes}, element);
    // vector.load/store require the last dimension to be contiguous. For a
    // transposed view, transpose the memref descriptor and the indices too.
    auto canonical = [&](Value value) -> Value {
      if (axis == 1)
        return value;
      auto map =
          AffineMap::getPermutationMap(ArrayRef<unsigned>{1, 0}, &context);
      return memref::TransposeOp::create(b, loc, value,
                                         AffineMapAttr::get(map));
    };
    if (axis == 0)
      std::swap(indices[0], indices[1]);
    Value fill;
    if (!load) {
      auto constant = store.getValue().getDefiningOp<arith::ConstantOp>();
      fill = arith::ConstantOp::create(b, loc, constant.getValue());
    }
    Value value = load ? Value(vector::LoadOp::create(
                             b, loc, vt, canonical(load.getMemref()), indices))
                       : Value(vector::BroadcastOp::create(b, loc, vt, fill));
    vector::StoreOp::create(b, loc, value, canonical(store.getMemref()),
                            indices);
    outer.erase();
  }
}

// Converter-agnostic preparation the standard lowering expects to have run:
// linalg to loops (there is no linalg-to-LLVM lowering), strided-metadata
// expansion, memref expansion and vector-to-SCF.  None of them looks at
// BOSCAME operations.
LogicalResult runPreparationPipeline(ModuleOp module, MLIRContext &context) {
  // The generic math-to-libm pass turns scalar abs into fabsf calls.  Opaque
  // scalar loops (for example im2col's per-patch max reduction) do not pass
  // through the pointwise planner, so preserve their exact sign-bit operation
  // here as well.  This keeps NaN payloads, infinities and signed zero intact
  // while avoiding a call and its register spills for each input element.
  SmallVector<math::AbsFOp> scalarAbs;
  module.walk([&](math::AbsFOp op) {
    if (op.getType().isF32())
      scalarAbs.push_back(op);
  });
  IRRewriter rewriter(&context);
  for (math::AbsFOp op : scalarAbs) {
    rewriter.setInsertionPoint(op);
    auto bits = arith::BitcastOp::create(
        rewriter, op.getLoc(), rewriter.getI32Type(), op.getOperand());
    auto mask =
        arith::ConstantIntOp::create(rewriter, op.getLoc(), 0x7fffffff, 32);
    auto positive = arith::AndIOp::create(rewriter, op.getLoc(), bits, mask);
    auto result = arith::BitcastOp::create(rewriter, op.getLoc(),
                                           rewriter.getF32Type(), positive);
    rewriter.replaceOp(op, result.getResult());
  }
  PassManager pm(&context, PassManager::getAnyOpAnchorName(),
                 PassManager::Nesting::Implicit);
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(memref::createExpandOpsPass());
  pm.addPass(createConvertVectorToSCFPass());
  pm.addPass(createConvertMathToLibmPass());
  return pm.run(module);
}

// Rewrite vector operations into the forms the LLVM conversion understands.
// This mirrors the `-convert-vector-to-llvm` pipeline (default options) that
// the shipped Makefile runs after the BOSCAME lowering.
void prepareVectorOps(ModuleOp module, MLIRContext &context) {
  RewritePatternSet patterns(&context);
  vector::populateVectorToVectorCanonicalizationPatterns(patterns);
  vector::populateVectorBitCastLoweringPatterns(patterns);
  vector::populateVectorBroadcastLoweringPatterns(patterns);
  vector::populateVectorContractLoweringPatterns(
      patterns, vector::VectorContractLowering::Dot);
  vector::populateVectorMaskOpLoweringPatterns(patterns);
  vector::populateVectorShapeCastLoweringPatterns(patterns);
  vector::populateVectorInterleaveLoweringPatterns(patterns);
  vector::populateVectorTransposeLoweringPatterns(
      patterns, vector::VectorTransposeLowering::EltWise);
  vector::populateVectorTransferLoweringPatterns(patterns, 1);
  vector::populateVectorMaskMaterializationPatterns(patterns, false);
  vector::populateVectorInsertExtractStridedSliceTransforms(patterns);
  vector::populateVectorRankReducingFMAPattern(patterns);
  vector::populateVectorGatherLoweringPatterns(patterns);
  vector::populateVectorFromElementsUnrollPatterns(patterns);
  vector::populateVectorToElementsUnrollPatterns(patterns);
  (void)applyPatternsGreedily(module, std::move(patterns));
}

// The final conversion: BOSCAME / SPM / DMA / tile plus every standard dialect
// that still stands between the module and LLVM.
LogicalResult convertToLLVM(ModuleOp module, MLIRContext &context,
                            const TileLoweringOptions &tileOptions,
                            bool scalarF32OrderedReductions,
                            bool scalarF32MaxReductions) {
  DataLayoutAnalysis dataLayoutAnalysis(module);
  LowerToLLVMOptions options(&context, dataLayoutAnalysis.getAtOrAbove(module));
  LLVMTypeConverter converter(&context, options, &dataLayoutAnalysis);
  addBOSCAMETypeConversions(converter);

  LLVMConversionTarget target(context);
  target.addDynamicallyLegalDialect<BOSCAMEDialect>([](Operation *op) {
    return op->getName().getStringRef().starts_with("bosc_ame.intr.");
  });
  target.addIllegalDialect<
      arith::ArithDialect, affine::AffineDialect, cf::ControlFlowDialect,
      func::FuncDialect, index::IndexDialect, linalg::LinalgDialect,
      math::MathDialect, memref::MemRefDialect, scf::SCFDialect, ub::UBDialect,
      vector::VectorDialect>();

  SymbolTableCollection symbolTables;
  RewritePatternSet patterns(&context);
  patterns.add<BOSCAMEToIntrinsicLowering>(converter, &context);
  patterns.add<SpmViewOpConversion>(converter, &context);
  patterns.add<DDRFrameConversion, DDRReleaseConversion>(converter, &context);
  patterns.add<DmaStartOpConversion>(converter, &context);
  patterns.add<DmaWaitOpConversion>(converter, &context);
  if (scalarF32OrderedReductions || scalarF32MaxReductions)
    patterns.add<ScalarOrderedSnapshotLoadLowering,
                 ScalarOrderedSnapshotStoreLowering>(converter);
  // The tile patterns share this converter, this target and this pattern set;
  // they do not run a conversion of their own.
  populateBOSCAMETileLoweringPatterns(converter, patterns, tileOptions);

  // A loop that carries an AME matrix value changes the type of its iter
  // arguments.  The structural SCF conversion has to run inside this same
  // conversion - and before SCF is lowered to a CFG - so that the loop itself,
  // the value it carries and the tile operations that produce and consume that
  // value are all rewritten against one converter; a separate type-conversion
  // pass would have to bridge the two conversions with casts that could not be
  // reconciled afterwards.
  scf::populateSCFStructuralTypeConversions(converter, patterns,
                                            PatternBenefit(10));
  arith::populateArithToLLVMConversionPatterns(converter, patterns);
  index::populateIndexToLLVMConversionPatterns(converter, patterns);
  ub::populateUBToLLVMConversionPatterns(converter, patterns);
  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns, &symbolTables);
  populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns,
                                                 &symbolTables);
  populateVectorToLLVMConversionPatterns(converter, patterns);
  vector::populateVectorTransferLoweringPatterns(patterns);
  populateMathToLLVMConversionPatterns(converter, patterns);

  return applyPartialConversion(module, target, std::move(patterns));
}

// Remove the unrealized conversion casts the conversion may have left behind
// (the equivalent of the final `-reconcile-unrealized-casts` stage).
void reconcileCasts(ModuleOp module) {
  SmallVector<UnrealizedConversionCastOp> casts;
  module.walk([&](UnrealizedConversionCastOp cast) { casts.push_back(cast); });
  reconcileUnrealizedCasts(casts);
}

//===----------------------------------------------------------------------===//
// Pass body
//===----------------------------------------------------------------------===//

// Options of the shared lowering body.
struct BOSCAMELoweringOptions {
  // `-lower-bosc-ame-tile`: the caller states that this pipeline is the FPGA
  // tile path, so the module has to select an FPGA AME contract even when the
  // tile operations are not present yet (they may be introduced earlier in the
  // same pipeline).
  bool requireFpgaContract = false;
  // Opt-in FPGA workaround: ordered f32 reductions keep scalar accumulators.
  bool scalarF32OrderedReductions = false;
  // Opt-in FPGA workaround: f32 max reductions keep scalar accumulators.
  bool scalarF32MaxReductions = false;
};

// Policy transitions do not clobber RVV registers on the delivered NR target.
// Keeping them as ABI calls would spill live vectors using
// vlenb/whole-register instructions that this target does not implement. Run
// this only after the conservative call-based matrix/DMA check, while
// declaration provenance is still available; genuine runtime calls retain
// their ordinary ABI.
LogicalResult inlineNrCpuDdrPolicies(ModuleOp module) {
  auto target = module->getAttrOfType<StringAttr>("bosc_ame.target");
  if (!target || target.getValue() != "nr-fpga")
    return success();
  SmallVector<func::CallOp> calls;
  module.walk([&](func::CallOp call) {
    if (call->hasAttr(buddy::boscame::lowering::kCpuDdrPolicy))
      calls.push_back(call);
  });
  for (func::CallOp call : calls) {
    StringRef name = call.getCallee();
    bool write = name == "bosc_ame_ddr_write_begin";
    auto function = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        call, call.getCalleeAttr());
    if ((!write && name != "bosc_ame_ddr_read_begin") || !function ||
        !function.isExternal() || function.getNumArguments() ||
        function.getNumResults() || call.getNumOperands() ||
        call.getNumResults() ||
        !call->getAttrOfType<UnitAttr>(buddy::boscame::lowering::kCpuDdrPolicy))
      return call.emitOpError(
          "compiler DDR policy transition requires a reserved external "
          "() -> () helper and unit policy provenance");
    OpBuilder builder(call);
    // Same defaults as BOSC_AME_DMA_SHARED_READS/WRITES in the NR runtime.
    // These target intrinsics deliberately do not use an overridable ABI call.
    int64_t policy =
        write ? INT64_C(0x0f0f0f0f0f0f0f0f) : INT64_C(0x0f0f000f000f000f);
    Value value =
        LLVM::ConstantOp::create(builder, call.getLoc(), builder.getI64Type(),
                                 builder.getI64IntegerAttr(policy));
    LLVM::InlineAsmOp::create(
        builder, call.getLoc(), TypeRange{}, ValueRange{value},
        "fence iorw, iorw\n\tcsrw 0xbc0, $0", "r,~{memory}", true, false,
        LLVM::TailCallKind::None, false, LLVM::AsmDialectAttr(), ArrayAttr());
    call.erase();
  }
  return success();
}

// The single lowering body behind both `-lower-bosc-ame` and
// `-lower-bosc-ame-tile`; the two differ only in which contract they demand
// from the module, never in how they convert.
LogicalResult runBOSCAMELowering(ModuleOp module, MLIRContext &context,
                                 const BOSCAMELoweringOptions &options) {
  // Stage 1: contract checks, while the SPM annotation is still present.
  if (failed(verifySpmDmaContract(module)))
    return failure();
  if (failed(validateSpmSurface(module)))
    return failure();

  // The tile path is opt-in: it is only taken when the module both uses tile
  // operations and selects a hardware contract this build implements.
  TileLoweringOptions tileOptions;
  if (hasTileOps(module) || options.requireFpgaContract) {
    TileTargetProfile profile;
    if (failed(resolveTileTargetProfile(module, profile)))
      return failure();
    tileOptions.profile = profile;
    if (failed(verifyBOSCAMETileValues(module)))
      return failure();
  }
  declareDmaRuntimeHelpers(module);
  if (failed(declareFrameRuntimeHelpers(module)))
    return failure();

  // Stage 2: preparation.
  // Reject unsupported ordered forms before canonicalization can erase masks
  // or supply implicit accumulators. Recheck after preparation, where other
  // vector rewrites can introduce reductions that also need scalarization.
  if (options.scalarF32OrderedReductions &&
      failed(scalarizeF32OrderedReductions(module, false)))
    return failure();
  if (options.scalarF32MaxReductions &&
      failed(scalarizeF32MaxReductions(module, false)))
    return failure();
  // Preserve emitted view roots for the private/external alias proof.
  vectorizeGeneratedMemoryTasks(module, context);
  if (failed(runPreparationPipeline(module, context)))
    return failure();
  prepareVectorOps(module, context);
  if (options.scalarF32OrderedReductions &&
      failed(scalarizeF32OrderedReductions(module, true)))
    return failure();
  if (options.scalarF32MaxReductions &&
      failed(scalarizeF32MaxReductions(module, true)))
    return failure();
  if (failed(buddy::boscame::lowering::emitCpuDdrPolicies(module)))
    return failure();
  // Policy helpers remain calls for this check: an otherwise legal CPU memory
  // operation may lie inside an ACC lifetime. The NR rewrite below must not
  // weaken the existing matrix/DMA lifetime boundary.
  if (module->hasAttr(buddy::boscame::lowering::kCpuDdrPolicy) &&
      failed(verifySpmDmaContract(module)))
    return failure();
  if (failed(inlineNrCpuDdrPolicies(module)))
    return failure();

  // Stage 3: the loop structure follows automatically from the type conversion
  // of stage 4 (see `convertToLLVM`), which converts the matrix value type and
  // the operations that carry it - the loop included - in one conversion, so
  // that no fixed-vector bridge can survive between them.

  // Stage 4: the unified BOSCAME / SPM / DMA / standard conversion.
  if (failed(convertToLLVM(module, context, tileOptions,
                           options.scalarF32OrderedReductions,
                           options.scalarF32MaxReductions)))
    return failure();
  reconcileCasts(module);
  return success();
}

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

// `-lower-bosc-ame`: the unified BOSCAME / SPM / DMA / tile entry point.
struct LegalizeBOSCAMEForLLVMExport
    : public PassWrapper<LegalizeBOSCAMEForLLVMExport,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeBOSCAMEForLLVMExport)

  StringRef getArgument() const final { return "lower-bosc-ame"; }
  StringRef getDescription() const final {
    return "Lower the BOSCAME / SPM / DMA surface and the surrounding standard "
           "dialects to the LLVM dialect.";
  }

  LegalizeBOSCAMEForLLVMExport() = default;
  LegalizeBOSCAMEForLLVMExport(const LegalizeBOSCAMEForLLVMExport &) {}

  Option<bool> scalarF32OrderedReductions{
      *this, "scalar-f32-ordered-reductions",
      llvm::cl::desc("Lower fixed rank-one ordered f32 ADD reductions with an "
                     "accumulator to sequential scalar loads/adds"),
      llvm::cl::init(false)};

  Option<bool> scalarF32MaxReductions{
      *this, "scalar-f32-max-reductions",
      llvm::cl::desc("Lower fixed rank-one f32 maxnum/maximum/maximumnum "
                     "reductions to scalar loads and matching comparisons"),
      llvm::cl::init(false)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<BOSCAMEDialect, LLVM::LLVMDialect, affine::AffineDialect,
                arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect,
                index::IndexDialect, linalg::LinalgDialect, math::MathDialect,
                memref::MemRefDialect, scf::SCFDialect, ub::UBDialect,
                vector::VectorDialect>();
  }

  void runOnOperation() override {
    BOSCAMELoweringOptions options;
    options.scalarF32OrderedReductions = scalarF32OrderedReductions;
    options.scalarF32MaxReductions = scalarF32MaxReductions;
    if (failed(runBOSCAMELowering(getOperation(), getContext(), options)))
      signalPassFailure();
  }
};

// `-lower-bosc-ame-tile`: the FPGA tile path of the same lowering.
//
// It produces exactly the same boundary as `-lower-bosc-ame` (LLVM dialect,
// ready for `buddy-translate`) and shares its implementation: the separate
// entry point only exists so a pipeline can state that it is lowering the
// FPGA AME tile path and therefore require `bosc_ame.target` to name a
// contract this build implements, even before the tile operations appear.
struct LowerBOSCAMETileOps
    : public PassWrapper<LowerBOSCAMETileOps, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerBOSCAMETileOps)

  StringRef getArgument() const final { return "lower-bosc-ame-tile"; }
  StringRef getDescription() const final {
    return "Lower the BOSCAME tile operations (and the SPM/DMA and standard "
           "dialect surface they need) to the LLVM dialect on the FPGA target.";
  }

  LowerBOSCAMETileOps() = default;
  LowerBOSCAMETileOps(const LowerBOSCAMETileOps &) {}

  Option<bool> scalarF32OrderedReductions{
      *this, "scalar-f32-ordered-reductions",
      llvm::cl::desc("Lower fixed rank-one ordered f32 ADD reductions with an "
                     "accumulator to sequential scalar loads/adds"),
      llvm::cl::init(false)};

  Option<bool> scalarF32MaxReductions{
      *this, "scalar-f32-max-reductions",
      llvm::cl::desc("Lower fixed rank-one f32 maxnum/maximum/maximumnum "
                     "reductions to scalar loads and matching comparisons"),
      llvm::cl::init(false)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<BOSCAMEDialect, LLVM::LLVMDialect, affine::AffineDialect,
                arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect,
                index::IndexDialect, linalg::LinalgDialect, math::MathDialect,
                memref::MemRefDialect, scf::SCFDialect, ub::UBDialect,
                vector::VectorDialect>();
  }

  void runOnOperation() override {
    BOSCAMELoweringOptions options;
    options.requireFpgaContract = true;
    options.scalarF32OrderedReductions = scalarF32OrderedReductions;
    options.scalarF32MaxReductions = scalarF32MaxReductions;
    if (failed(runBOSCAMELowering(getOperation(), getContext(), options)))
      signalPassFailure();
  }
};

} // namespace

void mlir::populateBOSCAMELegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  addBOSCAMETypeConversions(converter);
  patterns.add<BOSCAMEToIntrinsicLowering>(converter, patterns.getContext());
  // NOTE: callers of this helper that lower the SPM/DMA surface or
  // loop-carried matrix values must add the SPM view / DMA patterns and
  // `scf::populateSCFStructuralTypeConversionsAndLegality` themselves; see the
  // unified pass above.
}

void mlir::configureBOSCAMELegalizeForExportTarget(
    LLVMConversionTarget &target) {
  target.addDynamicallyLegalDialect<BOSCAMEDialect>([](Operation *op) {
    return op->getName().getStringRef().starts_with("bosc_ame.intr.");
  });
}

std::unique_ptr<Pass> buddy::boscame::createLegalizeForLLVMExportPass() {
  return std::make_unique<LegalizeBOSCAMEForLLVMExport>();
}

std::unique_ptr<Pass> buddy::boscame::createLowerBOSCAMETilePass() {
  return std::make_unique<LowerBOSCAMETileOps>();
}

namespace mlir {
namespace buddy {
void registerLowerBOSCAMEPass() {
  PassRegistration<LegalizeBOSCAMEForLLVMExport>();
}

void registerLowerBOSCAMETilePass() { PassRegistration<LowerBOSCAMETileOps>(); }
} // namespace buddy
} // namespace mlir
