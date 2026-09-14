//===- LegalizeForLLVMExport.cpp - Prepare BOSCAME for LLVM translation -===//
//
// Lower the SSA-based BOSC AME dialect to bosc_ame.intr.* operations.
//
// This file deliberately uses LLVM::getVectorType instead of concrete
// LLVMVectorType / LLVMScalableVectorType classes. getVectorType is the
// stable LLVM-dialect API in the Buddy/MLIR branches where concrete class
// names differ.
//===----------------------------------------------------------------------===//

#include "Dialect/BOSCAME/BOSCAMEDialect.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/Transform.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>
#include <memory>
#include <string>

using namespace mlir;
using namespace buddy::boscame;

namespace {

// The source vector shape, for example 4x4, represents active tile shape.
// This type represents the storage capacity of one backend matrix register.
static Type getBOSCAMERegisterType(Type type) {
  VectorType matrixType = dyn_cast<VectorType>(type);
  if (!matrixType)
    return Type();

  Type elementType = matrixType.getElementType();
  unsigned lanes = 0;
  if (IntegerType integerType = dyn_cast<IntegerType>(elementType)) {
    switch (integerType.getWidth()) {
    case 8:
      lanes = 128;
      break;
    case 16:
      lanes = 64;
      break;
    case 32:
      lanes = 32;
      break;
    case 64:
      lanes = 16;
      break;
    default:
      return Type();
    }
  } else {
    FloatType floatType = dyn_cast<FloatType>(elementType);
    if (!floatType)
      return Type();
    switch (floatType.getWidth()) {
    case 16:
      lanes = 64;
      break;
    case 32:
      lanes = 32;
      break;
    case 64:
      lanes = 16;
      break;
    default:
      return Type();
    }
  }

  // i32 + 32 + scalable produces <vscale x 32 x i32> in LLVM IR, i.e.
  // RISCV MVT::nxv32i32, registered to TileReg/AccReg by your backend code.
  return LLVM::getVectorType(elementType, lanes, /*isScalable=*/true);
}

static void addBOSCAMETypeConversions(LLVMTypeConverter &converter) {
  // LLVMTypeConverter normally lowers a fixed 2-D vector to nested LLVM
  // arrays. BOSCAME matrix registers are represented by scalable vectors in
  // the backend, so keep those values in the register form throughout this
  // conversion instead of materializing unrealized array casts.
  //
  // Vectors that are *not* matrix tiles must convert to themselves.  Returning
  // a null type here would make the framework report "unable to convert type
  // for operand" as soon as a loop carries an already-converted register type
  // (the accumulator chain), which then fails to legalize the generated
  // `scf.for` even though its types are correct.
  converter.addConversion([](VectorType type) -> Type {
    if (type.getRank() != 2)
      return type;
    Type registerType = getBOSCAMERegisterType(type);
    return registerType ? registerType : type;
  });
}

static bool isMemRefValue(Value value) {
  return isa<MemRefType, UnrankedMemRefType>(value.getType());
}

// Materialize the effective base address of `memref` as an LLVM pointer.
//
// The aligned pointer alone describes the allocation base; it ignores the
// descriptor offset introduced by subviews, reinterpret_cast and strided
// layouts. Feeding that pointer to a matrix load/store silently addresses the
// wrong tile, so decode the strided metadata and fold the element offset in:
//
//   effective = alignedPointer(baseBuffer) + offset * bytesPerElement
//
// `memref.extract_strided_metadata` resolves arbitrarily nested
// subview/reinterpret_cast chains into a single descriptor, so the offset is
// applied exactly once even for multi-level views.
static Value extractPointerFromMemref(ConversionPatternRewriter &rewriter,
                                      Location loc, Value memref) {
  MLIRContext *context = rewriter.getContext();
  Type pointerType = LLVM::LLVMPointerType::get(context);
  Type i64Type = IntegerType::get(context, 64);

  auto metadata =
      memref::ExtractStridedMetadataOp::create(rewriter, loc, memref);
  Value baseBuffer = metadata.getBaseBuffer();
  Value offset = metadata.getOffset();
  Value pointerAsIndex =
      memref::ExtractAlignedPointerAsIndexOp::create(rewriter, loc, baseBuffer);

  Type elementType = getElementTypeOrSelf(memref.getType());
  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  // Sub-byte element types (e.g. i4) share a byte; an offset is only
  // expressible in whole elements, so keep the element granularity.
  unsigned bytesPerElement = std::max(1u, bitWidth / 8u);
  Value elementBytes =
      arith::ConstantIndexOp::create(rewriter, loc, bytesPerElement);
  Value byteOffset =
      arith::MulIOp::create(rewriter, loc, offset, elementBytes);
  Value effectiveAsIndex =
      arith::AddIOp::create(rewriter, loc, pointerAsIndex, byteOffset);
  Value pointerAsI64 =
      arith::IndexCastOp::create(rewriter, loc, i64Type, effectiveAsIndex);
  return LLVM::IntToPtrOp::create(rewriter, loc, pointerType, pointerAsI64);
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

static Operation *createBOSCAMEIntrinsicOp(ConversionPatternRewriter &rewriter,
                                           Location loc, StringRef sourceName,
                                           TypeRange resultTypes,
                                           ValueRange operands) {
  // bosc_ame.mma.w.mm becomes the TableGen-defined bosc_ame.intr.mma.w.mm.
  static const StringRef dialectPrefix = "bosc_ame.";
  StringRef suffix = sourceName.drop_front(dialectPrefix.size());
  std::string intrinsicName = "bosc_ame.intr.";
  intrinsicName += suffix;

  OperationState state(loc, intrinsicName);
  state.addOperands(operands);
  state.addTypes(resultTypes);
  return rewriter.create(state);
}

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

    Location loc = op->getLoc();

    // Matrix load: memref, i64 stride -> scalable LLVM vector.
    if (op->getNumResults() == 1 && op->getNumOperands() == 2 &&
        isMemRefValue(op->getOperand(0))) {
      Type resultType = getBOSCAMERegisterType(op->getResult(0).getType());
      if (!resultType)
        return failure();
      Value base = extractPointerFromMemref(rewriter, loc, op->getOperand(0));
      Operation *intrinsic =
          createBOSCAMEIntrinsicOp(rewriter, loc, name, TypeRange{resultType},
                                   ValueRange{base, operands[1]});
      rewriter.replaceOp(op, intrinsic->getResults());
      return success();
    }

    // Matrix store: scalable LLVM vector, memref, i64 stride -> no result.
    if (op->getNumResults() == 0 && op->getNumOperands() == 3 &&
        isMemRefValue(op->getOperand(1))) {
      Value base = extractPointerFromMemref(rewriter, loc, op->getOperand(1));
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

struct LegalizeBOSCAMEForLLVMExport
    : public PassWrapper<LegalizeBOSCAMEForLLVMExport,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeBOSCAMEForLLVMExport)

  StringRef getArgument() const final { return "lower-bosc-ame"; }
  StringRef getDescription() const final {
    return "BOSCAME dialect lowering pass.";
  }

  LegalizeBOSCAMEForLLVMExport() = default;
  LegalizeBOSCAMEForLLVMExport(const LegalizeBOSCAMEForLLVMExport &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<BOSCAMEDialect>();
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext &context = getContext();
    LLVMConversionTarget target(context);
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addDynamicallyLegalDialect<BOSCAMEDialect>([](Operation *op) {
      return op->getName().getStringRef().starts_with("bosc_ame.intr.");
    });

    LLVMTypeConverter converter(&context);
    addBOSCAMETypeConversions(converter);
    RewritePatternSet patterns(&context);
    patterns.add<BOSCAMEToIntrinsicLowering>(converter, &context);
    // A matrix value that is carried across an `scf.for` reduction (the FPGA
    // accumulator chain) has to be converted consistently with the value
    // produced by the intrinsic, otherwise the loop result keeps the fixed
    // `vector<4x4xT>` type and a permanent unrealized_conversion_cast bridges
    // it to the scalable register type.  `reconcile-unrealized-casts` cannot
    // remove that bridge once the loop is lowered to a CFG, and the LLVM
    // translation refuses it.
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::populateBOSCAMELegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  addBOSCAMETypeConversions(converter);
  patterns.add<BOSCAMEToIntrinsicLowering>(converter, patterns.getContext());
  // NOTE: callers of this helper that also lower loop-carried matrix values
  // must add `scf::populateSCFStructuralTypeConversionsAndLegality` themselves;
  // see the pass below.
}

void mlir::configureBOSCAMELegalizeForExportTarget(
    LLVMConversionTarget &target) {
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<memref::MemRefDialect>();
  target.addDynamicallyLegalDialect<BOSCAMEDialect>([](Operation *op) {
    return op->getName().getStringRef().starts_with("bosc_ame.intr.");
  });
}

std::unique_ptr<Pass> buddy::boscame::createLegalizeForLLVMExportPass() {
  return std::make_unique<LegalizeBOSCAMEForLLVMExport>();
}

namespace mlir {
namespace buddy {
void registerLowerBOSCAMEPass() {
  PassRegistration<LegalizeBOSCAMEForLLVMExport>();
}
} // namespace buddy
} // namespace mlir
