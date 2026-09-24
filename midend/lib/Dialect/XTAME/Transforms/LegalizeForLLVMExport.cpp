//===- LegalizeForLLVMExport.cpp - Prepare XTAME for LLVM translation ----===//
//
// Lower SSA XTAME operations to their TableGen-defined xt_ame.intr.* bridge
// operations. LLVM dialect translation then emits LLVM intrinsics rather than
// inline assembly text.
//
//===----------------------------------------------------------------------===//

#include "Dialect/XTAME/Transform.h"
#include "Dialect/XTAME/XTAMEDialect.h"
#include "Dialect/XTAME/XTAMEOps.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <memory>
#include <string>

using namespace mlir;
using namespace buddy::xtame;

namespace {

// The source vector shape represents the active matrix tile. This scalable
// vector uses the smallest MatrixReg MVT supported by the XTAME backend
// (RLEN=128, M=4), so instruction selection can assign it to MatrixReg.
static Type getXTAMERegisterType(Type type) {
  VectorType matrixType = dyn_cast<VectorType>(type);
  if (!matrixType)
    return Type();

  // A one-dimensional scalable vector is already the flattened MatrixReg
  // representation. This form is useful when a matrix value is carried by
  // an SCF loop, whose LLVM conversion cannot preserve rank-2 vectors.
  if (matrixType.getRank() == 1) {
    if (!matrixType.isScalable())
      return Type();
    return matrixType;
  }

  if (matrixType.getRank() != 2)
    return Type();

  Type elementType = matrixType.getElementType();
  unsigned lanes = 0;
  if (IntegerType integerType = dyn_cast<IntegerType>(elementType)) {
    switch (integerType.getWidth()) {
    case 8:
      lanes = 64;
      break;
    case 16:
      lanes = 32;
      break;
    case 32:
      lanes = 16;
      break;
    case 64:
      lanes = 8;
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
      lanes = 32;
      break;
    case 32:
      lanes = 16;
      break;
    case 64:
      lanes = 8;
      break;
    default:
      return Type();
    }
  }

  return LLVM::getVectorType(elementType, lanes, /*isScalable=*/true);
}

static void addXTAMETypeConversions(LLVMTypeConverter &converter) {
  converter.addConversion(
      [](VectorType type) -> Type { return getXTAMERegisterType(type); });
}

static bool isMemRefValue(Value value) {
  return isa<MemRefType, UnrankedMemRefType>(value.getType());
}

static Value extractPointerFromMemref(ConversionPatternRewriter &rewriter,
                                      Location loc, Value memref) {
  MLIRContext *context = rewriter.getContext();
  Type pointerType = LLVM::LLVMPointerType::get(context);
  Type i64Type = IntegerType::get(context, 64);
  Value pointerAsIndex =
      memref::ExtractAlignedPointerAsIndexOp::create(rewriter, loc, memref);
  Value pointerAsI64 =
      arith::IndexCastOp::create(rewriter, loc, i64Type, pointerAsIndex);
  return LLVM::IntToPtrOp::create(rewriter, loc, pointerType, pointerAsI64);
}

// Immediate configuration and broadcast operands stay attributes in the user
// dialect and become SSA i64 constants only on the intrinsic bridge op.
static Value materializeXTAMEImmediate(Operation *op,
                                       ConversionPatternRewriter &rewriter) {
  static const char *const attributeNames[] = {"tilem", "tilen", "tilek",
                                               "uimm3"};
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

static Operation *createXTAMEIntrinsicOp(ConversionPatternRewriter &rewriter,
                                         Location loc, StringRef sourceName,
                                         TypeRange resultTypes,
                                         ValueRange operands) {
  static const StringRef dialectPrefix = "xt_ame.";
  StringRef suffix = sourceName.drop_front(dialectPrefix.size());
  std::string intrinsicName = "xt_ame.intr.";
  intrinsicName += suffix;

  OperationState state(loc, intrinsicName);
  state.addOperands(operands);
  state.addTypes(resultTypes);
  return rewriter.create(state);
}

class XTAMEToIntrinsicLowering : public ConversionPattern {
public:
  XTAMEToIntrinsicLowering(const TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, Pattern::MatchAnyOpTypeTag(),
                          PatternBenefit(1), context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef name = op->getName().getStringRef();
    if (!name.starts_with("xt_ame.") || name.starts_with("xt_ame.intr."))
      return failure();

    Location loc = op->getLoc();

    // Loads have source operands (stride, memref), while the LLVM intrinsic
    // takes (stride, pointer) and returns a MatrixReg value.
    if (op->getNumResults() == 1 && op->getNumOperands() == 2 &&
        isMemRefValue(op->getOperand(1))) {
      Type resultType = getXTAMERegisterType(op->getResult(0).getType());
      if (!resultType)
        return failure();
      Value base = extractPointerFromMemref(rewriter, loc, op->getOperand(1));
      Operation *intrinsic =
          createXTAMEIntrinsicOp(rewriter, loc, name, TypeRange{resultType},
                                 ValueRange{operands[0], base});
      rewriter.replaceOp(op, intrinsic->getResults());
      return success();
    }

    // Prefetches take the same (stride, pointer) operands as loads but do not
    // define a matrix value.
    if (op->getNumResults() == 0 && op->getNumOperands() == 2 &&
        isMemRefValue(op->getOperand(1))) {
      Value base = extractPointerFromMemref(rewriter, loc, op->getOperand(1));
      createXTAMEIntrinsicOp(rewriter, loc, name, TypeRange(),
                             ValueRange{operands[0], base});
      rewriter.eraseOp(op);
      return success();
    }

    // Stores map (matrix, stride, memref) to (matrix, stride, pointer).
    if (op->getNumResults() == 0 && op->getNumOperands() == 3 &&
        isMemRefValue(op->getOperand(2))) {
      Value base = extractPointerFromMemref(rewriter, loc, op->getOperand(2));
      createXTAMEIntrinsicOp(rewriter, loc, name, TypeRange(),
                             ValueRange{operands[0], operands[1], base});
      rewriter.eraseOp(op);
      return success();
    }

    SmallVector<Value, 4> intrinsicOperands(operands.begin(), operands.end());
    if (Value immediate = materializeXTAMEImmediate(op, rewriter))
      intrinsicOperands.push_back(immediate);

    SmallVector<Type, 1> resultTypes;
    for (Value result : op->getResults()) {
      Type resultType = result.getType();
      if (isa<VectorType>(resultType))
        resultType = getXTAMERegisterType(resultType);
      if (!resultType)
        return failure();
      resultTypes.push_back(resultType);
    }

    Operation *intrinsic = createXTAMEIntrinsicOp(
        rewriter, loc, name, TypeRange(resultTypes), intrinsicOperands);
    rewriter.replaceOp(op, intrinsic->getResults());
    return success();
  }
};

struct LegalizeXTAMEForLLVMExport
    : public PassWrapper<LegalizeXTAMEForLLVMExport, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeXTAMEForLLVMExport)

  StringRef getArgument() const final { return "lower-xt-ame"; }
  StringRef getDescription() const final {
    return "XTAME dialect lowering pass.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<XTAMEDialect>();
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext &context = getContext();
    LLVMConversionTarget target(context);
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addDynamicallyLegalDialect<XTAMEDialect>([](Operation *op) {
      return op->getName().getStringRef().starts_with("xt_ame.intr.");
    });

    LLVMTypeConverter converter(&context);
    addXTAMETypeConversions(converter);
    RewritePatternSet patterns(&context);
    patterns.add<XTAMEToIntrinsicLowering>(converter, &context);
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::populateXTAMELegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  addXTAMETypeConversions(converter);
  patterns.add<XTAMEToIntrinsicLowering>(converter, patterns.getContext());
}

void mlir::configureXTAMELegalizeForExportTarget(LLVMConversionTarget &target) {
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<memref::MemRefDialect>();
  target.addDynamicallyLegalDialect<buddy::xtame::XTAMEDialect>(
      [](Operation *op) {
        return op->getName().getStringRef().starts_with("xt_ame.intr.");
      });
}

std::unique_ptr<Pass> buddy::xtame::createLegalizeForLLVMExportPass() {
  return std::make_unique<LegalizeXTAMEForLLVMExport>();
}

namespace mlir {
namespace buddy {
void registerLowerXTAMEPass() {
  PassRegistration<LegalizeXTAMEForLLVMExport>();
}
} // namespace buddy
} // namespace mlir
