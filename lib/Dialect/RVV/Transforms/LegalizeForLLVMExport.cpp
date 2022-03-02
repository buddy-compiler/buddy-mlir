//===- LegalizeForLLVMExport.cpp - Prepare RVV for LLVM translation -------===//
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

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

#include "RVV/RVVDialect.h"
#include "RVV/Transforms.h"

using namespace mlir;
using namespace buddy::rvv;

// Extract an LLVM IR type from the LLVM IR dialect type.
static Type unwrap(Type type) {
  if (!type)
    return nullptr;
  auto *mlirContext = type.getContext();
  if (!LLVM::isCompatibleType(type))
    emitError(UnknownLoc::get(mlirContext),
              "conversion resulted in a non-LLVM type");
  return type;
}

// Scalable vector type in RVV dialect uses LMUL and SEW as parameters to
// provide better semantics. This is the helper function to bridge the gap
// of scalable vector type between the RVV dialect and LLVM dialect.
unsigned typeMapping(ScalableVectorType rvvSVType) {
  auto elementType = rvvSVType.getElementType();
  auto *elementContext = elementType.getContext();
  auto sizeType = rvvSVType.getSizeType();
  auto *sizeContext = sizeType.getContext();
  // TODO: support more element type.
  if (elementType.isa<IntegerType>()) {
    // Mapping LMUL and Mask type for different SEW type.
    switch (elementType.cast<IntegerType>().getWidth()) {
    case 64:
      if (sizeType.isa<MF8Type>() || sizeType.isa<MF4Type>() ||
          sizeType.isa<MF2Type>()) {
        emitError(UnknownLoc::get(sizeContext), "unsupported LMUL Type for ")
            << elementType << " type.";
      }
      return llvm::TypeSwitch<Type, unsigned>(sizeType)
          .Case<M1Type>([&](Type) { return 1; })
          .Case<M2Type>([&](Type) { return 2; })
          .Case<M4Type>([&](Type) { return 4; })
          .Case<M8Type>([&](Type) { return 8; })
          .Default([](Type) -> unsigned {
            llvm_unreachable("incompatible with RISC-V vector type");
          });
      break;
    case 32:
      if (sizeType.isa<MF8Type>() || sizeType.isa<MF4Type>()) {
        emitError(UnknownLoc::get(sizeContext), "unsupported LMUL Type for ")
            << elementType << " type.";
      }
      return llvm::TypeSwitch<Type, unsigned>(sizeType)
          .Case<MF2Type>([&](Type) { return 1; })
          .Case<M1Type>([&](Type) { return 2; })
          .Case<M2Type>([&](Type) { return 4; })
          .Case<M4Type>([&](Type) { return 8; })
          .Case<M8Type>([&](Type) { return 16; })
          .Default([](Type) -> unsigned {
            llvm_unreachable("incompatible with RISC-V vector type");
          });
      break;
    case 16:
      if (sizeType.isa<MF8Type>()) {
        emitError(UnknownLoc::get(sizeContext), "unsupported LMUL type for ")
            << elementType << " type.";
      }
      return llvm::TypeSwitch<Type, unsigned>(sizeType)
          .Case<MF4Type>([&](Type) { return 1; })
          .Case<MF2Type>([&](Type) { return 2; })
          .Case<M1Type>([&](Type) { return 4; })
          .Case<M2Type>([&](Type) { return 8; })
          .Case<M4Type>([&](Type) { return 16; })
          .Case<M8Type>([&](Type) { return 32; })
          .Default([](Type) -> unsigned {
            llvm_unreachable("incompatible with RISC-V vector type");
          });
      break;
    case 8:
      return llvm::TypeSwitch<Type, unsigned>(sizeType)
          .Case<MF8Type>([&](Type) { return 1; })
          .Case<MF4Type>([&](Type) { return 2; })
          .Case<MF2Type>([&](Type) { return 4; })
          .Case<M1Type>([&](Type) { return 8; })
          .Case<M2Type>([&](Type) { return 16; })
          .Case<M4Type>([&](Type) { return 32; })
          .Case<M8Type>([&](Type) { return 64; })
          .Default([](Type) -> unsigned {
            llvm_unreachable("incompatible with RISC-V vector type");
          });
      break;
    case 1:
      return llvm::TypeSwitch<Type, unsigned>(sizeType)
          .Case<Mask64Type>([&](Type) { return 1; })
          .Case<Mask32Type>([&](Type) { return 2; })
          .Case<Mask16Type>([&](Type) { return 4; })
          .Case<Mask8Type>([&](Type) { return 8; })
          .Case<Mask4Type>([&](Type) { return 16; })
          .Case<Mask2Type>([&](Type) { return 32; })
          .Case<Mask1Type>([&](Type) { return 64; })
          .Default([](Type) -> unsigned {
            llvm_unreachable("incompatible with RISC-V vector type");
          });
      break;
    default:
      emitError(UnknownLoc::get(elementContext), "unsupported ")
          << elementType << " SEW type.";
    }
  } else {
    emitError(UnknownLoc::get(elementContext), "unsupported ")
        << elementType << " SEW type.";
  }
  return 0;
}

static Optional<Type>
convertScalableVectorTypeToLLVM(ScalableVectorType svType,
                                LLVMTypeConverter &converter) {
  auto elementType = unwrap(converter.convertType(svType.getElementType()));
  if (!elementType)
    return {};
  auto sVectorType =
      LLVM::LLVMScalableVectorType::get(elementType, typeMapping(svType));
  return sVectorType;
}

template <typename OpTy>
class ForwardOperands : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getOperands().getTypes() == op->getOperands().getTypes())
      return rewriter.notifyMatchFailure(op, "operand types already match");

    rewriter.updateRootInPlace(
        op, [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

template <typename SourceOp, typename TargetOp>
class ConvertMaskedOpToLLVMPattern : public ConvertOpToLLVMPattern<SourceOp> {
public:
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;

  // TODO: Add vta attribute to make this pattern correct.

  /// The masked operations have a `vta` attribute. This pattern converts the
  /// `vta` attribute to a value, append the `vta` value to the operand list,
  /// and create the intrinsic operation.
  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    unsigned numResults = op->getNumResults();
    Type packedType;
    ValueRange operands = adaptor.getOperands();
    SmallVector<Value, 6> operandsVector(operands);
    operandsVector.pop_back();
    // Get the type of the `vl` value.
    Type vlType = operands.back().getType();
    auto attrs = op->getAttrs();
    if (attrs.empty()) {
      // Default attribute for the vta setting (vta = 1).
      // Add the vta = 1 to the operand list.
      Attribute vtaDefaultAttr = rewriter.getIntegerAttr(
          vlType, APInt(vlType.cast<IntegerType>().getWidth(), 1));
      Value vtaDefaultValue =
          rewriter.create<LLVM::ConstantOp>(loc, vlType, vtaDefaultAttr);
      operandsVector.push_back(vtaDefaultValue);
    } else if (attrs.size() == 1) {
      // Add the vta to the operand list according to the attribute value.
      Attribute attr = attrs[0].getValue();
      IntegerAttr vtaAttr = attr.cast<IntegerAttr>();
      Value vtaValue = rewriter.create<LLVM::ConstantOp>(loc, vlType, vtaAttr);
      operandsVector.push_back(vtaValue);
    } else {
      return failure();
    }

    LLVMTypeConverter typeConverter = *this->getTypeConverter();
    if (numResults != 0) {
      packedType = typeConverter.packFunctionResults(op->getResultTypes());
      if (!packedType)
        return failure();
    }
    // Create the intrinsic operation.
    OperationState state(op->getLoc(), TargetOp::getOperationName());
    state.addTypes(packedType);
    state.addOperands(operandsVector);
    Operation *newOp = rewriter.createOperation(state);
    return rewriter.replaceOp(op, newOp->getResult(0)), success();
  }
};

struct RVVLoadOpLowering : public ConvertOpToLLVMPattern<RVVLoadOp> {
  using ConvertOpToLLVMPattern<RVVLoadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RVVLoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = loadOp.getMemRefType();
    if (!isConvertibleAndHasIdentityMaps(type))
      return failure();

    LLVMTypeConverter converter(loadOp.getContext());

    auto resultType = loadOp.result().getType();
    LLVM::LLVMPointerType llvmDataTypePtr;
    if (resultType.isa<VectorType>()) {
      llvmDataTypePtr =
          LLVM::LLVMPointerType::get(resultType.cast<VectorType>());
    } else if (resultType.isa<ScalableVectorType>()) {
      llvmDataTypePtr = LLVM::LLVMPointerType::get(
          convertScalableVectorTypeToLLVM(resultType.cast<ScalableVectorType>(),
                                          converter)
              .getValue());
    }
    Value dataPtr = getStridedElementPtr(loadOp.getLoc(), type, adaptor.base(),
                                         adaptor.index(), rewriter);
    Value bitCastedPtr = rewriter.create<LLVM::BitcastOp>(
        loadOp.getLoc(), llvmDataTypePtr, dataPtr);
    Value vl = loadOp.getOperand(2);
    rewriter.replaceOpWithNewOp<RVVIntrLoadEleOp>(
        loadOp,
        convertScalableVectorTypeToLLVM(resultType.cast<ScalableVectorType>(),
                                        converter)
            .getValue(),
        bitCastedPtr, vl);
    return success();
  }
};

struct RVVStoreOpLowering : public ConvertOpToLLVMPattern<RVVStoreOp> {
  using ConvertOpToLLVMPattern<RVVStoreOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RVVStoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = storeOp.getMemRefType();
    if (!isConvertibleAndHasIdentityMaps(type))
      return failure();

    LLVMTypeConverter converter(storeOp.getContext());

    auto resultType = storeOp.value().getType();
    LLVM::LLVMPointerType llvmDataTypePtr;
    if (resultType.isa<VectorType>()) {
      llvmDataTypePtr =
          LLVM::LLVMPointerType::get(resultType.cast<VectorType>());
    } else if (resultType.isa<ScalableVectorType>()) {
      llvmDataTypePtr = LLVM::LLVMPointerType::get(
          convertScalableVectorTypeToLLVM(resultType.cast<ScalableVectorType>(),
                                          converter)
              .getValue());
    }
    Value dataPtr = getStridedElementPtr(storeOp.getLoc(), type, adaptor.base(),
                                         adaptor.index(), rewriter);
    Value bitCastedPtr = rewriter.create<LLVM::BitcastOp>(
        storeOp.getLoc(), llvmDataTypePtr, dataPtr);
    Value vl = storeOp.getOperand(3);
    rewriter.replaceOpWithNewOp<RVVIntrStoreEleOp>(storeOp, adaptor.value(),
                                                   bitCastedPtr, vl);
    return success();
  }
};

using RVVAddOpLowering = OneToOneConvertToLLVMPattern<RVVAddOp, RVVIntrAddOp>;
using RVVSubOpLowering = OneToOneConvertToLLVMPattern<RVVSubOp, RVVIntrSubOp>;
using RVVMulOpLowering = OneToOneConvertToLLVMPattern<RVVMulOp, RVVIntrMulOp>;
using RVVDivOpLowering = OneToOneConvertToLLVMPattern<RVVDivOp, RVVIntrDivOp>;
using RVVMaskedAddOpLowering =
    ConvertMaskedOpToLLVMPattern<RVVMaskedAddOp, RVVMaskedIntrAddOp>;
using RVVMaskedSubOpLowering =
    ConvertMaskedOpToLLVMPattern<RVVMaskedSubOp, RVVMaskedIntrSubOp>;
using RVVMaskedMulOpLowering =
    ConvertMaskedOpToLLVMPattern<RVVMaskedMulOp, RVVMaskedIntrMulOp>;
using RVVMaskedDivOpLowering =
    ConvertMaskedOpToLLVMPattern<RVVMaskedDivOp, RVVMaskedIntrDivOp>;

/// Populate the given list with patterns that convert from RVV to LLVM.
void mlir::populateRVVLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  // Populate conversion patterns.
  // Remove any RVV-specific types from function signatures and results.
  populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns, converter);
  converter.addConversion([&converter](ScalableVectorType rvvSVType) {
    return convertScalableVectorTypeToLLVM(rvvSVType, converter);
  });

  // clang-format off
  patterns.add<ForwardOperands<func::CallOp>,
               ForwardOperands<func::CallIndirectOp>,
               ForwardOperands<func::ReturnOp>>(converter, &converter.getContext());
  patterns.add<RVVLoadOpLowering,
               RVVStoreOpLowering>(converter);
  patterns.add<RVVAddOpLowering,
               RVVSubOpLowering,
               RVVMulOpLowering,
               RVVDivOpLowering,
               RVVSubOpLowering,
               RVVMaskedAddOpLowering,
               RVVMaskedSubOpLowering,
               RVVMaskedMulOpLowering,
               RVVMaskedDivOpLowering>(converter);
  // clang-format on
}

void mlir::configureRVVLegalizeForExportTarget(LLVMConversionTarget &target) {
  // clang-format off
  target.addLegalOp<RVVIntrLoadEleOp,
                    RVVIntrStoreEleOp,
                    RVVIntrAddOp,
                    RVVIntrSubOp,
                    RVVIntrMulOp,
                    RVVIntrDivOp,
                    RVVMaskedIntrAddOp,
                    RVVMaskedIntrSubOp,
                    RVVMaskedIntrMulOp,
                    RVVMaskedIntrDivOp>();
  target.addIllegalOp<RVVLoadOp,
                      RVVStoreOp,
                      RVVAddOp,
                      RVVSubOp,
                      RVVMulOp,
                      RVVDivOp,
                      RVVMaskedAddOp,
                      RVVMaskedSubOp,
                      RVVMaskedMulOp,
                      RVVMaskedDivOp>();
  // clang-format on

  auto hasScalableVectorType = [](TypeRange types) {
    for (Type type : types)
      if (type.isa<ScalableVectorType>())
        return true;
    return false;
  };
  target.addDynamicallyLegalOp<FuncOp>([hasScalableVectorType](FuncOp op) {
    return !hasScalableVectorType(op.getType().getInputs()) &&
           !hasScalableVectorType(op.getType().getResults());
  });
  target.addDynamicallyLegalOp<func::CallOp, func::CallIndirectOp, func::ReturnOp>(
      [hasScalableVectorType](Operation *op) {
        return !hasScalableVectorType(op->getOperandTypes()) &&
               !hasScalableVectorType(op->getResultTypes());
      });
}
