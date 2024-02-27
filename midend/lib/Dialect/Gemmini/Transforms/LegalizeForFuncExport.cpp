//====------------ LegalizeForFuncExport.cpp - Gemmini To Func ------------===//
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
// This file defines Gemmini dialect lowering to 3rd party Library.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Gemmini/GemminiDialect.h"
#include "Gemmini/GemminiOps.h"
#include "Gemmini/Transform.h"

using namespace mlir;
using namespace buddy::gemmini;

namespace {

int64_t getNumberFromValue(Value &value) {
  return dyn_cast<IntegerAttr>(value.getDefiningOp()->getAttr("value")).getInt();
}

acc_scale_t_bits acc_scale_t_to_acc_scale_t_bits(acc_scale_t x) {
  union {
    acc_scale_t_bits b;
    acc_scale_t f;
  } un;

  un.f = x;
  return un.b;
}

scale_t_bits scale_t_to_scale_t_bits(scale_t x) {
  union {
    scale_t_bits b;
    scale_t f;
  } un;

  un.f = x;
  return un.b;
}

template <typename IntrOp = Mvin_IntrOp>
void gemminiMvinOffset(const Value &mem, const size_t offset,
                       const uint32_t SpAddr, const size_t cols,
                       const size_t rows, int64_t addrLen,
                       ConversionPatternRewriter &rewriter) {
  Location loc = mem.getLoc();
  Value offsetOp = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(offset));
  IntegerType i64Type = rewriter.getI64Type();
  Value configPtr = rewriter.create<arith::AddIOp>(loc, i64Type, mem, offsetOp);
  uint64_t spadAddrInt = (uint64_t)rows << (addrLen + 16) |
                         (uint64_t)cols << addrLen | (uint64_t)SpAddr;
  Value spad = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(spadAddrInt));
  rewriter.create<IntrOp>(loc, configPtr, spad);
}

void gemminiMvoutOffset(const Value &mem, const size_t offset,
                        const uint32_t SpAddr, const size_t cols,
                        const size_t rows, int64_t addrLen,
                        ConversionPatternRewriter &rewriter) {
  Location loc = mem.getLoc();
  Value offsetOp = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(offset));
  IntegerType i64Type = rewriter.getI64Type();
  Value configPtr = rewriter.create<arith::AddIOp>(loc, i64Type, mem, offsetOp);
  uint64_t spadAddrInt = (uint64_t)rows << (addrLen + 16) |
                         (uint64_t)cols << addrLen | (uint64_t)SpAddr;
  Value spad = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(spadAddrInt));
  rewriter.create<Mvout_IntrOp>(loc, configPtr, spad);
}

static MemRefType makeStridedLayoutDynamic(MemRefType type) {
  // Dynamic layout for the type
  auto dynamicLayout = StridedLayoutAttr::get(
      type.getContext(), ShapedType::kDynamic,
      SmallVector<int64_t>(type.getRank(), ShapedType::kDynamic));

  // Dynamic shape for the type
  auto dynamicShape = MemRefType::get(
    SmallVector<int64_t>(type.getRank(), ShapedType::kDynamic),
    type.getElementType(), dynamicLayout);

  return MemRefType::Builder(dynamicShape);
}

/// Helper function to extract the operand types that are passed to the
/// generated CallOp. MemRefTypes have their layout canonicalized since the
/// information is not used in signature generation.
/// Note that static size information is not modified.
static SmallVector<Type, 4> extractOperandTypes(Operation *op) {
  SmallVector<Type, 4> result;
  result.reserve(op->getNumOperands());
  for (auto type : op->getOperandTypes()) {
    // The underlying descriptor type (e.g. LLVM) does not have layout
    // information. Canonicalizing the type at the level of std when going into
    // a library call avoids needing to introduce DialectCastOp.
    if (auto memrefType = type.dyn_cast<MemRefType>())
      result.push_back(makeStridedLayoutDynamic(memrefType));
    else
      result.push_back(type);
  }
  return result;
}

// Get a SymbolRefAttr containing the library function name for the Gemmini oeparation.
// If the library function does not exist, insert a declaration.
// Here is an example: llvm/mlir/lib/Conversion/LinalgToStandard/LinalgToStandard.cpp
static FailureOr<FlatSymbolRefAttr> 
getLibraryCallSymbolRef(Operation *op, std::string fnName, PatternRewriter &rewriter) {
  if (fnName.empty()) {
    return rewriter.notifyMatchFailure(op, "No library call defined for: ");
  }

  // fnName is a dynamic std::String, unique it via a SymbolRefAttr.
  FlatSymbolRefAttr fnNameAttr =
      SymbolRefAttr::get(rewriter.getContext(), fnName);
  auto module = op->getParentOfType<ModuleOp>();
  if (module.lookupSymbol(fnName)) {
    return fnNameAttr;
  }

  SmallVector<Type, 4> inputTypes({IntegerType::get(rewriter.getContext(), 64), 
                                  IntegerType::get(rewriter.getContext(), 64)});
  if (op->getNumResults() != 0) {
    return rewriter.notifyMatchFailure(
        op,
        "Library call can be generated only for ops that "
        "have void return types");
  }
  auto libFnType = rewriter.getFunctionType(inputTypes, op->getResultTypes());

  OpBuilder::InsertionGuard guard(rewriter);
  // Insert before module terminator.
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));
  func::FuncOp funcOp = rewriter.create<func::FuncOp>(op->getLoc(), fnNameAttr.getValue(), libFnType);
  // Insert a function attribute that will trigger the emission of the
  // corresponding `_mlir_ciface_xxx` interface so that external libraries see
  // a normalized ABI. This interface is added during std to llvm conversion.
  funcOp->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                  UnitAttr::get(op->getContext()));
  funcOp.setPrivate();

//   module.dump();
  return fnNameAttr;
}

} // namespace

struct GemminiConfigStLowering : public OpRewritePattern<ConfigStOp> {
  using OpRewritePattern<ConfigStOp>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(ConfigStOp configStOp, PatternRewriter &rewriter) const override {
    Value strideValue = configStOp.getStride();
    int stride = getNumberFromValue(strideValue);
    float scale = configStOp.getScale().convertToFloat();
    Location loc = configStOp.getLoc();
    uint64_t rs1 = ((uint64_t)configStOp.getActivation() << 2) | CONFIG_ST;
    uint64_t arg = (uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)scale)
                       << 32 |
                   (uint32_t)stride;
    Value value1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value value2 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(arg));

    auto libraryCallName = getLibraryCallSymbolRef(configStOp, "config_st", rewriter);
    if (failed(libraryCallName))
      return failure();

    rewriter.replaceOpWithNewOp<func::CallOp>(
      configStOp, libraryCallName->getValue(), TypeRange(), 
      ValueRange({value1, value2}));
    return success();
  }
};

struct GemminiConfigLdLowering : public OpRewritePattern<ConfigLdOp> {
  using OpRewritePattern<ConfigLdOp>::OpRewritePattern;
  explicit GemminiConfigLdLowering(MLIRContext *context, int64_t dim)
      : OpRewritePattern(context), dim(dim) {}
  LogicalResult
  matchAndRewrite(ConfigLdOp configLdOp, PatternRewriter &rewriter) const override {
    Value rs2Value = configLdOp.getStride();
    float scale = configLdOp.getScale().convertToFloat();
    uint64_t blockMvinStride = configLdOp.getBlockMvinStride();
    if (blockMvinStride == (uint64_t)-1)
      blockMvinStride = dim;
    uint64_t pixelRepeats = configLdOp.getPixelRepeats();
    uint64_t rs1 = (uint64_t)scale_t_to_scale_t_bits(scale) << 32 |
                   (blockMvinStride << 16) | pixelRepeats << 8 |
                   configLdOp.getId() << 3 | configLdOp.getShrunk() << 2 |
                   CONFIG_LD;
    Location loc = configLdOp.getLoc();
    Value rs1value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));

    auto libraryCallName = getLibraryCallSymbolRef(configLdOp, "config_ld", rewriter);
    if (failed(libraryCallName))
      return failure();

    rewriter.replaceOpWithNewOp<func::CallOp>(
      configLdOp, libraryCallName->getValue(), TypeRange(), 
      ValueRange({rs1value, rs2Value}));
    return success();
  }

private:
  int64_t dim;
};

struct GemminiMvinLowering : public OpRewritePattern<MvinOp> {
  using OpRewritePattern<MvinOp>::OpRewritePattern;
  explicit GemminiMvinLowering(MLIRContext *context, int64_t addrLen)
      : OpRewritePattern(context), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(MvinOp mvinOp, PatternRewriter &rewriter) const override {
    Value input = mvinOp.getInput();
    Location loc = input.getLoc();
    MemRefType memRefType = dyn_cast<MemRefType>(mvinOp.getOperandTypes().front());
    llvm::ArrayRef<int64_t> memRefShape = memRefType.getShape();
    TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
    Value extractOp = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, resultType, input);
    IntegerType i64Type = rewriter.getI64Type();
    Value indexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOp);
    Value spadAddrValue = mvinOp.getAddr();
    uint64_t number = getNumberFromValue(spadAddrValue);
    uint64_t spadAddrInt = (uint64_t)memRefShape[0] << (addrLen + 16) |
                           (uint64_t)memRefShape[1] << addrLen | number;
    Value spad = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(spadAddrInt));

    auto libraryCallName = getLibraryCallSymbolRef(mvinOp, "mvin", rewriter);
    if (failed(libraryCallName))
      return failure();

    rewriter.replaceOpWithNewOp<func::CallOp>(
      mvinOp, libraryCallName->getValue(), TypeRange(), 
      ValueRange({indexCastOp, spad}));
    return success();
  }

private:
  int64_t addrLen;
};

struct GemminiMvoutLowering : public OpRewritePattern<MvoutOp> {
  using OpRewritePattern<MvoutOp>::OpRewritePattern;
  explicit GemminiMvoutLowering(MLIRContext *context, int64_t addrLen)
      : OpRewritePattern(context), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(MvoutOp mvoutOp, PatternRewriter &rewriter) const override {
    Value output = mvoutOp.getOutput();
    TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
    Location loc = mvoutOp.getLoc();
    Value extractOp = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, resultType, output);
    IntegerType i64Type = rewriter.getI64Type();
    Value indexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOp);
    Value spadAddr = mvoutOp.getAddr();
    uint64_t number = getNumberFromValue(spadAddr);
    MemRefType memRefType =dyn_cast<MemRefType>(mvoutOp.getOperandTypes().front());
    llvm::ArrayRef<int64_t> memRefShape = memRefType.getShape();
    uint64_t spadAddrInt = (uint64_t)memRefShape[0] << (addrLen + 16) |
                           (uint64_t)memRefShape[1] << addrLen | number;
    Value newSpad = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(spadAddrInt));

    auto libraryCallName = getLibraryCallSymbolRef(mvoutOp, "mvout", rewriter);
    if (failed(libraryCallName))
      return failure();

    rewriter.replaceOpWithNewOp<func::CallOp>(
      mvoutOp, libraryCallName->getValue(), TypeRange(), 
      ValueRange({indexCastOp, newSpad}));
    return success();
  }

private:
  int64_t addrLen;
};

void mlir::populateGemminiLegalizeForFuncExportPatterns(
    RewritePatternSet &patterns, int64_t dim, int64_t addrLen, 
    int64_t accRows, int64_t bankRows, size_t sizeOfElemT,
    size_t sizeOfAccT) {
//   patterns
//       .add<ForwardOperands<func::CallOp>, ForwardOperands<func::CallIndirectOp>,
//            ForwardOperands<func::ReturnOp>>(converter, &converter.getContext());
//   patterns.add<GemminiFlushLowering>(converter);
  patterns.add<GemminiConfigStLowering>(patterns.getContext());
  patterns.add<GemminiConfigLdLowering>(patterns.getContext(), dim);
  patterns.add<GemminiMvinLowering>(patterns.getContext(), addrLen);
//   patterns.add<GemminiMvin2Lowering>(converter, addrLen);
//   patterns.add<GemminiMvin3Lowering>(converter, addrLen);
  patterns.add<GemminiMvoutLowering>(patterns.getContext(), addrLen);
//   patterns.add<GemminiConfigExLowering>(converter);
//   patterns.add<GemminiConfigNormLowering>(converter);
//   patterns.add<GemminiPreloadZerosLowering>(converter, dim, addrLen);
//   patterns.add<GemminiPreloadLowering>(converter, addrLen);
//   patterns.add<GemminiComputePreloadedLowering>(converter, addrLen);
//   patterns.add<GemminiComputeAccumulatedLowering>(converter, addrLen);
//   patterns.add<GemminiTileMatMulLowering>(converter, dim, addrLen, accRows,
//                                           bankRows, sizeOfElemT, sizeOfAccT);
//   patterns.add<GemminiTileConvLowering>(converter, dim, addrLen, accRows,
//                                         bankRows, sizeOfElemT, sizeOfAccT);
}