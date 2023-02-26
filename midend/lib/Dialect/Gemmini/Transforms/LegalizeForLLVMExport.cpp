//===- LegalizeForLLVMExport.cpp - Prepare Gemmini for LLVM translation --===//
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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "Gemmini/GemminiDialect.h"
#include "Gemmini/GemminiOps.h"
#include "Gemmini/Transform.h"

using namespace mlir;
using namespace buddy::gemmini;

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

class ReturnOpTypeConversion : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.updateRootInPlace(
        op, [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

struct GemminiConfigStOpLowering : public ConvertOpToLLVMPattern<ConfigStOp> {
  using ConvertOpToLLVMPattern<ConfigStOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ConfigStOp configStOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value strideValue = configStOp.getStride();
    Operation *op = strideValue.getDefiningOp();
    Attribute attr = op->getAttr("value");
    IntegerAttr intAttr = attr.dyn_cast<IntegerAttr>();
    uint64_t stride = intAttr.getInt();
    Type i64Type = rewriter.getI64Type();
    Attribute input0 = rewriter.getI64IntegerAttr(CONFIG_ST);
    Location loc = configStOp.getLoc();
    uint64_t arg = (uint64_t)acc_scale_t_to_acc_scale_t_bits(
                       (acc_scale_t)ACC_SCALE_IDENTITY)
                       << 32 |
                   (uint32_t)stride;
    Attribute input1 = rewriter.getI64IntegerAttr(arg);
    Value value1 = rewriter.create<arith::ConstantOp>(loc, input0, i64Type);
    Value value2 = rewriter.create<arith::ConstantOp>(loc, input1, i64Type);
    rewriter.replaceOpWithNewOp<ConfigSt_IntrOp>(configStOp, value1, value2);
    return success();
  }
};

struct GemminiConfigLdOpLowering : public ConvertOpToLLVMPattern<ConfigLdOp> {
  using ConvertOpToLLVMPattern<ConfigLdOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ConfigLdOp configLdOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value rs2Value = configLdOp.getStride();
    float scale = configLdOp.getScale().convertToFloat();
    uint64_t rs1 = (uint64_t)scale_t_to_scale_t_bits(scale) << 32 |
                   ((uint64_t)16 << 16) | (uint64_t)1 << 8 |
                   configLdOp.getShrunk() << 2 | CONFIG_LD;
    Type i64Type = rewriter.getI64Type();
    Attribute rs1Attr = rewriter.getI64IntegerAttr(rs1);
    Location loc = configLdOp.getLoc();
    Value rs1value = rewriter.create<arith::ConstantOp>(loc, rs1Attr, i64Type);
    rewriter.replaceOpWithNewOp<ConifgLd_IntrOp>(configLdOp, rs1value,
                                                 rs2Value);
    return success();
  }
};

struct GemminiConfigExOpLowering : public ConvertOpToLLVMPattern<ConfigExOp> {
  using ConvertOpToLLVMPattern<ConfigExOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ConfigExOp configExOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    IntegerType i64Type = rewriter.getI64Type();
    Location loc = configExOp.getLoc();
    float scale = configExOp.getSysAccScale().convertToFloat();
    uint64_t rs1 =
        (uint64_t)acc_scale_t_to_acc_scale_t_bits(scale) << 32 |
        configExOp.getAStride() << 16 | configExOp.getBTranspose() << 9 |
        configExOp.getATranspose() << 8 | configExOp.getSysAct() << 3 |
        configExOp.getDataflow() << 2 | CONFIG_EX;

    uint64_t rs2 = (uint64_t)1 << 48 | configExOp.getSysShift();
    IntegerAttr rs1Attr = rewriter.getI64IntegerAttr(rs1);
    IntegerAttr rs2Attr = rewriter.getI64IntegerAttr(rs2);
    Value rs1Value = rewriter.create<arith::ConstantOp>(loc, rs1Attr, i64Type);
    Value rs2Value = rewriter.create<arith::ConstantOp>(loc, rs2Attr, i64Type);
    rewriter.replaceOpWithNewOp<ConfigEX_IntrOp>(configExOp, rs1Value,
                                                 rs2Value);
    return success();
  }
};

struct GemminiMvinOpLowering : public ConvertOpToLLVMPattern<MvinOp> {
  using ConvertOpToLLVMPattern<MvinOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MvinOp mvinOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = mvinOp.getInput();
    Location loc = input.getLoc();
    MemRefType memRefType =
        mvinOp.getOperandTypes().front().dyn_cast<MemRefType>();
    llvm::ArrayRef<int64_t> memRefShape = memRefType.getShape();
    TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
    Value extractOp = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, resultType, input);
    IntegerType i64Type = rewriter.getI64Type();
    Value indexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOp);
    Value spadAddrValue = mvinOp.getAddr();
    IntegerAttr spadAddrAttr =
        spadAddrValue.getDefiningOp()->getAttr("value").dyn_cast<IntegerAttr>();
    uint64_t spadAddrInt = (uint64_t)memRefShape[0] << (ADDR_LEN + 16) |
                           (uint64_t)memRefShape[1] << ADDR_LEN |
                           spadAddrAttr.getInt();
    Attribute newSpadAddr = rewriter.getI64IntegerAttr(spadAddrInt);
    Value spad = rewriter.create<arith::ConstantOp>(loc, newSpadAddr, i64Type);
    rewriter.replaceOpWithNewOp<Mvin_IntrOp>(mvinOp, indexCastOp, spad);
    return success();
  }
};

struct GemminiMvoutLowering : public ConvertOpToLLVMPattern<MvoutOp> {
  using ConvertOpToLLVMPattern<MvoutOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MvoutOp mvoutOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value output = mvoutOp.getOutput();
    TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
    Location loc = mvoutOp.getLoc();
    Value extractOp = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, resultType, output);
    IntegerType i64Type = rewriter.getI64Type();
    Value indexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOp);
    Value spadAddr = mvoutOp.getAddr();
    IntegerAttr spadAddrAttr =
        spadAddr.getDefiningOp()->getAttr("value").dyn_cast<IntegerAttr>();
    MemRefType memRefType =
        mvoutOp.getOperandTypes().front().dyn_cast<MemRefType>();
    llvm::ArrayRef<int64_t> memRefShape = memRefType.getShape();
    uint64_t spadAddrInt = (uint64_t)memRefShape[0] << (ADDR_LEN + 16) |
                           (uint64_t)memRefShape[1] << ADDR_LEN |
                           spadAddrAttr.getInt();
    Attribute newSpadAddr = rewriter.getI64IntegerAttr(spadAddrInt);
    Value newSpad =
        rewriter.create<arith::ConstantOp>(loc, newSpadAddr, i64Type);
    rewriter.replaceOpWithNewOp<Mvout_IntrOp>(mvoutOp, indexCastOp, newSpad);
    return success();
  }
};

void mlir::populateGemminiLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns
      .add<ForwardOperands<func::CallOp>, ForwardOperands<func::CallIndirectOp>,
           ForwardOperands<func::ReturnOp>>(converter, &converter.getContext());
  patterns.add<GemminiConfigStOpLowering>(converter);
  patterns.add<GemminiConfigLdOpLowering>(converter);
  patterns.add<GemminiMvinOpLowering>(converter);
  patterns.add<GemminiMvoutLowering>(converter);
  patterns.add<GemminiConfigExOpLowering>(converter);
}

void mlir::configureGemminiegalizeForExportTarget(
    LLVMConversionTarget &target) {
  target.addLegalOp<ConfigSt_IntrOp, ConifgLd_IntrOp, ConfigEX_IntrOp,
                    Mvin_IntrOp, Mvout_IntrOp>();
  target.addIllegalOp<ConfigStOp, ConfigLdOp, ConfigExOp, MvinOp, MvoutOp,
                      PrintOp>();
}
