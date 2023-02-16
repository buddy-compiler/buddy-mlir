#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "Gemmini/Transform.h"
#include "Gemmini/GemminiDialect.h"
#include "Gemmini/GemminiOps.h"

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
    mlir::Value strideValue = configStOp.getStride();
    mlir::Operation* op = strideValue.getDefiningOp();
    mlir::Attribute attr = op->getAttr("value");
    mlir::IntegerAttr intAttr = attr.dyn_cast<IntegerAttr>();
    uint64_t stride = intAttr.getInt();
    mlir::Type i64Type = rewriter.getI64Type();
    mlir::Attribute input0 = rewriter.getI64IntegerAttr(CONFIG_ST); 
    mlir::Location loc = configStOp.getLoc(); 
    uint64_t arg = (uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)ACC_SCALE_IDENTITY) << 32 | (uint32_t)stride;
    mlir::Attribute input1 = rewriter.getI64IntegerAttr(arg);
    mlir::Value value1 = rewriter.create<arith::ConstantOp>(loc, input0, i64Type);
    mlir::Value value2 = rewriter.create<arith::ConstantOp>(loc, input1, i64Type);
    rewriter.replaceOpWithNewOp<ConfigSt_IntrOp>(configStOp, value1, value2);
    return success();
  }
};

struct GemminiConfigLdOpLowering : public ConvertOpToLLVMPattern<ConfigLdOp> {
  using ConvertOpToLLVMPattern<ConfigLdOp>::ConvertOpToLLVMPattern;
  LogicalResult 
  matchAndRewrite(ConfigLdOp configLdOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    mlir::Value strideValue = configLdOp.getStride();
    uint64_t arg = (uint64_t)scale_t_to_scale_t_bits(MVIN_SCALE_IDENTITY) << 32 | ((uint64_t)16 << 16) | (uint64_t)1 << 8 | 0 << 2 | CONFIG_LD;
    mlir::Type i64Type = rewriter.getI64Type();
    mlir::Attribute input = rewriter.getI64IntegerAttr(arg);
    mlir::Location loc = configLdOp.getLoc();
    mlir::Value value = rewriter.create<arith::ConstantOp>(loc, input, i64Type);
    rewriter.replaceOpWithNewOp<ConifgLd_IntrOp>(configLdOp, value, strideValue);
    return success();
  }
};


struct GemminiMvinOpLowering : public ConvertOpToLLVMPattern<MvinOp> {
  using ConvertOpToLLVMPattern<MvinOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MvinOp mvinOp, OpAdaptor adaptor, 
                  ConversionPatternRewriter &rewriter) const override {
    mlir::Value inputValue = mvinOp.getInput();
    mlir::Value spadAddrValue = mvinOp.getAddr();
    mlir::IntegerAttr spadAddrAttr = spadAddrValue.getDefiningOp()->getAttr("value").dyn_cast<IntegerAttr>();
    mlir::Type i64Type = rewriter.getI64Type();
    uint64_t spadAddrInt = (uint64_t)DIM << (ADDR_LEN + 16) | (uint64_t)DIM << ADDR_LEN | spadAddrAttr.getInt();
    mlir::Attribute input = rewriter.getI64IntegerAttr(spadAddrInt);
    mlir::Location loc = mvinOp.getLoc();
    mlir::Value value = rewriter.create<arith::ConstantOp>(loc, input, i64Type);
    rewriter.replaceOpWithNewOp<Mvin_IntrOp>(mvinOp, inputValue, value);
    return success();
  }
};

struct GemminiMvoutLowering : public ConvertOpToLLVMPattern<MvoutOp> {
    using ConvertOpToLLVMPattern<MvoutOp>::ConvertOpToLLVMPattern;
    LogicalResult 
    matchAndRewrite(MvoutOp mvoutOp, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
    mlir::Value outputValue = mvoutOp.getOutput();
    mlir::Value spadAddrValue = mvoutOp.getAddr();
    mlir::IntegerAttr spadAddrAttr = spadAddrValue.getDefiningOp()->getAttr("value").dyn_cast<IntegerAttr>();
    mlir::Type i64Type = rewriter.getI64Type();
    uint64_t spadAddrInt = (uint64_t)DIM << (ADDR_LEN + 16) | (uint64_t)DIM << ADDR_LEN | spadAddrAttr.getInt();
    mlir::Attribute output = rewriter.getI64IntegerAttr(spadAddrInt);
    mlir::Location loc = mvoutOp.getLoc();
    mlir::Value value = rewriter.create<arith::ConstantOp>(loc, output, i64Type);
    rewriter.replaceOpWithNewOp<Mvout_IntrOp>(mvoutOp, outputValue, value);
    return success();
  }
};

void mlir::populateGemminiLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,RewritePatternSet &patterns) {
  patterns.add<ForwardOperands<func::CallOp>,
               ForwardOperands<func::CallIndirectOp>,
               ForwardOperands<func::ReturnOp>
               >(converter, &converter.getContext());
  patterns.add<GemminiConfigStOpLowering>(converter);
  patterns.add<GemminiConfigLdOpLowering>(converter);
  patterns.add<GemminiMvinOpLowering>(converter);
  patterns.add<GemminiMvoutLowering>(converter);
}

void mlir::configureGemminiegalizeForExportTarget(LLVMConversionTarget &target) {
  target.addLegalOp<ConfigSt_IntrOp, ConifgLd_IntrOp, Mvin_IntrOp, Mvout_IntrOp>();
  target.addIllegalOp<ConfigStOp, ConfigLdOp, MvinOp, MvoutOp, PrintOp>();
}