#include "RVV/RVVDialect.h"
#include "RVV/Transforms.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
using namespace ::buddy::rvv;

namespace {
template <typename Source, typename Target>
struct FloatBinaryLowering : ConvertOpToLLVMPattern<Source> {
  using ConvertOpToLLVMPattern<Source>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(Source op, typename Source::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = this->getTypeConverter()->convertType(op.getResult().getType());
    Value passthru = LLVM::PoisonOp::create(rewriter, op.getLoc(), type);
    auto width = this->getTypeConverter()->getIndexType();
    Value rounding = LLVM::ConstantOp::create(
        rewriter, op.getLoc(), width, rewriter.getIntegerAttr(width, 7));
    Value length = adaptor.getLength();
    if (length.getType() != width) {
      op.emitError("vector length must match target XLEN");
      return failure();
    }
    rewriter.template replaceOpWithNewOp<Target>(
        op, type, passthru, adaptor.getSrc1(), adaptor.getSrc2(), rounding,
        length);
    return success();
  }
};
} // namespace

void mlir::populateRVVArithmeticPatterns(LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns) {
  patterns.add<FloatBinaryLowering<RVVFAddOp, RVVIntrFAddOp>,
               FloatBinaryLowering<RVVFSubOp, RVVIntrFSubOp>,
               FloatBinaryLowering<RVVFMulOp, RVVIntrFMulOp>,
               FloatBinaryLowering<RVVFDivOp, RVVIntrFDivOp>>(converter);
}
