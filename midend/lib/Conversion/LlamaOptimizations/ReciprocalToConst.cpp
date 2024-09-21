#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

#include "Utils/Utils.h"

#include <mlir/IR/IntegerSet.h>
#include <mlir/Pass/Pass.h>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace vector;

namespace {

struct ReciprocalToConstPattern : public RewritePattern {
  ReciprocalToConstPattern(MLIRContext *context)
      : RewritePattern(tosa::ReciprocalOp::getOperationName(), 1, context) {}

  // matchAndRewrite是重写模式的核心函数
  LogicalResult matchAndRewrite(Operation *op,
                                      PatternRewriter &rewriter) const override {
    // 检查操作是否为tosa.reciprocal
    auto reciprocalOp = llvm::dyn_cast<tosa::ReciprocalOp>(op);
    if (!reciprocalOp)
      return failure();

    // 检查输入是否为tosa.const
    auto constOp = reciprocalOp.getOperand().getDefiningOp<tosa::ConstOp>();
    if (!constOp)
      return failure();

    // 提取常量值
    auto denseAttr = constOp.getValue().dyn_cast<DenseElementsAttr>();
    if (!denseAttr)
      return failure();

    // 计算常量值的倒数
    auto inputValues = denseAttr.getValues<float>();
    // llvm::SmallVector<float> reciprocalValues;
    // for (float val : inputValues) {
    //   if (val != 0.0f)
    //     reciprocalValues.push_back(1.0f / val);
    //   else
    //     return failure(); // 避免除零错误
    // }
    auto reciprocalValue = 1.0f / inputValues[0];

    // 创建新的常量操作存储倒数
    auto newDenseAttr = DenseElementsAttr::get(denseAttr.getType(), reciprocalValue);
    auto newConstOp = rewriter.create<tosa::ConstOp>(op->getLoc(), reciprocalOp.getType(), newDenseAttr);

    // 用新的常量操作替换tosa.reciprocal
    rewriter.replaceOp(op, newConstOp.getResult());

    return mlir::success();
  }
};
} // end anonymous namespace

namespace {
class EliminateReciprocalPass : public PassWrapper<EliminateReciprocalPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EliminateReciprocalPass)
  StringRef getArgument() const final { return "reciprocal-elimination"; }
  StringRef getDescription() const final { return "combine const and reciprocal."; }
  EliminateReciprocalPass() = default;
  EliminateReciprocalPass(const EliminateReciprocalPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tosa::TosaDialect>();
  }

};
} // end anonymous namespace.

void EliminateReciprocalPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<tosa::TosaDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<ReciprocalToConstPattern>(context);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}


namespace mlir {
namespace buddy {
void registerEliminateReciprocalPass() { PassRegistration<EliminateReciprocalPass>(); }
} // namespace buddy
} // namespace mlir