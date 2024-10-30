#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

#include "Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class SimplifyAddAndReshapePattern : public RewritePattern {
public:
  explicit SimplifyAddAndReshapePattern(MLIRContext *context)
      : RewritePattern(tosa::ReshapeOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto reshapeOp = cast<tosa::ReshapeOp>(op);
    auto addOp = reshapeOp.getOperand().getDefiningOp<tosa::AddOp>();
    if (!addOp)
      return failure();

    auto constOp1 = addOp.getOperand(0).getDefiningOp<arith::ConstantOp>();
    auto constOp2 = addOp.getOperand(1).getDefiningOp<tosa::ConstOp>();
    if (!constOp1 || !constOp2)
      return failure();

    DenseElementsAttr constAttr2 =
        constOp2.getValue().cast<DenseElementsAttr>();
    if (!constAttr2.isSplat() || !constAttr2.getSplatValue<APFloat>().isZero())
      return failure();

    auto resultTy = cast<ShapedType>(reshapeOp.getType());
    rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
        reshapeOp, reshapeOp.getType(), constOp1,
        rewriter.getDenseI64ArrayAttr(resultTy.getShape()));

    return success();
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {

struct SimplifyTosaAddAndReshapePass
    : public PassWrapper<SimplifyTosaAddAndReshapePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimplifyTosaAddAndReshapePass)
  StringRef getArgument() const final { return "simplify-tosa-add-reshape"; }
  StringRef getDescription() const final {
    return "Simplify tosa.add and tosa.reshape operations.";
  }
  SimplifyTosaAddAndReshapePass() = default;
  SimplifyTosaAddAndReshapePass(const SimplifyTosaAddAndReshapePass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tosa::TosaDialect, arith::ArithDialect>();
  }
};

} // end anonymous namespace

void SimplifyTosaAddAndReshapePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                         scf::SCFDialect, memref::MemRefDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<SimplifyAddAndReshapePattern>(context);

  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace buddy {
void registerSimplifyTosaAddAndReshapePass() {
  PassRegistration<SimplifyTosaAddAndReshapePass>();
}
} // namespace buddy
} // namespace mlir
