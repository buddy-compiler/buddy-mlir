//====- LowerImgPass.cpp - Img Dialect Lowering Pass  ---------------------===//
//
// This file defines image dialect lowering pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Pass/Pass.h"

#include "Img/ImgDialect.h"
#include "Img/ImgOps.h"

using namespace mlir;
using namespace buddy;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
class LowerImgPattern : public ConversionPattern {
public:
  explicit LowerImgPattern(MLIRContext *context)
      : ConversionPattern(img::PrewittOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();
    auto result = op->getResult(0);
    // Create constant index.
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    Value c2 = rewriter.create<ConstantIndexOp>(loc, 2);
    Value c3 = rewriter.create<ConstantIndexOp>(loc, 3);
    // Define the kernel with dense attribute.
    FloatType f32 = mlir::FloatType::getF32(ctx);
    VectorType vectorTy = mlir::VectorType::get({3, 3}, f32);
    Value kernel = rewriter.create<ConstantOp>(
        loc, DenseFPElementsAttr::get<float>(
                 vectorTy, {-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0}));
    // Construct result memref.
    auto memref = result.getType().cast<MemRefType>();
    Value alloc =
        rewriter.create<memref::AllocOp>(loc, memref, ValueRange{c3, c3});
    Value row1 = rewriter.create<vector::ExtractOp>(loc, kernel, 0);
    Value row2 = rewriter.create<vector::ExtractOp>(loc, kernel, 1);
    Value row3 = rewriter.create<vector::ExtractOp>(loc, kernel, 2);
    rewriter.create<vector::StoreOp>(loc, row1, result, ValueRange{c0, c0});
    rewriter.create<vector::StoreOp>(loc, row2, result, ValueRange{c1, c0});
    rewriter.create<vector::StoreOp>(loc, row3, result, ValueRange{c2, c0});

    rewriter.replaceOp(op, alloc);
    return success();
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// LowerImgPass
//===----------------------------------------------------------------------===//

namespace {
class LowerImgPass : public PassWrapper<LowerImgPass, OperationPass<ModuleOp>> {
public:
  LowerImgPass() = default;
  LowerImgPass(const LowerImgPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<buddy::img::ImgDialect, linalg::LinalgDialect,
                    StandardOpsDialect, memref::MemRefDialect,
                    vector::VectorDialect>();
  }
};
} // end anonymous namespace.

void LowerImgPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect,
                         memref::MemRefDialect, vector::VectorDialect>();
  target.addLegalOp<ModuleOp, FuncOp, ReturnOp>();

  RewritePatternSet patterns(context);
  patterns.add<LowerImgPattern>(context);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerImgPass() {
  PassRegistration<LowerImgPass>("lower-img", "Lower Img Dialect.");
}
} // namespace buddy
} // namespace mlir
