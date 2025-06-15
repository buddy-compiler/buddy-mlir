#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"



using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

class SiLUVectorizePattern : public ConversionPattern {
public:
  explicit SiLUVectorizePattern(MLIRContext *context, int64_t vectorSizeParam)
      : ConversionPattern(linalg::GenericOp::getOperationName(), 1, context) {
    vectorSize = vectorSizeParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    linalg::GenericOp sigmoidOp = cast<linalg::GenericOp>(op);
    
    //--------------sigmoid OP--------
    // 1. Check input/output
    if (sigmoidOp.getNumDpsInputs() != 1 || sigmoidOp.getNumDpsInits() != 1){
      llvm::errs() << "1\n";
      return failure();}

    // Check the body of the op for sigmoid computation.
    // The IR should be: negf, exp, addf, divf, yield.
    Block &block = sigmoidOp.getRegion().front();
    if (block.getOperations().size() != 5) // negf, exp, addf, divf, yield
      {llvm::errs() << "4\n";
      return failure();}

    Operation &negfOp = block.getOperations().front();
    Operation &yieldOp = block.getOperations().back();

    // Check the type of the two operations.
    if (!isa<arith::NegFOp>(negfOp) || !isa<linalg::YieldOp>(yieldOp))
      {llvm::errs() << "5\n";
      return failure();}


    //-----------Find the consumer mul operation.------------------------------
    // The result of the sigmoid op must be used by another linalg.generic op.
    Value outputBuffer = sigmoidOp.getDpsInitOperand(0)->get();

    // 遍历所有 uses，寻找满足条件的 consumer op
    linalg::GenericOp mulOp = nullptr;

    for (auto &use : outputBuffer.getUses()) {
      Operation *user = use.getOwner();

      // 要求是 linalg.generic，且 %alloc 是 input operand（即 ins()）
      auto linalgOp = dyn_cast<linalg::GenericOp>(user);
      if (!linalgOp)
        continue;

      bool foundInInput = false;
      for (OpOperand *input : linalgOp.getDpsInputOperands()) {
        if (input->get() == outputBuffer) {
          foundInInput = true;
          break;
        }
      }
      if (!foundInInput)
        continue;

      // 检查其内部是否有 arith.mulf 操作
      for (auto &nestedOp : linalgOp.getRegion().front()) {
        if (isa<arith::MulFOp>(nestedOp)) {
          mulOp = linalgOp;
          break;
        }
      }

      if (mulOp)
        break;
    }

    if (!mulOp) {
      llvm::errs() << "Didn't find a consumer linalg.generic using sigmoid output with mulf.\n";
      return failure();
    }

    // Set the insertion point before the mulOp. This ensures that the new affine
    // loop is inserted at a point that is dominated by the allocation of the
    // output buffer.
    // rewriter.setInsertionPoint(mulOp);

    // Now we have matched the silu pattern: sigmoid followed by a mul.
    // The rewrite logic will be applied to the sigmoidOp, and the mulOp will be
    // erased.

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(mulOp);
    Location loc = sigmoidOp.getLoc();
    Value input = sigmoidOp.getDpsInputOperand(0)->get();
    // The final output buffer comes from the mulOp.
    Value output = mulOp.getDpsInitOperand(0)->get();

    auto inputMemRefType = input.getType().cast<MemRefType>();
    Type elementType = inputMemRefType.getElementType();
    VectorType vectorType = VectorType::get({vectorSize}, elementType);

    // Define constants.
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    const int64_t unrollFactor = 2;
    Value cUnrollVec =
        rewriter.create<arith::ConstantIndexOp>(loc, vectorSize * unrollFactor);
    Value cst1f = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(1.0));
    Value vec1f = rewriter.create<vector::BroadcastOp>(loc, vectorType, cst1f);

    // Get dimensions.
    Value d0 = rewriter.create<memref::DimOp>(loc, input, 0);
    Value d1 = rewriter.create<memref::DimOp>(loc, input, 1);
    Value d2 = rewriter.create<memref::DimOp>(loc, input, 2);

    // Create loop nest.
    scf::ForOp iLoop = rewriter.create<scf::ForOp>(loc, c0, d0, c1);
    rewriter.setInsertionPointToStart(iLoop.getBody());
    Value iv_i = iLoop.getInductionVar();

    scf::ForOp jLoop = rewriter.create<scf::ForOp>(loc, c0, d1, c1);
    rewriter.setInsertionPointToStart(jLoop.getBody());
    Value iv_j = jLoop.getInductionVar();

    scf::ForOp kLoop = rewriter.create<scf::ForOp>(loc, c0, d2, cUnrollVec);
    rewriter.setInsertionPointToStart(kLoop.getBody());
    Value iv_k = kLoop.getInductionVar();

    // Prefetch
    Value k_next = rewriter.create<arith::AddIOp>(loc, iv_k, cUnrollVec);
    rewriter.create<memref::PrefetchOp>(loc, input, ValueRange{iv_i, iv_j, k_next},
                                      /*isWrite=*/false, /*localityHint=*/3,
                                      /*isDataCache=*/true);

    // Unrolled loop body
    for (int i = 0; i < unrollFactor; ++i) {
      Value k_offset =
          rewriter.create<arith::ConstantIndexOp>(loc, i * vectorSize);
      Value k_i = rewriter.create<arith::AddIOp>(loc, iv_k, k_offset);

      // --- Process Vector ---
      Value x_vec = rewriter.create<vector::LoadOp>(
          loc, vectorType, input, ValueRange{iv_i, iv_j, k_i});
      Value neg_x_vec = rewriter.create<arith::NegFOp>(loc, x_vec);
      Value exp_neg_x_vec = rewriter.create<math::ExpOp>(loc, neg_x_vec);
      Value one_plus_exp_vec =
          rewriter.create<arith::AddFOp>(loc, vec1f, exp_neg_x_vec);
      Value sigmoid_x_vec =
          rewriter.create<arith::DivFOp>(loc, vec1f, one_plus_exp_vec);
      Value silu_vec = rewriter.create<arith::MulFOp>(loc, x_vec, sigmoid_x_vec);
      rewriter.create<vector::StoreOp>(loc, silu_vec, output,
                                      ValueRange{iv_i, iv_j, k_i});
    }

    // Replace the original mulOp with the result from our new computation.
    // The 'output' buffer now holds the final result. `replaceOp` will
    // replace all uses of mulOp's results with `output` and then erase mulOp.
    rewriter.eraseOp(mulOp);
    rewriter.eraseOp(sigmoidOp);

    return success();
  }

private:
  int64_t vectorSize;
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class SiluOptimizationPass
    : public PassWrapper<SiluOptimizationPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SiluOptimizationPass)
  StringRef getArgument() const final { return "silu-optimization"; }
  StringRef getDescription() const final {
    return "Vectorize linalg.generic representing SiLU.";
  }
  SiluOptimizationPass() = default;
  SiluOptimizationPass(const SiluOptimizationPass &) {}
  explicit SiluOptimizationPass(int64_t vectorSizeParam) {
    vectorSize = vectorSizeParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, affine::AffineDialect,
                    vector::VectorDialect, math::MathDialect, scf::SCFDialect,
                    arith::ArithDialect, memref::MemRefDialect>();
  }

  Option<int64_t> vectorSize{*this, "vector-size",
                               llvm::cl::desc("Vector size for SiLU."),
                               llvm::cl::init(8)};
};

void SiluOptimizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                         memref::MemRefDialect, vector::VectorDialect,
                         func::FuncDialect, math::MathDialect,
                         scf::SCFDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp>();
  
  // We will manually mark linalg.generic as illegal if it is part of a SiLU pattern.
  // The pattern itself will handle the legality checks and replacements.
  // Therefore, we don't need to addIllegalOp<linalg::GenericOp>() here.
  
  RewritePatternSet patterns(context);
  patterns.add<SiLUVectorizePattern>(context, vectorSize);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
} // end anonymous namespace 
namespace mlir {
namespace buddy {
void registerSiluOptimizationPass() {
  PassRegistration<SiluOptimizationPass>();
}
} // namespace buddy
} // namespace mlir
