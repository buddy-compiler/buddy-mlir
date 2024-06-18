#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <cstdio>
#include <mlir/Dialect/Affine/Analysis/AffineAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;
using namespace vector;
using namespace affine;

namespace {

class BatchMatMulToMatmulPattern : public ConversionPattern {
public:
  explicit BatchMatMulToMatmulPattern(MLIRContext *context)
      : ConversionPattern(linalg::BatchMatmulOp::getOperationName(), 1,
                          context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // Retrieve input tensors A, B, and C.
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);

    ShapedType ATy = A.getType().cast<ShapedType>();
    ShapedType BTy = B.getType().cast<ShapedType>();
    ShapedType CTy = C.getType().cast<ShapedType>();
    // Acquire the element type of input tensors.
    Type elementType = A.getType().cast<MemRefType>().getElementType();

    // Define constants.
    const Value c0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value c1 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    const AffineExpr d0 = rewriter.getAffineDimExpr(0);
    const AffineExpr zeroAffine = rewriter.getAffineConstantExpr(0);

    // Get dimensions of input tensors.
    Value batch = rewriter.create<memref::DimOp>(loc, A, 0);
    Value aRow = rewriter.create<memref::DimOp>(loc, A, 1);
    Value bCol = rewriter.create<memref::DimOp>(loc, B, 2);
    Value bRow = rewriter.create<memref::DimOp>(loc, B, 1);

    SmallVector<Value, 4U> reducedValues = llvm::to_vector<4>(
        llvm::map_range(ArrayRef<LoopReduction>{},
                        [](const LoopReduction &red) { return red.value; }));

    // Create the primary parallel batch level loop.
    AffineParallelOp parallelBatchLoop =
        rewriter.create<affine::AffineParallelOp>(
            loc, ValueRange(reducedValues).getTypes(), ValueRange{batch},
            ArrayRef<NamedAttribute>{
                rewriter.getNamedAttr("lowerBoundsGroups",
                                      rewriter.getI32TensorAttr({1})),
                rewriter.getNamedAttr("upperBoundsGroups",
                                      rewriter.getI32TensorAttr({1})),
                rewriter.getNamedAttr(
                    "lowerBoundsMap",
                    AffineMapAttr::get(AffineMap::get(0, 0, {zeroAffine},
                                                      rewriter.getContext()))),
                rewriter.getNamedAttr("upperBoundsMap",
                                      AffineMapAttr::get(AffineMap::get(
                                          1, 0, {d0}, rewriter.getContext()))),
                rewriter.getNamedAttr("reductions", rewriter.getArrayAttr({})),
                rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr({1}))});

    // Create the loop body for the parallel loop.
    Block *loopBody = new Block();
    rewriter.setInsertionPointToStart(loopBody);
    loopBody->addArgument(rewriter.getIndexType(), loc);
    Value loopVarBatchIdx = loopBody->getArguments()[0];

    MemRefType sliceATy = MemRefType::get(
        SmallVector<int64_t>(ATy.getShape().begin() + 1, ATy.getShape().end()),
        elementType); // get a slice from <?x?x?> to <?x?>

    MemRefType sliceBTy = MemRefType::get(
        SmallVector<int64_t>(BTy.getShape().begin() + 1, BTy.getShape().end()),
        elementType);

    MemRefType sliceCTy = MemRefType::get(
        SmallVector<int64_t>(CTy.getShape().begin() + 1, CTy.getShape().end()),
        elementType);

    auto aptr = rewriter.create<memref::SubViewOp>(
        loc, sliceATy, A, SmallVector<OpFoldResult>{loopVarBatchIdx, c0, c0},
        SmallVector<OpFoldResult>{c1, aRow, bRow},
        SmallVector<OpFoldResult>{batch, aRow, bRow});

    auto bptr = rewriter.create<memref::SubViewOp>(
        loc, sliceBTy, B, SmallVector<OpFoldResult>{loopVarBatchIdx, c0, c0},
        SmallVector<OpFoldResult>{c1, bRow, bCol},
        SmallVector<OpFoldResult>{batch, bRow, bCol});

    auto cptr = rewriter.create<memref::SubViewOp>(
        loc, sliceCTy, C, SmallVector<OpFoldResult>{loopVarBatchIdx, c0, c0},
        SmallVector<OpFoldResult>{c1, aRow, bCol},
        SmallVector<OpFoldResult>{batch, aRow, bCol});

    rewriter.create<linalg::MatmulOp>(loc, ValueRange{aptr, bptr},
                                      ValueRange{cptr});

    rewriter.create<affine::AffineYieldOp>(loc);
    // Finalize the loop and erase the original operation.
    parallelBatchLoop.getRegion().push_back(loopBody);
    rewriter.setInsertionPointAfter(parallelBatchLoop);

    rewriter.eraseOp(op);
    return success();
  }
};
} // end anonymous namespace

namespace {
class BatchMatMulToMatmulPass
    : public PassWrapper<BatchMatMulToMatmulPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BatchMatMulToMatmulPass)
  StringRef getArgument() const final { return "lower-batchmatmul-to-matmul"; }
  StringRef getDescription() const final {
    return "Lower batchMatMul to Matmul.";
  }
  BatchMatMulToMatmulPass() = default;
  BatchMatMulToMatmulPass(const BatchMatMulToMatmulPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, VectorDialect>();
  }
};
} // namespace

void BatchMatMulToMatmulPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                       scf::SCFDialect, memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<BatchMatMulToMatmulPattern>(context);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerBatchMatMulToMatmulPass() {
  PassRegistration<BatchMatMulToMatmulPass>();
}
} // namespace buddy
} // namespace mlir
