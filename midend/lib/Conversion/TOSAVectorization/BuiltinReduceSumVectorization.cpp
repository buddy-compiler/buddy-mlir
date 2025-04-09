#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include <cassert>
#include <cstdint>
#include <llvm/ADT/SmallVector.h>

using namespace mlir;
using namespace vector;
using namespace affine;
using namespace linalg;

namespace {
/// 针对 linalg::GenericOp 的转置操作进行向量化优化。
class ReduceSumVectorizationPattern : public ConversionPattern {
public:
  explicit ReduceSumVectorizationPattern(MLIRContext *context,
                                         int64_t affineVectorSizeParam)
      : ConversionPattern(linalg::ReduceOp::getOperationName(), 1, context),
        affineVectorSize(affineVectorSizeParam) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto reduceOp = cast<linalg::ReduceOp>(op);
    auto dimensionsAttr = reduceOp.getDimensions();

    // 只有在规约最后一个维度时，才进行向量化优化
    if (dimensionsAttr.size() != 1 ||
        dimensionsAttr[0] !=
            reduceOp.getOperand(0).getType().cast<MemRefType>().getRank() - 1)
      return failure();

    // 获取输入和输出
    Value input = reduceOp.getOperand(0);
    Value output = reduceOp.getOperand(1);
    auto loc = op->getLoc();
    Type elementType = input.getType().cast<MemRefType>().getElementType();

    // 获取输入张量的维度
    int rank = input.getType().cast<MemRefType>().getRank();

    // 获取各维度大小（使用 memref::DimOp）。
    llvm::SmallVector<Value> dims;
    for (int i = 0; i < rank; i++) {
      dims.push_back(rewriter.create<memref::DimOp>(loc, input, i));
    }
    Value innerDim = dims.back();

    // 计算 innerUpperBound = (innerDim floordiv affineVectorSize) *
    // affineVectorSize
    Value innerUpperBound = rewriter.create<affine::AffineApplyOp>(
        loc,
        AffineMap::get(1, 0,
                       rewriter.getAffineDimExpr(0).floorDiv(affineVectorSize) *
                           affineVectorSize),
        ValueRange{innerDim});

    // 计算 innerUnalignedLength = innerDim mod affineVectorSize
    Value innerUnalignedLength = rewriter.create<affine::AffineApplyOp>(
        loc,
        AffineMap::get(1, 0, rewriter.getAffineDimExpr(0) % affineVectorSize),
        ValueRange{innerDim});
    
    

    // 删除原始操作
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t affineVectorSize;

  // bool isReduceSumOp(GenericOp genericOp) const {
  //   if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1)
  //     return false;
  //   llvm::SmallVector<AffineMap, 2> indexingMaps =
  //       genericOp.getIndexingMapsArray();
  //   AffineMap inputMap = indexingMaps[0], outputMap = indexingMaps[1];
  //   if (inputMap.getNumDims() != outputMap.getNumDims() + 1) //
  //   输入比输出多一维
  //     return false;
  //   for (unsigned i = 0; i < outputMap.getNumDims(); ++i) {
  //     AffineExpr inputExpr = inputMap.getResult(i);
  //     AffineExpr outputExpr = outputMap.getResult(i);
  //     if (inputExpr != outputExpr)
  //       return false;
  //   }
  //   for (unsigned i = 0; i < genericOp.getIteratorTypesArray().size(); ++i) {
  //     auto it = genericOp.getIteratorTypesArray()[i];
  //     if (i == genericOp.getIteratorTypesArray().size() - 1) {
  //       if (it != utils::IteratorType::reduction)
  //         return false;
  //     } else {
  //       if (it != utils::IteratorType::parallel)
  //         return false;
  //     }
  //   }
  //   Region &body = genericOp.getRegion();
  //   if (!body.hasOneBlock())
  //     return false;

  //   auto &block = body.front();
  //   for (Operation &op : block) {
  //     if (!isa<arith::AddFOp>(op) && !isa<linalg::YieldOp>(op))
  //       return false;
  //   }
  //   return true;
  // }
};
} // namespace

//===----------------------------------------------------------------------===//
// ReduceSumVectorizationPass
//===----------------------------------------------------------------------===//

namespace {
class ReduceSumVectorizationPass
    : public PassWrapper<ReduceSumVectorizationPass, OperationPass<ModuleOp>>

{
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReduceSumVectorizationPass)

  StringRef getArgument() const

      final {
    return "tosa-reduce-sum-vectorization";
  }

  StringRef getDescription() const

      final {
    return "ReduceSum Optimization for any rank tensor.";
  }

  ReduceSumVectorizationPass() = default;

  ReduceSumVectorizationPass(const ReduceSumVectorizationPass &) {}

  explicit ReduceSumVectorizationPass(int64_t affineVectorSizeParam) {
    affineVectorSize = affineVectorSizeParam;
  }

  void runOnOperation()

      override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();
    ConversionTarget target(*context);
    target.

        addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                        memref::MemRefDialect, VectorDialect,
                        bufferization::BufferizationDialect>();

    target.

        addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp, linalg::FillOp>();

    RewritePatternSet patterns(context);
    patterns.add<ReduceSumVectorizationPattern>(context, affineVectorSize);
    if (failed(applyPartialConversion(module, target, std::move(patterns))))

      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const

      override {
    registry.

        insert<linalg::LinalgDialect, affine::AffineDialect, VectorDialect,
               memref::MemRefDialect, bufferization::BufferizationDialect>();
  }

  Option<int64_t> affineVectorSize{*this, "vector-size",
                                   llvm::cl::desc("Affine Vector size."),
                                   llvm::cl::init(16)};
};
} // namespace

namespace mlir {
namespace buddy {
void registerReduceSumVectorizationPass() {
  PassRegistration<ReduceSumVectorizationPass>();
}
} // namespace buddy
} // namespace mlir
