#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
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

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class ReduceSumVectorizationPattern : public ConversionPattern {
public:
  explicit ReduceSumVectorizationPattern(MLIRContext *context,
                                         int64_t affineVectorSizeParam)
      : ConversionPattern(linalg::ReduceOp::getOperationName(), 1, context) {
    affineVectorSize = affineVectorSizeParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto reduceOp = cast<linalg::ReduceOp>(op);

    // Get input and output tensors
    Value input = reduceOp.getOperand(0);
    Value output = reduceOp.getOperand(1);
    auto loc = op->getLoc();

    // Get element type and ranks
    Type elementType = input.getType().cast<MemRefType>().getElementType();
    int inputRank = input.getType().cast<MemRefType>().getRank();
    int outputRank = output.getType().cast<MemRefType>().getRank();

    // Get reduction dimensions
    auto dimensionsAttr = op->getAttr(rewriter.getStringAttr("dimensions"))
                              .cast<DenseI64ArrayAttr>()
                              .asArrayRef();
    if (dimensionsAttr.size() != 1 || dimensionsAttr[0] != inputRank - 1) {
      return failure();
    }

    // Define constants
    const Value index0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value zeroElementType = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));

    // Get input dimensions
    SmallVector<Value, 4> dims;
    for (int i = 0; i < inputRank; i++) {
      dims.push_back(rewriter.create<memref::DimOp>(loc, input, i));
    }

    // Create indices for input and output
    SmallVector<Value> inputIndices(inputRank, nullptr);
    SmallVector<Value> outputIndices(outputRank, nullptr);

    // Since we're only handling innermost reduction, we only need to process
    // the last dimension
    Value dimSize = dims.back();

    // Calculate upperBound = (dimSize floordiv vectorSize) * vectorSize
    Value upperBound = rewriter.create<affine::AffineApplyOp>(
        loc,
        AffineMap::get(1, 0,
                       rewriter.getAffineDimExpr(0).floorDiv(affineVectorSize) *
                           affineVectorSize),
        ValueRange{dimSize});

    // Calculate unalignedLength = dimSize mod vectorSize
    Value unalignedLength = rewriter.create<affine::AffineApplyOp>(
        loc,
        AffineMap::get(1, 0, rewriter.getAffineDimExpr(0) % affineVectorSize),
        ValueRange{dimSize});

    // Create vector mask for aligned and unaligned parts - now just 1D vector
    Value mask = rewriter.create<vector::CreateMaskOp>(
        loc, VectorType::get({affineVectorSize}, rewriter.getI1Type()),
        ValueRange{unalignedLength});

    // Create nested loops for all dimensions
    createNestedLoops(rewriter, loc, input, output, dims, elementType, 0,
                      inputIndices, outputIndices, index0, upperBound,
                      zeroElementType, affineVectorSize, mask, dimensionsAttr,
                      inputRank - 1);

    // Erase the original operation
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t affineVectorSize;

  void createNestedLoops(ConversionPatternRewriter &rewriter, Location loc,
                         Value input, Value output, SmallVector<Value, 4> &dims,
                         Type elementType, int64_t currentDim,
                         SmallVector<Value> &inputIndices,
                         SmallVector<Value> &outputIndices, Value index0,
                         Value upperBound, Value zeroElementType,
                         int64_t vectorSize, Value mask,
                         ArrayRef<int64_t> reductionDims,
                         int64_t currentReductionDim) const {
    // Base case: we've processed all dimensions
    if (currentDim >= static_cast<int64_t>(dims.size())) {
      return;
    }

    // Check if current dimension is the reduction dimension (innermost)
    bool isReductionDim = currentDim == currentReductionDim;

    // Create loop bounds
    Value dimSize = dims[currentDim];
    AffineMap lbMap = AffineMap::get(1, 0, {rewriter.getAffineConstantExpr(0)},
                                     rewriter.getContext());
    AffineMap ubMap = AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0)},
                                     rewriter.getContext());

    // For reduction dimensions, use vector size as step
    int64_t step = isReductionDim ? vectorSize : 1;
    Value bound = isReductionDim ? upperBound : dimSize;

    // Create the loop
    rewriter.create<affine::AffineForOp>(
        loc, ValueRange{index0}, lbMap, ValueRange{bound}, ubMap, step,
        ValueRange{},
        [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
            ValueRange iterArgs) {
          // Set current dimension index
          inputIndices[currentDim] = iv;

          // For non-reduction dimensions, copy index to output
          if (!isReductionDim) {
            outputIndices[currentDim] = iv;
          }

          // If this is the innermost reduction dimension
          if (isReductionDim) {
            // Create vector read operation
            SmallVector<Value> readOperands;
            readOperands.push_back(input);
            for (Value idx : inputIndices) {
              readOperands.push_back(idx);
            }
            readOperands.push_back(zeroElementType);

            auto readValue = nestedBuilder.create<vector::TransferReadOp>(
                nestedLoc, VectorType::get({vectorSize}, elementType),
                readOperands,
                ArrayRef<NamedAttribute>{
                    rewriter.getNamedAttr("in_bounds",
                                          rewriter.getBoolArrayAttr({true})),
                    rewriter.getNamedAttr(
                        "operand_segment_sizes",
                        rewriter.getDenseI32ArrayAttr(
                            {1, static_cast<int32_t>(inputIndices.size()), 1,
                             0})),
                    rewriter.getNamedAttr(
                        "permutation_map",
                        AffineMapAttr::get(AffineMap::getMinorIdentityMap(
                            inputIndices.size(), 1, rewriter.getContext())))});

            // Perform vector reduction
            auto reducedValue = nestedBuilder.create<vector::ReductionOp>(
                nestedLoc, vector::CombiningKind::ADD, readValue,
                /*acc=*/Value());

            // Store result
            nestedBuilder.create<memref::StoreOp>(nestedLoc, reducedValue,
                                                  output, outputIndices);

            // Handle unaligned tail if needed
            if (input.getType().cast<MemRefType>().isDynamicDim(currentDim) ||
                input.getType().cast<MemRefType>().getDimSize(currentDim) %
                        vectorSize !=
                    0) {
              affine::AffineIfOp tailIf =
                  nestedBuilder.create<affine::AffineIfOp>(
                      nestedLoc,
                      IntegerSet::get(
                          1, 0, {rewriter.getAffineDimExpr(0) % vectorSize - 1},
                          {false}),
                      ValueRange{dims[currentDim]}, false);

              OpBuilder tailBuilder = tailIf.getThenBodyBuilder();

              // Handle tail computation with mask
              SmallVector<Value> tailReadOperands;
              tailReadOperands.push_back(input);
              for (Value idx : inputIndices) {
                tailReadOperands.push_back(idx);
              }
              tailReadOperands.push_back(zeroElementType);
              tailReadOperands.push_back(mask);

              auto tailReadValue = tailBuilder.create<vector::TransferReadOp>(
                  nestedLoc, VectorType::get({vectorSize}, elementType),
                  tailReadOperands,
                  ArrayRef<NamedAttribute>{
                      rewriter.getNamedAttr("in_bounds",
                                            rewriter.getBoolArrayAttr({true})),
                      rewriter.getNamedAttr(
                          "operand_segment_sizes",
                          rewriter.getDenseI32ArrayAttr(
                              {1, static_cast<int32_t>(inputIndices.size()), 1,
                               1})),
                      rewriter.getNamedAttr(
                          "permutation_map",
                          AffineMapAttr::get(AffineMap::getMinorIdentityMap(
                              inputIndices.size(), 1,
                              rewriter.getContext())))});

              // Perform masked reduction
              auto tailReducedValue = tailBuilder.create<vector::ReductionOp>(
                  nestedLoc, vector::CombiningKind::ADD, tailReadValue,
                  /*acc=*/Value());

              // Store tail result
              tailBuilder.create<memref::StoreOp>(nestedLoc, tailReducedValue,
                                                  output, outputIndices);
            }
          } else {
            // Recursively create nested loops for non-reduction dimensions
            createNestedLoops(rewriter, loc, input, output, dims, elementType,
                              currentDim + 1, inputIndices, outputIndices,
                              index0, upperBound, zeroElementType, vectorSize,
                              mask, reductionDims, currentReductionDim);
          }

          nestedBuilder.create<affine::AffineYieldOp>(nestedLoc);
        });
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// ReduceSumVectorizationPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg pooling operations to mixture of
/// Affine + Vector operations.
namespace {
class ReduceSumVectorizationPass
    : public PassWrapper<ReduceSumVectorizationPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReduceSumVectorizationPass)
  StringRef getArgument() const final { return "reduce-sum-vectorize"; }
  StringRef getDescription() const final { return "Reduce Sum Vectorization."; }
  ReduceSumVectorizationPass() = default;
  ReduceSumVectorizationPass(const ReduceSumVectorizationPass &) {}
  explicit ReduceSumVectorizationPass(int64_t affineVectorSizeParam) {
    affineVectorSize = affineVectorSizeParam;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                           memref::MemRefDialect, VectorDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
    target.addLegalOp<linalg::FillOp>();

    RewritePatternSet patterns(context);
    patterns.add<ReduceSumVectorizationPattern>(context, affineVectorSize);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, affine::AffineDialect, VectorDialect>();
  }

  Option<int64_t> affineVectorSize{*this, "vector-size",
                                   llvm::cl::desc("Affine Vector size."),
                                   llvm::cl::init(16)};
};
} // end anonymous namespace.

namespace mlir {
namespace buddy {

std::unique_ptr<Pass> createReduceSumVectorizationPass() {
  return std::make_unique<ReduceSumVectorizationPass>();
}

void registerReduceSumVectorizationPass() {
  PassRegistration<ReduceSumVectorizationPass>();
}

} // namespace buddy
} // namespace mlir