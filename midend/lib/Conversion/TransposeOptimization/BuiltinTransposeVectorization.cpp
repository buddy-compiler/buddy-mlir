//===- BuiltinTransposeVectorization.cpp ----------------------------------===//
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
//
// This file implements the transpose optimization.
//
//===----------------------------------------------------------------------===//
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

class TransposeOptimizationPattern : public ConversionPattern {
public:
  explicit TransposeOptimizationPattern(MLIRContext *context,
                                        int64_t affineVectorSizeParam)
      : ConversionPattern(linalg::TransposeOp::getOperationName(), 1, context) {
    affineVectorSize = affineVectorSizeParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto permutationArrayAttr =
        op->getAttr(rewriter.getStringAttr("permutation"))
            .cast<DenseI64ArrayAttr>()
            .asArrayRef();

    // Retrieve input tensors A, B.
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);

    // Only to rewrite the rank 2 tensor transpose.
    if (permutationArrayAttr[0] != 1 or permutationArrayAttr[1] != 0 or
        A.getType().cast<MemRefType>().getRank() != 2) {
      return failure();
    }

    auto loc = op->getLoc();

    // Acquire the element type of input tensors.
    Type elementType = A.getType().cast<MemRefType>().getElementType();

    // Define constants.
    const Value index0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value indexVecSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(affineVectorSize));
    const AffineExpr d0 = rewriter.getAffineDimExpr(0);
    const AffineExpr d1 = rewriter.getAffineDimExpr(1);
    const AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
    const AffineExpr zeroAffine = rewriter.getAffineConstantExpr(0);

    const Value zeroElementType = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));

    // Get dimensions of input tensor.
    Value Row = rewriter.create<memref::DimOp>(loc, A, 0);
    Value Col = rewriter.create<memref::DimOp>(loc, A, 1);

    // Calculate the length of the tail, which might not fit in a vector.
    Value rowUnalignedLength = rewriter.create<affine::AffineApplyOp>(
        loc, AffineMap::get(1, 0, d0 % affineVectorSize), ValueRange{Row});
    Value colUnalignedLength = rewriter.create<affine::AffineApplyOp>(
        loc, AffineMap::get(1, 0, d0 % affineVectorSize), ValueRange{Col});
    Value rowUpperBound = rewriter.create<affine::AffineApplyOp>(
        loc,
        AffineMap::get(1, 0, d0.floorDiv(affineVectorSize) * affineVectorSize),
        ValueRange{Row});

    // Generate a mask vector based on the tail length.
    Value rowEndMaskLoad = rewriter.create<vector::CreateMaskOp>(
        loc,
        VectorType::get({affineVectorSize, affineVectorSize},
                        rewriter.getI1Type()),
        ValueRange{rowUnalignedLength, indexVecSize});
    Value colEndMaskLoad = rewriter.create<vector::CreateMaskOp>(
        loc,
        VectorType::get({affineVectorSize, affineVectorSize},
                        rewriter.getI1Type()),
        ValueRange{indexVecSize, colUnalignedLength});
    Value rowEndMaskStore = rewriter.create<vector::CreateMaskOp>(
        loc,
        VectorType::get({affineVectorSize, affineVectorSize},
                        rewriter.getI1Type()),
        ValueRange{indexVecSize, rowUnalignedLength});
    Value colEndMaskStore = rewriter.create<vector::CreateMaskOp>(
        loc,
        VectorType::get({affineVectorSize, affineVectorSize},
                        rewriter.getI1Type()),
        ValueRange{colUnalignedLength, indexVecSize});

    SmallVector<Value, 4U> reducedValues = llvm::to_vector<4>(
        llvm::map_range(ArrayRef<LoopReduction>{},
                        [](const LoopReduction &red) { return red.value; }));

    // Create the primary parallel loop.
    AffineParallelOp parallelColLoop =
        rewriter.create<affine::AffineParallelOp>(
            loc, ValueRange(reducedValues).getTypes(), ValueRange{Col},
            ArrayRef<NamedAttribute>{
                rewriter.getNamedAttr("lowerBoundsGroups",
                                      rewriter.getI32TensorAttr({1})),
                rewriter.getNamedAttr("upperBoundsGroups",
                                      rewriter.getI32TensorAttr({1})),
                rewriter.getNamedAttr(
                    "lowerBoundsMap",
                    AffineMapAttr::get(AffineMap::get(0, 0, {zeroAffine},
                                                      rewriter.getContext()))),
                rewriter.getNamedAttr(
                    "upperBoundsMap",
                    AffineMapAttr::get(AffineMap::get(
                        0, 1,
                        {s0.floorDiv(affineVectorSize) * affineVectorSize},
                        rewriter.getContext()))),
                rewriter.getNamedAttr("reductions", rewriter.getArrayAttr({})),
                rewriter.getNamedAttr(
                    "steps", rewriter.getI64ArrayAttr({affineVectorSize}))});

    // Create the loop body for the parallel loop.
    Block *loopBody = new Block();
    rewriter.setInsertionPointToStart(loopBody);
    loopBody->addArgument(rewriter.getIndexType(), loc);
    Value colIdx = loopBody->getArguments()[0];

    affine::buildAffineLoopNest(
        rewriter, loc, {index0}, {rowUpperBound}, affineVectorSize,
        [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
          Value rowIdx = ivRange.front();

          auto tiledMatrix = rewriter.create<vector::TransferReadOp>(
              loc,
              TypeRange{VectorType::get(
                  ArrayRef<int64_t>{affineVectorSize, affineVectorSize},
                  elementType)},
              ValueRange{A, rowIdx, colIdx, zeroElementType},
              ArrayRef<NamedAttribute>{
                  rewriter.getNamedAttr(
                      "in_bounds",
                      rewriter.getBoolArrayAttr(ArrayRef<bool>{false, true})),
                  rewriter.getNamedAttr(
                      "operand_segment_sizes",
                      rewriter.getDenseI32ArrayAttr(ArrayRef<int>{1, 2, 1, 0})),
                  rewriter.getNamedAttr(
                      "permutation_map",
                      AffineMapAttr::get(AffineMap::get(
                          2, 0, {d0, d1}, rewriter.getContext()))),
              });

          rewriter.create<vector::TransferWriteOp>(
              loc, TypeRange{}, ValueRange{tiledMatrix, B, colIdx, rowIdx},
              ArrayRef<NamedAttribute>{
                  rewriter.getNamedAttr(
                      "in_bounds",
                      rewriter.getBoolArrayAttr(ArrayRef<bool>{true, true})),
                  rewriter.getNamedAttr(
                      "operand_segment_sizes",
                      rewriter.getDenseI32ArrayAttr(ArrayRef<int>{1, 1, 2, 0})),
                  rewriter.getNamedAttr(
                      "permutation_map",
                      AffineMapAttr::get(AffineMap::get(2, 0, {d1, d0},
                                                        builder.getContext()))),
              });

          // Compile time branch detection.
          if (A.getType().cast<MemRefType>().isDynamicDim(0) or
              A.getType().cast<MemRefType>().getDimSize(0) % affineVectorSize !=
                  0) {
            // Depending on the position, use either full vectors or tail
            // vectors.
            affine::AffineIfOp branchingRowUnaligned =
                builder.create<affine::AffineIfOp>(
                    loc,
                    IntegerSet::get(1, 0, {d0 % affineVectorSize - 1}, {false}),
                    ValueRange{Row}, false);

            // Branch handling unaligned rows.
            OpBuilder trueRowUnalignedBranchBuilder =
                branchingRowUnaligned.getThenBodyBuilder();

            auto rowUnalignedTiledMatrix =
                trueRowUnalignedBranchBuilder.create<vector::TransferReadOp>(
                    loc,
                    TypeRange{VectorType::get(
                        ArrayRef<int64_t>{affineVectorSize, affineVectorSize},
                        elementType)},
                    ValueRange{A, rowUpperBound, colIdx, zeroElementType,
                               rowEndMaskLoad},
                    ArrayRef<NamedAttribute>{
                        rewriter.getNamedAttr("in_bounds",
                                              rewriter.getBoolArrayAttr(
                                                  ArrayRef<bool>{false, true})),
                        rewriter.getNamedAttr("operand_segment_sizes",
                                              rewriter.getDenseI32ArrayAttr(
                                                  ArrayRef<int>{1, 2, 1, 1})),
                        rewriter.getNamedAttr(
                            "permutation_map",
                            AffineMapAttr::get(AffineMap::get(
                                2, 0, {d0, d1}, rewriter.getContext()))),
                    });
            trueRowUnalignedBranchBuilder.create<vector::TransferWriteOp>(
                loc, TypeRange{},
                ValueRange{rowUnalignedTiledMatrix, B, colIdx, rowUpperBound,
                           rowEndMaskStore},
                ArrayRef<NamedAttribute>{
                    rewriter.getNamedAttr(
                        "in_bounds",
                        rewriter.getBoolArrayAttr(ArrayRef<bool>{true, false})),
                    rewriter.getNamedAttr("operand_segment_sizes",
                                          rewriter.getDenseI32ArrayAttr(
                                              ArrayRef<int>{1, 1, 2, 1})),
                    rewriter.getNamedAttr(
                        "permutation_map",
                        AffineMapAttr::get(AffineMap::get(
                            2, 0, {d1, d0}, builder.getContext()))),
                });
          }
        });

    rewriter.create<affine::AffineYieldOp>(loc);

    // Finalize the loop.
    parallelColLoop.getRegion().push_back(loopBody);
    rewriter.setInsertionPointAfter(parallelColLoop);

    if (A.getType().cast<MemRefType>().isDynamicDim(1) or
        A.getType().cast<MemRefType>().getDimSize(1) % affineVectorSize != 0) {

      affine::AffineIfOp branchingColUnaligned =
          rewriter.create<affine::AffineIfOp>(
              loc, IntegerSet::get(1, 0, {d0 % affineVectorSize - 1}, {false}),
              ValueRange{Col}, false);

      OpBuilder trueColUnalignedBranchBuilder =
          branchingColUnaligned.getThenBodyBuilder();
      Value colUpperBound =
          trueColUnalignedBranchBuilder.create<affine::AffineApplyOp>(
              loc,
              AffineMap::get(1, 0,
                             d0.floorDiv(affineVectorSize) * affineVectorSize),
              ValueRange{Col});

      affine::buildAffineLoopNest(
          trueColUnalignedBranchBuilder, loc, {index0}, {rowUpperBound},
          affineVectorSize,
          [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
            Value rowIdx = ivRange.front();
            auto colUnalignedTiledMatrix =
                builder.create<vector::TransferReadOp>(
                    loc,
                    TypeRange{VectorType::get(
                        ArrayRef<int64_t>{affineVectorSize, affineVectorSize},
                        elementType)},
                    ValueRange{A, rowIdx, colUpperBound, zeroElementType,
                               colEndMaskLoad},
                    ArrayRef<NamedAttribute>{
                        builder.getNamedAttr("in_bounds",
                                             builder.getBoolArrayAttr(
                                                 ArrayRef<bool>{false, false})),
                        builder.getNamedAttr("operand_segment_sizes",
                                             builder.getDenseI32ArrayAttr(
                                                 ArrayRef<int>{1, 2, 1, 1})),
                        builder.getNamedAttr(
                            "permutation_map",
                            AffineMapAttr::get(AffineMap::get(
                                2, 0, {d0, d1}, builder.getContext()))),
                    });
            builder.create<vector::TransferWriteOp>(
                loc, TypeRange{},
                ValueRange{colUnalignedTiledMatrix, B, colUpperBound, rowIdx,
                           colEndMaskStore},
                ArrayRef<NamedAttribute>{
                    rewriter.getNamedAttr(
                        "in_bounds",
                        rewriter.getBoolArrayAttr(ArrayRef<bool>{false, true})),
                    rewriter.getNamedAttr("operand_segment_sizes",
                                          rewriter.getDenseI32ArrayAttr(
                                              ArrayRef<int>{1, 1, 2, 1})),
                    rewriter.getNamedAttr(
                        "permutation_map",
                        AffineMapAttr::get(AffineMap::get(
                            2, 0, {d1, d0}, builder.getContext()))),
                });
          });

      if (A.getType().cast<MemRefType>().isDynamicDim(0) or
          A.getType().cast<MemRefType>().getDimSize(0) % affineVectorSize !=
              0) {
        affine::AffineIfOp branchingRowColUnaligned =
            trueColUnalignedBranchBuilder.create<affine::AffineIfOp>(
                loc,
                IntegerSet::get(1, 0, {d0 % affineVectorSize - 1}, {false}),
                ValueRange{Col}, false);

        OpBuilder trueRowColUnalignedBranchBuilder =
            branchingRowColUnaligned.getThenBodyBuilder();
        Value rowColEndMaskLoad =
            trueRowColUnalignedBranchBuilder.create<vector::CreateMaskOp>(
                loc,
                VectorType::get({affineVectorSize, affineVectorSize},
                                trueRowColUnalignedBranchBuilder.getI1Type()),
                ValueRange{rowUnalignedLength, colUnalignedLength});
        Value rowColEndMaskStore =
            trueRowColUnalignedBranchBuilder.create<vector::CreateMaskOp>(
                loc,
                VectorType::get({affineVectorSize, affineVectorSize},
                                trueRowColUnalignedBranchBuilder.getI1Type()),
                ValueRange{colUnalignedLength, rowUnalignedLength});
        auto rowColUnalignedTiledMatrix =
            trueRowColUnalignedBranchBuilder.create<vector::TransferReadOp>(
                loc,
                TypeRange{VectorType::get(
                    ArrayRef<int64_t>{affineVectorSize, affineVectorSize},
                    elementType)},
                ValueRange{A, rowUpperBound, colUpperBound, zeroElementType,
                           rowColEndMaskLoad},
                ArrayRef<NamedAttribute>{
                    trueRowColUnalignedBranchBuilder.getNamedAttr(
                        "in_bounds",
                        trueRowColUnalignedBranchBuilder.getBoolArrayAttr(
                            ArrayRef<bool>{false, false})),
                    trueRowColUnalignedBranchBuilder.getNamedAttr(
                        "operand_segment_sizes",
                        trueRowColUnalignedBranchBuilder.getDenseI32ArrayAttr(
                            ArrayRef<int>{1, 2, 1, 1})),
                    trueRowColUnalignedBranchBuilder.getNamedAttr(
                        "permutation_map",
                        AffineMapAttr::get(AffineMap::get(
                            2, 0, {d0, d1},
                            trueRowColUnalignedBranchBuilder.getContext()))),
                });
        trueRowColUnalignedBranchBuilder.create<vector::TransferWriteOp>(
            loc, TypeRange{},
            ValueRange{rowColUnalignedTiledMatrix, B, colUpperBound,
                       rowUpperBound, rowColEndMaskStore},
            ArrayRef<NamedAttribute>{
                rewriter.getNamedAttr(
                    "in_bounds",
                    rewriter.getBoolArrayAttr(ArrayRef<bool>{false, false})),
                rewriter.getNamedAttr(
                    "operand_segment_sizes",
                    rewriter.getDenseI32ArrayAttr(ArrayRef<int>{1, 1, 2, 1})),
                rewriter.getNamedAttr(
                    "permutation_map",
                    AffineMapAttr::get(AffineMap::get(
                        2, 0, {d1, d0},
                        trueRowColUnalignedBranchBuilder.getContext()))),
            });
      }
    }

    // Erase the original operation.
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t affineVectorSize;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// TransposeOptimizationPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg pooling operations to mixture of
/// Affine + Vector operations.
namespace {
class TransposeOptimizationPass
    : public PassWrapper<TransposeOptimizationPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransposeOptimizationPass)
  StringRef getArgument() const final { return "transpose-optimize"; }
  StringRef getDescription() const final {
    return "Transpose Optimization only for rank 2 tensor.";
  }
  TransposeOptimizationPass() = default;
  TransposeOptimizationPass(const TransposeOptimizationPass &) {}
  explicit TransposeOptimizationPass(int64_t affineVectorSizeParam) {
    affineVectorSize = affineVectorSizeParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, affine::AffineDialect, VectorDialect>();
  }

  Option<int64_t> affineVectorSize{*this, "vector-size",
                                   llvm::cl::desc("Affine Vector size."),
                                   llvm::cl::init(16)};
};
} // end anonymous namespace.

void TransposeOptimizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                         memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<TransposeOptimizationPattern>(context, affineVectorSize);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerTransposeOptimizationPass() {
  PassRegistration<TransposeOptimizationPass>();
}
} // namespace buddy
} // namespace mlir
