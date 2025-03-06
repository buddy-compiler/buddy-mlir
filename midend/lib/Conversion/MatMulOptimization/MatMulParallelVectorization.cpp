//===- MatMulParallelVectorization.cpp ------------------------------------===//
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
// This file implements the matmul-parallel-vectorization optimization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
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

class MatMulParallelVectorizationPattern : public ConversionPattern {
public:
  explicit MatMulParallelVectorizationPattern(MLIRContext *context,
                                              int64_t affineVectorSizeParam,
                                              int64_t kBlockSizeParam)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), 1, context) {
    affineVectorSize = affineVectorSizeParam;
    kBlockSize = kBlockSizeParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // Retrieve input tensors A, B, and C.
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);

    // Acquire the element type of input tensors.
    Type elementType = A.getType().cast<MemRefType>().getElementType();

    // Define constants.
    const Value zeroIndex =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value oneIndex = 
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    const AffineExpr d0 = rewriter.getAffineDimExpr(0);
    const AffineExpr d1 = rewriter.getAffineDimExpr(1);
    const AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
    const AffineExpr zeroAffine = rewriter.getAffineConstantExpr(0);

    const Value zeroElementType = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    const Value zeroElementTypeVec = rewriter.create<vector::SplatOp>(
        loc, VectorType::get({affineVectorSize}, elementType), zeroElementType);

    // Get dimensions of input tensors.
    Value aRow = rewriter.create<memref::DimOp>(loc, A, zeroIndex);
    Value bCol = rewriter.create<memref::DimOp>(loc, B, oneIndex);
    Value bRow = rewriter.create<memref::DimOp>(loc, B, zeroIndex);

    SmallVector<Value, 4U> reducedValues = llvm::to_vector<4>(
        llvm::map_range(ArrayRef<LoopReduction>{},
                        [](const LoopReduction &red) { return red.value; }));

    // Apply the column of matrix B
    Value appliedColOfB = rewriter.create<affine::AffineApplyOp>(
                          loc, AffineMap::get(1, 0, d0 - affineVectorSize + 1),
                          ValueRange{bCol});                        

    // Create the primary parallel loop for matrix multiplication.
    AffineParallelOp parallelLoop = rewriter.create<affine::AffineParallelOp>(
        loc, ValueRange(reducedValues).getTypes(), ValueRange{appliedColOfB},
        ArrayRef<NamedAttribute>{
            rewriter.getNamedAttr("lowerBoundsGroups",
                                  rewriter.getI32TensorAttr({1})),
            rewriter.getNamedAttr("upperBoundsGroups",
                                  rewriter.getI32TensorAttr({1})),
            rewriter.getNamedAttr(
                "lowerBoundsMap",
                AffineMapAttr::get(
                    AffineMap::get(0, 0, {zeroAffine}, rewriter.getContext()))),
            rewriter.getNamedAttr("upperBoundsMap",
                                  AffineMapAttr::get(AffineMap::get(
                                      1, 0, {d0}, rewriter.getContext()))),
            rewriter.getNamedAttr("reductions", rewriter.getArrayAttr({})),
            rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr({affineVectorSize}))}); 

    // Create the loop body for the parallel loop.
    Block *loopBody = new Block();
    rewriter.setInsertionPointToStart(loopBody);
    loopBody->addArgument(rewriter.getIndexType(), loc);
    Value loopVarColOfB = loopBody->getArguments()[0];

    // Prefetching data from tensor 'A' for better cache utilization.
    rewriter.create<affine::AffinePrefetchOp>(
        loc, A, AffineMap::get(2, 0, {d0, d1}, rewriter.getContext()),
        ArrayRef<Value>{aRow, bRow}, false, 3, true);

    affine::buildAffineLoopNest(
        rewriter, loc, {zeroIndex}, {bRow}, /*Step=*/{kBlockSize},
        [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
            Value kLow = ivRange.front();
            Value kHigh = builder.create<affine::AffineMinOp>(
                loc,
                AffineMap::get(1,1,{d0 + kBlockSize, s0},
                                builder.getContext()),
                SmallVector<Value>{kLow, bRow});
            affine::buildAffineLoopNest(
                builder, loc, {zeroIndex}, {aRow}, 1,
                [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
                    Value loopVarRowOfA = ivRange.front();
                    Value cVec = builder.create<affine::AffineVectorLoadOp>(
                        loc, VectorType::get({affineVectorSize}, elementType), C,
                        AffineMap::get(2, 0, {d0, d1},
                                       rewriter.getContext()),
                        ValueRange{loopVarRowOfA, loopVarColOfB});
                    auto iter_vec = builder.create<scf::ForOp>(
                        loc, kLow, kHigh, /*Step=*/oneIndex, ValueRange{cVec},
                        [&](OpBuilder &builder, Location loc, Value iv1,
                            ValueRange itrArgs0){
                                Value bVec = builder.create<vector::LoadOp>(
                                    loc, VectorType::get({affineVectorSize}, elementType), B,
                                    ValueRange{iv1, loopVarColOfB});
                                Value aElement = builder.create<memref::LoadOp>(
                                        loc, A, ValueRange{loopVarRowOfA, iv1});
                                Value aVec = builder.create<vector::BroadcastOp>(
                                        loc, VectorType::get({affineVectorSize}, elementType),
                                        aElement);
                                Value computedVec;

                                // Compute the result vector either through integer
                                // multiplication and addition or fused multiply-add
                                // based on the element type.
                                if (isa<IntegerType>(elementType)) {
                                    Value mulVec =
                                        builder.create<arith::MulIOp>(loc, aVec, bVec);
                                    computedVec =
                                        builder.create<arith::AddIOp>(loc, mulVec, itrArgs0[0]);
                                  } else {
                                    computedVec =
                                        builder.create<vector::FMAOp>(loc, aVec, bVec, itrArgs0[0]);
                                  }
                                  builder.create<scf::YieldOp>(loc, computedVec);
                            });
                            builder.create<vector::StoreOp>(
                                loc, iter_vec.getResult(0), C,
                                ValueRange{loopVarRowOfA, loopVarColOfB});
            });
        });
        rewriter.create<affine::AffineYieldOp>(loc);

        // Finalize the loop and erase the original operation.
        parallelLoop.getRegion().push_back(loopBody);
        rewriter.setInsertionPointAfter(parallelLoop);       

    // Compile time branch detection.
    if (C.getType().cast<MemRefType>().isDynamicDim(1) or
        C.getType().cast<MemRefType>().getDimSize(1) % affineVectorSize != 0) {
            
      // Depending on the position, use either full vectors or tail vectors.
      affine::AffineIfOp branchingOp = rewriter.create<affine::AffineIfOp>(
          loc,
          IntegerSet::get(
              1, 0, {d0 % affineVectorSize - 1}, {false}),
          ValueRange{bCol}, /*hasElse=*/false);

      // Branch handling operations on the tail.
      OpBuilder trueBranchBuilder = branchingOp.getThenBodyBuilder();
      Value tailSize = trueBranchBuilder.create<affine::AffineApplyOp>(
        loc, AffineMap::get(1, 0, d0 % affineVectorSize), ValueRange{bCol});
      Value maskVector = trueBranchBuilder.create<vector::CreateMaskOp>(
        loc, VectorType::get({affineVectorSize}, rewriter.getI1Type()),
        ValueRange{tailSize});
      Value loopVarColOfBTail = trueBranchBuilder.create<arith::SubIOp>(loc, bCol, tailSize);

      affine::buildAffineLoopNest(
        trueBranchBuilder, loc, {zeroIndex}, {bRow}, {kBlockSize},
        [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
            Value kLow = ivRange.front();
            Value kHigh = builder.create<affine::AffineMinOp>(
                loc, 
                AffineMap::get(1,1,{d0 + kBlockSize, s0},
                    builder.getContext()),
                SmallVector<Value>{kLow, bRow});
            affine::buildAffineLoopNest(
                builder, loc, {zeroIndex}, {aRow}, 1,
                [&](OpBuilder &builder, Location loc, ValueRange ivRange){
                    Value loopVarRowOfA = ivRange.front();
                    Value cVec = builder.create<vector::MaskedLoadOp>(
                        loc, VectorType::get({affineVectorSize}, elementType), C,
                        ValueRange{loopVarRowOfA, loopVarColOfBTail}, maskVector,
                        zeroElementTypeVec);
                    auto iter_vec = builder.create<scf::ForOp>(
                        loc, kLow, kHigh, /*Step=*/oneIndex, ValueRange{cVec},
                        [&](OpBuilder &builder, Location loc, Value iv1,
                        ValueRange itrArgs0){
                            Value bVec = builder.create<vector::MaskedLoadOp>(
                                loc, VectorType::get({affineVectorSize}, elementType), B,
                                ValueRange{iv1, loopVarColOfBTail}, maskVector,
                                zeroElementTypeVec);
                                
                            Value aElement = builder.create<memref::LoadOp>(
                                loc, A, ValueRange{loopVarRowOfA, iv1});
                            Value aVec = builder.create<vector::BroadcastOp>(
                                loc, VectorType::get({affineVectorSize}, elementType),
                                aElement);
                            Value computedVec;
          
                            // Compute the result vector either through integer
                            // multiplication and addition or fused multiply-add
                            // based on the element type.
                            if (isa<IntegerType>(elementType)) {
                              Value mulVec =
                                  builder.create<arith::MulIOp>(loc, aVec, bVec);
                              computedVec =
                                  builder.create<arith::AddIOp>(loc, mulVec, itrArgs0[0]);
                            } else {
                              computedVec =
                                  builder.create<vector::FMAOp>(loc, aVec, bVec, itrArgs0[0]);
                            }
                            builder.create<scf::YieldOp>(loc, computedVec);
                        }); 
                        builder.create<vector::MaskedStoreOp>(
                            loc, C, ValueRange{loopVarRowOfA, loopVarColOfBTail},
                            maskVector, iter_vec.getResult(0));
                    });
                });
            }
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t affineVectorSize;
  int64_t kBlockSize;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// MatMulParallelVectorizationPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg pooling operations to mixture of
/// Affine + Vector operations.
namespace {
class MatMulParallelVectorizationPass
    : public PassWrapper<MatMulParallelVectorizationPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulParallelVectorizationPass)
  StringRef getArgument() const final {
    return "matmul-parallel-vectorization-optimize";
  }
  StringRef getDescription() const final {
    return "MatMulParallelVectorization Optimization.";
  }
  MatMulParallelVectorizationPass() = default;
  MatMulParallelVectorizationPass(const MatMulParallelVectorizationPass &) {}
  explicit MatMulParallelVectorizationPass(int64_t affineVectorSizeParam, int64_t kBlockSizeParam) {
    affineVectorSize = affineVectorSizeParam;
    kBlockSize = kBlockSizeParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, VectorDialect>();
  }

  Option<int64_t> affineVectorSize{*this, "vector-size",
                                   llvm::cl::desc("Affine Vector size."),
                                   llvm::cl::init(64)};

  Option<int64_t> kBlockSize{*this, "k-block-size",
                                    llvm::cl::desc("K block size."),
                                    llvm::cl::init(32)};
};
} // end anonymous namespace.

void MatMulParallelVectorizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                       scf::SCFDialect, memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<MatMulParallelVectorizationPattern>(context, affineVectorSize, kBlockSize);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerMatMulParallelVectorizationPass() {
  PassRegistration<MatMulParallelVectorizationPass>();
}
} // namespace buddy
} // namespace mlir