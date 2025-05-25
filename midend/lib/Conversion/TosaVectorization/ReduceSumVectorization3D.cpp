//===- ReduceSumVectorization3D.cpp ----------------------------------===//
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
// This file implements the reduce sum vectorization for 3D tensors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
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

class ReduceSumVectorization3DPattern : public ConversionPattern {
public:
  explicit ReduceSumVectorization3DPattern(MLIRContext *context,
                                           int64_t affineVectorSizeParam)
      : ConversionPattern(linalg::ReduceOp::getOperationName(), 1, context),
        affineVectorSize(affineVectorSizeParam) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto reduceOp = cast<linalg::ReduceOp>(op);

    // Check if it's a 3D to 2D reduction
    if (!reduceOp.getOperand(0).getType().isa<MemRefType>() ||
        !reduceOp.getOperand(1).getType().isa<MemRefType>())
      return failure();

    auto inputType = reduceOp.getOperand(0).getType().cast<MemRefType>();
    auto outputType = reduceOp.getOperand(1).getType().cast<MemRefType>();

    // Verify dimensions
    if (inputType.getRank() != 3 || outputType.getRank() != 2)
      return failure();

    // Get input and output
    auto input = reduceOp.getOperand(0);
    auto output = reduceOp.getOperand(1);
    auto loc = op->getLoc();

    // Get element type of input tensor
    Type elementType = inputType.getElementType();

    // Define constants
    const Value index0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value indexVecSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(affineVectorSize));
    // const Value c8 =
    //     rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(8));
    // const Value c1 =
    //     rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    const Value zeroFloat = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));

    // Get input tensor dimensions
    Value dim0 = rewriter.create<memref::DimOp>(loc, input, 0);
    Value dim1 = rewriter.create<memref::DimOp>(loc, input, 1);
    Value dim2 = rewriter.create<memref::DimOp>(loc, input, 2);

    // Outer loop - first dimension
    affine::buildAffineLoopNest(
        rewriter, loc, {index0}, {dim0}, 1,
        [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
          Value i0 = ivRange.front();

          // Middle loop - second dimension, step 8
          affine::buildAffineLoopNest(
              builder, loc, {index0}, {dim1}, 8,
              [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
                Value j0 = ivRange.front();

                // Create parallel op to process 8 blocks
                SmallVector<Value, 4U> reducedValues =
                    llvm::to_vector<4>(llvm::map_range(
                        ArrayRef<LoopReduction>{},
                        [](const LoopReduction &red) { return red.value; }));

                AffineParallelOp parallelOp =
                    builder.create<affine::AffineParallelOp>(
                        loc, ValueRange(reducedValues).getTypes(), ValueRange{},
                        ArrayRef<NamedAttribute>{
                            builder.getNamedAttr("lowerBoundsGroups",
                                                 builder.getI32TensorAttr({1})),
                            builder.getNamedAttr("upperBoundsGroups",
                                                 builder.getI32TensorAttr({1})),
                            builder.getNamedAttr(
                                "lowerBoundsMap",
                                AffineMapAttr::get(AffineMap::get(
                                    0, 0, {builder.getAffineConstantExpr(0)},
                                    builder.getContext()))),
                            builder.getNamedAttr(
                                "upperBoundsMap",
                                AffineMapAttr::get(AffineMap::get(
                                    0, 0, {builder.getAffineConstantExpr(8)},
                                    builder.getContext()))),
                            builder.getNamedAttr("steps",
                                                 builder.getI64ArrayAttr({1})),
                            builder.getNamedAttr("reductions",
                                                 builder.getArrayAttr({}))});

                // Create parallel block body
                Block *parallelBody = new Block();
                builder.setInsertionPointToStart(parallelBody);
                parallelBody->addArgument(builder.getIndexType(), loc);
                Value idx = parallelBody->getArguments()[0];

                // Calculate actual j index
                Value j = builder.create<arith::AddIOp>(loc, j0, idx);

                // Check if j is within valid range
                Value j_in_range = builder.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::slt, j, dim1);

                builder.create<scf::IfOp>(
                    loc, j_in_range, [&](OpBuilder &builder, Location loc) {
                      // Initialize accumulator
                      Value acc = builder.create<arith::ConstantOp>(
                          loc, builder.getZeroAttr(elementType));

                      // Vectorized reduction in the innermost dimension
                      auto lbMap = AffineMap::get(
                          /*dimCount=*/0, /*symbolCount=*/0,
                          builder.getAffineConstantExpr(0),
                          builder.getContext());
                      auto ubMap = AffineMap::get(
                          /*dimCount=*/1, /*symbolCount=*/0,
                          builder.getAffineDimExpr(0), builder.getContext());

                      affine::AffineForOp reductionLoop = builder.create<
                          affine::AffineForOp>(
                          loc,
                          /*lbOperands=*/ValueRange{},
                          /*lbMap=*/lbMap,
                          /*ubOperands=*/ValueRange{dim2},
                          /*ubMap=*/ubMap,
                          /*step=*/affineVectorSize,
                          /*iterArgs=*/ValueRange{acc},
                          [&](OpBuilder &builder, Location loc, Value iv,
                              ValueRange iterArgs) {
                            Value curr_acc = iterArgs[0];

                            // Prefetch next data block
                            Value next_k = builder.create<arith::AddIOp>(
                                loc, iv, indexVecSize);
                            Value next_valid = builder.create<arith::CmpIOp>(
                                loc, arith::CmpIPredicate::slt, next_k, dim2);

                            builder.create<scf::IfOp>(
                                loc, next_valid,
                                [&](OpBuilder &builder, Location loc) {
                                  builder.create<memref::PrefetchOp>(
                                      loc, input, ValueRange{i0, j, next_k},
                                      /*isWrite=*/false,
                                      /*locality=*/3,
                                      /*isDataCache=*/true);
                                  builder.create<scf::YieldOp>(loc);
                                });

                            // Calculate current block size and mask
                            Value remaining =
                                builder.create<arith::SubIOp>(loc, dim2, iv);
                            Value vl = builder.create<arith::MinSIOp>(
                                loc, remaining, indexVecSize);
                            Value mask = builder.create<vector::CreateMaskOp>(
                                loc,
                                VectorType::get({(int64_t)affineVectorSize},
                                                builder.getI1Type()),
                                ValueRange{vl});

                            // Vectorized read
                            auto vecType = VectorType::get(
                                {(int64_t)affineVectorSize}, elementType);
                            auto map = AffineMap::get(
                                /*dimCount=*/3, // 3D输入
                                /*symbolCount=*/0,
                                {rewriter.getAffineDimExpr(2)}, // 只映射k维度
                                rewriter.getContext());
                            Value vec = builder.create<vector::TransferReadOp>(
                                loc, vecType, input, ValueRange{i0, j, iv}, map,
                                zeroFloat, mask,
                                ArrayAttr::get(builder.getContext(),
                                               {builder.getBoolAttr(false)}));

                            // Vector reduction sum
                            Value block_sum =
                                builder.create<vector::ReductionOp>(
                                    loc, vector::CombiningKind::ADD, vec);

                            // Update accumulator
                            Value next_acc = builder.create<arith::AddFOp>(
                                loc, curr_acc, block_sum);

                            builder.create<affine::AffineYieldOp>(loc,
                                                                  next_acc);
                          });

                      // Store result
                      builder.create<memref::StoreOp>(
                          loc, reductionLoop.getResult(0), output,
                          ValueRange{i0, j});

                      builder.create<scf::YieldOp>(loc);
                    });

                builder.create<affine::AffineYieldOp>(loc);
                parallelOp.getRegion().push_back(parallelBody);
              });
        });

    // Remove original operation
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t affineVectorSize;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ReduceSumVectorizationPass
//===----------------------------------------------------------------------===//

namespace {
class ReduceSumVectorizationPass
    : public PassWrapper<ReduceSumVectorizationPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReduceSumVectorizationPass)

  StringRef getArgument() const final { return "reduce-sum-vectorization-3d"; }

  StringRef getDescription() const final {
    return "Reduce Sum Vectorization for 3D tensors.";
  }

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
                           memref::MemRefDialect, VectorDialect,
                           scf::SCFDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp, linalg::FillOp>();
    RewritePatternSet patterns(context);
    patterns.add<ReduceSumVectorization3DPattern>(context, affineVectorSize);
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, affine::AffineDialect, VectorDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
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
