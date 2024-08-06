//===- BatchMatMulOptimize.cpp --------------------------------------------===//
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
// This file implements the batchmatmul optimization.
//
//===----------------------------------------------------------------------===//
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

class BatchMatMulOptimizePattern : public ConversionPattern {
private:
  int64_t vecSize, kernelM, kernelN;

public:
  explicit BatchMatMulOptimizePattern(MLIRContext *context,
                                      int64_t vecSizeParam,
                                      int64_t kernelMParam,
                                      int64_t kernelNParam)
      : ConversionPattern(linalg::BatchMatmulOp::getOperationName(), 1,
                          context) {
    vecSize = vecSizeParam;
    kernelM = kernelMParam;
    kernelN = kernelNParam;
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
    ShapedType ATy = A.getType().cast<ShapedType>();

    // Define constants.
    const Value c0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value c1 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    const AffineExpr d0 = rewriter.getAffineDimExpr(0);
    const AffineExpr d1 = rewriter.getAffineDimExpr(1);
    const AffineExpr d2 = rewriter.getAffineDimExpr(2);
    const AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
    const AffineExpr s1 = rewriter.getAffineSymbolExpr(1);
    const AffineExpr s2 = rewriter.getAffineSymbolExpr(2);

    const AffineMap mapBroadcast =
        AffineMap::get(3, 0, rewriter.getAffineConstantExpr(0));
    const AffineExpr zeroAffine = rewriter.getAffineConstantExpr(0);
    const Value zeroElementType = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));

    // Get dimensions of input tensors.
    Value batch = rewriter.create<memref::DimOp>(loc, A, 0);
    Value M = rewriter.create<memref::DimOp>(loc, A, 1); // aRow
    Value K = rewriter.create<memref::DimOp>(loc, B, 1); // bRow
    Value N = rewriter.create<memref::DimOp>(loc, B, 2); // bCol

    SmallVector<Value, 4U> reducedValues = llvm::to_vector<4>(
        llvm::map_range(ArrayRef<LoopReduction>{},
                        [](const LoopReduction &red) { return red.value; }));

    // Configs
    int64_t kNLen = vecSize * kernelN;

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

    // Prefetching data from tensor 'A' for better cache utilization.
    rewriter.create<affine::AffinePrefetchOp>(
        loc, A, AffineMap::get(3, 0, {d0, d1, d2}, rewriter.getContext()),
        ArrayRef<Value>{loopVarBatchIdx, M, K}, false, 3, true);

    // build loop body
    affine::buildAffineLoopNest(
        rewriter, loc, {c0}, {N}, kNLen,
        [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
          auto ivJ = ivRange.front();
          affine::buildAffineLoopNest(
              builder, loc, {c0}, {M}, kernelM,
              [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
                Value ivI = ivRange.front();
                SmallVector<memref::SubViewOp> cptrs;

                const VectorType vTy =
                    VectorType::get(vecSize, ATy.getElementType());

                for (int i = 0; i < kernelM; i++) {
                  Value fixedIV = builder.create<affine::AffineMinOp>(
                      loc,
                      AffineMap::get(1, 1, {d0 + i, s0 - 1},
                                     builder.getContext()),
                      SmallVector<Value>{ivI, M});
                  MemRefType resTy = MemRefType::get(
                      ATy.getShape(), ATy.getElementType(),
                      AffineMap::get(3, 3, d1 * s2 + d0 * s1 + s0 + d2));
                  auto cptr = builder.create<memref::SubViewOp>(
                      loc, resTy, C,
                      SmallVector<OpFoldResult>{loopVarBatchIdx, fixedIV, c0},
                      SmallVector<OpFoldResult>{c1, c1, N},
                      SmallVector<OpFoldResult>{c1, c1, c1});
                  cptrs.push_back(cptr);
                }
                affine::buildAffineLoopNest(
                    builder, loc, {c0}, {K}, 1,
                    [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
                      Value ivK = ivRange.front();
                      SmallVector<Value> as;
                      SmallVector<Value> bs;
                      SmallVector<Value> ds;
                      for (int i = 0; i < kernelM; ++i) {
                        Value fixedIV = ivI;
                        if (i != 0) {
                          fixedIV = builder.create<affine::AffineMinOp>(
                              loc,
                              AffineMap::get(1, 1, {d0 + i, s0 - 1},
                                             builder.getContext()),
                              SmallVector<Value>{ivI, M});
                        }
                        Value ksubAElement = builder.create<memref::LoadOp>(
                            loc, A, ValueRange{loopVarBatchIdx, fixedIV, ivK});
                        as.push_back(
                            builder.create<SplatOp>(loc, vTy, ksubAElement));
                      }
                      for (int i = 0; i < kernelM; ++i) {
                        for (int j = 0; j < kernelN; j++) {
                          Value fixedIV = builder.create<affine::AffineApplyOp>(
                              loc, AffineMap::get(1, 0, d0 + j * vecSize), ivJ);
                          ds.push_back(builder.create<LoadOp>(
                              loc, vTy, cptrs[i], ValueRange{c0, c0, fixedIV}));
                        }
                      }

                      for (int i = 0; i < kernelN; i++) {
                        Value fixedIV = ivJ;
                        if (i != 0) {
                          fixedIV = builder.create<affine::AffineApplyOp>(
                              loc, AffineMap::get(1, 0, d0 + i * vecSize), ivJ);
                        }
                        bs.push_back(builder.create<LoadOp>(
                            loc, vTy, B,
                            ValueRange{loopVarBatchIdx, ivK, fixedIV}));
                      }

                      for (int i = 0; i < kernelM; i++) {
                        for (int j = 0; j < kernelN; j++) {
                          ds[i * kernelN + j] = builder.create<vector::FMAOp>(
                              loc, vTy, as[i], bs[j], ds[i * kernelN + j]);
                        }
                      }

                      for (int i = 0; i < kernelM; ++i) {
                        for (int j = 0; j < kernelN; ++j) {
                          Value fixedJV = builder.create<affine::AffineApplyOp>(
                              loc, AffineMap::get(1, 0, d0 + j * vecSize), ivJ);
                          Value tailLength =
                              builder.create<affine::AffineApplyOp>(
                                  loc, AffineMap::get(2, 0, -d0 + d1),
                                  ValueRange{fixedJV, N});
                          affine::AffineIfOp nBranchingOp =
                              builder.create<affine::AffineIfOp>(
                                  loc,
                                  IntegerSet::get(1, 0, {-vecSize + d0},
                                                  {false}),
                                  ValueRange{tailLength}, true);
                          // Calculate the length of the tail, which might not
                          // fit in a vector.
                          OpBuilder nTrueBranchBuilder =
                              nBranchingOp.getThenBodyBuilder();
                          nTrueBranchBuilder.create<StoreOp>(
                              loc, ds[i * kernelN + j], cptrs[i],
                              ValueRange{c0, c0, fixedJV});
                          OpBuilder nFalseBranchBuilder =
                              nBranchingOp.getElseBodyBuilder();
                          // Generate a mask vector based on the tail length.
                          Value maskVector =
                              nFalseBranchBuilder.create<vector::CreateMaskOp>(
                                  loc,
                                  VectorType::get({vecSize},
                                                  rewriter.getI1Type()),
                                  ValueRange{tailLength});
                          nFalseBranchBuilder.create<MaskedStoreOp>(
                              loc, cptrs[i], ValueRange{c0, c0, fixedJV},
                              maskVector, ds[i * kernelN + j]);
                        }
                      }
                    });
              });
        });

    rewriter.create<affine::AffineYieldOp>(loc);

    // Finalize the loop and erase the original operation.
    parallelBatchLoop.getRegion().push_back(loopBody);
    rewriter.setInsertionPointAfter(parallelBatchLoop);

    rewriter.eraseOp(op);
    return success();
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// BatchMatMulOptimizePass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg pooling operations to mixture of
/// Affine + Vector operations.
namespace {
class BatchMatMulOptimizePass
    : public PassWrapper<BatchMatMulOptimizePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BatchMatMulOptimizePass)
  StringRef getArgument() const final { return "batchmatmul-tile-optimize"; }
  StringRef getDescription() const final { return "BatchMatMul Optimization."; }
  BatchMatMulOptimizePass() = default;
  BatchMatMulOptimizePass(const BatchMatMulOptimizePass &) {}
  explicit BatchMatMulOptimizePass(int64_t vecSizeParam, int64_t kernelMParam,
                                   int64_t kernelNParam) {
    vecSize = vecSizeParam;
    kernelM = kernelMParam;
    kernelN = kernelNParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, VectorDialect>();
  }

  Option<int64_t> vecSize{*this, "vec-size",
                          llvm::cl::desc("Strip mining size."),
                          llvm::cl::init(16)};

  Option<int64_t> kernelM{*this, "kernel-m",
                          llvm::cl::desc("Strip mining size."),
                          llvm::cl::init(4)};

  Option<int64_t> kernelN{*this, "kernel-n",
                          llvm::cl::desc("Strip mining size."),
                          llvm::cl::init(2)};
};
} // end anonymous namespace.

void BatchMatMulOptimizePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                       scf::SCFDialect, memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<BatchMatMulOptimizePattern>(context, vecSize, kernelM, kernelN);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
// add to buddy-opt.cpp
namespace mlir {
namespace buddy {
void registerBatchMatMulOptimizePass() {
  PassRegistration<BatchMatMulOptimizePass>();
}
} // namespace buddy
} // namespace mlir
