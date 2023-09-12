//===- BatchMatMulOptimize.cpp
//-------------------------------------------------===//
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
#include "mlir/IR/TypeRange.h"
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

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class BatchMatMulOptimizePattern : public ConversionPattern {
public:
  explicit BatchMatMulOptimizePattern(MLIRContext *context,
                                      int64_t affineVectorSizeParam)
      : ConversionPattern(linalg::BatchMatmulOp::getOperationName(), 1,
                          context) {
    affineVectorSize = affineVectorSizeParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // Get input A, B, C.
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);
    // Get ElementType of input and output.
    auto A_elementType = A.getType().cast<MemRefType>().getElementType();

    // Some constants.
    const Value c0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value step = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(affineVectorSize));
    const AffineExpr d0 = rewriter.getAffineDimExpr(0);
    const AffineExpr d1 = rewriter.getAffineDimExpr(1);
    const AffineExpr d2 = rewriter.getAffineDimExpr(2);
    const AffineExpr c0_affine = rewriter.getAffineConstantExpr(0);

    const Value c0_dynamicType = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(A_elementType));
    const Value c0_dynamicType_vec = rewriter.create<vector::SplatOp>(
        loc, VectorType::get({affineVectorSize}, A_elementType), c0_dynamicType);

    // Dims
    Value BATCH = rewriter.create<memref::DimOp>(loc, A, 0); // Batch size
    Value M = rewriter.create<memref::DimOp>(loc, A, 1);     // A row
    Value N = rewriter.create<memref::DimOp>(loc, B, 2);     // B col
    Value K = rewriter.create<memref::DimOp>(loc, B, 1);     // B row

    auto reducedValues = llvm::to_vector<4>(llvm::map_range(
        ArrayRef<mlir::affine::LoopReduction>{},
        [](const mlir::affine::LoopReduction &red) { return red.value; }));

    // Build parallel loop body.
    auto parallelLoop = rewriter.create<affine::AffineParallelOp>(
        loc, ValueRange(reducedValues).getTypes(), ValueRange{BATCH},
        ArrayRef<NamedAttribute>{
            rewriter.getNamedAttr(
                "lowerBoundsGroups",
                rewriter.getI32TensorAttr(ArrayRef<int32_t>{1})),
            rewriter.getNamedAttr(
                "upperBoundsGroups",
                rewriter.getI32TensorAttr(ArrayRef<int32_t>{1})),
            rewriter.getNamedAttr("lowerBoundsMap",
                                  AffineMapAttr::get(AffineMap::get(
                                      0, 0, {c0_affine},
                                      rewriter.getContext()))),
            rewriter.getNamedAttr("upperBoundsMap",
                                  AffineMapAttr::get(AffineMap::get(
                                      1, 0, {d0}, rewriter.getContext()))),
            rewriter.getNamedAttr("reductions", rewriter.getArrayAttr({})),
            rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr(1))});

    auto body = new Block();
    rewriter.setInsertionPointToStart(body);
    body->addArgument(rewriter.getIndexType(), loc);

    Value ivBatch = body->getArguments()[0];

    rewriter.create<affine::AffinePrefetchOp>(
        loc, A, AffineMap::get(3, 0, {d0, d1, d2}, rewriter.getContext()),
        ArrayRef<Value>{ivBatch, c0, c0}, false, 3, true);
    affine::buildAffineLoopNest(
        rewriter, loc, {c0}, {K}, 1,
        [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
          Value ivB_row = ivRange.front();
          affine::buildAffineLoopNest(
              builder, loc, {c0}, {M}, 1,
              [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
                Value ivA_row = ivRange.front();
                Value applied_n = builder.create<affine::AffineApplyOp>(
                    loc, AffineMap::get(1, 0, d0.ceilDiv(affineVectorSize)),
                    ValueRange{N});
                affine::buildAffineLoopNest(
                    builder, loc, {c0}, {applied_n}, 1,
                    [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
                      Value ivB_col = ivRange.front();
                      Value a_ele = builder.create<affine::AffineLoadOp>(
                          loc, A, ValueRange{ivBatch, ivA_row, ivB_row});
                      Value a_vec = builder.create<vector::BroadcastOp>(
                          loc,
                          VectorType::get({affineVectorSize}, A_elementType),
                          a_ele);
                      Value b_col_cur =
                          builder.create<arith::MulIOp>(loc, ivB_col, step);
                      Value tail_len =
                          builder.create<arith::SubIOp>(loc, N, b_col_cur);
                      Value tail_flag = builder.create<arith::CmpIOp>(
                          loc, mlir::arith::CmpIPredicate::sge, tail_len, step);
                      builder.create<scf::IfOp>(
                          loc, tail_flag,
                          [&](OpBuilder &builder, Location loc) {
                            Value b_vec =
                                builder.create<affine::AffineVectorLoadOp>(
                                    loc,
                                    VectorType::get({affineVectorSize},
                                                    A_elementType),
                                    B,
                                    AffineMap::get(
                                        3, 0, {d0, d1, d2 * affineVectorSize},
                                        rewriter.getContext()),
                                    ValueRange{ivBatch, ivB_row, ivB_col});
                            Value c_vec =
                                builder.create<affine::AffineVectorLoadOp>(
                                    loc,
                                    VectorType::get({affineVectorSize},
                                                    A_elementType),
                                    C,
                                    AffineMap::get(
                                        3, 0, {d0, d1, d2 * affineVectorSize},
                                        rewriter.getContext()),
                                    ValueRange{ivBatch, ivA_row, ivB_col});
                            Value result_vec;
                            if (A_elementType.isIntOrFloat() && 0) { // bug
                              Value add_vec = builder.create<arith::MulIOp>(
                                  loc, a_vec, b_vec);
                              result_vec = builder.create<arith::AddIOp>(
                                  loc, add_vec, c_vec);
                            } else {
                              result_vec = builder.create<vector::FMAOp>(
                                  loc, a_vec, b_vec, c_vec);
                            }
                            builder.create<affine::AffineVectorStoreOp>(
                                loc, result_vec, C,
                                AffineMap::get(3, 0,
                                               {d0, d1, d2 * affineVectorSize},
                                               rewriter.getContext()),
                                ValueRange{ivBatch, ivA_row, ivB_col});
                            builder.create<scf::YieldOp>(loc);
                          },
                          [&](OpBuilder &builder, Location loc) {
                            Value mask_vec =
                                builder.create<vector::CreateMaskOp>(
                                    loc,
                                    VectorType::get({affineVectorSize},
                                                    rewriter.getI1Type()),
                                    ValueRange{tail_len});
                            Value b_col_idx_tail =
                                builder.create<arith::MulIOp>(loc, ivB_col,
                                                              step);
                            Value b_vec_tail =
                                builder.create<vector::MaskedLoadOp>(
                                    loc,
                                    VectorType::get({affineVectorSize},
                                                    A_elementType),
                                    B,
                                    ValueRange{ivBatch, ivB_row,
                                               b_col_idx_tail},
                                    mask_vec, c0_dynamicType_vec);
                            Value c_vec_tail =
                                builder.create<vector::MaskedLoadOp>(
                                    loc,
                                    VectorType::get({affineVectorSize},
                                                    A_elementType),
                                    C,
                                    ValueRange{ivBatch, ivA_row,
                                               b_col_idx_tail},
                                    mask_vec, c0_dynamicType_vec);
                            Value result_vec_tail;
                            if (A_elementType.isIntOrFloat() && 0) { // bug
                              Value add_vec = builder.create<arith::MulIOp>(
                                  loc, a_vec, b_vec_tail);
                              result_vec_tail = builder.create<arith::AddIOp>(
                                  loc, add_vec, c_vec_tail);
                            } else {
                              result_vec_tail = builder.create<vector::FMAOp>(
                                  loc, a_vec, b_vec_tail, c_vec_tail);
                            }
                            builder.create<vector::MaskedStoreOp>(
                                loc, C,
                                ValueRange{ivBatch, ivA_row, b_col_idx_tail},
                                mask_vec, result_vec_tail);
                            builder.create<scf::YieldOp>(loc);
                          });
                    });
              });
        });

    rewriter.create<affine::AffineYieldOp>(loc);

    parallelLoop.getRegion().push_back(body);
    rewriter.setInsertionPointAfter(parallelLoop);

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t affineVectorSize;
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
  StringRef getArgument() const final { return "batchmatmul-optimize"; }
  StringRef getDescription() const final { return "BatchMatMul Optimization."; }
  BatchMatMulOptimizePass() = default;
  BatchMatMulOptimizePass(const BatchMatMulOptimizePass &) {}
  explicit BatchMatMulOptimizePass(int64_t affineVectorSizeParam) {
    affineVectorSize = affineVectorSizeParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, VectorDialect>();
  }

  Option<int64_t> affineVectorSize{
      *this, "vector-size",
      llvm::cl::desc("Affine Vector size."), llvm::cl::init(64)};
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
  patterns.add<BatchMatMulOptimizePattern>(context, affineVectorSize);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerBatchMatMulOptimizePass() {
  PassRegistration<BatchMatMulOptimizePass>();
}
} // namespace buddy
} // namespace mlir
