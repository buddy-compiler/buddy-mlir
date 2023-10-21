//===- MatMulVectorization.cpp --------------------------------------------===//
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
// This file implements the matmul vectorization.
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

#include "Utils/Utils.h"

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class MatMulVectorizationPattern : public ConversionPattern {
public:
  explicit MatMulVectorizationPattern(MLIRContext *context,
                                      int64_t vecSizeParam)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), 1, context) {
    vecSize = vecSizeParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    // Get input A, B, C.
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);
    // Get shape of input and output
    ShapedType ATy = A.getType().cast<ShapedType>();
    Type eleTy = ATy.getElementType();
    // ShapedType BTy = B.getType().cast<ShapedType>();
    // ShapedType CTy = C.getType().cast<ShapedType>();

    auto ctx = op->getContext();
    // Get i1 as the element type for mask vector.
    IntegerType i1 = IntegerType::get(ctx, 1);
    // Define `*Type`.
    VectorType vectorTy = mlir::VectorType::get({vecSize}, eleTy);
    VectorType vectorMaskTy = VectorType::get({vecSize}, i1);
    // Some constants.
    const Value c0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value c1 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    const Value step = rewriter.create<arith::ConstantIndexOp>(loc, vecSize);
    // Create pass through vector.
    const Value c0Ele = buddy::insertZeroConstantOp(ctx, rewriter, loc, eleTy);
    Value passthruVec = rewriter.create<SplatOp>(loc, vectorTy, c0Ele);

    // Create DimOp.
    const Value aRow = rewriter.create<memref::DimOp>(loc, A, c0);
    // This algorithm does not use the column A index.
    // const Value aCol = rewriter.create<memref::DimOp>(loc, A, c1);
    const Value bRow = rewriter.create<memref::DimOp>(loc, B, c0);
    const Value bCol = rewriter.create<memref::DimOp>(loc, B, c1);
    // Size of vector type.
    AffineExpr d0;
    bindDims(ctx, d0);
    AffineMap vecTailMap = AffineMap::get(1, 0, {d0.ceilDiv(vecSize)}, ctx);
    SmallVector<Value, 8> lowerBounds(2, c0);
    SmallVector<Value, 8> uperBounds{bRow, aRow};
    SmallVector<int64_t, 8> steps(2, /*Value=*/1);
    // clang-format off
    affine::buildAffineLoopNest(
        rewriter, loc, lowerBounds, uperBounds, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
      // Create loop based on vector size.
      builder.create<affine::AffineForOp>(
          loc, ValueRange{c0}, builder.getDimIdentityMap(),
          ValueRange{bCol}, vecTailMap, /*Step=*/1, std::nullopt,
          [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
              ValueRange itrArgs) {
        // Load element and broadcast to vector.
        Value aEle = builder.create<memref::LoadOp>(
            loc, A, ValueRange{ivs[1], ivs[0]});
        Value aVec = builder.create<vector::BroadcastOp>(loc, vectorTy, aEle);
        // Check tail.
        AffineExpr m, n, k;
        bindDims(ctx, m, n, k);
        AffineMap BVectorMap = AffineMap::get(
            /*dimCount=*/3, /*symbolCount=*/0, {m, k * vecSize}, ctx);
        AffineExpr x, y, z;
        bindDims(ctx, x, y, z);
        AffineMap CVectorMap = AffineMap::get(
            /*dimCount=*/3, /*symbolCount=*/0, {y, z * vecSize}, ctx);
        // Calculate the tail.
        Value bColCur = builder.create<arith::MulIOp>(loc, iv, step);
        Value tailLen = builder.create<arith::SubIOp>(loc, bCol, bColCur);
        Value tailFlag = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sge, tailLen, step);
        // If the current column does not reach the tail.
        builder.create<scf::IfOp>(loc, tailFlag,
            [&](OpBuilder &builder, Location loc) {
          Value bVec = builder.create<affine::AffineVectorLoadOp>(
              loc, vectorTy, B, BVectorMap, ValueRange{ivs[0], ivs[1], iv});
          Value cVec = builder.create<affine::AffineVectorLoadOp>(
              loc, vectorTy, C, CVectorMap, ValueRange{ivs[0], ivs[1], iv});
          // FMA = Fused Multiply + Add
          // FMAOp only supports floating point type input.
          // TODO: Write a utils function for FMA to support both int and float.
          Value resultVector = builder.create<FMAOp>(loc, aVec, bVec, cVec);
          builder.create<affine::AffineVectorStoreOp>(
              loc, resultVector, C, CVectorMap, ValueRange{ivs[0], ivs[1], iv});
          builder.create<scf::YieldOp>(loc);
        },
        // The else branch (the current column reaches the tail).
        [&](OpBuilder &builder, Location loc) {
          // Create mask according to the tail.
          Value maskVec = builder.create<CreateMaskOp>(
              loc, vectorMaskTy, tailLen);
          Value bColIdxTail = builder.create<arith::MulIOp>(loc, iv, step);
          // Masked load input and output.
          Value bVecTail = builder.create<MaskedLoadOp>(
              loc, vectorTy, B, ValueRange{ivs[0], bColIdxTail},
              maskVec, passthruVec);
          Value cVecTail = builder.create<MaskedLoadOp>(
              loc, vectorTy, C, ValueRange{ivs[1], bColIdxTail},
              maskVec, passthruVec);
          // FMA.
          Value resultVecTail =
              builder.create<FMAOp>(loc, aVec, bVecTail, cVecTail);
          builder.create<MaskedStoreOp>(
              loc, C, ValueRange{ivs[1], bColIdxTail}, maskVec, resultVecTail);
          builder.create<scf::YieldOp>(loc);
        });
        builder.create<affine::AffineYieldOp>(loc);
      });
    });
    // clang-format on
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t vecSize;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// MatMulVectorizationPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg matmul operations to mixture of
/// Affine + Vector operations.
namespace {
class MatMulVectorizationPass
    : public PassWrapper<MatMulVectorizationPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulVectorizationPass)
  StringRef getArgument() const final { return "matmul-vectorization"; }
  StringRef getDescription() const final { return "MatMul Vectorization."; }
  MatMulVectorizationPass() = default;
  MatMulVectorizationPass(const MatMulVectorizationPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, VectorDialect>();
  }
  Option<int64_t> vecSize{*this, "vector-size",
                          llvm::cl::desc("Specify vector type size."),
                          llvm::cl::init(32)};
};
} // end anonymous namespace.

void MatMulVectorizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                       scf::SCFDialect, memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<MatMulVectorizationPattern>(context, vecSize);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerMatMulVectorizationPass() {
  PassRegistration<MatMulVectorizationPass>();
}
} // namespace buddy
} // namespace mlir
