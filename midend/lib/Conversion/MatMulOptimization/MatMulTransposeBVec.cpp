//===- MatMulTransposeBVec.cpp --------------------------------------------===//
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
// This file implements the Matmul_TransposeB vectorization.
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

#include "Utils/Utils.h"

using namespace mlir;
using namespace vector;
using namespace affine;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
class MatMulTransposeBVecPattern : public ConversionPattern {
public:
  explicit MatMulTransposeBVecPattern(MLIRContext *context,
                                      int64_t vecSizeparam)
      : ConversionPattern(linalg::MatmulTransposeBOp::getOperationName(), 1,
                          context) {
    vecSize = vecSizeparam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();
    // Get input A, B, C.
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);

    const Value c0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value c1 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    const Value aRow = rewriter.create<memref::DimOp>(loc, A, c0);
    const Value bRow = rewriter.create<memref::DimOp>(loc, B, c0);
    const Value bCol = rewriter.create<memref::DimOp>(loc, B, c1);

    // Get shape of input and output.
    ShapedType ATy = A.getType().cast<ShapedType>();
    Type eleTy = ATy.getElementType();

    // the element type for mask vector.
    IntegerType i1 = IntegerType::get(ctx, 1);

    VectorType vectorTy = mlir::VectorType::get({vecSize}, eleTy);
    VectorType vectorMaskTy = VectorType::get({vecSize}, i1);

    const Value step = rewriter.create<arith::ConstantIndexOp>(loc, vecSize);

    const Value c0Ele = buddy::insertZeroConstantOp(ctx, rewriter, loc, eleTy);
    Value passthruVec = rewriter.create<SplatOp>(loc, vectorTy, c0Ele);

    AffineExpr t;
    bindDims(ctx, t);
    AffineMap vecTailMap = AffineMap::get(1, 0, {t.floorDiv(vecSize)}, ctx);
    AffineExpr d0 = rewriter.getAffineDimExpr(0);
    AffineExpr d1 = rewriter.getAffineDimExpr(1);

    // AffineMap vecTailMap = AffineMap::get(1, 0, {d0.ceilDiv(vecSize)}, ctx);
    SmallVector<Value, 8> lowerBounds(2, c0);
    SmallVector<Value, 8> uperBounds{bRow, bCol};
    SmallVector<int64_t, 8> steps(2, 1);
    // clang-format off

    SmallVector<Value, 4U> reducedValues = llvm::to_vector<4>(
    llvm::map_range(ArrayRef<mlir::affine::LoopReduction>{},
                    [](const LoopReduction &red) { return red.value; }));

    mlir::affine::AffineParallelOp parallelLoop = rewriter.create<affine::AffineParallelOp>(
        loc,ValueRange(reducedValues).getTypes(),ValueRange{aRow}
        ,ArrayRef<NamedAttribute>{
        rewriter.getNamedAttr("lowerBoundsGroups",
                                rewriter.getI32TensorAttr({1})),
        rewriter.getNamedAttr("upperBoundsGroups",
                                rewriter.getI32TensorAttr({1})),
        rewriter.getNamedAttr(
            "lowerBoundsMap",
            AffineMapAttr::get(
                AffineMap::get(0, 0, {rewriter.getAffineConstantExpr(0)}, rewriter.getContext()))),
        rewriter.getNamedAttr("upperBoundsMap",
                                AffineMapAttr::get(AffineMap::get(
                                    1, 0, {rewriter.getAffineDimExpr(0)}, rewriter.getContext()))),
        rewriter.getNamedAttr("reductions", rewriter.getArrayAttr({})),
        //rewriter.getNamedAttr("reductions", rewriter.getArrayAttr({rewriter.getStringAttr("add")})),
        rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr({1}))}
        );

        Block *LoopBody = new Block();
        rewriter.setInsertionPointToStart(LoopBody);
        LoopBody->addArgument(rewriter.getIndexType(), loc);
        Value RowofA = LoopBody->getArguments()[0];

        if(C.getType().cast<MemRefType>().getDimSize(1) % vecSize != 0
            || C.getType().cast<MemRefType>().getDimSize(1)<0){
            affine::buildAffineLoopNest(
                rewriter,loc,{c0},{bRow},1,
                [&](OpBuilder &builder,Location loc, ValueRange ivRange){
                    Value RowofB = ivRange[0];
                    Value tailIndex0 = c0;
                    Value sum0 = c0Ele;
                    auto result = builder.create<affine::AffineForOp>(
                        loc,ValueRange{c0},builder.getDimIdentityMap(),
                        ValueRange{bCol},vecTailMap,1,ValueRange{tailIndex0,sum0} ,
                        [&](OpBuilder &builder,Location loc, Value iv,ValueRange iter){
                        Value ColofB = iv;
                        AffineMap AVectorMap = AffineMap::get(2, 0, {d0, d1 * vecSize},
                            ctx);
                        Value aVec = builder.create<affine::AffineVectorLoadOp>(
                            loc, vectorTy, A, AVectorMap,
                            ValueRange{RowofA,ColofB});
                        AffineMap BVectorMap = AffineMap::get(2, 0, {d0, d1 * vecSize},
                            ctx);
                        Value bVec = builder.create<affine::AffineVectorLoadOp>(
                            loc, vectorTy, B, BVectorMap,
                            ValueRange{RowofB,ColofB});
                        Value resvec = builder.create<arith::MulFOp>(loc,aVec,bVec);
                        Value res1 = builder.create<mlir::vector::ReductionOp>(
                            loc,mlir::vector::CombiningKind::ADD,resvec);

                        Value tailBegin = builder.create<arith::AddIOp>(loc,iter[0],step);
                        Value curSum = builder.create<arith::AddFOp>(loc,iter[1],res1);
                        builder.create<affine::AffineYieldOp>(loc,ValueRange{tailBegin,curSum});
                        }
                    );
                    Value tailIndex = result.getResult(0);
                    Value tailLength = rewriter.create<arith::SubIOp>(loc,bCol,tailIndex);
                    Value maskVec = builder.create<CreateMaskOp>(loc, vectorMaskTy, tailLength);
                    Value aVecTail = builder.create<MaskedLoadOp>(
                                loc, vectorTy, A, ValueRange{RowofA,tailIndex},
                                maskVec, passthruVec);
                    Value bVecTail = builder.create<MaskedLoadOp>(
                        loc, vectorTy, B, ValueRange{RowofB,tailIndex},
                        maskVec, passthruVec);
                    Value resvec = builder.create<arith::MulFOp>(loc,aVecTail,bVecTail);
                    Value res1 = builder.create<mlir::vector::ReductionOp>(
                        loc,mlir::vector::CombiningKind::ADD,resvec);

                    Value sum = builder.create<arith::AddFOp>(loc, res1, result.getResult(1));
                    builder.create<memref::StoreOp>(loc, sum, C, ValueRange{RowofA,RowofB});
                }
            );
        }

        else{
            affine::buildAffineLoopNest(
                rewriter,loc,{c0},{bRow},1,
                [&](OpBuilder &builder,Location loc, ValueRange ivRange){
                    Value RowofB = ivRange[0];
                    Value sum0 = c0Ele;
                    auto result = builder.create<affine::AffineForOp>(
                        loc,ValueRange{c0},builder.getDimIdentityMap(),
                        ValueRange{bCol},vecTailMap,1, ValueRange{sum0},
                        [&](OpBuilder &builder,Location loc, Value iv,ValueRange iter){
                        Value ColofB = iv;
                        AffineMap AVectorMap = AffineMap::get(2, 0, {d0, d1 * vecSize},
                            ctx);
                        Value aVec = builder.create<affine::AffineVectorLoadOp>(
                            loc, vectorTy, A, AVectorMap,
                            ValueRange{RowofA,ColofB});
                        AffineMap BVectorMap = AffineMap::get(2, 0, {d0, d1 * vecSize},
                            ctx);
                        Value bVec = builder.create<affine::AffineVectorLoadOp>(
                            loc, vectorTy, B, BVectorMap,
                            ValueRange{RowofB,ColofB});
                        Value resvec = builder.create<arith::MulFOp>(loc,aVec,bVec);
                        Value res1 = builder.create<mlir::vector::ReductionOp>(
                            loc,mlir::vector::CombiningKind::ADD,resvec);
                        Value sum = builder.create<arith::AddFOp>(loc, res1, iter[0]);
                        rewriter.create<affine::AffineYieldOp>(loc,sum);
                        }
                    );
                    builder.create<memref::StoreOp>(loc, result.getResult(0), C, ValueRange{RowofA,RowofB});
                }
            );
        }
        rewriter.create<affine::AffineYieldOp>(loc);

        // Finalize the loop and erase the original operation.
        parallelLoop.getRegion().push_back(LoopBody);
        rewriter.setInsertionPointAfter(parallelLoop);

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

namespace {
class MatMulTransposeBVecPass
    : public PassWrapper<MatMulTransposeBVecPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulTransposeBVecPass)
  StringRef getArgument() const final {
    return "matmul-transpose-b-vectorization";
  }
  StringRef getDescription() const final {
    return "vectorize linalg MatmulTransposeBOp";
  }
  MatMulTransposeBVecPass() = default;
  MatMulTransposeBVecPass(const MatMulTransposeBVecPass &) {}
  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, VectorDialect>();
  }
  Option<int64_t> vecSize{*this, "vec-size",
                          llvm::cl::desc("The size of vectorization"),
                          llvm::cl::init(8)};
};
} // namespace

void MatMulTransposeBVecPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                       scf::SCFDialect, memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<MatMulTransposeBVecPattern>(context, vecSize);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerMatMulTransposeBVecPass() {
  PassRegistration<MatMulTransposeBVecPass>();
}
} // namespace buddy
} // namespace mlir
