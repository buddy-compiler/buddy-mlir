

//===- MatMulVectorization.cpp -------------------------------------------------===//
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

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class MatMulVectorizationPattern : public ConversionPattern {
public:
  explicit MatMulVectorizationPattern(MLIRContext *context, int64_t vecSizeParam, int64_t strideParam,
                                 int64_t kernelMParam, int64_t kernelNParam)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), 1, context) {
    vecSize = vecSizeParam;
    kernelM = kernelMParam;
    kernelN = kernelNParam;
    stride = strideParam;
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
    // ShapedType BTy = B.getType().cast<ShapedType>();
    // ShapedType CTy = C.getType().cast<ShapedType>();

    auto ctx = op->getContext();
    // Currently use f32 as the element type.
    // TODO: replace f32 with input type.
    FloatType f32 = mlir::FloatType::getF32(ctx);
    // Get i1 as the element type for mask vector.
    IntegerType i1 = IntegerType::get(ctx, 1);
    // Define `*Type`.
    VectorType vectorTy32 = mlir::VectorType::get({stride}, f32);
    VectorType vectorMaskTy = VectorType::get({stride}, i1);
    // Some constants.
    const Value c0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value c1 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    const Value step = 
        rewriter.create<arith::ConstantIndexOp>(loc, stride);
    const Value c0_f32 = rewriter.create<arith::ConstantFloatOp>(
      loc, APFloat::getZero(f32.getFloatSemantics()), f32);
    // Create pass through vector.
    Value c0_f32_vec = rewriter.create<SplatOp>(loc, vectorTy32, c0_f32);

    // Create DimOp.
    const Value a_row = rewriter.create<memref::DimOp>(loc, A, c0);
    const Value a_col = rewriter.create<memref::DimOp>(loc, A, c1);
    const Value b_row = rewriter.create<memref::DimOp>(loc, B, c0);
    const Value b_col = rewriter.create<memref::DimOp>(loc, B, c1);
    // Size of strip mining.
    AffineExpr d0;
    bindDims(ctx, d0);
    AffineMap stripMap = AffineMap::get(1, 0, {d0.ceilDiv(stride)}, ctx);
    SmallVector<Value, 8> lowerBounds(3, c0);
    SmallVector<Value, 8> uperBounds{b_row, a_row, b_col};
    SmallVector<int64_t, 8> steps(3, /*Value=*/1);
    affine::buildAffineLoopNest(

      rewriter, loc, lowerBounds, uperBounds, steps,
      [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        Value a_ele = builder.create<memref::LoadOp>(
                  loc, A, ValueRange{ivs[1], ivs[0]});
        Value a_vec =
            builder.create<vector::BroadcastOp>(loc, vectorTy32, a_ele);

        // Load input vector from memref.
        AffineExpr m, n, k;
        bindDims(ctx, m, n, k);
        AffineMap BVectorMap = AffineMap::get(
            /*dimCount=*/3, /*symbolCount=*/0,
            {m, k * stride}, ctx);
        AffineExpr x, y, z;
        bindDims(ctx, x, y, z);
        AffineMap CVectorMap = AffineMap::get(
            /*dimCount=*/3, /*symbolCount=*/0,
            {y, z * stride}, ctx);
        // Calculate the tail.
        Value b_col_cur =
            builder.create<arith::MulIOp>(loc, ivs[2], step);
        Value tail_len = builder.create<arith::SubIOp>(
            loc, b_col, b_col_cur);
        Value tail_flag = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sge, tail_len, step);





        // If the current column does not reach the tail.
        builder.create<scf::IfOp>(
            loc, tail_flag,
            [&](OpBuilder &builder, Location loc) {
                //if
        Value b_vec =
                builder.create<affine::AffineVectorLoadOp>(
                loc, vectorTy32, B, BVectorMap,
                ValueRange{ivs[0], ivs[1], ivs[2]});

        
        Value c_vec =
                builder.create<affine::AffineVectorLoadOp>(
                loc, vectorTy32, C, CVectorMap,
                ValueRange{ivs[0], ivs[1], ivs[2]});

        // FMA = Fused Multiply + Add
        Value resultVector = builder.create<FMAOp>(
                loc, a_vec, b_vec, c_vec);

        builder.create<affine::AffineVectorStoreOp>(
                              loc, resultVector, C, CVectorMap,
                              ValueRange{ivs[0], ivs[1], ivs[2]});
        
        builder.create<scf::YieldOp>(loc);
              },
        // The else branch (the current column reaches the
        // tail).
            [&](OpBuilder &builder, Location loc) {

        // Create mask according to the tail.
                //else
        Value mask_vec = builder.create<CreateMaskOp>(
                loc, vectorMaskTy, tail_len);
        Value b_col_idx_tail = builder.create<arith::MulIOp>(
                loc, ivs[2], step);
        // Masked load input and output.
        Value b_vec_tail = builder.create<MaskedLoadOp>(
                loc, vectorTy32, B,
                ValueRange{ivs[0], b_col_idx_tail}, mask_vec,
                c0_f32_vec);
        Value c_vec_tail = builder.create<MaskedLoadOp>(
                loc, vectorTy32, C,
                ValueRange{ivs[1], b_col_idx_tail}, mask_vec,
                c0_f32_vec);
        // FMA.
        Value result_vec_tail = builder.create<FMAOp>(
                loc, a_vec, b_vec_tail,
                c_vec_tail);

        builder.create<MaskedStoreOp>(
                loc, C, ValueRange{ivs[1], b_col_idx_tail},
                mask_vec, result_vec_tail);

        builder.create<scf::YieldOp>(loc);
              });  

    });
    

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t vecSize;
  int64_t kernelM;
  int64_t kernelN;
  int64_t stride;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// MatMulVectorizationPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg pooling operations to mixture of
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
  explicit MatMulVectorizationPass(int64_t vecSizeParam, int64_t kernelMParam,
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
  Option<int64_t> stride{*this, "strip-mining",
                         llvm::cl::desc("Strip mining size."),
                         llvm::cl::init(32)};

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
  patterns.add<MatMulVectorizationPattern>(context, vecSize, stride, kernelM, kernelN);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerMatMulVectorizationPass() { PassRegistration<MatMulVectorizationPass>(); }
} // namespace buddy
} // namespace mlir
