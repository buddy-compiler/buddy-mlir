//===- MatMulLlamaModePass.cpp --------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may not use this file except in compliance with the License.
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
// Corresponds one-to-one with try.mlir, including both top-level chunked
// parallelism and inner one_chunk tiled kernel.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace vector;

namespace {

class MatMulLlamaModePattern : public ConversionPattern {
public:
  explicit MatMulLlamaModePattern(MLIRContext *context)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto ctx = rewriter.getContext();

    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);

    // Types and constants
    auto elemTy = A.getType().cast<MemRefType>().getElementType();
    Value zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    Value one =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    Value chunk16 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(16));
    Value chunk64 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(64));

    // Dimensions
    Value M = rewriter.create<memref::DimOp>(loc, A, zero);
    Value K = rewriter.create<memref::DimOp>(loc, A, one);
    Value N = rewriter.create<memref::DimOp>(loc, B, one);

    // Compute chunk size logic (adaptive, as in try.mlir)
    Value cond_nr0_one =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, M, one);
    Value cond_nr1_one =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, N, one);
    Value chunk_size_tmp =
        rewriter.create<arith::SelectOp>(loc, cond_nr0_one, chunk64, chunk16);
    Value chunk_size = rewriter.create<arith::SelectOp>(
        loc, cond_nr1_one, chunk64, chunk_size_tmp);

    // nchunk0 = ceildiv(M, chunk_size)
    // nchunk1 = ceildiv(N, chunk_size)
    Value tmp0 = rewriter.create<arith::AddIOp>(loc, M, chunk_size);
    Value tmp0_1 = rewriter.create<arith::SubIOp>(loc, tmp0, one);
    Value nchunk0 = rewriter.create<arith::DivUIOp>(loc, tmp0_1, chunk_size);

    Value tmp1 = rewriter.create<arith::AddIOp>(loc, N, chunk_size);
    Value tmp1_1 = rewriter.create<arith::SubIOp>(loc, tmp1, one);
    Value nchunk1 = rewriter.create<arith::DivUIOp>(loc, tmp1_1, chunk_size);

    // dr0 = ceildiv(M, nchunk0)
    Value dr0_tmp = rewriter.create<arith::AddIOp>(loc, M, nchunk0);
    Value dr0_tmp1 = rewriter.create<arith::SubIOp>(loc, dr0_tmp, one);
    Value dr0 = rewriter.create<arith::DivUIOp>(loc, dr0_tmp1, nchunk0);

    // dr1 = ceildiv(N, nchunk1)
    Value dr1_tmp = rewriter.create<arith::AddIOp>(loc, N, nchunk1);
    Value dr1_tmp1 = rewriter.create<arith::SubIOp>(loc, dr1_tmp, one);
    Value dr1 = rewriter.create<arith::DivUIOp>(loc, dr1_tmp1, nchunk1);

    Value total_chunks = rewriter.create<arith::MulIOp>(loc, nchunk0, nchunk1);

    // Outer parallel over total_chunks (one chunk per thread)
    rewriter.create<scf::ParallelOp>(
        loc, ValueRange{zero}, ValueRange{total_chunks}, ValueRange{one},
        [&](OpBuilder &parBuilder, Location ploc, ValueRange parIvs) {
          Value current_chunk = parIvs[0];

          // ith0 = current_chunk % nchunk0
          Value ith0 =
              parBuilder.create<arith::RemUIOp>(ploc, current_chunk, nchunk0);
          // ith1 = current_chunk / nchunk0
          Value ith1 =
              parBuilder.create<arith::DivUIOp>(ploc, current_chunk, nchunk0);

          // row block indices
          Value ir0_start = parBuilder.create<arith::MulIOp>(ploc, dr0, ith0);
          Value ir0_end_tmp =
              parBuilder.create<arith::AddIOp>(ploc, ir0_start, dr0);
          Value ir0_lt = parBuilder.create<arith::CmpIOp>(
              ploc, arith::CmpIPredicate::slt, ir0_end_tmp, M);
          Value ir0_end =
              parBuilder.create<arith::SelectOp>(ploc, ir0_lt, ir0_end_tmp, M);

          // col block indices
          Value ir1_start = parBuilder.create<arith::MulIOp>(ploc, dr1, ith1);
          Value ir1_end_tmp =
              parBuilder.create<arith::AddIOp>(ploc, ir1_start, dr1);
          Value ir1_lt = parBuilder.create<arith::CmpIOp>(
              ploc, arith::CmpIPredicate::slt, ir1_end_tmp, N);
          Value ir1_end =
              parBuilder.create<arith::SelectOp>(ploc, ir1_lt, ir1_end_tmp, N);

          // Number of rows per dot (usually 1 for generality)
          Value num_rows_per_vec_dot = parBuilder.create<arith::ConstantOp>(
              ploc, parBuilder.getIndexAttr(1));
          Value blck_0 = parBuilder.create<arith::ConstantOp>(
              ploc, parBuilder.getIndexAttr(16));
          Value blck_1 = parBuilder.create<arith::ConstantOp>(
              ploc, parBuilder.getIndexAttr(16));

          // Now call mul_mat_one_chunk for this chunk
          parBuilder.create<scf::ForOp>(
              ploc, ir1_start, ir1_end, blck_1, ValueRange{},
              [&](OpBuilder &b1, Location loc, Value iir1, ValueRange) {
                Value iir1_end_tmp =
                    b1.create<arith::AddIOp>(loc, iir1, blck_1);
                Value iir1_lt = b1.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::slt, iir1_end_tmp, ir1_end);
                Value iir1_end = b1.create<arith::SelectOp>(
                    loc, iir1_lt, iir1_end_tmp, ir1_end);

                b1.create<scf::ParallelOp>(
                    loc, ValueRange{ir0_start}, ValueRange{ir0_end},
                    ValueRange{blck_0},
                    [&](OpBuilder &parInnerBuilder, Location loc,
                        ValueRange ivs) {
                      Value iir0 = ivs[0];
                      Value iir0_end_tmp =
                          parInnerBuilder.create<arith::AddIOp>(loc, iir0,
                                                                blck_0);
                      Value iir0_lt = parInnerBuilder.create<arith::CmpIOp>(
                          loc, arith::CmpIPredicate::slt, iir0_end_tmp,
                          ir0_end);
                      Value iir0_end = parInnerBuilder.create<arith::SelectOp>(
                          loc, iir0_lt, iir0_end_tmp, ir0_end);

                      parInnerBuilder.create<scf::ForOp>(
                          loc, iir1, iir1_end, num_rows_per_vec_dot,
                          ValueRange{},
                          [&](OpBuilder &b2, Location loc, Value ir1,
                              ValueRange) {
                            b2.create<scf::ForOp>(
                                loc, iir0, iir0_end, num_rows_per_vec_dot,
                                ValueRange{},
                                [&](OpBuilder &b3, Location loc, Value ir0,
                                    ValueRange) {
                                  // Vectorized dot-add kernel
                                  int vecSize = 8; // SIMD width
                                  bool scalable = false;

                                  VectorType vectorTy = VectorType::get(
                                      {vecSize}, elemTy, {scalable});
                                  Value c0 = b3.create<arith::ConstantOp>(
                                      loc, b3.getIndexAttr(0));
                                  Value c1 = b3.create<arith::ConstantOp>(
                                      loc, b3.getIndexAttr(1));
                                  Value vecStep = b3.create<arith::ConstantOp>(
                                      loc, b3.getIndexAttr(vecSize));

                                  // Compute vectorization bounds for K
                                  Value vecLenVal = vecStep;
                                  Value vecIters = b3.create<arith::DivUIOp>(
                                      loc, K, vecLenVal);
                                  Value vecLimit = b3.create<arith::MulIOp>(
                                      loc, vecIters, vecLenVal);
                                  // K_tail = K % vecSize
                                  Value tailSize = b3.create<arith::RemUIOp>(
                                      loc, K, vecLenVal);

                                  // Vector sum initialization to 0
                                  Value zeroF = b3.create<arith::ConstantOp>(
                                      loc, elemTy,
                                      b3.getFloatAttr(elemTy, 0.0));
                                  Value accVecInit = b3.create<vector::SplatOp>(
                                      loc, vectorTy, zeroF);

                                  auto vecAccFor = b3.create<scf::ForOp>(
                                      loc, c0, vecLimit, vecLenVal,
                                      ValueRange{accVecInit},
                                      [&](OpBuilder &vecBuilder, Location loc,
                                          Value kk, ValueRange iterArgs) {
                                        Value vecSum = iterArgs[0];
                                        Value avec =
                                            vecBuilder.create<vector::LoadOp>(
                                                loc, vectorTy, A,
                                                ValueRange{ir0, kk});
                                        Value bvec =
                                            vecBuilder.create<vector::LoadOp>(
                                                loc, vectorTy, B,
                                                ValueRange{kk, ir1});
                                        Value sumvec =
                                            vecBuilder.create<vector::FMAOp>(
                                                loc, avec, bvec, vecSum);
                                        vecBuilder.create<scf::YieldOp>(
                                            loc, ValueRange{sumvec});
                                      });
                                  Value accVec = vecAccFor.getResult(0);
                                  Value sumScalar =
                                      b3.create<arith::ConstantOp>(
                                          loc, elemTy,
                                          b3.getFloatAttr(elemTy, 0.0));
                                  for (int i = 0; i < vecSize; ++i) {
                                    Value idx = b3.create<arith::ConstantOp>(
                                        loc, b3.getIndexAttr(i));
                                    Value item =
                                        b3.create<vector::ExtractElementOp>(
                                            loc, accVec, idx);
                                    sumScalar = b3.create<arith::AddFOp>(
                                        loc, sumScalar, item);
                                  }

                                  // Scalar tail accumulation
                                  auto tailFor = b3.create<scf::ForOp>(
                                      loc, vecLimit, K, c1,
                                      ValueRange{sumScalar},
                                      [&](OpBuilder &tailBuilder, Location loc,
                                          Value kk, ValueRange iterArgs) {
                                        Value sum = iterArgs[0];
                                        Value aVal =
                                            tailBuilder.create<memref::LoadOp>(
                                                loc, A, ValueRange{ir0, kk});
                                        Value bVal =
                                            tailBuilder.create<memref::LoadOp>(
                                                loc, B, ValueRange{kk, ir1});
                                        Value prod =
                                            tailBuilder.create<arith::MulFOp>(
                                                loc, aVal, bVal);
                                        Value newSum =
                                            tailBuilder.create<arith::AddFOp>(
                                                loc, sum, prod);
                                        tailBuilder.create<scf::YieldOp>(
                                            loc, ValueRange{newSum});
                                      });

                                  Value finalScalar = tailFor.getResult(0);

                                  // Add the original C[ir0, ir1]
                                  Value cOrig = b3.create<memref::LoadOp>(
                                      loc, C, ValueRange{ir0, ir1});
                                  Value cNew = b3.create<arith::AddFOp>(
                                      loc, finalScalar, cOrig);

                                  // Store the result
                                  b3.create<memref::StoreOp>(
                                      loc, cNew, C, ValueRange{ir0, ir1});
                                  b3.create<scf::YieldOp>(loc);
                                });
                            b2.create<scf::YieldOp>(loc);
                          });
                    });
                b1.create<scf::YieldOp>(loc);
              });
        });
    // Erase linalg.matmul
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

namespace {
class MatMulLlamaMode
    : public PassWrapper<MatMulLlamaMode, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulLlamaMode)
  StringRef getArgument() const final { return "matmul-vectorization-llama"; }
  StringRef getDescription() const final {
    return "Llama mode matmul (full chunk + tile parallelization, matches "
           "try.mlir)";
  }
  MatMulLlamaMode() = default;
  MatMulLlamaMode(const MatMulLlamaMode &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, mlir::vector::VectorDialect,
                    memref::MemRefDialect, arith::ArithDialect>();
  }
};
} // namespace

void MatMulLlamaMode::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                         scf::SCFDialect, memref::MemRefDialect,
                         mlir::vector::VectorDialect>();
  target.addLegalOp<ModuleOp>();
  target.addLegalOp<func::FuncOp>();
  target.addLegalOp<func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();
  target.addIllegalOp<linalg::MatmulOp>();

  RewritePatternSet patterns(context);
  patterns.add<MatMulLlamaModePattern>(context);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerMatMulLlamaMode() { PassRegistration<MatMulLlamaMode>(); }
} // namespace buddy
} // namespace mlir
