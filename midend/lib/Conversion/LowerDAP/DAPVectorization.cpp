//====- DAPVectorization.cpp - DAP Dialect Vectorization Pass  ------------===//
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
// This file defines DAP dialect vectorization pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "DAP/DAPDialect.h"
#include "DAP/DAPOps.h"

using namespace mlir;
using namespace buddy;
using namespace vector;
using namespace mlir::arith;
using namespace mlir::linalg;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class DAPIirVectorization : public OpRewritePattern<dap::IirOp> {
public:
  using OpRewritePattern<dap::IirOp>::OpRewritePattern;

  explicit DAPIirVectorization(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(dap::IirOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);

    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    Value c2 = rewriter.create<ConstantIndexOp>(loc, 2);
    Value c4 = rewriter.create<ConstantIndexOp>(loc, 4);
    Value c5 = rewriter.create<ConstantIndexOp>(loc, 5);
    Value c15 = rewriter.create<ConstantIndexOp>(loc, 15);

    Value N = rewriter.create<memref::DimOp>(loc, input, c0);
    Value filterSize = rewriter.create<memref::DimOp>(loc, kernel, c0);

    FloatType f32 = FloatType::getF32(ctx);

    // TODO : Change the vector length value to an Attribute
    VectorType vectorTy32 = VectorType::get(16, f32);

    Value f0 = rewriter.create<ConstantFloatOp>(loc, APFloat(0.0f), f32);
    Value f1 = rewriter.create<ConstantFloatOp>(loc, APFloat(1.0f), f32);

    Value initB0 = rewriter.create<vector::SplatOp>(loc, vectorTy32, f1);
    Value initB1 = rewriter.create<vector::SplatOp>(loc, vectorTy32, f0);
    Value initB2 = rewriter.create<vector::SplatOp>(loc, vectorTy32, f0);
    Value initA1 = rewriter.create<vector::SplatOp>(loc, vectorTy32, f0);
    Value initA2 = rewriter.create<vector::SplatOp>(loc, vectorTy32, f0);

    // Distribute all params into 5 param vectors
    auto vecDistribute = rewriter.create<scf::ForOp>(
        loc, c0, filterSize, c1,
        ValueRange{initB0, initB1, initB2, initA1, initA2},
        [&](OpBuilder &builder, Location loc, Value iv, ValueRange iargs) {
          Value b0 =
              builder.create<memref::LoadOp>(loc, kernel, ValueRange{iv, c0});
          Value b1 =
              builder.create<memref::LoadOp>(loc, kernel, ValueRange{iv, c1});
          Value b2 =
              builder.create<memref::LoadOp>(loc, kernel, ValueRange{iv, c2});
          Value a1 =
              builder.create<memref::LoadOp>(loc, kernel, ValueRange{iv, c4});
          Value a2 =
              builder.create<memref::LoadOp>(loc, kernel, ValueRange{iv, c5});

          Value B0_next =
              builder.create<vector::InsertElementOp>(loc, b0, iargs[0], iv);
          Value B1_next =
              builder.create<vector::InsertElementOp>(loc, b1, iargs[1], iv);
          Value B2_next =
              builder.create<vector::InsertElementOp>(loc, b2, iargs[2], iv);
          Value A1_next =
              builder.create<vector::InsertElementOp>(loc, a1, iargs[3], iv);
          Value A2_next =
              builder.create<vector::InsertElementOp>(loc, a2, iargs[4], iv);

          builder.create<scf::YieldOp>(
              loc,
              std::vector<Value>{B0_next, B1_next, B2_next, A1_next, A2_next});
        });

    Value vecB0 = vecDistribute.getResult(0);
    Value vecB1 = vecDistribute.getResult(1);
    Value vecB2 = vecDistribute.getResult(2);
    Value vecA1 = vecDistribute.getResult(3);
    Value vecA2 = vecDistribute.getResult(4);

    Value vecOut = rewriter.create<vector::SplatOp>(loc, vectorTy32, f0);
    Value vecS1 = rewriter.create<vector::SplatOp>(loc, vectorTy32, f0);
    Value vecS2 = rewriter.create<vector::SplatOp>(loc, vectorTy32, f0);

    // The SIMD version for IIR operation can represented as a pipeline with
    // {vector length} stages. This loop represent the injection section, loop
    // {stages-1} times.
    auto injectionResult = rewriter.create<scf::ForOp>(
        loc, c0, c15, c1, ValueRange{vecOut, vecS1, vecS2},
        [&](OpBuilder &builder, Location loc, Value iv, ValueRange iargs) {
          Value in_elem = builder.create<memref::LoadOp>(loc, input, iv);
          Value vecIn_move_right = builder.create<vector::ShuffleOp>(
              loc, iargs[0], iargs[0],
              ArrayRef<int64_t>{0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                14});
          Value vecIn_next = builder.create<vector::InsertElementOp>(
              loc, in_elem, vecIn_move_right, c0);
          Value vecOut_next =
              builder.create<vector::FMAOp>(loc, vecB0, vecIn_next, iargs[1]);

          Value vecS1_lhs =
              builder.create<vector::FMAOp>(loc, vecB1, vecIn_next, iargs[2]);
          Value vecS1_rhs =
              builder.create<arith::MulFOp>(loc, vecA1, vecOut_next);
          Value vecS1_next =
              builder.create<arith::SubFOp>(loc, vecS1_lhs, vecS1_rhs);

          Value vecS2_lhs =
              builder.create<arith::MulFOp>(loc, vecB2, vecIn_next);
          Value vecS2_rhs =
              builder.create<arith::MulFOp>(loc, vecA2, vecOut_next);
          Value vecS2_next =
              builder.create<arith::SubFOp>(loc, vecS2_lhs, vecS2_rhs);

          builder.create<scf::YieldOp>(
              loc, std::vector<Value>{vecOut_next, vecS1_next, vecS2_next});
        });

    Value vecOut_tmp1 = injectionResult.getResult(0);
    Value vecS1_tmp1 = injectionResult.getResult(1);
    Value vecS2_tmp1 = injectionResult.getResult(2);

    Value i15 =
        rewriter.create<arith::ConstantIntOp>(loc, /*value=*/15, /*width=*/64);
    Value upperBound = rewriter.create<arith::SubIOp>(loc, N, c15);

    // This loop start to produce output.
    auto processResult = rewriter.create<scf::ForOp>(
        loc, c0, upperBound, c1,
        ValueRange{vecOut_tmp1, vecS1_tmp1, vecS2_tmp1},
        [&](OpBuilder &builder, Location loc, Value iv, ValueRange iargs) {
          Value index = builder.create<arith::AddIOp>(loc, iv, c15);
          Value in_elem = builder.create<memref::LoadOp>(loc, input, index);
          Value vecIn_move_right = builder.create<vector::ShuffleOp>(
              loc, iargs[0], iargs[0],
              ArrayRef<int64_t>{0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                14});
          Value vecIn_next = builder.create<vector::InsertElementOp>(
              loc, in_elem, vecIn_move_right, c0);
          Value vecOut_next =
              builder.create<vector::FMAOp>(loc, vecB0, vecIn_next, iargs[1]);
          Value out_elem =
              builder.create<vector::ExtractElementOp>(loc, vecOut_next, i15);
          builder.create<memref::StoreOp>(loc, out_elem, output, iv);

          Value vecS1_lhs =
              builder.create<vector::FMAOp>(loc, vecB1, vecIn_next, iargs[2]);
          Value vecS1_rhs =
              builder.create<arith::MulFOp>(loc, vecA1, vecOut_next);
          Value vecS1_next =
              builder.create<arith::SubFOp>(loc, vecS1_lhs, vecS1_rhs);

          Value vecS2_lhs =
              builder.create<arith::MulFOp>(loc, vecB2, vecIn_next);
          Value vecS2_rhs =
              builder.create<arith::MulFOp>(loc, vecA2, vecOut_next);
          Value vecS2_next =
              builder.create<arith::SubFOp>(loc, vecS2_lhs, vecS2_rhs);

          builder.create<scf::YieldOp>(
              loc, std::vector<Value>{vecOut_next, vecS1_next, vecS2_next});
        });

    Value vecOut_tmp2 = processResult.getResult(0);
    Value vecS1_tmp2 = processResult.getResult(1);
    Value vecS2_tmp2 = processResult.getResult(2);

    // This loop represent tail ending section.
    rewriter.create<scf::ForOp>(
        loc, upperBound, N, c1, ValueRange{vecOut_tmp2, vecS1_tmp2, vecS2_tmp2},
        [&](OpBuilder &builder, Location loc, Value iv, ValueRange iargs) {
          Value vecIn_move_right = builder.create<vector::ShuffleOp>(
              loc, iargs[0], iargs[0],
              ArrayRef<int64_t>{0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                14});
          Value vecIn_next = builder.create<vector::InsertElementOp>(
              loc, f0, vecIn_move_right, c0);
          Value vecOut_next =
              builder.create<vector::FMAOp>(loc, vecB0, vecIn_next, iargs[1]);
          Value out_elem =
              builder.create<vector::ExtractElementOp>(loc, vecOut_next, i15);
          builder.create<memref::StoreOp>(loc, out_elem, output, iv);

          Value vecS1_lhs =
              builder.create<vector::FMAOp>(loc, vecB1, vecIn_next, iargs[2]);
          Value vecS1_rhs =
              builder.create<arith::MulFOp>(loc, vecA1, vecOut_next);
          Value vecS1_next =
              builder.create<arith::SubFOp>(loc, vecS1_lhs, vecS1_rhs);

          Value vecS2_lhs =
              builder.create<arith::MulFOp>(loc, vecB2, vecIn_next);
          Value vecS2_rhs =
              builder.create<arith::MulFOp>(loc, vecA2, vecOut_next);
          Value vecS2_next =
              builder.create<arith::SubFOp>(loc, vecS2_lhs, vecS2_rhs);

          builder.create<scf::YieldOp>(
              loc, std::vector<Value>{vecOut_next, vecS1_next, vecS2_next});
        });

    rewriter.eraseOp(op);
    return success();
  }
};

} // end anonymous namespace

void populateVectorizeDAPConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<DAPIirVectorization>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// VectorizeDAPPass
//===----------------------------------------------------------------------===//

namespace {
class VectorizeDAPPass : public PassWrapper<VectorizeDAPPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizeDAPPass)
  VectorizeDAPPass() = default;
  VectorizeDAPPass(const VectorizeDAPPass &) {}

  StringRef getArgument() const final { return "vectorize-dap"; }
  StringRef getDescription() const final { return "Vectorize DAP Dialect."; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<buddy::dap::DAPDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect, VectorDialect,
                    affine::AffineDialect, arith::ArithDialect,linalg::LinalgDialect>();
  }
};
} // end anonymous namespace.

void VectorizeDAPPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<affine::AffineDialect, scf::SCFDialect,
                         func::FuncDialect, memref::MemRefDialect,
                         VectorDialect, arith::ArithDialect,
                         linalg::LinalgDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();

  RewritePatternSet patterns(context);
  populateVectorizeDAPConversionPatterns(patterns);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
  void registerDAPVectorizePass() { PassRegistration<VectorizeDAPPass>(); }
} // namespace buddy
} // namespace mlir
