//====- DAPVectorization.cpp - DAP Dialect Vectorization Pass  ------------===//
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
// This file defines DAP dialect vectorization pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

#include "DAP/DAPDialect.h"
#include "DAP/DAPOps.h"
#include "Utils/DAPUtils.h"
#include <optional>

using namespace mlir;
using namespace buddy;
using namespace vector;
using namespace mlir::arith;
using namespace mlir::linalg;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class DAPFIRVectorization : public OpRewritePattern<dap::FirOp> {
public:
  using OpRewritePattern<dap::FirOp>::OpRewritePattern;

  explicit DAPFIRVectorization(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(dap::FirOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);

    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    Value c2 = rewriter.create<ConstantIndexOp>(loc, 2);

    // 1. Get the total length of the workload.
    Value inputSize = rewriter.create<memref::DimOp>(loc, input, c0);
    Value kernelSize = rewriter.create<memref::DimOp>(loc, kernel, c0);

    // 2. Set the iteration step (tile size and vector size).
    Value tileStep = rewriter.create<ConstantIndexOp>(loc, 2048);
    Value vlStep = rewriter.create<ConstantIndexOp>(loc, 16);
    Value vlStepMinusOne = rewriter.create<arith::SubIOp>(loc, vlStep, c1);
    FloatType f32 = Float32Type::get(ctx);
    VectorType vecTy = VectorType::get(16, f32);

    // 3. Calculate full vectorization part.

    // 3.1 Calculate upbound for outer loop(tile input).
    // `lastKernelElementUsedInputSize` = `inputSize` - `kernelSize` + 1
    // `inputUpbound` = `lastKernelElementUsedInputSize` - `tileStep` + 1
    Value lastKernelElementUsedInputSize_ =
        rewriter.create<arith::SubIOp>(loc, inputSize, kernelSize);
    Value inputUpbound_ = rewriter.create<arith::SubIOp>(
        loc, lastKernelElementUsedInputSize_, tileStep);
    Value inputUpbound = rewriter.create<arith::AddIOp>(loc, inputUpbound_, c2);

    Value inputOffset =
        rewriter
            .create<scf::ForOp>(
                loc, c0, inputUpbound, tileStep, ValueRange{c0},
                [&](OpBuilder &builder, Location loc, Value address,
                    ValueRange iargs) {
                  Value upbound =
                      builder.create<arith::AddIOp>(loc, address, tileStep);
                  // 3.2 Broadcast each kernel element to a vector.
                  builder.create<scf::ForOp>(
                      loc, c0, kernelSize, c1, ValueRange{std::nullopt},
                      [&](OpBuilder &b, Location loc, Value iv_n,
                          ValueRange iargs) {
                        Value kElem =
                            b.create<memref::LoadOp>(loc, kernel, iv_n);
                        Value kVec =
                            b.create<vector::SplatOp>(loc, vecTy, kElem);
                        // 3.3 Vectorized computation.
                        b.create<scf::ForOp>(
                            loc, address, upbound, vlStep,
                            ValueRange{std::nullopt},
                            [&](OpBuilder &b, Location loc, Value iv_i,
                                ValueRange iargs) {
                              Value inVec = b.create<vector::LoadOp>(
                                  loc, vecTy, input, ValueRange{iv_i});
                              Value outOffset =
                                  b.create<arith::AddIOp>(loc, iv_i, iv_n);
                              Value outVec = b.create<vector::LoadOp>(
                                  loc, vecTy, output, ValueRange{outOffset});
                              Value fmaVec = b.create<vector::FMAOp>(
                                  loc, kVec, inVec, outVec);
                              b.create<vector::StoreOp>(loc, fmaVec, output,
                                                        ValueRange{outOffset});
                              b.create<scf::YieldOp>(loc);
                            });

                        b.create<scf::YieldOp>(loc);
                      });
                  builder.create<scf::YieldOp>(loc, ValueRange{upbound});
                })
            .getResult(0);

    // 4. Calculate tail processing part.
    // 4.1 Calculate upbound for tail processing
    Value tailUpbound_ = rewriter.create<arith::SubIOp>(loc, inputSize, vlStep);
    Value tailUpboundInit =
        rewriter.create<arith::AddIOp>(loc, tailUpbound_, c1);

    // 4.2 Loop through each kernel element.
    rewriter.create<scf::ForOp>(
        loc, c0, kernelSize, c1, ValueRange{tailUpboundInit},
        [&](OpBuilder &builder, Location loc, Value iv_n, ValueRange iargs) {
          Value kElem = builder.create<memref::LoadOp>(loc, kernel, iv_n);
          Value kVec = builder.create<vector::SplatOp>(loc, vecTy, kElem);

          // 4.3 Perform the vectorization body (for tail process).
          Value iterIdx =
              builder
                  .create<scf::ForOp>(
                      loc, inputOffset, iargs[0], vlStep,
                      ValueRange{inputOffset},
                      [&](OpBuilder &b, Location loc, Value iv_i,
                          ValueRange iargs) {
                        Value inVec = b.create<vector::LoadOp>(
                            loc, vecTy, input, ValueRange{iv_i});
                        Value outOffset =
                            b.create<arith::AddIOp>(loc, iv_i, iv_n);
                        Value outVec = b.create<vector::LoadOp>(
                            loc, vecTy, output, ValueRange{outOffset});
                        Value fmaVec =
                            b.create<vector::FMAOp>(loc, kVec, inVec, outVec);
                        b.create<vector::StoreOp>(loc, fmaVec, output,
                                                  ValueRange{outOffset});
                        Value iNext =
                            b.create<arith::AddIOp>(loc, iv_i, vlStep);
                        b.create<scf::YieldOp>(loc, ValueRange{iNext});
                      })
                  .getResult(0);

          // 4.4 Process the remainder of tail process with scalar operations.
          Value tailUpboundScalar =
              builder.create<arith::AddIOp>(loc, iargs[0], vlStepMinusOne);
          builder.create<scf::ForOp>(
              loc, iterIdx, tailUpboundScalar, c1, ValueRange{std::nullopt},
              [&](OpBuilder &b, Location loc, Value iv_i, ValueRange iargs) {
                Value inElem = b.create<memref::LoadOp>(loc, input, iv_i);
                Value outOffset = b.create<arith::AddIOp>(loc, iv_i, iv_n);
                Value outElem =
                    b.create<memref::LoadOp>(loc, output, outOffset);
                Value mulElem = b.create<arith::MulFOp>(loc, inElem, kElem);
                Value addElem = b.create<arith::AddFOp>(loc, mulElem, outElem);
                b.create<memref::StoreOp>(loc, addElem, output, outOffset);
                b.create<scf::YieldOp>(loc);
              });
          Value tailUpboundNext =
              builder.create<arith::SubIOp>(loc, iargs[0], c1);
          builder.create<scf::YieldOp>(loc, ValueRange{tailUpboundNext});
        });

    rewriter.eraseOp(op);
    return success();
  }
};

class DAPIirVectorization : public OpRewritePattern<dap::IirOp> {
public:
  using OpRewritePattern<dap::IirOp>::OpRewritePattern;

  explicit DAPIirVectorization(MLIRContext *context)
      : OpRewritePattern(context) {}

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
    Value c8 = rewriter.create<ConstantIndexOp>(loc, 8);
    Value c16 = rewriter.create<ConstantIndexOp>(loc, 16);
    Value c32 = rewriter.create<ConstantIndexOp>(loc, 32);

    Value N = rewriter.create<memref::DimOp>(loc, input, c0);
    Value filterSize = rewriter.create<memref::DimOp>(loc, kernel, c0);

    FloatType f32 = Float32Type::get(ctx);
    Value f0 = rewriter.create<ConstantFloatOp>(loc, APFloat(0.0f), f32);
    Value f1 = rewriter.create<ConstantFloatOp>(loc, APFloat(1.0f), f32);

    Value cond4 =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::ule, filterSize, c4);
    Value cond8 =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::ule, filterSize, c8);
    Value cond16 =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::ule, filterSize, c16);
    Value cond32 =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::ule, filterSize, c32);

    // clang-format off
    rewriter.create<scf::IfOp>(loc, cond4,
    /*thenBuilder=*/
    [&](OpBuilder &builder, Location loc) {
        dap::iirVectorizationProcess(builder, loc, 4, f32, f0, f1, c0, c1, c2, c4, c5,
                                     filterSize, kernel, ArrayRef<int64_t>{0, 0, 1, 2},
                                     N, input, output);

        builder.create<scf::YieldOp>(loc);
    },
    /*elseBuilder=*/
    [&](OpBuilder &builder, Location loc) {
        builder.create<scf::IfOp>(loc, cond8,
        /*thenBuilder=*/
        [&](OpBuilder &builder, Location loc){
            dap::iirVectorizationProcess(builder, loc, 8, f32, f0, f1, c0, c1, c2, c4, c5,
                                         filterSize, kernel,
                                         ArrayRef<int64_t>{0, 0, 1, 2, 3, 4, 5, 6}, N,
                                         input, output);

            builder.create<scf::YieldOp>(loc);
        },
        /*elseBuilder=*/
        [&](OpBuilder &builder, Location loc) {
            builder.create<scf::IfOp>(loc, cond16,
            /*thenBuilder=*/
            [&](OpBuilder &builder, Location loc){
                dap::iirVectorizationProcess(builder, loc, 16, f32, f0, f1, c0, c1, c2, c4, c5,
                                             filterSize, kernel, ArrayRef<int64_t>{0, 0, 1, 2,
                                             3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, N,
                                             input, output);

                builder.create<scf::YieldOp>(loc);
            },
            /*elseBuilder=*/
            [&](OpBuilder &builder, Location loc) {
                builder.create<scf::IfOp>(loc, cond32,
                /*thenBuilder=*/
                [&](OpBuilder &builder, Location loc){
                    dap::iirVectorizationProcess(builder, loc, 32, f32, f0, f1, c0, c1, c2, c4, c5,
                                                 filterSize, kernel, ArrayRef<int64_t>{0, 0, 1, 2,
                                                 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                                 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                 30}, N, input, output);

                    builder.create<scf::YieldOp>(loc);
                },
                /*elseBuilder=*/
                [&](OpBuilder &builder, Location loc) {
                    dap::iirVectorizationProcess(builder, loc, 64, f32, f0, f1, c0, c1, c2, c4, c5,
                                                 filterSize, kernel, ArrayRef<int64_t>{0, 0, 1, 2, 3, 4,
                                                 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                                                 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                                 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                                                 62}, N, input, output);

                    builder.create<scf::YieldOp>(loc);
                }
                );
                builder.create<scf::YieldOp>(loc);
            });

            builder.create<scf::YieldOp>(loc);
        });

        builder.create<scf::YieldOp>(loc);
    });
    // clang-format on

    rewriter.eraseOp(op);
    return success();
  }
};

} // end anonymous namespace

void populateVectorizeDAPConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<DAPFIRVectorization>(patterns.getContext());
  patterns.add<DAPIirVectorization>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// VectorizeDAPPass
//===----------------------------------------------------------------------===//

namespace {
class VectorizeDAPPass
    : public PassWrapper<VectorizeDAPPass, OperationPass<ModuleOp>> {
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
                    affine::AffineDialect, arith::ArithDialect,
                    linalg::LinalgDialect>();
  }
};
} // end anonymous namespace.

void VectorizeDAPPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  // clang-format off
  target.addLegalDialect<
      affine::AffineDialect,
      scf::SCFDialect,
      func::FuncDialect,
      memref::MemRefDialect,
      VectorDialect,
      arith::ArithDialect,
      linalg::LinalgDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  // clang-format on

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
