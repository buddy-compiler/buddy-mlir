//===- MatMulOptimize.cpp -------------------------------------------------===//
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
// This file implements the matmul optimization.
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

class MatMulOptimizePattern : public ConversionPattern {
public:
  explicit MatMulOptimizePattern(MLIRContext *context, int64_t vecSizeParam,
                                 int64_t kernelMParam, int64_t kernelNParam)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), 1, context) {
    vecSize = vecSizeParam;
    kernelM = kernelMParam;
    kernelN = kernelNParam;
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

    // Some constants.
    const Value c0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value c1 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    const AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
    const AffineExpr s1 = rewriter.getAffineSymbolExpr(1);
    const AffineExpr d0 = rewriter.getAffineDimExpr(0);
    const AffineExpr d1 = rewriter.getAffineDimExpr(1);
    const AffineMap mapBroadcast =
        AffineMap::get(2, 0, rewriter.getAffineConstantExpr(0));
    const VectorType vTy = VectorType::get(16, ATy.getElementType());

    // Configs
    int64_t kNLen = vecSize * kernelN;

    // Dims
    Value M = rewriter.create<memref::DimOp>(loc, A, 0);
    Value N = rewriter.create<memref::DimOp>(loc, B, 1);
    Value K = rewriter.create<memref::DimOp>(loc, A, 1);

    // build loop body
    affine::buildAffineLoopNest(
        rewriter, loc, {c0}, {N}, kNLen,
        [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
          auto ivJ = ivRange.front();
          affine::buildAffineLoopNest(
              builder, loc, {c0}, {M}, kernelM,
              [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
                Value ivI = ivRange.front();
                SmallVector<memref::SubViewOp> aptrs;
                SmallVector<memref::SubViewOp> cptrs;
                for (int i = 0; i < kernelM; ++i) {
                  Value fixedIV = ivI;
                  if (i != 0) {
                    fixedIV = builder.create<affine::AffineMinOp>(
                        loc,
                        AffineMap::get(1, 1, {d0 + i, s0 - 1},
                                       builder.getContext()),
                        SmallVector<Value>{ivI, M});
                  }
                  MemRefType resTy =
                      MemRefType::get(ATy.getShape(), ATy.getElementType(),
                                      AffineMap::get(2, 2, d0 * s1 + s0 + d1));
                  auto aptr = builder.create<memref::SubViewOp>(
                      loc, resTy, A, SmallVector<OpFoldResult>{fixedIV, c0},
                      SmallVector<OpFoldResult>{c1, K},
                      SmallVector<OpFoldResult>{c1, c1});
                  aptrs.push_back(aptr);
                }
                for (int i = 0; i < kernelM; ++i) {
                  Value fixedIV = builder.create<affine::AffineMinOp>(
                      loc,
                      AffineMap::get(1, 1, {d0 + i, s0 - 1},
                                     builder.getContext()),
                      SmallVector<Value>{ivI, M});
                  MemRefType resTy =
                      MemRefType::get(ATy.getShape(), ATy.getElementType(),
                                      AffineMap::get(2, 2, d0 * s1 + s0 + d1));
                  auto cptr = builder.create<memref::SubViewOp>(
                      loc, resTy, C, SmallVector<OpFoldResult>{fixedIV, c0},
                      SmallVector<OpFoldResult>{c1, N},
                      SmallVector<OpFoldResult>{c1, c1});
                  cptrs.push_back(cptr);
                }
                affine::buildAffineLoopNest(
                    builder, loc, {c0}, {K}, 1,
                    [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
                      Value ivK = ivRange.front();
                      SmallVector<Value> as;
                      SmallVector<Value> bs;
                      for (int i = 0; i < kernelM; ++i) {
                        Value a = builder.create<TransferReadOp>(
                            loc, vTy, aptrs[i], ValueRange{c0, ivK},
                            mapBroadcast);
                        as.push_back(a);
                      }
                      SmallVector<Value> ds;
                      for (int i = 0; i < kernelM; ++i) {
                        Value c = cptrs[i];
                        for (int j = 0; j < kernelN; ++j) {
                          Value fixedIV = builder.create<affine::AffineApplyOp>(
                              loc, AffineMap::get(1, 0, d0 + j * vecSize), ivJ);
                          Value d = builder.create<TransferReadOp>(
                              loc, vTy, c, ValueRange{c0, fixedIV});
                          ds.push_back(d);
                        }
                      }
                      for (int i = 0; i < kernelN; ++i) {
                        Value fixedIV = ivJ;
                        if (i != 0) {
                          fixedIV = builder.create<affine::AffineApplyOp>(
                              loc, AffineMap::get(1, 0, d0 + i * vecSize), ivJ);
                        }
                        Value b = builder.create<TransferReadOp>(
                            loc, vTy, B, ValueRange{ivK, fixedIV});
                        bs.push_back(b);
                      }

                      for (int i = 0; i < kernelM; ++i) {
                        for (int j = 0; j < kernelN; ++j) {
                          ds[i * kernelN + j] = builder.create<vector::FMAOp>(
                              loc, vTy, as[i], bs[j], ds[i * kernelN + j]);
                        }
                      }

                      for (int i = 0; i < kernelM; ++i) {
                        for (int j = 0; j < kernelN; ++j) {
                          Value fixedIV = builder.create<affine::AffineApplyOp>(
                              loc, AffineMap::get(1, 0, d0 + j * vecSize), ivJ);
                          builder.create<TransferWriteOp>(
                              loc, ds[i * kernelN + j], cptrs[i],
                              ValueRange{c0, fixedIV});
                        }
                      }
                    });
              });
        });

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t vecSize;
  int64_t kernelM;
  int64_t kernelN;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// MatMulOptimizePass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg pooling operations to mixture of
/// Affine + Vector operations.
namespace {
class MatMulOptimizePass
    : public PassWrapper<MatMulOptimizePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulOptimizePass)
  StringRef getArgument() const final { return "matmul-optimize"; }
  StringRef getDescription() const final { return "MatMul Optimization."; }
  MatMulOptimizePass() = default;
  MatMulOptimizePass(const MatMulOptimizePass &) {}
  explicit MatMulOptimizePass(int64_t vecSizeParam, int64_t kernelMParam,
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

void MatMulOptimizePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                       scf::SCFDialect, memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<MatMulOptimizePattern>(context, vecSize, kernelM, kernelN);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerMatMulOptimizePass() { PassRegistration<MatMulOptimizePass>(); }
} // namespace buddy
} // namespace mlir
