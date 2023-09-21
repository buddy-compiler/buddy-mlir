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

class BatchMatMulOptimizePattern : public ConversionPattern {
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
    Value Batch = rewriter.create<memref::DimOp>(loc, A, 0);
    Value M = rewriter.create<memref::DimOp>(loc, A, 1);
    Value N = rewriter.create<memref::DimOp>(loc, B, 2);
    Value K = rewriter.create<memref::DimOp>(loc, A, 2);

    // build loop body
    affine::buildAffineLoopNest(
        rewriter, loc, {c0}, {Batch}, 1,
        [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
          // not impletment
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

namespace mlir {
namespace buddy {
void registerBatchMatMulOptimizePass() {
  PassRegistration<BatchMatMulOptimizePass>();
}
} // namespace buddy
} // namespace mlir
