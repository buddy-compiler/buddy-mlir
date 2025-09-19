//===- MatMulVec.cpp ------------------------------------------------------===//
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
// This file implements the MatMulVec pass for AVX-512 vectorization.
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
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

class MatMulVecPattern : public ConversionPattern {
public:
  explicit MatMulVecPattern(MLIRContext *context, int64_t vecSizeParam)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), 1, context) {
    vecSize = vecSizeParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);

    ShapedType ATy = mlir::cast<ShapedType>(A.getType());
    Type eleTy = ATy.getElementType();

    MLIRContext *ctx = op->getContext();

    VectorType vectorTy = mlir::VectorType::get({vecSize}, eleTy);

    Value c0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    Value c1 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, vecSize);

    Value c0Ele = buddy::insertZeroConstantOp(ctx, rewriter, loc, eleTy);

    Value aRow = rewriter.create<memref::DimOp>(loc, A, c0); // M = 40
    Value bRow = rewriter.create<memref::DimOp>(loc, B, c0); // K = 3072
    Value bCol = rewriter.create<memref::DimOp>(loc, B, c1); // N = 1536

    // for i = 0 to M (40)
    rewriter.create<scf::ForOp>(
        loc, c0, aRow, c1, ValueRange{},
        [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
          // for j = 0 to N step vecSize (1536 step 16)
          builder.create<scf::ForOp>(
              loc, c0, bCol, step, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value j,
                  ValueRange iterArgs) {
                Value accVec = builder.create<vector::TransferReadOp>(
                    loc, vectorTy, C, ValueRange{i, j}, c0Ele);

                // for k = 0 to K (3072)
                auto forOp = builder.create<scf::ForOp>(
                    loc, c0, bRow, c1, ValueRange{accVec},
                    [&](OpBuilder &builder, Location loc, Value k,
                        ValueRange iterArgs) {
                      Value currentAcc = iterArgs[0];

                      Value aScalar = builder.create<memref::LoadOp>(
                          loc, A, ValueRange{i, k});
                      Value aVec = builder.create<vector::BroadcastOp>(
                          loc, vectorTy, aScalar);

                      Value bVec = builder.create<vector::TransferReadOp>(
                          loc, vectorTy, B, ValueRange{k, j}, c0Ele);

                      // acc = a * b + acc
                      Value newAcc =
                          builder.create<FMAOp>(loc, aVec, bVec, currentAcc);

                      builder.create<scf::YieldOp>(loc, ValueRange{newAcc});
                    });

                builder.create<vector::TransferWriteOp>(loc, forOp.getResult(0),
                                                        C, ValueRange{i, j});

                builder.create<scf::YieldOp>(loc);
              });
          builder.create<scf::YieldOp>(loc);
        });

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t vecSize;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// MatMulVecPass
//===----------------------------------------------------------------------===//

namespace {
class MatMulVecPass
    : public PassWrapper<MatMulVecPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulVecPass)
  StringRef getArgument() const final { return "matmul-vectorization-512"; }
  StringRef getDescription() const final {
    return "MatMul AVX-512 Vectorization.";
  }
  MatMulVecPass() = default;
  MatMulVecPass(const MatMulVecPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, VectorDialect>();
  }

  Option<int64_t> vecSize{*this, "vector-size",
                          llvm::cl::desc("Specify vector size for AVX-512."),
                          llvm::cl::init(16)};
};
} // end anonymous namespace

void MatMulVecPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithDialect, scf::SCFDialect,
                         memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp,
                    linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<MatMulVecPattern>(context, vecSize);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerMatMulVecPass() { PassRegistration<MatMulVecPass>(); }
} // namespace buddy
} // namespace mlir
