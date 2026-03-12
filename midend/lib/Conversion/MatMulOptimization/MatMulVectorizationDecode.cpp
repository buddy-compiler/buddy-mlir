//===- MatMulVectorizationDecode.cpp --------------------------------------===//
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
// This file implements a specialized matmul vectorization for decode
// workloads, where the `m` dimension is statically known to be 1.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
class MatMulVectorizationDecodePattern : public ConversionPattern {
public:
  MatMulVectorizationDecodePattern(MLIRContext *ctx, int64_t vecSize,
                                   bool scalableParam)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), 1, ctx),
        vecSize(vecSize), scalable(scalableParam) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto matmulOp = cast<linalg::MatmulOp>(op);
    auto loc = matmulOp.getLoc();

    Value A = matmulOp.getInputs()[0];
    Value B = matmulOp.getInputs()[1];
    Value C = matmulOp.getOutputs()[0];

    auto aType = dyn_cast<MemRefType>(A.getType());
    auto bType = dyn_cast<MemRefType>(B.getType());
    auto cType = dyn_cast<MemRefType>(C.getType());
    if (!aType || !bType || !cType)
      return failure();
    if (!aType.hasStaticShape() || aType.getRank() != 2)
      return failure();
    if (aType.getDimSize(0) != 1)
      return failure();
    if (cType.getRank() != 2 || cType.getDimSize(0) != 1)
      return failure();

    // Type elementType = cType.getElementType();
    // auto vectorType = VectorType::get({vecSize}, elementType, {scalable});
    // Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    // Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    // Value step = rewriter.create<arith::ConstantIndexOp>(loc, vecSize);
    int64_t nDim = cType.getDimSize(1);
    int64_t dynamicVecSize = vecSize;
    if (nDim != ShapedType::kDynamic) {
      int64_t suggestedSize = (nDim <= 256) ? 8 : (nDim >= 8192 ? 64 : 32);
      dynamicVecSize = std::min({suggestedSize, vecSize, nDim});
    }
    Type elementType = cType.getElementType();
    auto vectorType =
        VectorType::get({dynamicVecSize}, elementType, {scalable});
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, dynamicVecSize);

    if (scalable) {
      Value vscale = rewriter.create<vector::VectorScaleOp>(loc);
      step = rewriter.create<arith::MulIOp>(loc, step, vscale);
    }

    Value n = rewriter.create<memref::DimOp>(loc, C, c1);
    Value k = rewriter.create<memref::DimOp>(loc, A, c1);

    auto parOp = rewriter.create<scf::ParallelOp>(
        loc,
        /*lowerBounds=*/ValueRange{c0},
        /*upperBounds=*/ValueRange{n},
        /*steps=*/ValueRange{step},
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          Value nIdx = ivs.front();
          Value cVec = builder.create<vector::LoadOp>(loc, vectorType, C,
                                                      ValueRange{c0, nIdx});
          auto sumIter = builder.create<scf::ForOp>(
              loc, c0, k, c1, ValueRange{cVec},
              [&](OpBuilder &builder, Location loc, Value kIdx,
                  ValueRange iterArgs) {
                Value aElem = builder.create<memref::LoadOp>(
                    loc, A, ValueRange{c0, kIdx});
                Value aVec =
                    builder.create<vector::BroadcastOp>(loc, vectorType, aElem);
                Value bVec = builder.create<vector::LoadOp>(
                    loc, vectorType, B, ValueRange{kIdx, nIdx});
                Value res = builder.create<vector::FMAOp>(loc, aVec, bVec,
                                                          iterArgs.front());
                builder.create<scf::YieldOp>(loc, res);
              });

          builder.create<vector::StoreOp>(loc, sumIter.getResult(0), C,
                                          ValueRange{c0, nIdx});
        });

    rewriter.eraseOp(op);

    return success();
  }

private:
  int64_t vecSize;
  bool scalable;
};

class MatMulVectorizationDecodePass
    : public PassWrapper<MatMulVectorizationDecodePass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulVectorizationDecodePass)

  StringRef getArgument() const final { return "matmul-vectorization-decode"; }
  StringRef getDescription() const final {
    return "Vectorize linalg.matmul with m==1 for decode workloads.";
  }

  MatMulVectorizationDecodePass() = default;
  MatMulVectorizationDecodePass(const MatMulVectorizationDecodePass &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                           scf::SCFDialect, vector::VectorDialect,
                           func::FuncDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
    target.addDynamicallyLegalOp<linalg::MatmulOp>(
        [&](linalg::MatmulOp op) -> bool {
          auto aType = dyn_cast<MemRefType>(op.getInputs()[0].getType());
          auto cType = dyn_cast<MemRefType>(op.getOutputs()[0].getType());
          if (!aType || !cType)
            return true;
          if (!aType.hasStaticShape() || aType.getRank() != 2)
            return true;
          if (aType.getDimSize(0) != 1)
            return true;
          if (cType.getRank() != 2 || cType.getDimSize(0) != 1)
            return true;
          return false;
        });

    RewritePatternSet patterns(context);
    bool isScalable = (vectorType == "scalable");
    patterns.add<MatMulVectorizationDecodePattern>(context, vectorSize,
                                                   isScalable);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }

  Option<int64_t> vectorSize{
      *this, "vector-size",
      llvm::cl::desc("Specify the vector width for n-dimension iteration."),
      llvm::cl::init(32)};
  Option<std::string> vectorType{
      *this, "vector-type",
      llvm::cl::desc("Specify vector type: fixed or scalable."),
      llvm::cl::init("fixed")};
};
} // namespace

namespace mlir {
namespace buddy {
void registerMatMulVectorizationDecodePass() {
  PassRegistration<MatMulVectorizationDecodePass>();
}
} // namespace buddy
} // namespace mlir
