//===- BatchMatMulOptimize.cpp --------------------------------------------===//
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
// This file implements the batchmatmul optimization.
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
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <mlir/Dialect/Affine/Analysis/AffineAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/Transforms/VectorTransforms.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
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

class BatchMatMulTransVecPattern : public ConversionPattern {
public:
  explicit BatchMatMulTransVecPattern(MLIRContext *context,
                                      int64_t vecSizeParam,
                                      bool scalableParam)
      : ConversionPattern(linalg::BatchMatmulTransposeBOp::getOperationName(),
                          1, context) {
    vecSize = vecSizeParam;
    scalable = scalableParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Retrieve input tensors A, B, and C.
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);

    // Acquire the element type of input tensors.
    Type elementType = A.getType().cast<MemRefType>().getElementType();
    VectorType vectorTy = mlir::VectorType::get({vecSize}, elementType, {scalable});

    // Define constants.
    const Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    const Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    const Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    Value vlStep = rewriter.create<arith::ConstantIndexOp>(loc, vecSize);
    if (scalable) {
      Value vscale = rewriter.create<vector::VectorScaleOp>(loc);
      vlStep = rewriter.create<arith::MulIOp>(loc, vlStep, vscale);
    }
    const Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));

    // Create initial zero vector for accumulation.
    Value zeroVec = rewriter.create<SplatOp>(loc, vectorTy, zero);

    // Get dimensions of input tensors.
    Value batch = rewriter.create<memref::DimOp>(loc, A, c0);
    Value aRow = rewriter.create<memref::DimOp>(loc, A, c1);
    Value bRow = rewriter.create<memref::DimOp>(loc, A, c2);
    Value bCol = rewriter.create<memref::DimOp>(loc, B, c1);

    // Compute main (vectorized) part and tail part sizes along reduction dim.
    Value tailSize = rewriter.create<arith::RemUIOp>(loc, bRow, vlStep);
    Value mainSize = rewriter.create<arith::SubIOp>(loc, bRow, tailSize);

    // Create nested scf.for loops instead of affine loops.
    auto batchLoop = rewriter.create<scf::ForOp>(loc, c0, batch, c1);
    rewriter.setInsertionPointToStart(batchLoop.getBody());
    Value batchIdx = batchLoop.getInductionVar();

    auto aRowLoop = rewriter.create<scf::ForOp>(loc, c0, aRow, c1);
    rewriter.setInsertionPointToStart(aRowLoop.getBody());
    Value aRowIdx = aRowLoop.getInductionVar();

    auto bColLoop = rewriter.create<scf::ForOp>(loc, c0, bCol, c1);
    rewriter.setInsertionPointToStart(bColLoop.getBody());
    Value bColIdx = bColLoop.getInductionVar();

    // Inner vectorized loop using vector.fma for accumulation.
    auto vecLoop = rewriter.create<scf::ForOp>(
        loc, c0, mainSize, vlStep, ValueRange{zeroVec},
        [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
            ValueRange itrArgs) {
          Value aVec = nestedBuilder.create<vector::LoadOp>(
              nestedLoc, vectorTy, A, ValueRange{batchIdx, aRowIdx, iv});
          Value bVec = nestedBuilder.create<vector::LoadOp>(
              nestedLoc, vectorTy, B, ValueRange{batchIdx, bColIdx, iv});
          // Use vector.fma for fused multiply-add accumulation.
          Value resultVec;
          if (isa<IntegerType>(elementType)) {
            // For integer types, use mul + add since fma doesn't apply.
            Value mulVec =
                nestedBuilder.create<arith::MulIOp>(nestedLoc, aVec, bVec);
            resultVec = nestedBuilder.create<arith::AddIOp>(nestedLoc, mulVec,
                                                            itrArgs[0]);
          } else {
            // For floating point types, use vector.fma.
            resultVec = nestedBuilder.create<vector::FMAOp>(nestedLoc, aVec,
                                                            bVec, itrArgs[0]);
          }
          nestedBuilder.create<scf::YieldOp>(nestedLoc, resultVec);
        });

    // Load the initial value from output memref.
    Value initVal = rewriter.create<memref::LoadOp>(
        loc, elementType, C, ValueRange{batchIdx, aRowIdx, bColIdx});

    // Perform reduction on the accumulated vector (main vectorized part).
    Value partialResult = rewriter.create<vector::ReductionOp>(
        loc, vector::CombiningKind::ADD, vecLoop.getResult(0), initVal,
        ::mlir::arith::FastMathFlags::reassoc);

    // Tail processing for remaining elements that do not fit into a full
    // vector.
    auto tailLoop = rewriter.create<scf::ForOp>(
        loc, mainSize, bRow, c1, ValueRange{partialResult},
        [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
            ValueRange itrArgs) {
          Value aElem = nestedBuilder.create<memref::LoadOp>(
              nestedLoc, elementType, A, ValueRange{batchIdx, aRowIdx, iv});
          Value bElem = nestedBuilder.create<memref::LoadOp>(
              nestedLoc, elementType, B, ValueRange{batchIdx, bColIdx, iv});
          Value resultScalar;
          if (isa<IntegerType>(elementType)) {
            Value mulVal =
                nestedBuilder.create<arith::MulIOp>(nestedLoc, aElem, bElem);
            resultScalar = nestedBuilder.create<arith::AddIOp>(
                nestedLoc, mulVal, itrArgs[0]);
          } else {
            Value mulVal =
                nestedBuilder.create<arith::MulFOp>(nestedLoc, aElem, bElem);
            resultScalar = nestedBuilder.create<arith::AddFOp>(
                nestedLoc, mulVal, itrArgs[0]);
          }
          nestedBuilder.create<scf::YieldOp>(nestedLoc, resultScalar);
        });

    Value finalResult = tailLoop.getResult(0);

    // Store the result back.
    rewriter.create<memref::StoreOp>(loc, finalResult, C,
                                     ValueRange{batchIdx, aRowIdx, bColIdx});
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t vecSize;
  bool scalable;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// BatchMatMulTransVecPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg pooling operations to mixture of
/// Affine + Vector operations.
namespace {
class BatchMatMulTransVecPass
    : public PassWrapper<BatchMatMulTransVecPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BatchMatMulTransVecPass)
  StringRef getArgument() const final {
    return "batchmatmul-transpose-b-vectorization";
  }
  StringRef getDescription() const final {
    return "BatchMatMulTransposeBOp vectorization.";
  }
  BatchMatMulTransVecPass() = default;
  BatchMatMulTransVecPass(const BatchMatMulTransVecPass &) {}
  explicit BatchMatMulTransVecPass(int64_t vecSizeParam) {
    vecSize = vecSizeParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, VectorDialect>();
  }

  Option<int64_t> vecSize{*this, "vector-size",
                          llvm::cl::desc("Affine Vector size."),
                          llvm::cl::init(32)};
  Option<std::string> vectorType{
      *this, "vector-type",
      llvm::cl::desc("Specify vector type: fixed or scalable."),
      llvm::cl::init("fixed")};
};
} // end anonymous namespace.

void BatchMatMulTransVecPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                       scf::SCFDialect, memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  bool isScalable = (vectorType == "scalable");
  patterns.add<BatchMatMulTransVecPattern>(context, vecSize, isScalable);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerBatchMatMulTransVecPass() {
  PassRegistration<BatchMatMulTransVecPass>();
}
} // namespace buddy
} // namespace mlir
