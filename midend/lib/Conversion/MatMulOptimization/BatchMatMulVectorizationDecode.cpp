//===---------- BatchMatMulVectorizationDecode.cpp ------------------------===//
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
// This file implements the BatchMatMul vectorization decode optimization.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
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
using namespace affine;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class BatchMatMulVectorizationDecodePattern : public ConversionPattern {
public:
  explicit BatchMatMulVectorizationDecodePattern(MLIRContext *context,
                                                 int64_t vecSizeParam,
                                                 bool scalableParam)
      : ConversionPattern(linalg::BatchMatmulOp::getOperationName(), 1,
                          context) {
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
    Type elementType =
    mlir::cast<mlir::MemRefType>(A.getType()).getElementType();
    VectorType vectorTy = mlir::VectorType::get({vecSize}, elementType, {scalable});

    // Define constants.
    llvm::SmallVector<Value, 8> constantVals;
    for (int i = 0; i <= 8; ++i) {
      auto val =
          arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(i));
      constantVals.push_back(val);
    }
    Value vlStep = arith::ConstantIndexOp::create(rewriter, loc, vecSize);
    if (scalable) {
      Value vscale = vector::VectorScaleOp::create(rewriter, loc);
      vlStep = arith::MulIOp::create(rewriter, loc, vlStep, vscale);
    }

    // Get dimensions of input tensors.
    Value batch = memref::DimOp::create(rewriter, loc, A, constantVals[0]);
    Value aCol = memref::DimOp::create(rewriter, loc, A, constantVals[2]);
    Value bCol = memref::DimOp::create(rewriter, loc, C, constantVals[2]);

    // Decode-specialized loop structure:
    // scf.parallel (b) = (0) to (batch) step (1) {
    //   scf.for n = 0 to bCol step vlStep {
    //     c_vec = vector.load C[b, 0, n] : vector<vecSize x elt>
    //     %sum = scf.for k = 0 to aCol step 1 iter_args(%acc = c_vec)
    //          -> (vector<...>) {
    //       a_ele = memref.load A[b, 0, k]
    //       a_vec = vector.broadcast a_ele
    //       b_vec = vector.load B[b, k, n]
    //       r_vec = vector.fma a_vec, b_vec, %acc
    //       scf.yield r_vec
    //     }
    //     vector.store %sum, C[b, 0, n]
    //   }
    // }
    auto parOp = scf::ParallelOp::create(rewriter, 
        loc,
        /*lowerBounds=*/ValueRange{constantVals[0]},
        /*upperBounds=*/ValueRange{batch},
        /*steps=*/ValueRange{constantVals[1]},
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          Value bIdx = ivs[0];
          (void)builder;
          auto outerFor = scf::ForOp::create(rewriter, 
              loc, constantVals[0], bCol, vlStep, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value nIdx,
                  ValueRange /*iterArgs*/) {
                // Load initial C vector
                auto cVecInit = vector::LoadOp::create(rewriter, 
                    loc, vectorTy, C, ValueRange{bIdx, constantVals[0], nIdx});
                auto kFor = scf::ForOp::create(rewriter, 
                    loc, constantVals[0], aCol, constantVals[1],
                    ValueRange{cVecInit},
                    [&](OpBuilder &builder, Location loc, Value kIdx,
                        ValueRange accVecs) {
                      Value aEle = memref::LoadOp::create(rewriter, 
                          loc, A, ValueRange{bIdx, constantVals[0], kIdx});
                      Value aVec = vector::BroadcastOp::create(rewriter, 
                          loc, vectorTy, aEle);
                      Value bVec = vector::LoadOp::create(rewriter, 
                          loc, vectorTy, B, ValueRange{bIdx, kIdx, nIdx});
                      Value newAcc;
                      if (isa<IntegerType>(elementType)) {
                        Value mulVec =
                            arith::MulIOp::create(rewriter, loc, aVec, bVec);
                        newAcc = arith::AddIOp::create(rewriter, loc, mulVec,
                                                                accVecs[0]);
                      } else {
                        newAcc = vector::FMAOp::create(rewriter, loc, aVec, bVec,
                                                                accVecs[0]);
                      }
                      scf::YieldOp::create(builder, loc, ValueRange{newAcc});
                    });
                vector::StoreOp::create(rewriter, 
                    loc, kFor.getResult(0), C,
                    ValueRange{bIdx, constantVals[0], nIdx});
                scf::YieldOp::create(builder, loc);
              });
        });

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t vecSize;
  bool scalable;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// BatchMatMulVectorizationDecodePass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg pooling operations to mixture of
/// Affine + Vector operations.
namespace {
class BatchMatMulVectorizationDecodePass
    : public PassWrapper<BatchMatMulVectorizationDecodePass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      BatchMatMulVectorizationDecodePass)
  StringRef getArgument() const final {
    return "batch-matmul-vectorization-decode";
  }
  StringRef getDescription() const final {
    return "BatchMatMul Vectorization Decode Optimization.";
  }
  BatchMatMulVectorizationDecodePass() = default;
  BatchMatMulVectorizationDecodePass(
      const BatchMatMulVectorizationDecodePass &) {}
  explicit BatchMatMulVectorizationDecodePass(int64_t vecSizeParam) {
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

void BatchMatMulVectorizationDecodePass::runOnOperation() {
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
  patterns.add<BatchMatMulVectorizationDecodePattern>(context, vecSize, isScalable);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerBatchMatMulVectorizationDecodePass() {
  PassRegistration<BatchMatMulVectorizationDecodePass>();
}
} // namespace buddy
} // namespace mlir
