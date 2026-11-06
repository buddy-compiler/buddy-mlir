//===- BgeM3BatchMatMulTransposeBVec.cpp ----------------------------------===//
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
// BGE-M3 attention Q@K^T / scores@V batch matmul vectorization.
//
// This is a BGE-M3 specific variant of the transposed-B batch matmul
// vectorization. It is kept in a separate file so that the generic
// BatchMatMulTransBVec.cpp stays untouched: this variant additionally
// guards the rewrite with a `linalg::BatchMatmulTransposeBOp` dyn_cast, so
// default-map batch matmuls are never mis-lowered, and it defaults to
// vector-size=64 which matches the x86 AVX-512 (and RVV 512-bit) targets.
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
#include "llvm/ADT/SmallVector.h"
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

/// Checks the shared shape preconditions for transposed-B batch matmul
/// lowering: memref operands, rank 3, uniform integer/float element type.
static bool isSupportedTransBShape(Operation *op) {
  auto aType = dyn_cast<MemRefType>(op->getOperand(0).getType());
  auto bType = dyn_cast<MemRefType>(op->getOperand(1).getType());
  auto cType = dyn_cast<MemRefType>(op->getOperand(2).getType());
  if (!aType || !bType || !cType)
    return false;
  if (aType.getRank() != 3 || bType.getRank() != 3 || cType.getRank() != 3)
    return false;
  Type elementType = aType.getElementType();
  if (bType.getElementType() != elementType ||
      cType.getElementType() != elementType)
    return false;
  return isa<IntegerType, FloatType>(elementType);
}

class BgeM3BatchMatMulTransVecPattern : public ConversionPattern {
public:
  explicit BgeM3BatchMatMulTransVecPattern(MLIRContext *context,
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
    // `BatchMatmulTransposeBOp` shares the `linalg.batch_matmul` op name and
    // is only a typed wrapper; dyn_cast enforces the transposed-B indexing
    // maps so that default-map batch matmuls are left untouched (they are
    // handled by `batchmatmul-optimize`).
    if (!dyn_cast<linalg::BatchMatmulTransposeBOp>(op))
      return failure();
    if (!isSupportedTransBShape(op))
      return failure();

    auto loc = op->getLoc();

    // Retrieve input tensors A, B, and C.
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);

    // Acquire the element type of input tensors.
    Type elementType =
        mlir::cast<mlir::MemRefType>(A.getType()).getElementType();
    VectorType vectorTy =
        mlir::VectorType::get({vecSize}, elementType, {scalable});

    // Define constants.
    const Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    const Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    const Value c2 = arith::ConstantIndexOp::create(rewriter, loc, 2);
    Value vlStep = arith::ConstantIndexOp::create(rewriter, loc, vecSize);
    if (scalable) {
      Value vscale = vector::VectorScaleOp::create(rewriter, loc);
      vlStep = arith::MulIOp::create(rewriter, loc, vlStep, vscale);
    }
    const Value zero = arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(elementType));

    // Create initial zero vector for accumulation.
    Value zeroVec = vector::BroadcastOp::create(rewriter, loc, vectorTy, zero);

    // Get dimensions of input tensors.
    Value batch = memref::DimOp::create(rewriter, loc, A, c0);
    Value aRow = memref::DimOp::create(rewriter, loc, A, c1);
    Value bRow = memref::DimOp::create(rewriter, loc, A, c2);
    Value bCol = memref::DimOp::create(rewriter, loc, B, c1);

    // Compute main (vectorized) part and tail part sizes along reduction dim.
    Value tailSize = arith::RemUIOp::create(rewriter, loc, bRow, vlStep);
    Value mainSize = arith::SubIOp::create(rewriter, loc, bRow, tailSize);

    // Create nested scf.for loops instead of affine loops.
    auto batchLoop = scf::ForOp::create(rewriter, loc, c0, batch, c1);
    rewriter.setInsertionPointToStart(batchLoop.getBody());
    Value batchIdx = batchLoop.getInductionVar();

    auto aRowLoop = scf::ForOp::create(rewriter, loc, c0, aRow, c1);
    rewriter.setInsertionPointToStart(aRowLoop.getBody());
    Value aRowIdx = aRowLoop.getInductionVar();

    auto bColLoop = scf::ForOp::create(rewriter, loc, c0, bCol, c1);
    rewriter.setInsertionPointToStart(bColLoop.getBody());
    Value bColIdx = bColLoop.getInductionVar();

    // Inner vectorized loop using vector.fma for accumulation.
    auto vecLoop = scf::ForOp::create(
        rewriter, loc, c0, mainSize, vlStep, ValueRange{zeroVec},
        [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
            ValueRange itrArgs) {
          Value aVec =
              vector::LoadOp::create(nestedBuilder, nestedLoc, vectorTy, A,
                                     ValueRange{batchIdx, aRowIdx, iv});
          Value bVec =
              vector::LoadOp::create(nestedBuilder, nestedLoc, vectorTy, B,
                                     ValueRange{batchIdx, bColIdx, iv});
          // Use vector.fma for fused multiply-add accumulation.
          Value resultVec;
          if (isa<IntegerType>(elementType)) {
            // For integer types, use mul + add since fma doesn't apply.
            Value mulVec =
                arith::MulIOp::create(nestedBuilder, nestedLoc, aVec, bVec);
            resultVec = arith::AddIOp::create(nestedBuilder, nestedLoc, mulVec,
                                              itrArgs[0]);
          } else {
            // For floating point types, use vector.fma.
            resultVec = vector::FMAOp::create(nestedBuilder, nestedLoc, aVec,
                                              bVec, itrArgs[0]);
          }
          scf::YieldOp::create(nestedBuilder, nestedLoc, resultVec);
        });

    // Load the initial value from output memref.
    Value initVal = memref::LoadOp::create(
        rewriter, loc, elementType, C, ValueRange{batchIdx, aRowIdx, bColIdx});

    // Perform reduction on the accumulated vector (main vectorized part).
    Value partialResult = vector::ReductionOp::create(
        rewriter, loc, vector::CombiningKind::ADD, vecLoop.getResult(0),
        initVal, ::mlir::arith::FastMathFlags::reassoc);

    // Tail processing for remaining elements that do not fit into a full
    // vector.
    auto tailLoop = scf::ForOp::create(
        rewriter, loc, mainSize, bRow, c1, ValueRange{partialResult},
        [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
            ValueRange itrArgs) {
          Value aElem =
              memref::LoadOp::create(nestedBuilder, nestedLoc, elementType, A,
                                     ValueRange{batchIdx, aRowIdx, iv});
          Value bElem =
              memref::LoadOp::create(nestedBuilder, nestedLoc, elementType, B,
                                     ValueRange{batchIdx, bColIdx, iv});
          Value resultScalar;
          if (isa<IntegerType>(elementType)) {
            Value mulVal =
                arith::MulIOp::create(nestedBuilder, nestedLoc, aElem, bElem);
            resultScalar = arith::AddIOp::create(nestedBuilder, nestedLoc,
                                                 mulVal, itrArgs[0]);
          } else {
            Value mulVal =
                arith::MulFOp::create(nestedBuilder, nestedLoc, aElem, bElem);
            resultScalar = arith::AddFOp::create(nestedBuilder, nestedLoc,
                                                 mulVal, itrArgs[0]);
          }
          scf::YieldOp::create(nestedBuilder, nestedLoc, resultScalar);
        });

    Value finalResult = tailLoop.getResult(0);

    // Store the result back.
    memref::StoreOp::create(rewriter, loc, finalResult, C,
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
// BgeM3BatchMatMulTransVecPass
//===----------------------------------------------------------------------===//

/// Partial lowering of transposed-B linalg batch matmuls to a mixture of
/// SCF + Vector operations, guarded for the BGE-M3 attention patterns.
namespace {
class BgeM3BatchMatMulTransVecPass
    : public PassWrapper<BgeM3BatchMatMulTransVecPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BgeM3BatchMatMulTransVecPass)
  StringRef getArgument() const final {
    return "bge-m3-batchmatmul-transpose-b-vec";
  }
  StringRef getDescription() const final {
    return "BGE-M3 BatchMatMulTransposeBOp vectorization (guarded).";
  }
  BgeM3BatchMatMulTransVecPass() = default;
  BgeM3BatchMatMulTransVecPass(const BgeM3BatchMatMulTransVecPass &) {}
  explicit BgeM3BatchMatMulTransVecPass(int64_t vecSizeParam) {
    vecSize = vecSizeParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, VectorDialect>();
  }

  Option<int64_t> vecSize{*this, "vector-size",
                          llvm::cl::desc("Affine Vector size."),
                          llvm::cl::init(64)};
  Option<std::string> vectorType{
      *this, "vector-type",
      llvm::cl::desc("Specify vector type: fixed or scalable."),
      llvm::cl::init("fixed")};
};
} // end anonymous namespace.

void BgeM3BatchMatMulTransVecPass::runOnOperation() {
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
  patterns.add<BgeM3BatchMatMulTransVecPattern>(context, vecSize, isScalable);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerBgeM3BatchMatMulTransVecPass() {
  PassRegistration<BgeM3BatchMatMulTransVecPass>();
}
} // namespace buddy
} // namespace mlir
