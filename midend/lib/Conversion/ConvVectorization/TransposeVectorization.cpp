//===--------TransposeVectorization.cpp-------------------------------===//
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
// This file implements the Transpoese Vectorization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/ArrayRef.h"
#include <mlir/Dialect/Affine/Analysis/AffineAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Vector/Transforms/VectorTransforms.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
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

class TransposeVectorizationPattern : public ConversionPattern {
public:
  explicit TransposeVectorizationPattern(MLIRContext *context,
                                         int64_t stripParam)
      : ConversionPattern(tosa::TransposeOp::getOperationName(), 1, context) {
    strip = stripParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Get perms
    Value perms = op->getOperand(1);
    TensorType permsTy = dyn_cast<TensorType>(perms.getType());
    if (!permsTy || !permsTy.getElementType().isInteger(32) ||
        permsTy.getShape() != llvm::ArrayRef<int64_t>({4})) {
      return failure();
    }

    // Check if perms is {0, 2, 1, 3}
    // DenseElementsAttr permsAttr =
    // perms.getDefiningOp<arith::ConstantOp>().getValue().cast<DenseElementsAttr>();
    // SmallVector<int32_t, 4> expectedPerms = {0, 2, 1, 3};
    // if (!std::equal(permsAttr.getValues<int32_t>().begin(),
    // permsAttr.getValues<int32_t>().end(), expectedPerms.begin())) {
    //   return failure();
    // }

    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Get i1 as the element type for mask vector.
    IntegerType i1 = IntegerType::get(ctx, 1);
    VectorType vectorMaskTy = mlir::VectorType::get({strip}, i1);

    // Get input.
    Value input = op->getOperand(0);

    // Use perms to create a memref::AllocaOp
    ShapedType inputTy = input.getType().cast<ShapedType>();
    SmallVector<int64_t> inputNumVec;
    for (auto in : inputTy.getShape())
      inputNumVec.push_back(in);

    auto tmpValue = inputNumVec[1];
    inputNumVec[1] = inputNumVec[2];
    inputNumVec[2] = tmpValue;
    llvm::ArrayRef<int64_t> outputNum(inputNumVec);
    // Get ElementType of input.
    Type elementTy = input.getType().cast<ShapedType>().getElementType();
    VectorType vectorTy = mlir::VectorType::get({strip}, elementTy);
    Value inputMem = rewriter.create<bufferization::ToMemrefOp>(
        loc, MemRefType::get(inputTy.getShape(), elementTy), input);
    Value alloc = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(outputNum, elementTy));

    // Get Constants.
    const Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    const Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    const Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    const Value c3 = rewriter.create<arith::ConstantIndexOp>(loc, 3);
    const Value vlStep = rewriter.create<arith::ConstantIndexOp>(loc, strip);
    const Value zero =
        buddy::insertZeroConstantOp(ctx, rewriter, loc, elementTy);

    // Create pass through vector.
    Value passThroughVec = rewriter.create<SplatOp>(loc, vectorTy, zero);

    Value inputDim0 = rewriter.create<memref::DimOp>(loc, inputMem, c0);
    Value inputDim1 = rewriter.create<memref::DimOp>(loc, inputMem, c1);
    Value inputDim2 = rewriter.create<memref::DimOp>(loc, inputMem, c2);
    Value inputDim3 = rewriter.create<memref::DimOp>(loc, inputMem, c3);

    // Calculate the upper bound for vectorized processing
    // - Subtract `vlStep` is to avoid overflow at the vectorization tail.
    // - Add 1 to ensure the final loop runs when the workload length
    //   is divisible by the vector size.
    Value upperBoundTmp =
        rewriter.create<arith::SubIOp>(loc, inputDim3, vlStep);
    Value upperBound = rewriter.create<arith::AddIOp>(loc, upperBoundTmp, c1);

    SmallVector<Value, 8> lowerBounds(3, c0);
    SmallVector<Value, 8> uperBounds{inputDim0, inputDim2, inputDim1};
    SmallVector<int64_t, 8> steps(3, /*Value=*/1);
    affine::buildAffineLoopNest(
        rewriter, loc, lowerBounds, uperBounds, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          // Create strip mining loop.
          auto iterIdx = builder.create<scf::ForOp>(
              loc, c0, upperBound, /*Step=*/vlStep, ValueRange{c0},
              [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
                  ValueRange itrArgs) {
                Value inputVector = nestedBuilder.create<vector::LoadOp>(
                    loc, vectorTy, inputMem,
                    ValueRange{ivs[0], ivs[2], ivs[1], iv});
                nestedBuilder.create<vector::StoreOp>(
                    loc, inputVector, alloc,
                    ValueRange{ivs[0], ivs[1], ivs[2], iv});
                Value idx =
                    nestedBuilder.create<arith::AddIOp>(loc, iv, vlStep);
                nestedBuilder.create<scf::YieldOp>(loc, idx);
              });
          // Compute the tail size and Process the remaining elements
          // using masked vector operations.
          Value idx = iterIdx.getResult(0);
          Value tailSize = builder.create<arith::SubIOp>(loc, inputDim3, idx);
          Value tailMask =
              builder.create<CreateMaskOp>(loc, vectorMaskTy, tailSize);
          // Masked load input.
          Value maskedOutputVec = builder.create<MaskedLoadOp>(
              loc, vectorTy, inputMem, ValueRange{ivs[0], ivs[2], ivs[1], idx},
              tailMask, passThroughVec);
          // Masked store the result to output.
          builder.create<MaskedStoreOp>(loc, alloc,
                                        ValueRange{ivs[0], ivs[1], ivs[2], idx},
                                        tailMask, maskedOutputVec);
        });
    Value output = rewriter.create<bufferization::ToTensorOp>(
        loc, input.getType().cast<TensorType>().cloneWith(outputNum, elementTy),
        alloc, /*restrict=*/true);

    // Remove the origin convolution operation.
    rewriter.replaceOp(op, output);
    return success();
  }

private:
  int64_t strip;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// TransposeVectorizationPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering tosa transpose operations to mixture of
/// Arith + Vector operations.
namespace {
class TransposeVectorizationPass
    : public PassWrapper<TransposeVectorizationPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransposeVectorizationPass)
  StringRef getArgument() const final { return "transpose-vectorization"; }
  StringRef getDescription() const final { return "Transpose Vectorization."; }
  TransposeVectorizationPass() = default;
  TransposeVectorizationPass(const TransposeVectorizationPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tosa::TosaDialect, linalg::LinalgDialect, scf::SCFDialect,
                    bufferization::BufferizationDialect, affine::AffineDialect,
                    VectorDialect, func::FuncDialect, tensor::TensorDialect>();
  }
  Option<int64_t> strip{*this, "vector-size",
                        llvm::cl::desc("Specify vector type size."),
                        llvm::cl::init(64)};
};
} // end anonymous namespace.

void TransposeVectorizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                         scf::SCFDialect, func::FuncDialect,
                         memref::MemRefDialect, VectorDialect,
                         bufferization::BufferizationDialect, math::MathDialect,
                         tensor::TensorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();
  target.addLegalOp<tensor::CastOp>();

  RewritePatternSet patterns(context);
  patterns.add<TransposeVectorizationPattern>(context, strip);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerTransposeVectorizationPass() {
  PassRegistration<TransposeVectorizationPass>();
}
} // namespace buddy
} // namespace mlir
