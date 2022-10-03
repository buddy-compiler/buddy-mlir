//===- GEMMPointwiseConv.cpp - transfer Convolution to GEMM----------------===//
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
// This file implements the algorithm to transfer Pointwise Convolution to GEMM.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/Pass/Pass.h"

#include <iostream>

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
class GEMMPointwiseConvPattern : public ConversionPattern {
public:
  explicit GEMMPointwiseConvPattern(MLIRContext *context)
      : ConversionPattern(linalg::Conv2DNhwcHwcfOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // Get input, kernel and output.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    // Get shape of input and output
    ShapedType inputShapeType = input.getType().cast<ShapedType>();
    ShapedType filterShapeType = kernel.getType().cast<ShapedType>();
    ShapedType outputShapeType = output.getType().cast<ShapedType>();

    auto inputShape = inputShapeType.getShape();
    auto filterShape = filterShapeType.getShape();
    auto outputShape = outputShapeType.getShape();
    // Assertions
    if (filterShape[0] != 1 || filterShape[1] != 1)
      return failure();

    if (inputShape[0] != 1)
      return failure();

    auto convOp = dyn_cast<linalg::Conv2DNhwcHwcfOp>(op);

    if (!llvm::all_of(convOp.getStrides(), [](APInt element) {
          return element.getSExtValue() == 1;
        }))
      return failure();

    if (!llvm::all_of(convOp.getDilations(), [](APInt element) {
          return element.getSExtValue() == 1;
        }))
      return failure();

    // start arrange
    SmallVector<ReassociationIndices, 4> reassociationIndices = {{0, 1, 2},
                                                                 {3}};

    auto reshapedInputType =
        RankedTensorType::get({inputShape[1] * inputShape[2], inputShape[3]},
                              inputShapeType.getElementType());
    auto reshapedFilterType = RankedTensorType::get(
        {filterShape[2], filterShape[3]}, filterShapeType.getElementType());

    auto reshapedOutputType =
        RankedTensorType::get({outputShape[1] * outputShape[2], outputShape[3]},
                              outputShapeType.getElementType());

    Value reshapedInput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedInputType, input, reassociationIndices);
    Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedFilterType, kernel, reassociationIndices);
    Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedOutputType, output, reassociationIndices);

    // Create MutmulOp
    auto matmulResult = rewriter.create<linalg::MatmulOp>(
        loc, reshapedOutputType, ArrayRef<Value>{reshapedInput, reshapedFilter},
        ArrayRef<Value>{reshapedOutput});

    auto reshapedResult = rewriter.create<tensor::ExpandShapeOp>(
        loc, outputShapeType, matmulResult.getResults()[0],
        reassociationIndices);

    // Remove the origin convolution operation.
    rewriter.replaceOp(op, ArrayRef<Value>{reshapedResult});
    // rewriter.eraseOp(op);
    return success();
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// PointwiseConvToGemmPass
//===----------------------------------------------------------------------===//

namespace {
class PointwiseConvToGemmPass
    : public PassWrapper<PointwiseConvToGemmPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PointwiseConvToGemmPass)
  StringRef getArgument() const final { return "pointwise-conv-to-gemm"; }
  StringRef getDescription() const final {
    return "Pointwise Convolution to Gemm.";
  }
  PointwiseConvToGemmPass() = default;
  PointwiseConvToGemmPass(const PointwiseConvToGemmPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    scf::SCFDialect, func::FuncDialect>();
  }
};
} // end anonymous namespace.

void PointwiseConvToGemmPass::runOnOperation() {
  MLIRContext *context = &getContext();

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithDialect, scf::SCFDialect,
                         func::FuncDialect, memref::MemRefDialect,
                         tensor::TensorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp, tensor::CollapseShapeOp, linalg::MatmulOp,
                    tensor::ExpandShapeOp>();
}

namespace mlir {
namespace buddy {
void registerPointwiseConvToGemmPass() {
  PassRegistration<PointwiseConvToGemmPass>();
}
} // namespace buddy
} // namespace mlir
