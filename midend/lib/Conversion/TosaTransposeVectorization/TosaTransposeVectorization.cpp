//===- TosaTransposeVectorization.cpp ----------------------------------===//
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
// This file implements the Tosa transpose optimization via Linalg GenericOp.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>

using namespace mlir;
using namespace vector;
using namespace affine;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
class TosaTransposeVectorization
    : public OpConversionPattern<linalg::GenericOp> {
public:
  explicit TosaTransposeVectorization(MLIRContext *context,
                                      int64_t affineVectorSizeParam)
      : OpConversionPattern(context) {
    affineVectorSize = affineVectorSizeParam;
  }

  LogicalResult
  matchAndRewrite(linalg::GenericOp genericOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check if the operation is a transpose.
    if (!isTransposeOp(genericOp, rewriter)) {
      return failure();
    }

    // Get input and output tensors.
    Value input = genericOp.getInputs()[0];
    Value output = genericOp.getOutputs()[0];

    // Convert TensorType to MemRefType if necessary.
    if (auto tensorType = input.getType().dyn_cast<TensorType>()) {
      input = rewriter.create<bufferization::ToMemrefOp>(
          genericOp.getLoc(),
          MemRefType::get(tensorType.getShape(), tensorType.getElementType()),
          input);
    }
    if (auto tensorType = output.getType().dyn_cast<TensorType>()) {
      output = rewriter.create<bufferization::ToMemrefOp>(
          genericOp.getLoc(),
          MemRefType::get(tensorType.getShape(), tensorType.getElementType()),
          output);
    }

    // Get the element type and rank of the input tensor.
    Type elementType = input.getType().cast<MemRefType>().getElementType();
    int64_t rank = input.getType().cast<MemRefType>().getRank();

    // Get dimensions of the input tensor.
    SmallVector<Value, 4> dims;
    for (int64_t i = 0; i < rank; ++i) {
      dims.push_back(
          rewriter.create<memref::DimOp>(genericOp.getLoc(), input, i));
    }

    // Initialize input and output indices.
    SmallVector<Value, 4> inputIndices(rank, nullptr);
    SmallVector<Value, 4> outputIndices(rank, nullptr);

    // Create affine.for loops for vectorization.
    Value transposedOutput =
        createNestedLoops(rewriter, genericOp.getLoc(), input, output, dims,
                          elementType, 0, inputIndices, outputIndices);

    // Convert the result back to TensorType if necessary.
    if (auto memrefType = transposedOutput.getType().dyn_cast<MemRefType>()) {
      transposedOutput = rewriter.create<bufferization::ToTensorOp>(
          genericOp.getLoc(),
          RankedTensorType::get(memrefType.getShape(),
                                memrefType.getElementType()),
          transposedOutput);
    }

    // Replace the result of the original GenericOp with the transposed output.
    rewriter.replaceOp(genericOp, transposedOutput);

    return success();
  }

private:
  // Check if a GenericOp is a transpose operation.
  bool isTransposeOp(linalg::GenericOp genericOp,
                     ConversionPatternRewriter &rewriter) const {
    // Check if the input tensor rank is same as the output tensor rank.
    if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
      return false;
    }

    // Check indexing_maps.
    SmallVector<AffineMap> indexingMaps = genericOp.getIndexingMapsArray();
    if (indexingMaps.size() != 2) {
      return false;
    }

    AffineMap inputMap = indexingMaps[0];
    AffineMap outputMap = indexingMaps[1];

    // Check that input and output have the same rank.
    if (inputMap.getNumDims() != outputMap.getNumDims()) {
      return false;
    }

    // Check that the output map is a permutation of the input map.
    for (unsigned i = 0; i < inputMap.getNumDims(); ++i) {
      AffineExpr inputExpr = inputMap.getResult(i);
      AffineExpr outputExpr = outputMap.getResult(i);

      // If the input and output expressions are not the same, check if they are
      // a permutation.
      if (inputExpr != outputExpr) {
        bool isPermutation = false;
        for (unsigned j = 0; j < inputMap.getNumDims(); ++j) {
          if (inputMap.getResult(j) == outputExpr &&
              outputMap.getResult(j) == inputExpr) {
            isPermutation = true;
            break;
          }
        }
        if (!isPermutation) {
          return false;
        }
      }
    }

    // Check iterator_types.
    SmallVector<utils::IteratorType> iteratorTypes =
        genericOp.getIteratorTypesArray();
    for (utils::IteratorType type : iteratorTypes) {
      if (type != utils::IteratorType::parallel) {
        return false;
      }
    }

    // Check basic block.
    Region &body = genericOp.getRegion();
    if (!body.hasOneBlock()) {
      return false;
    }

    Block &block = body.front();
    if (!llvm::hasSingleElement(block.getOperations())) {
      return false;
    }

    Operation &onlyOp = block.front();
    if (!isa<linalg::YieldOp>(onlyOp)) {
      return false;
    }

    return true;
  }

  // Create nested affine.for loops for vectorization.
  Value createNestedLoops(ConversionPatternRewriter &rewriter, Location loc,
    Value input, Value output, SmallVector<Value, 4> &dims,
    Type elementType, size_t currentDim,
    SmallVector<Value, 4> &inputIndices,
    SmallVector<Value, 4> &outputIndices) const {
    AffineMap lbMap = AffineMap::get(0, 0, {rewriter.getAffineConstantExpr(0)},
                                     rewriter.getContext());
    AffineMap ubMap = AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0)},
                                     rewriter.getContext());

    if (currentDim == dims.size() - 1) {
      // Innermost loop: perform vectorized load and store.
      rewriter.create<affine::AffineForOp>(
          loc, ValueRange{},            // Lower bound operands.
          lbMap,                        // Lower bound map.
          ValueRange{dims[currentDim]}, // Upper bound operands.
          ubMap,                        // Upper bound map.
          affineVectorSize,             // Step.
          ValueRange{},                 // Iteration arguments.
          [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
              ValueRange iterArgs) {
            // Update input and output indices for the current dimension.
            inputIndices[currentDim] = iv;
            outputIndices[currentDim] = iv;

            // Adjust output indices for transpose.
            outputIndices[1] = inputIndices[2]; // Swap d1 and d2.
            outputIndices[2] = inputIndices[1];

            // Read a vector tile from the input.
            Value vectorTile = nestedBuilder.create<vector::LoadOp>(
                nestedLoc, VectorType::get({affineVectorSize}, elementType),
                input, inputIndices);

            // Write the transposed vector tile to the output.
            nestedBuilder.create<vector::StoreOp>(nestedLoc, vectorTile, output,
                                                  outputIndices);

            // Add a terminator to the loop body.
            nestedBuilder.create<affine::AffineYieldOp>(nestedLoc);
          });
    } else {
      // Outer loops: recursively create nested loops.
      rewriter.create<affine::AffineForOp>(
          loc, ValueRange{},            // Lower bound operands.
          lbMap,                        // Lower bound map.
          ValueRange{dims[currentDim]}, // Upper bound operands.
          ubMap,                        // Upper bound map.
          1,                            // Step.
          ValueRange{},                 // Iteration arguments.
          [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
              ValueRange iterArgs) {
            // Update input and output indices for the current dimension.
            inputIndices[currentDim] = iv;
            outputIndices[currentDim] = iv;

            createNestedLoops(rewriter, nestedLoc, input, output, dims,
                              elementType, currentDim + 1, inputIndices,
                              outputIndices);

            // Add a terminator to the loop body.
            nestedBuilder.create<affine::AffineYieldOp>(nestedLoc);
          });
    }

    // Return the transposed output.
    return output;
  }

  int64_t affineVectorSize;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// TosaTransposeVectorizationPass
//===----------------------------------------------------------------------===//

namespace {
class TosaTransposeVectorizationPass
    : public PassWrapper<TosaTransposeVectorizationPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TosaTransposeVectorizationPass)
  StringRef getArgument() const final { return "tosa-transpose-vectorization"; }
  StringRef getDescription() const final { return "Transpose Optimization ."; }
  TosaTransposeVectorizationPass() = default;
  TosaTransposeVectorizationPass(const TosaTransposeVectorizationPass &) {}
  explicit TosaTransposeVectorizationPass(int64_t affineVectorSizeParam) {
    affineVectorSize = affineVectorSizeParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, affine::AffineDialect, VectorDialect,
                memref::MemRefDialect, bufferization::BufferizationDialect>();
  }

  Option<int64_t> affineVectorSize{*this, "vector-size",
                                   llvm::cl::desc("Affine Vector size."),
                                   llvm::cl::init(64)};
};
} // end anonymous namespace

void TosaTransposeVectorizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                         memref::MemRefDialect, VectorDialect,
                         bufferization::BufferizationDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<TosaTransposeVectorization>(context, affineVectorSize);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerTosaTransposeVectorizationPass() {
  PassRegistration<TosaTransposeVectorizationPass>();
}
} // namespace buddy
} // namespace mlir