//===- SigmoidVectorization.cpp ----------------------------------===//
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
// This file implements the sigmoid vectorization optimization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"



using namespace mlir;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class SigmoidVectorizationPattern : public ConversionPattern {
public:
  explicit SigmoidVectorizationPattern(MLIRContext *context, int64_t vectorSizeParam)
      : ConversionPattern(tosa::SigmoidOp::getOperationName(), 1, context) {
    vectorSize = vectorSizeParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto sigmoidOp = cast<tosa::SigmoidOp>(op);
    
    // Get input and output tensors
    Value input = sigmoidOp.getInput();
    Value output = sigmoidOp.getOutput();
    
    // Convert tensors to memrefs
    auto inputType = input.getType().dyn_cast<TensorType>();
    if (!inputType)
      return failure();
    
    // Only handle 3D tensors for now (batch x rows x cols)
    if (inputType.getRank() != 3)
      return failure();
    
    // Create constants
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value oneF = rewriter.create<arith::ConstantOp>(loc, 
                  rewriter.getFloatAttr(inputType.getElementType(), 1.0));
    Value vectorSizeVal = rewriter.create<arith::ConstantIndexOp>(loc, vectorSize);
    
    // Get dimensions
    Value batch = rewriter.create<tensor::DimOp>(loc, input, 0);
    Value rows = rewriter.create<tensor::DimOp>(loc, input, 1);
    Value cols = rewriter.create<tensor::DimOp>(loc, input, 2);
    
    // Allocate output memref
    auto memrefType = MemRefType::get(inputType.getShape(), inputType.getElementType());
    Value outputBuffer = rewriter.create<memref::AllocOp>(loc, memrefType);
    
    // Create parallel loops for batch and rows
    rewriter.create<scf::ParallelOp>(
        loc, ValueRange{zero, zero}, ValueRange{batch, rows}, ValueRange{one, one},
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange ivs) {
          // Extract loop indices
          Value batchIdx = ivs[0];
          Value rowIdx = ivs[1];
          Value initVal = zero;

          MLIRContext *ctx = nestedBuilder.getContext();
          AffineMap perMap = AffineMap::get(3,0,{getAffineDimExpr(2,ctx)},ctx);
          AffineMapAttr perMapAttr = AffineMapAttr::get(perMap);
          ArrayAttr inBounds = nestedBuilder.getArrayAttr({nestedBuilder.getBoolAttr(true)});     

          // Prefetch next block of data
          //Value nextIdx = nestedBuilder.create<arith::AddIOp>(nestedLoc,initVal,vectorSizeVal);
          // Load vector from input tensor
          Value inputVec = nestedBuilder.create<vector::TransferReadOp>(
            nestedLoc ,VectorType::get({vectorSize},inputType.getElementType()),
            input ,ValueRange{batchIdx ,rowIdx ,initVal}
          );
          // Compute sigmoid: 1 / (1 + exp(-x))
          // Step 1: Negate the input
          Value negInputVec = nestedBuilder.create<arith::NegFOp>(nestedLoc, inputVec);

          // Step 2: Compute exp(-x)
          Value expVec = nestedBuilder.create<math::ExpOp>(nestedLoc, negInputVec);

          // Step 3: Add 1 to exp(-x)
          Value oneVec = nestedBuilder.create<vector::BroadcastOp>(
              nestedLoc, VectorType::get({vectorSize}, inputType.getElementType()), oneF);
          Value denomVec = nestedBuilder.create<arith::AddFOp>(nestedLoc, expVec, oneVec);

          // Step 4: Compute 1 / (1 + exp(-x))
          Value resultVec = nestedBuilder.create<arith::DivFOp>(nestedLoc, oneVec, denomVec);

          // Store result to output buffer
          nestedBuilder.create<vector::TransferWriteOp>(
              nestedLoc, resultVec, outputBuffer, ValueRange{batchIdx, rowIdx, initVal},perMapAttr,nullptr,inBounds);
          
    });
  
    rewriter.replaceOp(op, outputBuffer);
    
    return success();
  }

private:
  int64_t vectorSize;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// SigmoidVectorizationPass
//===----------------------------------------------------------------------===//

namespace {
class SigmoidVectorizationPass
    : public PassWrapper<SigmoidVectorizationPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SigmoidVectorizationPass)
  StringRef getArgument() const final { return "sigmoid-vectorize-manual"; }
  StringRef getDescription() const final {
    return "Vectorize sigmoid operations for better performance.";
  }
  SigmoidVectorizationPass() = default;
  SigmoidVectorizationPass(const SigmoidVectorizationPass &) {}
  explicit SigmoidVectorizationPass(int64_t vectorSizeParam) {
    vectorSize = vectorSizeParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tosa::TosaDialect, scf::SCFDialect, vector::VectorDialect,
                    math::MathDialect, memref::MemRefDialect, tensor::TensorDialect,
                    bufferization::BufferizationDialect, func::FuncDialect>();
  }

  Option<int64_t> vectorSize{*this, "vector-size",
                             llvm::cl::desc("Vector size for sigmoid vectorization."),
                             llvm::cl::init(8)};
};
} // end anonymous namespace.


void SigmoidVectorizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  if (!context->getLoadedDialect<func::FuncDialect>()) {
    llvm::errs() << "ERROR: Func dialect not registered!\n";
    signalPassFailure();
    return;
  }
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithDialect, scf::SCFDialect,
                         memref::MemRefDialect, vector::VectorDialect,
                         math::MathDialect, tensor::TensorDialect,
                         bufferization::BufferizationDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();

  RewritePatternSet patterns(context);
  patterns.add<SigmoidVectorizationPattern>(context, vectorSize);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerSigmoidVectorizationPass() {
  PassRegistration<SigmoidVectorizationPass>();
}
} // namespace buddy
} // namespace mlir

