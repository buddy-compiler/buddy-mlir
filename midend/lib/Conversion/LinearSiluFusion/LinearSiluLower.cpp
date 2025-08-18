//===- LinearSiluLower.cpp - Linear+SiLU Lower Pass ----------------------===//
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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tensor;
using namespace mlir::arith;
using namespace mlir::linalg;
using namespace mlir::bufferization;
using namespace mlir::affine;
using namespace mlir::vector;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Check if a linalg.generic operation implements matrix multiplication
bool isMatMulOp(linalg::GenericOp op) {
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1) {
    return false;
  }

  // Check iterator types: should have 3 parallel + 1 reduction
  auto iteratorTypes = op.getIteratorTypesArray();
  if (iteratorTypes.size() != 4) {
    return false;
  }

  int parallelCount = 0;
  int reductionCount = 0;

  for (auto type : iteratorTypes) {
    if (type == utils::IteratorType::parallel) {
      parallelCount++;
    } else if (type == utils::IteratorType::reduction) {
      reductionCount++;
    }
  }

  if (parallelCount != 3 || reductionCount != 1) {
    return false;
  }

  // Check the body: should have mul and add operations
  Block &block = op.getRegion().front();
  bool hasMul = false;
  bool hasAdd = false;

  for (auto &nestedOp : block) {
    if (isa<arith::MulFOp>(nestedOp)) {
      hasMul = true;
    } else if (isa<arith::AddFOp>(nestedOp)) {
      hasAdd = true;
    }
  }

  return hasMul && hasAdd;
}

/// Check if a linalg.generic operation implements bias addition + SiLU
bool isBiasSiluOp(linalg::GenericOp op) {
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1) {
    return false;
  }

  // Check iterator types: should have 3 parallel
  auto iteratorTypes = op.getIteratorTypesArray();
  if (iteratorTypes.size() != 3) {
    return false;
  }

  for (auto type : iteratorTypes) {
    if (type != utils::IteratorType::parallel) {
      return false;
    }
  }

  // Check the body: should have add, neg, exp, add, div, mul operations
  Block &block = op.getRegion().front();
  auto it = block.getOperations().begin();

  // Check: addf, negf, exp, addf, divf, mulf, yield
  if (!isa<arith::AddFOp>(*it++)) return false;
  if (!isa<arith::NegFOp>(*it++)) return false;
  if (!isa<math::ExpOp>(*it++)) return false;
  if (!isa<arith::AddFOp>(*it++)) return false;
  if (!isa<arith::DivFOp>(*it++)) return false;
  if (!isa<arith::MulFOp>(*it++)) return false;
  if (!isa<linalg::YieldOp>(*it)) return false;

  return true;
}

/// Find the next linalg.generic operation that uses the given value
linalg::GenericOp findNextGenericOp(Value value) {
  for (auto &use : value.getUses()) {
    Operation *user = use.getOwner();
    auto linalgOp = dyn_cast<linalg::GenericOp>(user);
    if (linalgOp) {
      return linalgOp;
    }
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

class LinearSiluLowerPattern : public OpRewritePattern<linalg::GenericOp> {
public:
  explicit LinearSiluLowerPattern(MLIRContext *context)
      : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(linalg::GenericOp matmulOp,
                  PatternRewriter &rewriter) const override {
    // Step 1: Check if this is a matmul operation
    if (!isMatMulOp(matmulOp)) {
      return failure();
    }

    // Step 2: Find the next generic operation that uses matmul result
    Value matmulResult = matmulOp.getResult(0);
    linalg::GenericOp biasSiluOp = findNextGenericOp(matmulResult);
    
    if (!biasSiluOp) {
      return failure();
    }

    // Step 3: Check if the next operation is bias + SiLU
    if (!isBiasSiluOp(biasSiluOp)) {
      return failure();
    }

    // Step 4: Verify that matmul result is used as first input in bias+SiLU
    Value biasSiluFirstInput = biasSiluOp.getDpsInputOperand(0)->get();
    if (biasSiluFirstInput != matmulResult) {
      return failure();
    }

    // Step 5: Check tensor types compatibility
    auto matmulOutputType = matmulResult.getType().cast<RankedTensorType>();
    auto biasSiluOutputType = biasSiluOp.getResult(0).getType().cast<RankedTensorType>();
    if (matmulOutputType != biasSiluOutputType) {
      return failure();
    }

    // Step 6: Verify the operations are in the same function/block
    if (matmulOp->getParentOp() != biasSiluOp->getParentOp()) {
      return failure();
    }
    
    // Step 7: Find intermediate operations between matmul and bias+silu
    SmallVector<Operation*> intermediateOps;
    Operation* expandShapeOp = nullptr;
    Operation* emptyOp = nullptr;
    
    for (auto &op : matmulOp->getParentOp()->getRegion(0).front()) {
      if (&op != matmulOp && op.getNumResults() > 0 && 
          op.getResult(0).getType().isa<RankedTensorType>() &&
          op.getResult(0).hasOneUse() &&
          op.getResult(0).use_begin()->getOwner() == biasSiluOp) {
        intermediateOps.push_back(&op);
        
        // Identify specific operations
        if (isa<tensor::ExpandShapeOp>(op)) {
          expandShapeOp = &op;
        } else if (isa<tensor::EmptyOp>(op)) {
          emptyOp = &op;
        }
      }
    }
    
    // =================================================================
    // Pattern matched successfully - start lowering to vectorized affine
    // =================================================================
    
    rewriter.setInsertionPoint(biasSiluOp);
    OpBuilder::InsertionGuard guard(rewriter);
    Location loc = biasSiluOp.getLoc();
    
    // Get input tensors
    Value inputA = matmulOp.getDpsInputOperand(0)->get();
    Value inputB = matmulOp.getDpsInputOperand(1)->get();
    Value biasInput = biasSiluOp.getDpsInputOperand(1)->get();
    
    // Get tensor types
    auto inputAType = inputA.getType().cast<RankedTensorType>();
    auto inputBType = inputB.getType().cast<RankedTensorType>();
    auto biasType = biasInput.getType().cast<RankedTensorType>();
    auto outputType = biasSiluOp.getResult(0).getType().cast<RankedTensorType>();
    
    // Create constants
    Value cst_0 = rewriter.create<arith::ConstantFloatOp>(loc, APFloat(0.0f), rewriter.getF32Type());
    Value cst_1 = rewriter.create<arith::ConstantFloatOp>(loc, APFloat(1.0f), rewriter.getF32Type());
    
    // Create output tensor
    Value outputTensor = rewriter.create<tensor::EmptyOp>(loc, outputType.getShape(), outputType.getElementType());

    // Convert tensors to memrefs for fused affine operation
    Value inputAMemref = rewriter.create<bufferization::ToMemrefOp>(
        loc, MemRefType::get(inputAType.getShape(), inputAType.getElementType()), inputA);
    Value inputBMemref = rewriter.create<bufferization::ToMemrefOp>(
        loc, MemRefType::get(inputBType.getShape(), inputBType.getElementType()), inputB);
    Value biasMemref = rewriter.create<bufferization::ToMemrefOp>(
        loc, MemRefType::get(biasType.getShape(), biasType.getElementType()), biasInput);
    Value outputMemref = rewriter.create<bufferization::ToMemrefOp>(
        loc, MemRefType::get(outputType.getShape(), outputType.getElementType()), outputTensor);
    
    // Allocate memory for vector accumulator
    auto vecType = VectorType::get({8}, rewriter.getF32Type());
    auto vecMemrefType = MemRefType::get({}, vecType);
    Value vecAccMemref = rewriter.create<memref::AllocOp>(loc, vecMemrefType);
    
    // Vectorized operation: MatMul + Bias + SiLU with vectorization
    Value iLower = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value iUpper = rewriter.create<arith::ConstantIndexOp>(loc, inputAType.getShape()[0]);
    
    buildAffineLoopNest(
        rewriter, loc, {iLower}, {iUpper}, {1},
        [&](OpBuilder &iBuilder, Location iLoc, ValueRange iIvs) {
          Value i = iIvs[0];
          
          Value jLower = iBuilder.create<arith::ConstantIndexOp>(iLoc, 0);
          Value jUpper = iBuilder.create<arith::ConstantIndexOp>(iLoc, inputAType.getShape()[1]);
          
          buildAffineLoopNest(
              iBuilder, iLoc, {jLower}, {jUpper}, {1},
              [&](OpBuilder &jBuilder, Location jLoc, ValueRange jIvs) {
                Value j = jIvs[0];
                
                // Vectorize the k loop by processing 8 elements at a time
                Value kLower = jBuilder.create<arith::ConstantIndexOp>(jLoc, 0);
                Value kUpper = jBuilder.create<arith::ConstantIndexOp>(jLoc, outputType.getShape()[2]);
                
                buildAffineLoopNest(
                    jBuilder, jLoc, {kLower}, {kUpper}, {8},
                    [&](OpBuilder &kBuilder, Location kLoc, ValueRange kIvs) {
                      Value k = kIvs[0];
                      
                      // Initialize vector accumulator for matmul
                      Value vecAccInit = kBuilder.create<vector::SplatOp>(kLoc, cst_0, vecType);
                      kBuilder.create<memref::StoreOp>(kLoc, vecAccInit, vecAccMemref);
                      
                      // MatMul reduction loop with vectorization
                      Value lLower = kBuilder.create<arith::ConstantIndexOp>(kLoc, 0);
                      Value lUpper = kBuilder.create<arith::ConstantIndexOp>(kLoc, inputAType.getShape()[2]);
                      
                      buildAffineLoopNest(
                          kBuilder, kLoc, {lLower}, {lUpper}, {1},
                          [&](OpBuilder &lBuilder, Location lLoc, ValueRange lIvs) {
                            Value l = lIvs[0];
                            
                            // Load A[i, j, l]
                            Value aVal = lBuilder.create<AffineLoadOp>(lLoc, inputAMemref, ValueRange{i, j, l});
                            Value aVec = lBuilder.create<vector::SplatOp>(lLoc, aVal, vecType);
                            
                            // Load 8 consecutive elements from weight matrix
                            Value bVec = lBuilder.create<vector::LoadOp>(lLoc, vecType, inputBMemref, ValueRange{i, l, k});
                            
                            // Vector multiply and accumulate
                            Value mulVec = lBuilder.create<arith::MulFOp>(lLoc, aVec, bVec);
                            Value vecAccOld = lBuilder.create<memref::LoadOp>(lLoc, vecAccMemref);
                            Value vecAccNew = lBuilder.create<arith::AddFOp>(lLoc, vecAccOld, mulVec);
                            lBuilder.create<memref::StoreOp>(lLoc, vecAccNew, vecAccMemref);
                          });
                      
                      // Load bias vector and add
                      Value vecAccFinal = kBuilder.create<memref::LoadOp>(kLoc, vecAccMemref);
                      Value biasVec = kBuilder.create<vector::LoadOp>(kLoc, vecType, biasMemref, ValueRange{i, j, k});
                      Value withBiasVec = kBuilder.create<arith::AddFOp>(kLoc, vecAccFinal, biasVec);
                      
                      // Vectorized SiLU computation: x * sigmoid(x)
                      Value negVec = kBuilder.create<arith::NegFOp>(kLoc, withBiasVec);
                      Value expVec = kBuilder.create<math::ExpOp>(kLoc, negVec);
                      Value cstVec = kBuilder.create<vector::SplatOp>(kLoc, cst_1, vecType);
                      Value addOneVec = kBuilder.create<arith::AddFOp>(kLoc, expVec, cstVec);
                      Value sigVec = kBuilder.create<arith::DivFOp>(kLoc, cstVec, addOneVec);
                      Value siluResultVec = kBuilder.create<arith::MulFOp>(kLoc, withBiasVec, sigVec);
                      
                      // Store final result vector
                      kBuilder.create<vector::StoreOp>(kLoc, siluResultVec, outputMemref, ValueRange{i, j, k});
                    });
              });
        });
    
    // Convert back to tensor
    Value resultTensor = rewriter.create<bufferization::ToTensorOp>(loc, outputMemref, /*restrict=*/true);
    
    // Replace the bias+silu operation result with our new result
    rewriter.replaceAllUsesWith(biasSiluOp.getResult(0), resultTensor);
    
    // Replace uses of intermediate operations with their inputs
    if (expandShapeOp) {
      // Don't replace expand_shape op usage since it changes shape
      // We'll handle this differently
    }
    if (emptyOp) {
      // For empty op, we need to replace with our output tensor
      rewriter.replaceAllUsesWith(emptyOp->getResult(0), resultTensor);
    }
    
    // Clean up the original operations and intermediate operations
    rewriter.eraseOp(biasSiluOp);
    rewriter.eraseOp(matmulOp);
    
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class LinearSiluLowerPass
    : public PassWrapper<LinearSiluLowerPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinearSiluLowerPass)
  StringRef getArgument() const final { return "linear-silu-lower"; }
  StringRef getDescription() const final {
    return "Lower Linear+SiLU operations to optimized implementation.";
  }
  LinearSiluLowerPass() = default;
  LinearSiluLowerPass(const LinearSiluLowerPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, affine::AffineDialect,
                    math::MathDialect, arith::ArithDialect,
                    memref::MemRefDialect, bufferization::BufferizationDialect,
                    vector::VectorDialect>();
  }
};

void LinearSiluLowerPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  RewritePatternSet patterns(context);
  patterns.add<LinearSiluLowerPattern>(context);
  GreedyRewriteConfig config;
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns), config))) {
    signalPassFailure();
  }
}

} // end anonymous namespace

namespace mlir {
namespace buddy {
void registerLinearSiluLowerPass() {
  PassRegistration<LinearSiluLowerPass>();
}
} // namespace buddy
} // namespace mlir