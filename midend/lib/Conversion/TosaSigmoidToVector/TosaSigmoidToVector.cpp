//===- TosaSigmoidToVector.cpp ------------------------------------------------------===//
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
// This file implements the TosaSigmoidToVector pass for vectorization.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace vector;
using namespace arith;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//
namespace  {

// TOSA Sigmoid 到向量优化的转换模式
class TosaSigmoidToVectorPattern : public ConversionPattern {
public:
  explicit TosaSigmoidToVectorPattern(MLIRContext *context,int64_t vectorWidth = 8, PatternBenefit benefit = 1)
      : ConversionPattern(tosa::SigmoidOp::getOperationName(),benefit,context) {
        vectorWidth=vectorWidth;
      }

  LogicalResult 
  matchAndRewrite(Operation *op,ArrayRef<Value> /*operands*/,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    
    Value input = op->getOperand(0);
    Value output = op->getResult(0);
    

    ShapedType inputType = input.getType().cast<ShapedType>();
    ShapedType outputType = output.getType().cast<ShapedType>();


    SmallVector<Value> inputDims;
    for (unsigned i = 0; i < inputType.getRank(); ++i) {
      if (inputType.isDynamicDim(i)) {
        inputDims.push_back(rewriter.create<memref::DimOp>(loc, input, i));
      } else {
        inputDims.push_back(rewriter.create<arith::ConstantIndexOp>(
            loc, inputType.getDimSize(i)));
      }
    }


    auto memRefType = MemRefType::get(inputType.getShape(), inputType.getElementType());
    
    Value tempMemRef = rewriter.create<memref::AllocOp>(
        loc, memRefType, ValueRange{}, rewriter.getI64IntegerAttr(64));
    

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);
    
    VectorType vecType = VectorType::get({vectorWidth}, Float32Type::get(getContext()));
    

    Value constOne = rewriter.create<arith::ConstantFloatOp>(
        loc, APFloat(1.0f), Float32Type::get(getContext()));
    Value constZero = rewriter.create<arith::ConstantFloatOp>(
        loc, APFloat(0.0f), Float32Type::get(getContext()));
    
    Value vecOne = rewriter.create<vector::BroadcastOp>(loc, vecType, constOne);
    Value vecZero = rewriter.create<vector::BroadcastOp>(loc, vecType, constZero);
    
    Value effectiveUpperBound = rewriter.create<arith::SubIOp>(
      loc, inputDims[2], 
      rewriter.create<arith::RemSIOp>(loc, inputDims[2], step));
  effectiveUpperBound = rewriter.create<arith::MaxSIOp>(
      loc, effectiveUpperBound, zero);
  
    auto parallelOp = rewriter.create<scf::ParallelOp>(
        loc, 
        SmallVector<Value>{zero, zero}, 
        SmallVector<Value>{inputDims[1], inputDims[2]}, 
        SmallVector<Value>{one, step}, 
        ValueRange{});
    
    rewriter.setInsertionPointToStart(parallelOp.getBody());
    Block *body = parallelOp.getBody();
    Value j = body->getArgument(0); 
    Value k = body->getArgument(1);  
    
 
    Value nextK = rewriter.create<arith::AddIOp>(loc, k, step);
    Value inBound = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, nextK, inputDims[2]);
    
    rewriter.create<scf::IfOp>(
        loc, inBound, [&](OpBuilder &builder, Location loc) {
          builder.create<memref::PrefetchOp>(
              loc, input, ValueRange{zero, j, nextK}, "r", 1, 0);
        });
    

    Value inputVec = rewriter.create<vector::LoadOp>(
        loc, vecType, input, ValueRange{zero, j, k});
    
    Value negX = rewriter.create<arith::NegFOp>(loc, inputVec);
    Value expNegX = rewriter.create<math::ExpOp>(loc, negX);
    Value onePlusExp = rewriter.create<arith::AddFOp>(loc, vecOne, expNegX);
    Value result = rewriter.create<arith::DivFOp>(loc, vecOne, onePlusExp);
    

    Value threshold = rewriter.create<vector::BroadcastOp>(
        loc, vecType, rewriter.create<arith::ConstantFloatOp>(
            loc, APFloat(16.0f), Float32Type::get(getContext())));
    
    Value xGT16 = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OGT, inputVec, threshold);
    Value xLT16 = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OLT, inputVec, 
        rewriter.create<arith::NegFOp>(loc, threshold));
    
    result = rewriter.create<arith::SelectOp>(loc, xGT16, vecOne, result);
    result = rewriter.create<arith::SelectOp>(loc, xLT16, vecZero, result);
    

    rewriter.create<vector::StoreOp>(
        loc, result, tempMemRef, ValueRange{zero, j, k});
    
    rewriter.create<scf::YieldOp>(loc);
    

    rewriter.replaceOp(op, tempMemRef);
    
    return success();
  }

private:
  int64_t vectorWidth;
};

//===----------------------------------------------------------------------===//
// TosaSigmoidToVectorPass
//===----------------------------------------------------------------------===//
class TosaSigmoidToVectorPass
    : public PassWrapper<TosaSigmoidToVectorPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TosaSigmoidToVectorPass)
  StringRef getArgument() const final { return "tosa-sigmoid-to-vector"; }
  StringRef getDescription() const final {
    return "Convert TOSA Sigmoid to vectorized implementation";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();


    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, scf::SCFDialect,
                          vector::VectorDialect, math::MathDialect,
                          memref::MemRefDialect, func::FuncDialect>();
    target.addIllegalDialect<tosa::TosaDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();


    RewritePatternSet patterns(context);
    patterns.add<TosaSigmoidToVectorPattern>(context, /*vectorWidth=*/8);


    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override{
    registry.insert<scf::SCFDialect, vector::VectorDialect >();
  }
};

} // namespace mlir

namespace mlir {
namespace buddy {
void registerTosaSigmoidToVectorPass() {
  PassRegistration<TosaSigmoidToVectorPass>();
}

} // namespace buddy
} // namespace mlir
