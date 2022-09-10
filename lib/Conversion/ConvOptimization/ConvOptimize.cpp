//====- ConvOptimize.cpp ----------------------------------------===//
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
// This file implements the conv optimize.
//
//===----------------------------------------------------------------------===//
#include <iostream>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
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

// PoolingNhwcSum vectorization pattern
class ConvOptimizePattern : public ConversionPattern {
public:
  explicit ConvOptimizePattern(MLIRContext *context,
                                                int64_t stripParam, int64_t kernelMParam, int64_t kernelNParam)
      : ConversionPattern(linalg::Conv2DNchwFchwOp::getOperationName(), 1,
                          context) {
    strip = stripParam;
    kernelM = kernelMParam;
    kernelN = kernelNParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // Some constant we need.
    const Value c0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value cf0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.));
    const AffineMap mapBroadcast = AffineMap::get(4, 0, rewriter.getAffineConstantExpr(0));
    const AffineExpr d0 = rewriter.getAffineDimExpr(0);
    const AffineExpr d1 = rewriter.getAffineDimExpr(1);

    Value input = op->getOperand(0);
    Value filter = op->getOperand(1);
    Value output = op->getOperand(2);

    ShapedType inputTy = input.getType().cast<ShapedType>();
    
    Type elemTy = inputTy.getElementType();
    VectorType vecTy = VectorType::get(strip, elemTy);

    // Dims
    Value a = rewriter.create<memref::DimOp>(loc, output, 0);
    Value b = rewriter.create<memref::DimOp>(loc, output, 1);
    Value d = rewriter.create<memref::DimOp>(loc, output, 3);
    Value c = rewriter.create<memref::DimOp>(loc, output, 2);
    Value e = rewriter.create<memref::DimOp>(loc, input, 1);
    Value f = rewriter.create<memref::DimOp>(loc, filter, 2);
    Value g = rewriter.create<memref::DimOp>(loc, filter, 3);

    // memref<1xvector<stripxf32>>
    MemRefType bufferTy = MemRefType::get(1, vecTy);
    Value buffer = rewriter.create<memref::AllocOp>(loc, bufferTy);

    buildAffineLoopNest(rewriter, loc, c0, a, 1, [&](OpBuilder& builder, Location loc, ValueRange ivRange){
      Value ivA = ivRange.front();
      buildAffineLoopNest(rewriter, loc, c0, b, 1, [&](OpBuilder& builder, Location loc, ValueRange ivRange){
        Value ivB = ivRange.front();
        buildAffineLoopNest(rewriter, loc, c0, d, 1, [&](OpBuilder& builder, Location loc, ValueRange ivRange){
          Value ivD = ivRange.front();
          buildAffineLoopNest(rewriter, loc, c0, c, 1, [&](OpBuilder& builder, Location loc, ValueRange ivRange){

            Value ivC = ivRange.front();
	    Value t = builder.create<SplatOp>(loc, vecTy, cf0);
	    builder.create<memref::StoreOp>(loc, t, buffer, c0);
            buildAffineLoopNest(rewriter, loc, c0, e, 1, [&](OpBuilder& builder, Location loc, ValueRange ivRange){
              Value ivE = ivRange.front();
              buildAffineLoopNest(rewriter, loc, c0, f, kernelM, [&](OpBuilder& builder, Location loc, ValueRange ivRange){
	        Value ivF = ivRange.front();
                buildAffineLoopNest(rewriter, loc, c0, g, kernelN * strip, [&](OpBuilder& builder, Location loc, ValueRange ivRange){
	          Value ivG = ivRange.front();

		  SmallVector<Value> iList;
		  SmallVector<Value> fList;
		  for(int i = 0; i < kernelM; ++ i){
		    Value rowInput = builder.create<AffineApplyOp>(loc, AffineMap::get(2, 0, d0 + i + d1), ValueRange{ivC, ivF});
		    Value rowFilter = builder.create<AffineApplyOp>(loc, AffineMap::get(1, 0, d0 + i), ivF);
		    for(int j = 0; j < kernelN; ++ j){
		      Value columnInput = builder.create<AffineApplyOp>(loc, AffineMap::get(2, 0, d0 + d1 + j * strip), ValueRange{ivD, ivG});
		      Value columnFilter = builder.create<AffineApplyOp>(loc, AffineMap::get(1, 0, d0 + j * strip), ivG);

			Value i = builder.create<TransferReadOp>(loc, vecTy, input, ValueRange{ivA, ivE, rowInput, columnInput});
			Value f = builder.create<TransferReadOp>(loc, vecTy, filter, ValueRange{ivB, ivE, rowFilter, columnFilter});

			iList.push_back(i);
			fList.push_back(f);
		    }
		  }
		  Value lastResult = builder.create<memref::LoadOp>(loc, buffer, c0);
		  for(int i = 0; i < kernelM; ++ i){
		    for(int j = 0; j < kernelN; ++ j){
                      lastResult = builder.create<vector::FMAOp>(loc, vecTy, iList[i * kernelN + j], fList[i * kernelN + j], lastResult);
		    }
		  }

		  builder.create<memref::StoreOp>(loc, lastResult, buffer, c0);

		});
	      });
	    });

	    Value reduceVec = builder.create<memref::LoadOp>(loc, buffer, c0);
	    Value reducedRes = builder.create<vector::ReductionOp>(loc, vector::CombiningKind::ADD, reduceVec);
	    Value bias = builder.create<memref::LoadOp>(loc, output, ValueRange{ivA, ivB, ivC, ivD});
	    Value addRes = builder.create<arith::AddFOp>(loc, bias, reducedRes);
	    builder.create<memref::StoreOp>(loc, addRes, output, ValueRange{ivA, ivB, ivC, ivD});

	    // Now handle tail issue.
            buildAffineLoopNest(rewriter, loc, c0, e, 1, [&](OpBuilder& builder, Location loc, ValueRange ivRange){
              Value ivE = ivRange.front();
	      Value remainStart = rewriter.create<AffineApplyOp>(loc, AffineMap::get(1, 0, d0 - d0 % kernelM), f);
              buildAffineLoopNest(rewriter, loc, remainStart, f, 1, [&](OpBuilder& builder, Location loc, ValueRange ivRange){
	        Value ivF = ivRange.front();
	        Value remainStart = rewriter.create<AffineApplyOp>(loc, AffineMap::get(1, 0, d0 - d0 % (kernelN * strip)), g);
                buildAffineLoopNest(rewriter, loc, remainStart, g, 1, [&](OpBuilder& builder, Location loc, ValueRange ivRange){
	          Value ivG = ivRange.front();
		  Value fixedRow = builder.create<AffineApplyOp>(loc, AffineMap::get(2, 0, d0 + d1), ValueRange{ivC, ivF});
		  Value fixedColumn = builder.create<AffineApplyOp>(loc, AffineMap::get(2, 0, d0 + d1), ValueRange{ivD, ivG});
		  Value i = rewriter.create<memref::LoadOp>(loc, input, ValueRange{ivA, ivE, fixedRow, fixedColumn});
		  Value f = rewriter.create<memref::LoadOp>(loc, filter, ValueRange{ivB, ivE, ivF, ivG});
		  Value o = rewriter.create<memref::LoadOp>(loc, output, ValueRange{ivA, ivB, ivC, ivD});
		  Value ITimesF = rewriter.create<arith::MulFOp>(loc, i, f);
		  Value res = rewriter.create<arith::AddFOp>(loc, ITimesF, o);
		  rewriter.create<memref::StoreOp>(loc, res, output, ValueRange{ivA, ivB, ivC, ivD});
		});
	      });
	    });
          });
        });
      });
    });

    rewriter.create<memref::DeallocOp>(loc, buffer);

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t strip;
  int64_t kernelM;
  int64_t kernelN;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ConvOptimizePass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg pooling operations to mixture of
/// Affine + Vector operations.
namespace {
class ConvOptimizePass
    : public PassWrapper<ConvOptimizePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvOptimizePass)
  StringRef getArgument() const final { return "conv-optimize"; }
  StringRef getDescription() const final { return "Conv vectorization."; }
  ConvOptimizePass() = default;
  ConvOptimizePass(const ConvOptimizePass &) {}
  explicit ConvOptimizePass(int64_t stripParam, int64_t kernelMParam, int64_t kernelNParam) { strip = stripParam; kernelM = kernelMParam; kernelN = kernelNParam; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect, AffineDialect,
                    VectorDialect>();
  }

  Option<int64_t> strip{*this, "vec-size",
                        llvm::cl::desc("Vector size using in kernel."),
                        llvm::cl::init(16)};
  
  Option<int64_t> kernelM{*this, "kernel-m",
                        llvm::cl::desc("Specify how many rows kernel will contain."),
                        llvm::cl::init(4)};

  Option<int64_t> kernelN{*this, "kernel-n",
                        llvm::cl::desc("Specify how many columns kernel will cantain."),
                        llvm::cl::init(2)};
};
} // end anonymous namespace.

void ConvOptimizePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithmeticDialect, AffineDialect, scf::SCFDialect,
                       memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<ConvOptimizePattern>(context, strip, kernelM, kernelN);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerConvOptimizePass() {
  PassRegistration<ConvOptimizePass>();
}
} // namespace buddy
} // namespace mlir
