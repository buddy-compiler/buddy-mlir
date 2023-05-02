//====- ConvOptimize.cpp --------------------------------------------------===//
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
// This file implements the Conv optimize.
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/IR/IntegerSet.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class ConvOptimizePattern : public ConversionPattern {
public:
  explicit ConvOptimizePattern(MLIRContext *context, int64_t vecSizeParam, int64_t kernelMParam, int64_t kernelNParam)
      : ConversionPattern(linalg::Conv2DNchwFchwOp::getOperationName(), 1, context) {
    vecSize = vecSizeParam;
    kernelM = kernelMParam;
    kernelN = kernelNParam;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/, ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // Some constant we need.
    const Value c0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value cf0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.));

    const AffineExpr d0 = rewriter.getAffineDimExpr(0);
    const AffineExpr d1 = rewriter.getAffineDimExpr(1);
    const AffineExpr s0 = rewriter.getAffineSymbolExpr(0);

    Value input = op->getOperand(0);
    Value filter = op->getOperand(1);
    Value output = op->getOperand(2);

    ShapedType inputTy = input.getType().cast<ShapedType>();

    Type elemTy = inputTy.getElementType();
    VectorType vecTy = VectorType::get(vecSize, elemTy);

    // Dims
    Value a = rewriter.create<memref::DimOp>(loc, output, 0);
    Value b = rewriter.create<memref::DimOp>(loc, output, 1);
    Value c = rewriter.create<memref::DimOp>(loc, output, 2);
    Value d = rewriter.create<memref::DimOp>(loc, output, 3);
    Value e = rewriter.create<memref::DimOp>(loc, input, 1);
    Value f = rewriter.create<memref::DimOp>(loc, filter, 2);
    Value g = rewriter.create<memref::DimOp>(loc, filter, 3);

    // memref<1xvector<vecsize x elemTy>>
    MemRefType bufferTy = MemRefType::get(1, vecTy);
    Value buffer = rewriter.create<memref::AllocOp>(loc, bufferTy);

    // Step 1: Create outer most loops.
    affine::buildAffineLoopNest(rewriter, loc, c0, a, 1, [&](OpBuilder &, Location loc, ValueRange ivRange) {
      Value ivA = ivRange.front();
      affine::buildAffineLoopNest(rewriter, loc, c0, b, 1, [&](OpBuilder &, Location loc, ValueRange ivRange) {
        Value ivB = ivRange.front();
        affine::buildAffineLoopNest(rewriter, loc, c0, d, 1, [&](OpBuilder &, Location loc, ValueRange ivRange) {
          Value ivD = ivRange.front();
          affine::buildAffineLoopNest(rewriter, loc, c0, c, 1, [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
            Value ivC = ivRange.front();
            Value t = builder.create<SplatOp>(loc, vecTy, cf0);
            builder.create<memref::StoreOp>(loc, t, buffer, c0);
            affine::buildAffineLoopNest(rewriter, loc, c0, e, 1, [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
              Value ivE = ivRange.front();

              Value fixed = builder.create<affine::AffineApplyOp>(loc, AffineMap::get(1, 0, d0.ceilDiv(kernelM) * kernelM), ValueRange{f});

              affine::buildAffineLoopNest(rewriter, loc, c0, fixed, kernelM, [&]([[maybe_unused]] OpBuilder &builder, Location loc, ValueRange ivRange) {
                Value ivF = ivRange.front();
                affine::buildAffineLoopNest(rewriter, loc, c0, g, kernelN * vecSize, [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
                  Value ivG = ivRange.front();

                  SmallVector<Value> iList;
                  SmallVector<Value> fList;
                  for (int i = 0; i < kernelM; ++i) {
                    Value rowInput = builder.create<affine::AffineApplyOp>(loc, AffineMap::get(2, 0, d0 + i + d1), ValueRange{ivC, ivF});
                    Value rowFilter = builder.create<affine::AffineApplyOp>(loc, AffineMap::get(1, 0, d0 + i), ivF);
                    for (int j = 0; j < kernelN; ++j) {
                      Value columnInput = builder.create<affine::AffineApplyOp>(loc, AffineMap::get(2, 0, d0 + d1 + j * vecSize), ValueRange{ivD, ivG});
                      Value columnFilter = builder.create<affine::AffineApplyOp>(loc, AffineMap::get(1, 0, d0 + j * vecSize), ivG);

                      Value i = builder.create<TransferReadOp>(loc, vecTy, input, ValueRange{ivA, ivE, rowInput, columnInput});

                      auto protectedF =
                          builder.create<affine::AffineIfOp>(loc, vecTy, IntegerSet::get(1, 1, {s0 - 1 - d0}, {false}), ValueRange{rowFilter, f}, true);

                      // if row in range, read normally.
                      auto thenBuilder = protectedF.getThenBodyBuilder();
                      Value normalReadVec = thenBuilder.create<TransferReadOp>(loc, vecTy, filter, ValueRange{ivB, ivE, rowFilter, columnFilter});
                      thenBuilder.create<affine::AffineYieldOp>(loc, normalReadVec);

                      // if row out of range, give back a empty vector.
                      auto elseBuilder = protectedF.getElseBodyBuilder();
                      Value emptyVec = elseBuilder.create<SplatOp>(loc, vecTy, cf0);
                      elseBuilder.create<affine::AffineYieldOp>(loc, emptyVec);

                      iList.push_back(i);
                      fList.push_back(protectedF->getOpResult(0));
                    }
                  }
                  Value lastResult = builder.create<memref::LoadOp>(loc, buffer, c0);
                  for (int i = 0; i < kernelM; ++i) {
                    for (int j = 0; j < kernelN; ++j) {
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
          });
        });
      });
    });

    rewriter.create<memref::DeallocOp>(loc, buffer);

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t vecSize;
  int64_t kernelM;
  int64_t kernelN;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ConvOptimizePass
//===----------------------------------------------------------------------===//

namespace {
class ConvOptimizePass : public PassWrapper<ConvOptimizePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvOptimizePass)
  StringRef getArgument() const final { return "conv-optimize"; }
  StringRef getDescription() const final { return "Conv optimize."; }
  ConvOptimizePass() = default;
  ConvOptimizePass(const ConvOptimizePass &) {}
  explicit ConvOptimizePass(int64_t vecSizeParam, int64_t kernelMParam, int64_t kernelNParam) {
    vecSize = vecSizeParam;
    kernelM = kernelMParam;
    kernelN = kernelNParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect, affine::AffineDialect, VectorDialect>();
  }

  Option<int64_t> vecSize{*this, "vec-size", llvm::cl::desc("Vector size using in kernel."), llvm::cl::init(16)};

  Option<int64_t> kernelM{*this, "kernel-m", llvm::cl::desc("Specify how many rows kernel will contain."), llvm::cl::init(4)};

  Option<int64_t> kernelN{*this, "kernel-n", llvm::cl::desc("Specify how many columns kernel will cantain."), llvm::cl::init(2)};
};
} // end anonymous namespace.

void ConvOptimizePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithDialect, affine::AffineDialect, scf::SCFDialect, memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<ConvOptimizePattern>(context, vecSize, kernelM, kernelN);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerConvOptimizePass() { PassRegistration<ConvOptimizePass>(); }
} // namespace buddy
} // namespace mlir
