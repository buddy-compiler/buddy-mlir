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

#include <llvm/CodeGen/MachineInstrBuilder.h>
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
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/IntegerSet.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace vector;

LogicalResult buildOptimizedConv2DNchwFchwOp(
    Location loc, ConversionPatternRewriter &rewriter, int64_t vecSize,
    int64_t kM, int64_t kN, Value input, Value filter, Value output) {
  // Some constant we need.
  const Value c0 =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
  // TODO: Support more data types.
  const Value cf0 =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.));

  // Some Affine Expr constant we need to build AffineExpr.
  const AffineExpr d0 = rewriter.getAffineDimExpr(0);
  const AffineExpr d1 = rewriter.getAffineDimExpr(1);
  const AffineExpr s0 = rewriter.getAffineSymbolExpr(0);

  ShapedType inputTy = input.getType().cast<ShapedType>();

  Type elemTy = inputTy.getElementType();
  VectorType vecTy = VectorType::get(vecSize, elemTy);

  // Dims
  Value dimA = rewriter.create<memref::DimOp>(loc, output, 0);
  Value dimB = rewriter.create<memref::DimOp>(loc, output, 1);
  Value dimC = rewriter.create<memref::DimOp>(loc, output, 2);
  Value dimD = rewriter.create<memref::DimOp>(loc, output, 3);
  Value dimE = rewriter.create<memref::DimOp>(loc, input, 1);
  Value dimF = rewriter.create<memref::DimOp>(loc, filter, 2);
  Value dimG = rewriter.create<memref::DimOp>(loc, filter, 3);

  // memref<1xvector<vecsize x elemTy>>
  MemRefType bufferTy = MemRefType::get(1, vecTy);
  Value buffer = rewriter.create<memref::AllocOp>(loc, bufferTy);

  // TODO: Support higher dim.
  auto buildOuterLoops =
      [&](Location loc, PatternRewriter &rewriter,
          function_ref<void(Location, OpBuilder &, ValueRange)>
              buildFilterLoopsFn) {
        buildAffineLoopNest(
            rewriter, loc, c0, dimA, 1,
            [&](OpBuilder &, Location, ValueRange ivRange) {
              Value ivA = ivRange.front();
              buildAffineLoopNest(
                  rewriter, loc, c0, dimB, 1,
                  [&](OpBuilder &, Location, ValueRange ivRange) {
                    Value ivB = ivRange.front();
                    // Reorder loop order for optimization.
                    buildAffineLoopNest(
                        rewriter, loc, c0, dimD, 1,
                        [&](OpBuilder &, Location, ValueRange ivRange) {
                          Value ivD = ivRange.front();
                          buildAffineLoopNest(
                              rewriter, loc, c0, dimC, 1,
                              [&](OpBuilder &builder, Location loc,
                                  ValueRange ivRange) {
                                Value ivC = ivRange.front();
                                buildFilterLoopsFn(
                                    loc, builder,
                                    ValueRange{ivA, ivB, ivC, ivD});
                              });
                        });
                  });
            });
        rewriter.create<memref::DeallocOp>(loc, buffer);
      };

  // TODO: Support higher dim.
  auto buildFilterLoops =
      [&](Location loc, OpBuilder &builder, ValueRange outerLoopsIvs,
          function_ref<void(Location, OpBuilder &, ValueRange, ValueRange)>
              buildKernelFn) {
        Value splatBufferOp = builder.create<SplatOp>(loc, vecTy, cf0);
        builder.create<memref::StoreOp>(loc, splatBufferOp, buffer, c0);
        buildAffineLoopNest(
            rewriter, loc, c0, dimE, 1,
            [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
              Value ivE = ivRange.front();

              Value dimFTileUb = builder.create<AffineApplyOp>(
                  loc, AffineMap::get(1, 0, d0.ceilDiv(kM) * kM),
                  ValueRange{dimF});

              buildAffineLoopNest(
                  rewriter, loc, c0, dimFTileUb, kM,
                  [&](OpBuilder &, Location loc, ValueRange ivRange) {
                    Value ivF = ivRange.front();
                    buildAffineLoopNest(rewriter, loc, c0, dimG, kN * vecSize,
                                        [&](OpBuilder &builder, Location loc,
                                            ValueRange ivRange) {
                                          Value ivG = ivRange.front();
                                          buildKernelFn(
                                              loc, builder, outerLoopsIvs,
                                              ValueRange{ivE, ivF, ivG});
                                        });
                  });
            });
      };

  buildOuterLoops(
      loc, rewriter,
      [&](Location loc, OpBuilder &builder, ValueRange outerLoopsIVs) {
        buildFilterLoops(
            loc, builder, outerLoopsIVs,
            [&](Location loc, OpBuilder &builder, ValueRange outerLoopsIVs,
                ValueRange filterLoopsIVs) {
              auto ivA = outerLoopsIVs[0];
              auto ivB = outerLoopsIVs[1];
              auto ivC = outerLoopsIVs[2];
              auto ivD = outerLoopsIVs[3];

              auto ivE = filterLoopsIVs[0];
              auto ivF = filterLoopsIVs[1];
              auto ivG = filterLoopsIVs[2];

              SmallVector<Value> iList;
              SmallVector<Value> fList;
              for (int64_t i = 0; i < kM; ++i) {
                AffineMap rowInputMap = AffineMap::get(2, 0, d0 + i + d1);
                Value rowInput = builder.create<AffineApplyOp>(
                    loc, rowInputMap, ValueRange{ivC, ivF});

                AffineMap rowFilterMap = AffineMap::get(1, 0, d0 + i);
                Value rowFilter =
                    builder.create<AffineApplyOp>(loc, rowFilterMap, ivF);

                for (int64_t j = 0; j < kN; ++j) {

                  AffineMap columnInputMap =
                      AffineMap::get(2, 0, d0 + d1 + j * vecSize);
                  Value columnInput = builder.create<AffineApplyOp>(
                      loc, columnInputMap, ValueRange{ivD, ivG});

                  AffineMap columnFilterMap =
                      AffineMap::get(1, 0, d0 + j * vecSize);
                  Value columnFilter =
                      builder.create<AffineApplyOp>(loc, columnFilterMap, ivG);

                  Value i = builder.create<TransferReadOp>(
                      loc, vecTy, input,
                      ValueRange{ivA, ivE, rowInput, columnInput});

                  IntegerSet protectedFSet =
                      IntegerSet::get(1, 1, {s0 - 1 - d0}, {false});
                  auto protectedF = builder.create<AffineIfOp>(
                      loc, vecTy, protectedFSet, ValueRange{rowFilter, dimF},
                      true);

                  // if row in range, read normally.
                  auto thenBuilder = protectedF.getThenBodyBuilder();
                  Value normalReadVec = thenBuilder.create<TransferReadOp>(
                      loc, vecTy, filter,
                      ValueRange{ivB, ivE, rowFilter, columnFilter});
                  thenBuilder.create<AffineYieldOp>(loc, normalReadVec);

                  // if row out of range, give back a empty vector.
                  auto elseBuilder = protectedF.getElseBodyBuilder();
                  Value emptyVec = elseBuilder.create<SplatOp>(loc, vecTy, cf0);
                  elseBuilder.create<AffineYieldOp>(loc, emptyVec);

                  iList.push_back(i);
                  fList.push_back(protectedF->getOpResult(0));
                }
              }

              Value lastResult =
                  builder.create<memref::LoadOp>(loc, buffer, c0);

              for (int64_t i = 0; i < kM; ++i) {
                for (int64_t j = 0; j < kM; ++j) {
                  auto inputOp = iList[i * kM + j];
                  auto filterOp = iList[i * kN + j];
                  lastResult = builder.create<vector::FMAOp>(
                      loc, vecTy, inputOp, filterOp, lastResult);
                }
              }

              builder.create<memref::StoreOp>(loc, lastResult, buffer, c0);

              Value reduceVec = builder.create<memref::LoadOp>(loc, buffer, c0);
              Value reducedRes = builder.create<vector::ReductionOp>(
                  loc, vector::CombiningKind::ADD, reduceVec);
              Value bias = builder.create<memref::LoadOp>(
                  loc, output, ValueRange{ivA, ivB, ivC, ivD});
              Value addRes =
                  builder.create<arith::AddFOp>(loc, bias, reducedRes);

              builder.create<memref::StoreOp>(loc, addRes, output,
                                              ValueRange{ivA, ivB, ivC, ivD});
            });
      });

  return success();
}

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class ConvOptimizePattern : public ConversionPattern {
public:
  explicit ConvOptimizePattern(MLIRContext *context, int64_t vecSizeParam,
                               int64_t kernelMParam, int64_t kernelNParam)
      : ConversionPattern(linalg::Conv2DNchwFchwOp::getOperationName(), 1,
                          context) {
    vecSize = vecSizeParam;
    kernelM = kernelMParam;
    kernelN = kernelNParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {

    Value input = op->getOperand(0);
    Value filter = op->getOperand(0);
    Value output = op->getOperand(0);

    if (buildOptimizedConv2DNchwFchwOp(op->getLoc(), rewriter, vecSize, kernelM,
                                       kernelN, input, filter, output)
            .failed()) {
      return failure();
    } else {
      rewriter.eraseOp(op);
      return success();
    }
  }

private:
  int64_t vecSize;
  int64_t kernelM;
  int64_t kernelN;

}; // end anonymous namespace
} // namespace

//===----------------------------------------------------------------------===//
// ConvOptimizePass
//===----------------------------------------------------------------------===//

namespace {
class ConvOptimizePass
    : public PassWrapper<ConvOptimizePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvOptimizePass)
  StringRef getArgument() const final { return "conv-optimize"; }
  StringRef getDescription() const final { return "Conv optimize."; }
  ConvOptimizePass() = default;
  ConvOptimizePass(const ConvOptimizePass &) {}
  explicit ConvOptimizePass(int64_t vecSizeParam, int64_t kernelMParam,
                            int64_t kernelNParam) {
    vecSize = vecSizeParam;
    kernelM = kernelMParam;
    kernelN = kernelNParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect, AffineDialect,
                    VectorDialect>();
  }

  Option<int64_t> vecSize{*this, "vec-size",
                          llvm::cl::desc("Vector size using in kernel."),
                          llvm::cl::init(16)};

  Option<int64_t> kernelM{
      *this, "kernel-m",
      llvm::cl::desc("Specify how many rows kernel will contain."),
      llvm::cl::init(4)};

  Option<int64_t> kernelN{
      *this, "kernel-n",
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
  patterns.add<ConvOptimizePattern>(context, vecSize, kernelM, kernelN);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerConvOptimizePass() { PassRegistration<ConvOptimizePass>(); }
} // namespace buddy
} // namespace mlir
