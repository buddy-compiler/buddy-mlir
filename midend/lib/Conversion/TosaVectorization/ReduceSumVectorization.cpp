//===- ReduceSumVectorization.cpp ----------------------------------===//
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
// This file implements the reduce sum vectorization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <mlir/Dialect/Affine/Analysis/AffineAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;
using namespace vector;
using namespace affine;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class ReduceSumVectorizationPattern : public ConversionPattern {
public:
  explicit ReduceSumVectorizationPattern(MLIRContext *context,
                                         int64_t affineVectorSizeParam)
      : ConversionPattern(linalg::ReduceOp::getOperationName(), 1, context) {
    affineVectorSize = affineVectorSizeParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto reduceOp = cast<linalg::ReduceOp>(op);

    // 获取输入和输出张量
    Value input = reduceOp.getOperand(0);
    Value output = reduceOp.getOperand(1);
    auto loc = op->getLoc();

    // 获取元素类型和维度信息
    Type elementType = input.getType().cast<MemRefType>().getElementType();
    int inputRank = input.getType().cast<MemRefType>().getRank();
    int outputRank = output.getType().cast<MemRefType>().getRank();
    if (inputRank != outputRank + 1) {
      return failure();
    }

    // 获取规约维度
    auto dimensionsAttr = op->getAttr(rewriter.getStringAttr("dimensions"))
                              .cast<DenseI64ArrayAttr>()
                              .asArrayRef();
    if (dimensionsAttr.size() != 1 || dimensionsAttr[0] != inputRank - 1) {
      return failure();
    }

    // 定义常量
    const Value index0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value zeroElementType = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));

    // 获取输入张量的各个维度大小
    SmallVector<Value, 4> dims;
    for (int i = 0; i < inputRank; i++) {
      dims.push_back(rewriter.create<memref::DimOp>(loc, input, i));
    }

    // 创建嵌套循环进行处理
    SmallVector<Value> inputIndices(inputRank);
    SmallVector<Value> outputIndices(outputRank);
    Value upperBound =
        rewriter.create<arith::ConstantIndexOp>(loc, affineVectorSize);

    // 调用createNestedLoops开始创建循环嵌套
    createNestedLoops(rewriter, loc, input, output, dims, elementType,
                      /*currentDim=*/0, inputIndices, outputIndices, index0,
                      upperBound, zeroElementType,
                      /*vectorSize=*/static_cast<int64_t>(affineVectorSize),
                      /*mask=*/nullptr, dimensionsAttr,
                      /*currentReductionDim=*/dimensionsAttr[0],
                      /*blockSize=*/8);

    // 删除原始操作
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t affineVectorSize;

  void createNestedLoops(ConversionPatternRewriter &rewriter, Location loc,
                         Value input, Value output, SmallVector<Value, 4> &dims,
                         Type elementType, int64_t currentDim,
                         SmallVector<Value> &inputIndices,
                         SmallVector<Value> &outputIndices, Value index0,
                         Value upperBound, Value zeroElementType,
                         int64_t vectorSize, Value mask,
                         ArrayRef<int64_t> reductionDims,
                         int64_t currentReductionDim, int64_t blockSize) const {
    // 基本情况：所有维度都已处理完
    if (currentDim >= static_cast<int64_t>(dims.size())) {
      return;
    }

    // 检查当前维度是否是规约维度（最内层）
    bool isReductionDim = currentDim == currentReductionDim;

    // 获取当前维度的大小
    Value dimSize = dims[currentDim];

    if (!isReductionDim) {
      // 第一步：为分块处理创建外层affine.for循环
      auto zeroExpr = rewriter.getAffineConstantExpr(0);
      auto dimExpr = rewriter.getAffineDimExpr(0);

      // 创建外层循环，步长为blockSize
      auto outerLoop = rewriter.create<affine::AffineForOp>(
          loc,
          /*lbOperands=*/ValueRange{},
          /*lbMap=*/AffineMap::get(0, 0, {zeroExpr}),
          /*ubOperands=*/ValueRange{dimSize},
          /*ubMap=*/AffineMap::get(1, 0, {dimExpr}),
          /*step=*/blockSize);

      rewriter.setInsertionPointToStart(outerLoop.getBody());
      Value outerIv = outerLoop.getInductionVar();

      // 如果是最后一个非规约维度，创建parallel循环进行分块处理
      if (currentDim == currentReductionDim - 1) {
        // 创建用于边界的affine maps
        auto lbMap = AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0)});
        auto ubMap =
            AffineMap::get(1, 0,
                           {rewriter.getAffineDimExpr(0) +
                            rewriter.getAffineConstantExpr(blockSize)});

        // 创建parallel循环
        AffineParallelOp parallelLoop =
            rewriter.create<affine::AffineParallelOp>(
                loc, TypeRange{}, ValueRange{outerIv},
                ArrayRef<NamedAttribute>{
                    rewriter.getNamedAttr("lowerBoundsGroups",
                                          rewriter.getI32TensorAttr({1})),
                    rewriter.getNamedAttr("upperBoundsGroups",
                                          rewriter.getI32TensorAttr({1})),
                    rewriter.getNamedAttr("lowerBoundsMap",
                                          AffineMapAttr::get(lbMap)),
                    rewriter.getNamedAttr("upperBoundsMap",
                                          AffineMapAttr::get(ubMap)),
                    rewriter.getNamedAttr("steps",
                                          rewriter.getI64ArrayAttr({1})),
                    rewriter.getNamedAttr("reductions",
                                          rewriter.getArrayAttr({}))});

        // 创建循环体
        Block *parallelBody = new Block();
        parallelBody->addArgument(rewriter.getIndexType(), loc);
        Value parallelIv = parallelBody->getArguments()[0];

        // 创建一个新构建器用于parallel体
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(parallelBody);

        // 初始化累加器
        Value init_acc = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getZeroAttr(elementType));

        // 确保索引有效后进行存储
        inputIndices[currentDim] = parallelIv;
        outputIndices[currentDim] = parallelIv;

        // Validate all indices are initialized
        bool allIndicesValid = true;
        for (size_t idx = 0; idx < outputIndices.size(); ++idx) {
          if (!outputIndices[idx]) {
            outputIndices[idx] = index0;
          }
          allIndicesValid &= (outputIndices[idx] != nullptr);
        }

        // Only store if all indices are valid
        if (allIndicesValid && output) {
          rewriter.create<memref::StoreOp>(loc, init_acc, output,
                                           outputIndices);

          // 创建向量化的规约循环
          createNestedLoops(rewriter, loc, input, output, dims, elementType,
                            currentDim + 1, inputIndices, outputIndices, index0,
                            upperBound, zeroElementType, vectorSize, mask,
                            reductionDims, currentReductionDim, blockSize);
        }

        // 将循环体添加到parallel循环中
        parallelLoop.getRegion().push_back(parallelBody);
      } else {
        // Continue creating nested loops for remaining dimensions
        createNestedLoops(rewriter, loc, input, output, dims, elementType,
                          currentDim + 1, inputIndices, outputIndices, index0,
                          upperBound, zeroElementType, vectorSize, mask,
                          reductionDims, currentReductionDim, blockSize);
      }
    } else {
      // 对于规约维度，使用向量化处理
      // 初始化累加器为零
      Value init_acc = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(elementType));

      // 确保所有索引都正确初始化
      SmallVector<Value> validInputIndices;
      SmallVector<Value> validOutputIndices;
      validInputIndices.reserve(inputIndices.size());
      validOutputIndices.reserve(outputIndices.size());

      // 初始化输入索引
      for (size_t idx = 0; idx < inputIndices.size(); ++idx) {
        if (!inputIndices[idx]) {
          validInputIndices.push_back(index0);
        } else {
          validInputIndices.push_back(inputIndices[idx]);
        }
      }

      // 初始化输出索引
      for (size_t idx = 0; idx < outputIndices.size(); ++idx) {
        if (!outputIndices[idx]) {
          validOutputIndices.push_back(index0);
        } else {
          validOutputIndices.push_back(outputIndices[idx]);
        }
      }

      // 存储初始累加器值
      if (!output || validOutputIndices.empty()) {
        return;
      }
      rewriter.create<memref::StoreOp>(loc, init_acc, output,
                                       validOutputIndices);

      // 创建用于边界的affine表达式
      auto zeroExpr = rewriter.getAffineConstantExpr(0);
      auto dimExpr = rewriter.getAffineDimExpr(0);

      // 创建主向量化循环处理对齐数据
      auto mainLoop = rewriter.create<affine::AffineForOp>(
          loc,
          /*lbOperands=*/ValueRange{},
          /*lbMap=*/AffineMap::get(0, 0, {zeroExpr}),
          /*ubOperands=*/ValueRange{dimSize},
          /*ubMap=*/AffineMap::get(1, 0, {dimExpr}),
          /*step=*/vectorSize);

      // 设置插入点到循环体开始
      rewriter.setInsertionPointToStart(mainLoop.getBody());

      Value iv = mainLoop.getInductionVar();
      validInputIndices[currentDim] = iv;

      // 创建向量读取操作
      auto vectorType = VectorType::get({vectorSize}, elementType);
      auto permMap = AffineMap::getMinorIdentityMap(validInputIndices.size(), 1,
                                                    rewriter.getContext());
      auto permMapAttr = AffineMapAttr::get(permMap);

      auto vec = rewriter.create<vector::TransferReadOp>(
          loc, vectorType, input, validInputIndices, permMapAttr,
          ArrayAttr::get(rewriter.getContext(), {rewriter.getBoolAttr(true)}));

      // 创建向量规约操作
      auto blockSum = rewriter.create<vector::ReductionOp>(
          loc, elementType, CombiningKind::ADD, vec, Value(),
          arith::FastMathFlags::none);

      // 加载当前累加器值
      Value currentAcc =
          rewriter.create<memref::LoadOp>(loc, output, validOutputIndices);

      // 更新累加器值
      Value newAcc = rewriter.create<arith::AddFOp>(loc, currentAcc, blockSum);

      // 存储结果
      rewriter.create<memref::StoreOp>(loc, newAcc, output, validOutputIndices);

      // 在最内层循环末尾创建yield
      rewriter.create<affine::AffineYieldOp>(loc);
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// ReduceSumVectorizationPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg pooling operations to mixture of
/// Affine + Vector operations.
namespace {
class ReduceSumVectorizationPass
    : public PassWrapper<ReduceSumVectorizationPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReduceSumVectorizationPass)
  StringRef getArgument() const final { return "reduce-sum-vectorize"; }
  StringRef getDescription() const final { return "Reduce Sum Vectorization."; }
  ReduceSumVectorizationPass() = default;
  ReduceSumVectorizationPass(const ReduceSumVectorizationPass &) {}
  explicit ReduceSumVectorizationPass(int64_t affineVectorSizeParam) {
    affineVectorSize = affineVectorSizeParam;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                           memref::MemRefDialect, VectorDialect,
                           scf::SCFDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
    target.addLegalOp<linalg::FillOp>();

    // Make sure all memref operations are legal
    target.addLegalOp<memref::AllocOp, memref::DeallocOp, memref::LoadOp,
                      memref::StoreOp, memref::DimOp>();

    // Make sure vector operations are legal
    target.addLegalOp<vector::TransferReadOp, vector::TransferWriteOp,
                      vector::ReductionOp>();

    RewritePatternSet patterns(context);
    patterns.add<ReduceSumVectorizationPattern>(context, affineVectorSize);

    // Add necessary conversion patterns
    mlir::populateAffineToStdConversionPatterns(patterns);
    mlir::populateVectorToSCFConversionPatterns(patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, affine::AffineDialect, VectorDialect,
                    scf::SCFDialect>();
  }

  Option<int64_t> affineVectorSize{*this, "vector-size",
                                   llvm::cl::desc("Affine Vector size."),
                                   llvm::cl::init(16)};
};
} // end anonymous namespace.

namespace mlir {
namespace buddy {

std::unique_ptr<Pass> createReduceSumVectorizationPass() {
  return std::make_unique<ReduceSumVectorizationPass>();
}

void registerReduceSumVectorizationPass() {
  PassRegistration<ReduceSumVectorizationPass>();
}

} // namespace buddy
} // namespace mlir