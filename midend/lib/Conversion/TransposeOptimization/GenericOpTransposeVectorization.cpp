//===- GenericOpTransposeVectorization.cpp --------------------------------===//
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
// This file implements the GenericOp transpose optimization based on methods
// from BuiltinTransposeVectorization, using only Affine ops for loops and
// conditions, and obtaining dimensions via memref::DimOp.
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
#include <cassert>
#include <cstdint>
#include <llvm/ADT/SmallVector.h>

using namespace mlir;
using namespace vector;
using namespace affine;
using namespace linalg;

namespace {

/// 针对 linalg::GenericOp 的转置操作进行向量化优化。
class GenericOpTransposeVectorizationPattern
    : public OpConversionPattern<GenericOp> {
public:
  explicit GenericOpTransposeVectorizationPattern(MLIRContext *context,
                                                  int64_t affineVectorSizeParam)
      : OpConversionPattern(context), affineVectorSize(affineVectorSizeParam) {}

  LogicalResult
  matchAndRewrite(GenericOp genericOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // 判断是否为转置操作。
    if (!isTransposeOp(genericOp))
      return failure();

    // 获取输入的参数
    Value input = genericOp.getInputs()[0];
    Value output = genericOp.getOutputs()[0];
    auto loc = genericOp.getLoc();
    // 输入类型确定为TensorType
    Type elementType = input.getType().cast<MemRefType>().getElementType();
    // 获取输入张量的维度
    int rank = input.getType().cast<MemRefType>().getRank();

    // 获取各维度大小（使用 memref::DimOp）。
    llvm::SmallVector<Value, 4> dims;
    for (int i = 0; i < rank; i++) {
      dims.push_back(rewriter.create<memref::DimOp>(loc, input, i));
    }
    Value innerDim = dims.back();

    const Value index0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));

    // const Value indexVecSize = rewriter.create<arith::ConstantOp>(
    //     loc, rewriter.getIndexAttr(affineVectorSize));

    const Value zeroElementType = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));

    // 计算 innerUpperBound = (innerDim floordiv affineVectorSize) *
    // affineVectorSize
    Value innerUpperBound = rewriter.create<affine::AffineApplyOp>(
        loc,
        AffineMap::get(1, 0,
                       rewriter.getAffineDimExpr(0).floorDiv(affineVectorSize) *
                           affineVectorSize),
        ValueRange{innerDim});

    // innerUnalignedLength = innerDim mod affineVectorSize
    Value innerUnalignedLength = rewriter.create<affine::AffineApplyOp>(
        loc,
        AffineMap::get(1, 0, rewriter.getAffineDimExpr(0) % affineVectorSize),
        ValueRange{innerDim});
    llvm::SmallVector<Value> inputIndices(rank, nullptr);
    llvm::SmallVector<Value> outputIndices(rank, nullptr);

    // 创建读写掩码
    llvm::SmallVector<int64_t> readShape(rank, 1);
    readShape.back() = affineVectorSize;

    llvm::SmallVector<int64_t> writeShape = readShape;
    writeShape[swapDims.first] = readShape[swapDims.second];
    writeShape[swapDims.second] = readShape[swapDims.first];


    llvm::SmallVector<Value, 4> readMaskOperands;
    for (int i = 0; i < rank - 1; i++) {
      readMaskOperands.push_back(rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1)));
    }
    readMaskOperands.push_back(innerUnalignedLength);
    llvm::SmallVector<Value, 4> writeMaskOperands = readMaskOperands;
    writeMaskOperands[swapDims.first] = readMaskOperands[swapDims.second];
    writeMaskOperands[swapDims.second] = readMaskOperands[swapDims.first];

    Value readMask = rewriter.create<vector::CreateMaskOp>(
        loc, VectorType::get(readShape, rewriter.getI1Type()),
        ValueRange(readMaskOperands));

    Value writeMask = rewriter.create<vector::CreateMaskOp>(
        loc, VectorType::get(writeShape, rewriter.getI1Type()),
        ValueRange(writeMaskOperands));

    // ForOp嵌套循环的Yield位置应该有bugs，需要修复。用递归！！！
    for (int i = 0; i < rank; i++) {
      AffineMap lbMap = AffineMap::get(
          1, 0, {rewriter.getAffineConstantExpr(0)}, rewriter.getContext());
      AffineMap ubMap = AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0)},
                                       rewriter.getContext());
      // 最内层循环进行向量化
      if (i == rank - 1) {
        rewriter.create<affine::AffineForOp>(
            loc, ValueRange{index0}, lbMap, ValueRange{innerUpperBound}, ubMap,
            affineVectorSize, ValueRange{},
            [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
                ValueRange iterArgs) {
              inputIndices[i] = iv;
              outputIndices[i] = iv;

              // 交换输入输出维度Indices
              Value temp = outputIndices[swapDims.first];
              outputIndices[swapDims.first] = outputIndices[swapDims.second];
              outputIndices[swapDims.second] = temp;

              llvm::SmallVector<Value, 4> transferOperands;
              transferOperands.push_back(input); // 第一个操作数为输入张量
              for (Value idx : inputIndices)
                transferOperands.push_back(idx);           // 后续各维的索引
              transferOperands.push_back(zeroElementType); // 可选的填充值

              // 读取输入张量
              auto readValue = rewriter.create<vector::TransferReadOp>(
                  loc,
                  TypeRange{VectorType::get(readShape, elementType)},
                  transferOperands,
                  ArrayRef<NamedAttribute>{
                      rewriter.getNamedAttr(
                          "in_bounds",
                          rewriter.getBoolArrayAttr(llvm::SmallVector<bool, 4>(
                              inputIndices.size(), true))),
                      rewriter.getNamedAttr(
                          "operand_segment_sizes",
                          rewriter.getDenseI32ArrayAttr(
                              llvm::SmallVector<int, 4>{
                                  1, static_cast<int>(inputIndices.size()), 1,
                                  0})),
                      rewriter.getNamedAttr(
                          "permutation_map",
                          AffineMapAttr::get(AffineMap::getMultiDimIdentityMap(
                              inputIndices.size(), rewriter.getContext())))});
              // 写入输出张量
              // 构造写入操作的操作数：首先是待写入的向量值，然后是目标张量，后面跟随每个维度的索引。
              llvm::SmallVector<Value, 4> writeOperands;
              writeOperands.push_back(readValue); // 待写入的向量数据
              writeOperands.push_back(output);    // 输出张量（MemRef）
              for (Value idx : outputIndices)
                writeOperands.push_back(idx);

              // 创建 TransferWriteOp 操作，注意此操作没有返回值，所以 TypeRange
              // 为空。
              rewriter.create<vector::TransferWriteOp>(
                  loc, TypeRange{}, // TransferWriteOp 不产生结果
                  writeOperands,
                  ArrayRef<NamedAttribute>{
                      // in_bounds 属性：对所有维度（包括 memref
                      // 之后的索引）都执行边界检查
                      rewriter.getNamedAttr(
                          "in_bounds",
                          rewriter.getBoolArrayAttr(llvm::SmallVector<bool, 4>(
                              inputIndices.size(), true))),
                      // operand_segment_sizes 属性：指定操作数分段，1
                      // 个向量数据，1 个目标张量，
                      // 后续 inputIndices.size() 个索引，最后 0 个额外操作数。
                      rewriter.getNamedAttr(
                          "operand_segment_sizes",
                          rewriter.getDenseI32ArrayAttr(
                              llvm::SmallVector<int, 4>{
                                  1, 1, static_cast<int>(inputIndices.size()),
                                  0})),
                      // permutation_map
                      // 属性：使用多维度恒等映射，确保索引顺序与操作数顺序一致
                      rewriter.getNamedAttr(
                          "permutation_map",
                          AffineMapAttr::get(AffineMap::getMultiDimIdentityMap(
                              outputIndices.size(), rewriter.getContext())))});
              // 尾部数据处理
              if (input.getType().cast<MemRefType>().isDynamicDim(rank - 1) or
                  input.getType().cast<MemRefType>().getDimSize(rank - 1) %
                          affineVectorSize !=
                      0) {
                affine::AffineIfOp tailIfOp =
                    rewriter.create<affine::AffineIfOp>(
                        loc,
                        IntegerSet::get(
                            1, 0,
                            {rewriter.getAffineDimExpr(0) % affineVectorSize -
                             1},
                            {false}),
                        ValueRange{innerDim}, false);
                OpBuilder tailBranchBuilder = tailIfOp.getThenBodyBuilder();

                // 构造尾部数据各维索引（tailIndices），其中除最内层维度外，其它维度采用当前循环中已有的索引，最内层维度使用
                llvm::SmallVector<Value> intailIndices(rank, nullptr);
                for (int i = 0; i < rank - 1; i++) {
                  // 假设 outer 维度的索引存储在 outputIndices 或 inputIndices
                  // 中（根据实际情况选用）
                  intailIndices[i] = inputIndices[i];
                }
                // 最内层维度使用 innerUpperBound 作为尾部起始索引
                intailIndices[rank - 1] = innerUpperBound;

                llvm::SmallVector<Value> outtailIndices = intailIndices;
                // 交换输入输出维度Indices
                outtailIndices[swapDims.first] = intailIndices[swapDims.second];
                outtailIndices[swapDims.second] = intailIndices[swapDims.first];

                // 构造 TransferReadOp 的操作数：
                // 第一个操作数为输入张量，后续依次为各个维度的索引，再附加一个填充值和尾部掩码
                llvm::SmallVector<Value, 4> tailReadOperands;
                tailReadOperands.push_back(input);
                for (Value idx : intailIndices)
                  tailReadOperands.push_back(idx);
                tailReadOperands.push_back(zeroElementType); // 填充值
                tailReadOperands.push_back(readMask);        // 尾部掩码

                auto tailReadValue =
                    tailBranchBuilder.create<vector::TransferReadOp>(
                        loc,
                        TypeRange{
                            VectorType::get(readShape, elementType)},
                        tailReadOperands,
                        ArrayRef<NamedAttribute>{
                            tailBranchBuilder.getNamedAttr(
                                "in_bounds",
                                tailBranchBuilder.getBoolArrayAttr(
                                    llvm::SmallVector<bool, 4>(
                                        intailIndices.size() + 2, true))),
                            tailBranchBuilder.getNamedAttr(
                                "operand_segment_sizes",
                                tailBranchBuilder.getDenseI32ArrayAttr(
                                    llvm::SmallVector<int, 4>{
                                        1,
                                        static_cast<int>(intailIndices.size()),
                                        1, 1})),
                            tailBranchBuilder.getNamedAttr(
                                "permutation_map",
                                AffineMapAttr::get(
                                    AffineMap::getMultiDimIdentityMap(
                                        intailIndices.size(),
                                        rewriter.getContext())))});

                // 构造 TransferWriteOp 的操作数：
                // 第一个操作数为待写入的向量数据（tailRead），第二个为输出张量，其余依次为各个维度的索引
                llvm::SmallVector<Value, 4> tailWriteOperands;
                tailWriteOperands.push_back(tailReadValue);
                tailWriteOperands.push_back(output);
                for (Value idx : outtailIndices)
                  tailWriteOperands.push_back(idx);
                tailWriteOperands.push_back(writeMask); // 尾部掩码cd bi

                // 创建 TransferWriteOp 操作，注意此操作没有返回值，所以
                // TypeRange 为空。
                tailBranchBuilder.create<vector::TransferWriteOp>(
                    loc, TypeRange{}, // TransferWriteOp 不产生结果
                    tailWriteOperands,
                    ArrayRef<NamedAttribute>{
                        // in_bounds 属性：对所有维度（包括 memref
                        // 之后的索引）都执行边界检查
                        tailBranchBuilder.getNamedAttr(
                            "in_bounds",
                            tailBranchBuilder.getBoolArrayAttr(
                                llvm::SmallVector<bool, 4>(
                                    outtailIndices.size(), true))),
                        // operand_segment_sizes 属性：指定操作数分段，1
                        // 个向量数据，1 个目标张量，
                        // 后续 outtailIndices.size() 个索引，最后 0
                        // 个额外操作数。
                        tailBranchBuilder.getNamedAttr(
                            "operand_segment_sizes",
                            tailBranchBuilder.getDenseI32ArrayAttr(
                                llvm::SmallVector<int, 4>{
                                    1, 1,
                                    static_cast<int>(outtailIndices.size()),
                                    1})),
                        // permutation_map
                        // 属性：使用多维度恒等映射，确保索引顺序与操作数顺序一致
                        tailBranchBuilder.getNamedAttr(
                            "permutation_map",
                            AffineMapAttr::get(
                                AffineMap::getMultiDimIdentityMap(
                                    outtailIndices.size(),
                                    rewriter.getContext())))});
              }

              // 插入 affine::AffineYieldOp
              nestedBuilder.create<affine::AffineYieldOp>(nestedLoc);
            });
      } else {
        rewriter.create<affine::AffineForOp>(
            loc, ValueRange{index0}, lbMap, ValueRange{dims[i]}, ubMap,
            affineVectorSize, ValueRange{},
            [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
                ValueRange iterArgs) {
              inputIndices[i] = iv;
              outputIndices[i] = iv;

              // 插入 affine::AffineYieldOp
              nestedBuilder.create<affine::AffineYieldOp>(nestedLoc);
            });
      }
    }

    // 删除原始操作
    rewriter.eraseOp(genericOp);
    return success();
  }

private:
  int64_t affineVectorSize;
  // 记录转置时交换的两个维度，在 isTransposeOp 中确定。
  mutable std::pair<unsigned, unsigned> swapDims;
  /// 判断 GenericOp 是否为转置操作。
  bool isTransposeOp(GenericOp genericOp) const {
    if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1)
      return false;
    llvm::SmallVector<AffineMap, 2> indexingMaps =
        genericOp.getIndexingMapsArray();
    AffineMap inputMap = indexingMaps[0], outputMap = indexingMaps[1];
    if (inputMap.getNumDims() != outputMap.getNumDims())
      return false;
    for (unsigned i = 0; i < inputMap.getNumDims(); ++i) {
      AffineExpr inputExpr = inputMap.getResult(i);
      AffineExpr outputExpr = outputMap.getResult(i);
      if (inputExpr != outputExpr) {
        bool isPermutation = false;
        for (unsigned j = 0; j < inputMap.getNumDims(); ++j) {
          if (inputMap.getResult(j) == outputMap.getResult(i) &&
              outputMap.getResult(j) == inputMap.getResult(i)) {
            isPermutation = true;
            swapDims = std::make_pair(i, j);
            break;
          }
        }
        if (!isPermutation)
          return false;
      }
    }
    for (auto it : genericOp.getIteratorTypesArray())
      if (it != utils::IteratorType::parallel)
        return false;
    Region &body = genericOp.getRegion();
    if (!body.hasOneBlock() ||
        !llvm::hasSingleElement(body.front().getOperations()))
      return false;
    Operation &onlyOp = body.front().front();
    // 要求循环体只有一个 yield 操作，这里使用 linalg::YieldOp 作为判断条件。
    if (!isa<linalg::YieldOp>(onlyOp))
      return false;
    return true;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// GenericOpTransposeVectorizationPass
//===----------------------------------------------------------------------===//

namespace {
class GenericOpTransposeVectorizationPass
    : public PassWrapper<GenericOpTransposeVectorizationPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenericOpTransposeVectorizationPass)
  StringRef getArgument() const final {
    return "genericOp-transpose-vectorization";
  }
  StringRef getDescription() const final {
    return "Transpose Optimization for any rank tensor.";
  }
  GenericOpTransposeVectorizationPass() = default;
  GenericOpTransposeVectorizationPass(
      const GenericOpTransposeVectorizationPass &) {}
  explicit GenericOpTransposeVectorizationPass(int64_t affineVectorSizeParam) {
    affineVectorSize = affineVectorSizeParam;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();
    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                           memref::MemRefDialect, VectorDialect,
                           bufferization::BufferizationDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp, linalg::FillOp>();
    RewritePatternSet patterns(context);
    patterns.add<GenericOpTransposeVectorizationPattern>(context,
                                                         affineVectorSize);
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, affine::AffineDialect, VectorDialect,
                memref::MemRefDialect, bufferization::BufferizationDialect>();
  }

  Option<int64_t> affineVectorSize{*this, "vector-size",
                                   llvm::cl::desc("Affine Vector size."),
                                   llvm::cl::init(16)};
};
} // namespace

namespace mlir {
namespace buddy {
void registerGenericOpTransposeVectorizationPass() {
  PassRegistration<GenericOpTransposeVectorizationPass>();
}
} // namespace buddy
} // namespace mlir