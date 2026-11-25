//===- LowerLinalgToMagia.cpp - Linalg matrix products to MAGIA ----------===//
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
// The FP32 matrix products of a bufferized graph become general matrix
// products of the MAGIA mesh. RedMulE computes in FP16, so the operands are
// converted on the way in and the result on the way out.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Magia/MagiaDialect.h"
#include "Magia/MagiaOps.h"

using namespace mlir;
using namespace buddy;

namespace {

/// The elements of `source` converted into `target` (FP32 to FP16 or back);
/// `transposed` reads `source` with its two dimensions exchanged.
void convert(PatternRewriter &rewriter, Location loc, Value source,
             Value target, bool transposed = false) {
  AffineMap identity = rewriter.getMultiDimIdentityMap(2);
  AffineMap exchanged = AffineMap::getPermutationMap(ArrayRef<unsigned>{1, 0},
                                                     rewriter.getContext());
  SmallVector<AffineMap> maps = {transposed ? exchanged : identity, identity};
  SmallVector<utils::IteratorType> iterators(2, utils::IteratorType::parallel);
  linalg::GenericOp::create(
      rewriter, loc, ValueRange{source}, ValueRange{target}, maps, iterators,
      [](OpBuilder &builder, Location loc, ValueRange elements) {
        Type to = elements[1].getType();
        Value value =
            to.isF16()
                ? Value(arith::TruncFOp::create(builder, loc, to, elements[0]))
                : Value(arith::ExtFOp::create(builder, loc, to, elements[0]));
        linalg::YieldOp::create(builder, loc, value);
      });
}

/// A contiguous FP16 copy of the FP32 matrix `source`.
Value half(PatternRewriter &rewriter, Location loc, Value source,
           bool transposed = false) {
  auto type = cast<MemRefType>(source.getType());
  SmallVector<int64_t> shape(type.getShape());
  SmallVector<Value> dynamic;
  if (transposed)
    std::swap(shape[0], shape[1]);
  for (unsigned i = 0; i < 2; ++i)
    if (ShapedType::isDynamic(shape[i]))
      dynamic.push_back(
          memref::DimOp::create(rewriter, loc, source, transposed ? 1 - i : i));
  Value target = memref::AllocOp::create(
      rewriter, loc, MemRefType::get(shape, rewriter.getF16Type()), dynamic);
  convert(rewriter, loc, source, target, transposed);
  return target;
}

/// C += A * B in FP32 becomes Z = Z + X * W in FP16 on the mesh. B is read as
/// [K, N], or as [N, K] the way a Linear keeps its weight. A B already in
/// FP16 is W itself: the mesh reads it in place.
class MatmulLowering : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Value a = op.getInputs()[0], b = op.getInputs()[1], c = op.getOutputs()[0];
    for (Value operand : {a, c}) {
      auto type = dyn_cast<MemRefType>(operand.getType());
      if (!type || !type.getElementType().isF32())
        return failure();
    }
    auto weight = dyn_cast<MemRefType>(b.getType());
    if (!weight)
      return failure();
    bool converted = weight.getElementType().isF16();
    if (!converted && !weight.getElementType().isF32())
      return failure();
    if (converted && op.hasUserDefinedMaps())
      return failure();
    bool transposed = false;
    if (op.hasUserDefinedMaps()) {
      AffineExpr m, n, k;
      bindDims(op.getContext(), m, n, k);
      SmallVector<AffineMap> weight = {
          AffineMap::get(3, 0, {m, k}, op.getContext()),
          AffineMap::get(3, 0, {n, k}, op.getContext()),
          AffineMap::get(3, 0, {m, n}, op.getContext())};
      if (op.getIndexingMapsArray() != weight)
        return failure();
      transposed = true;
    }
    Location loc = op.getLoc();
    Value x = half(rewriter, loc, a);
    Value w = converted ? b : half(rewriter, loc, b, transposed);
    Value z = half(rewriter, loc, c);
    magia::GemmOp::create(rewriter, loc, x, w, z);
    convert(rewriter, loc, z, c);
    memref::DeallocOp::create(rewriter, loc, x);
    if (!converted)
      memref::DeallocOp::create(rewriter, loc, w);
    memref::DeallocOp::create(rewriter, loc, z);
    rewriter.eraseOp(op);
    return success();
  }
};

/// The products of a batch, one after another; the matrices may have dynamic
/// sizes, the batch is static.
class BatchMatmulLowering : public OpRewritePattern<linalg::BatchMatmulOp> {
public:
  using OpRewritePattern<linalg::BatchMatmulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::BatchMatmulOp op,
                                PatternRewriter &rewriter) const override {
    if (op.hasUserDefinedMaps())
      return failure();
    SmallVector<Value> operands = {op.getInputs()[0], op.getInputs()[1],
                                   op.getOutputs()[0]};
    for (Value operand : operands) {
      auto type = dyn_cast<MemRefType>(operand.getType());
      if (!type || !type.getElementType().isF32() || type.isDynamicDim(0))
        return failure();
    }
    Location loc = op.getLoc();
    int64_t batch = cast<MemRefType>(operands[0].getType()).getShape()[0];
    for (int64_t index = 0; index < batch; ++index) {
      SmallVector<Value> matrices;
      for (Value operand : operands) {
        auto type = cast<MemRefType>(operand.getType());
        SmallVector<OpFoldResult> offsets = {rewriter.getIndexAttr(index),
                                             rewriter.getIndexAttr(0),
                                             rewriter.getIndexAttr(0)};
        SmallVector<OpFoldResult> sizes =
            memref::getMixedSizes(rewriter, loc, operand);
        sizes[0] = rewriter.getIndexAttr(1);
        SmallVector<OpFoldResult> strides(3, rewriter.getIndexAttr(1));
        MemRefType matrix = memref::SubViewOp::inferRankReducedResultType(
            type.getShape().drop_front(), type, offsets, sizes, strides);
        matrices.push_back(memref::SubViewOp::create(
            rewriter, loc, matrix, operand, offsets, sizes, strides));
      }
      linalg::MatmulOp::create(rewriter, loc,
                               ValueRange{matrices[0], matrices[1]},
                               ValueRange{matrices[2]});
    }
    rewriter.eraseOp(op);
    return success();
  }
};

class LowerLinalgToMagiaPass
    : public PassWrapper<LowerLinalgToMagiaPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLinalgToMagiaPass);
  LowerLinalgToMagiaPass() = default;
  LowerLinalgToMagiaPass(const LowerLinalgToMagiaPass &) {}
  StringRef getArgument() const final { return "convert-linalg-to-magia"; }
  StringRef getDescription() const final {
    return "convert linalg matrix products to the magia dialect";
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MatmulLowering, BatchMatmulLowering>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<magia::MagiaDialect, memref::MemRefDialect,
                    linalg::LinalgDialect, arith::ArithDialect>();
  }
};
} // namespace

namespace mlir {
namespace buddy {
void registerLowerLinalgToMagiaPass() {
  PassRegistration<LowerLinalgToMagiaPass>();
}
} // namespace buddy
} // namespace mlir
